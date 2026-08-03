from __future__ import annotations

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.runtime import ForecastRetryActual, SpectrumH3Runtime

TOPOLOGY = (("video", (1, 24, 2, 4, 4)), ("audio", (1, 32, 2, 8)), ("hidden", 4))
LABEL = ((0, "positive"),)


def _runtime(**overrides):
    values = {
        "degree": 1,
        "max_history": 4,
        "warmup_steps": 2,
        "tail_actual_steps": 0,
        "window_size": 2.0,
    }
    values.update(overrides)
    return SpectrumH3Runtime(SpectrumH3Config(**values))


def _actual_step(runtime, timestep, records):
    decision = runtime.begin_step(torch.tensor([timestep]))
    assert decision["actual"]
    for labels, feature in records:
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=labels,
            expected_shape=tuple(feature.shape),
        )
        assert actual
        runtime.observe_actual(decision["run_id"], decision["step_id"], call_id, feature)
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    return decision


def _forecast_step(runtime, timestep, labels=LABEL):
    decision = runtime.begin_step(torch.tensor([timestep]))
    assert not decision["actual"]
    shape = (len(labels), 3, 4)
    call_id, actual = runtime.begin_model_call(
        decision["run_id"],
        decision["step_id"],
        topology=TOPOLOGY,
        labels=labels,
        expected_shape=shape,
    )
    assert not actual
    prediction = runtime.predict(
        decision["run_id"], decision["step_id"], call_id, device=torch.device("cpu"), dtype=torch.float32
    )
    assert prediction is not None
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    return prediction


def test_scheduler_counts_warmup_forecasts_recomputes_and_window_growth():
    runtime = _runtime()
    run_id = runtime.start_run(torch.linspace(1.0, 0.0, 7), "sample_euler", supported_sampler=True)
    for step, sigma in enumerate(torch.linspace(1.0, 1.0 / 6.0, 6)):
        if step in (0, 1, 3, 5):
            _actual_step(runtime, float(sigma), [(LABEL, torch.full((1, 3, 4), float(step)))])
        else:
            _forecast_step(runtime, float(sigma))
    assert runtime.stats.actual_steps == 4
    assert runtime.stats.forecast_steps == 2
    assert runtime.stats.actual_transformer_calls == 4
    assert runtime.stats.current_window == pytest.approx(3.5)
    runtime.end_run(run_id)
    assert runtime.forecaster.history_length == 0


def test_split_branches_are_canonicalized_and_reordered_transactionally():
    runtime = _runtime()
    runtime.start_run(torch.linspace(1.0, 0.0, 5), "sample_euler", supported_sampler=True)
    positive = ((0, "positive"),)
    negative = ((1, "negative"),)
    _actual_step(runtime, 1.0, [(negative, torch.full((1, 3, 4), -1.0)), (positive, torch.ones(1, 3, 4))])
    _actual_step(runtime, 0.75, [(positive, torch.full((1, 3, 4), 2.0)), (negative, torch.full((1, 3, 4), -2.0))])

    decision = runtime.begin_step(torch.tensor([0.5]))
    predictions = {}
    for labels in (negative, positive):
        call_id, actual = runtime.begin_model_call(
            decision["run_id"], decision["step_id"], topology=TOPOLOGY, labels=labels, expected_shape=(1, 3, 4)
        )
        assert not actual
        predictions[labels] = runtime.predict(
            decision["run_id"], decision["step_id"], call_id, device=torch.device("cpu"), dtype=torch.float32
        )
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    assert predictions[positive].mean() > 0
    assert predictions[negative].mean() < 0


def test_incomplete_forecast_requires_whole_step_actual_retry():
    runtime = _runtime()
    runtime.start_run(torch.linspace(1.0, 0.0, 5), "sample_euler", supported_sampler=True)
    labels = (((0, "a"), (1, "b")))
    _actual_step(runtime, 1.0, [(labels, torch.ones(2, 3, 4))])
    _actual_step(runtime, 0.75, [(labels, torch.full((2, 3, 4), 2.0))])
    decision = runtime.begin_step(torch.tensor([0.5]))
    call_id, _ = runtime.begin_model_call(
        decision["run_id"], decision["step_id"], topology=TOPOLOGY, labels=(labels[0],), expected_shape=(1, 3, 4)
    )
    runtime.predict(decision["run_id"], decision["step_id"], call_id, device=torch.device("cpu"), dtype=torch.float32)
    with pytest.raises(ForecastRetryActual):
        runtime.finalize_step(decision["run_id"], decision["step_id"])
    runtime.prepare_actual_retry(decision["run_id"], decision["step_id"], "incomplete branch set")
    for label in labels:
        call_id, actual = runtime.begin_model_call(
            decision["run_id"], decision["step_id"], topology=TOPOLOGY, labels=(label,), expected_shape=(1, 3, 4)
        )
        assert actual
        runtime.observe_actual(decision["run_id"], decision["step_id"], call_id, torch.zeros(1, 3, 4))
    runtime.finalize_step(decision["run_id"], decision["step_id"])
    assert runtime.stats.forecast_fallbacks == 1
    assert runtime.stats.actual_steps == 3


def test_abort_rolls_back_solver_step_id():
    runtime = _runtime(force_actual=True)
    runtime.start_run(torch.tensor([1.0, 0.5, 0.0]), "sample_euler", supported_sampler=True)
    first = runtime.begin_step(torch.tensor([1.0]))
    runtime.abort_step(first["run_id"], first["step_id"])
    repeated = runtime.begin_step(torch.tensor([1.0]))
    assert repeated["step_id"] == first["step_id"] == 0


def test_unsupported_sampler_never_enters_forecast_state():
    runtime = _runtime()
    run_id = runtime.start_run(torch.tensor([1.0, 0.5, 0.0]), "sample_heun", supported_sampler=False)
    assert not runtime.supported_sampler
    assert "not allowlisted" in runtime.disabled_reason
    runtime.end_run(run_id)


def test_invalid_sigma_span_uses_a_neutral_coordinate_on_the_native_path():
    runtime = _runtime()
    runtime.start_run(torch.tensor([1.0, 1.0]), "sample_euler", supported_sampler=True)
    decision = runtime.begin_step(torch.tensor([1.0]))
    assert decision["actual"]
    assert decision["coordinate"] == 0.0
    runtime.abort_step(decision["run_id"], decision["step_id"])
    runtime.end_run(decision["run_id"])


def test_history_dtype_change_disables_forecasting_and_keeps_actual_progress():
    runtime = _runtime()
    runtime.start_run(torch.linspace(1.0, 0.0, 4), "sample_euler", supported_sampler=True)
    _actual_step(runtime, 1.0, [(LABEL, torch.ones(1, 3, 4, dtype=torch.float32))])
    _actual_step(runtime, 2.0 / 3.0, [(LABEL, torch.ones(1, 3, 4, dtype=torch.float16))])
    assert runtime.stats.actual_steps == 2
    assert runtime.stats.disabled
    assert "feature dtype changed" in runtime.disabled_reason


def test_adaptive_window_is_capped_by_the_history_bound():
    runtime = _runtime(flex_window=10.0, max_history=4)
    runtime.start_run(torch.linspace(1.0, 0.0, 6), "sample_euler", supported_sampler=True)
    _actual_step(runtime, 1.0, [(LABEL, torch.zeros(1, 3, 4))])
    _actual_step(runtime, 0.8, [(LABEL, torch.ones(1, 3, 4))])
    _forecast_step(runtime, 0.6)
    _actual_step(runtime, 0.4, [(LABEL, torch.full((1, 3, 4), 3.0))])
    assert runtime.stats.current_window == 4.0


@pytest.mark.parametrize(
    ("flex_window", "tail_actual_steps", "expected_actual", "expected_indices"),
    [
        (0.75, 3, 12, [0, 1, 2, 3, 4, 6, 8, 11, 15, 17, 18, 19]),
        (3.0, 2, 9, [0, 1, 2, 3, 4, 6, 11, 18, 19]),
    ],
)
def test_twenty_step_preset_schedule_counts(flex_window, tail_actual_steps, expected_actual, expected_indices):
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(flex_window=flex_window, tail_actual_steps=tail_actual_steps)
    )
    runtime.start_run(torch.linspace(1.0, 0.0, 21), "sample_euler", supported_sampler=True)
    actual_indices = []
    for step, sigma in enumerate(torch.linspace(1.0, 0.05, 20)):
        decision = runtime.begin_step(sigma)
        call_id, actual = runtime.begin_model_call(
            decision["run_id"],
            decision["step_id"],
            topology=TOPOLOGY,
            labels=LABEL,
            expected_shape=(1, 3, 4),
        )
        if actual:
            actual_indices.append(step)
            runtime.observe_actual(
                decision["run_id"], decision["step_id"], call_id, torch.full((1, 3, 4), float(step))
            )
        else:
            runtime.predict(
                decision["run_id"],
                decision["step_id"],
                call_id,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        runtime.finalize_step(decision["run_id"], decision["step_id"])
    assert runtime.stats.actual_steps == expected_actual
    assert runtime.stats.forecast_steps == 20 - expected_actual
    assert actual_indices == expected_indices
