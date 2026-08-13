from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import generic_correction as generic_module
from comfyui_spectrum_h3 import replay_component_shadow as component_module
from comfyui_spectrum_h3 import replay_generic_correction_gate as gate_module
from comfyui_spectrum_h3 import replay_trust_shadow as replay_module
from comfyui_spectrum_h3 import trust_probe as trust_module
from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.experiments import OfflineFeatureArchive, OfflineSmoother
from comfyui_spectrum_h3.model_aware import ModelAwareForecastDecision
from comfyui_spectrum_h3.nodes import SpectrumApplyMiniMaxH3
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime


def _decision(
    *,
    audio_correction: float = -0.20,
    video_correction: float = -0.10,
    audio_blend: float = 0.0,
    video_blend: float = 0.5,
) -> ModelAwareForecastDecision:
    return ModelAwareForecastDecision(
        trajectory_risk=0.2,
        model_risk=0.2,
        patch_risk=0.1,
        combined_risk=0.2,
        confidence=0.8,
        ridge_lambda=0.1,
        degree=1,
        audio_blend_weight=audio_blend,
        video_blend_weight=video_blend,
        audio_correction_gain=audio_correction,
        video_correction_gain=video_correction,
        forecast_horizon=1.0,
        force_actual=False,
    )


def _archive(
    *,
    with_correction: bool = True,
    packed: bool = False,
) -> OfflineFeatureArchive:
    archive = OfflineFeatureArchive(
        total_steps=5,
        sampler_name="sample_er_sde",
        history_storage="system_ram",
    )
    coordinates = torch.linspace(-1.0, 1.0, 5).tolist()
    correction = -0.20 if with_correction else 0.0
    video_correction = -0.10 if with_correction else 0.0
    audio_blend = 0.5 if packed else 0.0
    decision = _decision(
        audio_correction=correction,
        video_correction=(correction if packed else video_correction),
        audio_blend=audio_blend,
        video_blend=0.5,
    )
    for step_id, coordinate in enumerate(coordinates):
        actual = step_id % 2 == 0
        archive.record_step(
            step_id,
            coordinate,
            actual,
            model_aware_decision=(None if actual else decision),
        )
    topology = () if packed else (("target_audio_rows", 1), ("target_video_rows", 1))
    values = {
        0: ((0.0, 0.2, -0.1), (0.0, 0.1, 0.3)),
        2: ((1.0, 1.4, 0.8), (1.4, 1.0, 1.6)),
        4: ((3.0, 2.6, 3.2), (2.2, 3.0, 2.7)),
    }
    for step_id in (0, 2, 4):
        feature = torch.tensor([values[step_id]], dtype=torch.float32)
        archive.record_actual(
            step_id,
            coordinates[step_id],
            feature,
            labels=((0, "positive"),),
            topology=topology,
            take_ownership=False,
        )
    assert archive.complete(minimum_anchors=2)
    return archive


def _smoother(
    archive: OfflineFeatureArchive,
    *,
    gate_enabled: bool | None = None,
    packed: bool = False,
) -> OfflineSmoother:
    if gate_enabled is not None:
        setattr(archive, gate_module._ARCHIVE_GATE_ATTR, gate_enabled)
    return OfflineSmoother(
        archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
        audio_blend_weight=(0.5 if packed else 0.0),
    )


def _forecast_weight_items(smoother: OfflineSmoother):
    return {
        key: value.clone()
        for key, value in smoother._forecast_weights.items()
    }


def _assert_weight_maps_equal(left, right) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert torch.equal(left[key], right[key]), key


def _counted_first_pass(gate_enabled: bool) -> dict[str, object]:
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=6,
            warmup_steps=2,
            tail_actual_steps=1,
            bootstrap_first_forecast=False,
            model_aware_mode="full",
            model_aware_risk_threshold=1.0,
            model_aware_replay_generic_correction=gate_enabled,
            offline_smoothing_replay=True,
            audio_blend_weight=0.0,
            blend_weight=0.5,
        )
    )
    sigmas = torch.tensor([1.0, 0.82, 0.64, 0.46, 0.28, 0.10, 0.0])
    topology = (("target_audio_rows", 1), ("target_video_rows", 1))
    labels = ("branch",)
    runtime.begin_offline_capture(total_steps=6, sampler_name="sample_er_sde")
    run_id = runtime.start_run(
        sigmas,
        "sample_er_sde",
        supported_sampler=True,
        max_consecutive_forecasts=1,
        min_actual_steps_after_forecast=1,
        min_tail_actual_steps=0,
    )
    decisions: list[bool] = []
    reasons: list[str] = []
    try:
        for step_id, sigma in enumerate(sigmas[:-1]):
            decision = runtime.begin_step(sigma)
            decisions.append(bool(decision["actual"]))
            reasons.append(str(decision["reason"]))
            call_id, actual = runtime.begin_model_call(
                decision["run_id"],
                decision["step_id"],
                topology=topology,
                labels=labels,
                expected_shape=(1, 2, 2),
            )
            if actual:
                base = float(step_id + 1)
                runtime.observe_actual(
                    decision["run_id"],
                    decision["step_id"],
                    call_id,
                    torch.tensor(
                        [[[base, base + 0.25], [1.5 * base, 1.5 * base + 0.5]]],
                        dtype=torch.float32,
                    ),
                )
            else:
                prediction = runtime.predict(
                    decision["run_id"],
                    decision["step_id"],
                    call_id,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                assert prediction is not None
            runtime.finalize_step(decision["run_id"], decision["step_id"])
        actual_calls = runtime.stats.actual_transformer_calls
        fallback_count = runtime.stats.forecast_fallbacks
    finally:
        runtime.end_run(run_id)
    assert runtime.complete_offline_capture()
    archive = runtime.offline_archive
    assert archive is not None
    telemetry = getattr(archive, gate_module._ARCHIVE_TELEMETRY_ATTR)
    result = {
        "decisions": tuple(decisions),
        "reasons": tuple(reasons),
        "actual_calls": actual_calls,
        "fallbacks": fallback_count,
        "telemetry": telemetry,
    }
    runtime.release_offline_archive()
    return result


def _shadow_archive(gate_enabled: bool) -> OfflineFeatureArchive:
    archive = OfflineFeatureArchive(
        total_steps=9,
        sampler_name="sample_er_sde",
        history_storage="system_ram",
    )
    coordinates = {step: -1.0 + 0.25 * step for step in range(9)}
    decision = _decision()
    features = {
        0: torch.tensor([[[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]]]),
        2: torch.tensor([[[1.0, 1.4, 0.9], [2.0, 2.5, 1.8]]]),
        4: torch.tensor([[[1.4, 2.0, 1.2], [2.8, 3.2, 2.4]]]),
        6: torch.tensor([[[3.0, 2.4, 3.4], [4.5, 3.8, 4.2]]]),
        8: torch.tensor([[[4.0, 5.0, 4.5], [6.0, 6.8, 5.5]]]),
    }
    for step_id in range(9):
        actual = step_id in features
        archive.record_step(
            step_id,
            coordinates[step_id],
            actual,
            model_aware_decision=(None if actual else decision),
        )
        if actual:
            archive.record_actual(
                step_id,
                coordinates[step_id],
                features[step_id],
                labels=("branch",),
                topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
                take_ownership=False,
            )
    assert archive.complete(minimum_anchors=2)
    aggregate = trust_module._TrustAggregate()
    archive._model_aware_trust_aggregate = aggregate
    records: list[trust_module._ReplayShadowRecord] = []
    for index, step_id in enumerate((2, 4, 6)):
        for stream_name, blend, gain in (
            ("audio", 0.0, -0.20),
            ("video", 0.5, -0.10),
        ):
            records.append(
                trust_module._ReplayShadowRecord(
                    step_id=step_id,
                    coordinate=coordinates[step_id],
                    latest_anchor_id=step_id - 2,
                    stream_name=stream_name,
                    degree=1,
                    ridge_lambda=0.1,
                    blend_weight=blend,
                    correction_gain=gain,
                    disagreement=(0.2, 0.5, 0.8)[index],
                    kappa=0.2,
                )
            )
    archive._model_aware_trust_replay_shadow_records = records
    setattr(archive, replay_module._ARCHIVE_SHADOW_ONLY_ATTR, True)
    setattr(archive, gate_module._ARCHIVE_GATE_ATTR, gate_enabled)
    return archive


def test_replay_generic_correction_setting_defaults_to_current_behavior_and_round_trips():
    config = SpectrumH3Config()
    assert config.model_aware_replay_generic_correction is True
    round_trip = SpectrumH3Config(**asdict(config))
    assert round_trip == config
    with pytest.raises(TypeError, match="model_aware_replay_generic_correction"):
        SpectrumH3Config(model_aware_replay_generic_correction="no")

    optional = SpectrumApplyMiniMaxH3.INPUT_TYPES()["optional"]
    assert optional["model_aware_replay_generic_correction"][1]["default"] is True


def test_replay_generic_correction_is_inert_outside_full_offline_replay():
    single_pass = SpectrumH3Config(
        model_aware_mode="full",
        offline_smoothing_replay=False,
        model_aware_replay_generic_correction=False,
    ).validate()
    non_full = SpectrumH3Config(
        model_aware_mode="schedule_confidence",
        model_aware_replay_generic_correction=False,
    ).validate()
    assert single_pass.model_aware_replay_generic_correction is False
    assert non_full.model_aware_replay_generic_correction is False

    runtime = SpectrumH3Runtime(non_full)
    runtime.begin_offline_capture(total_steps=2, sampler_name="sample_euler")
    assert runtime.offline_archive is not None
    assert getattr(runtime.offline_archive, gate_module._ARCHIVE_GATE_ATTR) is True
    runtime.release_offline_archive()


def test_default_and_explicit_enabled_replay_are_exactly_identical():
    default = _smoother(_archive(with_correction=True))
    explicit = _smoother(_archive(with_correction=True), gate_enabled=True)
    _assert_weight_maps_equal(
        _forecast_weight_items(default),
        _forecast_weight_items(explicit),
    )
    telemetry = getattr(default.archive, gate_module._ARCHIVE_TELEMETRY_ATTR)
    assert telemetry.enabled is True
    assert telemetry.path == "current_causal_gain_transfer"
    assert telemetry.applications == 4
    assert telemetry.skips == 0
    assert telemetry.extra_transformer_nfe == 0
    assert default.model_aware_offline_correction_applications == 4


def test_disabled_replay_equals_uncorrected_blend_b_and_preserves_archive():
    archive = _archive(with_correction=True)
    original_steps = archive.steps
    original_decisions = tuple(record.model_aware_decision for record in archive.steps)
    disabled = _smoother(archive, gate_enabled=False)
    uncorrected = _smoother(_archive(with_correction=False), gate_enabled=True)
    enabled = _smoother(_archive(with_correction=True), gate_enabled=True)

    _assert_weight_maps_equal(
        _forecast_weight_items(disabled),
        _forecast_weight_items(uncorrected),
    )
    assert any(
        not torch.equal(disabled._forecast_weights[key], enabled._forecast_weights[key])
        for key in disabled._forecast_weights
    )
    assert archive.steps is original_steps
    assert tuple(record.model_aware_decision for record in archive.steps) == original_decisions
    assert archive.steps[1].model_aware_decision is not None
    assert archive.steps[1].model_aware_decision.audio_correction_gain == pytest.approx(-0.20)

    telemetry = getattr(archive, gate_module._ARCHIVE_TELEMETRY_ATTR)
    assert telemetry.enabled is False
    assert telemetry.path == "disabled_replay_geometry_experiment"
    assert telemetry.applications == 0
    assert telemetry.skips == 4
    assert telemetry.extra_transformer_nfe == 0
    assert disabled.model_aware_offline_correction_applications == 0
    assert disabled.model_aware_offline_correction_seconds == pytest.approx(0.0)


def test_disabled_gate_preserves_exact_anchors_validation_and_blends():
    disabled_archive = _archive(with_correction=True)
    disabled = _smoother(disabled_archive, gate_enabled=False)
    uncorrected_archive = _archive(with_correction=False)
    uncorrected = _smoother(uncorrected_archive, gate_enabled=True)

    assert disabled._validation_scores == uncorrected._validation_scores
    assert disabled.effective_blend_stream_stats == uncorrected.effective_blend_stream_stats
    assert disabled.attenuated_prediction_counts == uncorrected.attenuated_prediction_counts
    assert disabled.local_only_prediction_counts == uncorrected.local_only_prediction_counts

    kwargs = {"rows": (0,), "device": torch.device("cpu"), "dtype": torch.float32}
    for step_id in (0, 2, 4):
        expected = next(
            anchor.feature
            for anchor in disabled_archive.anchors
            if anchor.step_id == step_id
        )
        torch.testing.assert_close(disabled.predict(step_id, **kwargs), expected)


def test_replay_gate_does_not_change_causal_generic_weight_segments():
    call = SimpleNamespace(
        topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
        expected_shape=(1, 2, 2),
    )
    decision = _decision()

    def build(gate_enabled: bool):
        runtime = SpectrumH3Runtime(
            SpectrumH3Config(
                model_aware_mode="full",
                offline_smoothing_replay=False,
                model_aware_replay_generic_correction=gate_enabled,
            )
        )
        runtime.forecaster.update(
            -1.0,
            torch.tensor([[[0.0, 0.2], [1.0, 1.2]]]),
            anchor_id=0,
            take_ownership=False,
        )
        runtime.forecaster.update(
            -0.4,
            torch.tensor([[[0.8, 1.0], [1.8, 2.0]]]),
            anchor_id=1,
            take_ownership=False,
        )
        return generic_module._weight_segments(
            runtime,
            call,
            decision,
            coordinate=0.2,
        )

    enabled = build(True)
    disabled = build(False)
    assert len(enabled) == len(disabled) == 2
    for enabled_segment, disabled_segment in zip(enabled, disabled, strict=True):
        assert enabled_segment[:2] == disabled_segment[:2]
        assert torch.equal(enabled_segment[2], disabled_segment[2])


def test_first_pass_schedule_nfe_and_er_sde_tail_are_gate_independent():
    enabled = _counted_first_pass(True)
    disabled = _counted_first_pass(False)
    assert enabled["decisions"] == disabled["decisions"]
    assert enabled["reasons"] == disabled["reasons"]
    assert enabled["actual_calls"] == disabled["actual_calls"]
    assert enabled["fallbacks"] == disabled["fallbacks"] == 0
    assert enabled["decisions"][-2:] == (True, True)
    assert enabled["actual_calls"] == sum(enabled["decisions"])
    assert enabled["telemetry"].extra_transformer_nfe == 0
    assert disabled["telemetry"].extra_transformer_nfe == 0


def test_packed_topology_gate_removes_only_scalar_replay_correction():
    disabled = _smoother(
        _archive(with_correction=True, packed=True),
        gate_enabled=False,
        packed=True,
    )
    uncorrected = _smoother(
        _archive(with_correction=False, packed=True),
        gate_enabled=True,
        packed=True,
    )
    _assert_weight_maps_equal(
        _forecast_weight_items(disabled),
        _forecast_weight_items(uncorrected),
    )
    telemetry = getattr(disabled.archive, gate_module._ARCHIVE_TELEMETRY_ATTR)
    assert telemetry.applications == 0
    assert telemetry.skips == 2


def test_abcd_shadow_remains_counterfactual_d_under_b_production():
    enabled_archive = _shadow_archive(True)
    disabled_archive = _shadow_archive(False)
    enabled = _smoother(enabled_archive, gate_enabled=True)
    disabled = _smoother(disabled_archive, gate_enabled=False)

    assert any(
        not torch.equal(enabled._forecast_weights[key], disabled._forecast_weights[key])
        for key in enabled._forecast_weights
    )

    enabled_component = getattr(
        enabled_archive,
        component_module._ARCHIVE_COMPONENT_ATTR,
    )
    disabled_component = getattr(
        disabled_archive,
        component_module._ARCHIVE_COMPONENT_ATTR,
    )
    for stream_name in ("audio", "video"):
        left = getattr(enabled_component, stream_name)
        right = getattr(disabled_component, stream_name)
        assert left.count == right.count > 0
        for candidate in component_module._CANDIDATES:
            assert left.candidate_ratio_sums[candidate] == pytest.approx(
                right.candidate_ratio_sums[candidate]
            )
            assert left.candidate_advantage_sums[candidate] == pytest.approx(
                right.candidate_advantage_sums[candidate]
            )

    disabled_telemetry = getattr(
        disabled_archive,
        gate_module._ARCHIVE_TELEMETRY_ATTR,
    )
    assert disabled_telemetry.enabled is False
    assert disabled_telemetry.skips > 0


def test_gate_telemetry_distinguishes_correction_application_from_skip():
    enabled_archive = _archive(with_correction=True)
    enabled = _smoother(enabled_archive, gate_enabled=True)
    disabled_archive = _archive(with_correction=True)
    disabled = _smoother(disabled_archive, gate_enabled=False)

    runtime_enabled = SpectrumH3Runtime(SpectrumH3Config())
    runtime_enabled._offline_archive = enabled_archive
    runtime_enabled._offline_smoother = enabled
    summary_enabled = runtime_enabled.debug_summary()
    assert "model_aware_replay_generic_correction_enabled=1" in summary_enabled
    assert "model_aware_replay_generic_correction_path=current_causal_gain_transfer" in summary_enabled
    assert "model_aware_replay_generic_correction_applications=4" in summary_enabled
    assert "model_aware_replay_generic_correction_skips=0" in summary_enabled
    assert "model_aware_replay_generic_correction_extra_transformer_nfe=0" in summary_enabled

    runtime_disabled = SpectrumH3Runtime(SpectrumH3Config())
    runtime_disabled._offline_archive = disabled_archive
    runtime_disabled._offline_smoother = disabled
    summary_disabled = runtime_disabled.debug_summary()
    assert "model_aware_replay_generic_correction_enabled=0" in summary_disabled
    assert (
        "model_aware_replay_generic_correction_path=disabled_replay_geometry_experiment"
        in summary_disabled
    )
    assert "model_aware_replay_generic_correction_applications=0" in summary_disabled
    assert "model_aware_replay_generic_correction_skips=4" in summary_disabled
    assert "model_aware_replay_generic_correction_extra_transformer_nfe=0" in summary_disabled


def test_trust_toggle_does_not_implicitly_change_replay_correction_gate():
    for trust_enabled in (False, True):
        runtime = SpectrumH3Runtime(
            SpectrumH3Config(
                model_aware_mode="full",
                model_aware_trust_shrinkage=trust_enabled,
                model_aware_replay_generic_correction=True,
            )
        )
        runtime.begin_offline_capture(total_steps=2, sampler_name="sample_er_sde")
        assert runtime.offline_archive is not None
        assert getattr(runtime.offline_archive, gate_module._ARCHIVE_GATE_ATTR) is True
        runtime.release_offline_archive()


def test_gate_restores_archive_steps_when_underlying_builder_raises(monkeypatch):
    archive = _archive(with_correction=True)
    smoother = _smoother(archive, gate_enabled=True)
    setattr(archive, gate_module._ARCHIVE_GATE_ATTR, False)
    original_steps = archive.steps

    def ordinary_failure(_smoother):
        assert _smoother.archive.steps is not original_steps
        raise ValueError("ordinary replay failure")

    monkeypatch.setattr(
        gate_module,
        "_ORIGINAL_OFFLINE_BUILD_FORECAST_WEIGHTS",
        ordinary_failure,
    )
    with pytest.raises(ValueError, match="ordinary replay failure"):
        gate_module._build_forecast_weights_with_replay_generic_gate(smoother)
    assert archive.steps is original_steps

    def oom_failure(_smoother):
        assert _smoother.archive.steps is not original_steps
        raise torch.cuda.OutOfMemoryError("synthetic OOM")

    monkeypatch.setattr(
        gate_module,
        "_ORIGINAL_OFFLINE_BUILD_FORECAST_WEIGHTS",
        oom_failure,
    )
    with pytest.raises(torch.cuda.OutOfMemoryError, match="synthetic OOM"):
        gate_module._build_forecast_weights_with_replay_generic_gate(smoother)
    assert archive.steps is original_steps


def test_gate_state_is_archive_scoped_and_resets_with_new_capture():
    disabled_runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            model_aware_mode="full",
            model_aware_replay_generic_correction=False,
        )
    )
    disabled_runtime.begin_offline_capture(total_steps=2, sampler_name="sample_er_sde")
    disabled_archive = disabled_runtime.offline_archive
    assert disabled_archive is not None
    assert getattr(disabled_archive, gate_module._ARCHIVE_GATE_ATTR) is False
    disabled_runtime.release_offline_archive()

    enabled_runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            model_aware_mode="full",
            model_aware_replay_generic_correction=True,
        )
    )
    enabled_runtime.begin_offline_capture(total_steps=2, sampler_name="sample_er_sde")
    enabled_archive = enabled_runtime.offline_archive
    assert enabled_archive is not None
    assert enabled_archive is not disabled_archive
    assert getattr(enabled_archive, gate_module._ARCHIVE_GATE_ATTR) is True
    assert not hasattr(enabled_archive, gate_module._ARCHIVE_TELEMETRY_ATTR)
    enabled_runtime.release_offline_archive()
