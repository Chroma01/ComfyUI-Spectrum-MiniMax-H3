from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import trust_probe as trust_module
from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.experiments import OfflineFeatureArchive, OfflineSmoother
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
from comfyui_spectrum_h3.trust_probe import (
    apply_trust_to_history_weights,
    oracle_segment_kappa,
    trust_kappa,
)


def _archive(*, packed: bool = False) -> OfflineFeatureArchive:
    archive = OfflineFeatureArchive(
        total_steps=4,
        sampler_name="sample_er_sde",
        history_storage="system_ram",
    )
    labels = ("branch",)
    topology = (
        ()
        if packed
        else (("target_audio_rows", 1), ("target_video_rows", 1))
    )
    features = {
        0: torch.tensor([[[0.0, 0.2], [1.0, 1.2]]]),
        2: torch.tensor([[[0.8, 1.0], [1.8, 2.0]]]),
        3: torch.tensor([[[1.2, 1.4], [2.2, 2.4]]]),
    }
    coordinates = {0: -1.0, 1: -0.6, 2: 0.2, 3: 1.0}
    for step_id in range(4):
        actual = step_id in features
        if actual:
            archive.record_actual(
                step_id,
                coordinates[step_id],
                features[step_id],
                labels=labels,
                topology=topology,
                take_ownership=False,
            )
        archive.record_step(step_id, coordinates[step_id], actual)
    assert archive.complete(minimum_anchors=2)
    return archive


def _attach_trust(
    archive: OfflineFeatureArchive,
    *,
    audio_kappa: float,
    video_kappa: float,
) -> trust_module._TrustAggregate:
    aggregate = trust_module._TrustAggregate()
    archive._model_aware_trust_aggregate = aggregate
    archive._model_aware_trust_forecasts = {
        1: trust_module._ForecastTrustDecision(
            step_id=1,
            horizon=1.0,
            latest_anchor_id=0,
            audio=trust_module._StreamTrustDecision(
                disagreement=0.7,
                kappa=audio_kappa,
            ),
            video=trust_module._StreamTrustDecision(
                disagreement=0.3,
                kappa=video_kappa,
            ),
            compute_seconds=0.0,
            scalar_transfer_seconds=0.0,
        )
    }
    return aggregate


def test_trust_kappa_decreases_with_disagreement_and_horizon():
    low_risk = trust_kappa(0.05, 1.0, theta=0.25)
    high_risk = trust_kappa(0.50, 1.0, theta=0.25)
    long_horizon = trust_kappa(0.05, 3.0, theta=0.25)

    assert 0.0 < high_risk < low_risk < 1.0
    assert 0.0 < long_horizon < low_risk


@pytest.mark.parametrize("risk", [0.0, 0.15, 1.0, 100.0])
@pytest.mark.parametrize("horizon", [1.0, 2.0, 10.0])
def test_trust_kappa_is_finite_and_bounded(risk, horizon):
    result = trust_kappa(risk, horizon, theta=0.15)
    assert torch.isfinite(torch.tensor(result))
    assert 0.0 <= result <= 1.0


def test_oracle_segment_kappa_recovers_best_interpolation():
    latest = torch.tensor([0.0, 0.0])
    proposal = torch.tensor([2.0, 0.0])
    actual = torch.tensor([1.0, 0.0])

    kappa = oracle_segment_kappa(actual, latest, proposal)

    assert float(kappa) == pytest.approx(0.5)
    corrected = latest + kappa * (proposal - latest)
    assert torch.equal(corrected, actual)


def test_oracle_segment_kappa_never_extrapolates_past_segment():
    latest = torch.tensor([0.0, 0.0])
    proposal = torch.tensor([1.0, 0.0])

    beyond = oracle_segment_kappa(torch.tensor([3.0, 0.0]), latest, proposal)
    behind = oracle_segment_kappa(torch.tensor([-2.0, 0.0]), latest, proposal)

    assert float(beyond) == pytest.approx(1.0)
    assert float(behind) == pytest.approx(0.0)


def test_history_weight_trust_endpoints_and_affine_invariant():
    baseline = torch.tensor([-0.2, 0.3, 0.9], dtype=torch.float32)

    unchanged = apply_trust_to_history_weights(baseline, 1.0, 2)
    held = apply_trust_to_history_weights(baseline, 0.0, 2)
    halfway = apply_trust_to_history_weights(baseline, 0.5, 2)

    assert torch.equal(unchanged, baseline)
    assert torch.equal(held, torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(halfway, torch.tensor([-0.1, 0.15, 0.95]))
    assert float(halfway.sum()) == pytest.approx(float(baseline.sum()))


def test_default_config_disables_applied_trust_and_rejects_non_boolean():
    assert SpectrumH3Config().model_aware_trust_shrinkage is False
    with pytest.raises(TypeError, match="model_aware_trust_shrinkage"):
        SpectrumH3Config(model_aware_trust_shrinkage="yes")


def test_insufficient_history_returns_baseline_no_trust_decision():
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            model_aware_mode="full",
            model_aware_trust_shrinkage=True,
        )
    )
    result = trust_module._compute_forecast_trust(
        runtime,
        SimpleNamespace(),
        SimpleNamespace(degree=1, ridge_lambda=0.1, forecast_horizon=1.0),
        coordinate=0.0,
    )
    assert result is None


def test_offline_replay_applies_audio_video_kappa_independently():
    archive = _archive()
    aggregate = _attach_trust(
        archive,
        audio_kappa=0.0,
        video_kappa=1.0,
    )
    smoother = OfflineSmoother(
        archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
        audio_blend_weight=0.0,
    )

    audio = smoother._forecast_weights[(1, 0, 0)]
    video = smoother._forecast_weights[(1, 0, 1)]

    assert torch.equal(audio, torch.tensor([1.0, 0.0, 0.0]))
    assert not torch.equal(video, audio)
    assert aggregate.applications == 2
    assert aggregate.failures == 0


def test_offline_replay_kappa_one_is_exact_baseline():
    baseline_archive = _archive()
    baseline = OfflineSmoother(
        baseline_archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
        audio_blend_weight=0.0,
    )

    trusted_archive = _archive()
    aggregate = _attach_trust(
        trusted_archive,
        audio_kappa=1.0,
        video_kappa=1.0,
    )
    trusted = OfflineSmoother(
        trusted_archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
        audio_blend_weight=0.0,
    )

    for key, weights in baseline._forecast_weights.items():
        assert torch.equal(trusted._forecast_weights[key], weights)
    assert aggregate.applications == 2


def test_offline_replay_packed_topology_does_not_fabricate_stream_split():
    archive = _archive(packed=True)
    aggregate = _attach_trust(
        archive,
        audio_kappa=0.0,
        video_kappa=0.0,
    )
    smoother = OfflineSmoother(
        archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
        audio_blend_weight=0.5,
    )

    assert aggregate.applications == 0
    assert aggregate.failures == 0
    assert (1, 0, 0) in smoother._forecast_weights


def test_offline_replay_trust_is_deterministic():
    def build():
        archive = _archive()
        _attach_trust(archive, audio_kappa=0.25, video_kappa=0.4)
        return OfflineSmoother(
            archive,
            degree=1,
            ridge_lambda=0.1,
            blend_weight=0.5,
            audio_blend_weight=0.0,
        )

    first = build()
    second = build()
    for key in first._forecast_weights:
        assert torch.equal(first._forecast_weights[key], second._forecast_weights[key])
    first_prediction = first.predict(
        1,
        rows=(0,),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second_prediction = second.predict(
        1,
        rows=(0,),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert torch.equal(first_prediction, second_prediction)


def test_non_oom_trust_failure_falls_back_and_oom_propagates(monkeypatch):
    runtime = SimpleNamespace(
        _run=SimpleNamespace(run_id=1),
        _step=SimpleNamespace(step_id=2),
        _offline_phase=None,
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("test failure")

    monkeypatch.setattr(trust_module, "_compute_forecast_trust", fail)
    assert trust_module._ensure_step_trust(
        runtime,
        SimpleNamespace(),
        SimpleNamespace(),
        coordinate=0.0,
    ) is None
    assert trust_module._state(runtime).trust.failures == 1

    runtime._step = SimpleNamespace(step_id=3)

    def oom(*_args, **_kwargs):
        raise torch.cuda.OutOfMemoryError("oom")

    monkeypatch.setattr(trust_module, "_compute_forecast_trust", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        trust_module._ensure_step_trust(
            runtime,
            SimpleNamespace(),
            SimpleNamespace(),
            coordinate=0.0,
        )


def test_trust_state_resets_between_runtime_runs():
    runtime = SpectrumH3Runtime(SpectrumH3Config())
    sigmas = torch.tensor([1.0, 0.5, 0.0])

    run_id = runtime.start_run(sigmas, "sample_euler", supported_sampler=True)
    trust_module._state(runtime).trust.failures = 7
    runtime.end_run(run_id)

    next_run = runtime.start_run(sigmas, "sample_euler", supported_sampler=True)
    assert trust_module._state(runtime).trust.failures == 0
    runtime.end_run(next_run)


def test_er_sde_tail_policy_and_extra_nfe_invariant_remain_visible():
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            model_aware_mode="full",
            model_aware_trust_shrinkage=True,
        )
    )
    sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    run_id = runtime.start_run(
        sigmas,
        "sample_er_sde",
        supported_sampler=True,
    )
    assert runtime._run is not None
    assert runtime._run.min_tail_actual_steps >= 2
    summary = runtime.debug_summary()
    assert "model_aware_trust_extra_transformer_nfe=0" in summary
    runtime.end_run(run_id)


def test_debug_summary_distinguishes_shadow_probe_and_opt_in_controller():
    disabled = SpectrumH3Runtime(SpectrumH3Config(model_aware_mode="full"))
    disabled_summary = disabled.debug_summary()
    assert "trust_probe=shadow_only" in disabled_summary
    assert "trust_probe_observer=unblended_spectral_vs_linear" in disabled_summary
    assert "trust_probe_applied=0" in disabled_summary
    assert "trust_probe_failures=0" in disabled_summary
    assert "trust_probe_extra_transformer_nfe=0" in disabled_summary
    assert "trust_probe_audio_samples=0" in disabled_summary
    assert "trust_probe_video_samples=0" in disabled_summary
    assert "model_aware_trust_enabled=0" in disabled_summary
    assert "model_aware_trust_applied=0" in disabled_summary
    assert "feature3_applied_correction=generic_scalar_latest_delta" in disabled_summary

    enabled = SpectrumH3Runtime(
        SpectrumH3Config(
            model_aware_mode="full",
            model_aware_trust_shrinkage=True,
        )
    )
    enabled_summary = enabled.debug_summary()
    assert "model_aware_trust_enabled=1" in enabled_summary
    assert "model_aware_trust_applied=0" in enabled_summary
    assert "model_aware_trust_path=causal_single_pass" in enabled_summary
    assert "model_aware_trust_failures=0" in enabled_summary
    assert "model_aware_trust_compute_s=0.000000" in enabled_summary
    assert "model_aware_trust_scalar_transfer_s=0.000000" in enabled_summary
    assert "model_aware_trust_weight_apply_s=0.000000" in enabled_summary
    assert "model_aware_trust_extra_transformer_nfe=0" in enabled_summary
