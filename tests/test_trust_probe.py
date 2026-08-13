from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from comfyui_spectrum_h3 import trust_probe as trust_module
from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.experiments import OfflineFeatureArchive, OfflineSmoother
from comfyui_spectrum_h3.model_aware import ModelForecastabilityProfile, ProfileLookup
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


def _shadow_record() -> trust_module._ReplayShadowRecord:
    return trust_module._ReplayShadowRecord(
        step_id=2,
        coordinate=0.2,
        latest_anchor_id=0,
        stream_name="audio",
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.0,
        correction_gain=0.0,
        disagreement=0.4,
        kappa=0.2,
    )


def _profile_lookup() -> ProfileLookup:
    return ProfileLookup(
        profile=ModelForecastabilityProfile(
            cache_key=("base", "patches"),
            base_model_identity="fake:base",
            patch_identity="patches",
            active_patch_count=0,
            active_patch_keys=0,
            recognized_lora_count=0,
            unknown_patch_count=0,
            sampled_base_tensors=8,
            profile_confidence=1.0,
            aggregate_sensitivity=0.2,
            patch_perturbation=0.0,
            final_block_perturbation=0.0,
            audio_sensitivity=0.8,
            video_sensitivity=1.2,
            audio_head_weight=None,
            video_head_weight=None,
            audio_head_gram_diagonal=None,
            video_head_gram_diagonal=None,
            forecast_risk_prior=0.2,
            build_seconds=0.001,
            estimated_bytes=1024,
            transient_workspace_bytes=4096,
        ),
        cache_hit=True,
        lookup_seconds=0.0,
    )


def _persistence_archive() -> tuple[
    OfflineFeatureArchive,
    dict[int, torch.Tensor],
    dict[int, float],
    tuple[tuple[str, int], ...],
]:
    archive = OfflineFeatureArchive(
        total_steps=6,
        sampler_name="sample_euler",
        history_storage="system_ram",
    )
    topology = (("target_audio_rows", 1), ("target_video_rows", 1))
    labels = ("branch",)
    coordinates = {
        0: -1.0,
        1: -0.6,
        2: -0.2,
        3: 0.2,
        4: 0.6,
        5: 1.0,
    }
    features = {
        0: torch.tensor([[[0.0, 0.2], [1.0, 1.2]]]),
        1: torch.tensor([[[0.4, 0.7], [1.3, 1.7]]]),
        3: torch.tensor([[[1.1, 1.5], [2.0, 2.5]]]),
        5: torch.tensor([[[2.0, 2.5], [3.2, 3.8]]]),
    }
    for step_id in range(6):
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
    return archive, features, coordinates, topology


def _counted_er_sde_first_pass(trust_enabled: bool) -> dict[str, object]:
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            degree=1,
            max_history=6,
            warmup_steps=2,
            tail_actual_steps=1,
            bootstrap_first_forecast=False,
            model_aware_mode="full",
            model_aware_risk_threshold=1.0,
            model_aware_trust_shrinkage=trust_enabled,
            offline_smoothing_replay=True,
            audio_blend_weight=0.0,
            blend_weight=0.5,
        )
    )
    runtime.set_model_profile(_profile_lookup())
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
                feature = torch.tensor(
                    [[[base, base + 0.25], [1.5 * base, 1.5 * base + 0.5]]],
                    dtype=torch.float32,
                )
                runtime.observe_actual(
                    decision["run_id"],
                    decision["step_id"],
                    call_id,
                    feature,
                )
            else:
                predicted = runtime.predict(
                    decision["run_id"],
                    decision["step_id"],
                    call_id,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                assert predicted is not None
            runtime.finalize_step(decision["run_id"], decision["step_id"])
        actual_calls = runtime.stats.actual_transformer_calls
        fallbacks = runtime.stats.forecast_fallbacks
    finally:
        runtime.end_run(run_id)

    assert runtime.complete_offline_capture()
    archive = runtime.offline_archive
    aggregate = (
        getattr(archive, "_model_aware_trust_aggregate", None)
        if archive is not None
        else None
    )
    result = {
        "decisions": tuple(decisions),
        "reasons": tuple(reasons),
        "actual_calls": actual_calls,
        "fallbacks": fallbacks,
        "applications": (
            aggregate.applications
            if isinstance(aggregate, trust_module._TrustAggregate)
            else 0
        ),
        "failures": (
            aggregate.failures
            if isinstance(aggregate, trust_module._TrustAggregate)
            else 0
        ),
    }
    runtime.release_offline_archive()
    return result


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
    with pytest.raises(ValueError, match="requires model_aware_mode='full'"):
        SpectrumH3Config(
            model_aware_mode="schedule",
            model_aware_trust_shrinkage=True,
        ).validate()


def test_forecast_trust_fits_shared_weights_once_for_audio_and_video(monkeypatch):
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            model_aware_mode="full",
            model_aware_trust_shrinkage=True,
        )
    )
    segments = (("audio", 0, 1), ("video", 1, 2))
    runtime.forecaster.update(
        -1.0,
        torch.tensor([[[0.0, 0.3], [1.0, 1.4]]]),
        anchor_id=0,
        evidence_segments=segments,
        take_ownership=False,
    )
    runtime.forecaster.update(
        -0.4,
        torch.tensor([[[0.4, 0.9], [1.5, 2.1]]]),
        anchor_id=1,
        evidence_segments=segments,
        take_ownership=False,
    )
    call = SimpleNamespace(
        topology=(("target_audio_rows", 1), ("target_video_rows", 1)),
        expected_shape=(1, 2, 2),
    )
    decision = SimpleNamespace(degree=1, ridge_lambda=0.1, forecast_horizon=1.0)
    counts = {"spectral": 0, "linear": 0}
    original_spectral = runtime.forecaster._spectral_weights_configured
    original_linear = runtime.forecaster._linear_weights

    def spectral(*args, **kwargs):
        counts["spectral"] += 1
        return original_spectral(*args, **kwargs)

    def linear(*args, **kwargs):
        counts["linear"] += 1
        return original_linear(*args, **kwargs)

    monkeypatch.setattr(runtime.forecaster, "_spectral_weights_configured", spectral)
    monkeypatch.setattr(runtime.forecaster, "_linear_weights", linear)

    resolved = trust_module._compute_forecast_trust(
        runtime,
        call,
        decision,
        coordinate=0.0,
    )

    assert resolved is not None
    assert counts == {"spectral": 1, "linear": 1}


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


def test_offline_first_pass_persists_trust_for_only_matching_replay_step(monkeypatch):
    archive, features, coordinates, topology = _persistence_archive()
    runtime = SpectrumH3Runtime(
        SpectrumH3Config(
            model_aware_mode="full",
            model_aware_trust_shrinkage=True,
        )
    )
    runtime._offline_phase = "first_pass"
    runtime._offline_archive = archive
    runtime._run = SimpleNamespace(run_id=1)
    segments = (("audio", 0, 1), ("video", 1, 2))
    for anchor_id in (0, 1):
        runtime.forecaster.update(
            coordinates[anchor_id],
            features[anchor_id],
            anchor_id=anchor_id,
            evidence_segments=segments,
            take_ownership=False,
        )
    decision = SimpleNamespace(degree=1, ridge_lambda=0.1, forecast_horizon=1.0)
    runtime._step = SimpleNamespace(
        step_id=2,
        coordinate=coordinates[2],
        model_aware_decision=decision,
    )
    call = SimpleNamespace(topology=topology, expected_shape=(1, 2, 2))
    monkeypatch.setattr(runtime, "_model_aware_enabled", lambda: True)

    runtime._prediction_segments(call)

    persisted = archive._model_aware_trust_forecasts[2]
    assert persisted.step_id == 2
    assert persisted.latest_anchor_id == 1
    assert 0.0 <= persisted.audio.kappa < 1.0
    assert 0.0 <= persisted.video.kappa < 1.0

    baseline_archive, _features, _coordinates, _topology = _persistence_archive()
    baseline = OfflineSmoother(
        baseline_archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
        audio_blend_weight=0.0,
    )
    trusted = OfflineSmoother(
        archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
        audio_blend_weight=0.0,
    )

    step2_keys = [key for key in baseline._forecast_weights if key[0] == 2]
    step4_keys = [key for key in baseline._forecast_weights if key[0] == 4]
    assert step2_keys and step4_keys
    assert any(
        not torch.equal(trusted._forecast_weights[key], baseline._forecast_weights[key])
        for key in step2_keys
    )
    assert all(
        torch.equal(trusted._forecast_weights[key], baseline._forecast_weights[key])
        for key in step4_keys
    )


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


def test_replay_shadow_setup_failure_does_not_abort_applied_replay(monkeypatch):
    archive = _archive()
    aggregate = _attach_trust(archive, audio_kappa=0.25, video_kappa=0.4)
    archive._model_aware_trust_replay_shadow_records = [_shadow_record()]

    def fail_shadow_setup(*_args, **_kwargs):
        raise ValueError("diagnostic setup failed")

    monkeypatch.setattr(trust_module, "_sample_archive_stream", fail_shadow_setup)
    smoother = OfflineSmoother(
        archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
        audio_blend_weight=0.0,
    )

    assert aggregate.applications == 2
    assert aggregate.failures == 0
    assert aggregate.replay_shadow_failures == 1
    assert (1, 0, 0) in smoother._forecast_weights


def test_replay_shadow_setup_oom_still_propagates(monkeypatch):
    archive = _archive()
    _attach_trust(archive, audio_kappa=0.25, video_kappa=0.4)
    archive._model_aware_trust_replay_shadow_records = [_shadow_record()]

    def oom(*_args, **_kwargs):
        raise torch.cuda.OutOfMemoryError("oom")

    monkeypatch.setattr(trust_module, "_sample_archive_stream", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        OfflineSmoother(
            archive,
            degree=1,
            ridge_lambda=0.1,
            blend_weight=0.5,
            audio_blend_weight=0.0,
        )


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


def test_er_sde_trust_preserves_counted_schedule_transformer_calls_and_tail():
    disabled = _counted_er_sde_first_pass(False)
    enabled = _counted_er_sde_first_pass(True)

    assert enabled["decisions"] == disabled["decisions"]
    assert enabled["actual_calls"] == disabled["actual_calls"]
    assert enabled["actual_calls"] == sum(enabled["decisions"])
    assert enabled["fallbacks"] == disabled["fallbacks"] == 0
    assert enabled["decisions"][-2:] == (True, True)
    assert enabled["reasons"][-2:] == ("final actual tail", "final actual tail")
    assert enabled["applications"] > 0
    assert enabled["failures"] == 0


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
