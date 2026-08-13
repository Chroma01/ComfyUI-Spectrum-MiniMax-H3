from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from . import generic_correction as _generic
from .model_aware import AnchorEvidence, ModelAwareForecastDecision
from .runtime import SpectrumH3Runtime

_EPS = torch.finfo(torch.float32).eps
_CORRECTION_GAIN_LIMIT = 0.25
_RACER_A = 0.3
_RACER_B = 4.0
# Shadow-only threshold sweep. These deliberately span the recent video
# calibration regime rather than becoming shipping constants for MiniMax-H3.
_TRUST_THETAS = (0.15, 0.25, 0.40)

_ORIGINAL_GENERIC_ANCHOR_EVIDENCE: Callable[..., AnchorEvidence | None] | None = None
_ORIGINAL_RUNTIME_DEBUG_SUMMARY: Callable[[SpectrumH3Runtime], str] | None = None


def _tensor_rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value.to(torch.float32).square()))


def _stable_sigmoid(value: float) -> float:
    resolved = float(value)
    if resolved >= 0.0:
        exp_term = math.exp(-min(resolved, 80.0))
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(max(resolved, -80.0))
    return exp_term / (1.0 + exp_term)


def trust_kappa(
    disagreement: float,
    horizon: float,
    *,
    theta: float,
    a: float = _RACER_A,
    b: float = _RACER_B,
) -> float:
    """Cache-only trust coefficient used only for shadow candidate scoring."""
    risk = max(0.0, float(disagreement))
    horizon_decay = math.exp(-float(a) * max(float(horizon) - 1.0, 0.0))
    confidence = _stable_sigmoid(float(b) * (float(theta) - risk))
    return max(0.0, min(1.0, horizon_decay * confidence))


def oracle_segment_kappa(
    actual: torch.Tensor,
    latest: torch.Tensor,
    proposal: torch.Tensor,
) -> torch.Tensor:
    """Best [latest, proposal] interpolation coefficient for diagnostic use."""
    target = (actual - latest).reshape(-1).to(torch.float32)
    direction = (proposal - latest).reshape(-1).to(torch.float32)
    actual_rms = _tensor_rms(actual)
    epsilon = actual_rms.mul(1e-6).clamp_min(_EPS)
    denominator = torch.dot(direction, direction).clamp_min(
        epsilon.square() * max(1, int(direction.numel()))
    )
    return (torch.dot(target, direction) / denominator).clamp(0.0, 1.0)


@dataclass(slots=True)
class _RunningPair:
    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def add(self, x: float, y: float) -> None:
        x_value = float(x)
        y_value = float(y)
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            return
        self.count += 1
        self.sum_x += x_value
        self.sum_y += y_value
        self.sum_x2 += x_value * x_value
        self.sum_y2 += y_value * y_value
        self.sum_xy += x_value * y_value

    def correlation(self) -> float:
        if self.count < 2:
            return 0.0
        n = float(self.count)
        covariance = n * self.sum_xy - self.sum_x * self.sum_y
        variance_x = n * self.sum_x2 - self.sum_x * self.sum_x
        variance_y = n * self.sum_y2 - self.sum_y * self.sum_y
        denominator = math.sqrt(max(0.0, variance_x) * max(0.0, variance_y))
        if denominator <= 1e-12:
            return 0.0
        result = covariance / denominator
        return max(-1.0, min(1.0, result)) if math.isfinite(result) else 0.0


@dataclass(slots=True)
class _StreamProbe:
    count: int = 0
    horizon_sum: float = 0.0
    horizon_max: float = 0.0
    disagreement_sum: float = 0.0
    disagreement_max: float = 0.0
    corrected_ratio_sum: float = 0.0
    delta_bound_gain_sum: float = 0.0
    delta_bound_ratio_sum: float = 0.0
    delta_bound_advantage_sum: float = 0.0
    oracle_ratio_sum: float = 0.0
    oracle_kappa_sum: float = 0.0
    oracle_advantage_sum: float = 0.0
    error_correlation: _RunningPair = field(default_factory=_RunningPair)
    shrink_correlation: _RunningPair = field(default_factory=_RunningPair)
    candidate_ratio_sums: dict[float, float] = field(
        default_factory=lambda: {theta: 0.0 for theta in _TRUST_THETAS}
    )
    candidate_advantage_sums: dict[float, float] = field(
        default_factory=lambda: {theta: 0.0 for theta in _TRUST_THETAS}
    )

    def record(
        self,
        *,
        horizon: float,
        disagreement: float,
        corrected_ratio: float,
        delta_bound_gain: float,
        delta_bound_ratio: float,
        oracle_ratio: float,
        oracle_kappa: float,
        candidate_ratios: dict[float, float],
    ) -> None:
        values = (
            float(horizon),
            float(disagreement),
            float(corrected_ratio),
            float(delta_bound_gain),
            float(delta_bound_ratio),
            float(oracle_ratio),
            float(oracle_kappa),
            *[float(candidate_ratios[theta]) for theta in _TRUST_THETAS],
        )
        if not all(math.isfinite(value) for value in values):
            return
        baseline = max(float(corrected_ratio), 1e-12)
        delta_bound_advantage = (
            float(corrected_ratio) - float(delta_bound_ratio)
        ) / baseline
        oracle_advantage = (float(corrected_ratio) - float(oracle_ratio)) / baseline
        self.count += 1
        self.horizon_sum += float(horizon)
        self.horizon_max = max(self.horizon_max, float(horizon))
        self.disagreement_sum += float(disagreement)
        self.disagreement_max = max(self.disagreement_max, float(disagreement))
        self.corrected_ratio_sum += float(corrected_ratio)
        self.delta_bound_gain_sum += float(delta_bound_gain)
        self.delta_bound_ratio_sum += float(delta_bound_ratio)
        self.delta_bound_advantage_sum += delta_bound_advantage
        self.oracle_ratio_sum += float(oracle_ratio)
        self.oracle_kappa_sum += float(oracle_kappa)
        self.oracle_advantage_sum += oracle_advantage
        self.error_correlation.add(float(disagreement), float(corrected_ratio))
        self.shrink_correlation.add(float(disagreement), 1.0 - float(oracle_kappa))
        for theta in _TRUST_THETAS:
            ratio = float(candidate_ratios[theta])
            self.candidate_ratio_sums[theta] += ratio
            self.candidate_advantage_sums[theta] += (
                float(corrected_ratio) - ratio
            ) / baseline

    def mean(self, value: float) -> float:
        return float(value) / self.count if self.count else 0.0


@dataclass(slots=True)
class _ProbeState:
    run_id: int | None = None
    failures: int = 0
    audio: _StreamProbe = field(default_factory=_StreamProbe)
    video: _StreamProbe = field(default_factory=_StreamProbe)


def _active_run_id(runtime: SpectrumH3Runtime) -> int | None:
    run = getattr(runtime, "_run", None)
    return None if run is None else int(run.run_id)


def _state(runtime: SpectrumH3Runtime) -> _ProbeState:
    run_id = _active_run_id(runtime)
    state = getattr(runtime, "_forecast_trust_probe_state", None)
    if not isinstance(state, _ProbeState) or state.run_id != run_id:
        state = _ProbeState(run_id=run_id)
        runtime._forecast_trust_probe_state = state
    return state


def _combine_samples(
    weights: torch.Tensor,
    samples: list[torch.Tensor],
) -> torch.Tensor:
    if int(weights.numel()) != len(samples):
        raise ValueError("trust probe weights are not aligned with evidence history")
    output = torch.zeros_like(samples[0])
    for weight, sample in zip(weights.tolist(), samples, strict=True):
        if weight != 0.0:
            output.add_(sample, alpha=float(weight))
    return output


def _logical_horizon(step: Any, forecaster: Any) -> float:
    if not forecaster._history:
        return 1.0
    anchor_id = forecaster._history[-1].anchor_id
    if anchor_id is None:
        return 1.0
    return float(max(1, int(step.step_id) - int(anchor_id)))


def _record_shadow_probe(
    runtime: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    decision: ModelAwareForecastDecision,
    evidence: AnchorEvidence,
) -> None:
    if runtime.config.model_aware_mode != "full":
        return
    forecaster = runtime.forecaster
    if forecaster.history_length < 2 or not forecaster._evidence_history:
        return
    if len(forecaster._evidence_history) != forecaster.history_length:
        return

    state = _state(runtime)
    horizon = _logical_horizon(step, forecaster)
    horizon_decay = math.exp(-_RACER_A * max(horizon - 1.0, 0.0))
    for name, start, end in runtime._stream_ranges(step.calls[0]):
        if name == "packed":
            # The shipping generic path already rejects modality-specific packed
            # settings. A packed equal-blend run has no stream-specific signal to
            # learn from, so keep the probe inert rather than invent attribution.
            continue
        stream_probe = getattr(state, name, None)
        stream_evidence = getattr(evidence, name, None)
        if not isinstance(stream_probe, _StreamProbe) or stream_evidence is None:
            continue
        blend = (
            decision.audio_blend_weight
            if name == "audio"
            else decision.video_blend_weight
        )
        gain = (
            decision.audio_correction_gain
            if name == "audio"
            else decision.video_correction_gain
        )
        history_samples = [entry[name] for entry in forecaster._evidence_history]
        if not history_samples:
            continue
        actual = forecaster._sample_segment_device(combined, start, end)
        spectral_weights = forecaster._spectral_weights_configured(
            step.coordinate,
            degree=decision.degree,
            ridge_lambda=decision.ridge_lambda,
        )
        linear_weights = forecaster._linear_weights(step.coordinate)
        spectral = _combine_samples(spectral_weights, history_samples)
        linear = _combine_samples(linear_weights, history_samples)
        blended_weights = (
            spectral_weights
            if blend >= 1.0 - 1e-12
            else linear_weights
            if blend <= 1e-12
            else blend * spectral_weights + (1.0 - blend) * linear_weights
        )
        predicted = _combine_samples(blended_weights, history_samples)
        latest = history_samples[-1]
        previous = history_samples[-2]
        delta = latest - previous
        proposal = predicted + float(gain) * delta

        epsilon = _tensor_rms(actual).mul(1e-6).clamp_min(_EPS)
        hold_rms = _tensor_rms(actual - latest).clamp_min(epsilon)
        disagreement = _tensor_rms(spectral - linear) / _tensor_rms(spectral).clamp_min(
            epsilon
        )
        corrected_ratio = _tensor_rms(actual - proposal) / hold_rms

        projection = float(stream_evidence.residual_projection)
        delta_bound_gain = projection / (
            1.0 + abs(projection) / _CORRECTION_GAIN_LIMIT
        )
        delta_bound = predicted + float(delta_bound_gain) * delta
        delta_bound_ratio = _tensor_rms(actual - delta_bound) / hold_rms

        oracle_kappa = oracle_segment_kappa(actual, latest, proposal)
        oracle = latest + oracle_kappa.to(dtype=proposal.dtype) * (proposal - latest)
        oracle_ratio = _tensor_rms(actual - oracle) / hold_rms

        candidate_ratios: dict[float, torch.Tensor] = {}
        for theta in _TRUST_THETAS:
            kappa = float(horizon_decay) * torch.sigmoid(
                disagreement.new_tensor(_RACER_B * float(theta))
                - _RACER_B * disagreement
            )
            candidate = latest + kappa.to(dtype=proposal.dtype) * (proposal - latest)
            candidate_ratios[theta] = _tensor_rms(actual - candidate) / hold_rms

        values = torch.stack(
            (
                disagreement,
                corrected_ratio,
                delta_bound_ratio,
                oracle_ratio,
                oracle_kappa,
                *[candidate_ratios[theta] for theta in _TRUST_THETAS],
            )
        )
        resolved = values.detach().to(device="cpu", dtype=torch.float32).tolist()
        stream_probe.record(
            horizon=horizon,
            disagreement=float(resolved[0]),
            corrected_ratio=float(resolved[1]),
            delta_bound_gain=delta_bound_gain,
            delta_bound_ratio=float(resolved[2]),
            oracle_ratio=float(resolved[3]),
            oracle_kappa=float(resolved[4]),
            candidate_ratios={
                theta: float(resolved[5 + index])
                for index, theta in enumerate(_TRUST_THETAS)
            },
        )


def _generic_anchor_evidence_with_trust_probe(
    runtime: SpectrumH3Runtime,
    step: Any,
    combined: torch.Tensor,
    decision: ModelAwareForecastDecision,
) -> AnchorEvidence | None:
    if _ORIGINAL_GENERIC_ANCHOR_EVIDENCE is None:
        raise RuntimeError("forecast trust probe was not installed correctly")
    evidence = _ORIGINAL_GENERIC_ANCHOR_EVIDENCE(runtime, step, combined, decision)
    if evidence is None:
        return None
    try:
        _record_shadow_probe(runtime, step, combined, decision, evidence)
    except torch.cuda.OutOfMemoryError:
        raise
    except (RuntimeError, TypeError, ValueError, KeyError, IndexError):
        # This probe is diagnostic and must never disable or perturb the already
        # validated generic correction path. Missing probe evidence is counted
        # instead of disabling the shipping correction path.
        _state(runtime).failures += 1
    return evidence


def _stream_summary(name: str, probe: _StreamProbe) -> str:
    fields = [
        f"trust_probe_{name}_samples={probe.count}",
        f"trust_probe_{name}_horizon_mean={probe.mean(probe.horizon_sum):.6f}",
        f"trust_probe_{name}_horizon_max={probe.horizon_max:.6f}",
        f"trust_probe_{name}_disagreement_mean={probe.mean(probe.disagreement_sum):.6f}",
        f"trust_probe_{name}_disagreement_max={probe.disagreement_max:.6f}",
        f"trust_probe_{name}_corrected_ratio_mean={probe.mean(probe.corrected_ratio_sum):.6f}",
        f"trust_probe_{name}_delta_bound_gain_mean={probe.mean(probe.delta_bound_gain_sum):.6f}",
        f"trust_probe_{name}_delta_bound_ratio_mean={probe.mean(probe.delta_bound_ratio_sum):.6f}",
        f"trust_probe_{name}_delta_bound_advantage_mean={probe.mean(probe.delta_bound_advantage_sum):.6f}",
        f"trust_probe_{name}_oracle_ratio_mean={probe.mean(probe.oracle_ratio_sum):.6f}",
        f"trust_probe_{name}_oracle_kappa_mean={probe.mean(probe.oracle_kappa_sum):.6f}",
        f"trust_probe_{name}_oracle_advantage_mean={probe.mean(probe.oracle_advantage_sum):.6f}",
        f"trust_probe_{name}_error_corr={probe.error_correlation.correlation():.6f}",
        f"trust_probe_{name}_shrink_corr={probe.shrink_correlation.correlation():.6f}",
    ]
    for theta in _TRUST_THETAS:
        suffix = f"{theta:.2f}".replace(".", "p")
        fields.extend(
            (
                f"trust_probe_{name}_theta_{suffix}_ratio_mean={probe.mean(probe.candidate_ratio_sums[theta]):.6f}",
                f"trust_probe_{name}_theta_{suffix}_advantage_mean={probe.mean(probe.candidate_advantage_sums[theta]):.6f}",
            )
        )
    return " ".join(fields)


def _debug_summary(self: SpectrumH3Runtime) -> str:
    if _ORIGINAL_RUNTIME_DEBUG_SUMMARY is None:
        raise RuntimeError("forecast trust probe was not installed correctly")
    summary = _ORIGINAL_RUNTIME_DEBUG_SUMMARY(self)
    state = _state(self)
    return (
        f"{summary} "
        "trust_probe=shadow_only "
        "trust_probe_observer=unblended_spectral_vs_linear "
        "trust_probe_applied=0 "
        f"trust_probe_failures={state.failures} "
        "trust_probe_extra_transformer_nfe=0 "
        f"{_stream_summary('audio', state.audio)} "
        f"{_stream_summary('video', state.video)}"
    )


def install_forecast_trust_probe() -> None:
    """Install a telemetry-only trust-region experiment on top of PR #39."""
    global _ORIGINAL_GENERIC_ANCHOR_EVIDENCE, _ORIGINAL_RUNTIME_DEBUG_SUMMARY
    if getattr(SpectrumH3Runtime, "_forecast_trust_probe_installed", False):
        return
    _ORIGINAL_GENERIC_ANCHOR_EVIDENCE = _generic._generic_anchor_evidence
    _ORIGINAL_RUNTIME_DEBUG_SUMMARY = SpectrumH3Runtime.debug_summary
    _generic._generic_anchor_evidence = _generic_anchor_evidence_with_trust_probe
    SpectrumH3Runtime.debug_summary = _debug_summary
    SpectrumH3Runtime._forecast_trust_probe_installed = True


__all__ = [
    "install_forecast_trust_probe",
    "oracle_segment_kappa",
    "trust_kappa",
]
