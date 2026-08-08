from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any

import torch

from .forecast import HistoryWeightForecaster


DEFAULT_CHUNK_BYTES = 32 * 1024 * 1024


def _chunk_elements(chunk_bytes: int) -> int:
    if chunk_bytes < 4096:
        raise ValueError("chunk_bytes must be >= 4096")
    return max(1024, int(chunk_bytes) // torch.tensor([], dtype=torch.float32).element_size())


def tensor_all_finite(value: torch.Tensor, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> bool:
    flat = value.detach().reshape(-1)
    chunk = _chunk_elements(chunk_bytes)
    for offset in range(0, flat.numel(), chunk):
        if not bool(torch.isfinite(flat.narrow(0, offset, min(chunk, flat.numel() - offset))).all().item()):
            return False
    return True


@dataclass(frozen=True, slots=True)
class StreamResidualScore:
    forecast_rms: float
    hold_rms: float
    actual_rms: float
    epsilon: float
    score: float
    chunks: int


def measure_stream_residual(
    actual: torch.Tensor,
    shadow: torch.Tensor,
    hold: torch.Tensor,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> StreamResidualScore:
    if actual.shape != shadow.shape or actual.shape != hold.shape:
        raise ValueError("actual, shadow, and hold outputs must have identical shapes")
    if not actual.dtype.is_floating_point or not shadow.dtype.is_floating_point or not hold.dtype.is_floating_point:
        raise ValueError("residual measurement requires floating-point outputs")
    count = actual.numel()
    if count == 0:
        raise ValueError("residual measurement cannot reduce an empty output")

    actual_flat = actual.detach().reshape(-1)
    shadow_flat = shadow.detach().reshape(-1)
    hold_flat = hold.detach().reshape(-1)
    chunk = _chunk_elements(chunk_bytes)
    actual_sq = 0.0
    forecast_sq = 0.0
    hold_sq = 0.0
    chunks = 0
    for offset in range(0, count, chunk):
        length = min(chunk, count - offset)
        actual_chunk = actual_flat.narrow(0, offset, length).to(torch.float32)
        shadow_chunk = shadow_flat.narrow(0, offset, length).to(device=actual_chunk.device, dtype=torch.float32)
        hold_chunk = hold_flat.narrow(0, offset, length).to(device=actual_chunk.device, dtype=torch.float32)
        actual_sq += float(torch.sum(actual_chunk * actual_chunk, dtype=torch.float32).item())
        forecast_delta = actual_chunk - shadow_chunk
        hold_delta = actual_chunk - hold_chunk
        forecast_sq += float(torch.sum(forecast_delta * forecast_delta, dtype=torch.float32).item())
        hold_sq += float(torch.sum(hold_delta * hold_delta, dtype=torch.float32).item())
        chunks += 1

    actual_rms = math.sqrt(actual_sq / count)
    forecast_rms = math.sqrt(forecast_sq / count)
    hold_rms = math.sqrt(hold_sq / count)
    if not all(math.isfinite(value) for value in (actual_rms, forecast_rms, hold_rms)):
        raise ValueError("residual measurement produced a nonfinite RMS")

    epsilon = max(actual_rms * 1e-6, torch.finfo(torch.float32).eps)
    if forecast_rms <= epsilon and hold_rms <= epsilon:
        score = 0.0
    else:
        score = forecast_rms / max(hold_rms, epsilon)
    if not math.isfinite(score):
        raise ValueError("residual measurement produced a nonfinite score")
    return StreamResidualScore(
        forecast_rms=forecast_rms,
        hold_rms=hold_rms,
        actual_rms=actual_rms,
        epsilon=epsilon,
        score=score,
        chunks=chunks,
    )


def apply_hidden_residual(
    prediction: torch.Tensor,
    residual: torch.Tensor,
    gain: float,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> int:
    if prediction.shape != residual.shape:
        raise ValueError("prediction and residual shapes must match")
    if not prediction.dtype.is_floating_point or not residual.dtype.is_floating_point:
        raise ValueError("hidden residual correction requires floating-point tensors")
    gain_value = float(gain)
    if not math.isfinite(gain_value) or not 0.0 <= gain_value <= 1.0:
        raise ValueError("residual gain must be finite and in [0, 1]")
    if not prediction.is_contiguous():
        raise ValueError("hidden residual correction requires a contiguous prediction tensor")
    prediction_flat = prediction.view(-1)
    residual_flat = residual.detach().reshape(-1)
    chunk = _chunk_elements(chunk_bytes)
    chunks = 0
    for offset in range(0, prediction_flat.numel(), chunk):
        length = min(chunk, prediction_flat.numel() - offset)
        target = prediction_flat.narrow(0, offset, length)
        corrected = target.to(torch.float32)
        corrected.add_(
            residual_flat.narrow(0, offset, length).to(device=target.device, dtype=torch.float32),
            alpha=gain_value,
        )
        target.copy_(corrected.to(target.dtype))
        chunks += 1
    return chunks


@dataclass(frozen=True, slots=True)
class OfflineStepRecord:
    step_id: int
    coordinate: float
    actual: bool


@dataclass(slots=True)
class OfflineAnchor:
    step_id: int
    coordinate: float
    feature: torch.Tensor


class OfflineFeatureArchive:
    def __init__(self, *, total_steps: int, sampler_name: str) -> None:
        self.total_steps = int(total_steps)
        self.sampler_name = str(sampler_name)
        self.steps: list[OfflineStepRecord] = []
        self.anchors: list[OfflineAnchor] = []
        self.labels: tuple[Any, ...] | None = None
        self.topology: tuple[Any, ...] | None = None
        self.feature_shape: tuple[int, ...] | None = None
        self.feature_dtype: torch.dtype | None = None
        self.valid = True
        self.failure_reason: str | None = None

    @property
    def tensor_bytes(self) -> int:
        return sum(anchor.feature.numel() * anchor.feature.element_size() for anchor in self.anchors)

    @property
    def estimated_tensor_bytes(self) -> int:
        if not self.anchors:
            return 0
        feature = self.anchors[0].feature
        return self.total_steps * feature.numel() * feature.element_size()

    def invalidate(self, reason: str) -> None:
        if self.valid:
            self.valid = False
            self.failure_reason = str(reason)

    def record_step(self, step_id: int, coordinate: float, actual: bool) -> None:
        if not self.valid:
            return
        expected = len(self.steps)
        if int(step_id) != expected:
            self.invalidate(f"offline step sequence changed: expected {expected}, got {step_id}")
            return
        self.steps.append(OfflineStepRecord(int(step_id), float(coordinate), bool(actual)))

    def record_actual(
        self,
        step_id: int,
        coordinate: float,
        feature: torch.Tensor,
        *,
        labels: tuple[Any, ...],
        topology: tuple[Any, ...],
        take_cpu_ownership: bool,
    ) -> None:
        if not self.valid:
            return
        shape = tuple(int(value) for value in feature.shape)
        if self.labels is None:
            self.labels = tuple(labels)
            self.topology = tuple(topology)
            self.feature_shape = shape
            self.feature_dtype = feature.dtype
        elif tuple(labels) != self.labels or tuple(topology) != self.topology:
            self.invalidate("offline branch labels or topology changed across actual anchors")
            return
        elif shape != self.feature_shape or feature.dtype != self.feature_dtype:
            self.invalidate("offline actual feature shape or dtype changed")
            return

        detached = feature.detach()
        if take_cpu_ownership and detached.device.type == "cpu" and detached.is_contiguous():
            archived = detached
        else:
            archived = detached.to(device="cpu", dtype=feature.dtype, copy=True).contiguous()
        self.anchors.append(OfflineAnchor(int(step_id), float(coordinate), archived))

    def complete(self, *, minimum_anchors: int) -> bool:
        if len(self.steps) != self.total_steps:
            self.invalidate(
                f"offline first pass recorded {len(self.steps)} of {self.total_steps} logical steps"
            )
        actual_ids = [step.step_id for step in self.steps if step.actual]
        anchor_ids = [anchor.step_id for anchor in self.anchors]
        if actual_ids != anchor_ids:
            self.invalidate("offline actual-step schedule does not match the retained anchor archive")
        if len(self.anchors) < int(minimum_anchors):
            self.invalidate(
                f"offline smoothing requires at least {minimum_anchors} actual anchors"
            )
        if any(
            not step.actual
            and not any(anchor.step_id < step.step_id for anchor in self.anchors)
            for step in self.steps
        ):
            self.invalidate("offline forecast step has no earlier actual anchor")
        if any(
            not step.actual
            and not any(anchor.step_id > step.step_id for anchor in self.anchors)
            for step in self.steps
        ):
            self.invalidate("offline forecast step has no future actual anchor")
        return self.valid

    def release(self) -> None:
        self.steps.clear()
        self.anchors.clear()
        self.labels = None
        self.topology = None
        self.feature_shape = None
        self.feature_dtype = None


class OfflineSmoother:
    def __init__(
        self,
        archive: OfflineFeatureArchive,
        *,
        degree: int,
        ridge_lambda: float,
        blend_weight: float,
        chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    ) -> None:
        if not archive.valid or not archive.anchors or archive.labels is None:
            raise ValueError("offline archive is incomplete")
        self.archive = archive
        self.blend_weight = float(blend_weight)
        self._anchor_ids = [anchor.step_id for anchor in archive.anchors]
        self._anchor_by_step = {anchor.step_id: anchor for anchor in archive.anchors}
        self._forecaster = HistoryWeightForecaster(
            degree=degree,
            ridge_lambda=ridge_lambda,
            max_history=max(len(archive.anchors), degree + 1, 2),
            chunk_bytes=chunk_bytes,
            history_storage="system_ram",
        )
        for anchor in archive.anchors:
            self._forecaster.update(anchor.coordinate, anchor.feature, take_ownership=True)

    def _weights_for_step(self, record: OfflineStepRecord) -> torch.Tensor:
        spectral = self._forecaster.spectral_weights(record.coordinate)
        position = bisect.bisect_left(self._anchor_ids, record.step_id)
        if position == 0 or position == len(self._anchor_ids):
            raise RuntimeError("offline forecast requires bracketing actual anchors")
        left = self.archive.anchors[position - 1]
        right = self.archive.anchors[position]
        spacing = right.coordinate - left.coordinate
        if abs(spacing) <= 1e-12:
            raise RuntimeError("offline bracketing anchors have duplicate coordinates")
        ratio = (record.coordinate - left.coordinate) / spacing
        local = torch.zeros(len(self.archive.anchors), dtype=torch.float32)
        local[position - 1] = 1.0 - ratio
        local[position] = ratio
        return self.blend_weight * spectral + (1.0 - self.blend_weight) * local

    def predict(
        self,
        step_id: int,
        *,
        rows: tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        record = self.archive.steps[int(step_id)]
        anchor = self._anchor_by_step.get(record.step_id)
        if anchor is not None:
            weights = torch.zeros(len(self.archive.anchors), dtype=torch.float32)
            weights[self._anchor_ids.index(record.step_id)] = 1.0
        else:
            weights = self._weights_for_step(record)
        return self._forecaster.predict_with_weights(
            weights,
            rows=rows,
            device=device,
            dtype=dtype,
        )
