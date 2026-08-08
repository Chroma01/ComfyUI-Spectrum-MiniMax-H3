from __future__ import annotations

import pytest
import torch

from comfyui_spectrum_h3.experiments import (
    OfflineFeatureArchive,
    OfflineSmoother,
    apply_hidden_residual,
    measure_stream_residual,
)


def _archive(right_value: float) -> OfflineFeatureArchive:
    archive = OfflineFeatureArchive(total_steps=3, sampler_name="sample_euler")
    archive.record_step(0, -1.0, True)
    archive.record_step(1, 0.0, False)
    archive.record_step(2, 1.0, True)
    labels = ((0, "positive"),)
    topology = (("shape", 1),)
    archive.record_actual(
        0,
        -1.0,
        torch.zeros(1, 1, 2, dtype=torch.float16),
        labels=labels,
        topology=topology,
        take_cpu_ownership=True,
    )
    archive.record_actual(
        2,
        1.0,
        torch.full((1, 1, 2), right_value, dtype=torch.float16),
        labels=labels,
        topology=topology,
        take_cpu_ownership=True,
    )
    assert archive.complete(minimum_anchors=2)
    return archive


def test_residual_score_uses_scale_aware_zero_case():
    actual = torch.full((2048,), 1000.0)
    score = measure_stream_residual(actual, actual.clone(), actual.clone(), chunk_bytes=4096)
    assert score.score == 0.0
    assert score.epsilon == pytest.approx(1e-3)
    assert score.chunks == 2


def test_residual_score_compares_forecast_with_hold_baseline():
    actual = torch.full((8,), 3.0)
    shadow = torch.zeros(8)
    hold = torch.full((8,), 2.0)
    score = measure_stream_residual(actual, shadow, hold)
    assert score.forecast_rms == pytest.approx(3.0)
    assert score.hold_rms == pytest.approx(1.0)
    assert score.score == pytest.approx(3.0)


def test_hidden_residual_is_applied_in_bounded_chunks_without_dtype_promotion():
    prediction = torch.ones(4096, dtype=torch.float16)
    residual = torch.full((4096,), 2.0, dtype=torch.float16)
    chunks = apply_hidden_residual(prediction, residual, 0.5, chunk_bytes=4096)
    assert chunks == 4
    assert prediction.dtype is torch.float16
    torch.testing.assert_close(prediction, torch.full_like(prediction, 2.0))


def test_offline_smoother_uses_future_anchor_and_reuses_actual_features_exactly():
    first_archive = _archive(2.0)
    second_archive = _archive(6.0)
    first = OfflineSmoother(
        first_archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
    )
    second = OfflineSmoother(
        second_archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.5,
    )
    kwargs = {"rows": (0,), "device": torch.device("cpu"), "dtype": torch.float16}
    first_middle = first.predict(1, **kwargs)
    second_middle = second.predict(1, **kwargs)
    assert not torch.equal(first_middle, second_middle)
    torch.testing.assert_close(first.predict(0, **kwargs), first_archive.anchors[0].feature)
    torch.testing.assert_close(first.predict(2, **kwargs), first_archive.anchors[1].feature)


def test_offline_local_component_is_bracketing_interpolation():
    archive = _archive(8.0)
    smoother = OfflineSmoother(
        archive,
        degree=1,
        ridge_lambda=0.1,
        blend_weight=0.0,
    )
    middle = smoother.predict(
        1,
        rows=(0,),
        device=torch.device("cpu"),
        dtype=torch.float16,
    )
    torch.testing.assert_close(middle, torch.full_like(middle, 4.0))


def test_offline_archive_requires_a_future_anchor_for_every_forecast():
    archive = OfflineFeatureArchive(total_steps=2, sampler_name="sample_euler")
    archive.record_step(0, -1.0, True)
    archive.record_step(1, 0.0, False)
    archive.record_actual(
        0,
        -1.0,
        torch.zeros(1, 1, 1),
        labels=((0, "positive"),),
        topology=(("shape", 1),),
        take_cpu_ownership=True,
    )
    assert not archive.complete(minimum_anchors=1)
    assert "future actual anchor" in archive.failure_reason
