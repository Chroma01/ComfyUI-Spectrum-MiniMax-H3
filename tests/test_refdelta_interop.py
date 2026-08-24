from __future__ import annotations

import pytest
import torch

from comfyui_spectrum_h3.er_sde_stochastic import (
    ERSDEStepDescriptor,
    ERSDEStochasticTracker,
)
from comfyui_spectrum_h3.refdelta_interop import RefDeltaInteropBridge
from comfyui_spectrum_h3.sampling import _refdelta_sampler_contract


def _installed_refdelta():
    try:
        from comfyui_refdelta_solver.config import RefDeltaSamplerConfig
        from comfyui_refdelta_solver.sampler import sample_refdelta_er_sde
    except ImportError:
        pytest.skip("reviewed RefDelta package is not on PYTHONPATH")
    return RefDeltaSamplerConfig, sample_refdelta_er_sde


def test_installed_refdelta_api_v1_is_admitted_for_exact_increment_ownership():
    config_type, function = _installed_refdelta()

    accepted, reason, external_increment = _refdelta_sampler_contract(
        function,
        {"config": config_type(), "s_noise": 1.0, "max_stage": 3},
    )

    assert accepted, reason
    assert external_increment


def test_refdelta_native_equivalence_uses_native_increment_tracking():
    config_type, function = _installed_refdelta()
    config = config_type(
        adaptive_order=False,
        stochastic_adaptation_strength=0.0,
        trajectory_correction=False,
        telemetry=False,
    )

    accepted, reason, external_increment = _refdelta_sampler_contract(
        function,
        {"config": config, "s_noise": 1.0, "max_stage": 3},
    )

    assert accepted, reason
    assert not external_increment


def test_bridge_and_tracker_transfer_exact_gated_increment_end_to_end():
    noise = torch.tensor([[1.0, -2.0]])
    tracker = ERSDEStochasticTracker(
        noise_sampler=lambda _sigma, _sigma_next: noise,
        noise_scaler=lambda value: value**2,
        effective_s_noise=1.0,
        max_stage=3,
        debug=False,
        run_id=19,
        external_increment=True,
    )
    bridge = RefDeltaInteropBridge(run_id=19, tracker=tracker)
    actual = ERSDEStepDescriptor(19, 0, "actual", None, False)
    forecast = ERSDEStepDescriptor(19, 1, "forecast", None, True)

    bridge.note_model_result(actual)
    assert bridge.model_result_is_actual(0)
    tracker.noise_scaler(torch.tensor(0.5))
    tracker.noise_scaler(torch.tensor(1.0))
    tracker.noise_sampler(torch.tensor(0.8), torch.tensor(0.4))
    gated_increment = torch.tensor([[0.15, -0.3]])
    bridge.publish_stochastic_increment(0, gated_increment)

    raw = torch.tensor([[3.0, 4.0]]) + gated_increment
    corrected = tracker.consume(raw, forecast)
    bridge.note_model_result(forecast)

    torch.testing.assert_close(corrected, torch.tensor([[3.0, 4.0]]), rtol=0, atol=0)
    assert not bridge.model_result_is_actual(1)
