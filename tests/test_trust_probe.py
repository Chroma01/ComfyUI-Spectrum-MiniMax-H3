from __future__ import annotations

import pytest
import torch

from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime
from comfyui_spectrum_h3.trust_probe import oracle_segment_kappa, trust_kappa


def test_trust_kappa_decreases_with_disagreement_and_horizon():
    low_risk = trust_kappa(0.05, 1.0, theta=0.25)
    high_risk = trust_kappa(0.50, 1.0, theta=0.25)
    long_horizon = trust_kappa(0.05, 3.0, theta=0.25)

    assert 0.0 < high_risk < low_risk < 1.0
    assert 0.0 < long_horizon < low_risk


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


def test_debug_summary_declares_probe_shadow_only():
    runtime = SpectrumH3Runtime(SpectrumH3Config(model_aware_mode="full"))
    summary = runtime.debug_summary()

    assert "trust_probe=shadow_only" in summary
    assert "trust_probe_observer=unblended_spectral_vs_linear" in summary
    assert "trust_probe_applied=0" in summary
    assert "trust_probe_failures=0" in summary
    assert "trust_probe_extra_transformer_nfe=0" in summary
    assert "trust_probe_audio_samples=0" in summary
    assert "trust_probe_video_samples=0" in summary
    assert "feature3_applied_correction=generic_scalar_latest_delta" in summary
