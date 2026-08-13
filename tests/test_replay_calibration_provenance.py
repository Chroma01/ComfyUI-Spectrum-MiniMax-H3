from __future__ import annotations

from types import SimpleNamespace

from comfyui_spectrum_h3 import replay_calibration as calibration
from comfyui_spectrum_h3 import replay_calibration_provenance as provenance
from comfyui_spectrum_h3 import sampling
from comfyui_spectrum_h3.config import SpectrumH3Config
from comfyui_spectrum_h3.runtime import SpectrumH3Runtime


def _base_block() -> dict[str, object]:
    return {
        "schema_version": 1,
        "provenance": {
            "source_schema_revision": "pr45-replay-calibration-v1",
            "package_version": "0.2.7",
            "source_revision": None,
            "config_hash": "old",
            "schedule_fingerprint": "schedule",
            "topology_fingerprint": "topology",
            "trace_fingerprint": "old-trace",
        },
        "config": {
            "model_aware_mode": "full",
            "model_aware_risk_threshold": 0.65,
            "offline_smoothing_replay": True,
            "model_aware_replay_generic_correction": False,
        },
        "metadata": {
            "sampler": "sample_er_sde",
            "steps": 25,
            "scheduler": None,
        },
        "target_rows": [
            {
                "target_step_id": 3,
                "coordinate": 0.75,
                "left_anchor_step_id": 2,
                "right_anchor_step_id": 4,
                "current_weight": 0.25,
                "causal_disagreement": 0.4,
                "validation_penalty": 1.5,
                "spectral_gap": 0.2,
                "trace_fingerprint": "old-trace",
            }
        ],
    }


def test_observed_seed_accepts_only_clean_integral_values():
    assert provenance._observed_seed(None) is None
    assert provenance._observed_seed(True) is None
    assert provenance._observed_seed(123) == 123
    assert provenance._observed_seed(123.0) == 123
    assert provenance._observed_seed(123.5) is None
    assert provenance._observed_seed("123") is None


def test_outer_sample_wrapper_exposes_seed_only_during_run(monkeypatch):
    runtime = SpectrumH3Runtime(SpectrumH3Config())
    guider = SimpleNamespace(
        model_options={sampling.BINDING_KEY: sampling.SpectrumH3Binding(runtime)}
    )
    executor = SimpleNamespace(class_obj=guider)
    observed = []

    def original(*_args, **_kwargs):
        observed.append(getattr(runtime, provenance._RUNTIME_SEED_ATTR, None))
        return "result"

    monkeypatch.setattr(provenance, "_ORIGINAL_OUTER_SAMPLE_WRAPPER", original)
    result = provenance._outer_sample_with_provenance(
        executor,
        object(),
        object(),
        object(),
        object(),
        seed=987654321,
    )
    assert result == "result"
    assert observed == [987654321]
    assert not hasattr(runtime, provenance._RUNTIME_SEED_ATTR)


def test_block_provenance_uses_run_config_hash_and_seed_in_trace_fingerprint(monkeypatch):
    runtime = SpectrumH3Runtime(SpectrumH3Config())
    state = calibration._CalibrationState(enabled=True, config_snapshot={})
    monkeypatch.setattr(
        provenance,
        "_ORIGINAL_BUILD_BLOCK",
        lambda _runtime, _state: _base_block(),
    )
    setattr(runtime, provenance._RUNTIME_SEED_ATTR, 42)

    first = provenance._build_block_with_provenance(runtime, state)
    second = provenance._build_block_with_provenance(runtime, state)
    assert first == second
    assert first["provenance"]["seed"] == 42
    assert first["metadata"]["seed_source"] == "ComfyUI OUTER_SAMPLE seed"
    expected_config_hash = calibration._sha256_json(
        {
            "spectrum_config": first["config"],
            "sampler": "sample_er_sde",
            "steps": 25,
            "scheduler": None,
        }
    )
    assert first["provenance"]["config_hash"] == expected_config_hash
    trace = first["provenance"]["trace_fingerprint"]
    assert len(trace) == 64
    assert first["target_rows"][0]["trace_fingerprint"] == trace


def test_seed_changes_trace_fingerprint_but_not_run_config_hash(monkeypatch):
    runtime = SpectrumH3Runtime(SpectrumH3Config())
    state = calibration._CalibrationState(enabled=True, config_snapshot={})
    monkeypatch.setattr(
        provenance,
        "_ORIGINAL_BUILD_BLOCK",
        lambda _runtime, _state: _base_block(),
    )

    setattr(runtime, provenance._RUNTIME_SEED_ATTR, 1)
    first = provenance._build_block_with_provenance(runtime, state)
    setattr(runtime, provenance._RUNTIME_SEED_ATTR, 2)
    second = provenance._build_block_with_provenance(runtime, state)

    assert first["provenance"]["config_hash"] == second["provenance"]["config_hash"]
    assert first["provenance"]["trace_fingerprint"] != second["provenance"]["trace_fingerprint"]


def test_unknown_seed_is_explicit_not_fabricated(monkeypatch):
    runtime = SpectrumH3Runtime(SpectrumH3Config())
    state = calibration._CalibrationState(enabled=True, config_snapshot={})
    monkeypatch.setattr(
        provenance,
        "_ORIGINAL_BUILD_BLOCK",
        lambda _runtime, _state: _base_block(),
    )
    result = provenance._build_block_with_provenance(runtime, state)
    assert result["provenance"]["seed"] is None
    assert result["metadata"]["seed_source"] == "unavailable"
