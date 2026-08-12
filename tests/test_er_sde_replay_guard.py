from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfyui_spectrum_h3.sampling import sampler_supports_seeded_replay


def _er_sde_sampler(**options):
    def sample_er_sde():
        pass

    return SimpleNamespace(sampler_function=sample_er_sde, extra_options=options)


def _native_sampler_er_sde_scaler(name: str, eta: float = 0.75):
    if name == "er_sde_noise_scaler":
        def er_sde_noise_scaler(x):
            return x * ((x ** 0.3).exp() + 10.0) ** eta

        scaler = er_sde_noise_scaler
    elif name == "reverse_time_sde_noise_scaler":
        def reverse_time_sde_noise_scaler(x):
            return x ** (eta + 1)

        scaler = reverse_time_sde_noise_scaler
    elif name == "ode_noise_scaler":
        def ode_noise_scaler(x):
            return x

        scaler = ode_noise_scaler
    else:
        raise AssertionError(f"unexpected native scaler name: {name}")

    scaler.__module__ = "comfy_extras.nodes_custom_sampler"
    scaler.__qualname__ = f"SamplerER_SDE.execute.<locals>.{name}"
    return scaler


@pytest.mark.parametrize(
    "scaler_name",
    (
        "er_sde_noise_scaler",
        "reverse_time_sde_noise_scaler",
        "ode_noise_scaler",
    ),
)
def test_native_sampler_er_sde_scalers_are_seeded_replay_safe(scaler_name):
    sampler = _er_sde_sampler(
        s_noise=0.8,
        max_stage=3,
        noise_scaler=_native_sampler_er_sde_scaler(scaler_name),
    )

    assert sampler_supports_seeded_replay(sampler)


def test_custom_er_sde_noise_sampler_remains_replay_unsafe():
    sampler = _er_sde_sampler(noise_sampler=lambda *_args: None)

    assert not sampler_supports_seeded_replay(sampler)


def test_unknown_er_sde_noise_scaler_remains_replay_unsafe():
    sampler = _er_sde_sampler(noise_scaler=lambda value: value)

    assert not sampler_supports_seeded_replay(sampler)


def test_native_named_er_sde_scaler_with_changed_closure_contract_fails_closed():
    mutable_state = []

    def er_sde_noise_scaler(x):
        mutable_state.append(x)
        return x

    er_sde_noise_scaler.__module__ = "comfy_extras.nodes_custom_sampler"
    er_sde_noise_scaler.__qualname__ = (
        "SamplerER_SDE.execute.<locals>.er_sde_noise_scaler"
    )
    sampler = _er_sde_sampler(noise_scaler=er_sde_noise_scaler)

    assert not sampler_supports_seeded_replay(sampler)


def _loaded_names(node: ast.AST) -> set[str]:
    return {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }


def test_comfyui_sampler_er_sde_wires_only_reviewed_native_scaler_closures():
    comfyui_path = os.environ.get("COMFYUI_PATH")
    if not comfyui_path:
        pytest.skip("COMFYUI_PATH is required for native sampler contract tests")

    source = Path(comfyui_path) / "comfy_extras/nodes_custom_sampler.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    sampler_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SamplerER_SDE"
    )
    execute = next(
        node
        for node in sampler_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "execute"
    )

    local_scalers = {
        node.name: node
        for node in execute.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert set(local_scalers) == {
        "er_sde_noise_scaler",
        "reverse_time_sde_noise_scaler",
        "ode_noise_scaler",
    }
    assert _loaded_names(local_scalers["er_sde_noise_scaler"]) <= {"x", "eta", "torch"}
    assert _loaded_names(local_scalers["reverse_time_sde_noise_scaler"]) <= {"x", "eta"}
    assert _loaded_names(local_scalers["ode_noise_scaler"]) <= {"x"}

    ksampler_call = next(
        call
        for call in ast.walk(execute)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "ksampler"
    )
    assert len(ksampler_call.args) >= 2
    options = ksampler_call.args[1]
    assert isinstance(options, ast.Dict)
    option_names = {
        key.value
        for key in options.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert option_names == {"s_noise", "noise_scaler", "max_stage"}
    assert "noise_sampler" not in option_names
