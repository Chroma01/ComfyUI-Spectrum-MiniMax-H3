from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

RES_VARIANTS = (
    "sample_res_multistep",
    "sample_res_multistep_cfg_pp",
    "sample_res_multistep_ancestral",
    "sample_res_multistep_ancestral_cfg_pp",
)


def _native_sampling_functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    comfyui_path = os.environ.get("COMFYUI_PATH")
    if not comfyui_path:
        pytest.skip("COMFYUI_PATH is required for native sampler contract tests")
    source_path = Path(comfyui_path) / "comfy" / "k_diffusion" / "sampling.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _named_calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name
    ]


def test_native_res_multistep_makes_one_model_call_per_solver_iteration():
    function = _native_sampling_functions()["res_multistep"]
    loops = [node for node in function.body if isinstance(node, (ast.For, ast.AsyncFor))]

    assert len(loops) == 1
    assert len(_named_calls(loops[0], "model")) == 1
    assert len(_named_calls(function, "model")) == 1


@pytest.mark.parametrize("function_name", RES_VARIANTS)
def test_native_res_variant_delegates_once_to_reviewed_core(function_name):
    function = _native_sampling_functions()[function_name]

    assert len(_named_calls(function, "res_multistep")) == 1
    assert not _named_calls(function, "model")
