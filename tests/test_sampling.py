from __future__ import annotations

from types import SimpleNamespace

import pytest

from comfyui_spectrum_h3.sampling import (
    max_consecutive_forecasts,
    min_actual_steps_after_forecast,
    min_tail_actual_steps,
    sampler_is_supported,
    sampler_name,
)


def _sampler(function_name: str) -> SimpleNamespace:
    def sampler_function():
        pass

    sampler_function.__name__ = function_name
    return SimpleNamespace(sampler_function=sampler_function)


@pytest.mark.parametrize(
    "function_name",
    (
        "sample_euler",
        "sample_res_multistep",
        "sample_res_multistep_cfg_pp",
    ),
)
def test_reviewed_single_call_samplers_are_supported(function_name):
    sampler = _sampler(function_name)

    assert sampler_name(sampler) == function_name
    assert sampler_is_supported(sampler)


@pytest.mark.parametrize(
    "function_name",
    (
        "sample_euler_ancestral",
        "sample_res_multistep_ancestral",
        "sample_res_multistep_ancestral_cfg_pp",
        "res_multistep",
        "sample_res_multistep_experimental",
        "sample_heun",
    ),
)
def test_unreviewed_sampler_names_do_not_match_by_prefix(function_name):
    assert not sampler_is_supported(_sampler(function_name))


@pytest.mark.parametrize(
    "function_name",
    (
        "sample_euler",
        "sample_res_multistep",
        "sample_res_multistep_cfg_pp",
    ),
)
def test_supported_h3_samplers_limit_forecast_streaks(function_name):
    assert max_consecutive_forecasts(_sampler(function_name)) == 1


@pytest.mark.parametrize("function_name", ("sample_res_multistep", "sample_res_multistep_cfg_pp"))
def test_res_multistep_policy_refreshes_once_and_protects_tail(function_name):
    sampler = _sampler(function_name)

    assert min_actual_steps_after_forecast(sampler) == 1
    assert min_tail_actual_steps(sampler) == 3


def test_euler_policy_keeps_one_refresh_and_user_tail():
    sampler = _sampler("sample_euler")

    assert min_actual_steps_after_forecast(sampler) == 1
    assert min_tail_actual_steps(sampler) == 0


def test_unsupported_sampler_has_no_forecast_streak_policy():
    sampler = _sampler("sample_euler_ancestral")

    assert max_consecutive_forecasts(sampler) is None
    assert min_actual_steps_after_forecast(sampler) == 0
    assert min_tail_actual_steps(sampler) == 0
