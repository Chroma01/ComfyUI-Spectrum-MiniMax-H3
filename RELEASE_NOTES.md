# Spectrum MiniMax H3 v0.1.5

Fixes MiniMax H3 sampling on Apple MPS and includes the documentation and licensing corrections made since v0.1.4.

## Fixed

- Move detached solver timestep tensors to CPU before converting them to `float64`, avoiding the unsupported MPS `float64` conversion that stopped sampling at step 0.
- Apply the same transfer-before-cast ordering to sigma-schedule normalization, which contained the same latent defect.
- Centralize the conversion path so run initialization and per-step coordinate calculation cannot diverge.
- Correct the repository's GPL-3.0-or-later licensing files and package metadata.

## Validation

- Add regression coverage that explicitly enforces `detach -> CPU transfer -> float64 cast -> flatten`.
- Preserve CPU values, output shape, device, and dtype through the shared conversion helper.
- Confirm the fix on a physical Apple M4 Max system with a complete 20-step RES multistep run: 14 actual H3 transformer evaluations, 6 forecasted evaluations, and zero fallbacks.
- Retain the existing native MiniMax H3 fixture, scheduler, transaction, fallback, and forecasting test coverage.

## Documentation

- Clarify that Spectrum can produce trajectory deviations and localized degradation during fast or brief motion.
- Record the community-tested Radeon AI PRO R9700 / ROCm configuration and its measured warm-run result as a scoped compatibility datapoint.

## Scope

The MPS hardware confirmation covers the reported Apple M4 Max, PyTorch 2.10.0, ComfyUI 0.30.0, RES multistep configuration. It does not establish compatibility for every Apple Silicon model, PyTorch release, sampler, workflow topology, or MiniMax H3 configuration.
