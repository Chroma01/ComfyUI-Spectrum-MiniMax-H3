# Spectrum MiniMax H3 v0.1.0

Initial alpha release of the standalone Spectrum-style acceleration node for ComfyUI's native MiniMax H3 audio-video model.

## Highlights

- Forecasts selected post-transformer H3 features while preserving the native current-step output heads, reconstruction, sigma mapping, and return structure.
- Keeps runtime and history state isolated per model clone and rolls incomplete split-branch transactions back to a complete native step.
- Supports native Euler and Euler ancestral sampling, with explicit native fallback for unsupported samplers, incompatible topology, invalid forecasts, and multi-GPU parallel sampling.
- Bounds retained history on CPU and streams forecast accumulation in chunks to avoid persistent full-feature FP32 coefficient or right-hand-side tensors.
- Leaves the separate FLUX-focused ComfyUI-Spectrum-Proper repository unchanged.

## Validation

- GitHub Actions passes against native ComfyUI commit `e377e263049f9338b4d12a3dd417b36ae62948ff`.
- 37 automated tests cover forecasting mathematics, scheduling, rollback, clone isolation, native-path equivalence, and zero transformer-block execution on forecast steps.
- CodeRabbit review feedback was evaluated and all review threads were resolved.

## Current limits

Full MiniMax H3 checkpoint generation, reference modes, decoded output quality, audiovisual synchronization, long-duration memory use, VRAM/RSS peaks, and end-to-end speedup remain unverified. Presets and performance expectations are provisional.
