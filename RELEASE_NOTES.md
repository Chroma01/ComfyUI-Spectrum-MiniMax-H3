# Spectrum MiniMax H3 v0.1.10

Makes Spectrum tolerate valid downstream model-patch paths that complete a solver evaluation without reaching Spectrum's native MiniMax H3 wrapper.

## Fixed

- Accept the downstream patch's successful `predict_noise` result instead of raising `Spectrum H3 solver step completed without an H3 model call`.
- Disable Spectrum forecasting for the remainder of that sampling run after a bypass, preventing history from advancing across an unobserved solver step.
- Immediately release retained forecast history and reset refresh state at the bypass boundary.
- Emit one precise warning per run and report bypassed steps separately from actual and forecast work.
- Preserve normal Spectrum operation when downstream patches, including the current MiniMax H3 Sol-Attn path, continue through the native H3 wrapper.

## Documentation

- Identify the default degree-1, one-step-warmup, one-point-bootstrap schedule as the preliminary performance preset.
- Add a conservative degree-4, five-step-warmup, bootstrap-disabled starting preset for quality-sensitive workflows.
- Expand the fidelity guidance to cover trajectory changes, visual artifacts, generated/reference-audio distortion, and audiovisual validation.
- Record the reported reference-audio result precisely: increasing `degree` and `warmup_steps` helped, and a 30-step run with those increased settings produced clean audio on that setup.

Existing workflow inputs and normal native-H3 sampling behavior are unchanged. If a downstream patch bypasses the native H3 wrapper, that patch remains active while Spectrum safely becomes a passthrough for the rest of the run.

