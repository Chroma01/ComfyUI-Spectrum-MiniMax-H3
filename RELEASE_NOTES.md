# Spectrum MiniMax H3 v0.2.0

Adds three default-off trajectory-correction experiments and isolates MiniMax H3 audio from unsafe shared spectral blending. Existing workflows continue to load with every new experiment disabled.

## Audio and modality handling

- Keep `blend_weight` as the video spectral share for saved-workflow compatibility.
- Add `audio_blend_weight`, defaulting to `0.0`, so audio uses local prediction unless explicitly enabled for experiments.
- Apply audio and video history weights independently inside one bounded packed prediction buffer.
- Fail closed when native H3 cannot prove the target audio/video boundary.
- Report configured, causal, and effective per-modality blends in debug summaries.

Matched full-checkpoint runs reproduced speech stutter when a video spectral forecast entered the causal joint audio-video trajectory, even with direct audio blending disabled. The revised offline path fixes that observed seed by capturing with `video=0, audio=0` and applying the configured `video=0.5, audio=0` only during transformer-free replay. Disabling offline replay with the same weights reproduced the stutter. This is a validated case, not a universal quality guarantee.

## Experimental trajectory correction

- `anchor_residual_feedback`: a video-scored actual-refresh guard with no hidden-residual injection, a `1.5` trigger, and a three-refresh budget.
- `selective_rollback_correction`: thresholded deterministic Euler rollback with exact sampler-state restoration and a three-correction budget.
- `offline_smoothing_replay`: local-only capture followed by transformer-free bidirectional replay using exact actual anchors, bracketing interpolation, affine-corrected spectral weights, and leave-one-anchor-out per-modality attenuation.
- The three modes are mutually exclusive while Spectrum is enabled and remain disabled by default.
- Unsupported and recoverably incomplete experimental paths preserve a valid ordinary/native result according to each mode's documented contract; cancellation, OOM, and other fatal exceptions propagate with normal teardown.

## Memory, telemetry, and compatibility

- Share immutable offline anchors with causal history instead of keeping a duplicate archive.
- Keep offline anchors on the selected `history_storage` device; VRAM mode avoids repeated multi-GiB host transfers when sufficient memory is available.
- Add detailed residual, rollback, archive, validation, replay, blend, call-count, timing, and memory telemetry.
- Preserve exact actual anchors, constant hidden trajectories, callback order, cancellation, OOM propagation, clone-local state, and teardown.
- Support native `t2va`, `fl2va`, and `ref2va` conditioning; reference-only rows remain outside forecast history.
- Retain ordinary single-pass Spectrum when all three experiments are disabled.

## Validation

- Full current-ComfyUI contract suite, forecaster smoke test, compilation, and diff checks pass.
- Native tiny-H3 tests cover forced-actual equivalence, zero-transformer forecast replay, output-head behavior, modality splitting, and visual/audio reference conditioning.
- CUDA-only storage/parity checks remain skipped in CPU CI; full-checkpoint evidence comes from the documented user runs.

Recommended configuration for the reproduced speech case:

```text
offline_smoothing_replay = true
blend_weight = 0.5
audio_blend_weight = 0.0
```

The other two correction modes remain available for research. All three settings are experimental and require exact-seed video, audio, timing, and memory validation on the intended workflow.
