# Spectrum MiniMax H3 v0.2.1

Makes the validated H3 audio-quality correction the standard default path.

## Corrected default

- `offline_smoothing_replay` now defaults to `true` in the runtime configuration, node schema, and Python call signature.
- The mode is no longer classified as experimental.
- The default remains `blend_weight=0.5` for video and `audio_blend_weight=0.0` for audio.
- Offline capture uses causal `video=0, audio=0`; configured per-modality weights are applied only during transformer-free replay.

This default follows the matched full-checkpoint result from v0.2.0 development. Single-pass `video=0.5, audio=0` reproduced degraded speech and stuttering because video forecasts fed into later joint audio-video transformer calls. Offline capture/replay with the same weights removed the defect and restored high-quality audio on that seed while retaining the preferred image result. Disabling offline replay brought the defect back.

## Retained research modes

- `anchor_residual_feedback` remains experimental and default-off.
- `selective_rollback_correction` remains experimental and default-off.
- Both remain mutually exclusive with each other and with the standard offline path. Disable `offline_smoothing_replay` before enabling either research mode.
- Explicitly setting all three trajectory modes to `false` retains the single-pass comparison path.

## Upgrade behavior

New nodes use offline replay automatically. Workflows created before v0.2.0 did not store this input and therefore also receive the new default. Workflows saved with v0.2.0 may retain a serialized `offline_smoothing_replay=false`; enable it once in those existing nodes to use the corrected path.

Offline replay retains every actual hidden anchor plus cloned sampling inputs. `history_storage=system_ram` remains the default; `vram` avoids large host transfers when enough VRAM is available. Unsupported samplers run one valid native pass, and an incomplete replay archive returns the valid local-only first-pass result. Cancellation, OOM, and other fatal exceptions preserve normal teardown and propagation.

## Validation

- Default-value tests cover the dataclass, ComfyUI node schema, and Python call signature.
- The outer-sampler integration test now proves that the omitted/default setting executes capture plus transformer-free replay.
- Existing single-pass, residual-feedback, rollback, native-equivalence, `t2va`, `fl2va`, and `ref2va` contracts remain covered.
