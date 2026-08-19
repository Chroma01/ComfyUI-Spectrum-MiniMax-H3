# Spectrum MiniMax H3 v0.2.16

v0.2.16 releases the MiniMax H3 Untwisting RoPE compatibility consumer from PR #65 together with the post-run research isolation and process-lifetime hardening already merged in PRs #64 and #66.

The release does not change Spectrum's normal forecasting defaults, sampler cadence, generic-correction defaults, native ER-SDE stochastic ownership, or offline-replay policy.

## Untwisting RoPE visual-reference compatibility

Spectrum now recognizes a separate versioned external-patch contract for deterministic MiniMax H3 visual-reference attention modulation produced by the H3 Untwisting RoPE integration.

The new producer namespace is intentionally separate from the existing Diff-Aid contract:

```text
spectrum_h3_visual_reference_patch_profiles
spectrum_h3_visual_reference_patch_runtime
```

This preserves the strict v0.2.12 Diff-Aid `text_activation_modulation` schema while allowing Untwist to declare `visual_reference_attention_modulation` without being misidentified as a Diff-Aid patch or rejected into the all-actual fail-safe.

Spectrum validates the visual-reference profile's:

- schema version, provider, architecture, patch kind, and unique instance identity;
- H3 block count and zero-based block coverage;
- reference scope;
- active denoising-progress interval and producer-declared hard boundaries;
- high/low frequency scale endpoints;
- interpolation `beta`;
- temporal-axis scaling mode;
- scalar strength summary against the declared endpoint scales.

The full visual strength/configuration metadata participates in the external profile fingerprint, so behaviorally different Untwist profiles cannot alias the same Spectrum model-profile cache entry.

Per-call runtime progress is converted into Spectrum's existing normalized-sigma transaction guard. Only boundaries explicitly declared hard by the producer become compatibility transitions. If such a boundary is crossed on a step that Spectrum had scheduled as a forecast, that current step is promoted to one real H3 transformer evaluation. A transition landing on an already-actual step adds no duplicate NFE.

The default companion Untwist window ending at `end_percent=0.90` is covered by regression tests: the first forecast after crossing the hard end boundary is promoted to an actual anchor.

Diff-Aid and Untwist descriptors can be stacked in the same run. Their kinds remain distinct in runtime/model-aware telemetry, for example:

```text
external_patch_kinds=text_activation_modulation,visual_reference_attention_modulation
```

Malformed or inconsistent declared metadata continues to use Spectrum's existing fail-safe behavior rather than allowing an ambiguous forecast transaction. The compatibility layer adds no hard dependency on the Untwisting RoPE custom node; Spectrum only consumes the declared pure-data contract when it is present.

## Post-run research process isolation

PR #64 moves optional generic-correction post-run evaluation/report work out of the ComfyUI process and into an isolated Python subprocess.

The previous in-process daemon-thread boundary could not satisfy Spectrum's post-run safety invariant: a native SIGSEGV in any Python thread terminates the entire process and cannot be contained by `except Exception`. A completed generation could therefore be lost after sampling had already finished while optional research analysis was running.

The isolated path now:

- releases core Spectrum runtime/history state before optional research dispatch;
- starts the research worker through an isolated stdlib bootstrap with `-I` and `faulthandler`;
- keeps only a lightweight watcher thread in the ComfyUI process for child I/O and lifetime management;
- retains the single-worker bound so diagnostic jobs cannot accumulate;
- caps stalled analysis and reports Python failures, timeouts, and fatal signals without invalidating the completed sampler result;
- preserves the ordering invariant `calibration export -> core runtime/VRAM release -> optional research dispatch`.

Forecasting math, native ER-SDE behavior, offline replay, generic-correction controller math, and downstream VAE/output execution are unchanged by this isolation.

## Fatal-signal and timeout teardown hardening

The first isolation implementation exposed a runner-specific process-lifetime race after PR #64 merged. A child could already have emitted CPython faulthandler's canonical `Fatal Python error: Segmentation fault` diagnostic while still failing to become reapable before the watcher's timeout. Killing it at that point could replace the useful native-signal diagnosis with the cleanup signal. The timeout cleanup also had an unbounded final `communicate()` path if descendants retained the worker's stdout/stderr pipes.

PR #66 hardens that boundary by:

- recognizing only CPython faulthandler's canonical fatal-error markers and normalizing them to signal names such as `SIGSEGV`;
- preserving an already-observed fatal-signal diagnosis even when the child has not become reapable by the deadline;
- retaining direct negative-return-code signal reporting for normally reaped children;
- best-effort disabling POSIX core dumps before the worker executes;
- launching the worker in its own POSIX session/process group;
- terminating the whole research process group on timeout so descendants cannot keep inherited pipes alive;
- bounding the post-kill drain with a separate termination grace period and closing parent pipe handles if cleanup still cannot drain;
- preserving the Windows direct-process termination path;
- preserving the single-worker bound and non-blocking sampler teardown.

This fixes the violated lifetime/diagnostic invariant without broadening the research worker's authority over the generation process.

## Included merged work since v0.2.15

This release contains all runtime work merged after v0.2.15:

- **#64 — Isolate post-run generic-correction research from ComfyUI**
- **#66 — Harden isolated research crash and timeout teardown**
- **#65 — Recognize Untwist H3 visual-reference external patch profiles**

The v0.2.15 H3 Continuum interoperability and native ER-SDE post-prefix solver-space fix remain included unchanged, along with the v0.2.14 replay-preview safety, v0.2.13 Python 3.13 provenance normalization, and v0.2.12 Diff-Aid compatibility layer.

## Validation and release boundary

PR #65 passed the repository `tests` workflow on its implementation head before merge. Its regression coverage checks stacked Diff-Aid/Untwist recognition, distinct runtime patch kinds, strength-profile fingerprinting, hard `end_percent=0.90` boundary promotion, and required runtime/static contract consistency.

PR #64 added isolation coverage for non-blocking dispatch, worker-slot lifetime, child failures, package-entrypoint avoidance, and intentional child SIGSEGV containment. PR #66 added deterministic fatal-marker normalization, POSIX core-dump/bootstrap checks, delayed-reap SIGSEGV handling, process-group timeout cleanup, and descendant-pipe teardown coverage.

The v0.2.16 release commit changes release metadata only. The exact combined runtime tree is the already-merged `main` state containing #64, #66, and #65. The release remains gated by the repository's existing CI workflow: after this version bump is merged to `main`, the Comfy Registry publish workflow is triggered by `pyproject.toml`, and the GitHub release workflow publishes `v0.2.16` only from a successful tested `main` commit.
