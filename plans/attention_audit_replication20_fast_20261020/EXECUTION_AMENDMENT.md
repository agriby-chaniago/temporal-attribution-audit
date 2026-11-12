# Technical execution amendment: resident GPU, batch one

Date: 20 October 2026. The author authorized option 1, option 2 only if compatible,
and a small implementation of option 4. Batched model inference (option 3) was
declined. This is an execution amendment to the existing local design, not a new
scientific protocol, public registration, or a claim that earlier results were
unknown. The original design, builders, notebooks, checkpoint payloads and
research results remain unchanged.

## Scientific contract retained

- The sealed design ID, two architectures, two datasets, 20 seeds, five folds,
  training recipe, final-epoch checkpoint selection and normalizers are identical.
- The same five-source schedule, original attention, four intervention budgets,
  two replacements, donor plans, RNG keys and complete-control quotas apply.
- Matching still uses the original geometry sampler, exact run-length multiset,
  RMS caliper, direct-RMS check, search cap, and incomplete-support rule.
- The same response metrics, AOPC, G/D/H definitions, weighting, bootstrap,
  analysis sets, sensitivities and fixed Holm family are emitted unchanged.
- Each model call retains shape (1, T, 6), the original dtype, the original encoder
  forward and current numerical tolerances. No AMP, TF32 change, new padding,
  early stopping, source reduction or placebo reduction is introduced.

## Execution changes

The normalized input and six backgrounds remain on the GPU for a record. Each
mask copies replacements from the original background into a reusable input
buffer; the incomplete tail remains untouched. Sequential batch-one logits are
collected on the GPU and transferred together after each background/control group.
Explicit per-forward synchronizations are replaced with group boundaries. Host
synchronizations inside the original encoders are retained.

Patch energies use precisely the original CPU float64 expression, cached once per
record/background. Both matching and component validation use that same array.
The RNG and candidate search implementation are unchanged.

CUDA Graphs are **not enabled**. Static review of `src/model.py` found
`lengths.detach().cpu()` in `BiGRUEncoder.forward` and `lengths.tolist()` in
`BiMamba2Encoder._flip_valid`. Capturing the original full forward would encounter
host operations. Removing these would require a larger encoder execution change;
that is outside this deliberately narrow optimization. No CUDA capture was run.

## User-run numerical and performance gate

Before study interventions, deterministically select the shortest and longest
held-out records of the first seed/fold for each architecture. Check all original
own-mask budgets/backgrounds and controls available under the sealed matching
recipe. The replaced float32 input must be bit-identical to the NumPy reference.
Logits must satisfy the existing atol=rtol=1e-4, without widening tolerances.

Benchmark three repetitions in alternating reference/resident order. Both paths
include masking, input handling, model execution and output collection. Warm-up
is symmetric. Select resident mode per architecture only if all numerical checks
pass and the aggregate measured speedup is at least 1.05. Otherwise retain the
reference batch-one path. Input mismatch stops execution. Numerical output
mismatch selects the reference path and is reported. On each new record, compare
unmasked and masked logits under all backgrounds again; any mismatch uses the
reference for that record. These decisions use numerical agreement and runtime,
not the direction, significance, magnitude or coverage of scientific results.

The gate is part of user execution. Preparing these notebooks does not establish
a speedup, numerical equivalence on CUDA, or a completed research run.

## Cache and provenance

- Model weights remain in their existing store. No retraining is needed for a
  completed authenticated model. UCI still needs its planned bank if not trained.
- Prediction and complete/partial record caches are read from the old namespace
  only when the design, runtime, model, record, fold, input, background, mask and
  quota identities match. Their envelope checksums are verified.
- Complete records are returned without further intervention inference. Completed
  components in a partial record are reused; only missing components are run.
- New records/progress, CSVs, figures, manifests and validation logs are written
  in `results/attention_replication20_fast` or `figures/attention_replication20_fast`,
  under an execution ID separate from the original namespace.
- `cache_reuse.json` records the paths, roles and checksums of reused artefacts.
  Each new component reports its execution ID and backend. Conditional estimates
  can contain original results and numerically validated new results; they are not
  claimed to be bitwise identical across execution paths.
- The old active locks are inspected read-only. A live old run blocks the fast
  pipeline. Fast notebooks also share a single-GPU lock. Do not start the old
  notebook while a fast notebook is running; the old code cannot know about the
  new lock.

## Incidental completion-check correction

The parent's final plotting cell reused `expected` for a source-file checksum
after it had represented the expected pair-row count. This would invalidate the
final row-count check. The new copy renames the checksum loop variable to
`pinned_sha256`; the planned row count and all scientific estimates are unchanged.
The original builder/notebooks remain intact.

## Validation scope

The builder performs schema and Python-source checks only. Synthetic software
tests exercise tensor replacement on CPU, exact candidate/energy/component
preservation, cache identity/checksum handling and dependency/static contracts.
They never read study signals, call a trained network, use CUDA, bootstrap study
data or calculate new research findings. GPU validation and timing remain pending
until the author runs the generated notebooks.
