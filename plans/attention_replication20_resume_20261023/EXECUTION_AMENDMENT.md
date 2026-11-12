# Strict numerical resume, 23 October 2026

This is an execution repair of the sealed 20-seed design, not a change to its
scientific recipe or retrospective registration. The reference and FAST builders,
executed notebooks, checkpoints and result namespaces remain read-only parents.

## Observed failure and evidence

The saved FAST notebook stopped at BiMamba-2 seed 7, fold 4, recording index 191
(`newhandpd:P28:task1:seg0`). Read-only inspection of the already computed error
frame found 388 patches. At the 30% budget, the 117-patch mask exchanged patch 178
for patch 126. Saved/current logits were 9.937044143676758 and 9.93714427947998;
the original logit tolerance passed. Maximum attention-value difference was
0.0001417398452758789. The checkpoint hash matched its inventory and metadata;
three recent cache envelopes passed checksums. No new model forward was used to
obtain this diagnosis. The low-level source of numerical drift was not proven.

The old diagnostic gate correctly stopped. The repair retains exact operational
masks rather than accepting changed indices, rewriting maps, rounding scores,
widening tolerances or silently dropping the recording.

## Recovery contract

* Models, normalizers, input hashes, grid, batch one and original precision remain
  unchanged. LR, epochs, seed/fold schedule, all budgets/backgrounds, matching RNG,
  calipers, quotas, placebos, estimators and multiplicity family remain unchanged.
* Native execution is checked first. If original alignment or cached partial-point
  replay fails for Mamba-2, at most 128 predetermined profiles are considered.
  Configurations must already be registered in the five installed Mamba forward
  Triton autotuners. No package, CUDA setting or encoder source is changed.
* Candidate order is fixed: native profile; previously validated historical
  profiles, read-only and hash-bound; one-kernel alternatives in registered order;
  bounded combinations of small registered matrix tiles and cumsum/state-passing
  configurations. Selection uses numerical compatibility, never intervention
  effects, labels, AUROC or significance. A historical profile is a candidate,
  not evidence that it works for the new checkpoint.
* Acceptance requires two consecutive original forwards passing the original
  atol=rtol=1e-4 logit anchor and exactly the same stored top-patch sets at every
  budget. Every available partial cached target/control logit must replay within
  the same tolerance. A final original forward is checked again after replay.
* The first passing profile is pinned for the entire unfinished recording. It is
  saved with checkpoint/input/map/background/kernel-source bindings. On reuse it
  must pass again; failure stops without replacing that accepted profile.
* Previous autotuner configurations and caches are restored after the recording,
  including on an exception. GPU resource/compiler rejections are logged and
  skipped as in the original autotuner; unrelated CUDA failures propagate.
* Exhausting the bounded search stops with diagnostics. It does not prove that no
  compatible runtime exists. No record is counted as completed after that failure.

## Caches and provenance

New notebook/output names contain `resume`; the original design/runtime IDs are
retained and a new execution ID records this repair. Read-through precedence is
new resume cache, parent FAST cache, original reference cache. All cache contexts,
checksums and identities remain checked. Complete recording caches are reused
without new forwards. Every partial cached target/control point is replayed before
it is mixed with newly computed components. Already completed values are unchanged.

For NewHandPD an early probe checks the known failed recording before the main
loop. Its choice is operational, based on the recorded failure, and has no role in
selecting scientific effects. The probe writes alignment diagnostics/profile only;
it does not fabricate a completed recording evaluation. Once a complete cache for
that recording exists, the probe is skipped and the main loop authenticates it.

Reference batch-one is used throughout this version. Both architectures fell back
to that backend in the parent NewHandPD FAST benchmark. No resident acceleration,
CUDA Graph, new benchmark or speedup claim is introduced. UCI-SST likewise uses the
canonical reference path. Original CPU energy memoization is retained.

## Limits and execution ownership

Kernel pinning covers the five inspected forward autotuners, not every CUDA,
cuDNN/cuBLAS or host operation. Exact-mask recovery can fail. Matching original
anchors and masks cannot establish equivalence of every unseen perturbed output
when no historical output exists. New outputs are tagged by the separate execution
and recovered recording profile; historical outputs remain historical values.

The builder only generates notebook sources and performs static/software checks.
The user runs all training, forwards, compatibility search, interventions and
statistics. Until those cells are run, recovery on actual CUDA data is unverified.

