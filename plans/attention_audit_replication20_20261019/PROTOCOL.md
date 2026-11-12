# Prospective extension of the temporal-attention audit

Design date: 19 October 2026. Prepared after the three-seed NewHandPD results,
including notebooks 19–20, were known. Those results remain exploratory. This is
a local, prospectively specified extension, **not a public preregistration**.
No registration, upload, training, inference or statistical analysis is performed
by preparation of this package. The researcher executes the new notebooks.

## Question, design and independent units

Does the advantage of a target model's own temporal ranking over a transferred
ranking recur across newly trained seeds and a different handwriting acquisition?
Target functions stay fixed during a swap. Both rankings receive their own
RMS/run-geometry controls. Output reproducibility, map reproducibility,
intervention-ranking relevance and physiological validity remain distinct.

Two architectures: BiGRU (`gru`) and BiMamba-2 (`mamba2`). Two separately analysed
datasets: NewHandPD spirals and UCI 395 **SST only** (task code 0). The SST choice
is procedural: tracing a visible spiral template. DST and STCP are not pooled or
silently substituted. Sensor channels, time-grid provenance and participant
populations differ. No frozen-model transfer between datasets is claimed.

Twenty total training seeds per architecture and dataset: integers 0–19. Five
subject-grouped folds, shuffled with split seed 42, remain identical across
training seeds. All recordings/segments of a subject stay in the same fold.
For NewHandPD the stored record/fold identities must agree with the authenticated
baseline. SST folds are constructed from SST subjects only, then recorded.
This design does not estimate sensitivity to new fold partitions.

- **NewHandPD:** reuse the 30 authenticated seed-0/1/2 checkpoints read-only;
  train 170 new checkpoints for seeds 3–19. Primary replication restricts both
  target and source to these 17 new seeds. The complete 20-seed bank is a clearly
  labelled pooled sensitivity, not an independent replication of the discovery bank.
- **UCI-SST:** train all 200 checkpoints, seeds 0–19. All are new to this audit;
  all 20 seeds enter its primary analysis. Historical UCI models are not reused.
- Total planned: 400 checkpoints in the extended banks; **370 newly trained**.
  Additional seeds add training repetitions, not participants.

## Recipe fixed before new outcomes

Use the existing preprocessed caches without rebuilding or modifying them:
`data/cache/newhandpd_spiral_fs100.npz` and `data/cache/uci395_fs100.npz`.
Filter UCI task 0 before splitting. Pin cache and implementation hashes.
NewHandPD uses six first/second sensor-difference channels at 100 Hz. UCI uses
six coordinate-dynamics/pressure channels on its existing nominal time grid.
No new inference of physical units or physiological truth is made.

Training: attention pooling, patch 7, width 128, three bidirectional layers,
dropout 0.30, BiGRU directional width 64; AdamW through the existing `latih`
implementation, learning rate 0.0003, batch size 8, 35 epochs and the existing
stepwise cosine scheduler. Fit median/IQR/clipping only on training recordings.
Retain the final epoch. No tuning, validation-based selection or early stopping
is introduced into this replication recipe. An improved recipe would be another
bank. Abort if pinned training implementations changed.

## Scheduled transfer pairs and compute budget

Build an independently shuffled ring of the 20 seeds for each dataset and
architecture using scheduling seed 2101 and keyed randomness. Each target uses
sources at offsets **−2, −1, +1, +2, +10**. The schedule has no self-pairs, five
sources and five outgoing appearances per seed, and reciprocal directions.
Every seed is evaluated on every held-out recording in its fold. Generate and
save the exact schedule in the design preflight, before any new model response.

This is a scheduled-source estimand, not an exhaustive mean over all 19 possible
sources. In the NewHandPD primary new/new subgraph, each new target has 2–5
scheduled new sources. Average available sources within target before averaging
targets. Report planned as well as eligible target/source counts. The complete
20-seed scheduled graph is the pooled sensitivity.

For 264 NewHandPD recordings the schedule plans
`264 × 2 architectures × 100 directed pairs × 2 replacements = 105,600`
pair/replacement rows, versus 401,280 for all directed pairs. The exact primary
new/new count depends on the saved ring. UCI counts use the observed SST record
manifest, never the full dataset's 77-subject headline as a task denominator.
These are row counts, not wall-clock estimates. Budget, background, control and
placebo work is additional. Record training, prediction, candidate-search and
intervention-forward times separately; report cached work separately.

The target remains 20 regardless of effect signs or p-values. Do not extend to
30 because a result is disappointing. A resource interruption yields an incomplete
20-seed run; resume it or report the shortfall without relabelling it complete.

## Interventions and endpoints

Keep patch fractions 0.05/0.10/0.20/0.30, all six channels together, unchanged tail,
forward batch one, donor block length 32 patches and five same-task training-only
donor backgrounds. A zero normalized background is training-median replacement.
Matching uses exact masked-patch count and the sorted multiset of linear run
lengths, RMS relative caliper 0.05 plus absolute tolerance 1e-8, at most 4096
candidates in batches of 128. Require 20 median controls, or four controls for
each of five donor backgrounds. No quota reduction, nearest-match fallback,
outcome-based candidate selection or caliper widening is allowed.

The same input/normalizer/background/mask context shares its control component,
including self-swaps. AOPC integrates absolute logit changes at actual fractions,
anchored at zero and divided by the last actual fraction. Let T be ranking AOPC,
R its matched-control AOPC, G=T−R, and **D=G_own−G_transferred**.

Primary estimand: subject-balanced mean D separately in the eight
dataset × architecture × replacement strata. Average sources within target,
targets within record, records within subject, then subjects equally. Positive D
favors the own ranking under this intervention. It is not a clinical quality scale.
Nominal 95% class-stratified subject-bootstrap intervals use 5000 draws, seed 2102.
Bootstrap retains all model/record pairs for each sampled subject. Approximate
two-sided centered-bootstrap p-values receive Holm adjustment across the **eight
primary mean-D tests**. These are not exact randomization p-values.

Key secondary: G_transferred and G_own on exactly the same eligible support;
dimensionless H=(T−R)/(T+R), floor 1e-8, D_H=H_own−H_transferred; raw T contrast;
weighted Spearman JSD versus signed D and |D|; strict stable-output sensitivity
(same predicted class, |Δp|≤0.02 and |Δz|≤0.1). Continuous all-eligible analyses
remain primary. Secondary intervals are nominal without a claim of familywise
control. Base-2 JSD is divergence in [0,1], not its square-root distance.

## Placebos, dose and eligibility diagnostics

For each scheduled real source use circular-shift and permutation references
of the target's own attention: 199 candidates, keyed seed 2103, closest JSD with
absolute discrepancy ≤0.025. Choose one candidate per construction without model
responses; do not widen the caliper. Give it its own RMS/geometric controls.
Analyse real-minus-placebo G only on joint complete support. Multiple evaluated
placebo candidates and an extra attribution method remain later extensions.

At every budget/background export own and transferred RMS, run geometry, mask
overlap/Jaccard and symmetric relative dose gap
`abs(r_own−r_swap)/max(r_own,r_swap,1e-8)`. A pre-specified dose sensitivity keeps
only complete pairs with gap ≤0.05 at **all** budgets/backgrounds. The zero/zero
case has gap zero. Report its lost pairs, records, subjects and targets. Dose
restriction is diagnostic; it does not identify a causal effect.

Distinguish original own-control failure, transferred-control failure,
JSD-reference failure and placebo-control failure. Compare eligible/excluded
planned pairs using pre-intervention class, duration, patch count, initial
probability/confidence, map entropy/max mass, requested mask dose, and overlap.
Use hierarchical subject weights; report standardized differences and distributions
without selecting covariates by significance. Do not inverse-probability reweight
to claim missing intervention effects have been recovered. Age is not invented
or joined by row order; it is deferred unless an authenticated subject join is supplied.

## Uncertainty, comparison and interpretation rules

The subject interval conditions on the trained bank, split, source schedule and
matching. Separately report target-seed means, distributions and leave-one-seed-out
influence, removing a dropped seed in **both** target and source roles. These are
not seed-population confidence intervals. No unvalidated crossed seed bootstrap
or self-pairs from resampling are introduced.

Keep discovery, new-seed replication and pooled extension labels in every table.
Compare NewHandPD discovery and new-seed subject effects on common subjects with
a paired subject-bootstrap difference, reporting lost support. No raw-AOPC ranking
of architectures or datasets is made. Normalized contrasts are sensitivity, not
proof of calibration or removal of all confounding.

Replication is assessed by effect direction and size, intervals/precision, coverage,
and new-minus-discovery contrasts. Mixed strata are retained. No requirement that
every condition wins, no CI-overlap test of replication, no equivalence or zero-effect
claim without a separately justified margin, and no post-hoc power argument.
All public preregistration fields remain false unless an actual registration is
later provided. Record the known discovery outcomes when registering prospectively.

## Execution and preservation

Notebook 21 records design/preflight and exact schedules; it performs no training
or model inference. Notebooks 22 and 23 execute the same pipeline for NewHandPD
and UCI-SST. Notebook 24 verifies both complete datasets and applies the fixed
eight-test Holm family; a single completed dataset is not that combined result.
New checkpoints, predictions, rank components and masks are saved
atomically outside all old artifact directories. Completed folds/records/components
resume using identities and hashes. Mid-epoch training is not resumable; an
interrupted unfinished fold restarts its fixed 35-epoch training.

No manuscript, old notebook, builder 01–20, checkpoint or result is overwritten.
The notebook scientific code is generated only through the new builder. A
complete run requires all 200 checkpoint identities per dataset and all planned
record/pair/replacement rows, including explicit matching failures. Statistical
outputs are absent until the researcher runs the corresponding cells.

## Source context checked for this design

- UCI's official documentation distinguishes SST template tracing, blinking-template
  DST and point-holding STCP: https://doi.org/10.24432/C5Q01S.
- Šimić et al. (2025), Scientific Reports, DOI 10.1038/s41598-025-09538-2:
  operator and temporal region choices affect attribution evaluations.
- Existing implementations and complete manifests from notebooks 16–20 are pinned
  as local provenance; no new empirical outcome is asserted here.
