# Attention audit extension: development bank, 2026-10-16

Status: additional exploratory analysis, specified before the new swap outcomes
are generated. Existing NewHandPD E1, E2, RMS-control and bridge results are known.
This document is not a public preregistration. It changes neither the historical
preregistration nor any historical estimate. No new seed or UCI result is included.

## Question and bank

On an identical held-out recording, what happens to intervention-ranking relevance
when a ranking from another training seed is applied to a fixed target model?
How does this change relate to map disagreement and to displaced-map references?

Use only the authenticated NewHandPD bank: BiGRU and BiMamba-2, seeds 0, 1, 2,
five identical subject-grouped folds, 30 checkpoints. Historical ten-seed AUROC
and five-seed classical-baseline banks remain separate. No training is allowed.
The user executes all scientific computation, including stored-map diagnostics.

Pinned sources:

- E2: `results/e2/20261012T050124548170Z`.
- RMS controls: `results/e2_amplitude/20261014T024232464643Z`.
- Bridge: `results/e1_e2_rms_bridge/20261014T041914953496Z`.
- The exact manifests, source builders and bridge table checksums are bound in
  `source_pins.json`. Generated notebooks include their own scientific-source hash.

## Notebook 19: stored-artifact diagnostics

No model inference, interventions or training. First repeat the existing bridge's
identity, subject/fold, checkpoint, grid, intervention and matching verification.
Reconstruct its full-curve pair table from E1 plus RMS record contrasts and compare
it with the saved table before calculating new statistics.

1. **Map references.** For each unordered seed pair and recording, circularly shift
   the second map on its original patch grid. Use up to 199 distinct nonzero
   offsets, sampled without replacement with keyed seed 1900; enumerate all offsets
   if fewer exist. Use 199 independent permutations as a secondary reference.
   Preserve the map's weights and length. Keep periodic/unchanged outcomes and
   identify them. References are conditional constructions, not physiological
   ground truth or automatically valid population hypothesis tests. Report actual
   JSD, reference draws and per-pair means/quantiles; subject-balanced differences
   between actual JSD and the per-pair reference mean receive conditional CIs.
2. **Concentration.** Compute entropy in bits, entropy/log2(W), effective support
   2**entropy, effective-support fraction, maximum mass, and boundary gaps at each
   intervention budget. Report W and exact ties; for W=1 normalized entropy is
   undefined. JSD is base-2 divergence, not its square-root distance.
3. **Output-change sensitivity.** Retain the old weighted Spearman as the reference.
   Define the new sensitivity as weighted partial rank correlation: weighted
   midranks of JSD and gain-change are each residualized on an intercept and the
   weighted midrank of |delta logit|; correlate the residuals with the same weights.
   Recompute ranks and weighted least-squares fits in every subject-bootstrap draw.
   Constant/rank-deficient adjustment is reported as undefined, without fallback.
   Residual norm <=1e-12 times max(original weighted rank norm,1) is also treated
   as numerical degeneracy rather than an estimable residual correlation.
   |delta probability| is an alternative adjustment. Outcomes are |delta G| and
   |delta H|, H=(T-R)/(T+R), floor 1e-8; also report raw on normalized-valid support.
   This operational estimator is a descriptive
   sensitivity, not a causal effect or a replacement of the old primary estimator.
4. **Replacement contrast.** Join median and donor on the exact same architecture,
   subject, recording, fold and two checkpoint/seed identities. Use only pairs
   complete in both replacements. Estimate rho_median-rho_donor with paired subject
   draws. Normalized analyses require eligibility in both replacements. Also show
   raw results on that normalized-valid support. Do not compare significance flags
   or subtract the endpoints of separate CIs.

## Notebook 20: fixed-model ranking swap

### Target, ranking and scheduling

Within architecture and fold, evaluate every ordered pair a<-b with a!=b. For
three seeds these are all six directed pairs per recording. Each source seed's
map must come from the same held-out recording and grid. Transfer its **input
patch ranking**; do not replace the target's attention-pooling weights. Use the
target checkpoint, target normalizer and target original logit throughout.
Self-swaps are technical controls, excluded from scientific association analysis.

### Recipe and controls

Retain patch=7, fs=100 Hz, all six channels together, unchanged tail and length,
batch-one forward, budgets 0.05/0.10/0.20/0.30 and actual ceil(fraction*W)/W fractions.
Use the training-median background and the five original same-task training-donor
backgrounds reconstructed from their saved plans. No evaluation-subject donor and
no fitting, tuning or selection on masked outputs or test labels.

For each own/swapped/placebo mask separately, preserve the existing random-control
recipe: same patch count and exact multiset of linear contiguous-run lengths,
RMS caliper 1e-8+0.05*target_RMS, 4096 candidate cap, candidate batches of 128,
20 median controls or four per each of five donor backgrounds. Exclude the
identical target mask; allow and report other overlap. No nearest-match fallback.
Own-mask controls keep their original plans and completeness flags. New masks use
keyed matching seed 2000. Identical masks in an identical target/input/background/
budget context share one control plan and response, regardless of ranking role.
Hence an identical mask does not acquire spurious D from separate random draws.

Reuse the saved BiMamba-2 runtime profile from the RMS manifest; validate its hash,
environment and two original sentinels. Do not search for a new profile or modify
installed packages. Replay original attention points for every target recording;
replay old controls when reusing their outputs. Keep existing anchor tolerances
(atol=rtol=1e-4) and map checks (atol=1e-6, rtol=1e-4, exact operational masks).
Save diagnostics and stop on a mismatch. Never silently substitute a checkpoint.

### Endpoints

For target a, source b and recording r, let T be normalized-area AOPC of absolute
logit change and R the AOPC of its own RMS/geometrically matched controls. Mean
controls within background, then equally weight all backgrounds; integrate the
four actual fractions plus (0,0), divide by the final actual fraction.

- G_own=T_own-R_own; G_swap=T_swap-R_swap.
- **Primary:** subject-balanced mean D=G_own-G_swap per architecture/replacement.
  Positive favors the own ranking under this intervention; negative favors the
  transferred ranking. This is not automatically a change in explanation quality.
- Key secondary: mean G_swap and mean G_own on the same eligible support; mean |D|.
- Planned secondary associations: weighted Spearman(JSD, |D|) and
  weighted Spearman(JSD, D). Neither is a causal dose-response.
- Scale sensitivity: D_H=H_own-H_swap, H=(T-R)/(T+R), floor 1e-8, with floors
  1e-6/1e-4 reported separately. Report raw D on the same normalized-valid support.
- Dose diagnostics: per-budget/background RMS_own, RMS_swap, their difference,
  exact run lengths, mask overlap, and raw T_own-T_swap AOPC.
- Stable-output sensitivity: original target/source predicted classes agree,
  |delta p|<=0.02 and |delta z|<=0.1. The full continuous analysis remains primary.

JSD=0 implies identical maps and, with common tie handling, identical rankings.
JSD approaching zero does not guarantee identical top-k masks near a tie. Report
actual mask overlap and attention boundary gaps rather than assuming continuity.

### Displaced-map references (placebos)

For each real target-source pair, construct candidates by circularly shifting the
**target's own map** (primary reference) or permuting its weights (sensitivity).
Each preserves W, entropy and the multiset of weights. Use up to 199 distinct
nonzero circular offsets and 199 independent permutation candidates, keyed seed
2001. Select one candidate per construction by smallest absolute JSD discrepancy
from the real target-source JSD, ties resolved deterministically by candidate ID.
Require absolute discrepancy <=0.025 on the [0,1] JSD scale. This pragmatic
caliper is fixed for the development run; report its actual error and failures.
Do not use input-intervention responses to select a candidate or widen the caliper.

Each selected map gets its own RMS/geometric controls. Report G_swap-G_placebo,
D_placebo and |D_placebo| on the intersection of own, real-swap and placebo
eligibility. One selected reference per construction is a bounded diagnostic,
not exhaustive integration over all possible shifted/permuted maps. Matching JSD
does not match spatial organization, every budget's overlap, or all input effects.

### Eligibility and inferential units

Every planned directed pair remains in the audit table. Full-curve eligibility
requires own and swap complete in every budget and every background of the given
replacement; it does not require all three seeds to have complete matching.
Record reasons for loss, including own matching, swap matching and placebo-JSD
search. Report pairs, recordings, subjects, class, target seed and source seed.
Per-budget analyses have their own explicitly reported support.

Within eligible support weight subjects equally, recordings within subject equally,
eligible target seeds within recording equally, and source seeds within target
equally. After filtering, renormalize this hierarchy explicitly. For old bridge
unordered pairs retain its original equal-subject/record/pair weights.

Use 5000 class-stratified subject-bootstrap draws, seed 2002 for swap and 1902 for
stored diagnostics, shared across strata and endpoints. Preserve all recordings
and model-pair dependencies in each sampled subject. Recompute weighted ranks per
draw. CI: percentile 95%, conditional on available checkpoints/folds/seeds/plans/
matching; report only with >=10 eligible subjects and >=95% finite draws. This is
a reporting rule, not proof of power. Leave-one-seed-out is an influence diagnostic,
not a seed-population CI. No independence is claimed for the six directed pairs.

The four primary mean-D strata form one exploratory family. Provide nominal CIs
and approximate two-sided centered subject-bootstrap p values, plus Holm-adjusted
p values across all four planned strata (undefined tests retained as p=1 only for
adjustment bookkeeping and flagged). These p values are approximate, not exact
randomization tests. Secondary/reference/sensitivity intervals are nominal and
exploratory; no confirmatory selection by the most favorable result. Holm-adjusted
p values do not turn nominal intervals into simultaneous intervals.

### Control-response variability and computation profile

Save response count, mean, SD and range for each matched-control component.
As a conditional finite-pool sensitivity, generate 500 exponential reweightings
of saved control responses (seed 2003). Reuse weights for identical components,
and share weights for identical accepted candidate masks across donor backgrounds
within a matching search. Recompute R, AOPC and D, keeping subjects, targets and
intervention outputs fixed. Report quantiles/SD as **control-reweighting sensitivity**,
not population CIs or complete uncertainty of a new matching search. Rejected or
unobserved candidates are unavailable, and candidate-selection dependence is not
fully reconstructed. Keep this separate from subject-bootstrap intervals.

Time loading, original/replay forward, candidate search, intervention forward,
cache validation and serialization separately, with CUDA synchronization for GPU
timings. No training benchmark is claimed. Resume only identically hashed code,
protocol, inputs, checkpoint, profile and configuration; verify cache checksums.
Cached scientific outputs are replayed before reuse, without repeating search.
An OS-level advisory lock prevents simultaneous swap loops in this new cache
namespace and is released when a kernel exits. It does not modify old caches.

## Interpretation and subsequent decisions

- G_swap>0 supports above-control intervention relevance on eligible support.
- D summarizes own-versus-swap advantage; D around zero is not equivalence.
- Neither mean D nor an association establishes physiological validity.
- Different mean signs, replacements or architectures are reported separately.
- Replication N (10/20/30 scenarios), total-versus-new seeds, UCI primary task,
  scheduling for the new bank, replication criteria and formal seed-population
  inference are not locked by this development protocol. SST is the procedural
  candidate task; this is not frozen-model transfer.
- Prepare a prospective replication protocol after inspecting development results
  and actual cost; disclose that these development outcomes were already known.
  Any public registration requires the user's separate instruction.
- Manuscript V5 and all historical sources remain intact. A later manuscript
  version incorporates actual results only after completion/alignment review.

## Execution boundary

The agent generates notebooks through builders and performs static/synthetic
software checks only. The agent does not execute notebook cells, real-model
inference, interventions, bootstraps or new scientific analyses. There are no new
results merely because a builder or notebook file exists.
