# Current research status

Publication update: 12 November 2026. Experiment dates below come from the stored
run manifests. This Git publication occurs after those experiments; it does not
provide a contemporaneous public registration.

## Question and design

The audit asks whether a temporal attention ranking retains its intervention
advantage when it is transferred from another training seed to a fixed target
model on the same held-out recording. Output reproducibility, map
reproducibility, intervention-ranking relevance and physiological validity are
separate properties.

The extended banks contain BiGRU and BiMamba-2, twenty seeds per architecture and
dataset, and five subject-grouped folds: 400 checkpoint identities in total.
NewHandPD has 264 spiral recordings from 66 subjects. The UCI-SST audit has 77
recordings from 76 subjects; this task denominator differs from the full UCI 395
resource's 77 subjects. Neither HandPD nor PaHaW is substituted for NewHandPD.

NewHandPD seeds 0–2 form the known discovery bank. Its primary extension uses
both target and source from seeds 3–19; the full twenty-seed bank is a pooled
sensitivity. All twenty UCI-SST seeds are new to this audit. The banks contain
370 newly trained checkpoints and thirty reused discovery checkpoints. Training
was completed by the researcher, not during this repository publication.

The subject split remains fixed. Each target uses five scheduled source seeds
within its architecture and dataset. This is not an exhaustive evaluation of all
source–target pairs. Own and transferred masks each receive controls matched for
RMS input change, patch budget and run geometry. The two masks need not have the
same dose as each other.

## Reading the results

Let T denote area over the perturbation curve for absolute logit change, and R
the corresponding matched-control response. Ranking advantage is G = T − R.
The primary contrast is D = G_own − G_transferred. Positive D favors the target's
own ranking under the defined interventions. It is not a clinical quality score.

The stored joint report has positive mean D in all eight dataset × architecture ×
replacement strata. Holm adjustment covers those eight primary tests. Nominal
subject-bootstrap intervals condition on the trained bank, split, schedule and
matching; they are not intervals over a population of training seeds. Coverage,
placebo results, dose sensitivities and seed influence must be read alongside D.

The earlier changing-model bridge and the fixed-target swap estimate different
relationships. Map disagreement alone does not establish deterioration of an
explanation. A high AUROC does not validate a temporal map. Absolute logit
responses measure influence under the stated replacement rules, rather than
independent physiological validity.

Historical cross-dataset AUROC results describe separately retrained
architectures. The contrast interval includes zero. They do not establish
frozen-model transportability or superior retention. ECE/Brier results belong
to the historical ten-seed bank and the three-seed discovery bank; calibration
of the twenty-seed replication bank has not been computed. Brier measures
overall probability quality, not calibration alone.

## Analysis history

The three-seed additions are exploratory. The extension plan was locked locally
on 19 October 2026 after discovery outcomes were known. Its manifest explicitly
records `public_preregistration: false`. Publishing it now does not change that
status. Subject uncertainty, seed influence and multiple secondary analyses
remain distinct.

The original September README is preserved in
[the historical snapshot](history/README_20260929.md). Its `4–95×`, fidelity
ceiling and ensemble-recovery narrative belongs to that historical version and
is not the current audit's interpretation. Original results and Git history
remain intact. BiMamba-3 is a historical arm, outside the main extended audit.

## Navigation

- [Experiment timeline](EXPERIMENT_TIMELINE.md).
- [Notebook and builder map](WORKFLOW.md).
- [Published artifact scope](PUBLIC_ARTIFACTS.md).
- [Dataset attribution and handling](../LISENSI-DATA.md).
