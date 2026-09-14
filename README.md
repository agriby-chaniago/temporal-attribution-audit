# Temporal Attribution Audit

Code, analysis notebooks, and result artifacts for a pre-registered, multi-cohort audit of
**temporal attribution faithfulness and stability** in sequence models applied to Parkinsonian
handwriting.

The study asks three questions that are usually collapsed into one:

1. **Output fidelity** — does the attention map match what actually drove the model's prediction,
   measured against Shapley attribution?
2. **Physiological alignment** — do the localized moments correspond to an independent motor
   marker, measured from a signal channel deliberately withheld from the model?
3. **Reproducibility** — how much of the map survives retraining the same architecture with a
   different random seed?

Three interchangeable encoders are compared under an otherwise identical pipeline: **Bidirectional
GRU**, **Bidirectional Mamba-2**, and **Bidirectional Mamba-3**.

## Headline result

Across four independent perturbations — architecture, cohort, random seed alone, and patch
resolution — the attention map shifted **4 to 95 times more**, in matched units, than classification
accuracy did. Ensembling across ten seeds recovered Shapley fidelity (17% → 53% of a measured
ceiling) but did **not** recover physiological alignment under any of four aggregation rules, which
establishes those two properties as separable rather than travelling together.

This synthesis is **exploratory**: it was assembled after the four perturbations had been observed
and was not itself part of the frozen protocol. It is reported for confirmatory replication, not as
a settled finding.

## Repository layout

```
src/          model, data loaders, channel engineering
scripts/      analysis scripts, one per reported result
notebooks/    end-to-end pipeline; 00_gabungan_beku.ipynb is the archived frozen run
results/      104 result artifacts; every number in the manuscript traces to a file here
figures/      generated figures
draftSempro/  protocol document (see "Protocol record" below)
SETUP.md      full environment specification
```

## Data

Neither dataset is redistributed here. Both are public and are downloaded directly from their
original sources by the included download script.

| Dataset | Subjects | Acquisition | License / terms |
|---|---|---|---|
| UCI 395 | 77 (62 PD, 15 HC) | digitizing tablet, ≈127.52 Hz | CC BY 4.0 |
| NewHandPD | 66 (31 PD, 35 HC) | BiSP smart pen, 1000 Hz | citation of Pereira et al. (2016) required |

Full attribution, the exact modifications made to each dataset, and the personal-data handling
policy are documented in [`LISENSI-DATA.md`](LISENSI-DATA.md).

**Personal data.** Raw NewHandPD metadata headers contain identifying fields (names, an
identifier resembling a medical record number). Raw metadata is **not** included in this
repository. Metadata extraction is restricted to a non-identifying allow-list (`META_AMAN` in
`src/newhandpd.py`), and notebook outputs have been redacted where a raw header was printed during
data-structure verification.

## Protocol record

Effect-size thresholds, decision rules, and the confirmatory/exploratory label of each analysis were
frozen in a written protocol before the corresponding analysis was run. That protocol is the
research proposal in [`draftSempro/`](draftSempro/), which contains every prediction and threshold
and **no result values**.

This is a written, internally dated and defended protocol — it is **not** a prospective registration
on a public registry, and it should not be cited as one. Analyses conceived after earlier results
were seen are labelled exploratory in both the protocol and the manuscript.

## Reproducing

See [`SETUP.md`](SETUP.md) for the full environment specification.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# download both datasets from their original sources
python scripts/unduh_data.py
```

Reproducibility is enforced rather than hoped for: fold splits are fixed by `random_state=42` across
all scenarios, so subject partitions are identical across architectures and training seeds, and the
training seed is recorded explicitly in every result file.

## Citation

A manuscript reporting this work is in preparation. Until it appears, please cite this repository
directly.

## License

- **Code** (`src/`, `scripts/`, `notebooks/`): MIT — see [`LICENSE`](LICENSE).
- **Result artifacts, figures, and documentation**: CC BY 4.0.
- Datasets remain under their own terms; see [`LISENSI-DATA.md`](LISENSI-DATA.md).
- The Mamba-3 module used at runtime is redistributed by its own authors under their license and is
  not vendored here.
