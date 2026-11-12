"""Copy and bind stored presentation evidence. No scientific estimator is run."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
MANUSCRIPT = ROOT / "Q1-amin/manuscript_array_latex_v9"
DESIGN = "9f048349d216342f9280825fa3e12dea0fbcb8eeeac3387efccead22f3b540f5"
EXECUTION = "0082ecd5671e689ea2325bb0d45ed2ecfdffd7622ebc487e2edbbc2b0def0726"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    provenance = []
    bindings = []
    evidence = {"publication_kind": "presentation_from_stored_results", "replication": {}}

    def copy_file(source: Path, name: str, kind: str = "csv"):
        target = HERE / ("assets" if kind == "asset" else "sources") / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert digest(source) == digest(target)
        provenance.append({"source_path": str(source.relative_to(ROOT)),
                           "public_copy": str(target.relative_to(HERE)),
                           "sha256": digest(source), "bytes": source.stat().st_size})
        if kind == "csv":
            with target.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            for i, row in enumerate(rows):
                for field, value in row.items():
                    if value and re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value):
                        bindings.append({"source_copy": str(target.relative_to(HERE)),
                                         "row_index": i, "field": field, "value": value})
            return rows
        if kind == "json":
            return json.loads(target.read_text())
        return str(target.relative_to(HERE))

    combined = (ROOT / "results/attention_replication20_resume/combined" / DESIGN /
                EXECUTION / "20261025T073118047187Z")
    evidence["design"] = copy_file(ROOT / "plans/attention_audit_replication20_20261019/design_lock.json", "design_lock.json", "json")
    joint = json.loads((combined / "analysis_manifest.json").read_text())
    assert joint["completion_status"] == "complete"
    evidence["primary"] = copy_file(combined / "primary_eight_strata_holm.csv", "primary_eight_strata.csv")
    assert len(evidence["primary"]) == 8
    copy_file(combined / "analysis_manifest.json", "joint_manifest.json", "json")

    for cohort in ("newhandpd", "uci395"):
        pointer = json.loads((ROOT / f"results/attention_replication20_resume/{cohort}/latest_complete.json").read_text())
        run = ROOT / pointer["path"]
        manifest = json.loads((run / "analysis_manifest.json").read_text())
        assert digest(run / "analysis_manifest.json") == pointer["manifest_sha256"]
        assert manifest["design_id"] == DESIGN and manifest["execution_id"] == EXECUTION
        assert manifest["completion_status"] == "complete"
        assert manifest["public_preregistration"] is False
        output = {"manifest": copy_file(run / "analysis_manifest.json", f"{cohort}_manifest.json", "json")}
        for name in ("secondary_mean_contrasts", "matching_coverage", "jsd_associations",
                     "placebo_contrasts", "target_seed_effects", "leave_one_seed_out",
                     "eligible_excluded_characteristics", "new_minus_discovery_subject_contrast",
                     "replacement_common_support_contrasts", "subject_auroc_per_seed",
                     "record_manifest", "model_inventory", "directed_seed_schedule"):
            source = run / (name + ".csv")
            output[name] = copy_file(source, f"{cohort}_{name}.csv")
        evidence["replication"][cohort] = output

    discovery = ROOT / "results/e2/20261012T050124548170Z"
    evidence["discovery_e1"] = copy_file(discovery / "e1_ci_new.csv", "discovery_e1_ci.csv")
    evidence["discovery_e1_sensitivity"] = copy_file(discovery / "e1_sensitivity_new.csv", "discovery_e1_sensitivity.csv")
    evidence["discovery_auc"] = copy_file(discovery / "new_auroc_per_seed.csv", "discovery_auroc_per_seed.csv")
    evidence["amplitude"] = copy_file(ROOT / "results/e2_amplitude/20261014T024232464643Z/paired_ci.csv", "discovery_amplitude_ci.csv")
    evidence["bridge"] = copy_file(ROOT / "results/e1_e2_rms_bridge/20261014T041914953496Z/association_ci.csv", "discovery_bridge_associations.csv")
    evidence["historical_auc"] = copy_file(ROOT / "results/e0_e1/20261012T035813806939Z/e0_cohort.csv", "historical_auroc.csv")
    evidence["historical_auc_contrast"] = copy_file(ROOT / "results/e0_e1/20261012T035813806939Z/e0_contrast.csv", "historical_auroc_contrast.csv")
    evidence["historical_age"] = copy_file(ROOT / "results/p5_perancu_usia.csv", "historical_age.csv")
    evidence["historical_classic"] = copy_file(ROOT / "results/a_baseline_lintas_kohort.csv", "historical_classic.csv")
    calibration = ROOT / "results/calibration_oof/20261014T135037828430Z"
    evidence["calibration"] = copy_file(calibration / "mean_per_seed_scores_ci.csv", "calibration_scores.csv")
    evidence["calibration_contrasts"] = copy_file(calibration / "paired_architecture_contrasts_ci.csv", "calibration_contrasts.csv")
    evidence["examples"] = copy_file(MANUSCRIPT / "english/figures/source_data/attention_examples_v5.json", "attention_examples.json", "json")
    for example in evidence["examples"]:
        assert example["source"] == "results/e2/20261012T050124548170Z/joint_oof_maps.pkl"
        assert digest(ROOT / example["source"]) == example["source_sha256"]
        assert len(example["alpha_a"]) == len(example["alpha_b"]) == example["n_patches"]
        assert example["fold"] == 1 and example["record_id"] == "newhandpd:H10:task1:seg0"
    for stem in ("jsd_newhandpd_v7", "jsd_uci_v7"):
        copy_file(MANUSCRIPT / f"english/figures/{stem}.png", stem + ".png", "asset")

    # These snapshots identify documents, not new scientific evidence.
    for source in (MANUSCRIPT / "english/build/main/main.pdf",
                   ROOT / "untuk_presentasi_25sep2026/sempro-agriby-presentasi-Q1.pptx"):
        provenance.append({"source_path": str(source.relative_to(ROOT)), "public_copy": "not copied",
                           "sha256": digest(source), "bytes": source.stat().st_size})

    bib = (MANUSCRIPT / "english/references.bib").read_text()
    keys = ["theissler2022", "diaz2021", "toffoli2023", "turbe2023", "braganca2026",
            "simic2025", "navakauskas2026", "nielsen2021", "kapoor2023", "silva2023",
            "dimitriadis2021", "loy2022", "watson2023", "lakens2022", "mambasurvey2026"]
    references = []
    for key in keys:
        match = re.search(r"@article\{" + key + r",.*?(?=\n@|\Z)", bib, re.S)
        assert match, key
        block = match.group(0)
        def value(field):
            m = re.search(r"\b" + field + r"\s*=\s*\{(.*?)\}\s*[,\n]", block, re.S)
            if not m:
                return ""
            text = m.group(1)
            for old, new in ((r"{\v{S}}", "Š"), (r"\v{S}", "Š"),
                             (r"{\'c}", "ć"), (r"\'{c}", "ć"),
                             (r"\c{c}", "ç"), (r"\'{e}", "é"),
                             (r"{\'e}", "é"), (r"\'{i}", "í")):
                text = text.replace(old, new)
            text = text.replace(r"\'c", "ć").replace(r"\'e", "é").replace(r"\'i", "í")
            return text.replace("{", "").replace("}", "")
        year = int(value("year")); assert year >= 2021
        references.append(dict(key=key, author=value("author"), title=value("title"),
                               journal=value("journal"), year=year, doi=value("doi"),
                               source="V9 verified journal bibliography; selected presentation use"))
    evidence["references"] = references
    (HERE / "sources/evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    (HERE / "sources/PROVENANCE.json").write_text(json.dumps({
        "prepared_utc": datetime.now(timezone.utc).isoformat(), "event_date": "2026-11-16",
        "source_cutoff": "completed stored runs through 2026-10-25",
        "training": False, "inference": False, "new_interventions": False,
        "new_statistical_estimates": False, "sources": provenance,
    }, ensure_ascii=False, indent=2) + "\n")
    with (HERE / "sources/NUMERIC_BINDINGS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_copy", "row_index", "field", "value"])
        writer.writeheader(); writer.writerows(bindings)
    (HERE / "sources/references.json").write_text(json.dumps(references, ensure_ascii=False, indent=2) + "\n")
    print(f"Copied {len(provenance)-2} stored sources/assets; {len(bindings)} scalar bindings.")
    print("No statistical estimator, notebook, model, intervention or bootstrap was executed.")


if __name__ == "__main__":
    main()
