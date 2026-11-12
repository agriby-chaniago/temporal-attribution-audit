"""Verify stored slide values, CI ranges, package structure and preservation.

This is document verification. It never imports models, estimates associations,
resamples study subjects, executes notebooks or changes scientific sources.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import pymupdf

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    checks = []
    def check(passed, name, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})

    cache = {}
    def read_csv(relative):
        if relative not in cache:
            with (HERE / relative).open(newline="", encoding="utf-8") as handle:
                cache[relative] = list(csv.DictReader(handle))
        return cache[relative]

    provenance = json.loads((HERE / "sources/PROVENANCE.json").read_text())
    for item in provenance["sources"]:
        if item["public_copy"] != "not copied":
            check(digest(HERE / item["public_copy"]) == item["sha256"], "source copy hash", item["public_copy"])
        original = ROOT / item["source_path"]
        if original.is_file():
            check(digest(original) == item["sha256"], "original source unchanged", item["source_path"])

    bindings = read_csv("sources/NUMERIC_BINDINGS.csv")
    for binding in bindings:
        rows = read_csv(binding["source_copy"])
        check(rows[int(binding["row_index"])][binding["field"]] == binding["value"], "copied scalar binding", f"{binding['source_copy']}:{binding['row_index']}:{binding['field']}")

    displayed = json.loads((HERE / "review/DISPLAY_CI_BINDINGS.json").read_text())
    for binding in displayed:
        row = read_csv(binding["source_copy"])[binding["row_index"]]
        for dest, source in (("estimate", binding["estimate_field"]), ("ci_low", "ci_low"), ("ci_high", "ci_high")):
            check(str(binding[dest]) == row[source], "displayed CI source field", f"slide {binding['slide']}:{source}")

    specs = json.loads((HERE / "review/CHART_BINDINGS.json").read_text())
    pptx_path = HERE / "presentasi_attention_16nov2026.pptx"
    with ZipFile(pptx_path) as archive:
        check(archive.testzip() is None, "PPTX ZIP integrity")
        slide_paths = sorted([p for p in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", p)], key=lambda p:int(re.search(r"(\d+)\.xml", p)[1]))
        check(len(slide_paths) == 35, "slide count", len(slide_paths))
        notes = [p for p in archive.namelist() if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", p)]
        check(len(notes) == 35, "speaker notes on every slide", len(notes))
        slide_text = []
        for item in slide_paths:
            doc = ET.fromstring(archive.read(item))
            slide_text.append(" ".join(n.text or "" for n in doc.findall(".//a:t", NS)))
        check(not any(re.search(r"\bTODO\b|lorem ipsum|\[insert|Sisipkan Fig", text, re.I) for text in slide_text), "no unfinished slide placeholders")
        for spec in specs:
            doc = ET.fromstring(archive.read(spec["chart"]))
            series = doc.findall(".//c:ser", NS)
            check(len(series) == len(spec["series"]), "native chart series", spec["chart"])
            for index, (ser, source) in enumerate(zip(series, spec["series"])):
                values = [float(v.text) for v in ser.findall("./c:val/c:numRef/c:numCache/c:pt/c:v", NS)]
                check(values == source["values"], "native chart values", f"{spec['chart']} series {index}")
                if spec["errors"]:
                    expected = spec["errors"][index]
                    for i, point in enumerate(expected):
                        row = read_csv(point["source_copy"])[point["row_index"]]
                        check(all(float(row[field]) == point[out] for field, out in (("estimate", "estimate"), ("ci_low", "low"), ("ci_high", "high"))), "chart interval source endpoints", f"{spec['chart']}:{index}:{i}")
                    for side in ("plus", "minus"):
                        parent = ser.find(f"./c:errBars/c:{side}/c:numRef", NS)
                        check(parent is not None and parent.find("c:f", NS) is not None, "native CI workbook reference", f"{spec['chart']}:{index}:{side}")
                        actual = [float(v.text) for v in parent.findall("./c:numCache/c:pt/c:v", NS)]
                        lengths = [(p["high"]-p["estimate"] if side=="plus" else p["estimate"]-p["low"]) for p in expected]
                        check(actual == lengths, "native CI lengths", f"{spec['chart']}:{index}:{side}")
        check("Agriby Diandra Chaniago" in slide_text[0] and "Annastasya Nabila Elsa Wulandari" in slide_text[0], "current author names")
        check("Šimić" in " ".join(slide_text), "Unicode reference names")

    evidence = json.loads((HERE / "sources/evidence.json").read_text())
    for cohort, group in evidence["replication"].items():
        manifest = group["manifest"]
        check(manifest["completion_status"] == "complete", "cohort complete", cohort)
        check(manifest["public_preregistration"] is False, "public registration not retroactively changed", cohort)
        models = group["model_inventory"]
        identities = {(r["architecture"], int(r["seed"]), int(r["fold"])) for r in models}
        check(len(models)==200 and len(identities)==200, "checkpoint identities", cohort)
        records = group["record_manifest"]
        subjects = {r["subject"] for r in records}
        check(len(records)==manifest["n_records"] and len(subjects)==manifest["n_subjects"], "record/subject denominators", cohort)
        loso = group["leave_one_seed_out"]
        estimate_field = "estimate"
        selected = [r for r in loso if r.get("endpoint")=="D"]
        check(bool(selected), "LOSO endpoint exists", cohort)
        check(bool(selected) and all(float(r[estimate_field])>0 for r in selected), "stored LOSO mean D positive", cohort)
        for row in group["placebo_contrasts"]:
            check(float(row["ci_low"]) <= float(row["ci_high"]), "placebo CI order", condition_label(row))
    check(len(evidence["primary"])==8 and all(float(r["ci_low"])>0 for r in evidence["primary"]), "all eight stored primary CI lower bounds positive")
    for row in evidence["primary"]:
        check(round(float(row["p_holm_eight_primary_tests"]), 6)==0.001600, "printed Holm value", condition_label(row))
    check(len(evidence["references"])==15 and all(r["year"]>=2021 and r["doi"] for r in evidence["references"]), "15 recent journal records with DOI")
    check(not any("\\" in r["author"] for r in evidence["references"]), "no unresolved LaTeX in displayed author names")

    shapes = json.loads((HERE / "review/SHAPE_LAYOUT.json").read_text())
    outside = [r for r in shapes if min(r["x"],r["y"],r["w"],r["h"]) < -0.01 or r["x"]+r["w"]>13.33334 or r["y"]+r["h"]>7.50001]
    check(not outside, "slide boxes within canvas", f"{len(shapes)} recorded boxes")
    pdf = pymupdf.open(HERE / "presentasi_attention_16nov2026.pdf")
    check(len(pdf)==35, "PDF page count", len(pdf))
    pdf_text = "\n".join(p.get_text() for p in pdf)
    check("Šimić" in pdf_text, "PDF Unicode reference names")
    for binding in displayed:
        # The full editable slide package is the exact formatted-value source;
        # PDF whitespace can differ after line wrapping.
        text = slide_text[binding["slide"]-1].replace(" ", "")
        check(binding["rendered"].replace(" ", "") in text, "formatted CI appears on its slide", binding["slide"])
    preserved_snapshot = ROOT / ".git/presentation_16nov26_protected_before.json"
    preserved_count = 0
    if preserved_snapshot.is_file():
        preserved = json.loads(preserved_snapshot.read_text())
        preserved_count = len(preserved)
        check(all((ROOT/rel).is_file() and digest(ROOT/rel)==expected for rel,expected in preserved.items()), "old presentation and V9 preserved", preserved_count)
    failed = [row for row in checks if not row["passed"]]
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(), "n_checks": len(checks),
              "n_failed": len(failed), "source_scalar_bindings": len(bindings),
              "displayed_ci_bindings": len(displayed), "native_charts": len(specs),
              "preserved_files": preserved_count, "research_computation_performed": False,
              "checks": checks}
    (HERE / "review/DOCUMENT_CHECKS.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(f"Document checks: {len(checks)}; failed: {len(failed)}")
    print(f"{len(bindings)} copied scalar bindings; {len(displayed)} displayed CI bindings; {len(specs)} native charts.")
    for row in failed:
        print(row)
    raise SystemExit(bool(failed))


def condition_label(row):
    return f"{row.get('cohort','')}/{row.get('architecture','')}/{row.get('masker','')}"


if __name__ == "__main__":
    main()
