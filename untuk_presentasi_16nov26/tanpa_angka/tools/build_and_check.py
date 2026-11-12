"""Build/check a qualitative document, never research models or estimators."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import pymupdf
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parents[1]
PARENT = HERE.parent
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "c": "http://schemas.openxmlformats.org/drawingml/2006/chart"}
NAME = "presentasi_attention_16nov2026_tanpa_angka"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build:
        subprocess.run(["node", "build_presentation.cjs"], cwd=HERE, check=True)
        with tempfile.TemporaryDirectory(prefix="attention-qualitative-lo-") as profile:
            subprocess.run(["soffice", "-env:UserInstallation="+Path(profile).as_uri(),
                            "--headless", "--convert-to", "pdf", "--outdir", str(HERE),
                            str(HERE / (NAME+".pptx"))], check=True)
    checks=[]
    def check(value, description, detail=""):
        checks.append({"passed": bool(value), "check": description, "detail": detail})
    plan=json.loads((HERE/"SLIDE_PLAN.json").read_text())
    evidence=json.loads((PARENT/"sources/evidence.json").read_text())
    # Logical checks on SAVED estimates and intervals, not a statistical analysis.
    primary=evidence["primary"]
    check(len(primary)==8 and all(float(r["ci_low"])>0 for r in primary),
          "Own-minus-transferred primary contrasts support the stated direction")
    for cohort, group in evidence["replication"].items():
        transferred=[r for r in group["secondary_mean_contrasts"]
                     if r["endpoint"]=="G_swap" and r["scenario"]=="all_primary"]
        check(len(transferred)==4 and all(float(r["ci_low"])>0 for r in transferred),
              "Transferred ranking exceeds matched controls on saved support",cohort)
        associations=[r for r in group["jsd_associations"]
                      if r.get("endpoint")=="abs_D" and r.get("estimator")=="weighted_spearman"
                      and r.get("analysis_set")=="new_seed_replication"]
        check(len(associations)==4 and all(float(r["ci_low"])>0 for r in associations),
              "Saved JSD-absolute-transfer association supports its qualitative statement",cohort)
        placebo=group["placebo_contrasts"]
        check(any(float(r["ci_low"])>0 for r in placebo)
              and any(float(r["ci_low"])<=0<=float(r["ci_high"]) for r in placebo),
              "Placebo comparisons genuinely vary by condition",cohort)
        check(group["manifest"]["completion_status"]=="complete"
              and group["manifest"]["public_preregistration"] is False,
              "Completion and exploratory history retained",cohort)
    dose=[r for r in evidence["replication"]["uci395"]["secondary_mean_contrasts"]
          if r["scenario"]=="dose_gap_at_most_5percent" and r["endpoint"]=="D"
          and r["masker"]=="train_donor"]
    check(len(dose)==2 and all(float(r["ci_low"])<=0<=float(r["ci_high"]) for r in dose),
          "UCI donor dose-sensitivity intervals span zero on both architectures")
    magnitude=[r for r in evidence["amplitude"] if r["metric"]=="paired_magnitude_abs_delta_logit"]
    gru=[r for r in magnitude if r["architecture"]=="gru"]
    mamba_donor=[r for r in magnitude if r["architecture"]=="mamba2" and r["masker"]=="train_donor"]
    mamba_median={r["scope"]:r for r in magnitude
                  if r["architecture"]=="mamba2" and r["masker"]=="train_median"}
    check(len(gru)==4 and all(float(r["ci_low"])>0 for r in gru),
          "BiGRU magnitude comparison retains positive attention contrast on both supports")
    check(len(mamba_donor)==2 and all(float(r["ci_high"])<0 for r in mamba_donor),
          "Unfavourable BiMamba donor magnitude comparison retained on both supports")
    check(float(mamba_median["full_baseline"]["ci_high"])<0 and
          float(mamba_median["matched_common_support"]["ci_low"])<=0<=
          float(mamba_median["matched_common_support"]["ci_high"]),
          "BiMamba median magnitude interval interpretation depends on support")
    calibration={r["metric"]:r for r in evidence["calibration_contrasts"]
                 if r["bank"]=="current_E1_E2" and r["cohort"]=="newhandpd"}
    check(float(calibration["brier"]["ci_high"])<0,
          "Discovery Brier evidence is retained, not described as wholly inconclusive")
    check(float(calibration["ece_equal_width_10"]["ci_low"])<=0<=float(calibration["ece_equal_width_10"]["ci_high"]),
          "Primary discovery ECE uncertainty retained")
    retention=evidence["historical_auc_contrast"]
    check(len(retention)==1 and float(retention[0]["contrast"])<0
          and float(retention[0]["ci_low"])<=0<=float(retention[0]["ci_high"]),
          "Historical retention is descriptive with zero-spanning interval")
    paths=[]
    with ZipFile(HERE/(NAME+".pptx")) as archive:
        check(archive.testzip() is None,"PPTX ZIP integrity")
        paths=sorted((p for p in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml",p)),
                     key=lambda p:int(re.search(r"(\d+)\.xml",p)[1]))
        check(len(paths)==plan["total_slides"]==26,"Slide count")
        notes=[p for p in archive.namelist() if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml",p)]
        check(len(notes)==len(paths),"Every slide has speaker notes")
        numerals=[];texts=[]
        for index,p in enumerate(paths):
            value=" ".join(x.text or "" for x in ET.fromstring(archive.read(p)).findall(".//a:t",NS))
            texts.append(value)
            residual=value.replace("BiMamba-2","").replace("16 November 2026","")
            if re.search(r"\d",residual):numerals.append({"slide":index+1,"text":residual})
        check(not numerals,"No displayed study numbers; only official model name and event date allowed",numerals)
        check(not any(re.search(r"\bTODO\b|lorem|\[insert",t,re.I) for t in texts),"No unfinished placeholders")
        check("Annastasya Nabila Elsa Wulandari" in texts[0],"Current author names")
        spec=json.loads((HERE/"review/CHART_BINDINGS.json").read_text())
        check(len(spec)==1,"Only the real attention example is a data chart")
        chart=ET.fromstring(archive.read(spec[0]["chart"]))
        series=chart.findall(".//c:ser",NS)
        example=next(r for r in evidence["examples"] if r["architecture"]=="gru")
        for i,expected in enumerate((example["alpha_a"],example["alpha_b"])):
            observed=[float(p.text) for p in series[i].findall("./c:val/c:numRef/c:numCache/c:pt/c:v",NS)]
            check(observed==expected,"Original empirical attention values retained",i)
        check(all(a.find("c:delete",NS) is not None and a.find("c:delete",NS).get("val")=="1"
                  for a in chart.findall(".//c:catAx",NS)+chart.findall(".//c:valAx",NS)),
              "Chart numerical axes are hidden")
    shapes=json.loads((HERE/"review/SHAPE_LAYOUT.json").read_text())
    outside=[r for r in shapes if min(r["x"],r["y"],r["w"],r["h"])<0
             or r["x"]+r["w"]>13.33334 or r["y"]+r["h"]>7.50001]
    check(not outside,"Recorded shapes lie within slide canvas",outside)
    pdf=pymupdf.open(HERE/(NAME+".pdf"))
    check(len(pdf)==len(paths),"PDF page count matches PPTX")
    renders=HERE/"review/rendered";renders.mkdir(exist_ok=True)
    sheets=HERE/"review/contact_sheets";sheets.mkdir(exist_ok=True)
    outside_pdf=[];pdf_numbers=[]
    for i,page in enumerate(pdf):
        page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False).save(renders/f"slide_{i+1:02}.png")
        residual=page.get_text().replace("BiMamba-2","").replace("16 November 2026","")
        if re.search(r"\d",residual):pdf_numbers.append({"page":i+1,"text":residual})
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines",[]):
                for span in line["spans"]:
                    r=pymupdf.Rect(span["bbox"])
                    if r.x0 < -1 or r.y0 < -1 or r.x1>page.rect.width+1 or r.y1>page.rect.height+1:
                        outside_pdf.append({"page":i+1,"text":span["text"]})
    check(not outside_pdf,"No text spans outside PDF pages",outside_pdf)
    check(not pdf_numbers,"No numerical study results in rendered PDF",pdf_numbers)
    for offset in range(0,len(pdf),6):
        indices=list(range(offset,min(offset+6,len(pdf))))
        sheet=Image.new("RGB",(1920,1190),"#E8EEEB");draw=ImageDraw.Draw(sheet)
        for j,index in enumerate(indices):
            im=Image.open(renders/f"slide_{index+1:02}.png").convert("RGB");im.thumbnail((900,340))
            x=40+(j%2)*960;y=20+(j//2)*390
            sheet.paste(im,(x,y+25));draw.text((x,y),f"Slide {index+1}",fill="#18342F")
        sheet.save(sheets/f"slides_{offset+1:02}_{indices[-1]+1:02}.jpg",quality=92)
    failed=[r for r in checks if not r["passed"]]
    write("review/DOCUMENT_CHECKS.json",{
        "checked_utc":datetime.now(timezone.utc).isoformat(),"scope":"Document generation and saved-result checks only",
        "research_computations":False,"slides":len(paths),"main_slides":21,"n_checks":len(checks),
        "n_failed":len(failed),"checks":checks,"pdf_sha256":sha(HERE/(NAME+".pdf")),
        "pptx_sha256":sha(HERE/(NAME+".pptx")),"numeral_exceptions":["BiMamba-2","16 November 2026"],
        "manual_visual_review":"Recorded separately after rendered-page inspection"})
    print(f"Document checks: {len(checks)}; failed: {len(failed)}; rendered pages: {len(pdf)}")
    for failure in failed:print(failure)
    raise SystemExit(bool(failed))


if __name__=="__main__":main()
