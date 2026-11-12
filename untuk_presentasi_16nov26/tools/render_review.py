"""Render all delivered PDF pages and check page/text geometry, without research analysis."""

import json
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]


def main():
    doc = pymupdf.open(HERE / "presentasi_attention_16nov2026.pdf")
    renders = HERE / "review/rendered"; renders.mkdir(exist_ok=True)
    sheets = HERE / "review/contact_sheets"; sheets.mkdir(exist_ok=True)
    report = {"pages": len(doc), "text_spans_outside_page": [], "rendered": []}
    for index, page in enumerate(doc):
        target = renders / f"slide_{index+1:02}.png"
        page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(target)
        report["rendered"].append(target.name)
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    rect = pymupdf.Rect(span["bbox"])
                    if rect.x0 < -1 or rect.y0 < -1 or rect.x1 > page.rect.width+1 or rect.y1 > page.rect.height+1:
                        report["text_spans_outside_page"].append({"page": index+1, "text": span["text"]})
    for offset in range(0, len(doc), 6):
        indices = list(range(offset, min(offset+6, len(doc))))
        sheet = Image.new("RGB", (1920, 1190), "#E8EEEB")
        draw = ImageDraw.Draw(sheet)
        for j, index in enumerate(indices):
            im = Image.open(renders / f"slide_{index+1:02}.png").convert("RGB")
            im.thumbnail((900, 506))
            x = 40 + (j % 2) * 960; y = 20 + (j // 2) * 390
            # Keep each overview tile legible without changing the full-resolution render.
            im.thumbnail((900, 340))
            sheet.paste(im, (x, y+25))
            draw.text((x, y), f"Slide {index+1}", fill="#18342F")
        sheet.save(sheets / f"slides_{offset+1:02}_{indices[-1]+1:02}.jpg", quality=92)
    (HERE / "review/PDF_GEOMETRY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Rendered {len(doc)} pages; {len(report['text_spans_outside_page'])} text spans outside pages.")


if __name__ == "__main__":
    main()
