"""Add editable native-chart CI endpoints copied from stored estimates."""

from __future__ import annotations

import json
import posixpath
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from defusedxml import minidom

HERE = Path(__file__).resolve().parents[1]
C = "http://schemas.openxmlformats.org/drawingml/2006/chart"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def column_name(index):
    result = ""
    while index:
        index, remainder = divmod(index-1, 26)
        result = chr(65+remainder) + result
    return result


def add_ci_columns(workbook_bytes, all_bounds):
    """Extend the chart's own embedded workbook with literal CI lengths."""
    source = ZipFile(BytesIO(workbook_bytes))
    doc = minidom.parseString(source.read("xl/worksheets/sheet1.xml"))
    workbook = minidom.parseString(source.read("xl/workbook.xml"))
    sheet_name = workbook.getElementsByTagNameNS(S, "sheet")[0].getAttribute("name")
    rows = {int(row.getAttribute("r")): row for row in doc.getElementsByTagNameNS(S, "row")}
    existing_columns = len(all_bounds)+1  # category + original response series
    references = []
    for series_index, bounds in enumerate(all_bounds):
        sides = {}
        for side_index, side in enumerate(("plus", "minus")):
            col = column_name(existing_columns + 2*series_index + side_index + 1)
            sides[side] = f"'{sheet_name}'!${col}$2:${col}${len(bounds)+1}"
            header = doc.createElementNS(S, "c")
            header.setAttribute("r", f"{col}1"); header.setAttribute("t", "inlineStr")
            inline = doc.createElementNS(S, "is"); text = doc.createElementNS(S, "t")
            text.appendChild(doc.createTextNode(f"CI_{side}_{series_index}"))
            inline.appendChild(text);header.appendChild(inline);rows[1].appendChild(header)
            for index, point in enumerate(bounds, 2):
                length = (point["high"]-point["estimate"] if side=="plus"
                          else point["estimate"]-point["low"])
                assert length >= 0
                cell = doc.createElementNS(S, "c");cell.setAttribute("r", f"{col}{index}")
                value = doc.createElementNS(S, "v");value.appendChild(doc.createTextNode(repr(length)))
                cell.appendChild(value);rows[index].appendChild(cell)
        references.append(sides)
    last_column = column_name(existing_columns + 2*len(all_bounds))
    doc.getElementsByTagNameNS(S, "dimension")[0].setAttribute("ref", f"A1:{last_column}{max(rows)}")
    for row in rows.values():
        row.setAttribute("spans", f"1:{existing_columns + 2*len(all_bounds)}")
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as target:
        for info in source.infolist():
            target.writestr(info, doc.toxml(encoding="utf-8") if info.filename=="xl/worksheets/sheet1.xml" else source.read(info.filename))
    return buffer.getvalue(), references


def main() -> None:
    specs = json.loads((HERE / "review/CHART_BINDINGS.json").read_text())
    source = HERE / "review/presentasi_raw.pptx"
    target = HERE / "presentasi_attention_16nov2026.pptx"
    changed = []
    with ZipFile(source) as src, ZipFile(target, "w", ZIP_DEFLATED) as dst:
        mapping = {spec["chart"]: spec for spec in specs if spec["errors"]}
        workbooks = {}
        chart_ranges = {}
        for filename, spec in mapping.items():
            rels_path = posixpath.join(posixpath.dirname(filename), "_rels", posixpath.basename(filename)+".rels")
            rels = minidom.parseString(src.read(rels_path))
            target_rel = [r.getAttribute("Target") for r in rels.getElementsByTagName("Relationship") if r.getAttribute("Type").endswith("/package")]
            assert len(target_rel)==1
            workbook_path = posixpath.normpath(posixpath.join(posixpath.dirname(filename), target_rel[0]))
            workbooks[workbook_path], chart_ranges[filename] = add_ci_columns(src.read(workbook_path), spec["errors"])
        for info in src.infolist():
            payload = workbooks.get(info.filename, src.read(info.filename))
            if info.filename in mapping:
                spec = mapping[info.filename]
                doc = minidom.parseString(payload)
                series = doc.getElementsByTagNameNS(C, "ser")
                assert len(series) == len(spec["errors"])

                def element(local, value=None, namespace=C, prefix="c"):
                    node = doc.createElementNS(namespace, f"{prefix}:{local}")
                    if value is not None:
                        node.setAttribute("val", str(value))
                    return node

                for series_index, (ser, bounds) in enumerate(zip(series, spec["errors"])):
                    assert not ser.getElementsByTagNameNS(C, "errBars")
                    errors = element("errBars")
                    # Category charts use their value dimension; let the renderer
                    # rotate that dimension with a horizontal bar chart.
                    errors.appendChild(element("errBarType", "both"))
                    errors.appendChild(element("errValType", "cust"))
                    errors.appendChild(element("noEndCap", 0))
                    for side in ("plus", "minus"):
                        side_node = element(side)
                        reference = element("numRef")
                        formula = element("f")
                        formula.appendChild(doc.createTextNode(chart_ranges[info.filename][series_index][side]))
                        reference.appendChild(formula)
                        literals = element("numCache")
                        fmt = element("formatCode")
                        fmt.appendChild(doc.createTextNode("General"))
                        literals.appendChild(fmt)
                        literals.appendChild(element("ptCount", len(bounds)))
                        for i, point in enumerate(bounds):
                            # Rendering lengths, not recomputed uncertainty estimates.
                            length = (point["high"] - point["estimate"] if side == "plus"
                                      else point["estimate"] - point["low"])
                            assert length >= 0
                            p = element("pt"); p.setAttribute("idx", str(i))
                            value = element("v")
                            value.appendChild(doc.createTextNode(repr(length)))
                            p.appendChild(value); literals.appendChild(p)
                        reference.appendChild(literals);side_node.appendChild(reference); errors.appendChild(side_node)
                    shape = element("spPr")
                    line = element("ln", namespace=A, prefix="a");line.setAttribute("w", "15875")
                    fill = element("solidFill", namespace=A, prefix="a")
                    rgb = element("srgbClr", namespace=A, prefix="a");rgb.setAttribute("val", "263F36")
                    fill.appendChild(rgb);line.appendChild(fill);shape.appendChild(line);errors.appendChild(shape)
                    cat = ser.getElementsByTagNameNS(C, "cat")
                    assert len(cat) == 1
                    ser.insertBefore(errors, cat[0])
                payload = doc.toxml(encoding="utf-8")
                changed.append(info.filename)
            dst.writestr(info, payload)
    (HERE / "review/NATIVE_CI_REPORT.json").write_text(json.dumps({
        "charts_with_native_custom_intervals": changed,
        "interval_source": "stored estimates and CI endpoints; no new statistics",
        "original_response_values_preserved": True,
        "embedded_workbooks_extended_with_literal_ci_lengths": list(workbooks),
        "final_pptx": target.name,
    }, indent=2) + "\n")
    print(f"Final PPTX created; native CI ranges on {len(changed)} charts.")


if __name__ == "__main__":
    main()
