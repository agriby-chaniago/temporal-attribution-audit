"""Gabungkan seluruh notebook tahapan menjadi satu berkas beku.

Jalankan: python3 scripts/gabung_notebook.py

Menutup rencana yang ditetapkan di awal proyek: notebook dipisah selama
eksplorasi supaya tiap tahap dapat dijalankan ulang sendiri-sendiri, lalu
disatukan ketika sudah final dan tinggal menulis.

Berkas hasil bersifat **arsip beku**, bukan notebook yang dijalankan ulang dari
atas ke bawah. Alasannya jujur: tiap tahap memakai nama variabel yang tumpang
tindih dan memuat artefaknya sendiri, sehingga menjalankan gabungannya secara
berurutan akan saling menimpa. Yang dijaga adalah keterbacaan seluruh hasil dalam
satu tempat, beserta seluruh keluaran yang sudah dihitung.

Sumber kebenaran tetap `scripts/build_nb_*.py`. Notebook per tahap dibangun dari
sana, dan berkas gabungan ini dibangun dari notebook itu.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import nbformat as nbf

AKAR = Path(__file__).resolve().parent.parent
KELUARAN = AKAR / "notebooks" / "00_gabungan_beku.ipynb"
# kecualikan keluaran sendiri, agar menjalankan ulang tidak menggabung hasil sebelumnya
INDUK = AKAR / "notebooks" / "00_induk.ipynb"
# kecualikan keluaran sendiri dan notebook induk: induk adalah alur tersendiri yang
# dapat dijalankan dari atas ke bawah, bukan salah satu tahap.
SUMBER = sorted(f for f in (AKAR / "notebooks").glob("[0-9][0-9]_*.ipynb")
                if f not in {KELUARAN, INDUK})

JUDUL = {
    "01": "Verifikasi struktur data",
    "02": "Prapemrosesan dan rekayasa kanal",
    "03": "Arsitektur dan verifikasi encoder",
    "04": "Kontrol positif (S1)",
    "05": "Atribusi Shapley (S1 lanjutan)",
    "06": "Pilot resolusi patch (S2)",
    "07": "Eksperimen utama (S3)",
    "08": "Kesetiaan atensi (S6)",
    "09": "Keselarasan terhadap penanda motorik (S7)",
    "10": "Uji kewarasan pendukung (S8)",
    "11": "Generalisasi lintas tugas (S4)",
    "12": "Replikasi NewHandPD (S5)",
    "13": "Pembagian fold tingkat rekaman (C1)",
}


def main() -> int:
    if not SUMBER:
        print("tidak ada notebook sumber"); return 1

    gabungan = nbf.v4.new_notebook()
    sel = []

    daftar = "\n".join(
        f"{i}. **{JUDUL.get(f.name[:2], f.name)}** — `{f.name}`"
        for i, f in enumerate(SUMBER, 1)
    )
    sel.append(nbf.v4.new_markdown_cell(f"""# Arsip Beku — Audit Atribusi Temporal Bidirectional Mamba-2

Gabungan seluruh notebook tahapan, dibekukan pada **{date.today().isoformat()}**.

## Sifat berkas ini

Ini **arsip**, bukan notebook yang dijalankan ulang dari atas ke bawah. Tiap tahap memakai nama
variabel yang tumpang tindih dan memuat artefaknya sendiri, sehingga menjalankan gabungannya secara
berurutan akan saling menimpa. Yang dijaga adalah keterbacaan seluruh hasil dalam satu tempat,
beserta seluruh keluaran yang sudah dihitung.

Untuk menjalankan ulang satu tahap, pakai notebook aslinya di `notebooks/`. Sumber kebenaran seluruh
kode adalah `scripts/build_nb_*.py`; notebook per tahap dibangun dari sana, dan berkas ini dibangun
dari notebook itu.

## Isi

{daftar}

## Analisis berdiri sendiri

Sebagian analisis tidak berbentuk notebook melainkan skrip, dan tidak termasuk dalam arsip ini:

- `scripts/analisis_a0_perancu_stcp.py` — audit perancu STCP dan massa atensi
- `scripts/analisis_a3_penanda_lambat.py` — penanda lambat pada DST
- `scripts/analisis_b1_bootstrap.py` — bootstrap dan uji berpasangan peta
- `scripts/analisis_e7_spektrum.py` — spektrum rotasi state BiMamba-3
- `scripts/verifikasi_encoder.py` — tujuh pemeriksaan encoder
- `scripts/analisis_a_baseline_fitur.py` — baseline fitur kinematik agregat, dua kohort
- `scripts/analisis_rm5_newhandpd.py` — RM5 pada 35 subjek kontrol
- `scripts/analisis_rm2_tuning.py` — penyetelan hyperparameter berimbang tiga arm

## Notebook induk

`notebooks/00_induk.ipynb` adalah alur tunggal yang **dapat dijalankan dari atas ke bawah**: ia
memuat artefak `results/`, menghitung ulang seluruh angka kunci, menyimpan gambar ke `figures/`, dan
diakhiri validasi silang terhadap naskah. Ia bukan bagian arsip ini.
"""))

    n_sel, n_out = 0, 0
    for f in SUMBER:
        nb = nbf.read(f, as_version=4)
        kode = sum(1 for c in nb.cells if c.cell_type == "code")
        keluaran = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
        n_sel += len(nb.cells); n_out += keluaran
        sel.append(nbf.v4.new_markdown_cell(
            f"""---

# {JUDUL.get(f.name[:2], f.name)}

`{f.name}` · {len(nb.cells)} sel, {kode} sel kode, {keluaran} beroutput"""))
        for c in nb.cells:
            sel.append(c)

    gabungan["cells"] = sel
    nbf.validator.normalize(gabungan)   # beri id unik pada sel yang belum punya
    gabungan["metadata"] = {
        "kernelspec": {"display_name": "Skripsi (.venv)", "language": "python", "name": "skripsi"},
        "language_info": {"name": "python", "version": "3.11.8"},
        "beku": {"tanggal": date.today().isoformat(), "n_notebook": len(SUMBER)},
    }
    nbf.write(gabungan, open(KELUARAN, "w"))

    ukuran = KELUARAN.stat().st_size / 1e6
    print(f"ditulis: {KELUARAN}")
    print(f"  {len(SUMBER)} notebook digabung")
    print(f"  {len(sel)} sel total ({n_sel} dari sumber + {len(sel)-n_sel} penanda)")
    print(f"  {n_out} sel beroutput dipertahankan")
    print(f"  {ukuran:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
