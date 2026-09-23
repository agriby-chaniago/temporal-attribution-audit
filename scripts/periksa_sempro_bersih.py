"""Periksa apakah sempro benar-benar bebas hasil empiris.

Jalankan: python3 scripts/periksa_sempro_bersih.py

Latar
-----
Tiga kali berturut-turut pemeriksaan manual gagal menangkap kebocoran, dan tiap
kali sebabnya sama: pemeriksanya dibuat dari kebocoran yang **sudah diketahui**,
sehingga hanya mampu menemukan itu. Berkas ini mengunci polanya supaya
pemeriksaan berikutnya tidak bergantung pada ingatan.

Yang dicari **pola**, bukan daftar angka tertentu:

- angka berpresisi hasil (desimal empat digit, p-value, AUC)
- kosakata vonis: gagal, berhasil, terbukti, didukung, dibantah, ditarik
- penanda status lampiran: TERJAWAB, TERTUTUP, coret ~~...~~
- kronologi eksperimen: "setelah hasil ... diketahui", "sebelum lima seed"
- klaim ketersediaan artefak: "sudah memuat", "sudah dijalankan"
- klaim absolut tanpa pembatas: "pertama kali", "satu-satunya"

Sebagian pola bersifat sah pada konteks tertentu — aritmetika, aturan keputusan,
spesifikasi desain. Baris semacam itu didaftar pada PENGECUALIAN beserta
alasannya, sehingga pengecualiannya sendiri dapat ditinjau.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
SEMPRO = AKAR / "naskah" / "sempro-skripsi.md"

# Pola yang selalu merupakan kebocoran, apa pun konteksnya.
POLA_KERAS = [
    # Tanda minus/plus di depan memutus \b, dan tiga desimal juga lolos dari
    # pola empat desimal. Kedua celah itu pernah meloloskan kebocoran nyata.
    ("angka presisi hasil", r"(?<![\d,])[01],\d{3,4}(?![\d])"),
    # Rentang nilai terukur berdesimal dua, mis. "0,93 sampai 0,97". Pola ini
    # menangkap kebocoran yang lolos dari pola empat desimal di atas.
    ("rentang nilai terukur", r"[−+\-]?[01],\d{2}\s+(sampai|hingga|dan)\s+[−+\-]?[01],\d{2}\b"),
    ("Gini bernilai", r"Gini[^.]{0,40}\b[01],\d{2}\b"),
    ("p-value", r"\bp\s*[=<]\s*0,0\d"),
    ("AUC bernilai", r"AUC\s+[01],\d"),
    ("status lampiran", r"\bTERJAWAB\b|\bTERTUTUP\b|\bTERUKUR\b|~~"),   # huruf besar, peka huruf
    ("kronologi eksperimen", r"sebelum lima seed|ketika hanya \w+ dan \w+ yang ada|"
                             r"belum menjadi bagian penelitian|artefak Skenario S\d sudah"),
    ("klaim artefak tersedia", r"sudah memuat|sudah dijalankan|sudah tersedia"),
    ("klaim absolut", r"pertama kali|belum pernah ada"),
    ("bahasa laporan", r"hasil menunjukkan|hasil pengujian|jawaban yang diperoleh|"
                       r"pengukuran menunjukkan|Hasilnya,"),
]

# Kosakata vonis hanya menjadi kebocoran bila **satu baris dengan angka berbentuk
# hasil**. Tanpa angka, kata seperti "gagal" lazim dipakai pada pengandaian,
# aturan keputusan, prinsip metodologis, dan kutipan penelitian terdahulu —
# menandainya tanpa syarat menghasilkan ratusan temuan palsu.
VONIS = r"\b(gagal|berhasil|terbukti|dibantah|ditarik|tidak didukung|meleset|runtuh)\b"
ANGKA_HASIL = r"\b[01],\d{3,4}\b|\bp\s*[=<]\s*0,\d"

# Baris yang cocok pola namun sah. Kunci = potongan unik baris; nilai = alasannya.
PENGECUALIAN = {
    "minimum 2/32 = 0,0625": "aritmetika p minimum uji peringkat pada n=5, bukan hasil",
    "| p < 0,05 |": "aturan keputusan pra-registrasi, bukan hasil",
    "yaitu 0,025": "taraf alfa Bonferroni, parameter aturan keputusan",
    "Kemungkinan itu diuji langsung": "deskripsi prosedur; hasilnya sudah dipindah",
    "tidak dapat dipakai karena tidak tersedia rumus tertutup": "kata 'tertutup' pada 'rumus tertutup'",
    "nilai Shapley-nya memiliki bentuk tertutup": "kata 'tertutup' pada 'bentuk tertutup'",
    "apabila kolom koordinat sederhana ternyata tidak ada": "pengandaian, bukan laporan",
    "Apabila keselarasan terbukti": "pengandaian pada Manfaat, bukan laporan",
    "gagal atau sangat lemah": "isi ramalan pra-registrasi, bukan hasil",
    "pos_weight ≈ 37/170": "parameter pelatihan yang diturunkan dari komposisi data, sejenis rasio kelas yang memang sudah dinyatakan di sempro — bukan hasil eksperimen",
}


def sah(baris: str) -> str | None:
    for kunci, alasan in PENGECUALIAN.items():
        if kunci in baris:
            return alasan
    return None


def main() -> int:
    if not SEMPRO.exists():
        print(f"tidak ada: {SEMPRO}"); return 1
    L = SEMPRO.read_text().split("\n")
    kepala = [(i + 1, l) for i, l in enumerate(L) if l.startswith("#")]

    def judul(n: int) -> str:
        x = [k for k in kepala if k[0] <= n]
        return x[-1][1][:52] if x else "?"

    temuan, dimaafkan = [], []
    for i, l in enumerate(L, 1):
        cocok = None
        for nama, pat in POLA_KERAS:
            bendera = 0 if nama == "status lampiran" else re.I   # status peka huruf besar
            if re.search(pat, l, bendera):
                cocok = nama; break
        if cocok is None and re.search(VONIS, l, re.I) and re.search(ANGKA_HASIL, l):
            cocok = "vonis berangka"
        for nama in ([cocok] if cocok else []):
            if True:
                alasan = sah(l)
                (dimaafkan if alasan else temuan).append((nama, i, judul(i), l.strip()[:96], alasan))
                break

    print("=" * 78)
    print("PEMERIKSAAN SEMPRO BEBAS HASIL")
    print("=" * 78)
    print(f"\n{len(L)} baris diperiksa terhadap {len(POLA_KERAS)} pola keras + kosakata vonis berangka\n")

    if dimaafkan:
        print(f"--- {len(dimaafkan)} baris cocok pola namun sah ---")
        for nama, i, _, l, alasan in dimaafkan:
            print(f"  {i:5d} [{nama}] {alasan}")
        print()

    if temuan:
        print(f"--- {len(temuan)} KEBOCORAN ---")
        for nama, i, hd, l, _ in temuan:
            print(f"  {i:5d} [{nama}] {hd}\n        {l}")
        print("\nGAGAL: sempro masih memuat hasil empiris.")
        return 1

    print("LOLOS: tidak ada kebocoran hasil terdeteksi.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
