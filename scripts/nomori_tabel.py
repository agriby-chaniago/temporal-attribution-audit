"""Beri judul dan nomor pada tabel, mengikuti Panduan Tugas Akhir UHB.

Jalankan:
    python3 scripts/nomori_tabel.py --sisipkan   # tulis judul ke NASKAH-SUMBER.md (sekali)
    python3 scripts/nomori_tabel.py --nomori     # beri nomor pada sempro dan semhas

Kenapa judul dan nomor dipisah
------------------------------
Panduan menuntut nomor tabel **per bab** (`Tabel 3.1` = tabel pertama Bab III). Nomor itu
tidak dapat dibakukan di dalam `NASKAH-SUMBER.md`, sebab kedua dokumen turunannya memuat
himpunan tabel yang berbeda: Bab III sempro memuat 12 tabel, sedangkan Bab III semhas
memuat 20 — delapan tabel di dalam blok berpenanda `HASIL-ONLY` hilang dari sempro namun
tetap di tempatnya pada semhas. Menomori di sumber akan membuat salah satu dokumen bolong.

Karena itu **judul** tinggal di naskah sumber, sebab ia isi; **nomor** dihitung ulang pada
tiap dokumen sesudah pemisahan, sebab ia bergantung dokumen.

Yang tidak dinomori
-------------------
Tabel pada halaman depan (lembar pengesahan) dan pada Lampiran tidak memperoleh nomor bab.
Panduan menomori lampiran sebagai lampiran, bukan sebagai tabel bab.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
SUMBER = AKAR / "naskah" / "NASKAH-SUMBER.md"
DOKUMEN = [AKAR / "naskah" / "sempro-skripsi.md",
           AKAR / "naskah" / "semhas-skripsi.md"]

PENANDA = "Tabel — "
ROMAWI = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}

# Judul menurut urutan kemunculan tabel di NASKAH-SUMBER. Nomor 1 adalah tabel tanda tangan
# pada lembar pengesahan dan 88 ke atas berada di Lampiran; keduanya tidak dinomori.
JUDUL: dict[int, str] = {
    2: "Keaslian penelitian",
    3: "Tanda motorik Parkinson menurut sifat dan skala waktunya",
    4: "Tiga peta temporal yang dibandingkan",
    5: "Struktur state rekuren ketiga encoder",
    6: "Biaya komputasi BiMamba-2 dan BiGRU menurut panjang urutan",
    7: "Penelitian terdahulu deteksi Parkinson dari tulisan tangan",
    8: "Penelitian terdahulu tentang kesahihan atribusi pada deret waktu",
    9: "Performa lintas kohort dan simpangan antar seed",
    10: "Kesetiaan dan keselarasan peta pada NewHandPD",
    11: "Nilai p retensi menurut jumlah seed",
    12: "Definisi operasional variabel penelitian",
    13: "Spesifikasi perangkat keras dan perangkat lunak",
    14: "Tiga tugas pada UCI 395 dan perannya",
    15: "Fraksi baris bertekanan nol pada STCP menurut kelompok",
    16: "Identitas kolom sinyal NewHandPD",
    17: "Peran dan daya pisah tiap kanal BiSP",
    18: "Butir verifikasi struktur data",
    19: "Enam kanal masukan model",
    20: "Konfigurasi pelatihan yang dibekukan",
    21: "Delapan skenario pengujian",
    22: "Matriks transfer lintas tugas",
    23: "Pengaruh kebocoran subjek pada transfer lintas tugas",
    24: "Daya pisah penanda tremor menurut tugas",
    25: "Daya pisah penanda menurut pita frekuensi",
    26: "Redaman penanda lambat terhadap frekuensi gangguan",
    27: "Daya pisah penanda lambat menurut tugas",
    28: "Hitungan daya uji penanda lambat pada DST",
    29: "Daya pisah perancu tingkat subjek pada STCP",
    30: "Korelasi perancu terhadap keluaran model pada STCP",
    31: "Dukungan penanda dan episode melayang menurut kelompok",
    32: "Selisih perancu antara penderita dan kontrol",
    33: "Perbandingan perancu antar ketiga tugas",
    34: "Kesetiaan atensi terhadap atribusi Shapley menurut arsitektur",
    35: "Kesetiaan menurut resolusi patch dan cara agregasi",
    36: "Sebaran kesetiaan pada kontrol positif",
    37: "Kesetiaan berpasangan alpha dan phi",
    38: "Jumlah jendela per rekaman menurut resolusi patch",
    39: "Pemisahan kenaikan kesetiaan: agregasi berbanding resolusi model",
    40: "Tiga diagnostik atas kontrol label acak",
    41: "Uji kewarasan: permutasi bobot dan ablasi mean pooling",
    42: "Penurunan performa akibat pengacakan bobot",
    43: "Metrik evaluasi menurut aspek yang diukur",
    44: "Besaran aturan keputusan pada Skenario S3",
    45: "Performa klasifikasi kedua arm pra-registrasi",
    46: "Performa arm ketiga BiMamba-3",
    47: "Simpangan baku antar seed menurut jumlah seed",
    48: "Perbandingan arm ketiga terhadap arm pra-registrasi",
    49: "Tiga definisi retensi lintas kohort",
    50: "Unit penganggitan ulang menurut sumber ketidakpastian",
    51: "Pembanding netral bagi tiap metrik peta",
    52: "Kalibrasi ambang keselarasan dari kontrol positif",
    53: "Matriks empat kemungkinan hasil",
    54: "Sel yang terisi pada matriks skenario hasil",
    55: "Uji berpasangan besar keselarasan atas seluruh subjek",
    56: "Kekhususan keselarasan terhadap penyakit",
    57: "Ekspresivitas state dan keselarasan peta",
    58: "Kesepakatan peta antar pasangan arsitektur",
    59: "Ketahanan keselarasan pada lima seed",
    60: "Spektrum rotasi state BiMamba-3",
    61: "Penggelembungan akibat pembagian tingkat rekaman",
    62: "Empat cabang hasil retensi yang ditetapkan di muka",
    63: "Retensi lintas kohort per seed",
    64: "Retensi tiap arm beserta selang kepercayaannya",
    65: "Perbandingan retensi antar arm pada sepuluh seed",
    66: "Retensi menurut ketiga definisi",
    67: "Regresi retensi terhadap performa awal",
    68: "Pencocokan performa awal antar arm",
    69: "Performa baseline fitur agregat pada kohort utama",
    70: "Kehilangan performa baseline fitur agregat lintas kohort",
    71: "Perbandingan kehilangan antara model sekuens dan baseline",
    72: "Peringkat metode pada kohort kedua",
    73: "Perbandingan arsitektur pada kohort kedua",
    74: "Daya uji Rumusan Masalah 5 pada STCP berbanding NewHandPD",
    75: "Keselarasan peta pada NewHandPD menurut arm",
    76: "Jawaban Rumusan Masalah 5 pada kohort dengan kontrol lebih banyak",
    77: "Ruang pencarian penyetelan hyperparameter",
    78: "Performa sebelum dan sesudah penyetelan berimbang",
    79: "Konfigurasi yang paling sering terpilih tiap arm",
    80: "Perbandingan arsitektur pada konfigurasi tersetel",
    81: "Aturan agregasi peta antar seed",
    82: "Reproduktibilitas peta: satu seed berbanding ensemble",
    83: "Kurva reproduktibilitas menurut jumlah seed",
    84: "Peringkat arsitektur menurut keluarga tugas",
    85: "Usia subjek NewHandPD menurut kelompok",
    86: "Performa sebelum dan sesudah usia dicocokkan",
    87: "Jadwal penelitian",
}


def sisipkan() -> int:
    """Tulis baris judul di atas tiap tabel pada NASKAH-SUMBER.md."""
    L = SUMBER.read_text().split("\n")
    keluar: list[str] = []
    n = dipasang = 0
    for i, l in enumerate(L):
        if l.strip().startswith("|---"):
            n += 1
            judul = JUDUL.get(n)
            if judul and not (len(keluar) >= 2 and keluar[-2].startswith(PENANDA)):
                kepala = keluar.pop()                 # baris kepala tabel
                if keluar and keluar[-1].strip():
                    keluar.append("")
                keluar += [PENANDA + judul, "", kepala]
                dipasang += 1
        keluar.append(l)
    SUMBER.write_text("\n".join(keluar))
    print(f"{n} tabel ditemukan, {dipasang} judul disisipkan, "
          f"{n - dipasang} sengaja tanpa nomor (halaman depan dan Lampiran)")
    return 0


def nomori() -> int:
    """Ganti penanda judul menjadi 'Tabel N.M' pada tiap dokumen turunan."""
    for jalur in DOKUMEN:
        if not jalur.exists():
            continue
        L = jalur.read_text().split("\n")
        bab, hitung, n = None, {}, 0
        for i, l in enumerate(L):
            m = re.match(r"^# BAB ([IVX]+)", l)
            if m:
                bab = ROMAWI.get(m.group(1))
            elif re.match(r"^# (DAFTAR PUSTAKA|LAMPIRAN)", l):
                bab = None
            if l.startswith(PENANDA) and bab:
                hitung[bab] = hitung.get(bab, 0) + 1
                L[i] = f"Tabel {bab}.{hitung[bab]} {l[len(PENANDA):]}"
                n += 1
        jalur.write_text("\n".join(L))
        rincian = ", ".join(f"Bab {b}: {c}" for b, c in sorted(hitung.items()))
        print(f"  {jalur.name}: {n} tabel dinomori  ({rincian})")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sisipkan", action="store_true")
    p.add_argument("--nomori", action="store_true")
    a = p.parse_args()
    if a.sisipkan:
        return sisipkan()
    if a.nomori:
        return nomori()
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
