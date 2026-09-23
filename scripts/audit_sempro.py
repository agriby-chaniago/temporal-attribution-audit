"""Audit menyeluruh naskah sempro pada delapan sumbu.

Jalankan: python3 scripts/audit_sempro.py

Berbeda dari `periksa_sempro_bersih.py` yang hanya memburu kebocoran hasil,
berkas ini memeriksa keutuhan dokumen sebagai proposal: rujukan silang yang
menggantung, daftar isi yang tidak cocok, penomoran yang bolong atau kembar,
tabel yang kolomnya tidak sama, penanda tebal yang tidak tertutup, dan sitasi.

Sumbu rujukan silang adalah yang paling penting: sempro dibangun dengan
membuang banyak blok, sehingga rujukan ke subbab yang ikut terbuang akan
menggantung tanpa ada yang menyadarinya.

Sumbu kedelapan ditambahkan 6 September 2026 setelah sebuah cacat lolos dari
ketujuh sumbu lainnya. Naskah menyebut "tabel keruntuhan peta pada Subbab 3.8"
sebanyak dua kali, padahal tabel itu tidak pernah ada. Sumbu pertama meloloskannya
sebab yang diperiksanya hanya **nomor** subbab, dan Subbab 3.8 memang ada; yang
tidak ada adalah artefak **di dalamnya**. Sumbu kedelapan memeriksa hal itu:
tiap rujukan berbentuk "<tabel|matriks|daftar> ... pada Subbab N" harus mendarat
di subbab yang benar-benar memuat tabel.

Sumbu kedelapan dijalankan pada **kedua** dokumen, bukan hanya sempro. Cacat
semacam ini dapat hidup di dalam prosa yang berpenanda hasil, dan prosa itu
tidak pernah sampai ke sempro sehingga tidak akan pernah terlihat bila yang
diperiksa hanya satu dokumen.

Dua skema penomoran
-------------------
Ditambahkan 8 September 2026, sebelum naskah direstrukturisasi mengikuti Panduan
Tugas Akhir UHB. Panduan menuntut subbab berhuruf (`A.` `1.` `a.`), sementara
naskah masih memakai desimal (`3.6.14`). `struktur()` mengenali **keduanya**,
dan seluruh sumbu bersandar padanya.

Alasannya soal urutan. Memutakhirkan pemeriksa ke skema huruf lebih dahulu
membuatnya buta terhadap naskah yang masih desimal; memutakhirkannya belakangan
membuatnya buta sepanjang peralihan. Keduanya melepaskan pengawasan persis pada
saat naskah paling banyak berubah. Mendukung dua skema sekaligus menghapus
pilihan buruk itu.

Perilaku pada skema huruf diikat uji regresi: `python3 scripts/audit_sempro.py --uji`.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
BERKAS = AKAR / "naskah" / "sempro-skripsi.md"
SEMHAS = AKAR / "naskah" / "semhas-skripsi.md"

# Kata benda artefak yang, bila dirujuk "pada Subbab N", menuntut subbab itu
# benar-benar memuat tabel. Kata "gambar" sengaja tidak dimasukkan: naskah tidak
# menyisipkan berkas gambar, sehingga tidak ada jejak mekanis yang dapat diuji.
ARTEFAK = ("tabel", "matriks", "daftar")

# Jarak maksimum antara kata artefak dan "pada Subbab", dihitung dalam kata.
# Tiga sudah cukup bagi sebutan terpanjang yang dipakai naskah, dan menolak
# kalimat seperti "tabel di atas dinilai terhadap ambang yang dikalibrasi pada
# Subbab 3.6.7", yang "pada Subbab"-nya milik ambang, bukan milik tabel.
JARAK_KATA = 3

# Skenario yang memang dirancang; rujukan ke luar daftar ini menggantung.
SKENARIO = {f"S{i}" for i in range(1, 9)}


def muat() -> list[str]:
    return BERKAS.read_text().split("\n")


def heading(L: list[str]) -> list[tuple[int, int, str]]:
    out = []
    for i, l in enumerate(L, 1):
        m = re.match(r"^(#{1,6})\s+(.*)", l)
        if m:
            out.append((i, len(m.group(1)), m.group(2).strip()))
    return out


ROMAWI = "[IVXLC]+"

# Nomor kanonik pada skema huruf, mis. "III.F.3". Dipakai seragam oleh seluruh
# sumbu supaya kedua skema dapat diperiksa alat yang sama.
POLA_NOMOR = r"\d+(?:\.\d+)*|" + ROMAWI + r"\.[A-Z](?:\.\d+)*(?:\.[a-z])?"

_ROMAWI_KE_ANGKA = {"I": "1", "II": "2", "III": "3", "IV": "4", "V": "5"}


def struktur(L: list[str]) -> list[tuple[int, int, str, str]]:
    """Petakan tiap heading ke nomor kanoniknya. Mendukung DUA skema sekaligus.

    Desimal (sebelum restrukturisasi):
        '## 3.6.14 Reproduktibilitas'            -> '3.6.14'

    Huruf (sesudah, mengikuti Panduan Tugas Akhir UHB) — nomor disusun dari
    BAB yang sedang berlaku, sebab huruf subbab berulang di tiap bab:
        '# BAB III. METODE' + '## F. Prosedur'   -> 'III.F'
        '### 3. Skenario S3'                     -> 'III.F.3'
        '#### a. Kontrol positif'                -> 'III.F.3.a'

    Keduanya sengaja hidup berdampingan. Memutakhirkan pemeriksa ke skema huruf
    lebih dahulu akan membutakannya terhadap naskah yang masih desimal, dan
    memutakhirkannya belakangan membutakannya selama peralihan — yaitu persis
    ketika naskah paling banyak berubah dan pengawasan paling dibutuhkan.

    Mengembalikan (baris, tingkat heading, nomor kanonik, teks judul).
    """
    out: list[tuple[int, int, str, str]] = []
    bab: str | None = None
    jejak: dict[int, str] = {}
    for i, l in enumerate(L, 1):
        m = re.match(r"^(#{1,6})\s+(.*)", l)
        if not m:
            continue
        lv, teks = len(m.group(1)), m.group(2).strip()

        b = re.match(r"^BAB\s+(" + ROMAWI + r")\b", teks)
        if b:
            bab = b.group(1)
            jejak.clear()
            out.append((i, lv, bab, teks))
            continue

        d = re.match(r"^(\d+(?:\.\d+)+)\s", teks)
        if d:
            out.append((i, lv, d.group(1), teks))
            continue

        if bab:
            h = re.match(r"^([A-Z])\.\s+\S", teks)
            if h:
                jejak = {2: h.group(1)}
                out.append((i, lv, f"{bab}.{h.group(1)}", teks))
                continue
            a = re.match(r"^(\d+)\.\s+\S", teks)
            if a and 2 in jejak:
                jejak[3] = a.group(1)
                out.append((i, lv, f"{bab}.{jejak[2]}.{a.group(1)}", teks))
                continue
            c = re.match(r"^([a-z])\.\s+\S", teks)
            if c and 3 in jejak:
                out.append((i, lv, f"{bab}.{jejak[2]}.{jejak[3]}.{c.group(1)}", teks))
                continue
    return out


def nomor_subbab(L: list[str]) -> set[str]:
    """Nomor subbab yang benar-benar ada, beserta induknya."""
    ada: set[str] = set()
    for _, _, nomor, teks in struktur(L):
        ada.add(nomor)
        bag = nomor.split(".")
        for k in range(1, len(bag)):
            ada.add(".".join(bag[:k]))
        # 'BAB III' juga memenuhi rujukan gaya lama ke angka '3'
        if re.fullmatch(ROMAWI, nomor):
            ada.add(_ROMAWI_KE_ANGKA.get(nomor, nomor))
    return ada


def rujukan(L: list[str]) -> list[tuple[int, str]]:
    out = []
    for i, l in enumerate(L, 1):
        for m in re.finditer(r"Subbab\s+(" + POLA_NOMOR + r")", l):
            out.append((i, m.group(1)))
    return out


def lampiran(L: list[str]) -> tuple[set[str], list[tuple[int, str]]]:
    ada = {m.group(1) for _, _, t in heading(L)
           if (m := re.match(r"^LAMPIRAN\s+([A-Z])", t))}
    ruj = [(i, m.group(1)) for i, l in enumerate(L, 1)
           for m in re.finditer(r"Lampiran\s+([A-Z])\b", l)]
    return ada, ruj


def nomor_daftar_isi(blok: str) -> set[str]:
    """Nomor subbab tingkat dua yang tercantum pada blok daftar isi.

    Melayani dua bentuk. Pada skema desimal nomornya tertulis utuh ('1.1'),
    sehingga dibaca apa adanya. Pada skema huruf yang dituntut panduan, daftar
    isi hanya menuliskan hurufnya ('A. Latar Belakang') di bawah baris babnya,
    sehingga bab yang sedang berlaku harus diikuti agar 'A' dapat dipulihkan
    menjadi 'I.A' — tanpa itu huruf A dari tiga bab berbeda akan tampak kembar.
    """
    keluar: set[str] = set()
    bab: str | None = None
    for baris in blok.split("\n"):
        b = re.match(r"^\s*BAB\s+(" + ROMAWI + r")\b", baris)
        if b:
            bab = b.group(1)
            continue
        d = re.match(r"^\s*(\d+\.\d+)\s", baris)
        if d:
            keluar.add(d.group(1))
            continue
        h = re.match(r"^\s*([A-Z])\.\s+\S", baris)
        if h and bab:
            keluar.add(f"{bab}.{h.group(1)}")
    return keluar



def tabel_rusak(L: list[str]) -> list[tuple[int, str]]:
    """Baris tabel yang jumlah kolomnya beda dari baris kepala blok tabelnya."""
    rusak, kepala, n = [], None, 0
    for i, l in enumerate(L, 1):
        s = l.strip()
        if s.startswith("|") and s.endswith("|"):
            kol = len(s.split("|")) - 2
            if kepala is None:
                kepala, n = i, kol
            elif re.fullmatch(r"\|[\s:\-|]+\|", s):
                continue
            elif kol != n:
                rusak.append((i, f"{kol} kolom, kepala tabel baris {kepala} punya {n}"))
        else:
            kepala, n = None, 0
    return rusak


def tabel_kosong(L: list[str]) -> list[int]:
    """Tabel yang punya kepala dan pemisah tetapi tidak punya satu pun baris isi.

    Terjadi ketika baris isinya terbungkus penanda blok dan terbuang, sementara
    kepalanya tertinggal. Pemeriksa jumlah kolom tidak menangkap ini.
    """
    kosong = []
    for i in range(len(L) - 1):
        s, s2 = L[i].strip(), L[i + 1].strip()
        if (s.startswith("|") and s.endswith("|")
                and re.fullmatch(r"\|[\s:\-|]+\|", s2)):
            j = i + 2
            isi = 0
            while j < len(L) and L[j].strip().startswith("|"):
                isi += 1; j += 1
            if isi == 0:
                kosong.append(i + 1)
    return kosong


def rentang_subbab(L: list[str]) -> dict[str, tuple[int, int]]:
    """Nomor subbab -> rentang baris [awal, akhir) berbasis indeks 0.

    Bersandar pada `struktur`, sehingga kedua skema penomoran ikut terlayani.
    Berbeda dari `nomor_subbab`, induk TIDAK disintesis: rentang hanya bermakna
    bagi nomor yang benar-benar punya heading sendiri.
    """
    kepala = [(baris - 1, lv, nomor) for baris, lv, nomor, _ in struktur(L)]
    rentang: dict[str, tuple[int, int]] = {}
    for k, (i, lv, nomor) in enumerate(kepala):
        j = len(L)
        for i2, lv2, _ in kepala[k + 1:]:
            if lv2 <= lv:
                j = i2
                break
        rentang[nomor] = (i, j)
    return rentang


def artefak_menggantung(L: list[str]) -> list[tuple[int, str, str]]:
    """Rujukan ke tabel di dalam subbab yang subbabnya tidak memuat tabel.

    Mengembalikan (baris, kutipan, sebab). Rujukan ke subbab yang nomornya sendiri
    tidak ada sengaja TIDAK dilaporkan di sini — itu urusan sumbu pertama, dan
    melaporkannya dua kali membuat satu cacat tampak seperti dua.
    """
    rentang = rentang_subbab(L)
    pola = re.compile(r"(" + "|".join(ARTEFAK) + r")\s+((?:\S+\s+){0," + str(JARAK_KATA)
                      + r"}?)pada Subbab\s+(" + POLA_NOMOR + r")", re.I)
    keluar = []
    for i, l in enumerate(L, 1):
        for m in pola.finditer(l):
            nomor = m.group(3)
            if nomor not in rentang:
                continue
            a, b = rentang[nomor]
            if not any(x.strip().startswith("|") for x in L[a:b]):
                keluar.append((i, m.group(0).strip(),
                               f"Subbab {nomor} (baris {a+1}-{b}) tidak memuat satu pun tabel"))
    return keluar


def tebal_ganjil(L: list[str]) -> list[int]:
    """Penanda tebal dihitung per **paragraf**, bukan per baris.

    Markdown mengizinkan satu span tebal membentang lintas baris di dalam satu
    paragraf, sehingga menghitung per baris menghasilkan positif palsu untuk
    setiap span yang terpotong pembungkus baris.
    """
    ganjil, awal, jumlah = [], None, 0
    for i, l in enumerate(L, 1):
        if l.strip() == "":
            if awal is not None and jumlah % 2:
                ganjil.append(awal)
            awal, jumlah = None, 0
            continue
        if awal is None:
            awal = i
        jumlah += l.count("**")
    if awal is not None and jumlah % 2:
        ganjil.append(awal)
    return ganjil


def uji_sendiri() -> int:
    """Uji regresi sumbu kedelapan: python3 scripts/audit_sempro.py --uji

    Pemeriksa yang hanya pernah berkata LOLOS tidak dapat dipercaya, sebab tidak
    ada bukti ia masih mampu berkata GAGAL. Keempat kasus di bawah mengikat
    perilakunya: satu cacat sungguhan yang harus tertangkap, dan tiga kalimat
    yang mirip namun harus dibiarkan lewat.
    """
    kasus = [
        ("cacat yang memicu sumbu ini",
         ["## 3.8 Skenario Hasil", "", "Prosa tanpa tabel apa pun.", "",
          "tabel keruntuhan peta pada Subbab 3.8 menempatkan seed."], True),
        ("'pada Subbab' milik ambang, bukan milik tabel",
         ["## 3.6.7 Ambang", "", "Prosa tanpa tabel.", "",
          "Kedua istilah pada tabel di atas dinilai terhadap ambang yang "
          "dikalibrasi pada Subbab 3.6.7."], False),
        ("rujukan sah, subbabnya memuat tabel",
         ["## 3.8 Skenario", "", "| a | b |", "|---|---|", "| 1 | 2 |", "",
          "Lihat matriks keputusan pada Subbab 3.8."], False),
        ("nomor subbab tidak ada — urusan sumbu pertama, jangan dilapor dua kali",
         ["## 3.1 Apa pun", "", "teks", "",
          "tabel keruntuhan peta pada Subbab 9.9 disebut."], False),
    ]
    gagal = 0
    print("uji regresi sumbu [8]")
    for nama, L, harus in kasus:
        hasil = artefak_menggantung(L)
        ok = bool(hasil) == harus
        gagal += not ok
        print(f"  {'ok   ' if ok else 'GAGAL'} {nama}: {len(hasil)} temuan, "
              f"diharap {'ada' if harus else 'nol'}")

    # ── skema huruf: kode yang belum pernah dijalankan bukan kode yang bekerja ──
    HURUF = [
        "# BAB III. METODE PENELITIAN", "",
        "## A. Jenis dan Rancangan Penelitian", "", "teks", "",
        "## F. Prosedur Penelitian", "",
        "### 3. Skenario S3", "", "| a | b |", "|---|---|", "| 1 | 2 |", "",
        "Rinciannya ada pada Subbab III.A, dan matriks pada Subbab III.F.",
    ]
    print("\nuji skema huruf")
    periksa_huruf = [
        ("struktur mengenali subbab huruf",
         lambda: ("III.A", "III.F") == tuple(n for _, _, n, _ in struktur(HURUF)
                                             if re.fullmatch(r"III\.[A-Z]", n))),
        ("struktur menyusun anak subbab dari bab dan huruf induknya",
         lambda: "III.F.3" in {n for _, _, n, _ in struktur(HURUF)}),
        ("rujukan membaca 'Subbab III.F'",
         lambda: {"III.A", "III.F"} <= {r for _, r in rujukan(HURUF)}),
        ("rujukan huruf tidak menggantung terhadap heading huruf",
         lambda: not [r for _, r in rujukan(HURUF) if r not in nomor_subbab(HURUF)]),
        ("sumbu 8 menerima rujukan tabel yang subbabnya memuat tabel",
         lambda: artefak_menggantung(HURUF) == []),
        ("sumbu 8 menangkap rujukan tabel ke subbab huruf tanpa tabel",
         lambda: len(artefak_menggantung(HURUF[:6] + ["Lihat tabel keruntuhan peta pada Subbab III.A."])) == 1),
        ("daftar isi huruf dipulihkan menjadi 'I.A'",
         lambda: nomor_daftar_isi("BAB I  PENDAHULUAN\n  A. Latar Belakang\n  B. Rumusan") == {"I.A", "I.B"}),
        ("daftar isi desimal tetap terbaca",
         lambda: nomor_daftar_isi("  1.1 Latar Belakang\n  1.2 Rumusan") == {"1.1", "1.2"}),
    ]
    for nama, uji in periksa_huruf:
        try:
            ok = bool(uji())
        except Exception as e:
            ok = False
            nama = f"{nama} [{type(e).__name__}: {e}]"
        gagal += not ok
        print(f"  {'ok   ' if ok else 'GAGAL'} {nama}")

    print("\nLOLOS" if not gagal else f"\nGAGAL — {gagal} kasus")
    return 1 if gagal else 0


def main() -> int:
    if not BERKAS.exists():
        print(f"tidak ada: {BERKAS}"); return 1
    L = muat()
    galat, peringatan = [], []

    print("=" * 78)
    print(f"AUDIT SEMPRO — {BERKAS.name}, {len(L)} baris")
    print("=" * 78)

    # 1 rujukan silang subbab
    ada = nomor_subbab(L)
    gantung = [(i, r) for i, r in rujukan(L) if r not in ada]
    print(f"\n[1] Rujukan silang subbab — {len(rujukan(L))} rujukan, {len(ada)} subbab ada")
    if gantung:
        galat.append(f"{len(gantung)} rujukan subbab menggantung")
        for i, r in gantung:
            print(f"    GANTUNG baris {i:5d} -> Subbab {r}")
    else:
        print("    seluruhnya menunjuk subbab yang ada")

    # 2 rujukan lampiran
    ada_l, ruj_l = lampiran(L)
    g2 = [(i, r) for i, r in ruj_l if r not in ada_l]
    print(f"\n[2] Rujukan lampiran — ada {sorted(ada_l)}, {len(ruj_l)} rujukan")
    if g2:
        galat.append(f"{len(g2)} rujukan lampiran menggantung")
        for i, r in g2:
            print(f"    GANTUNG baris {i:5d} -> Lampiran {r}")
    else:
        print("    seluruhnya menunjuk lampiran yang ada")

    # 3 rujukan skenario
    ruj_s = [(i, m.group(1)) for i, l in enumerate(L, 1)
             for m in re.finditer(r"Skenario\s+(S\d+)", l)]
    g3 = [(i, r) for i, r in ruj_s if r not in SKENARIO]
    print(f"\n[3] Rujukan skenario — {len(ruj_s)} rujukan")
    if g3:
        galat.append(f"{len(g3)} rujukan skenario di luar S1-S8")
        for i, r in g3:
            print(f"    ASING baris {i:5d} -> {r}")
    else:
        print("    seluruhnya di dalam S1 sampai S8")

    # 4 penomoran subbab: kembar atau bolong
    # Diambil dari struktur() supaya skema huruf ikut terperiksa. Entri tingkat
    # BAB (romawi telanjang) dikeluarkan: ia bukan subbab.
    nomor = [n for _, _, n, _ in struktur(L) if "." in n]
    kembar = [k for k, v in Counter(nomor).items() if v > 1]
    print(f"\n[4] Penomoran subbab — {len(nomor)} bernomor")
    if kembar:
        galat.append(f"nomor subbab kembar: {kembar}")
        print(f"    KEMBAR: {kembar}")
    bolong = []
    for induk in sorted({".".join(n.split(".")[:-1]) for n in nomor if n.count(".") >= 1}):
        ekor = [n.split(".")[-1] for n in nomor
                if n.startswith(induk + ".") and n.count(".") == induk.count(".") + 1]
        if not ekor:
            continue
        # Deret bisa berupa angka (1,2,3), huruf besar (A,B,C), atau huruf kecil.
        # Ketiganya diubah ke ordinal supaya satu pemeriksaan melayani semuanya.
        if all(x.isdigit() for x in ekor):
            ke_ord, dari_ord = int, str
        elif all(len(x) == 1 and x.isupper() for x in ekor):
            ke_ord, dari_ord = lambda x: ord(x) - 64, lambda k: chr(k + 64)
        elif all(len(x) == 1 and x.islower() for x in ekor):
            ke_ord, dari_ord = lambda x: ord(x) - 96, lambda k: chr(k + 96)
        else:
            peringatan.append(f"deret penomoran campur di bawah {induk}: {sorted(set(ekor))}")
            continue
        anak = sorted({ke_ord(x) for x in ekor})
        hilang = [dari_ord(x) for x in range(1, max(anak) + 1) if x not in anak]
        if hilang:
            bolong.append(f"{induk}.{{{','.join(hilang)}}}")
    if bolong:
        peringatan.append(f"nomor bolong: {bolong}")
        print(f"    BOLONG: {bolong}")
    if not kembar and not bolong:
        print("    tidak ada yang kembar maupun bolong")

    # 5 daftar isi
    teks = "\n".join(L)
    m = re.search(r"# DAFTAR ISI\n+```(.*?)```", teks, re.S)
    print("\n[5] Daftar isi")
    if not m:
        peringatan.append("blok daftar isi tidak ditemukan")
        print("    blok daftar isi tidak ditemukan")
    else:
        di = nomor_daftar_isi(m.group(1))
        nyata = {n for n in nomor
                 if (n.count(".") == 1 and all(b.isdigit() for b in n.split(".")))
                 or re.fullmatch(ROMAWI + r"\.[A-Z]", n)}
        print(f"    daftar isi memuat {len(di)}, dokumen punya {len(nyata)} subbab tingkat dua")
        if di - nyata:
            galat.append(f"daftar isi menyebut subbab yang tidak ada: {sorted(di - nyata)}")
            print(f"    DI TAPI TIDAK ADA: {sorted(di - nyata)}")
        if nyata - di:
            peringatan.append(f"subbab tidak tercantum di daftar isi: {sorted(nyata - di)}")
            print(f"    ADA TAPI TIDAK DI DAFTAR ISI: {sorted(nyata - di)}")
        if not (di - nyata) and not (nyata - di):
            print("    cocok seluruhnya")

    # 6 tabel dan penanda tebal
    rusak = tabel_rusak(L)
    kosong = tabel_kosong(L)
    ganjil = tebal_ganjil(L)
    print(f"\n[6] Bentuk markdown")
    if rusak:
        galat.append(f"{len(rusak)} baris tabel kolomnya tidak konsisten")
        for i, ket in rusak[:8]:
            print(f"    TABEL baris {i:5d}: {ket}")
    else:
        print("    seluruh tabel kolomnya konsisten")
    if kosong:
        galat.append(f"{len(kosong)} tabel tanpa baris isi")
        for i in kosong:
            print(f"    TABEL KOSONG baris {i:5d}: kepala dan pemisah tanpa isi")
    else:
        print("    tidak ada tabel yang kehilangan baris isinya")
    if ganjil:
        galat.append(f"{len(ganjil)} baris penanda tebal ganjil")
        print(f"    TEBAL GANJIL di baris: {ganjil[:12]}")
    else:
        print("    penanda tebal seluruhnya berpasangan")

    # 7 sitasi, gaya APA nama-tahun
    #
    # Sumbu ini semula mencocokkan penanda IEEE `[n]`. Setelah naskah beralih ke APA pada
    # 9 September 2026, pola itu tidak pernah cocok lagi sehingga sumbunya melaporkan
    # "0 dipakai, 0 terdaftar" dan LOLOS secara hampa — pengawasan hilang tanpa satu pun
    # tanda. Pencocokan kini dilakukan atas nama belakang penulis pertama beserta tahunnya.
    i = teks.rfind("# DAFTAR PUSTAKA")
    # Wilayah daftar pustaka berakhir pada heading tingkat satu berikutnya. Tanpa batas itu
    # ia menelan seluruh LAMPIRAN, dan baris tabel lampiran terhitung sebagai entri pustaka.
    j = teks.find("\n# ", i + 1)
    badan, pustaka = teks[:i], teks[i:j if j > 0 else len(teks)]

    # Nama belakang boleh dua kata — 'San Luciano' pernah luput ketika polanya satu kata.
    # Tahun boleh berimbuhan huruf: APA menuntut '2016a' dan '2016b' bagi dua karya penulis
    # yang sama pada tahun yang sama. Pola yang menuntut tepat empat digit meleset pada
    # keduanya sekaligus — di teks maupun di daftar pustaka — sehingga cocok secara hampa.
    NAMA = r"[A-Z][\w'’-]+(?:\s+[A-Z][\w'’-]+)?"
    dipakai = {(m.group(1), m.group(2)) for m in
               re.finditer(r"[;(]\s*(" + NAMA + r")(?:\s+(?:dkk\.|&\s+" + NAMA + r"))?,\s*(\d{4}[a-z]?)\s*[;)]",
                           badan)}
    terdaftar = {(m.group(1), m.group(2)) for m in
                 re.finditer(r"^(" + NAMA + r"),[^(]*\((\d{4}[a-z]?)\)\.", pustaka, re.M)}
    yatim = sorted(dipakai - terdaftar)
    nganggur = sorted(terdaftar - dipakai)
    print(f"\n[7] Sitasi — {len(dipakai)} dipakai, {len(terdaftar)} terdaftar")
    if yatim:
        galat.append(f"sitasi yatim: {yatim}")
        print(f"    YATIM: {yatim}")
    if nganggur:
        peringatan.append(f"terdaftar tanpa dipakai: {nganggur}")
        print(f"    TIDAK DIPAKAI: {nganggur}")
    if not yatim and not nganggur:
        print("    seluruhnya sinkron")

    # 8 artefak yang dirujuk di dalam subbab
    print("\n[8] Artefak dalam subbab — rujukan \"tabel ... pada Subbab N\"")
    dokumen = [("sempro", L)]
    if SEMHAS.exists():
        dokumen.append(("semhas", SEMHAS.read_text().split("\n")))
    else:
        peringatan.append("semhas tidak ada; sumbu 8 hanya diperiksa pada sempro")
        print("    semhas  LEWAT (berkas tidak ada)")
    for label, baris in dokumen:
        g8 = artefak_menggantung(baris)
        if g8:
            galat.append(f"{len(g8)} rujukan artefak menggantung pada {label}")
            for i, kutip, sebab in g8:
                print(f"    GANTUNG {label} baris {i:5d}: {kutip!r}")
                print(f"             {sebab}")
        else:
            print(f"    {label:7s} tiap rujukan tabel mendarat di subbab yang memuat tabel")

    print("\n" + "=" * 78)
    if galat:
        print(f"GAGAL — {len(galat)} galat:")
        for g in galat:
            print(f"  - {g}")
    if peringatan:
        print(f"{len(peringatan)} peringatan:")
        for w in peringatan:
            print(f"  - {w}")
    if not galat:
        print("LOLOS — tidak ada galat struktural.")
    return 1 if galat else 0


if __name__ == "__main__":
    sys.exit(uji_sendiri() if "--uji" in sys.argv else main())
