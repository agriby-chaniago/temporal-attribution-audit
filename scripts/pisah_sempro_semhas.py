"""Pisahkan naskah tunggal menjadi dua dokumen: proposal (sempro) dan hasil (semhas).

Jalankan: python3 scripts/pisah_sempro_semhas.py

Latar
-----
Naskah tumbuh sebagai satu berkas selama penelitian berjalan, sehingga bagian
hasil ikut menumpuk di dalam Bab III. Per hitungan terakhir, sekitar 35 persen
Bab III sebenarnya hasil yang berpakaian metodologi. Sempro dan semhas adalah
dua dokumen berbeda dengan pembaca dan pertanyaan berbeda, sehingga dipisahkan.

Pembagian
---------
**Sempro** memuat Bab I sampai III tanpa satu pun angka hasil, ditambah jadwal
dan lampiran lubang rancangan. Dokumen ini berfungsi ganda sebagai **catatan
pra-registrasi**: ramalan, ambang, dan aturan keputusan seluruhnya ada di sini,
dan tidak satu pun hasil ada di sini.

**Semhas** memuat Bab I sampai III yang sama, ditambah Bab IV berisi seluruh
hasil yang diangkat dari Bab III, dan Bab V kesimpulan.

Sumber kebenaran tetap `NASKAH-SUMBER.md` di akar proyek. Kedua keluaran
dibangun ulang dari sana, sehingga koreksi cukup dilakukan di satu tempat.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

AKAR = Path(__file__).resolve().parent.parent
SUMBER = AKAR / "NASKAH-SUMBER.md"
OUT_SEMPRO = AKAR / "draftSempro" / "sempro-skripsi.md"
OUT_SEMHAS = AKAR / "draftSemhas" / "semhas-skripsi.md"

# Heading yang menandai blok HASIL di dalam Bab III. Blok berakhir tepat sebelum
# heading berikutnya yang levelnya sama atau lebih tinggi.
HASIL = [
    "### Hasil Skenario S4",
    "### Penanda lambat diterapkan pada DST, dan satu klaim daya uji yang ditarik",
    "### Audit perancu pada STCP",
    "### Hasil pengukuran",
    "### Hasil Skenario S8",
    "### Hasil Skenario S5, dan ramalan di atas yang DIBANTAH",
    "### Hasil sepuluh seed: cabang pertama, dan ramalan di atas yang meleset",
    "### Hasil penerapan aturan",
    "### Arm ketiga: BiMamba-3",
    "### Sel yang terisi",
    "### Hasil baseline fitur agregat",
    "### Hasil RM5 pada NewHandPD",
    "### Hasil tuning berimbang",
    "### Hasil reproduktibilitas peta",
    "### Hasil sumbu geser tugas",
    "### Hasil sapuan biaya",
    "### Hasil audit perancu usia",
]

# Judul bagi tiap blok ketika dipindahkan ke Bab IV.
JUDUL_BAB4 = {
    "### Hasil Skenario S4": "### 2. Generalisasi lintas tugas (S4)",
    "### Penanda lambat diterapkan pada DST, dan satu klaim daya uji yang ditarik":
        "### 7. Penanda lambat pada DST, dan satu klaim daya uji yang ditarik",
    "### Audit perancu pada STCP": "### 8. Audit perancu pada STCP",
    "### Hasil pengukuran": "### 5. Kesetiaan atensi terhadap atribusi Shapley (S6)",
    "### Hasil Skenario S8": "### 3. Uji kewarasan pendukung (S8)",
    "### Hasil Skenario S5, dan ramalan di atas yang DIBANTAH":
        "### 4. Replikasi pada NewHandPD (S5), dan ramalan yang dibantah",
    "### Hasil sepuluh seed: cabang pertama, dan ramalan di atas yang meleset":
        "### 9. Retensi lintas kohort pada sepuluh seed",
    "### Hasil penerapan aturan": "### 1. Performa klasifikasi dan penerapan aturan keputusan (S3)",
    "### Arm ketiga: BiMamba-3": "#### a. Arm ketiga: BiMamba-3",
    "### Sel yang terisi": "### 6. Keselarasan terhadap penanda motorik (S7)",
    "### Hasil baseline fitur agregat": "### 10. Baseline fitur kinematik agregat",
    "### Hasil RM5 pada NewHandPD": "### 11. Rumusan Masalah 5 pada 35 subjek kontrol",
    "### Hasil tuning berimbang": "### 12. Tuning hyperparameter berimbang (RM2)",
    "### Hasil reproduktibilitas peta": "### 13. Reproduktibilitas peta atribusi antar seed (RM6)",
    "### Hasil sumbu geser tugas": "### 14. Sumbu geser tugas di dalam kohort",
    "### Hasil sapuan biaya": "### 15. Biaya pelatihan terhadap panjang urutan (RM2)",
    "### Hasil audit perancu usia": "### 16. Audit perancu usia pada kohort lintas",
}

URUT_BAB4 = [
    "### Hasil penerapan aturan",
    "### Arm ketiga: BiMamba-3",
    "### Hasil Skenario S4",
    "### Hasil Skenario S8",
    "### Hasil Skenario S5, dan ramalan di atas yang DIBANTAH",
    "### Hasil pengukuran",
    "### Sel yang terisi",
    "### Penanda lambat diterapkan pada DST, dan satu klaim daya uji yang ditarik",
    "### Audit perancu pada STCP",
    "### Hasil sepuluh seed: cabang pertama, dan ramalan di atas yang meleset",
    "### Hasil baseline fitur agregat",
    "### Hasil RM5 pada NewHandPD",
    "### Hasil tuning berimbang",
    "### Hasil reproduktibilitas peta",
    "### Hasil sumbu geser tugas",
    "### Hasil sapuan biaya",
    "### Hasil audit perancu usia",
]


# ── Prosa penghubung antar subbab Bab IV ──────────────────────────────────────
# Blok hasil dipindahkan dari Bab III, sehingga sebagian membuka dengan rujukan
# ke konteks yang kini berada di bab lain — "paragraf sebelumnya", "tabel di
# atas", "verifikasi di atas". Paragraf penghubung ini memulihkan konteks
# sekaligus menyatakan pertanyaan yang dijawab tiap subbab, agar Bab IV terbaca
# sebagai satu alur dan bukan kumpulan blok.
TRANSISI = {
    "### Hasil reproduktibilitas peta": """Dua subbab sebelumnya memperlihatkan peta runtuh ketika kohort, seed, dan resolusi dipertukarkan.
Subbab ini menjawab **Rumusan Masalah 6** dan memisahkan dua kemungkinan yang selama ini bercampur:
apakah keruntuhan itu **variansi** yang dapat diredam dengan mengulang pelatihan, atau **bias** yang
tidak akan hilang berapa kali pun diulang. Aturan agregasi dan uji kurvanya dibekukan pada Subbab
III.G.9 sebelum satu angka pun dihitung.""",

    "### Hasil sumbu geser tugas": """Subbab IV.A.9 memperlihatkan peringkat antar arsitektur membalik ketika kohort berpindah. Perpindahan
kohort mengubah perangkat, negara, subjek, dan jenis tugas sekaligus, sehingga pembalikan itu tidak
dapat diatribusikan kepada satu pun di antaranya. Subbab ini memisahkan satu sumbu dari sisanya
dengan memakai 396 rekaman kohort lintas yang belum pernah dipakai: jenis tugas berganti, sementara
kohort, perangkat, dan subjek dikunci konstan.""",

    "### Hasil sapuan biaya": """Subbab II.A.3.b butir 3 menuliskan paralelisasi saat pelatihan sebagai keunggulan yang dapat diuji, dan
butir itu belum pernah diuji sampai titik ini. Subbab ini mengujinya pada sapuan yang membentang 27
sampai 1.560 token, beserta satu sumbu biaya kedua yang arahnya ternyata berlawanan.""",

    "### Hasil audit perancu usia": """Metadata subjek pada kohort lintas terlewat sepanjang penelitian sebab pemuat data membuang baris
header bersama seluruh isinya. Ketika akhirnya dibaca, satu kovariat memisahkan kedua kelompok
hampir sebaik model itu sendiri. Subbab ini memeriksa apakah model membacanya, memakai bentuk
pemisahan yang sama dengan audit perancu pada Subbab IV.A.8.""",

    "### Hasil penerapan aturan": """Subbab ini menjawab **Rumusan Masalah 2 pada sumbu akurasi**: apakah ketiga encoder berbeda
performa klasifikasinya ketika seluruh komponen lain dikunci identik. Aturan keputusan yang dipakai
ditetapkan pada Subbab III.G.1 sebelum data dilihat, dan dihitung hanya dari dua arm pra-registrasi.""",

    "### Arm ketiga: BiMamba-3": """Arm ketiga dilaporkan terpisah, sebab statusnya berbeda dan pemisahan itu ditegakkan pada tingkat
kode.""",

    "### Hasil Skenario S4": """Setelah performa dalam-tugas diketahui, pertanyaan berikutnya: **apakah yang dipelajari model
bertahan ketika tugasnya berganti?** Subbab ini menguji langsung lewat matriks tugas terhadap tugas,
dan hasilnya menjadi dasar ramalan pra-registrasi bagi replikasi lintas kohort pada Subbab IV.A.4.""",

    "### Hasil Skenario S8": """Sebelum hasil mana pun ditafsirkan, pipeline-nya sendiri perlu dibuktikan tidak bocor. Subbab ini
memuat dua kontrol yang, bila gagal, akan membatalkan seluruh subbab lain.""",

    "### Hasil Skenario S5, dan ramalan di atas yang DIBANTAH": """Subbab ini menjawab **Rumusan Masalah 3**: apakah metodenya dapat direplikasi pada basis data yang
berbeda perangkat, negara, tugas, dan arah komposisi kelas. Ramalan pra-registrasinya dicatat pada
Subbab II.D sebelum satu pun angka di bawah terlihat, dengan syarat pembantah yang ditulis lebih
dahulu.""",

    "### Hasil pengukuran": """Empat subbab sebelumnya membahas **prediksi**. Mulai di sini, yang diuji adalah **petanya**. Subbab
ini menjawab **Rumusan Masalah 4**: seberapa setia bobot atensi terhadap kontribusi yang sebenarnya
menggerakkan keluaran, diukur memakai atribusi Shapley pada grid temporal yang sama.""",

    "### Sel yang terisi": """Subbab ini menjawab **Rumusan Masalah 5**, dan memuat **satu-satunya besaran konfirmatori** pada
seluruh penelitian. Matriks empat kemungkinan hasil beserta kalibrasi kata "tinggi" dan "rendah"
ditetapkan pada Subbab III.G.4 sebelum data pasien disentuh; di bawah ini dilaporkan sel mana yang
akhirnya terisi.""",

    "### Penanda lambat diterapkan pada DST, dan satu klaim daya uji yang ditarik": """Analisis primer di atas memakai penanda **cepat** pada tugas STCP. Penanda **lambat** sudah
diverifikasi lewat empat uji pada Subbab III.F.5 namun belum dipakai satu analisis pun, padahal tugas
DST memiliki dua kali lipat subjek kontrol. Subbab ini memakainya, dan hitungan dayanya justru
memaksa satu klaim ditarik.""",

    "### Audit perancu pada STCP": """Dua subbab sebelumnya memperlihatkan analisis primer tidak melampaui ambangnya. Pertanyaan yang
wajar berikutnya: **mengapa?** Subbab ini mengaudit tugas STCP untuk mencari sebabnya, dan
jawabannya bukan sebab tunggal melainkan batas identifikasi.""",

    "### Hasil sepuluh seed: cabang pertama, dan ramalan di atas yang meleset": """Subbab IV.A.4 memperlihatkan replikasi berhasil, dan mencatat satu pola yang belum diuji secara formal:
BiGRU kehilangan jauh lebih banyak daripada kedua arm Mamba ketika berpindah kohort. Uji bagi pola
itu **dinamai pada Subbab III.G.5 sebelum seed diperbanyak**, beserta empat cabang hasil dan
peringatan bahwa hasil paling mungkin adalah mundur ke arah nol. Subbab ini melaporkan cabang mana
yang berlaku.""",

    "### Hasil baseline fitur agregat": """Seluruh perbandingan di atas berlangsung antar arsitektur sekuens. Subbab ini menjawab keberatan
yang paling wajar terhadap keseluruhannya: **apakah pemodelan sekuens memang diperlukan**, atau
statistik agregat pada kanal yang sama sudah cukup.""",

    "### Hasil RM5 pada NewHandPD": """Analisis primer Rumusan Masalah 5 bersandar pada tujuh subjek kontrol, dan Subbab IV.A.7 memperlihatkan
besaran efek minimum terdeteksi berada **di atas** ambangnya sendiri, sehingga hasil apa pun di sana
tidak dapat ditafsirkan sebagai penolakan hipotesis. Basis data lintas memiliki tiga puluh lima
kontrol. Subbab ini mengajukan pertanyaan yang sama di sana, dengan daya yang memadai.""",

    "### Hasil tuning berimbang": """Seluruh perbandingan arsitektur di atas berjalan pada satu konfigurasi tunggal, dan keberatan bahwa
konfigurasi itu mungkin tidak sama cocoknya bagi ketiga arm sudah dinyatakan sebagai batasan pada
Subbab III.G.8. Subbab ini mengujinya alih-alih membiarkannya sebagai catatan kaki.""",
}


# ── Penanda blok "hanya-hasil" ────────────────────────────────────────────────
# Blok yang diapit penanda ini memuat hasil eksperimen dan **dibuang seluruhnya
# dari sempro**, sementara pada semhas ia dipertahankan di tempatnya (hanya baris
# penandanya yang dibuang). Mekanisme ini dipakai untuk blok hasil yang tersebar
# di dalam Bab II dan III, yang tidak dapat dipindahkan ke Bab IV tanpa merusak
# alur argumennya.
AWAL = "<!-- HASIL-ONLY -->"
AKHIR = "<!-- /HASIL-ONLY -->"


def buang_blok_hasil(t: str) -> str:
    """Buang seluruh blok berpenanda. Dipakai untuk sempro."""
    # Kedalaman, bukan boolean: blok berpenanda dapat bersarang, dan penutup
    # bersarang tidak boleh menutup blok terluar terlalu dini.
    keluar, dalam, n = [], 0, 0
    for b in t.split("\n"):
        s = b.strip()
        if s == AWAL:
            if dalam == 0:
                n += 1
            dalam += 1; continue
        if s == AKHIR:
            dalam -= 1
            if dalam < 0:
                raise SystemExit("penanda /HASIL-ONLY tanpa pembuka")
            continue
        if dalam == 0:
            keluar.append(b)
    if dalam:
        raise SystemExit("penanda HASIL-ONLY tidak berpasangan")
    for b_ in t.split("\n"):
        if (AWAL in b_ or AKHIR in b_) and b_.strip() not in (AWAL, AKHIR):
            raise SystemExit(f"penanda HASIL-ONLY tersisip di tengah baris:\n  {b_[:90]}")
    buang_blok_hasil.n = n
    return "\n".join(keluar)


def lepas_penanda(t: str) -> str:
    """Buang baris penandanya saja, isinya tetap. Dipakai untuk semhas."""
    return "\n".join(b for b in t.split("\n")
                     if b.strip() not in (AWAL, AKHIR))


def tingkat(baris: str) -> int:
    m = re.match(r"^(#+)\s", baris)
    return len(m.group(1)) if m else 99


def cari_blok(L: list[str]) -> dict[str, tuple[int, int]]:
    """Petakan tiap heading hasil ke rentang barisnya [awal, akhir)."""
    idx = {}
    for i, b in enumerate(L):
        for h in HASIL:
            if b.strip() == h:
                idx[h] = i
    blok = {}
    for h, a in idx.items():
        lv = tingkat(h)
        akhir = len(L)
        for j in range(a + 1, len(L)):
            if tingkat(L[j]) <= lv:
                akhir = j
                break
        blok[h] = (a, akhir)
    hilang = [h for h in HASIL if h not in blok]
    if hilang:
        raise SystemExit("heading hasil tidak ditemukan:\n  " + "\n  ".join(hilang))
    return blok


def buang_banner(t: str) -> str:
    """Banner peringatan berkas sumber tidak ikut ke dokumen yang diserahkan."""
    return re.sub(r"<!-- SUMBER-BANNER-AWAL -->.*?<!-- SUMBER-BANNER-AKHIR -->\n+",
                  "", t, flags=re.S)


def main() -> int:
    L = buang_banner(SUMBER.read_text()).split("\n")
    blok = cari_blok(L)

    # ---- baris mana yang milik hasil ----
    milik_hasil = set()
    for a, b in blok.values():
        milik_hasil.update(range(a, b))

    # ---- SEMPRO: seluruh naskah minus blok hasil ----
    sempro = [b for i, b in enumerate(L) if i not in milik_hasil]
    teks_sempro = "\n".join(sempro)

    # ---- SEMHAS: kerangka + Bab IV dari blok hasil ----
    kerangka = list(sempro)
    bab4 = []
    for h in URUT_BAB4:
        a, b = blok[h]
        isi = L[a:b]
        isi[0] = JUDUL_BAB4[h]
        if h in TRANSISI:
            isi.insert(1, "\n" + TRANSISI[h] + "\n")
        # turunkan satu tingkat heading anak agar konsisten di bawah Bab IV
        for k in range(1, len(isi)):
            if isi[k].startswith("####"):
                isi[k] = "###" + isi[k][4:]
        bab4 += isi + [""]

    teks_semhas = "\n".join(kerangka)
    teks_semhas = sisip_bab4(teks_semhas, "\n".join(bab4))

    teks_sempro = buang_blok_hasil(teks_sempro)
    teks_semhas = lepas_penanda(teks_semhas)
    tulis(OUT_SEMPRO, sempro_kepala(teks_sempro))
    tulis(OUT_SEMHAS, semhas_kepala(teks_semhas))

    print(f"sumber   : {len(L):5d} baris")
    print(f"sempro   : {OUT_SEMPRO.relative_to(AKAR)}  "
          f"{len(OUT_SEMPRO.read_text().split(chr(10))):5d} baris")
    print(f"semhas   : {OUT_SEMHAS.relative_to(AKAR)}  "
          f"{len(OUT_SEMHAS.read_text().split(chr(10))):5d} baris")
    print(f"dipindah : {len(milik_hasil)} baris hasil dari Bab III ke Bab IV")
    lewat = getattr(terapkan_ganti, "lewat", [])
    if lewat:
        print(f"dilewati : {len(lewat)} entri SEMPRO_GANTI tidak menemukan jangkarnya.")
        print( "           Biasanya sah: paragrafnya sudah terbuang lebih dahulu, baik oleh")
        print( "           penanda HASIL-ONLY maupun karena ia berada di dalam blok '### Hasil'")
        print( "           yang dipindah utuh ke Bab IV. Tetapi entri juga lewat ketika teks")
        print( "           jangkarnya berubah karena penyuntingan, dan pada kasus itu paragraf")
        print( "           hasil lolos ke sempro tanpa peringatan. Bila baru menyunting salah")
        print( "           satu paragraf di bawah, periksa keluarannya di sempro:")
        for x in lewat:
            print(f"           - {x}")
    print(f"dibuang  : {getattr(buang_blok_hasil, 'n', 0)} blok berpenanda HASIL-ONLY dari sempro")

    # Nomor tabel dihitung PER DOKUMEN, sebab keduanya memuat himpunan tabel yang berbeda:
    # tabel di dalam blok HASIL-ONLY hilang dari sempro namun tetap di semhas. Dipanggil di
    # sini supaya nomor tidak pernah tertinggal setiap kali dokumen dibangun ulang.
    from nomori_tabel import nomori
    nomori()

    # Daftar tabel, lampiran, dan singkatan dibangkitkan dari dokumen yang sudah jadi,
    # sesudah tabel dinomori — sebab isinya bergantung dokumen, sama seperti nomornya.
    from bangun_daftar_pelengkap import main as pelengkap
    pelengkap()

    # Daftar isi paling akhir: ia menyebut halaman-halaman di atas, termasuk yang baru
    # saja dibangkitkan.
    from bangun_daftar_isi import main as daftar_isi
    daftar_isi()
    return 0


def sisip_bab4(teks: str, bab4: str) -> str:
    """Tempatkan blok hasil yang dipindah dari Bab III ke penampung tetapnya.

    Kerangka Bab IV dan V (prosa statis, IV.B Keterbatasan, seluruh Bab V) hidup
    permanen di `NASKAH-SUMBER.md` sendiri, dibungkus `<!-- HASIL-ONLY -->` sehingga
    otomatis terbuang dari sempro oleh mekanisme yang sama dipakai 30 blok hasil
    lain — tidak ada logika baru yang perlu dijaga bagi sempro. Yang benar-benar
    dinamis hanyalah isi Subbab IV.A.1–16 itu sendiri, sebab teksnya berasal dari
    blok `### Hasil ...` yang diangkat dari Bab III; satu-satunya titik sambung
    yang fungsi ini masih tangani adalah penampung `<!-- BAB4-ISI -->` itu.
    """
    penanda = "<!-- BAB4-ISI -->"
    if penanda not in teks:
        raise SystemExit(f"penampung {penanda} tidak ditemukan; periksa NASKAH-SUMBER.md")
    return teks.replace(penanda, bab4, 1)

# Paragraf yang melaporkan hasil eksperimen dan karena itu tidak boleh muncul di
# proposal. Diganti dengan rumusan bertense rancangan. Kunci dipakai sebagai
# awalan paragraf; pencocokannya wajib tepat satu, jika tidak skrip berhenti.
SEMPRO_GANTI = [
    ("**Hitungan daya yang dahulu tidak dilakukan, kini tersedia",
     "**Hitungan daya belum dilakukan, dan itu diakui sebagai kelemahan rancangan.** Besaran efek "
     "minimum terdeteksi pada jumlah kontrol yang tersedia akan dihitung dan dilaporkan bersama "
     "hasil, sehingga batas kemampuan rancangan ini terbaca sebagai angka alih-alih sebagai "
     "kualifikasi verbal. Hitungan itu tidak akan dipakai membatalkan hasil apa pun secara surut."),
    ("**Catatan status keseluruhan.**",
     "**Catatan status keseluruhan.** Skenario S1 sampai S8 dirancang sebagaimana Subbab III.F, "
     "dijalankan pada tiga arm: dua arsitektur pra-registrasi ditambah BiMamba-3 yang berstatus "
     "eksploratori. Aturan keputusan Subbab III.G.1 dihitung **hanya** dari dua arm pra-registrasi, dan "
     "penjagaan itu ditegakkan pada tingkat kode sehingga penambahan arm ketiga tidak dapat "
     "menggeser vonis konfirmatori.\n\n"
     "Hasil seluruh skenario dilaporkan pada naskah seminar hasil, terpisah dari dokumen ini. "
     "Pemisahan itu disengaja: ramalan, ambang, dan aturan keputusan yang tertulis di sini "
     "ditetapkan sebelum hasil dilihat, dan dokumen ini menjadi catatannya.\n\n"
     "Hal-hal ini dinyatakan agar ketidaklengkapan penelitian terlihat pada dokumen, bukan hanya "
     "diketahui penyusunnya."),
    ("**Klaim itu berlaku bagi penandanya, bukan bagi modelnya.**",
     "**Klaim itu berlaku bagi penandanya, dan belum tentu bagi modelnya.** STCP memuat sedikitnya "
     "empat besaran tingkat subjek yang tidak dimiliki SST maupun DST dan berpotensi menjadi "
     "perancu: perbedaan durasi rekaman, perbedaan kepatuhan menyentuh, jumlah episode melayang, "
     "dan dukungan penanda yang berbeda antar kelompok karena penanda hanya terdefinisi pada "
     "segmen melayang. Keempatnya akan diaudit dan dilaporkan bersama hasil, sebab tugas yang "
     "paling bersih bagi penanda belum tentu paling bersih bagi model. Sisa baris bertekanan "
     "(sentuhan sesaat) ditangani pada Subbab III.F.5."),
    ("Skenario S4 sudah dijalankan (Subbab III.F.4,",
     "Skenario S4 dirancang untuk menjawab pertanyaan ini secara langsung (Subbab III.F.4, "
     "`notebooks/11_lintas_tugas.ipynb`): matriks tugas terhadap tugas memperlihatkan apakah "
     "model buta tugas atau tidak, dan hasilnya dilaporkan pada naskah seminar hasil. Bila "
     "menggambar dan menahan pena merupakan dua masalah berbeda, konsekuensinya menyentuh "
     "penggabungan tugas pada seluruh skenario."),
    ("| Data hanya 77 subjek, cukupkah? |",
     "| Data hanya 77 subjek, cukupkah? | Ukuran efek terkecil yang dapat dideteksi diukur melalui "
     "kontrol positif pada Subbab III.F.3 dan hitungan daya, bukan diasumsikan. Keduanya dilaporkan "
     "bersama hasil dan membatasi bahasa klaim |"),
]


def terapkan_ganti(t: str) -> str:
    terapkan_ganti.lewat = []
    baris = t.split("\n")
    for awalan, ganti in SEMPRO_GANTI:
        cocok = [i for i, b in enumerate(baris) if b.startswith(awalan)]
        # Nol cocokan berarti paragrafnya sudah terbuang mekanisme penanda blok,
        # yang menggantikan sebagian entri lama; itu sah. Lebih dari satu berarti
        # penggantiannya ambigu dan harus dihentikan.
        if len(cocok) > 1:
            raise SystemExit(f"paragraf sempro cocok {len(cocok)} kali, ambigu: {awalan[:60]}")
        if not cocok:
            # Entri yang tidak menemukan jangkarnya biasanya berarti paragrafnya
            # sudah tercakup penanda blok. Tetapi ia juga terjadi ketika teks
            # jangkarnya berubah karena penyuntingan — dan pada kasus itu paragraf
            # hasil akan lolos ke sempro tanpa peringatan. Karena itu entri yang
            # lewat dilaporkan menonjol, bukan disebut sambil lalu.
            terapkan_ganti.lewat.append(awalan[:52])
            continue
        baris[cocok[0]] = ganti
    t = "\n".join(baris)
    # buang sisa paragraf status yang kini tergantikan
    for sisa in [
        "Rumusan Masalah 3 kini memiliki jawaban, dan jawabannya positif:",
        "Seluruh skenario kini dijalankan pada **tiga arm**:",
        "Skenario S8 sudah dijalankan dan hasilnya dilaporkan",
        "Hal-hal ini dinyatakan di sini agar ketidaklengkapan",
    ]:
        t = "\n".join(b for b in t.split("\n") if not b.startswith(sisa))
    return t


def sempro_kepala(t: str) -> str:
    t = terapkan_ganti(t)
    catatan = f"""
> **Catatan mengenai dokumen ini.** Dokumen ini memuat rancangan penelitian dan **tidak memuat satu
> pun angka hasil**. Seluruh ramalan, ambang, dan aturan keputusan yang tertulis di sini ditetapkan
> sebelum data hasil dilihat, sehingga dokumen ini sekaligus berfungsi sebagai **catatan
> pra-registrasi**. Hasilnya dilaporkan terpisah pada naskah seminar hasil.
> Dibangun otomatis dari `NASKAH-SUMBER.md` oleh `scripts/pisah_sempro_semhas.py`
> pada {date.today().isoformat()}.

"""
    return t.replace("\n# LEMBAR PERSETUJUAN", catatan + "\n# LEMBAR PERSETUJUAN", 1)


def semhas_kepala(t: str) -> str:
    t = t.replace("# PROPOSAL SKRIPSI", "# SKRIPSI", 1)
    t = t.replace("Proposal skripsi dengan judul", "Skripsi dengan judul", 1)
    # Halaman depan mengikuti Lampiran 3 dan 7; padanan naskah TA-nya Lampiran 4 dan 8.
    t = t.replace("**PROPOSAL TUGAS AKHIR**", "**TUGAS AKHIR**")
    t = t.replace("## PROPOSAL TUGAS AKHIR", "## TUGAS AKHIR")
    t = t.replace("seminar proposal Tugas Akhir pada Program Studi",
                  "seminar hasil Tugas Akhir pada Program Studi")
    t = t.replace("dinyatakan layak untuk dilakukan penelitian", "dinyatakan memenuhi syarat")

    # Judul semhas menjawab pertanyaan yang diajukan judul sempro. Naskah sumber
    # menyimpan versi pertanyaan sebab ia juga berfungsi sebagai catatan
    # pra-registrasi; pertukaran ke versi pernyataan hanya berlaku bagi semhas,
    # dan hanya sah setelah jawabannya diperoleh. Subjudulnya tidak berubah.
    for cari, ganti in [
        ('"BENARKAH YANG BENAR SELALU MENUNJUK YANG BENAR?"',
         '"BAHKAN YANG BENAR TIDAK SELALU MENUNJUK YANG BENAR"'),
        ('"Benarkah yang Benar Selalu Menunjuk yang Benar?"',
         '"Bahkan yang Benar Tidak Selalu Menunjuk yang Benar"'),
        ('"The Model Is Right, but Is It Always Pointing to the Right Moment?"',
         '"Even When the Model Is Right, It Does Not Always Point to the Right Moment"'),
    ]:
        if cari not in t:
            raise SystemExit(f"judul sempro tidak ditemukan untuk ditukar: {cari[:50]}")
        t = t.replace(cari, ganti)
    t = t.replace("telah disetujui untuk diseminarkan.",
                  "telah disetujui untuk diseminarkan pada Seminar Hasil.", 1)
    # Bab IV dan V dahulu ditambal manual ke sini lewat variabel `daftar_isi` di atas,
    # sebab `bangun_daftar_isi.py` belum menjangkau heading Bab IV/V. Tambalan itu sekarang
    # bukan cuma percuma — `bangun_daftar_isi.main()` di penghujung pipeline sudah membaca
    # SELURUH heading dokumen jadi (termasuk Bab IV, IV.B, dan Bab V) dan menulis ulang
    # blok fenced-nya secara utuh — tetapi aktif merusak: `t.replace("DAFTAR PUSTAKA", ...)`
    # tanpa tanda "#" mencocok heading asli "# DAFTAR PUSTAKA" di penghujung dokumen (satu-
    # satunya kemunculan literal string itu pada titik ini dalam pipeline), memakan tanda "#"-
    # nya dan menyisakan "DAFTAR PUSTAKA" sebagai teks polos, bukan heading — bab Daftar
    # Pustaka pada semhas kehilangan headingnya sama sekali. Baris itu sudah dihapus di sini.
    catatan = f"""
> **Catatan mengenai dokumen ini.** Bab I sampai III identik dengan naskah proposal yang sudah
> diseminarkan, sehingga ramalan dan aturan keputusan di dalamnya dapat dibaca sebagai catatan
> pra-registrasi. Bab IV memuat seluruh hasil.
> Dibangun otomatis dari `NASKAH-SUMBER.md` oleh `scripts/pisah_sempro_semhas.py`
> pada {date.today().isoformat()}.

"""
    return t.replace("\n# LEMBAR PERSETUJUAN", catatan + "\n# LEMBAR PERSETUJUAN", 1)


def tulis(p: Path, t: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(t)


if __name__ == "__main__":
    raise SystemExit(main())
