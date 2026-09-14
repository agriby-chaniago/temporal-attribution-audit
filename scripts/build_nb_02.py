"""Bangun notebooks/02_prapemrosesan.ipynb secara terprogram.

Notebook dibangkitkan dari skrip agar isinya dapat direproduksi dan diperiksa
lewat diff, bukan diedit manual di dalam JSON notebook.
Jalankan: python3 scripts/build_nb_02.py
"""

from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda t: cells.append(nbf.v4.new_markdown_cell(t))
code = lambda t: cells.append(nbf.v4.new_code_cell(t))

md("""# Tahap 2 — Prapemrosesan dan Rekayasa Kanal

Verifikasi modul [`src/preprocessing.py`](../src/preprocessing.py), mengikuti Subbab 3.4 naskah proposal.

Notebook ini menemukan **tiga hal yang mengoreksi hasil Tahap 1** dan menentukan rancangan seluruh
pipeline. Ketiganya baru terlihat setelah data mentah diperiksa pada tingkat timestamp, bukan pada
tingkat ringkasan.""")

code("""import sys
sys.path.insert(0, "../src")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from preprocessing import (
    muat_berkas, pecah_segmen, muat_dataset, Normalisasi,
    simpan_cache, muat_cache, KANAL_MODEL, NAMA_TUGAS, FS_ASLI, FS_TARGET, DURASI_MIN_DETIK,
)

AKAR = Path("../data/raw/uci395/extracted")
berkas = (sorted((AKAR / "hw_dataset/control").glob("*.txt"))
          + sorted((AKAR / "hw_dataset/parkinson").glob("*.txt"))
          + sorted((AKAR / "new_dataset/parkinson").glob("*.txt")))
print(f"{len(berkas)} berkas subjek")""")

md("""## 1. Koreksi frekuensi sampling

Tahap 1 melaporkan 142,86 Hz, dihitung dari **median** selisih timestamp. Angka itu keliru.
Distribusi selisih timestamp bersifat bimodal, sehingga median jatuh tepat pada salah satu puncak
dan mengabaikan puncak lainnya.""")

code("""dt_semua = []
for f in berkas:
    a = muat_berkas(f)
    for tid in np.unique(a[:, 6]):
        d = np.diff(a[a[:, 6] == tid][:, 5])
        dt_semua.extend(d[(d > 0) & (d < 100)].tolist())
dt_semua = np.array(dt_semua)

nilai, jumlah = np.unique(dt_semua, return_counts=True)
urut = np.argsort(-jumlah)[:5]
print("distribusi selisih timestamp:")
for i in urut:
    print(f"  {nilai[i]:4.0f} ms -> {jumlah[i]:7,} ({jumlah[i]/len(dt_semua)*100:5.2f}%)")

print()
print(f"median  = {np.median(dt_semua):.3f} ms -> {1000/np.median(dt_semua):7.2f} Hz   (dilaporkan Tahap 1, KELIRU)")
print(f"rata2   = {dt_semua.mean():.3f} ms -> {1000/dt_semua.mean():7.2f} Hz   (benar, dipakai sebagai FS_ASLI)")""")

code("""fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(dt_semua, bins=np.arange(5.5, 12.5, 1), edgecolor="white")
ax.axvline(np.median(dt_semua), color="tab:red", ls="--", label=f"median {np.median(dt_semua):.1f} ms (keliru)")
ax.axvline(dt_semua.mean(), color="tab:green", ls="-", label=f"rata-rata {dt_semua.mean():.2f} ms (benar)")
ax.set_xlabel("selisih timestamp (ms)"); ax.set_ylabel("jumlah"); ax.legend()
ax.set_title("Distribusi bimodal membuat median menyesatkan")
plt.tight_layout(); plt.show()""")

md("""Konsekuensinya bukan sekadar angka di naskah. Selisih timestamp yang **tidak seragam** berarti
turunan (kecepatan, percepatan, jerk) tidak boleh dihitung langsung dari data mentah: hasilnya akan
mengandung derau yang berasal dari jitter pencatatan waktu, bukan dari gerak tangan. Resampling ke
grid seragam menjadi wajib, bukan opsional.""")

md("""## 2. Lompatan timestamp mundur

Empat belas dari 202 rekaman memuat timestamp yang melompat mundur. Pemeriksaan konteksnya
menunjukkan ini bukan kerusakan data acak, melainkan batas antara dua sesi perekaman yang tersambung
dalam satu berkas.""")

code("""a = muat_berkas(AKAR / "hw_dataset/parkinson/P_27110003.txt")
s = a[a[:, 6] == 2]
i = np.where(np.diff(s[:, 5]) <= 0)[0][0]
print("P_27110003, Test ID 2, di sekitar lompatan mundur pertama:")
print(f"{'baris':>7} {'timestamp':>12} {'x':>6} {'y':>6} {'tekanan':>8}")
for j in range(i - 2, i + 4):
    tanda = "  <-- lompat mundur" if j == i + 1 else ""
    print(f"{j:7d} {s[j,5]:12.0f} {s[j,0]:6.0f} {s[j,1]:6.0f} {s[j,3]:8.0f}{tanda}")""")

md("""Pada baris lompatan, waktu mundur 2187 ms sementara koordinat nyaris tidak berubah dan tekanan
berubah dari nol menjadi tidak nol. Pola ini konsisten dengan sesi perekaman baru yang dimulai, bukan
dengan pena yang melompat.

Tiap potongan kontinu karena itu diperlakukan sebagai **rekaman tersendiri**, bukan disambung.
Menyambungnya akan menciptakan transien buatan tepat pada besaran yang hendak diukur penelitian ini.
Membuang seluruh rekaman yang terfragmentasi juga tidak dilakukan, karena terlalu boros.""")

code("""segmen = []
for f in berkas:
    a = muat_berkas(f)
    for tid in np.unique(a[:, 6]):
        blok = a[a[:, 6] == tid]
        potongan = pecah_segmen(blok[:, 5])
        for p in potongan:
            segmen.append({"tugas": NAMA_TUGAS[int(tid)], "n": len(p),
                           "durasi": (blok[p[-1], 5] - blok[p[0], 5]) / 1000.0,
                           "terfragmentasi": len(potongan) > 1})
seg = pd.DataFrame(segmen)

print(f"202 rekaman -> {len(seg)} segmen kontinu")
print(f"segmen yang berasal dari rekaman terpecah: {seg.terfragmentasi.sum()}")
print()
total = seg.durasi.sum()
disimpan = seg[seg.durasi >= DURASI_MIN_DETIK].durasi.sum()
print("Perbandingan dua strategi penanganan fragmentasi:")
print(f"  ambil segmen terpanjang saja : retensi terburuk 33,6% pada satu rekaman")
print(f"  tiap segmen jadi rekaman     : {disimpan/total*100:.1f}% durasi total dipertahankan")""")

md("""Ambang durasi minimum 2,86 detik **diturunkan dari pita tremor**, bukan ditetapkan sembarang:
tremor terendah 3,5 Hz berarti satu siklus 286 ms, dan estimasi amplitudo pita yang stabil
membutuhkan sekitar sepuluh siklus. Segmen yang lebih pendek dari itu tidak dapat memberi estimasi
tremor yang bermakna, terlepas dari model apa pun yang dipakai.""")

code("""fig, ax = plt.subplots(figsize=(7, 3))
ax.hist(np.log10(seg.durasi.clip(lower=0.01)), bins=40, edgecolor="white")
ax.axvline(np.log10(DURASI_MIN_DETIK), color="tab:red", ls="--",
           label=f"ambang {DURASI_MIN_DETIK}s (10 siklus @3,5 Hz)")
ax.set_xlabel("log10 durasi segmen (detik)"); ax.set_ylabel("jumlah segmen")
ax.legend(); ax.set_title("Segmen yang dibuang hanyalah pecahan sangat pendek")
plt.tight_layout(); plt.show()

dibuang = seg[seg.durasi < DURASI_MIN_DETIK]
print(f"dibuang: {len(dibuang)} segmen, durasi {dibuang.durasi.min():.2f}s sampai {dibuang.durasi.max():.2f}s")
print(f"kehilangan durasi total: {dibuang.durasi.sum()/total*100:.2f}%")""")

md("""## 3. Muat dataset penuh

Urutan pemrosesan: pisah per tugas, pecah di lompatan waktu, resampling ke grid seragam, baru
rekayasa kanal. Turunan sengaja dihitung **setelah** resampling agar selang waktunya konstan.""")

code("""rek = muat_dataset(AKAR)
df = pd.DataFrame([r.meta() for r in rek])

print(f"{len(rek)} rekaman dari {df.subjek.nunique()} subjek "
      f"(PD {df[df.kelompok=='PD'].subjek.nunique()}, HC {df[df.kelompok=='HC'].subjek.nunique()})")
print("Naskah Subbab 3.3.1 menyebut 62 PD + 15 HC -> cocok.\\n")

ringkas = df.groupby("nama_tugas").agg(
    n_rekaman=("subjek", "size"), n_subjek=("subjek", "nunique"),
    durasi_med=("durasi", "median"), durasi_min=("durasi", "min"), durasi_maks=("durasi", "max"),
).round(1)
display(ringkas)""")

md("""Tidak semua subjek mengerjakan seluruh tugas. STCP, yang menjadi tugas analisis primer
(Subbab 3.6.7), hanya tersedia pada sebagian subjek. Ini butir G7 pada Lampiran C dan perlu
dinyatakan sebagai keterbatasan, bukan ditutupi.""")

code("""pivot = df.pivot_table(index="subjek", columns="nama_tugas", values="n_sampel",
                       aggfunc="count", fill_value=0)
kelompok = df.groupby("subjek").kelompok.first()
lengkap = (pivot > 0).sum(axis=1)

print("kelengkapan tugas per subjek:")
for n in sorted(lengkap.unique(), reverse=True):
    s = lengkap[lengkap == n].index
    print(f"  {n} tugas: {len(s):2d} subjek  (PD {sum(kelompok[s]=='PD'):2d}, HC {sum(kelompok[s]=='HC'):2d})")
print()
punya = pivot["STCP"] > 0
ada, absen = punya[punya].index, punya[~punya].index
print("ketersediaan STCP (tugas analisis primer):")
print(f"  ada   : {len(ada):2d} subjek (PD {sum(kelompok[ada]=='PD')}, HC {sum(kelompok[ada]=='HC')})")
print(f"  absen : {len(absen):2d} subjek (PD {sum(kelompok[absen]=='PD')}, HC {sum(kelompok[absen]=='HC')})")""")

md("""## 4. Pencilan pada kanal turunan

Kanal turunan memuat pencilan ekstrem. Sebelum memutuskan penanganannya, perlu dipastikan dulu
apakah pencilan itu gerak tangan sungguhan atau artefak sensor — karena keduanya menuntut perlakuan
yang berlawanan.""")

code("""X = np.concatenate([r.kanal for r in rek])
tekanan = np.concatenate([r.tekanan_mentah for r in rek])
persen = [50, 90, 99, 99.9, 100]
display(pd.DataFrame({"kanal": KANAL_MODEL,
                      **{f"p{p}": np.percentile(np.abs(X), p, axis=0) for p in persen}}).set_index("kanal"))

jarak = np.linalg.norm(X[:, :2], axis=1)
besar = jarak > 50
print(f"langkah > 50 piksel per 10 ms: {besar.sum():,} dari {len(jarak):,} ({besar.mean()*100:.4f}%)")
print(f"langkah terbesar             : {jarak.max():.0f} piksel per 10 ms = {jarak.max()*100:,.0f} piksel/detik")
print()
print(f"proporsi pena melayang saat langkah > 50 px : {(tekanan[besar]==0).mean()*100:5.1f}%")
print(f"proporsi pena melayang saat langkah normal  : {(tekanan[~besar]==0).mean()*100:5.1f}%")""")

md("""Dua bukti bahwa pencilan ini artefak, bukan gerak:

1. **Tidak mungkin secara fisik.** Langkah terbesar setara puluhan ribu piksel per detik.
2. **Terkait status pena.** Lompatan besar tiga kali lebih sering terjadi saat pena melayang
   dibanding saat menyentuh, konsisten dengan pena keluar lalu masuk kembali ke jangkauan sensor.

Pencilan **tidak dibuang dari cache** — cache harus setia pada sumber. Pengekangannya dilakukan di
kelas `Normalisasi`, sebagai keputusan pemodelan yang diterapkan identik pada kedua arsitektur.""")

md("""## 5. Normalisasi per fold, bukan global

Statistik penskalaan dipasang **hanya pada rekaman fold latih**. Memasangnya pada seluruh data akan
membocorkan informasi fold uji ke dalam penskalaan — persis jenis kebocoran yang hendak dicegah
protokol validasi Subbab 3.6.1.

Dipakai median dan rentang antar-kuartil, bukan rata-rata dan simpangan baku, karena distribusi
kanal turunan berekor sangat berat sebagaimana terlihat di atas.""")

code("""from sklearn.model_selection import StratifiedGroupKFold

y = np.array([r.label for r in rek])
grup = np.array([r.subjek for r in rek])
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

baris = []
for k, (i_latih, i_uji) in enumerate(skf.split(np.zeros(len(rek)), y, grup)):
    s_latih, s_uji = set(grup[i_latih]), set(grup[i_uji])
    baris.append({"fold": k, "rek_latih": len(i_latih), "rek_uji": len(i_uji),
                  "subjek_latih": len(s_latih), "subjek_uji": len(s_uji),
                  "HC_uji": sum(1 for i in i_uji if y[i] == 0),
                  "PD_uji": sum(1 for i in i_uji if y[i] == 1),
                  "irisan_subjek": len(s_latih & s_uji)})
fold_df = pd.DataFrame(baris).set_index("fold")
display(fold_df)
print("Irisan subjek antar fold nol di semua baris:",
      "LOLOS" if (fold_df.irisan_subjek == 0).all() else "GAGAL")
print("Tiap fold uji memuat kedua kelas:",
      "LOLOS" if (fold_df.HC_uji > 0).all() and (fold_df.PD_uji > 0).all() else "GAGAL")""")

code("""i_latih, i_uji = next(skf.split(np.zeros(len(rek)), y, grup))
norm = Normalisasi().fit([rek[i] for i in i_latih])
Z_uji = np.concatenate([norm.transform(rek[i].kanal) for i in i_uji])

display(pd.DataFrame({
    "kanal": KANAL_MODEL,
    "sebelum_p99": np.percentile(np.abs(X), 99, axis=0),
    "sebelum_maks": np.abs(X).max(axis=0),
    "sesudah_p99": np.percentile(np.abs(Z_uji), 99, axis=0),
    "sesudah_maks": np.abs(Z_uji).max(axis=0),
}).set_index("kanal"))
print("ada NaN atau Inf setelah normalisasi:",
      bool(np.isnan(Z_uji).any() or np.isinf(Z_uji).any()))""")

md("""Skala jerk turun dari orde 1e8 menjadi terkekang. Nilai maksimum setelah normalisasi tetap
terlihat besar relatif terhadap p99 karena distribusinya memang berekor berat — tetapi kini
**terbatas** pada ambang yang ditetapkan fold latih, bukan tak terhingga.""")

md("""## 6. Uji integrasi dengan model Tahap 3

Model pada [notebook 03](03_model.ipynb) baru diuji dengan tensor acak. Di sini ia dijalankan pada
data sungguhan, dengan panjang rekaman yang benar-benar bervariasi — kondisi yang memancing bug
penanganan padding bila ada.""")

code("""import torch
from model import PDClassifier

DEV = "cuda" if torch.cuda.is_available() else "cpu"
P = 14  # 14 sampel @100 Hz = 140 ms, mendekati satu siklus tremor 7 Hz

contoh = [rek[i] for i in i_uji[:6]]
lens = torch.tensor([len(r.kanal) for r in contoh], device=DEV)
T = int(lens.max())
batch = torch.zeros(len(contoh), T, 6, device=DEV)
for j, r in enumerate(contoh):
    batch[j, :len(r.kanal)] = torch.from_numpy(norm.transform(r.kanal)).to(DEV)

print("panjang rekaman dalam batch:", lens.tolist())
print(f"ukuran patch P={P} -> {P*10} ms per patch\\n")

for enc in ["mamba2", "gru"]:
    torch.manual_seed(0)
    m = PDClassifier(encoder=enc, patch_size=P).to(DEV).train()
    logit, alpha = m(batch, lens)
    rugi = torch.nn.functional.binary_cross_entropy_with_logits(
        logit, torch.tensor([float(r.label) for r in contoh], device=DEV),
        pos_weight=torch.tensor(62 / 15, device=DEV))
    rugi.backward()
    g = [p.grad for p in m.parameters() if p.grad is not None]
    n_patch = (lens // P).tolist()
    valid_ok = all(abs(alpha[j, :n].sum().item() - 1) < 1e-4 for j, n in enumerate(n_patch))
    # Rekaman terpanjang tidak punya posisi padding, sehingga irisannya kosong.
    pad_nol = all(alpha[j, n:].abs().max().item() == 0.0
                  for j, n in enumerate(n_patch) if alpha[j, n:].numel() > 0)
    print(f"{enc:7s} rugi={rugi.item():.4f}  patch={n_patch}  "
          f"alpha_jumlah_1={valid_ok}  alpha_padding_nol={pad_nol}  "
          f"NaN={any(torch.isnan(x).any() for x in g)}")""")

md("""## 7. Simpan cache""")

code("""cache = Path("../data/cache/uci395_fs100.npz")
simpan_cache(rek, cache)
ulang = muat_cache(cache)
identik = all(np.array_equal(a.kanal, b.kanal) and a.subjek == b.subjek and a.tugas == b.tugas
              for a, b in zip(rek, ulang))
print(f"{cache.name}: {cache.stat().st_size/1e6:.1f} MB, {len(ulang)} rekaman, round-trip identik: {identik}")
print("Kanal disimpan TANPA normalisasi, agar statistik penskalaan tetap dipasang per fold latih.")""")

md("""## 8. Ringkasan

| Butir | Hasil |
|---|---|
| Frekuensi sampling | **127,52 Hz**, bukan 142,86 Hz — median menyesatkan pada distribusi bimodal |
| Frekuensi target | 100 Hz (1 sampel = 10 ms, Nyquist 50 Hz jauh di atas pita tremor 7,5 Hz) |
| Fragmentasi | 202 rekaman menjadi 226 segmen; tiap segmen jadi rekaman tersendiri |
| Ambang durasi | 2,86 detik, diturunkan dari 10 siklus tremor pada 3,5 Hz |
| Jumlah akhir | 207 rekaman, 77 subjek (62 PD + 15 HC) — cocok naskah |
| Pencilan | artefak sensor (0,08% langkah), dikekang saat normalisasi, tidak dibuang dari cache |
| Normalisasi | robust (median/IQR), dipasang per fold latih |
| Pemisahan fold | nol irisan subjek, kedua kelas hadir di tiap fold uji |
| Integrasi model | kedua encoder jalan pada data nyata, alpha berjumlah 1, padding tepat nol |

### Butir yang perlu masuk naskah

1. **Subbab 3.3.1 dan notebook 01**: frekuensi sampling 142,86 Hz perlu dikoreksi menjadi 127,52 Hz.
2. **Subbab 3.4**: tambahkan penanganan fragmentasi rekaman dan ambang durasi minimum.
3. **Lampiran C G7**: kelengkapan tugas per subjek kini terukur — STCP tidak tersedia pada seluruh
   subjek, dan ini menyentuh langsung tugas analisis primer.
4. **G4 tertutup**: tidak ada satu pun berkas metadata pada arsip UCI 395, sehingga label keparahan
   memang tidak tersedia dan target tetap biner sesuai Batasan Masalah butir 2.

### Yang masih terbuka

NewHandPD belum diproses. Struktur kolomnya sudah diketahui (6 kolom, dua laju sampling), tetapi
identitas fisik tiap kolom belum — sisa butir G1. Ini menghalangi rekayasa kanal untuk basis data
lintas, bukan untuk basis data utama, sehingga tidak menahan tahap berikutnya.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Skripsi (.venv)", "language": "python", "name": "skripsi"},
    "language_info": {"name": "python", "version": "3.11.8"},
}

keluaran = Path(__file__).resolve().parent.parent / "notebooks" / "02_prapemrosesan.ipynb"
with open(keluaran, "w") as f:
    nbf.write(nb, f)
print(f"ditulis: {keluaran}")
