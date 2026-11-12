# Implementasi tahap pengembangan setelah V5

## Ditambahkan

- Protokol operasional lokal bertanggal 2026-10-16, keputusan desain, source pins,
  petunjuk eksekusi dan pemeriksaan hasil.
- `scripts/build_nb_19.py` → `notebooks/19_attention_audit_diagnostics.ipynb`.
  Mencakup map references, entropy/effective support/boundary gaps, weighted
  partial-rank sensitivity, dan kontras replacement pada irisan pasangan.
- `scripts/build_nb_20.py` → `notebooks/20_fixed_model_ranking_swap.ipynb`.
  Mencakup seluruh pasangan terarah tiga seed pada model tetap, kontrol RMS untuk
  setiap mask, displaced-map reference yang dicocokkan JSD, self-swap, dosis,
  variasi kontrol, inferensi subjek, multiplicity dan profiling/resume.
- Sembilan tes perangkat lunak sintetis di
  `tests/test_attention_extension_builders.py`.

## Perubahan terhadap usulan awal diskusi

- Endpoint primer baru adalah mean D=G_own-G_swap. JSD–|D| menjadi analisis
  sekunder terencana. G_swap dilaporkan sebagai pendamping utama. Tidak ada
  perubahan terhadap estimator primer atau angka bridge V5 yang sudah ada.
- G dihitung terhadap RMS control milik setiap mask; selisih T mentah juga
  dilaporkan. Perbedaan dosis/geometri own–swap tidak dianggap hilang otomatis.
- Plasebo dibentuk dari bobot alpha sendiri, bukan ranking acak tanpa bobot yang
  kemudian dipaksakan mempunyai JSD. Kandidat hanya dipilih dengan peta/JSD.
- Self-swap menguji identitas mask/plan dan D tepat nol jika terdefinisi; numerical
  replay dinilai dengan toleransi lama. Matching yang gagal tidak diisi nol.
- Ketidakpastian subjek, finite-control-pool sensitivity dan seed influence
  dibedakan. Tidak ada klaim CI populasi seed atau ketidakpastian fresh search.
- Empat mean-D strata memiliki p bootstrap berpusat aproksimatif dan penyesuaian
  Holm; CI tetap nominal dan seluruh tahap tetap eksploratori.

## Dipertahankan

- V5, V4, DOCX awal dan arsip historis.
- Builder/notebook 15–18 beserta keluaran sel dan angka yang telah dilaporkan.
- Checkpoint, CSV, manifest, cache dan profil numerik lama.
- Protokol training, preprocessing, subject folds, seed, patch/grid, masker donor,
  budget, matching cap/caliper/kuota dan toleransi baseline.
- Status preregistrasi lama serta pemisahan bank historis dari bank checkpoint.

## Batas pekerjaan yang telah dilakukan

Builder dijalankan hanya untuk menghasilkan file notebook. Pemeriksaan sintaks,
struktur, nama variabel dan tes sintetis dilakukan. Tidak ada sel penelitian,
model nyata, training, intervensi, bootstrap subjek atau analisis sinyal nyata
yang dieksekusi. Belum ada hasil baru untuk dimasukkan ke manuskrip.

N replikasi, total-versus-new seed, task UCI final dan registrasi publik tetap
keputusan tahap berikutnya. Notebook replikasi dan manuskrip V6 belum dibuat.
