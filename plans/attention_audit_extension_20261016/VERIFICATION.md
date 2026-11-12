# Status pemeriksaan implementasi

Pemeriksaan ini menilai **perangkat lunak yang disiapkan**, bukan hasil penelitian
tambahan. Tidak ada sel penelitian atau model nyata yang dijalankan agen.

## Lolos

- Kedua builder menghasilkan notebook baru melalui workflow `.py` proyek.
- Seluruh code cell lolos parsing/kompilasi sintaks Python dan struktur nbformat.
- Pemeriksaan nama dengan Pyflakes bersih pada gabungan source setiap notebook.
- Seluruh cell tetap belum dieksekusi dan tanpa output.
- Notebook tersimpan cocok dengan source builder saat penyerahan.
- Sembilan tes sintetis lolos: generasi tanpa training, urutan header CSV,
  atomic cache/path argument dan penolakan overwrite, top-rank dekat tie,
  konstruksi plasebo, reuse mask identik tanpa forward, kelengkapan AOPC/kuota,
  pembobotan target–sumber, serta degenerasi partial rank.
- Inventori 30 checkpoint mencakup dua arsitektur, tiga seed dan lima fold.
  Hash checkpoint, identitas metadata/spec, hash dan identitas fold cache,
  serta hash source cocok. Pemeriksaan dilakukan pada byte/metadata, tanpa
  deserialisasi bobot atau inisialisasi model.
- Metadata versi paket yang terpasang cocok dengan manifest bank asli. CUDA,
  device dan numerical replay belum dijalankan untuk memverifikasi runtime.
- Seluruh **2.203 berkas lama yang dipantau** masih tersedia dan hash-nya tidak
  berubah, termasuk V4/V5, DOCX awal, notebook lama, source, checkpoint dan
  keluarga artefak hasil yang dicantumkan pada snapshot.

## Belum dilakukan

- Eksekusi notebook 19/20, bootstrap subjek, hasil referensi JSD atau partial rank.
- Pengujian model nyata, profil CUDA tersimpan, replay logit/attention dan kontrol.
- Pencarian/forward ranking-swap maupun plasebo; coverage belum diketahui.
- Pemeriksaan visual grafik baru yang memerlukan hasil nyata.
- Estimasi biaya komputasi aktual, training seed tambahan atau UCI.
- Registrasi publik dan perubahan manuskrip V5.

Pemeriksaan statis tidak menjamin kecocokan numerik GPU. Notebook 20 menyertakan
probe profil lama dan replay per rekaman; kegagalan menghentikan run dengan
diagnostik, tanpa memperlebar toleransi atau training fallback.

## Bukti pemeriksaan

- `validation/implementation_checks.json`: sintaks, nbformat, cell tanpa eksekusi,
  guard `allow_training=False`, Pyflakes, hash notebook dan batas pekerjaan.
- `validation/builder_notebook_consistency.json`: kecocokan source builder–notebook.
- `validation/static_name_check.json`: hasil pemeriksaan nama final.
- `validation/synthetic_tests.log`: log sembilan tes software sintetis.
- `validation/dependency_inventory.json`: pemeriksaan metadata/dependency/30 checkpoint.
- `validation/protected_files_check.json`: hasil pencocokan 2.203 hash lama.
- `protected_files_before.json`: snapshot sebelum implementasi.

File `validation/notebook_19_source.txt` dan `notebook_20_source.txt` adalah
ekstraksi source untuk audit statis, **bukan entry point analisis**. Source utama
penelitian tetap notebook yang dibangkitkan builder.
