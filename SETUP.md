# Setup Lingkungan

## Konteks perangkat

- Python 3.11.8
- GPU: NVIDIA RTX 3050 6GB Laptop, driver 610.57.04
- `nvcc` sistem: 12.1 (lebih lama dari wheel torch cu130 — tidak masalah, wheel PyTorch membawa runtime CUDA sendiri, `nvcc` sistem cuma dipakai Triton untuk kompilasi kernel saat runtime pertama kali dipanggil)

## Instalasi

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/patch_mamba_ssm.py
```

## Kenapa ada `wheels/` dan skrip patch

`mamba_ssm` dan `causal_conv1d` adalah ekstensi CUDA yang tidak punya wheel resmi di
PyPI untuk kombinasi Python 3.11 + torch 2.11 + CUDA 13.0 di mesin ini. Build dari
source butuh `nvcc` yang cocok dan `TORCH_CUDA_ARCH_LIST` yang benar (compute
capability 8.6 untuk RTX 3050) — proses yang dikenal rapuh.

Wheel di `wheels/` sudah dikompilasi dan diverifikasi jalan di mesin ini (lihat
riwayat kerja Tahap 1). Disimpan sebagai bagian proyek — bukan rujukan ke proyek lain
— supaya proyek ini tidak bergantung pada lingkungan eksternal untuk direproduksi.

Wheel `mamba_ssm==2.2.2` membawa bug ringan: `mamba_ssm/utils/generation.py`
mengimpor nama-nama dari `transformers.generation` yang sudah dihapus di versi
`transformers` baru, walau proyek ini tidak pernah memakai kode itu (cuma dipakai
blok `Mamba2` sebagai encoder, bukan `MambaLMHeadModel` untuk text-generation).
`scripts/patch_mamba_ssm.py` menambal ini dengan fallback try/except. Idempoten —
aman dijalankan berkali-kali, dan wajib dijalankan ulang tiap kali `mamba_ssm`
dipasang ulang dari `wheels/`.

## Apabila wheel perlu dibangun ulang dari nol

Source build asli (`/tmp/mamba-src`, `/tmp/causal-conv1d-src`) tidak disimpan —
lokasinya di `/tmp` dan sudah terhapus. Untuk membangun ulang:

```bash
git clone https://github.com/Dao-AILab/causal-conv1d && cd causal-conv1d && pip install -e . --no-build-isolation
git clone https://github.com/state-spaces/mamba && cd mamba && pip install -e . --no-build-isolation
```

Butuh `nvcc` yang cocok dengan versi CUDA torch, dan waktu kompilasi cukup lama
(kernel Triton + CUDA ekstensi). Setelah berhasil, salin wheel hasil build (`pip
wheel .` atau dari cache `~/.cache/pip/wheels/`) ke `wheels/` proyek ini agar tidak
perlu dibangun ulang lagi.

## Verifikasi instalasi

```bash
source .venv/bin/activate
python3 -c "
import torch
print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available())
from mamba_ssm.modules.mamba2 import Mamba2
block = Mamba2(d_model=128, d_state=64, d_conv=4, expand=2, headdim=32).cuda()
x = torch.randn(2, 500, 128, device='cuda')
y = block(x)
y.sum().backward()
print('Mamba2 forward+backward OK:', y.shape)
import shap
print('shap:', shap.__version__)
"
```

## Kernel Jupyter

Skrip `notebooks/01_verifikasi_struktur_data.ipynb` dibangun dengan kernel `skripsi`,
didaftarkan lewat:

```bash
python -m ipykernel install --user --name=skripsi --display-name="Skripsi (.venv)"
```

Sudah terpasang di mesin ini. Perlu diulang kalau `.venv` dibuat ulang dari nol.
