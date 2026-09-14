"""
Tempel setelah `pip install -r requirements.txt`.

Wheel mamba_ssm==2.2.2 yang dipakai proyek ini (lihat wheels/, SETUP.md) berisi
`mamba_ssm/utils/generation.py` yang mengimpor GreedySearchDecoderOnlyOutput,
SampleDecoderOnlyOutput, TextStreamer dari transformers.generation secara tanpa
syarat. Nama-nama ini sudah dihapus di transformers versi baru.

Proyek ini hanya memakai blok Mamba2 sebagai encoder (Subbab 3.5 naskah),
tidak pernah memakai utilitas text-generation di berkas tersebut
(GenerationMixin.generate dkk.), sehingga stub kosong aman dipakai sebagai
fallback. Idempoten — aman dijalankan berkali-kali.
"""

import re
import sys
from pathlib import Path

TARGET_LINE = (
    "from transformers.generation import "
    "GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, TextStreamer"
)

PATCHED_BLOCK = """try:
    from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput, TextStreamer
except ImportError:
    # Nama-nama ini dihapus di transformers versi baru. Proyek ini hanya memakai
    # Mamba2 sebagai blok encoder, tidak pernah memakai utilitas text-generation
    # di berkas ini (GenerationMixin.generate dkk.), sehingga stub kosong aman.
    GreedySearchDecoderOnlyOutput = SampleDecoderOnlyOutput = TextStreamer = None"""


def find_generation_py() -> Path:
    import mamba_ssm

    path = Path(mamba_ssm.__file__).parent / "utils" / "generation.py"
    if not path.exists():
        sys.exit(f"Tidak ditemukan: {path}. Apakah mamba_ssm terpasang?")
    return path


def main() -> None:
    path = find_generation_py()
    text = path.read_text()

    if "except ImportError:" in text and "GreedySearchDecoderOnlyOutput = SampleDecoderOnlyOutput" in text:
        print(f"Sudah dipatch: {path}")
        return

    if TARGET_LINE not in text:
        sys.exit(
            f"Baris target tidak ditemukan di {path}. "
            "Berkas mungkin berbeda dari yang diasumsikan skrip ini — patch manual diperlukan."
        )

    text = text.replace(TARGET_LINE, PATCHED_BLOCK)
    path.write_text(text)
    print(f"Dipatch: {path}")


if __name__ == "__main__":
    main()
