"""Verifikasi encoder — tujuh pemeriksaan, untuk arsitektur mana pun.

Jalankan: python3 scripts/verifikasi_encoder.py [nama_encoder ...]
Tanpa argumen: memeriksa ketiganya.

Berdiri sendiri karena notebooks/03_model.ipynb tidak punya builder, sehingga
ujinya tidak dapat dijalankan ulang untuk arsitektur baru. Dijalankan untuk
SELURUH arm, bukan hanya yang baru, sehingga sekaligus menjadi uji regresi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from model import PDClassifier  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
P = 7
ACUAN_PARAM = 233_808   # encoder BiMamba-2, patokan kesetaraan
TOLERANSI = 0.05


def n_param_encoder(m):
    return sum(p.numel() for p in m.encoder.parameters())


def periksa(nama: str) -> list[tuple[str, bool, str]]:
    hasil = []
    torch.manual_seed(0)
    m = PDClassifier(encoder=nama, patch_size=P).to(DEV).eval()
    n_enc, n_tot = n_param_encoder(m), sum(p.numel() for p in m.parameters())

    # 1. kesetaraan parameter
    selisih = (n_enc - ACUAN_PARAM) / ACUAN_PARAM
    hasil.append(("1. parameter encoder setara (+-5%)", abs(selisih) <= TOLERANSI,
                  f"encoder {n_enc:,} ({selisih:+.2%}), total {n_tot:,}"))

    # 2. forward dan backward tanpa NaN/Inf, 3. batch panjang beragam
    panjang = [222, 150, 29]
    T = max(panjang)
    x = torch.randn(len(panjang), T * P, 6, device=DEV, requires_grad=True)
    lengths = torch.tensor([n * P for n in panjang], device=DEV)
    for i, n in enumerate(panjang):
        x.data[i, n * P:] = 0
    m.train()
    logit, alpha = m(x, lengths)
    logit.square().mean().backward()
    bersih = (torch.isfinite(logit).all().item() and torch.isfinite(alpha).all().item()
              and x.grad is not None and torch.isfinite(x.grad).all().item())
    hasil.append(("2. forward+backward tanpa NaN/Inf", bersih,
                  f"logit {tuple(logit.shape)}, alpha {tuple(alpha.shape)}"))
    hasil.append(("3. batch dengan >=2 panjang berbeda", len(set(panjang)) >= 2,
                  f"panjang patch {panjang}"))

    # 4. kebocoran padding lintas-sampel: sampel B dalam batch == B solo
    m.eval()
    with torch.no_grad():
        lo_batch, _ = m(x.detach(), lengths)
        j = 1
        xs = x.detach()[j : j + 1, : panjang[j] * P]
        lo_solo, _ = m(xs, lengths[j : j + 1])
    d = float((lo_batch[j] - lo_solo[0]).abs())
    hasil.append(("4. kebocoran padding lintas-sampel <1e-4", d < 1e-4, f"selisih {d:.2e}"))

    # 5. regresi arah mundur: flip naif HARUS berbeda dari flip per panjang asli
    if hasattr(m.encoder, "_flip_valid"):
        h = torch.randn(2, 40, 128, device=DEV)
        pjg = torch.tensor([40, 17], device=DEV)
        h[1, 17:] = 0
        naif = torch.flip(h, dims=[1])
        benar = m.encoder._flip_valid(h, pjg)
        beda = float((naif - benar).abs().max())
        hasil.append(("5. flip naif != flip per panjang asli", beda > 1e-3,
                      f"selisih maks {beda:.3e} (nol berarti bug arah mundur kembali)"))
    else:
        hasil.append(("5. flip naif != flip per panjang asli", True,
                      "encoder tanpa _flip_valid (BiGRU memakai pack_padded_sequence)"))

    # 6. bobot alpha tepat nol pada posisi padding
    with torch.no_grad():
        _, alpha = m(x.detach(), lengths)
    maks_pad = max(float(alpha[i, n:].abs().max()) if n < alpha.shape[1] else 0.0
                   for i, n in enumerate(panjang))
    hasil.append(("6. alpha tepat nol di padding", maks_pad == 0.0, f"maks |alpha| {maks_pad:.3e}"))

    # 7. kasus tepi: panjang 29 sampai 467
    tepi_ok, catatan = True, []
    for n in (29, 467):
        try:
            with torch.no_grad():
                lo, _ = m(torch.randn(2, n * P, 6, device=DEV),
                          torch.tensor([n * P, n * P], device=DEV))
            ok = torch.isfinite(lo).all().item()
        except Exception as e:  # noqa: BLE001
            ok = False
            catatan.append(f"{n}: {type(e).__name__}")
        tepi_ok &= ok
    hasil.append(("7. panjang tepi 29 dan 467", tepi_ok, "; ".join(catatan) or "keduanya jalan"))
    return hasil


def main(argv: list[str]) -> int:
    daftar = argv[1:] or ["mamba2", "mamba3", "gru"]
    print(f"perangkat {DEV} | patch {P} | acuan encoder {ACUAN_PARAM:,} parameter\n")
    semua_lolos = True
    for nama in daftar:
        print("=" * 76)
        print(f"ENCODER: {nama}")
        print("=" * 76)
        try:
            hasil = periksa(nama)
        except Exception as e:  # noqa: BLE001
            print(f"  GAGAL DIBANGUN: {type(e).__name__}: {e}\n")
            semua_lolos = False
            continue
        for label, lolos, catatan in hasil:
            print(f"  [{'LOLOS' if lolos else 'GAGAL'}] {label:38s} {catatan}")
            semua_lolos &= lolos
        print()
    print("=" * 76)
    print("SELURUH PEMERIKSAAN LOLOS" if semua_lolos else "ADA PEMERIKSAAN YANG GAGAL")
    return 0 if semua_lolos else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
