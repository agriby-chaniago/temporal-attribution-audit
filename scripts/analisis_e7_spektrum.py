"""E7 — spektrum rotasi state kompleks BiMamba-3.

Pertanyaannya: adakah rotasi state yang frekuensi karakteristiknya jatuh di pita
tremor 3,5 sampai 7,5 Hz, tanpa model pernah diberi tahu pita itu.

Analisis ini hanya dapat dilakukan pada BiMamba-3. BiMamba-2 memakai peluruhan
bernilai riil, yang merepresentasikan skala waktu tetapi bukan frekuensi;
BiGRU tidak memiliki besaran yang sepadan sama sekali.

Mekanisme yang diukur, dibaca langsung dari kernel `angle_dt.py`:

    sudut  = tanh(angle_mentah) * pi        sudut per satuan waktu, (-pi, pi)
    DT     = softplus(dd_dt + dt_bias)      langkah diskretisasi, per head
    rotasi = sudut * DT                     kenaikan fase per langkah patch

sehingga frekuensinya, pada grid patch berfrekuensi fs_patch,

    f = rotasi / (2 pi) * fs_patch = tanh(angle) * DT * fs_patch / 2

**Sudutnya bergantung data**, dihitung dari in_proj pada setiap langkah waktu,
bukan parameter tetap per head. Yang diukur karena itu bukan "apakah arsitektur
punya head di 5 Hz", melainkan "apakah model MEMILIH laju rotasi di pita tremor
ketika melihat data pasien".

Batas keterwakilan: patch 7 pada 100 Hz memberi grid encoder 100/7 = 14,29 Hz,
sehingga Nyquist-nya 7,14 Hz. Pita tremor hanya terwakili sampai batas itu.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, "src")
from preprocessing import muat_cache, Normalisasi  # noqa: E402
from model import PDClassifier  # noqa: E402
from training import latih, susun_batch  # noqa: E402
from marker import PITA_TREMOR  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
P, EPOCHS, SEED, FS = 7, 35, 0, 100
FS_PATCH = FS / P                      # 14,2857 Hz
NYQUIST = FS_PATCH / 2                 # 7,1429 Hz
HASIL = Path("results")


@torch.no_grad()
def spektrum_blok(blok, h):
    """Frekuensi rotasi (Hz) yang dipilih satu blok Mamba3 untuk masukan h.

    Mereplikasi pemisahan in_proj pada mamba3.py baris 152-177, lalu
    menerapkan rumus frekuensi di atas. Mengembalikan (B, L, nheads, n_sudut).
    """
    keluar = blok.in_proj(h)
    bagian = [blok.d_inner, blok.d_inner,
              blok.d_state * blok.num_bc_heads * blok.mimo_rank,
              blok.d_state * blok.num_bc_heads * blok.mimo_rank,
              blok.nheads, blok.nheads, blok.nheads, blok.num_rope_angles]
    _, _, _, _, dd_dt, _, _, sudut_mentah = torch.split(keluar, bagian, dim=-1)

    DT = F.softplus(dd_dt + blok.dt_bias).float()                 # (B, L, nheads)
    sudut = torch.tanh(sudut_mentah.float())                      # (B, L, n_sudut), sudah dibagi pi
    # f = tanh(angle) * DT * fs_patch / 2
    f = sudut.unsqueeze(2) * DT.unsqueeze(-1) * (FS_PATCH / 2)
    return f


def kumpulkan(model, rek, indeks, norm, batch=8):
    """Frekuensi rotasi pada posisi valid saja, dikumpulkan per rekaman."""
    model.eval()
    per_rekaman = {}
    for b in range(0, len(indeks), batch):
        pilih = list(indeks[b:b + batch])
        bat = susun_batch(rek, pilih, norm, DEV)
        h, plen = model.patch_embed(bat.x, bat.lengths)
        for lapis in range(len(model.encoder.down)):
            hh = model.encoder.down[lapis](h)
            f = spektrum_blok(model.encoder.fwd[lapis], hh)       # (B, L, H, S)
            for j, idx in enumerate(pilih):
                n = int(plen[j])
                per_rekaman.setdefault(idx, []).append(
                    f[j, :n].abs().flatten().cpu().numpy())
            h = model.encoder(h, plen) if lapis == len(model.encoder.down) - 1 else h
    return {k: np.concatenate(v) for k, v in per_rekaman.items()}


def main():
    rek = muat_cache(Path("data/cache/uci395_fs100.npz"))
    y = np.array([r.label for r in rek])
    grup = np.array([r.subjek for r in rek])
    print(f"perangkat {DEV} | patch {P} | grid encoder {FS_PATCH:.4f} Hz | "
          f"Nyquist {NYQUIST:.4f} Hz")
    print(f"pita tremor {PITA_TREMOR[0]}-{PITA_TREMOR[1]} Hz -> terwakili hanya sampai "
          f"{min(PITA_TREMOR[1], NYQUIST):.4f} Hz\n")

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    i_lat, i_uji = next(iter(skf.split(np.zeros(len(rek)), y, grup)))
    norm = Normalisasi().fit([rek[i] for i in i_lat])
    torch.manual_seed(SEED)
    m = PDClassifier(encoder="mamba3", patch_size=P).to(DEV)
    print("melatih BiMamba-3 pada fold 0 ...", flush=True)
    latih(m, rek, list(i_lat), norm, epochs=EPOCHS, device=DEV, seed=SEED)

    f_per = kumpulkan(m, rek, i_uji, norm)
    semua = np.concatenate(list(f_per.values()))
    lo, hi = PITA_TREMOR[0], min(PITA_TREMOR[1], NYQUIST)

    print(f"\n{len(f_per)} rekaman uji, {len(semua):,} nilai frekuensi rotasi\n")
    print("sebaran |f| (Hz):")
    for q in (50, 75, 90, 95, 99, 99.9):
        print(f"  persentil {q:>5.1f} : {np.percentile(semua, q):.4f}")
    print(f"  maksimum       : {semua.max():.4f}")

    frac = float(((semua >= lo) & (semua <= hi)).mean())
    print(f"\nproporsi rotasi di pita tremor terwakili [{lo}, {hi:.2f}] Hz: {frac:.4%}")
    print(f"proporsi di bawah {lo} Hz (lebih lambat dari tremor)        : "
          f"{float((semua < lo).mean()):.4%}")

    baris = []
    for idx, f in f_per.items():
        baris.append({"subjek": grup[idx], "label": int(y[idx]),
                      "frac_pita": float(((f >= lo) & (f <= hi)).mean()),
                      "f_median": float(np.median(f)), "f_p99": float(np.percentile(f, 99))})
    df = pd.DataFrame(baris).groupby(["subjek", "label"]).mean().reset_index()
    print("\nper kelompok, diagregasi ke subjek:")
    print(df.groupby("label")[["frac_pita", "f_median", "f_p99"]].median().round(6).to_string())

    df.to_csv(HASIL / "e7_spektrum_subjek.csv", index=False)
    pd.DataFrame([{
        "fs_patch": FS_PATCH, "nyquist": NYQUIST,
        "pita_bawah": lo, "pita_atas_terwakili": hi,
        "frac_di_pita": frac, "f_median": float(np.median(semua)),
        "f_p99": float(np.percentile(semua, 99)), "f_maks": float(semua.max()),
        "n_nilai": int(semua.size), "n_rekaman": len(f_per),
        "sumber": "scripts/analisis_e7_spektrum.py",
    }]).to_csv(HASIL / "e7_spektrum_ringkas.csv", index=False)
    print(f"\ndisimpan ke {HASIL}/e7_spektrum_subjek.csv dan e7_spektrum_ringkas.csv")


if __name__ == "__main__":
    main()
