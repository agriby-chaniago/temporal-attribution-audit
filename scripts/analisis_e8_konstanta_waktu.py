"""E8 — konstanta waktu head BiMamba-2. TIDAK DIJALANKAN.

Status per 4 September 2026: skrip ini sengaja **tidak dijalankan**, dan tidak
menghasilkan berkas apa pun di `results/`. Naskah tidak merujuk hasilnya di mana
pun, sehingga tidak ada klaim yang bergantung padanya. Alasan lengkap keputusan
itu ada di DOSIER-PROYEK.md §10. Disimpan sebagai jalan masuk bila pertanyaannya
kelak dianggap layak dikejar.

Pasangan langsung bagi E7. E7 menanyakan apakah BiMamba-3 memilih laju *rotasi*
di pita tremor; E8 menanyakan apakah BiMamba-2 memisahkan *skala waktu* antar
head, dan berapa banyak head yang skala waktunya sepadan dengan pita tremor
3,5 sampai 7,5 Hz.

Pertanyaan ini menguji Subbab 2.2.4 butir keunggulan 1, "pemisahan skala waktu
antar head", yang sejauh ini hanya dinilai lewat AUC dan divonis tidak didukung.
Vonis itu menyangkut apakah kapasitasnya menjelma menjadi akurasi, bukan apakah
kapasitasnya terwujud pada bobot. E8 mengukur yang kedua.

Perbedaan penting terhadap E7, dan ini harus dinyatakan di naskah
--------------------------------------------------------------
Mamba-2 memakai peluruhan bernilai riil, sehingga yang terukur adalah **skala
waktu**, bukan frekuensi rotasi. Keduanya tidak boleh disamakan. Frekuensi yang
dilaporkan di sini adalah **frekuensi karakteristik hasil konversi** dari
konstanta waktu, bukan laju osilasi yang direpresentasikan model.

Mekanisme yang diukur, dari kernel Mamba2:

    A      = -exp(A_log)                 peluruhan per head, A_log parameter
    dt     = softplus(dd_dt + dt_bias)   langkah diskretisasi, per head
    dA     = exp(dt * A)                 faktor peluruhan per langkah patch
    laju   = dt * exp(A_log)             = -ln(dA), per langkah
    tau    = 1 / laju                    konstanta waktu, dalam langkah patch
    tau_s  = tau / fs_patch              konstanta waktu, dalam detik

Dua konvensi frekuensi karakteristik dilaporkan berdampingan, sebab pilihannya
mengubah head mana yang terhitung masuk pita dan karena itu tidak boleh
tersembunyi:

    f_invers = 1 / tau_s                 kebalikan konstanta waktu
    f_sudut  = 1 / (2 * pi * tau_s)      frekuensi sudut lag orde satu

Dua ragam dt
------------
`dt_bias` adalah nilai dasar yang dipelajari per head; dt sebenarnya bergantung
data lewat `dd_dt` dari in_proj. Keduanya dihitung:

    statis  — dt = softplus(dt_bias). Prior arsitektural per head, tidak
              bergantung masukan. Inilah yang menjawab "apakah head terpisah
              skala waktunya".
    dinamis — dt = softplus(dd_dt + dt_bias) pada rekaman uji. Sepadan dengan
              yang diukur E7, sehingga perbandingan terhadap angka Mamba-3
              menjadi setara. Tanpa ini, membandingkan proporsi E8 terhadap
              proporsi E7 membandingkan dua besaran yang berbeda.

Protokol dibuat identik dengan E7 supaya keduanya dapat disandingkan: basis data
UCI 395, patch 7, fold 0 dari StratifiedGroupKFold(5, random_state=42),
35 epoch, seed 0.

Keluaran:
    results/e8_konstanta_waktu_head.csv    satu baris per head (48 baris)
    results/e8_konstanta_waktu_ringkas.csv satu baris ringkasan
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, "src")
from preprocessing import muat_cache, Normalisasi  # noqa: E402
from model import PDClassifier  # noqa: E402
from training import latih, susun_batch  # noqa: E402
from marker import PITA_TREMOR  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
P, EPOCHS, SEED, FS = 7, 35, 0, 100
FS_PATCH = FS / P                      # 14,2857 Hz
HASIL = Path("results")


@torch.no_grad()
def dt_statis(blok) -> np.ndarray:
    """softplus(dt_bias) per head. Prior arsitektural, tanpa masukan."""
    return F.softplus(blok.dt_bias.float()).cpu().numpy()


@torch.no_grad()
def dt_dinamis(blok, h) -> np.ndarray:
    """softplus(dd_dt + dt_bias) pada masukan h, dirata-rata posisi valid.

    Pemisahan in_proj mengikuti mamba_ssm.modules.mamba2: keluarannya tersusun
    sebagai [z, xBC, dt] dengan lebar [d_inner, d_inner + 2*ngroups*d_state,
    nheads].
    """
    keluar = blok.in_proj(h)
    lebar = [blok.d_inner, blok.d_inner + 2 * blok.ngroups * blok.d_state, blok.nheads]
    _, _, dd_dt = torch.split(keluar, lebar, dim=-1)
    dt = F.softplus(dd_dt.float() + blok.dt_bias.float())          # (B, L, nheads)
    return dt


def konstanta(A_log: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """tau dalam detik. laju = dt * exp(A_log) per langkah patch."""
    laju = dt * np.exp(A_log)
    return 1.0 / (laju * FS_PATCH)


def main() -> None:
    rek = muat_cache(Path("data/cache/uci395_fs100.npz"))
    y = np.array([r.label for r in rek])
    grup = np.array([r.subjek for r in rek])
    lo, hi = PITA_TREMOR
    print(f"perangkat {DEV} | patch {P} | grid encoder {FS_PATCH:.4f} Hz")
    print(f"pita tremor {lo}-{hi} Hz\n")

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    i_lat, i_uji = next(iter(skf.split(np.zeros(len(rek)), y, grup)))
    norm = Normalisasi().fit([rek[i] for i in i_lat])
    torch.manual_seed(SEED)
    m = PDClassifier(encoder="mamba2", patch_size=P).to(DEV)
    print("melatih BiMamba-2 pada fold 0 ...", flush=True)
    latih(m, rek, list(i_lat), norm, epochs=EPOCHS, device=DEV, seed=SEED)
    m.eval()

    # --- dt dinamis: rerata atas posisi valid seluruh rekaman uji ---
    n_lapis = len(m.encoder.down)
    akum: dict[tuple[int, str], list[np.ndarray]] = {}
    for b in range(0, len(i_uji), 8):
        pilih = list(i_uji[b:b + 8])
        bat = susun_batch(rek, pilih, norm, DEV)
        h, plen = m.patch_embed(bat.x, bat.lengths)
        for lapis in range(n_lapis):
            hh = m.encoder.down[lapis](h)
            for arah, blok in (("maju", m.encoder.fwd[lapis]), ("mundur", m.encoder.bwd[lapis])):
                dt = dt_dinamis(blok, hh)                          # (B, L, nheads)
                for j in range(len(pilih)):
                    n = int(plen[j])
                    akum.setdefault((lapis, arah), []).append(
                        dt[j, :n].mean(dim=0).cpu().numpy())
            h = m.encoder(h, plen) if lapis == n_lapis - 1 else h

    baris = []
    for lapis in range(n_lapis):
        for arah, blok in (("maju", m.encoder.fwd[lapis]), ("mundur", m.encoder.bwd[lapis])):
            A_log = blok.A_log.detach().float().cpu().numpy()
            ds = dt_statis(blok)
            dd = np.stack(akum[(lapis, arah)]).mean(axis=0)
            tau_s = konstanta(A_log, ds)
            tau_d = konstanta(A_log, dd)
            for h_i in range(len(A_log)):
                baris.append({
                    "lapis": lapis, "arah": arah, "head": h_i,
                    "A_log": float(A_log[h_i]),
                    "dt_statis": float(ds[h_i]), "dt_dinamis": float(dd[h_i]),
                    "tau_statis_ms": float(tau_s[h_i] * 1000),
                    "tau_dinamis_ms": float(tau_d[h_i] * 1000),
                    "f_invers_statis": float(1.0 / tau_s[h_i]),
                    "f_sudut_statis": float(1.0 / (2 * np.pi * tau_s[h_i])),
                    "f_invers_dinamis": float(1.0 / tau_d[h_i]),
                    "f_sudut_dinamis": float(1.0 / (2 * np.pi * tau_d[h_i])),
                })
    df = pd.DataFrame(baris)
    assert len(df) == n_lapis * 2 * len(A_log), f"harusnya {n_lapis*2*len(A_log)} head, dapat {len(df)}"

    ring = {"n_head": len(df), "n_lapis": n_lapis, "fs_patch": FS_PATCH,
            "pita_bawah": lo, "pita_atas": hi, "n_rekaman_uji": len(i_uji),
            "sumber": "scripts/analisis_e8_konstanta_waktu.py"}
    for ragam in ("statis", "dinamis"):
        tau = df[f"tau_{ragam}_ms"].to_numpy()
        ring[f"tau_min_ms_{ragam}"] = float(tau.min())
        ring[f"tau_maks_ms_{ragam}"] = float(tau.max())
        ring[f"tau_median_ms_{ragam}"] = float(np.median(tau))
        ring[f"rentang_kali_{ragam}"] = float(tau.max() / tau.min())
        for konv in ("invers", "sudut"):
            f = df[f"f_{konv}_{ragam}"].to_numpy()
            n_pita = int(((f >= lo) & (f <= hi)).sum())
            ring[f"n_head_di_pita_{konv}_{ragam}"] = n_pita
            ring[f"frac_di_pita_{konv}_{ragam}"] = n_pita / len(f)

    print(f"\n{len(df)} head = {n_lapis} lapis x 2 arah x {len(A_log)} head\n")
    for ragam in ("statis", "dinamis"):
        print(f"dt {ragam}:")
        print(f"  tau  {ring[f'tau_min_ms_{ragam}']:.1f} ms .. "
              f"{ring[f'tau_maks_ms_{ragam}']:.1f} ms "
              f"(median {ring[f'tau_median_ms_{ragam}']:.1f} ms, "
              f"rentang {ring[f'rentang_kali_{ragam}']:.0f}x)")
        for konv in ("invers", "sudut"):
            print(f"  head di pita, konvensi {konv:>6}: "
                  f"{ring[f'n_head_di_pita_{konv}_{ragam}']}/{len(df)} "
                  f"({ring[f'frac_di_pita_{konv}_{ragam}']:.1%})")

    df.to_csv(HASIL / "e8_konstanta_waktu_head.csv", index=False)
    pd.DataFrame([ring]).to_csv(HASIL / "e8_konstanta_waktu_ringkas.csv", index=False)
    print(f"\ndisimpan ke {HASIL}/e8_konstanta_waktu_head.csv dan e8_konstanta_waktu_ringkas.csv")


if __name__ == "__main__":
    main()
