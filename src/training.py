"""
Infrastruktur pelatihan dan evaluasi, dipakai bersama oleh seluruh skenario S1-S8.

Mengikuti Subbab 3.6.1 naskah proposal. Tiga keputusan protokol yang dikodekan
di sini, bukan diserahkan pada pemanggil:

1. **Bobot kelas pada loss.** Binary cross-entropy berbobot dengan bobot kelas
   positif dihitung dari frekuensi kelas pada himpunan latih. SMOTE dan variannya
   tidak dipakai karena interpolasi antar deret waktu tremor dari subjek berbeda
   menghasilkan sinyal yang bukan tremor siapa pun.

2. **Statistik normalisasi dipasang hanya pada fold latih.** Fungsi di sini
   menerima objek Normalisasi yang sudah dipasang, dan tidak pernah memasangnya
   sendiri dari data gabungan.

3. **Hyperparameter dibekukan sebelum eksperimen utama.** Batas epoch adalah
   argumen wajib, bukan hasil pemilihan otomatis berdasarkan fold uji. Memilih
   epoch terbaik berdasarkan fold uji akan membocorkan fold uji ke dalam
   pemilihan model, meniadakan pemisahan tingkat subjek.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from preprocessing import Normalisasi, Rekaman

__all__ = ["Batch", "susun_batch", "bobot_kelas_positif", "latih", "prediksi", "metrik_biner"]


@dataclass
class Batch:
    x: torch.Tensor        # (B, T, C) terpadding
    lengths: torch.Tensor  # (B,)
    y: torch.Tensor        # (B,)
    indeks: list[int]      # indeks rekaman asal, untuk penelusuran balik


def susun_batch(rekaman: list[Rekaman], indeks: list[int], norm: Normalisasi,
                device: str, label: np.ndarray | None = None) -> Batch:
    """Padding ke panjang maksimum dalam batch, dengan normalisasi fold latih."""
    potongan = [norm.transform(rekaman[i].kanal) for i in indeks]
    lengths = torch.tensor([len(p) for p in potongan], device=device)
    T, C = int(lengths.max()), potongan[0].shape[1]
    x = torch.zeros(len(potongan), T, C, device=device)
    for j, p in enumerate(potongan):
        x[j, : len(p)] = torch.from_numpy(p).to(device)
    y_np = np.array([rekaman[i].label for i in indeks]) if label is None else label[indeks]
    return Batch(x=x, lengths=lengths, y=torch.tensor(y_np, dtype=torch.float32, device=device),
                 indeks=list(indeks))


def bobot_kelas_positif(y: np.ndarray) -> float:
    """Rasio invers frekuensi kelas, dihitung dari himpunan latih saja."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def latih(model: nn.Module, rekaman: list[Rekaman], indeks_latih: list[int],
          norm: Normalisasi, epochs: int, device: str = "cuda",
          batch_size: int = 8, lr: float = 3e-4, label: np.ndarray | None = None,
          seed: int = 0, verbose: bool = False) -> list[float]:
    """Latih satu model sampai batas epoch yang sudah dibekukan.

    Batas epoch adalah argumen wajib: tidak ada penghentian dini berdasarkan
    fold uji, karena itu akan membocorkan fold uji ke dalam pemilihan model.
    """
    g = torch.Generator().manual_seed(seed)
    y_latih = (np.array([rekaman[i].label for i in indeks_latih]) if label is None
               else label[indeks_latih])
    pos_weight = torch.tensor(bobot_kelas_positif(y_latih), device=device)
    kriteria = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    n_batch = max(len(indeks_latih) // batch_size, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs * n_batch)

    riwayat = []
    model.train()
    for ep in range(epochs):
        urutan = torch.randperm(len(indeks_latih), generator=g).tolist()
        total, n = 0.0, 0
        for b in range(0, len(urutan), batch_size):
            pilih = [indeks_latih[k] for k in urutan[b : b + batch_size]]
            if len(pilih) < 2:  # BatchNorm/LayerNorm aman, tapi batch 1 tak informatif
                continue
            batch = susun_batch(rekaman, pilih, norm, device, label)
            logit, _ = model(batch.x, batch.lengths)
            rugi = kriteria(logit, batch.y)
            optim.zero_grad(set_to_none=True)
            rugi.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            total += rugi.item() * len(pilih)
            n += len(pilih)
        riwayat.append(total / max(n, 1))
        if verbose and (ep + 1) % max(epochs // 5, 1) == 0:
            print(f"    epoch {ep+1:3d}/{epochs}  rugi={riwayat[-1]:.4f}")
    return riwayat


@torch.no_grad()
def prediksi(model: nn.Module, rekaman: list[Rekaman], indeks: list[int],
             norm: Normalisasi, device: str = "cuda", batch_size: int = 8,
             label: np.ndarray | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Hasilkan logit dan peta bobot atensi per rekaman.

    Peta alpha dipotong sesuai panjang valid tiap rekaman, sehingga tidak ada
    posisi padding yang ikut terbawa ke analisis lokalisasi.
    """
    model.eval()
    logits, alphas = [], []
    for b in range(0, len(indeks), batch_size):
        pilih = list(indeks[b : b + batch_size])
        batch = susun_batch(rekaman, pilih, norm, device, label)
        logit, alpha = model(batch.x, batch.lengths)
        logits.append(logit.cpu().numpy())
        n_patch = (batch.lengths // model.patch_embed.patch_size).tolist()
        for j, n in enumerate(n_patch):
            alphas.append(alpha[j, :n].cpu().numpy())
    return np.concatenate(logits), alphas


def metrik_biner(y: np.ndarray, logit: np.ndarray) -> dict[str, float]:
    """AUC, sensitivitas, dan spesifisitas. Akurasi sengaja tidak dilaporkan.

    Akurasi bergantung pada prior kelas, sedangkan rasio kelas pada penelitian ini
    timpang dan terbalik arah antar basis data (Subbab 3.6.1).
    """
    prob = 1 / (1 + np.exp(-logit))
    hasil: dict[str, float] = {}
    hasil["auc"] = float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan")
    tebak = (prob >= 0.5).astype(int)
    pos, neg = y == 1, y == 0
    hasil["sensitivitas"] = float(tebak[pos].mean()) if pos.any() else float("nan")
    hasil["spesifisitas"] = float((1 - tebak[neg]).mean()) if neg.any() else float("nan")
    return hasil
