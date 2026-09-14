"""
Atribusi Shapley temporal dengan WindowSHAP varian stasioner.

Mengikuti Subbab 2.2.7 dan 3.6.5 naskah proposal. Referensi metodenya: Nayebi,
Tipirneni, Reddy, Foreman, dan Subbian, "WindowSHAP: An efficient framework for
explaining time-series classifiers based on Shapley values", Journal of Biomedical
Informatics, 2023.

Gagasannya: langkah waktu bertetangga digabung menjadi jendela, lalu tiap jendela
diperlakukan sebagai satu pemain dalam permainan koalisi. Ini menurunkan jumlah
pemain dari ribuan langkah waktu menjadi puluhan sampai ratusan jendela, sekaligus
mengurangi pelanggaran asumsi independensi antar pemain — sebab langkah waktu
bertetangga pada sinyal gerak yang mulus berkorelasi sangat kuat.

Keputusan rancangan
-------------------
1. **Jendela dikunci sama dengan grid patch model.** Dengan begitu atribusi
   Shapley dan bobot atensi berada pada grid temporal identik dan dapat
   dibandingkan tanpa penyelarasan ulang (Subbab 2.2.7).

2. **Nilai latar bersifat parameter eksplisit, bukan pilihan tersembunyi.**
   Atribusi Shapley memerlukan acuan untuk kondisi "fitur tidak hadir", dan
   pilihan itu mengubah hasil secara material. Naskah Subbab 3.6.5 menuntut satu
   pilihan ditetapkan beserta satu ablasi alternatifnya, sehingga strategi latar
   di sini dapat dipertukarkan dan selalu tercatat pada hasilnya.

3. **Antarmuka model bersifat umum.** Kelas ini hanya menuntut sebuah pemanggil
   yang menerima tensor `(B, T, C)` beserta panjang valid dan mengembalikan logit
   `(B,)`. Tidak ada ketergantungan pada arsitektur tertentu, sehingga modul ini
   dapat dipakai untuk BiGRU maupun BiMamba-2, dan untuk model apa pun sesudahnya.

4. **Aksioma efisiensi dapat diperiksa.** Jumlah seluruh nilai Shapley harus sama
   dengan selisih keluaran pada masukan utuh dan pada masukan latar penuh. Sifat
   ini merupakan invarian yang dapat diuji, dan `periksa_efisiensi` memakainya
   sebagai uji kewarasan implementasi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import torch

__all__ = ["StrategiLatar", "HasilShapley", "WindowSHAP"]

StrategiLatar = Literal["nol", "median_lokal", "acak_latih"]
_STRATEGI_SAH = ("nol", "median_lokal", "acak_latih")


@dataclass
class HasilShapley:
    """Nilai Shapley per jendela beserta konteks yang diperlukan untuk menafsirkannya."""

    phi: np.ndarray                 # (W,) nilai Shapley per jendela, bertanda
    phi_0: float                    # keluaran pada latar penuh
    keluaran_utuh: float            # keluaran pada masukan utuh
    n_koalisi: int
    strategi_latar: str
    patch_size: int
    r2: float                       # kecocokan regresi berbobot; rendah berarti estimasi kasar
    sisa_efisiensi: float           # |sum(phi) - (keluaran_utuh - phi_0)|
    meta: dict = field(default_factory=dict)

    @property
    def phi_positif(self) -> np.ndarray:
        """Bagian positif, untuk metrik lokalisasi berbasis massa.

        Nilai Shapley bertanda. Metrik seperti Relevance Mass Accuracy pada
        Quantus didefinisikan atas atribusi positif, sehingga pilihan ini harus
        dinyatakan eksplisit saat melaporkan hasil.
        """
        return np.clip(self.phi, 0, None)

    def ringkas(self) -> dict:
        return {
            "n_jendela": len(self.phi),
            "n_koalisi": self.n_koalisi,
            "strategi_latar": self.strategi_latar,
            "phi_0": self.phi_0,
            "keluaran_utuh": self.keluaran_utuh,
            "r2": self.r2,
            "sisa_efisiensi": self.sisa_efisiensi,
            "frac_positif": float((self.phi > 0).mean()),
        }


class WindowSHAP:
    """WindowSHAP stasioner dengan pemecah KernelSHAP berbobot kernel Shapley.

    Parameters
    ----------
    fungsi_model : pemanggil `(x, lengths) -> logit`
        `x` bentuk `(B, T, C)`, `lengths` bentuk `(B,)`, keluaran bentuk `(B,)`.
        Sengaja tidak menuntut kelas model tertentu agar modul ini tetap berguna
        bagi arsitektur yang belum ada saat modul ini ditulis.
    patch_size : lebar jendela dalam satuan sampel, dikunci sama dengan grid patch model
    strategi_latar : cara mengisi jendela yang dianggap tidak hadir
        "nol"          : diisi nol. Setelah normalisasi robust, nol mendekati median
                         himpunan latih, sehingga bermakna "nilai lazim".
        "median_lokal" : diisi median kanal rekaman itu sendiri. Dipakai sebagai
                         ablasi untuk menguji ketergantungan hasil pada acuan global.
        "acak_latih"   : diisi cuplikan acak dari `contoh_latar`, mendekati
                         marginalisasi observasional.
    n_koalisi : anggaran jumlah koalisi. Bila None, dipakai `2 * W + 2048`
        mengikuti kelaziman implementasi KernelSHAP.
    batch : jumlah koalisi yang dievaluasi sekaligus pada satu forward pass
    """

    def __init__(
        self,
        fungsi_model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        patch_size: int,
        strategi_latar: StrategiLatar = "nol",
        n_koalisi: int | None = None,
        batch: int = 64,
        device: str = "cuda",
        contoh_latar: np.ndarray | None = None,
        seed: int = 0,
    ):
        if strategi_latar not in _STRATEGI_SAH:
            raise ValueError(f"strategi_latar harus salah satu dari {_STRATEGI_SAH}, bukan {strategi_latar!r}")
        if strategi_latar == "acak_latih" and contoh_latar is None:
            raise ValueError("strategi 'acak_latih' memerlukan contoh_latar berbentuk (N, C)")

        self.fungsi_model = fungsi_model
        self.patch_size = patch_size
        self.strategi_latar = strategi_latar
        self.n_koalisi = n_koalisi
        self.batch = batch
        self.device = device
        self.contoh_latar = None if contoh_latar is None else np.asarray(contoh_latar, dtype=np.float32)
        self.seed = seed

    # ------------------------------------------------------------------ latar

    def _nilai_latar(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Bentuk tensor latar sebentuk `x`, sesuai strategi yang dipilih."""
        if self.strategi_latar == "nol":
            return np.zeros_like(x)
        if self.strategi_latar == "median_lokal":
            return np.broadcast_to(np.median(x, axis=0, keepdims=True), x.shape).copy()
        # acak_latih
        idx = rng.integers(0, len(self.contoh_latar), size=len(x))
        return self.contoh_latar[idx]

    # -------------------------------------------------------------- koalisi

    @staticmethod
    def _bobot_ukuran(W: int) -> np.ndarray:
        """Peluang ukuran koalisi menurut kernel Shapley.

        Bobot kernel Shapley untuk satu koalisi berukuran s adalah
        `(W-1) / (C(W,s) * s * (W-s))`. Dijumlahkan atas seluruh koalisi berukuran
        s, totalnya sebanding dengan `1 / (s * (W-s))`. Mengambil cuplikan ukuran
        dari distribusi ini, lalu memilih himpunan acak berukuran itu, memberi
        penaksir tak bias tanpa perlu menimbang ulang tiap koalisi.
        """
        s = np.arange(1, W)
        p = 1.0 / (s * (W - s))
        return p / p.sum()

    def _cuplik_koalisi(self, W: int, n: int, rng: np.random.Generator) -> np.ndarray:
        """Matriks koalisi `(n, W)` bernilai 0/1, memakai cuplikan berpasangan.

        Tiap koalisi disertai komplemennya. Pemasangan ini menurunkan variansi
        penaksir secara nyata pada anggaran koalisi yang sama, dan merupakan
        praktik lazim pada implementasi KernelSHAP.
        """
        p_ukuran = self._bobot_ukuran(W)
        n_pasang = max(n // 2, 1)
        z = np.zeros((n_pasang * 2, W), dtype=np.float32)
        ukuran = rng.choice(np.arange(1, W), size=n_pasang, p=p_ukuran)
        for i, s in enumerate(ukuran):
            pilih = rng.choice(W, size=int(s), replace=False)
            z[2 * i, pilih] = 1.0
            z[2 * i + 1] = 1.0 - z[2 * i]
        return z

    # ---------------------------------------------------------------- inti

    @torch.no_grad()
    def _evaluasi(self, x: np.ndarray, latar: np.ndarray, z: np.ndarray,
                  panjang: int, T_penuh: int) -> np.ndarray:
        """Jalankan model pada seluruh koalisi, dikelompokkan menjadi batch."""
        W = z.shape[1]
        hasil = np.empty(len(z), dtype=np.float64)
        for awal in range(0, len(z), self.batch):
            blok = z[awal : awal + self.batch]
            xb = np.repeat(latar[None, ...], len(blok), axis=0)
            for j, baris in enumerate(blok):
                for w in np.flatnonzero(baris):
                    a = w * self.patch_size
                    b = min(a + self.patch_size, panjang)
                    xb[j, a:b] = x[a:b]
            # Padding sampai T_penuh agar bentuk masukan model tetap konsisten.
            if T_penuh > xb.shape[1]:
                pad = np.zeros((len(blok), T_penuh - xb.shape[1], xb.shape[2]), dtype=xb.dtype)
                xb = np.concatenate([xb, pad], axis=1)
            xt = torch.from_numpy(xb).to(self.device)
            lt = torch.full((len(blok),), panjang, dtype=torch.long, device=self.device)
            hasil[awal : awal + len(blok)] = self.fungsi_model(xt, lt).detach().cpu().numpy()
        return hasil

    def explain(self, kanal: np.ndarray, panjang: int | None = None) -> HasilShapley:
        """Hitung nilai Shapley per jendela untuk satu rekaman.

        Parameters
        ----------
        kanal : (T, C) sinyal yang SUDAH dinormalisasi dengan statistik fold latih
        panjang : panjang valid; bila None dipakai seluruh baris `kanal`

        Returns
        -------
        HasilShapley
        """
        x = np.asarray(kanal, dtype=np.float32)
        panjang = int(panjang if panjang is not None else len(x))
        W = panjang // self.patch_size
        if W < 2:
            raise ValueError(f"jendela terlalu sedikit (W={W}); rekaman terlalu pendek untuk patch {self.patch_size}")

        rng = np.random.default_rng(self.seed)
        x_potong = x[:panjang]
        latar = self._nilai_latar(x_potong, rng)

        n = self.n_koalisi if self.n_koalisi is not None else 2 * W + 2048
        z = self._cuplik_koalisi(W, n, rng)

        # Dua jangkar wajib: koalisi kosong memberi phi_0, koalisi penuh memberi
        # sisi kanan kendala efisiensi.
        z_jangkar = np.concatenate([np.zeros((1, W), np.float32), np.ones((1, W), np.float32)])
        y_jangkar = self._evaluasi(x_potong, latar, z_jangkar, panjang, len(x))
        phi_0, keluaran_utuh = float(y_jangkar[0]), float(y_jangkar[1])

        y = self._evaluasi(x_potong, latar, z, panjang, len(x))

        phi, r2 = self._pecahkan(z, y, phi_0, keluaran_utuh)
        sisa = abs(float(phi.sum()) - (keluaran_utuh - phi_0))

        return HasilShapley(
            phi=phi, phi_0=phi_0, keluaran_utuh=keluaran_utuh, n_koalisi=len(z),
            strategi_latar=self.strategi_latar, patch_size=self.patch_size,
            r2=r2, sisa_efisiensi=sisa,
            meta={"panjang": panjang, "n_jendela": W, "seed": self.seed},
        )

    @staticmethod
    def _pecahkan(z: np.ndarray, y: np.ndarray, phi_0: float,
                  keluaran_utuh: float) -> tuple[np.ndarray, float]:
        """Regresi kuadrat terkecil dengan kendala efisiensi ditegakkan tepat.

        Kendala `sum(phi) = keluaran_utuh - phi_0` disubstitusikan dengan
        mengeliminasi satu peubah, sehingga aksioma efisiensi terpenuhi secara
        aljabar dan bukan hanya secara hampiran. Cara ini mengikuti penurunan
        KernelSHAP pada Lundberg dan Lee (2017).
        """
        W = z.shape[1]
        total = keluaran_utuh - phi_0
        y_pusat = y - phi_0

        # phi_terakhir = total - sum(phi_1..W-1)  =>  y ~ sum_i (z_i - z_W) phi_i + z_W * total
        A = z[:, :-1] - z[:, -1:]
        b = y_pusat - z[:, -1] * total

        # Bobot kernel sudah terserap oleh cara pengambilan cuplikan ukuran
        # koalisi, sehingga di sini cukup kuadrat terkecil biasa. Regularisasi
        # ridge sangat kecil menjaga kestabilan bila kolomnya hampir kolinear.
        lam = 1e-6 * A.shape[0]
        AtA = A.T @ A + lam * np.eye(W - 1)
        Atb = A.T @ b
        phi_kurang = np.linalg.solve(AtA, Atb)

        phi = np.empty(W, dtype=np.float64)
        phi[:-1] = phi_kurang
        phi[-1] = total - phi_kurang.sum()

        prediksi = A @ phi_kurang + z[:, -1] * total
        ss_sisa = float(((b - A @ phi_kurang) ** 2).sum())
        ss_total = float(((y_pusat - y_pusat.mean()) ** 2).sum())
        r2 = 1.0 - ss_sisa / ss_total if ss_total > 0 else float("nan")
        return phi, r2

    # ------------------------------------------------------------ kewarasan

    def periksa_efisiensi(self, kanal: np.ndarray, panjang: int | None = None,
                          toleransi: float = 1e-6) -> dict:
        """Uji aksioma efisiensi sebagai invarian implementasi.

        Jumlah nilai Shapley harus sama dengan selisih keluaran pada masukan utuh
        dan pada latar penuh. Kendala itu ditegakkan secara aljabar oleh
        `_pecahkan`, sehingga sisa yang besar menandakan kekeliruan implementasi,
        bukan sekadar keterbatasan cuplikan.
        """
        hasil = self.explain(kanal, panjang)
        return {
            "sum_phi": float(hasil.phi.sum()),
            "selisih_keluaran": hasil.keluaran_utuh - hasil.phi_0,
            "sisa": hasil.sisa_efisiensi,
            "lolos": hasil.sisa_efisiensi <= toleransi,
            "r2": hasil.r2,
        }
