"""
Arsitektur model untuk lokalisasi temporal gangguan motorik pada tulisan tangan daring.

Mengikuti Subbab 3.5 naskah proposal (naskah/sempro-skripsi.md):

    Sinyal → PatchEmbedding → Encoder (dipertukarkan) → AttentionPooling → Head

Dua encoder tersedia dan dapat dipertukarkan tanpa mengubah bagian lain:
BiGRU sebagai baseline, BiMamba-2 sebagai yang diusulkan. Bottleneck (attention
pooling) dan kepala prediksi dikunci identik untuk keduanya, sehingga perbedaan
yang teramati dapat diatribusikan pada encoder.

Catatan implementasi penting
----------------------------
1. `Mamba2` mengunci dimensi masukan = dimensi keluaran = `d_model`, berbeda dari
   `nn.GRU` yang `input_size` dan `hidden_size`-nya boleh berbeda. Karena itu
   BiMamba2Encoder memproyeksikan d_model → d_dir terlebih dahulu. Proyeksi ini
   adalah versi eksplisit dari apa yang GRU lakukan implisit lewat W_ih, dan
   menyamakan lebar state rekuren per arah antara kedua arsitektur.

2. `causal_conv1d` mensyaratkan `d_in_proj` kelipatan 8, dengan
   `d_in_proj = 2*d_inner + 2*ngroups*d_state + nheads`. Kendala ini tidak
   terdokumentasi dan hanya muncul sebagai RuntimeError saat forward. Fungsi
   `check_mamba2_config` memvalidasinya di awal agar gagal cepat dan jelas.

3. Arah mundur pada BiMamba-2 WAJIB membalik hanya bagian valid tiap sampel.
   `torch.flip` atas tensor terpadding menaruh padding di awal urutan, dan
   rekurensi SSM akan menyebarkan kontaminasi itu ke seluruh posisi valid
   sesudahnya. Arah maju tidak memerlukan penanganan khusus karena kausalitas
   melindunginya. Lihat uji regresi di notebooks/03_model.ipynb.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mamba_ssm.modules.mamba2 import Mamba2

try:
    # Mamba-3 di-vendor ke .venv dari rilis v2.3.2.post1; jalur SISO hanya
    # bergantung pada Triton sehingga tidak menuntut ekstensi terkompilasi.
    # Dibungkus try agar modul tetap dapat diimpor bila vendoring dibatalkan.
    from mamba_ssm.modules.mamba3 import Mamba3
except ImportError:  # pragma: no cover
    Mamba3 = None

__all__ = [
    "check_mamba2_config",
    "lengths_to_mask",
    "PatchEmbedding",
    "BiGRUEncoder",
    "BiMamba2Encoder",
    "BiMamba3Encoder",
    "AttentionPooling",
    "MeanPooling",
    "PDClassifier",
]


def check_mamba2_config(d_model: int, d_state: int, headdim: int, expand: int = 2, ngroups: int = 1) -> int:
    """Validasi kendala `causal_conv1d` sebelum modul dibangun.

    Mengembalikan `d_in_proj` bila sah, melempar ValueError bila tidak.
    """
    d_inner = expand * d_model
    if d_inner % headdim != 0:
        raise ValueError(f"d_inner={d_inner} tidak habis dibagi headdim={headdim}")
    nheads = d_inner // headdim
    d_in_proj = 2 * d_inner + 2 * ngroups * d_state + nheads
    if d_in_proj % 8 != 0:
        raise ValueError(
            f"d_in_proj={d_in_proj} bukan kelipatan 8 (sisa {d_in_proj % 8}). "
            f"causal_conv1d akan gagal saat forward. "
            f"Konfigurasi: d_model={d_model}, d_state={d_state}, headdim={headdim}, "
            f"expand={expand}, ngroups={ngroups}. "
            f"Coba headdim lain agar nheads berubah, atau sesuaikan d_state."
        )
    return d_in_proj


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Mask boolean posisi valid, bentuk (B, max_len). True = valid."""
    device = lengths.device
    return torch.arange(max_len, device=device)[None, :] < lengths[:, None]


class PatchEmbedding(nn.Module):
    """Potong urutan panjang menjadi patch dan proyeksikan ke dimensi model.

    Conv1d dengan kernel dan stride sama dengan P, sehingga patch tidak tumpang
    tindih dan grid temporalnya sama persis dengan grid yang nanti dipakai
    WindowSHAP (Subbab 2.2.7 dan 3.6.5).
    """

    def __init__(self, in_channels: int, d_model: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, C) sinyal terpadding
        lengths : (B,) panjang valid tiap sampel dalam satuan titik data mentah

        Returns
        -------
        h : (B, T_patch, d_model)
        patch_lengths : (B,) panjang valid dalam satuan patch
        """
        h = self.proj(x.transpose(1, 2)).transpose(1, 2)
        # Patch dianggap valid hanya bila seluruh titik penyusunnya valid.
        patch_lengths = torch.div(lengths, self.patch_size, rounding_mode="floor")
        patch_lengths = patch_lengths.clamp(min=1, max=h.size(1))
        return h, patch_lengths


class BiGRUEncoder(nn.Module):
    """Encoder baseline: Bidirectional GRU bertumpuk.

    Memakai pack_padded_sequence sehingga padding tidak pernah ikut diproses.
    Wrapper (LayerNorm + residual) dibuat identik dengan BiMamba2Encoder agar
    perbandingan antar arsitektur hanya menyentuh inti rekurennya.
    """

    def __init__(self, d_model: int = 128, d_dir: int = 64, n_layers: int = 3):
        super().__init__()
        if 2 * d_dir != d_model:
            raise ValueError(f"2*d_dir ({2 * d_dir}) harus sama dengan d_model ({d_model})")
        self.layers = nn.ModuleList(
            nn.GRU(d_model, d_dir, batch_first=True, bidirectional=True) for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        max_len = x.size(1)
        lens_cpu = lengths.detach().cpu()
        for gru, norm in zip(self.layers, self.norms):
            packed = pack_padded_sequence(x, lens_cpu, batch_first=True, enforce_sorted=False)
            out, _ = gru(packed)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=max_len)
            x = norm(out + x)
        return x


class BiMamba2Encoder(nn.Module):
    """Encoder yang diusulkan: Bidirectional Mamba-2 bertumpuk.

    Tiap layer memproyeksikan d_model → d_dir, menjalankan dua blok Mamba2
    (maju dan mundur) pada lebar d_dir, lalu menggabungkan keduanya kembali ke
    d_model. Wrapper (LayerNorm + residual) identik dengan BiGRUEncoder.

    Arah mundur membalik hanya bagian valid tiap sampel. Lihat catatan nomor 3
    pada docstring modul.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_dir: int = 64,
        n_layers: int = 3,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 16,
    ):
        super().__init__()
        if 2 * d_dir != d_model:
            raise ValueError(f"2*d_dir ({2 * d_dir}) harus sama dengan d_model ({d_model})")
        check_mamba2_config(d_model=d_dir, d_state=d_state, headdim=headdim, expand=expand)

        mamba_kwargs = dict(d_model=d_dir, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
        self.down = nn.ModuleList(nn.Linear(d_model, d_dir) for _ in range(n_layers))
        self.fwd = nn.ModuleList(Mamba2(**mamba_kwargs) for _ in range(n_layers))
        self.bwd = nn.ModuleList(Mamba2(**mamba_kwargs) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))

    @staticmethod
    def _flip_valid(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Balik hanya bagian valid tiap sampel; posisi padding tetap nol.

        Ini yang mencegah padding berpindah ke awal urutan dan mencemari
        seluruh posisi valid lewat rekurensi SSM.
        """
        out = torch.zeros_like(x)
        for i, length in enumerate(lengths.tolist()):
            out[i, :length] = torch.flip(x[i, :length], dims=[0])
        return out

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        for down, fwd, bwd, norm in zip(self.down, self.fwd, self.bwd, self.norms):
            h = down(x)
            h_fwd = fwd(h)
            h_bwd = self._flip_valid(bwd(self._flip_valid(h, lengths)), lengths)
            x = norm(torch.cat([h_fwd, h_bwd], dim=-1) + x)
        return x


class BiMamba3Encoder(nn.Module):
    """Bidirectional Mamba-3, rakitan identik dengan BiMamba2Encoder.

    Hanya inti rekurennya yang berbeda; wrapper (proyeksi turun, concat,
    LayerNorm, residual) dan penanganan arah mundur sama persis, sesuai prinsip
    bottleneck dikunci identik pada Subbab 3.5. Perbedaan hasil karena itu hanya
    dapat berasal dari inti rekurennya.

    Perbedaan Mamba-3 dari Mamba-2 yang relevan di sini:

    1. **State bernilai kompleks.** Mamba-2 hanya dapat merepresentasikan
       peluruhan; Mamba-3 dapat merepresentasikan rotasi, yaitu osilasi. Ini
       yang membuatnya menarik bagi masalah ini, sebab tanda patologi yang
       dilokalisasi berupa osilasi pada pita 3,5 sampai 7,5 Hz.
    2. **Tanpa konvolusi kausal.** Mamba-3 tidak memakai `d_conv`, sehingga
       `causal_conv1d` tidak dibutuhkan sama sekali.
    3. **Tanpa kendala kelipatan delapan.** Kendala tak terdokumentasi pada
       Mamba-2 (`d_in_proj` wajib kelipatan 8) tidak muncul pada Mamba-3;
       keempat konfigurasi yang diuji berjalan tanpa galat.

    Dipakai varian SISO (`is_mimo=False`). Varian MIMO menuntut pustaka
    `tilelang` yang tidak di-vendor.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_dir: int = 64,
        n_layers: int = 3,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 16,
    ):
        super().__init__()
        if Mamba3 is None:
            raise ImportError(
                "Mamba3 tidak tersedia. Vendor mamba_ssm/modules/mamba3.py dan "
                "mamba_ssm/ops/triton/mamba3/ dari rilis v2.3.2.post1 ke .venv."
            )
        if 2 * d_dir != d_model:
            raise ValueError(f"2*d_dir ({2 * d_dir}) harus sama dengan d_model ({d_model})")

        kwargs = dict(d_model=d_dir, d_state=d_state, expand=expand,
                      headdim=headdim, is_mimo=False)
        self.down = nn.ModuleList(nn.Linear(d_model, d_dir) for _ in range(n_layers))
        self.fwd = nn.ModuleList(Mamba3(**kwargs) for _ in range(n_layers))
        self.bwd = nn.ModuleList(Mamba3(**kwargs) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))

    # Dipakai ulang, bukan disalin: bug arah mundur pernah terkonfirmasi nyata
    # (selisih 1,35e-02) dan menduplikasi logikanya mengundangnya kembali.
    _flip_valid = staticmethod(BiMamba2Encoder._flip_valid)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        for down, fwd, bwd, norm in zip(self.down, self.fwd, self.bwd, self.norms):
            h = down(x)
            h_fwd = fwd(h)
            h_bwd = self._flip_valid(bwd(self._flip_valid(h, lengths)), lengths)
            x = norm(torch.cat([h_fwd, h_bwd], dim=-1) + x)
        return x


class AttentionPooling(nn.Module):
    """Bottleneck attention pooling, identik untuk kedua encoder.

        e_i   = w · tanh(W · h_i)
        e_i   = nilai minimum dtype pada posisi padding
        alpha = softmax(e)
        z     = sum alpha_i · h_i

    Masking dilakukan SEBELUM softmax. Bila setelahnya, bobot sudah terdistribusi
    ke posisi kosong dan peta lokalisasi menjadi tidak sahih. Kesalahan semacam
    itu tidak terdeteksi lewat metrik akurasi (Subbab 3.5).

    Memakai torch.finfo(dtype).min alih-alih -inf agar aman secara numerik.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.W = nn.Linear(d_model, d_model)
        self.w = nn.Linear(d_model, 1, bias=False)

    def forward(self, h: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z : (B, d_model) representasi teragregasi
        alpha : (B, T) peta bobot atensi per segmen waktu
        """
        e = self.w(torch.tanh(self.W(h))).squeeze(-1)
        valid = lengths_to_mask(lengths, h.size(1))
        e = e.masked_fill(~valid, torch.finfo(e.dtype).min)
        alpha = torch.softmax(e, dim=-1)
        z = (alpha.unsqueeze(-1) * h).sum(dim=1)
        return z, alpha


class MeanPooling(nn.Module):
    """Agregasi rata-rata berbobot seragam, sebagai ablasi attention pooling.

    Dipakai pada uji kewarasan Subbab 3.6.8: apabila performanya setara dengan
    attention pooling, maka attention pooling tidak memberi keuntungan prediktif
    dan peta bobotnya merupakan produk sampingan gratis. Hasil itu dilaporkan apa
    adanya, karena menyentuh langsung nilai guna peta yang menjadi keluaran inti
    penelitian ini.

    Mengembalikan bobot seragam pada posisi valid agar antarmukanya identik
    dengan AttentionPooling dan dapat dipertukarkan tanpa mengubah pemanggil.
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.d_model = d_model

    def forward(self, h: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = lengths_to_mask(lengths, h.size(1)).to(h.dtype)
        alpha = valid / valid.sum(dim=1, keepdim=True).clamp(min=1)
        z = (alpha.unsqueeze(-1) * h).sum(dim=1)
        return z, alpha


class PDClassifier(nn.Module):
    """Model utuh: patch embedding, encoder, attention pooling, kepala prediksi.

    Menghasilkan logit prediksi sekaligus peta bobot atensi per segmen waktu,
    yang menjadi keluaran inti penelitian ini (Rumusan Masalah 1).
    """

    def __init__(
        self,
        in_channels: int = 6,
        d_model: int = 128,
        patch_size: int = 20,
        n_layers: int = 3,
        dropout: float = 0.3,
        encoder: str = "mamba2",
        d_dir: int = 64,
        pooling: str = "attention",
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoder_name = encoder
        self.pooling_name = pooling
        self.patch_embed = PatchEmbedding(in_channels, d_model, patch_size)

        if encoder == "mamba2":
            self.encoder = BiMamba2Encoder(d_model, d_dir, n_layers, **encoder_kwargs)
        elif encoder == "mamba3":
            self.encoder = BiMamba3Encoder(d_model, d_dir, n_layers, **encoder_kwargs)
        elif encoder == "gru":
            self.encoder = BiGRUEncoder(d_model, d_dir, n_layers, **encoder_kwargs)
        else:
            raise ValueError(
                f"encoder harus 'mamba2', 'mamba3', atau 'gru', bukan {encoder!r}")

        if pooling == "attention":
            self.pool = AttentionPooling(d_model)
        elif pooling == "mean":
            self.pool = MeanPooling(d_model)
        else:
            raise ValueError(f"pooling harus 'attention' atau 'mean', bukan {pooling!r}")
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor,
                alpha_pengganti: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, C) sinyal terpadding
        lengths : (B,) panjang valid dalam satuan titik data mentah
        alpha_pengganti : (B, T_patch), opsional. Bila diberikan, bobot ini dipakai
            menggantikan bobot yang dihasilkan kepala agregasi, sementara seluruh
            bagian model lain tetap sama.

            Diperlukan oleh uji permutasi bobot pada Subbab 3.6.8, yang mengikuti
            logika Jain dan Wallace [8]: bila prediksi tidak berubah berarti ketika
            bobotnya diacak, maka bobot itu tidak menentukan keluaran. Menyuntikkan
            bobot dari luar adalah satu-satunya cara menguji hal itu tanpa melatih
            ulang model, sehingga yang diuji benar-benar model yang sama.

        Returns
        -------
        logits : (B,) logit biner, belum melewati sigmoid
        alpha : (B, T_patch) bobot yang BENAR-BENAR dipakai untuk agregasi
        """
        h, patch_lengths = self.patch_embed(x, lengths)
        h = self.encoder(h, patch_lengths)
        z, alpha = self.pool(h, patch_lengths)
        if alpha_pengganti is not None:
            alpha = alpha_pengganti[:, : h.size(1)]
            z = (alpha.unsqueeze(-1) * h).sum(dim=1)
        return self.head(z).squeeze(-1), alpha

    def count_parameters(self) -> dict[str, int]:
        """Jumlah parameter per komponen, untuk penyetaraan antar arsitektur."""
        n = lambda m: sum(p.numel() for p in m.parameters())
        return {
            "patch_embed": n(self.patch_embed),
            "encoder": n(self.encoder),
            "pool": n(self.pool),
            "head": n(self.head),
            "total": n(self),
        }
