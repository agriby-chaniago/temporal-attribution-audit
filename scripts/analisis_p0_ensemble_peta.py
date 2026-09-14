"""P0 — peta ensemble antar seed dan kurva kestabilan. Tanpa GPU.

Jalankan: python3 scripts/analisis_p0_ensemble_peta.py

Pertanyaan yang dijawab
-----------------------
Benang merah penelitian ini adalah **akurasi hampir tak bergerak, peta runtuh**.
Salah satu dari empat perturbasi pada tabel keruntuhan itu adalah "ganti seed
saja", yang menggeser peta 0,2354 poin korelasi sementara akurasinya bergeser
0,0344. Berkas ini menanyakan lanjutannya, yang belum pernah ditanyakan:

    keruntuhan itu **variansi** yang dapat diredam, atau **bias** yang tidak?

Kalau variansi, merata-ratakan peta antar seed memulihkannya, dan obatnya murah:
jalankan beberapa seed lalu rata-ratakan. Kalau bias, tidak ada jumlah seed yang
menolong, dan sebabnya struktural.

Seluruh datanya sudah ada di `results/`. Tidak ada model yang dilatih di sini.

Kenapa agregasinya rata-rata PERINGKAT
--------------------------------------
Vonisnya `rho` Spearman, yang hanya melihat urutan patch. Untuk satu seed,
transformasi monoton apa pun pada alpha memberi rho identik — jadi normalisasi
tidak berpengaruh. Tetapi ensemble menggabungkan **sebelum** korelasi dihitung,
dan di situ pilihannya menentukan:

- rata-rata peringkat merata-ratakan langsung besaran yang nanti diukur;
- rata-rata aritmetik merata-ratakan besaran lalu membuang semuanya kecuali
  urutan, sehingga besaran memengaruhi hasil hanya lewat cara ia merusak urutan.

Ada pula alasan empiris. Gini alpha membentang 0,26 sampai 0,97 antar rekaman;
satu seed yang petanya berupa paku tunggal akan mendominasi rata-rata aritmetik
dan ensemble berubah menjadi "suara paling keras", bukan konsensus.

Tiga agregasi lain tetap dilaporkan sebagai sensitivitas. Bila keempatnya searah,
kesimpulannya kokoh; bila berbeda, itu sendiri temuan tentang kerapuhan peta.

Status: EKSPLORATORI. Analisis pra-registrasi tidak diganti, ambang tidak diubah,
aturan keputusan tidak disentuh.
"""

from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR / "src"))

from marker import (ke_grid_patch, mask_pena_melayang,  # noqa: E402
                    penanda_cepat)
from preprocessing import muat_cache  # noqa: E402

HASIL = AKAR / "results"
FS = 100
P = 7                       # dibekukan S2
MAKS_SUBSET = 200           # batas penarikan subhimpunan pada kurva-k
RNG = np.random.default_rng(20260904)

# Nilai jangkar untuk memeriksa bahwa penanda UCI dihitung ulang dengan benar.
# Sumber: results/s7_tiga_peta.csv, kolom "alpha vs penanda", tugas STCP.
JANGKAR_STCP = {"mamba2": 0.21592917, "gru": 0.14266051, "mamba3": -0.34696989}

# Analisis primer pra-registrasi: phi <-> penanda cepat, STCP, BiMamba-2, P=7.
# Hasilnya +0,011177 terhadap ambang 0,069410, p 0,58468 (results/s7_analisis_primer.csv).
AMBANG = 0.0694103145939695
N_PERMUTASI = 5000


# --------------------------------------------------------------- agregasi peta
def ens_peringkat(peta: list[np.ndarray]) -> np.ndarray:
    """Rata-rata peringkat antar seed; nilai seri memakai peringkat rata-rata."""
    return np.mean([rankdata(p) for p in peta], axis=0)


def ens_zskor(peta: list[np.ndarray]) -> np.ndarray:
    """Z-score per rekaman lalu rata-rata aritmetik."""
    z = []
    for p in peta:
        s = np.std(p)
        z.append((p - np.mean(p)) / s if s > 0 else np.zeros_like(p))
    return np.mean(z, axis=0)


def ens_rerata(peta: list[np.ndarray]) -> np.ndarray:
    """Rata-rata aritmetik mentah — yang naif, dilaporkan agar bedanya terlihat."""
    return np.mean(peta, axis=0)


AGREGASI = {"peringkat": ens_peringkat, "zskor": ens_zskor, "rerata": ens_rerata}


def korelasi_rekaman(nilai, acuan) -> float:
    """Korelasi peringkat pada satu rekaman. Persis nb 09: buang patch tanpa
    acuan sah, tuntut sekurangnya empat patch, tolak ragam nol."""
    n = min(len(nilai), len(acuan))
    a, b = np.asarray(nilai[:n], float), np.asarray(acuan[:n], float)
    sah = ~np.isnan(b)
    if sah.sum() < 4 or np.std(a[sah]) == 0 or np.std(b[sah]) == 0:
        return np.nan
    return float(spearmanr(a[sah], b[sah]).statistic)


# ------------------------------------------------------------------ pemuat data
def muat_uci() -> tuple[dict, list[int], dict]:
    """Kembalikan {(arm, seed): {idx: {alpha, phi}}}, daftar seed, dan penanda per idx.

    Kunci idx pada s7_artefak adalah posisi di dalam daftar `muat_cache`; itu
    sudah diperiksa (207 rekaman, seluruh kunci (arm, seed) berhimpunan sama).
    """
    art = pickle.load(open(HASIL / "s7_artefak.pkl", "rb"))
    seeds = sorted({k[1] for k in art})
    arms = sorted({k[0] for k in art})

    dasar = set(art[(arms[0], seeds[0])])
    for k in art:
        assert set(art[k]) == dasar, f"himpunan indeks berbeda pada {k}"

    rek = muat_cache(AKAR / "data" / "cache" / "uci395_fs100.npz")
    penanda = {}
    for idx in sorted(dasar):
        r = rek[idx]
        m = penanda_cepat(r.xy, FS, r.tugas != 2)
        w = mask_pena_melayang(r.tekanan_mentah) if r.tugas == 2 else None
        penanda[idx] = {"peta": ke_grid_patch(m, P, bobot=w),
                        "tugas": int(r.tugas), "subjek": r.subjek,
                        "label": int(r.label)}
    return art, seeds, penanda


def muat_newhandpd() -> tuple[dict, list[int], dict]:
    """Bentuk ulang daftar entri S5 menjadi bentuk yang sama dengan UCI."""
    entri = pickle.load(open(HASIL / "s5_artefak.pkl", "rb"))
    art, penanda = {}, {}
    for e in entri:
        isi = {}
        for p in e["peta"]:
            isi[int(p["idx"])] = {"alpha": np.asarray(p["alpha"], float),
                                  "phi": np.asarray(p["phi"], float)}
            penanda.setdefault(int(p["idx"]), {
                "peta": np.asarray(p["penanda"], float), "tugas": -1,
                "subjek": str(p["subjek"]), "label": int(p["label"])})
        art[(e["arsitektur"], int(e["seed"]))] = isi
    seeds = sorted({k[1] for k in art})
    return art, seeds, penanda


# ------------------------------------------------------------------- keluaran 1
def kesetiaan(art, seeds, kohort) -> list[dict]:
    """rho(alpha <-> phi): ensemble dibandingkan terhadap seed tunggal."""
    arms = sorted({k[0] for k in art})
    baris = []
    for arm in arms:
        idxs = sorted(art[(arm, seeds[0])])
        satu = [np.nanmedian([korelasi_rekaman(art[(arm, s)][i]["alpha"],
                                               art[(arm, s)][i]["phi"]) for i in idxs])
                for s in seeds]
        for nama, fn in AGREGASI.items():
            rho = []
            for i in idxs:
                a = fn([art[(arm, s)][i]["alpha"] for s in seeds])
                p = fn([art[(arm, s)][i]["phi"] for s in seeds])
                rho.append(korelasi_rekaman(a, p))
            ens = float(np.nanmedian(rho))
            baris.append({"kohort": kohort, "arsitektur": arm, "agregasi": nama,
                          "rho_satu_seed": float(np.mean(satu)),
                          "rho_ensemble": ens,
                          "delta": ens - float(np.mean(satu)),
                          "n_seed": len(seeds), "n_rekaman": len(idxs)})
        # Sensitivitas 3: alpha diensemble, phi tetap per seed.
        a_ens = {i: ens_peringkat([art[(arm, s)][i]["alpha"] for s in seeds]) for i in idxs}
        rho = [np.nanmedian([korelasi_rekaman(a_ens[i], art[(arm, s)][i]["phi"])
                             for i in idxs]) for s in seeds]
        baris.append({"kohort": kohort, "arsitektur": arm,
                      "agregasi": "peringkat_alpha_saja",
                      "rho_satu_seed": float(np.mean(satu)),
                      "rho_ensemble": float(np.mean(rho)),
                      "delta": float(np.mean(rho)) - float(np.mean(satu)),
                      "n_seed": len(seeds), "n_rekaman": len(idxs)})
    return baris


# ------------------------------------------------------------------- keluaran 2
def _rho_subjek(peta_idx, penanda, idx_pakai) -> dict:
    """rho per rekaman lalu dirata-ratakan per subjek. Persis nb 09."""
    per = {}
    for i in idx_pakai:
        rho = korelasi_rekaman(peta_idx[i], penanda[i]["peta"])
        if np.isfinite(rho):
            per.setdefault(penanda[i]["subjek"], []).append(rho)
    return {s: float(np.mean(v)) for s, v in per.items()}


def _selisih_kelompok(rho_subjek, penanda, idx_pakai):
    """MEDIAN(PD) - MEDIAN(HC), bukan mean. Persis `selisih_kelompok` nb 09."""
    lab = {penanda[i]["subjek"]: penanda[i]["label"] for i in idx_pakai}
    pd_ = [v for s, v in rho_subjek.items() if lab.get(s) == 1]
    hc = [v for s, v in rho_subjek.items() if lab.get(s) == 0]
    if not pd_ or not hc:
        return np.nan, len(pd_), len(hc)
    return float(np.median(pd_) - np.median(hc)), len(pd_), len(hc)


def uji_permutasi(pd_, hc, n=N_PERMUTASI, seed=0):
    """Uji permutasi label kelompok, tingkat subjek. Persis nb 09."""
    rng = np.random.default_rng(seed)
    semua = np.array(pd_ + hc); n_pd = len(pd_)
    obs = np.median(semua[:n_pd]) - np.median(semua[n_pd:])
    lebih = sum(abs(np.median((a := rng.permutation(semua))[:n_pd]) - np.median(a[n_pd:]))
                >= abs(obs) - 1e-12 for _ in range(n))
    return float(obs), (lebih + 1) / (n + 1)


def keselarasan(art, seeds, penanda, kohort, hanya_tugas=None, permutasi=False) -> list[dict]:
    """Keselarasan alpha terhadap penanda motorik, tingkat subjek, selisih PD-HC.

    Urutan agregasinya menentukan angkanya, dan harus sama dengan nb 09:
    korelasi per rekaman, dirata-ratakan atas rekaman milik subjek yang sama,
    lalu atas seed, baru selisih **median** antar kelompok diambil. Menukar
    urutannya atau memakai mean memberi angka yang berbeda jauh.
    """
    arms = sorted({k[0] for k in art})
    idx_pakai = [i for i in sorted(penanda)
                 if hanya_tugas is None or penanda[i]["tugas"] == hanya_tugas]
    baris = []
    for arm, besaran in itertools.product(arms, ["alpha", "phi"]):
        # Acuan: seed tunggal, digabung menurut urutan nb 09.
        kumpul, med = {}, []
        for s in seeds:
            peta = {i: art[(arm, s)][i][besaran] for i in idx_pakai}
            for subj, v in _rho_subjek(peta, penanda, idx_pakai).items():
                kumpul.setdefault(subj, []).append(v)
            med.append(np.nanmedian([korelasi_rekaman(peta[i], penanda[i]["peta"])
                                     for i in idx_pakai]))
        rs = {s: float(np.mean(v)) for s, v in kumpul.items()}
        selisih, n_pd, n_hc = _selisih_kelompok(rs, penanda, idx_pakai)
        baris.append({"kohort": kohort, "arsitektur": arm, "besaran": besaran,
                      "agregasi": "satu_seed",
                      "tugas": "STCP" if hanya_tugas == 2 else "semua",
                      "rho_median": float(np.nanmean(med)), "selisih_PD_HC": selisih,
                      "p_permutasi": np.nan, "ambang": AMBANG if permutasi else np.nan,
                      "n_PD": n_pd, "n_HC": n_hc, "n_rekaman": len(idx_pakai)})

        for nama, fn in AGREGASI.items():
            peta = {i: fn([art[(arm, s)][i][besaran] for s in seeds]) for i in idx_pakai}
            rs = _rho_subjek(peta, penanda, idx_pakai)
            selisih, n_pd, n_hc = _selisih_kelompok(rs, penanda, idx_pakai)
            med = np.nanmedian([korelasi_rekaman(peta[i], penanda[i]["peta"])
                                for i in idx_pakai])
            pval = np.nan
            if permutasi and nama == "peringkat":
                lab = {penanda[i]["subjek"]: penanda[i]["label"] for i in idx_pakai}
                a = [v for s, v in rs.items() if lab.get(s) == 1]
                b = [v for s, v in rs.items() if lab.get(s) == 0]
                if a and b:
                    _, pval = uji_permutasi(a, b)
            baris.append({"kohort": kohort, "arsitektur": arm, "besaran": besaran,
                          "agregasi": nama,
                          "tugas": "STCP" if hanya_tugas == 2 else "semua",
                          "rho_median": float(med), "selisih_PD_HC": selisih,
                          "p_permutasi": pval, "ambang": AMBANG if permutasi else np.nan,
                          "n_PD": n_pd, "n_HC": n_hc, "n_rekaman": len(idx_pakai)})
    return baris


# ------------------------------------------------------------------- keluaran 3
def kurva_seed(art, seeds, kohort) -> list[dict]:
    """Berapa seed sampai peta berhenti bergerak, diukur terhadap ensemble penuh."""
    arms = sorted({k[0] for k in art})
    baris = []
    for arm in arms:
        idxs = sorted(art[(arm, seeds[0])])
        penuh = {i: ens_peringkat([art[(arm, s)][i]["alpha"] for s in seeds]) for i in idxs}
        for k in range(1, len(seeds) + 1):
            semua = list(itertools.combinations(seeds, k))
            if len(semua) > MAKS_SUBSET:
                pilih = [semua[j] for j in RNG.choice(len(semua), MAKS_SUBSET, replace=False)]
            else:
                pilih = semua
            nilai = []
            for sub in pilih:
                rho = [korelasi_rekaman(ens_peringkat([art[(arm, s)][i]["alpha"] for s in sub]),
                                        penuh[i]) for i in idxs]
                nilai.append(np.nanmedian(rho))
            baris.append({"kohort": kohort, "arsitektur": arm, "k": k,
                          "rho_thd_ensemble_penuh": float(np.mean(nilai)),
                          "sb": float(np.std(nilai)), "n_subset": len(pilih)})
    return baris


def lacak_temuan7(art_u, seed_u, art_n, seed_n) -> pd.DataFrame:
    """Lacak angka keruntuhan peta pada Temuan 7 ke berkas yang menghasilkannya.

    Dosier sempat mencantumkan 0,0109 sebagai padanan NewHandPD bagi 0,1250 pada
    basis data utama. Angka itu tidak dapat dilacak ke satu pun berkas hasil, dan
    fungsi ini menguji lima agregasi kandidat agar penarikannya berdasar bukti,
    bukan berdasar ketidakmampuan menemukan.
    """
    u = pd.read_csv(HASIL / "s7_berpasangan_alpha_phi.csv")
    n = pd.read_csv(HASIL / "s5_keselarasan.csv")
    per_arm = {}
    for arm in sorted({k[0] for k in art_n}):
        idxs = sorted(art_n[(arm, seed_n[0])])
        per_arm[arm] = [np.nanmedian([korelasi_rekaman(art_n[(arm, s)][i]["alpha"],
                                                       art_n[(arm, s)][i]["phi"]) for i in idxs])
                        for s in seed_n]
    acuan = float(u.rho_alpha_median.mean())
    kandidat = [
        ("UCI rerata rho_alpha_median 3 arm", acuan, "results/s7_berpasangan_alpha_phi.csv"),
        ("NHP rerata rho_alpha 3 arm", float(n.rho_alpha.mean()), "results/s5_keselarasan.csv"),
        ("NHP rerata rho_alpha 2 arm prareg",
         float(n[n.arsitektur != "mamba3"].rho_alpha.mean()), "results/s5_keselarasan.csv"),
        ("NHP rerata rho(alpha,phi) 3 arm seluruh seed",
         float(np.mean([np.mean(v) for v in per_arm.values()])), "results/s5_artefak.pkl"),
        ("NHP rerata rho(alpha,phi) 3 arm lima seed pertama",
         float(np.mean([np.mean(v[:5]) for v in per_arm.values()])), "results/s5_artefak.pkl"),
    ]
    return pd.DataFrame([{"besaran": a, "nilai": b, "sumber": c,
                          "rasio_thd_UCI": acuan / b if b else np.nan,
                          "cocok_0_0109": abs(b - 0.0109) < 5e-4}
                         for a, b, c in kandidat])


def main() -> int:
    print("P0 — peta ensemble antar seed. Tanpa GPU.\n")

    art_u, seed_u, pen_u = muat_uci()
    art_n, seed_n, pen_n = muat_newhandpd()
    print(f"UCI 395   : {len(seed_u)} seed, {len(pen_u)} rekaman")
    print(f"NewHandPD : {len(seed_n)} seed, {len(pen_n)} rekaman\n")

    # Pemeriksaan jangkar: keselarasan satu seed pada STCP harus mereproduksi
    # results/s7_tiga_peta.csv. Kalau meleset, penanda dihitung ulang dengan salah
    # dan seluruh keluaran 2 tidak sah.
    print("Jangkar penanda UCI (STCP, alpha vs penanda, thd s7_tiga_peta.csv):")
    cek = keselarasan(art_u, seed_u, pen_u, "uci395", hanya_tugas=2)
    lolos = True
    for r in [x for x in cek if x["agregasi"] == "satu_seed" and x["besaran"] == "alpha"]:
        acuan = JANGKAR_STCP[r["arsitektur"]]
        got = r["selisih_PD_HC"]
        ok = abs(got - acuan) < 5e-3
        lolos &= ok
        print(f"  {r['arsitektur']:7s} dihitung {got:+.5f}  acuan {acuan:+.5f}  "
              f"n_PD={r['n_PD']} n_HC={r['n_HC']}  {'OK' if ok else 'MELESET'}")
    if not lolos:
        print("\nGAGAL: jangkar meleset, penanda dihitung ulang dengan salah.", file=sys.stderr)
        return 1
    print()

    k1 = kesetiaan(art_u, seed_u, "uci395") + kesetiaan(art_n, seed_n, "newhandpd")
    pd.DataFrame(k1).to_csv(HASIL / "p0_kesetiaan_ensemble.csv", index=False)
    print("p0_kesetiaan_ensemble.csv")
    print(pd.DataFrame(k1).query("agregasi == 'peringkat'")
          [["kohort", "arsitektur", "rho_satu_seed", "rho_ensemble", "delta"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"), "\n")

    k2 = (keselarasan(art_u, seed_u, pen_u, "uci395", hanya_tugas=2, permutasi=True)
          + keselarasan(art_u, seed_u, pen_u, "uci395", hanya_tugas=None)
          + keselarasan(art_n, seed_n, pen_n, "newhandpd", hanya_tugas=None))
    pd.DataFrame(k2).to_csv(HASIL / "p0_keselarasan_ensemble.csv", index=False)
    print("p0_keselarasan_ensemble.csv")
    print(pd.DataFrame(k2).query("agregasi in ['satu_seed', 'peringkat']")
          [["kohort", "tugas", "arsitektur", "besaran", "agregasi",
            "rho_median", "selisih_PD_HC", "p_permutasi"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"), "\n")

    # Analisis primer pra-registrasi diulang dengan peta ensemble. EKSPLORATORI:
    # analisis pra-registrasi yang lama tidak diganti dan ambangnya tidak diubah.
    pr = pd.DataFrame(k2).query(
        "tugas == 'STCP' and arsitektur == 'mamba2' and besaran == 'phi'")
    print("ANALISIS PRIMER DIULANG DENGAN PETA ENSEMBLE (eksploratori)")
    print(f"  pra-registrasi (1 seed) : selisih +0.011177  ambang {AMBANG:.6f}  p 0.5847  GAGAL")
    for _, r in pr.iterrows():
        v = "LOLOS" if (r.selisih_PD_HC > AMBANG and r.p_permutasi < 0.05) else "gagal"
        pp = "     -" if np.isnan(r.p_permutasi) else f"{r.p_permutasi:.4f}"
        print(f"  {r.agregasi:22s}: selisih {r.selisih_PD_HC:+.6f}  "
              f"ambang {AMBANG:.6f}  p {pp}  {v}")
    print()

    k3 = kurva_seed(art_u, seed_u, "uci395") + kurva_seed(art_n, seed_n, "newhandpd")
    pd.DataFrame(k3).to_csv(HASIL / "p0_kurva_seed.csv", index=False)
    print("p0_kurva_seed.csv")
    print(pd.DataFrame(k3).pivot(index=["kohort", "k"], columns="arsitektur",
                                 values="rho_thd_ensemble_penuh")
          .to_string(float_format=lambda v: f"{v:.4f}"))

    lac = lacak_temuan7(art_u, seed_u, art_n, seed_n)
    lac.to_csv(HASIL / "p0_lacak_temuan7.csv", index=False)
    print("\np0_lacak_temuan7.csv — pelacakan angka Temuan 7")
    print(lac[["besaran", "nilai", "rasio_thd_UCI", "cocok_0_0109"]]
          .to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    if not lac.cocok_0_0109.any():
        print("  Tak satu pun agregasi menghasilkan 0,0109 -> angka itu ditarik.")

    # Gerbang Play 5, dibekukan sebelum hasil dilihat: perbaikan < +0,010 pada
    # agregasi primer di KEDUA kohort memicu Play 5.
    d = pd.DataFrame(k1).query("agregasi == 'peringkat' and arsitektur == 'mamba2'")
    delta = dict(zip(d["kohort"], d["delta"]))
    picu = all(v < 0.010 for v in delta.values())
    print(f"\nGERBANG PLAY 5 — delta mamba2 (peringkat): "
          f"{', '.join(f'{k} {v:+.4f}' for k, v in delta.items())}")
    print(f"  ambang +0,0100 di kedua kohort -> Play 5 {'DIJALANKAN' if picu else 'TIDAK dijalankan'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
