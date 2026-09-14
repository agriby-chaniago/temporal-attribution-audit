"""P4 — sumbu geser tugas dari 396 rekaman NewHandPD yang belum pernah dipakai.

Jalankan: python3 scripts/analisis_p4_geser_tugas.py --periksa    (CPU, verifikasi)
          python3 scripts/analisis_p4_geser_tugas.py              (GPU, ~1 jam)

Apa yang dibuka, dan apa yang TIDAK
-----------------------------------
`data/raw/newhandpd/` memuat dua belas tugas; Skenario S5 hanya memakai empat
spiral. Empat meander dan dua Dia — 396 rekaman — sudah terunduh, lisensinya
sudah didokumentasikan, dan belum pernah disentuh.

**Ini bukan kohort ketiga.** Subjeknya orang yang sama, sehingga rekaman ini
tidak boleh dipakai menambah pasangan retensi lintas kohort. Yang dibuka adalah
sumbu geser **tugas** dengan perangkat, negara, dan subjek terkunci konstan —
justru sesuatu yang tidak dapat diberikan kohort baru mana pun.

Uji dinamai sebelum dijalankan
------------------------------
    Apakah peringkat arsitektur membalik antar TUGAS ketika kohort, perangkat,
    dan subjek dikunci?

Kedua hasilnya berharga, dan itu disebut di depan. Membalik berarti Temuan 6
menguat: ketidakstabilan peringkat bukan khusus perpindahan kohort. Tidak
membalik berarti yang membalikkan adalah kohort dan bukan tugas, sehingga
penjelasannya **menyempit** — dan menyempitkan penjelasan adalah kemajuan.

Verifikasi wajib sebelum melatih
--------------------------------
Asumsi pemuat dibuat untuk `sigSp*` dan belum tentu berlaku. Yang diperiksa:
identitas enam kanal lewat panjang run dan koefisien variasi norma kanal 4-6
(1,63 % pada spiral, ciri sensor tiga sumbu yang didominasi gravitasi), serta
anomali penamaan berkas. Kesahihan pemisahan kanal model/penanda bergantung
padanya — kalau ciri itu hilang, pemisahannya harus diperiksa ulang sebelum
dipakai.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from scipy import stats
from scipy.stats import spearmanr

AKAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AKAR / "src"))

from artefak import muat_artefak, simpan_atomik  # noqa: E402
from marker import ke_grid_patch, penanda_cepat_bisp  # noqa: E402
from model import PDClassifier  # noqa: E402
from newhandpd import (KELUARGA_BISP, META_AMAN, baca_meta_bisp,  # noqa: E402
                       muat_berkas_bisp, muat_cache_bisp, muat_newhandpd,
                       simpan_cache_bisp)
from preprocessing import Normalisasi  # noqa: E402
from training import latih, prediksi  # noqa: E402

HASIL = AKAR / "results"
MENTAH = AKAR / "data" / "raw" / "newhandpd"
CACHE = AKAR / "data" / "cache"
FS, P, EPOCHS = 100, 7, 35
SEEDS = [0, 1, 2, 3, 4]
ARSITEKTUR = ["mamba2", "gru", "mamba3"]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
KELUARGA_UJI = ["meander", "dia"]


# ---------------------------------------------------------------- verifikasi
def periksa() -> int:
    print("P4 — verifikasi sebelum melatih. Tanpa GPU.\n")

    # 1. Anomali penamaan. sigMea1 hanya punya 65 berkas, sisanya 66.
    pola = re.compile(r"(sig[A-Za-z]+\w*)-([HP]\d+)\.txt$")
    aneh = []
    for sub in ["extracted/Signal", "extracted_patient/Signal"]:
        for f in sorted((MENTAH / sub).glob("sig*.txt")):
            if not f.name.startswith("._") and not pola.search(f.name):
                aneh.append(f)
    print(f"1. Berkas tidak cocok pola nama: {[f.name for f in aneh]}")
    for f in aneh:
        meta = baca_meta_bisp(f)
        pid = meta.get("Person_ID_Number")
        cocok = []
        for g in sorted(f.parent.glob("sig*.txt")):
            if g != f and not g.name.startswith("._"):
                if baca_meta_bisp(g).get("Person_ID_Number") == pid:
                    cocok.append(pola.search(g.name).group(2) if pola.search(g.name) else g.name)
        subjek = sorted(set(cocok))
        print(f"   {f.name}: Person_ID {pid}")
        print(f"   -> subjek dengan Person_ID sama: {subjek}")
        print(f"   -> VONIS: {'salah nama, identitas terpulihkan' if len(subjek) == 1 else 'ambigu, DIBUANG'}")
    print()

    # 2. Muat tiap keluarga dan periksa ciri kanal.
    baris = []
    for kel in ["spiral"] + KELUARGA_UJI:
        rek = muat_newhandpd(MENTAH, keluarga=kel)
        durasi = np.array([len(r.kanal) / FS for r in rek])
        # Ciri sensor tiga sumbu: norma kanal 4-6 hampir tetap.
        cv = [float(np.std(n) / np.mean(n))
              for r in rek if (n := np.linalg.norm(r.kanal_ditahan, axis=1)).mean() > 0]
        # Panjang run identik, DIUKUR PADA BERKAS MENTAH 1000 Hz. Sesudah
        # resample_poly struktur tahan/ulang kanal 4-6 hancur dan metriknya
        # memberi 1,0 untuk semua keluarga, termasuk spiral.
        pola_glob = KELUARGA_BISP[kel][0]
        run_lo, run_hi = [], []
        contoh = []
        for sub in ["extracted/Signal", "extracted_patient/Signal"]:
            contoh += [f for f in sorted((MENTAH / sub).glob(pola_glob))
                       if not f.name.startswith("._")][:10]
        for f in contoh:
            d = muat_berkas_bisp(f)
            for k in range(6):
                v = d[:, k]
                panjang = len(v) / max(np.count_nonzero(np.diff(v)) + 1, 1)
                (run_lo if k < 3 else run_hi).append(panjang)
        baris.append({"keluarga": kel, "n_rekaman": len(rek),
                      "n_subjek": len({r.subjek for r in rek}),
                      "n_PD": len({r.subjek for r in rek if r.kelompok == "PD"}),
                      "n_HC": len({r.subjek for r in rek if r.kelompok == "HC"}),
                      "durasi_median_s": float(np.median(durasi)),
                      "durasi_min_s": float(durasi.min()),
                      "patch_median": int(np.median(durasi * FS // P)),
                      "cv_norma_ditahan": float(np.median(cv)),
                      "run_kanal_1_3": float(np.median(run_lo)),
                      "run_kanal_4_6": float(np.median(run_hi))})
        simpan_cache_bisp(rek, CACHE / f"newhandpd_{kel}_fs100.npz")
    t = pd.DataFrame(baris)
    print("2. Ciri per keluarga tugas (acuan dosier: run kanal 4-6 = 3,68-3,72; kanal 1-3 = 1,00)")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"), "\n")

    acuan = t[t.keluarga == "spiral"].iloc[0]
    lolos = True
    for _, r in t[t.keluarga != "spiral"].iterrows():
        # Ciri sensor tiga sumbu diuji lewat panjang run, bukan lewat cv norma:
        # cv mengukur seberapa banyak pena BERGERAK, dan tugas Dia memang gerak
        # melingkar sehingga cv-nya wajar lebih besar. Yang harus tetap adalah
        # STRUKTUR kanal, dan itulah yang panjang run ukur.
        ok_hi = abs(r.run_kanal_4_6 - acuan.run_kanal_4_6) < 0.5
        ok_lo = abs(r.run_kanal_1_3 - acuan.run_kanal_1_3) < 0.5
        lolos &= ok_hi and ok_lo
        print(f"   {r.keluarga:8s} run kanal 4-6 {r.run_kanal_4_6:.2f} "
              f"{'OK' if ok_hi else 'MENYIMPANG'}   kanal 1-3 {r.run_kanal_1_3:.2f} "
              f"{'OK' if ok_lo else 'MENYIMPANG'}   cv {r.cv_norma_ditahan:.4f}")
    print(f"   VONIS: pemisahan kanal {'tetap sah' if lolos else 'HARUS DIPERIKSA ULANG'}\n")
    t.to_csv(HASIL / "p4_verifikasi_kanal.csv", index=False)

    # 3. Meta subjek. HANYA kolom META_AMAN yang ditulis: header mentah memuat
    #    Surename, Forename, dan Notice, dan sekurangnya satu subjek pada rilis
    #    ini memiliki nama lengkap asli beserta nomor menyerupai rekam medis di
    #    sana. Menuliskannya berarti menyebarkan identitas pasien lewat G9.
    meta_baris = []
    for sub, kelompok in [("extracted/Signal", "HC"), ("extracted_patient/Signal", "PD")]:
        for f in sorted((MENTAH / sub).glob("sigSp1*.txt")):
            if f.name.startswith("._"):
                continue
            m = pola.search(f.name)
            if m:
                mm = baca_meta_bisp(f)
                meta_baris.append({"subjek": m.group(2), "kelompok": kelompok,
                                   **{k: mm.get(k) for k in META_AMAN}})
    md = pd.DataFrame(meta_baris)
    md.to_csv(HASIL / "p4_meta_subjek.csv", index=False)
    print(f"3. p4_meta_subjek.csv — {len(md)} subjek, kolom {list(md.columns)}")
    if "Age" in md:
        u = md.assign(Age=pd.to_numeric(md.Age, errors="coerce")).groupby("kelompok").Age
        print(f"   usia: {u.mean().round(1).to_dict()} (n {u.count().to_dict()})")
    return 0 if lolos else 1


# ------------------------------------------------------------------ pelatihan
def jalankan(rek, arsitektur, seed) -> dict:
    y = np.array([r.label for r in rek])
    grup = np.array([r.subjek for r in rek])
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    logit = np.zeros(len(rek))
    rho = []
    for i_latih, i_uji in skf.split(np.zeros(len(rek)), y, grup):
        norm = Normalisasi().fit([rek[i] for i in i_latih])
        torch.manual_seed(seed)
        m = PDClassifier(encoder=arsitektur, patch_size=P).to(DEV)
        latih(m, rek, list(i_latih), norm, epochs=EPOCHS, device=DEV, seed=seed)
        m.eval()
        lo, alphas = prediksi(m, rek, list(i_uji), norm, device=DEV)
        logit[i_uji] = lo
        for j, idx in enumerate(i_uji):
            pen = ke_grid_patch(penanda_cepat_bisp(rek[idx].kanal_ditahan, FS), P)
            n = min(len(alphas[j]), len(pen))
            a, b = np.asarray(alphas[j][:n], float), np.asarray(pen[:n], float)
            sah = ~np.isnan(b)
            if sah.sum() >= 4 and np.std(a[sah]) > 0 and np.std(b[sah]) > 0:
                rho.append((grup[idx], y[idx], float(spearmanr(a[sah], b[sah]).statistic)))
    # Divergensi pelatihan dicatat, BUKAN ditambal. Menambahkan gradient
    # clipping atau menurunkan lr akan membuat protokolnya berbeda dari S3 dan
    # S5, sehingga perbandingan lintas tugas tidak lagi sah. Seed yang divergen
    # dilaporkan apa adanya: sebuah arsitektur yang meledak pada satu dari lima
    # seed di tugas baru adalah keterangan tentang kestabilannya, bukan gangguan.
    if not np.isfinite(logit).all():
        n_buruk = int((~np.isfinite(logit)).sum())
        print(f"    DIVERGEN: {n_buruk}/{len(logit)} logit tidak hingga", flush=True)
        return {"arsitektur": arsitektur, "seed": seed, "auc": np.nan,
                "keselarasan_PD_HC": np.nan, "n_subjek": len(set(grup)),
                "divergen": True, "n_logit_tak_hingga": n_buruk}

    per = {}
    for s, lab, r in rho:
        per.setdefault((s, lab), []).append(r)
    nilai = {k: float(np.mean(v)) for k, v in per.items()}
    pd_ = [v for (s, lab), v in nilai.items() if lab == 1]
    hc = [v for (s, lab), v in nilai.items() if lab == 0]
    # AUC tingkat subjek, sama seperti S3 dan S5.
    subj = sorted(set(grup))
    ps = np.array([logit[grup == s].mean() for s in subj])
    ys = np.array([y[grup == s][0] for s in subj])
    return {"arsitektur": arsitektur, "seed": seed,
            "auc": float(roc_auc_score(ys, ps)),
            "keselarasan_PD_HC": float(np.median(pd_) - np.median(hc)) if pd_ and hc else np.nan,
            "n_subjek": len(subj), "divergen": False, "n_logit_tak_hingga": 0}


def main() -> int:
    if "--periksa" in sys.argv:
        return periksa()

    print(f"P4 — pelatihan sumbu geser tugas. {DEV}, patch {P}, {EPOCHS} epoch, "
          f"{len(ARSITEKTUR)} arm x {len(SEEDS)} seed x 5 fold\n")
    art_path = HASIL / "p4_artefak.pkl"
    art = muat_artefak(art_path) or []
    ada = {(a["keluarga"], a["arsitektur"], a["seed"]) for a in art}

    for kel in KELUARGA_UJI:
        rek = muat_cache_bisp(CACHE / f"newhandpd_{kel}_fs100.npz")
        print(f"{kel}: {len(rek)} rekaman, {len({r.subjek for r in rek})} subjek")
        for arsitektur in ARSITEKTUR:
            for seed in SEEDS:
                if (kel, arsitektur, seed) in ada:
                    continue
                t0 = time.time()
                h = jalankan(rek, arsitektur, seed)
                art.append({"keluarga": kel, **h})
                simpan_atomik(art_path, art)
                tanda = "DIVERGEN" if h.get("divergen") else f"AUC={h['auc']:.4f}"
                sel = ("" if h.get("divergen")
                       else f"  selaras={h['keselarasan_PD_HC']:+.4f}")
                print(f"  {kel:8s} {arsitektur:7s} seed={seed}  "
                      f"{time.time()-t0:6.1f}s  {tanda}{sel}", flush=True)

    t = pd.DataFrame(art)
    if "divergen" not in t:
        t["divergen"] = False
    t.to_csv(HASIL / "p4_geser_tugas.csv", index=False)

    # Uji peringkat lintas tugas. Keluarga spiral diambil dari retensi_per_seed
    # (kolom m_lintas = AUC NewHandPD) supaya ketiga keluarga dibandingkan pada
    # besaran yang sama persis, bukan dihitung ulang dengan jalur berbeda.
    sp = pd.read_csv(HASIL / "retensi_per_seed.csv")
    sp = sp.assign(keluarga="spiral", auc=sp.m_lintas)[["keluarga", "arsitektur", "seed", "auc"]]
    semua = pd.concat([t[["keluarga", "arsitektur", "seed", "auc"]], sp], ignore_index=True)
    baris = []
    for kel in ["spiral"] + KELUARGA_UJI:
        s = semua[semua.keluarga == kel].dropna(subset=["auc"])
        if s.empty:
            continue
        urut = " > ".join(s.groupby("arsitektur").auc.mean().sort_values(ascending=False).index)
        for a, b in [("mamba2", "gru"), ("mamba3", "gru"), ("mamba2", "mamba3")]:
            x = s[s.arsitektur == a].auc.values
            y = s[s.arsitektur == b].auc.values
            if len(x) < 2 or len(y) < 2:
                continue
            tt, pv = stats.ttest_ind(x, y, equal_var=False)
            baris.append({"keluarga": kel, "arm_a": a, "arm_b": b,
                          "auc_a": float(x.mean()), "auc_b": float(y.mean()),
                          "selisih": float(x.mean() - y.mean()), "t": float(tt),
                          "p_welch": float(pv), "nyata_05": bool(pv < 0.05),
                          "n_a": len(x), "n_b": len(y), "peringkat": urut})
    u = pd.DataFrame(baris)
    u.to_csv(HASIL / "p4_uji_lintas_tugas.csv", index=False)
    print("\np4_uji_lintas_tugas.csv — apakah peringkat membalik antar tugas")
    print(u[["keluarga", "arm_a", "arm_b", "selisih", "p_welch", "nyata_05"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    m2 = u[(u.arm_a == "mamba2") & (u.arm_b == "gru")]
    print(f"\n  mamba2 > gru: tanda positif pada {int((m2.selisih > 0).sum())}/{len(m2)} keluarga, "
          f"nyata pada {int(m2.nyata_05.sum())}/{len(m2)} -> "
          f"{'TIDAK membalik' if (m2.selisih > 0).all() else 'MEMBALIK'}")
    print("\np4_geser_tugas.csv")
    print(t.groupby(["keluarga", "arsitektur"])[["auc", "keselarasan_PD_HC"]]
          .agg(["mean", "std", "count"]).round(4).to_string())
    div = t[t.divergen.fillna(False)]
    if len(div):
        print(f"\nDIVERGEN — {len(div)} dari {len(t)} run, dikeluarkan dari rerata "
              f"dan dilaporkan apa adanya:")
        print(div[["keluarga", "arsitektur", "seed", "n_logit_tak_hingga"]]
              .to_string(index=False))
        print(t.assign(div=t.divergen.fillna(False))
              .groupby(["keluarga", "arsitektur"]).div.sum()
              .to_frame("n_seed_divergen").T.to_string())
    else:
        print("\nTidak ada run yang divergen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
