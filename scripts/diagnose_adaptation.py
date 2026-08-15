"""
diagnose_adaptation.py - JALANKAN INI PERTAMA
Author: Kukuh Yudhistiro, 2026

Menjawab satu pertanyaan yang menentukan arah seluruh revisi:
apakah mekanisme adaptif A-GWi benar-benar aktif pada data?

Latar belakang. Eq. (7) memusatkan sigmoid pada rho_L = 0.5 dengan
k_s = 25. Naskah sendiri melaporkan (Bagian 4.8) bahwa median rho_L pada
BSDS500 adalah 0.087. Pada nilai itu, argumen sigmoid adalah
25 * (0.087 - 0.5) = -10.3, sehingga sigmoid = 3.3e-05 dan
f0 = f_min + 0.4 * 3.3e-05 = 0.050013.

Pemeriksaan pada citra di dense_images memberi 79 sampai 99 persen piksel
dengan f0 dalam 1 persen dari f_min. Bila itu berlaku pada seluruh test set,
A-GWi secara efektif adalah filter STATIS pada f0 = f_min, dan seluruh
klaim adaptasi per piksel tidak didukung oleh implementasinya sendiri.

Skrip ini mengukur tiga hal pada dataset penuh:
  1. Distribusi rho_L dan f0_adapt, termasuk fraksi piksel yang tersaturasi.
  2. Korelasi antara keluaran A-GWi dan Gabor STATIS pada f0 = f_min.
     Korelasi tinggi berarti adaptasi tidak menyumbang apa pun.
  3. Norma L2 kernel sebagai fungsi f0, yang menunjukkan skala amplitudo
     berbeda sampai dua orde besaran di sepanjang rentang frekuensi.

Keluaran: diagnose_adaptation.csv dan diagnose_adaptation.png

Pemakaian:
  python diagnose_adaptation.py --data-root ./data --dataset BSDS500 \
      --max-images 50 --out-dir ./diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from methods_v2 import (
    preprocess, estimate_density_sobel, transform_rho,
    AGWiParams, SIGMA_COEF, ORIENTATIONS_8,
)

DATASET_LAYOUT = {
    "BSDS500": {"img_dir": "BSDS500/images/test", "img_ext": "*.jpg"},
    "BSDS500_train": {"img_dir": "BSDS500/images/train", "img_ext": "*.jpg"},
    "UDED": {"img_dir": "UDED/imgs", "img_ext": "*.jpg"},
}


def agwi_quantized(img, rho, params, levels=96):
    """A-GWi dengan f0 dikuantisasi ke `levels` tingkat.

    Setara dengan implementasi per piksel sampai galat kuantisasi, tetapi
    memakai filter2D sehingga cepat tanpa Numba. Dipakai hanya untuk
    diagnostik, bukan untuk angka yang dilaporkan.
    """
    f_sig = 1.0 / (1.0 + np.exp(-params.k_steepness *
                                (rho - params.rho_center)))
    f0map = params.f_min + (params.f_max - params.f_min) * f_sig
    edges = np.linspace(params.f_min, params.f_max, levels + 1)
    idx = np.clip(np.digitize(f0map, edges) - 1, 0, levels - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    coef = SIGMA_COEF[params.sigma_relation]
    half = params.kernel_size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    out = np.zeros_like(img)
    for li in range(levels):
        m = idx == li
        if not m.any():
            continue
        f0 = centers[li]
        sx = coef / f0
        sy = sx / params.aspect
        best = None
        for o in range(params.n_orientations):
            th = o * np.pi / params.n_orientations
            xt = x * np.cos(th) + y * np.sin(th)
            yt = -x * np.sin(th) + y * np.cos(th)
            k = np.exp(-0.5 * (xt ** 2 / sx ** 2 + yt ** 2 / sy ** 2)) \
                * np.sin(2 * np.pi * f0 * xt)
            if params.l2_normalize:
                n = np.linalg.norm(k)
                if n > 0:
                    k = k / n
            r = np.abs(cv2.filter2D(img, cv2.CV_64F, k,
                                    borderType=cv2.BORDER_REFLECT))
            best = r if best is None else np.maximum(best, r)
        out[m] = best[m]
    return out, f0map


def static_gabor(img, f0, params):
    coef = SIGMA_COEF[params.sigma_relation]
    sx = coef / f0
    sy = sx / params.aspect
    half = params.kernel_size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    best = None
    for o in range(params.n_orientations):
        th = o * np.pi / params.n_orientations
        xt = x * np.cos(th) + y * np.sin(th)
        yt = -x * np.sin(th) + y * np.cos(th)
        k = np.exp(-0.5 * (xt ** 2 / sx ** 2 + yt ** 2 / sy ** 2)) \
            * np.sin(2 * np.pi * f0 * xt)
        r = np.abs(cv2.filter2D(img, cv2.CV_64F, k,
                                borderType=cv2.BORDER_REFLECT))
        best = r if best is None else np.maximum(best, r)
    return best


def norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-12)


def kernel_norm_table(params, n=25):
    rows = []
    half = params.kernel_size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    coef = SIGMA_COEF[params.sigma_relation]
    for f0 in np.linspace(params.f_min, params.f_max, n):
        sx = coef / f0
        sy = sx / params.aspect
        k = np.exp(-0.5 * (x ** 2 / sx ** 2 + y ** 2 / sy ** 2)) \
            * np.sin(2 * np.pi * f0 * x)
        rows.append({"f0": f0, "sigma_x": sx, "sigma_y": sy,
                     "l2_norm": float(np.linalg.norm(k)),
                     "dc_sum": float(k.sum())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dataset", default="BSDS500")
    ap.add_argument("--max-images", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, default=Path("diagnostics"))
    ap.add_argument("--rho-transform", default="none",
                    choices=["none", "cdf"])
    ap.add_argument("--rho-center", type=float, default=0.5)
    ap.add_argument("--k-steepness", type=float, default=25.0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    params = AGWiParams(rho_center=args.rho_center,
                        k_steepness=args.k_steepness)
    layout = DATASET_LAYOUT[args.dataset]
    paths = sorted((args.data_root / layout["img_dir"]).glob(layout["img_ext"]))
    if not paths:
        for ext in ["*.png", "*.jpeg"]:
            paths.extend(sorted((args.data_root / layout["img_dir"]).glob(ext)))
    paths = sorted(paths)[:args.max_images]
    if not paths:
        raise SystemExit(f"Tidak ada citra di {args.data_root / layout['img_dir']}")

    print(f"[INFO] {len(paths)} citra, rho_transform={args.rho_transform}, "
          f"rho_center={args.rho_center}, k_s={args.k_steepness}")

    rows, all_rho, all_f0 = [], [], []
    band = 0.01 * (params.f_max - params.f_min)
    for i, p in enumerate(paths, 1):
        gray, _ = preprocess(p)
        rho = estimate_density_sobel(gray, params.sobel_ksize,
                                     params.density_blur)
        rho = transform_rho(rho, args.rho_transform)
        out, f0map = agwi_quantized(gray, rho, params)
        stat = static_gabor(gray, params.f_min, params)
        a, b = norm01(out), norm01(stat)
        corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
        rows.append({
            "image": p.stem,
            "rho_median": float(np.median(rho)),
            "rho_p90": float(np.percentile(rho, 90)),
            "rho_p99": float(np.percentile(rho, 99)),
            "rho_max": float(rho.max()),
            "f0_median": float(np.median(f0map)),
            "f0_p99": float(np.percentile(f0map, 99)),
            "frac_f0_at_fmin": float(np.mean(f0map < params.f_min + band)),
            "frac_f0_above_mid": float(np.mean(
                f0map > 0.5 * (params.f_min + params.f_max))),
            "corr_with_static_fmin": corr,
            "mad_with_static_fmin": float(np.abs(a - b).mean()),
        })
        all_rho.append(rho.ravel()[::17])
        all_f0.append(f0map.ravel()[::17])
        if i % 10 == 0:
            print(f"  {i}/{len(paths)}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "diagnose_adaptation.csv", index=False)

    kn = kernel_norm_table(params)
    kn.to_csv(args.out_dir / "kernel_norm_vs_f0.csv", index=False)

    print("\n" + "=" * 72)
    print("RINGKASAN DIAGNOSTIK")
    print("=" * 72)
    print(f"  rho_L median (dataset)          : {df.rho_median.median():.4f}")
    print(f"  rho_L p99 (dataset)             : {df.rho_p99.median():.4f}")
    print(f"  f0 median (dataset)             : {df.f0_median.median():.5f}  "
          f"(f_min={params.f_min}, f_max={params.f_max})")
    print(f"  Piksel dengan f0 ~= f_min       : "
          f"{100 * df.frac_f0_at_fmin.mean():.2f} persen")
    print(f"  Piksel dengan f0 > titik tengah : "
          f"{100 * df.frac_f0_above_mid.mean():.4f} persen")
    print(f"  Korelasi A-GWi vs Gabor statis f0=f_min : "
          f"{df.corr_with_static_fmin.mean():.4f} "
          f"(median {df.corr_with_static_fmin.median():.4f})")
    print(f"  Rasio norma L2 kernel f_min/f_max       : "
          f"{kn.l2_norm.iloc[0] / max(kn.l2_norm.iloc[-1], 1e-12):.1f}x")
    print("\n  Interpretasi:")
    if df.frac_f0_at_fmin.mean() > 0.5:
        print("  [KRITIS] Lebih dari separuh piksel tersaturasi pada f_min.")
        print("           Adaptasi tidak aktif. Jalankan ulang dengan")
        print("           --rho-transform cdf atau --rho-center <median rho>.")
    else:
        print("  [OK] Adaptasi mencakup rentang frekuensi yang berarti.")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].hist(np.concatenate(all_rho), bins=120, color="#4477aa")
    ax[0].axvline(args.rho_center, color="crimson", ls="--",
                  label=f"pusat sigmoid = {args.rho_center}")
    ax[0].set_xlabel(r"$\rho_L$")
    ax[0].set_ylabel("count")
    ax[0].set_title("(a) Distribusi $\\rho_L$")
    ax[0].legend()

    ax[1].hist(np.concatenate(all_f0), bins=120, color="#228833")
    ax[1].axvline(params.f_min, color="crimson", ls="--", label="$f_{min}$")
    ax[1].axvline(params.f_max, color="k", ls="--", label="$f_{max}$")
    ax[1].set_xlabel("$f_{0,adapt}$ (cycles/pixel)")
    ax[1].set_title("(b) Distribusi frekuensi adaptif")
    ax[1].legend()

    ax[2].plot(kn.f0, kn.l2_norm, "o-", color="#ee6677")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("$f_0$")
    ax[2].set_ylabel("kernel $L_2$ norm")
    ax[2].set_title("(c) Norma kernel terhadap $f_0$")
    fig.tight_layout()
    fig.savefig(args.out_dir / "diagnose_adaptation.png", dpi=160)
    print(f"\n[OK] {args.out_dir}/diagnose_adaptation.csv")
    print(f"[OK] {args.out_dir}/kernel_norm_vs_f0.csv")
    print(f"[OK] {args.out_dir}/diagnose_adaptation.png")


if __name__ == "__main__":
    main()
