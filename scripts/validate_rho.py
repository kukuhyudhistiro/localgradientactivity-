"""
validate_rho.py - Validasi empiris rho_L
Author: Kukuh Yudhistiro, 2026

Menjawab R1-4 dan R2-4: reviewer menyatakan bahwa rho_L adalah magnitudo
gradien yang dihaluskan, bukan densitas objek atau densitas batas, dan
naskah tidak pernah membuktikan bahwa besaran itu melacak densitas yang
diklaim.

Skrip ini mengukur, pada setiap citra:
  1. Spearman rank correlation antara rho_L dan densitas batas ground truth
     d_gt (jumlah piksel batas GT dalam jendela w x w, dinormalkan).
  2. Spearman antara rho_L dan kontras lokal (std intensitas jendela w x w),
     sebagai pembanding. Jika korelasi terhadap kontras lebih tinggi daripada
     terhadap densitas batas, penamaan "density" memang tidak tepat dan
     istilah "gradient activity" harus dipakai.
  3. Dua mode kegagalan yang disebut reviewer, dihitung secara eksplisit:
     - tepi terisolasi berkontras tinggi: d_gt rendah tetapi rho_L tinggi
     - tekstur rapat berkontras rendah: d_gt tinggi tetapi rho_L rendah

Keluaran: validate_rho.csv dan validate_rho.png

Pemakaian:
  python validate_rho.py --data-root ./data --dataset BSDS500 \
      --window 15 --out-dir ./diagnostics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import spearmanr
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from methods_v2 import preprocess, estimate_density_sobel

LAYOUT = {
    "BSDS500": {"img_dir": "BSDS500/images/test",
                "gt_dir": "BSDS500/groundTruth/test",
                "gt_format": "mat", "img_ext": ".jpg"},
    "UDED": {"img_dir": "UDED/imgs", "gt_dir": "UDED/gt",
             "gt_format": "png", "img_ext": ".jpg"},
}


def load_gt_union(gt_path, gt_format):
    """Peta batas gabungan: piksel dianggap batas bila ditandai >= 1 anotator."""
    if gt_format == "mat":
        mat = loadmat(str(gt_path))
        gt = mat["groundTruth"]
        acc = None
        for i in range(gt.shape[1]):
            b = gt[0, i]["Boundaries"][0, 0] > 0
            acc = b if acc is None else (acc | b)
        return acc.astype(np.float64)
    arr = np.array(Image.open(str(gt_path)).convert("L"))
    return (arr > 127).astype(np.float64)


def boundary_density(gt_bin, win):
    """Fraksi piksel batas GT dalam jendela win x win."""
    k = np.ones((win, win), dtype=np.float64) / (win * win)
    return cv2.filter2D(gt_bin, cv2.CV_64F, k, borderType=cv2.BORDER_REFLECT)


def local_contrast(gray, win):
    m = cv2.boxFilter(gray, cv2.CV_64F, (win, win))
    m2 = cv2.boxFilter(gray ** 2, cv2.CV_64F, (win, win))
    return np.sqrt(np.maximum(m2 - m ** 2, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dataset", default="BSDS500")
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--subsample", type=int, default=7,
                    help="Ambil setiap n piksel untuk korelasi (kecepatan)")
    ap.add_argument("--out-dir", type=Path, default=Path("diagnostics"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lay = LAYOUT[args.dataset]
    img_dir = args.data_root / lay["img_dir"]
    gt_dir = args.data_root / lay["gt_dir"]
    paths = sorted(img_dir.glob("*" + lay["img_ext"]))
    if not paths:
        for ext in [".png", ".jpeg"]:
            paths.extend(sorted(img_dir.glob("*" + ext)))
    paths = sorted(paths)
    if args.max_images:
        paths = paths[:args.max_images]
    if not paths:
        raise SystemExit(f"Tidak ada citra di {img_dir}")

    rows = []
    scatter_rho, scatter_dgt = [], []
    s = args.subsample
    for i, p in enumerate(paths, 1):
        gt_path = (gt_dir / f"{p.stem}.mat" if lay["gt_format"] == "mat"
                   else gt_dir / f"{p.stem}.png")
        if not gt_path.exists():
            continue
        gray, _ = preprocess(p)
        rho = estimate_density_sobel(gray)
        gt = load_gt_union(gt_path, lay["gt_format"])
        d_gt = boundary_density(gt, args.window)
        contrast = local_contrast(gray, args.window)

        a = rho.ravel()[::s]
        b = d_gt.ravel()[::s]
        c = contrast.ravel()[::s]
        rho_gt = spearmanr(a, b).statistic
        rho_ct = spearmanr(a, c).statistic

        # mode kegagalan
        hi_rho = a > np.percentile(a, 90)
        lo_dgt = b < np.percentile(b, 50)
        hi_dgt = b > np.percentile(b, 90)
        lo_rho = a < np.percentile(a, 50)
        rows.append({
            "image": p.stem,
            "spearman_rho_vs_gt_density": float(rho_gt),
            "spearman_rho_vs_local_contrast": float(rho_ct),
            "frac_highRho_lowGT": float(np.mean(hi_rho & lo_dgt)),
            "frac_highGT_lowRho": float(np.mean(hi_dgt & lo_rho)),
        })
        if i <= 30:
            scatter_rho.append(a[::5])
            scatter_dgt.append(b[::5])
        if i % 25 == 0:
            print(f"  {i}/{len(paths)}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / f"validate_rho_{args.dataset}.csv", index=False)

    q = df.spearman_rho_vs_gt_density.quantile([0.25, 0.5, 0.75])
    qc = df.spearman_rho_vs_local_contrast.quantile([0.25, 0.5, 0.75])
    print("\n" + "=" * 72)
    print(f"VALIDASI rho_L  ({args.dataset}, {len(df)} citra, "
          f"jendela {args.window}x{args.window})")
    print("=" * 72)
    print(f"  Spearman rho_L vs densitas batas GT : median {q[0.5]:.3f} "
          f"(IQR {q[0.25]:.3f} sampai {q[0.75]:.3f})")
    print(f"  Spearman rho_L vs kontras lokal     : median {qc[0.5]:.3f} "
          f"(IQR {qc[0.25]:.3f} sampai {qc[0.75]:.3f})")
    print(f"  Piksel rho_L tinggi tetapi GT rendah: "
          f"{100 * df.frac_highRho_lowGT.mean():.2f} persen")
    print(f"  Piksel GT tinggi tetapi rho_L rendah: "
          f"{100 * df.frac_highGT_lowRho.mean():.2f} persen")
    if qc[0.5] > q[0.5]:
        print("\n  [KESIMPULAN] rho_L lebih kuat melacak kontras lokal daripada")
        print("  densitas batas. Istilah 'local gradient activity' harus dipakai")
        print("  menggantikan 'local gradient density' di seluruh naskah.")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].hist(df.spearman_rho_vs_gt_density, bins=25, alpha=0.8,
               label="vs densitas batas GT", color="#4477aa")
    ax[0].hist(df.spearman_rho_vs_local_contrast, bins=25, alpha=0.6,
               label="vs kontras lokal", color="#ee6677")
    ax[0].set_xlabel("Spearman correlation")
    ax[0].set_ylabel("jumlah citra")
    ax[0].set_title("(a) Sebaran korelasi per citra")
    ax[0].legend()
    if scatter_rho:
        xr = np.concatenate(scatter_rho)
        yr = np.concatenate(scatter_dgt)
        ax[1].hexbin(xr, yr, gridsize=60, bins="log", cmap="viridis")
        ax[1].set_xlabel(r"$\rho_L$")
        ax[1].set_ylabel("densitas batas GT")
        ax[1].set_title("(b) Hubungan piksel-demi-piksel")
    fig.tight_layout()
    fig.savefig(args.out_dir / f"validate_rho_{args.dataset}.png", dpi=160)
    print(f"\n[OK] {args.out_dir}/validate_rho_{args.dataset}.csv")
    print(f"[OK] {args.out_dir}/validate_rho_{args.dataset}.png")


if __name__ == "__main__":
    main()
