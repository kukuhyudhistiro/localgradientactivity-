#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_final_figures.py - Seluruh figure untuk naskah revisi JESA 44740
Author: Kukuh Yudhistiro, 2026

Menggantikan versi sebelumnya. Perubahan utama:

  [R1-9, R2-9]  Anotasi Shannon entropy dihapus dari figure perbandingan
                citra tunggal. Diganti dengan peta parameter f0 dan sigma.
  [R1-15, R2-15] Peta f0_adapt dan sigma_adapt ditambahkan sebagai bukti
                visual untuk klaim mekanisme.
  [R1-4, R2-4]  Seluruh label memakai "gradient activity", bukan "density".
  [R1-3, R2-3]  Figure baru untuk verifikasi parameter pada validation split.
  [R1-7, R2-7]  Figure baru untuk diagnostik saturasi sigmoid.
  [R1-17, R2-17] Panel diberi label (a), (b), (c). Tidak ada lagi "top" dan
                "bottom" pada caption.
  Seluruh label figure dalam bahasa Inggris, siap masuk naskah.

Figure yang dihasilkan:

  fig3_parameter_grid.png       validation split, poin 3
  fig4_pr_curves.png            PR curves (a) BSDS500 (b) UDED
  fig5_qualitative_BSDS500.png  perbandingan kualitatif
  fig6_qualitative_UDED.png     perbandingan kualitatif
  fig7_activity_distribution.png  (a) sebaran rho_L (b) std per citra
  fig8_activity_validation.png    (a) korelasi Spearman (b) hexbin
  fig9_adaptation_diagnostic.png  (a) rho_L (b) f0 (c) kernel norm
  fig10_parameter_maps.png        original, rho_L, f0, sigma, GWi, A-GWi

Figure 1 dan Figure 2 adalah diagram dan tidak dihasilkan skrip ini.
Figure 2 masih perlu diperbaiki secara manual: kotak Stage 3 menyebut bahwa
kernel size k dimodulasi oleh rho_L, padahal k tetap 7 x 7.

Pemakaian:
  python generate_final_figures.py --data-root ./data --output-root ./output \
      --eval-results ./eval_results_v2 --eval-grid ./eval_grid \
      --figures-dir ./figures --figures all

  python generate_final_figures.py ... --figures fig4 fig10
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
cv2.setNumThreads(1)
from scipy.io import loadmat
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from methods_v2 import (
    preprocess, estimate_density_sobel, compute_f0_sigma_maps,
    AGWiParams, SIGMA_COEF,
)

# ============================================================================
# Konfigurasi
# ============================================================================
DATASET_LAYOUT = {
    "BSDS500": {"img_dir": "BSDS500/images/test",
                "gt_dir": "BSDS500/groundTruth/test", "gt_fmt": "mat"},
    "UDED": {"img_dir": "UDED/imgs", "gt_dir": "UDED/gt", "gt_fmt": "png"},
}

METHOD_ORDER = ["Canny", "Sobel", "PC", "GWC", "GWi", "AGWi"]
METHOD_LABELS = {"Canny": "Canny", "Sobel": "Sobel", "PC": "PC",
                 "GWC": "GWC", "GWi": "GWi", "AGWi": "A-GWi"}
METHOD_COLORS = {"Canny": "#4477AA", "Sobel": "#EE6677", "PC": "#228833",
                 "GWC": "#CCBB44", "GWi": "#66CCEE", "AGWi": "#AA3377"}

QUALITATIVE_SAMPLES = {
    "BSDS500": ["3063", "29030", "128035", "35049"],
    "UDED": ["04-0896x4", "05-WIREFRAME-2", "28-img_043_SRF_2_HR"],
}

# Citra untuk figure peta parameter (fig10)
PARAM_MAP_SAMPLES = [("BSDS500", "3063"), ("BSDS500", "29030"),
                     ("UDED", "04-0896x4")]

HUMAN_ODS_BSDS500 = 0.803
IMG_EXTS = [".jpg", ".png", ".jpeg", ".JPG", ".PNG"]


def setup_matplotlib():
    plt.rcParams.update({
        "font.size": 9,
        "font.family": "serif",
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7,
        "savefig.dpi": 400,
    })


def find_image_file(image_dir, image_id):
    for ext in IMG_EXTS:
        p = image_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("black")
        sp.set_linewidth(0.5)


def load_gt_display(dataset, gt_dir, image_id):
    fmt = DATASET_LAYOUT[dataset]["gt_fmt"]
    if fmt == "mat":
        p = gt_dir / f"{image_id}.mat"
        if not p.exists():
            return None
        gt = loadmat(str(p))["groundTruth"]
        acc = None
        for i in range(gt.shape[1]):
            b = (gt[0, i]["Boundaries"][0, 0] > 0).astype(np.float32)
            acc = b if acc is None else np.maximum(acc, b)
        return (acc * 255).astype(np.uint8)
    p = gt_dir / f"{image_id}.png"
    if not p.exists():
        return None
    arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return (arr > 127).astype(np.uint8) * 255


def load_edge_map(output_root, dataset, method, image_id):
    p = output_root / dataset / method / f"{image_id}.png"
    if not p.exists():
        return None
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)


def draw_iso_f(ax, levels=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    r = np.linspace(0.01, 1.0, 500)
    for f in levels:
        p = f * r / np.maximum(2 * r - f, 1e-9)
        m = (p >= 0) & (p <= 1)
        ax.plot(r[m], p[m], color="0.75", linewidth=0.5,
                linestyle=":", zorder=1)
        if m.any():
            ax.annotate(f"F={f:.1f}", xy=(r[m][-1], p[m][-1]),
                        fontsize=5.5, color="0.55",
                        xytext=(2, 0), textcoords="offset points",
                        va="center")


# ============================================================================
# Figure 3: verifikasi parameter pada validation split (poin 3)
# ============================================================================
def fig3_parameter_grid(eval_grid, figures_dir):
    csv = eval_grid / "ods_summary_v2.csv"
    if not csv.exists():
        print(f"  [SKIP] fig3: {csv} tidak ada")
        return False
    df = pd.read_csv(csv)
    rows = []
    for _, r in df.iterrows():
        name = str(r["method"])
        if not name.startswith("G_fmin"):
            continue
        try:
            fmin_tok = name.split("_")[1].replace("fmin", "")
            ks_tok = name.split("_")[2].replace("ks", "")
            fmin = float(fmin_tok) / (10 ** (len(fmin_tok) - 1))
            rows.append({"f_min": round(fmin, 3), "k_s": int(ks_tok),
                         "ods": r["ods_f"],
                         "std": r.get("f_at_ods_std", np.nan)})
        except Exception:
            continue
    if not rows:
        print("  [SKIP] fig3: tidak ada baris grid pada CSV")
        return False
    g = pd.DataFrame(rows).sort_values(["k_s", "f_min"])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

    ax = axes[0]
    for ks, sub in g.groupby("k_s"):
        marker = "o" if ks == 25 else "s"
        ls = "-" if ks == 25 else "--"
        ax.plot(sub.f_min, sub.ods, marker=marker, linestyle=ls,
                label=f"$k_s$ = {ks}")
    used = g[(g.k_s == 25) & (np.isclose(g.f_min, 0.05))]
    if not used.empty:
        ax.scatter(used.f_min, used.ods, s=140, facecolors="none",
                   edgecolors="crimson", linewidths=1.4, zorder=5)
        ax.annotate("setting used", xy=(used.f_min.iloc[0], used.ods.iloc[0]),
                    xytext=(6, -14), textcoords="offset points",
                    fontsize=7, color="crimson")
    ax.set_xlabel(r"$f_{min}$ (cycles/pixel)")
    ax.set_ylabel("Validation ODS")
    ax.set_title("(a) Validation ODS across the grid")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower left")

    ax = axes[1]
    for ks, sub in g.groupby("k_s"):
        marker = "o" if ks == 25 else "s"
        ls = "-" if ks == 25 else "--"
        ax.plot(sub.f_min, sub["std"], marker=marker, linestyle=ls,
                label=f"$k_s$ = {ks}")
    ax.set_xlabel(r"$f_{min}$ (cycles/pixel)")
    ax.set_ylabel("Per-image F standard deviation")
    ax.set_title("(b) Stability across images")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right")

    fig.tight_layout()
    out = figures_dir / "fig3_parameter_grid.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white",
                pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] {out}")
    return True


# ============================================================================
# Figure 4: PR curves (poin 1, 13)
# ============================================================================
def fig4_pr_curves(eval_results, figures_dir):
    summary = eval_results / "ods_summary_v2.csv"
    ods = pd.read_csv(summary) if summary.exists() else None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    panel = ["(a) BSDS500", "(b) UDED"]

    for i, dataset in enumerate(["BSDS500", "UDED"]):
        ax = axes[i]
        draw_iso_f(ax)
        drawn = 0
        for method in METHOD_ORDER:
            f = eval_results / f"pr_curve_{dataset}_{method}.csv"
            if not f.exists():
                print(f"  [WARN] fig4: {f.name} tidak ada")
                continue
            pr = pd.read_csv(f)
            val = np.nan
            if ods is not None:
                row = ods[(ods.dataset == dataset) & (ods.method == method)]
                if not row.empty:
                    val = float(row.ods_f.iloc[0])
            is_ours = method == "AGWi"
            lw = 2.2 if is_ours else 1.2
            ls = "-" if is_ours else ("-." if method == "GWi" else "--")
            z = 4 if is_ours else 2
            lbl = METHOD_LABELS[method] + (" (ours)" if is_ours else "")
            if np.isfinite(val):
                lbl = f"[F={val:.3f}] {lbl}"
            ax.plot(pr["recall"], pr["precision"],
                    color=METHOD_COLORS[method], linewidth=lw,
                    linestyle=ls, label=lbl, zorder=z)
            drawn += 1
        if drawn == 0:
            print(f"  [SKIP] fig4: tidak ada kurva untuk {dataset}")
        if dataset == "BSDS500":
            h = HUMAN_ODS_BSDS500
            ax.scatter([h], [h], s=70, color="#006400", marker="o",
                       edgecolor="black", linewidth=1.0, zorder=5,
                       label=f"[F={h:.3f}] Human")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(panel[i])
        ax.legend(loc="lower left", framealpha=0.95, handlelength=2.0,
                  handletextpad=0.5)

    fig.tight_layout()
    out = figures_dir / "fig4_pr_curves.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white",
                pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] {out}")
    return True


# ============================================================================
# Figure 5 dan 6: perbandingan kualitatif
# ============================================================================
def fig_qualitative(dataset, data_root, output_root, figures_dir,
                    sample_ids, fig_num):
    layout = DATASET_LAYOUT[dataset]
    img_dir = data_root / layout["img_dir"]
    gt_dir = data_root / layout["gt_dir"]

    row_labels = ["Original", "Ground truth"] + \
                 [METHOD_LABELS[m] for m in METHOD_ORDER]
    n_rows, n_cols = len(row_labels), len(sample_ids)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.1 * n_cols, 2.1 * n_rows),
                             squeeze=False)

    for c, image_id in enumerate(sample_ids):
        img_path = find_image_file(img_dir, image_id)
        if img_path is None:
            print(f"  [WARN] {dataset}/{image_id}: citra tidak ditemukan")
            for r in range(n_rows):
                axes[r, c].axis("off")
            continue
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        axes[0, c].imshow(gray, cmap="gray", vmin=0, vmax=255)
        clean_axis(axes[0, c])
        axes[0, c].set_title(image_id, fontsize=8)

        gt = load_gt_display(dataset, gt_dir, image_id)
        if gt is not None:
            axes[1, c].imshow(gt, cmap="gray_r", vmin=0, vmax=255)
        clean_axis(axes[1, c])

        for r, method in enumerate(METHOD_ORDER, start=2):
            edge = load_edge_map(output_root, dataset, method, image_id)
            if edge is not None:
                axes[r, c].imshow(edge, cmap="gray_r", vmin=0, vmax=255)
            clean_axis(axes[r, c])

    for r, label in enumerate(row_labels):
        is_ours = label == "A-GWi"
        axes[r, 0].set_ylabel(label + (" (ours)" if is_ours else ""),
                              fontsize=9, rotation=90, labelpad=8,
                              va="center",
                              color=METHOD_COLORS["AGWi"] if is_ours else "black",
                              weight="bold" if is_ours else "normal")

    fig.tight_layout(pad=0.3)
    out = figures_dir / f"fig{fig_num}_qualitative_{dataset}.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    print(f"[OK] {out}")
    return True


# ============================================================================
# Figure 7: sebaran gradient activity (menggantikan density distribution)
# ============================================================================
def collect_rho_stats(data_root, dataset="BSDS500", max_images=None,
                      subsample=13):
    layout = DATASET_LAYOUT[dataset]
    paths = sorted((data_root / layout["img_dir"]).glob("*.jpg"))
    if not paths:
        for e in ["*.png", "*.jpeg"]:
            paths.extend(sorted((data_root / layout["img_dir"]).glob(e)))
        paths = sorted(paths)
    if max_images:
        paths = paths[:max_images]
    pooled, per_image_std = [], []
    for i, p in enumerate(paths, 1):
        gray, _ = preprocess(p)
        rho = estimate_density_sobel(gray)
        pooled.append(rho.ravel()[::subsample])
        per_image_std.append(float(rho.std()))
        if i % 50 == 0:
            print(f"  rho stats {i}/{len(paths)}")
    return np.concatenate(pooled), np.array(per_image_std)


def fig7_activity_distribution(data_root, figures_dir, max_images=None):
    pooled, stds = collect_rho_stats(data_root, "BSDS500", max_images)
    med, mean = float(np.median(pooled)), float(pooled.mean())

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    ax = axes[0]
    ax.hist(pooled, bins=120, color="#4477AA", edgecolor="none")
    ax.axvline(med, color="crimson", linestyle="--", linewidth=1.2,
               label=f"median = {med:.3f}")
    ax.axvline(mean, color="darkorange", linestyle="-.", linewidth=1.2,
               label=f"mean = {mean:.3f}")
    ax.axvline(0.5, color="black", linestyle=":", linewidth=1.2,
               label="sigmoid centre = 0.5")
    ax.set_xlabel(r"$\rho_L$ (local gradient activity)")
    ax.set_ylabel("Pixel count")
    ax.set_title("(a) Distribution across BSDS500")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.hist(stds, bins=30, color="#228833", edgecolor="none")
    thr = float(np.percentile(stds, 90))
    ax.axvline(thr, color="crimson", linestyle="--", linewidth=1.2,
               label=f"90th percentile = {thr:.3f}")
    ax.set_xlabel(r"Per-image standard deviation of $\rho_L$")
    ax.set_ylabel("Image count")
    ax.set_title("(b) Within-image heterogeneity")
    ax.legend(loc="upper right")

    fig.tight_layout()
    out = figures_dir / "fig7_activity_distribution.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white",
                pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] {out}  (median={med:.4f}, mean={mean:.4f})")
    return True


# ============================================================================
# Figure 8: validasi rho_L (poin 4), label bahasa Inggris
# ============================================================================
def fig8_activity_validation(data_root, figures_dir, window=15,
                             max_images=None, subsample=7):
    layout = DATASET_LAYOUT["BSDS500"]
    img_dir = data_root / layout["img_dir"]
    gt_dir = data_root / layout["gt_dir"]
    paths = sorted(img_dir.glob("*.jpg"))
    if max_images:
        paths = paths[:max_images]

    k = np.ones((window, window), dtype=np.float64) / (window * window)
    r_gt, r_ct, sc_rho, sc_gt = [], [], [], []
    for i, p in enumerate(paths, 1):
        gp = gt_dir / f"{p.stem}.mat"
        if not gp.exists():
            continue
        gray, _ = preprocess(p)
        rho = estimate_density_sobel(gray)
        gt = loadmat(str(gp))["groundTruth"]
        acc = None
        for j in range(gt.shape[1]):
            b = (gt[0, j]["Boundaries"][0, 0] > 0).astype(np.float64)
            acc = b if acc is None else np.maximum(acc, b)
        d_gt = cv2.filter2D(acc, cv2.CV_64F, k,
                            borderType=cv2.BORDER_REFLECT)
        m = cv2.boxFilter(gray, cv2.CV_64F, (window, window))
        m2 = cv2.boxFilter(gray ** 2, cv2.CV_64F, (window, window))
        contrast = np.sqrt(np.maximum(m2 - m ** 2, 0))

        a = rho.ravel()[::subsample]
        b_ = d_gt.ravel()[::subsample]
        c_ = contrast.ravel()[::subsample]
        r_gt.append(spearmanr(a, b_).statistic)
        r_ct.append(spearmanr(a, c_).statistic)
        if i <= 30:
            sc_rho.append(a[::5])
            sc_gt.append(b_[::5])
        if i % 50 == 0:
            print(f"  rho validation {i}/{len(paths)}")

    r_gt, r_ct = np.array(r_gt), np.array(r_ct)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax = axes[0]
    ax.hist(r_gt, bins=25, alpha=0.85, color="#4477AA",
            label=f"vs GT boundary density (median {np.median(r_gt):.3f})")
    ax.hist(r_ct, bins=25, alpha=0.65, color="#EE6677",
            label=f"vs local contrast (median {np.median(r_ct):.3f})")
    ax.set_xlabel("Spearman correlation")
    ax.set_ylabel("Image count")
    ax.set_title("(a) Per-image correlation")
    ax.legend(loc="upper left")

    ax = axes[1]
    if sc_rho:
        hb = ax.hexbin(np.concatenate(sc_rho), np.concatenate(sc_gt),
                       gridsize=60, bins="log", cmap="viridis")
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("log pixel count", fontsize=8)
    ax.set_xlabel(r"$\rho_L$ (local gradient activity)")
    ax.set_ylabel("Ground-truth boundary density")
    ax.set_title("(b) Pixel-level relation")

    fig.tight_layout()
    out = figures_dir / "fig8_activity_validation.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white",
                pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] {out}  (median rho_gt={np.median(r_gt):.3f}, "
          f"rho_contrast={np.median(r_ct):.3f})")
    return True


# ============================================================================
# Figure 9: diagnostik saturasi sigmoid (poin 7), label bahasa Inggris
# ============================================================================
def fig9_adaptation_diagnostic(data_root, figures_dir, max_images=50,
                               subsample=17):
    params = AGWiParams()
    layout = DATASET_LAYOUT["BSDS500"]
    paths = sorted((data_root / layout["img_dir"]).glob("*.jpg"))[:max_images]

    all_rho, all_f0 = [], []
    band = 0.01 * (params.f_max - params.f_min)
    frac = []
    for i, p in enumerate(paths, 1):
        gray, _ = preprocess(p)
        rho = estimate_density_sobel(gray)
        f0, _ = compute_f0_sigma_maps(rho, params)
        all_rho.append(rho.ravel()[::subsample])
        all_f0.append(f0.ravel()[::subsample])
        frac.append(float(np.mean(f0 < params.f_min + band)))
        if i % 25 == 0:
            print(f"  diagnostic {i}/{len(paths)}")
    rho_all = np.concatenate(all_rho)
    f0_all = np.concatenate(all_f0)
    pct = 100.0 * float(np.mean(frac))

    # kernel L2 norm terhadap f0
    half = params.kernel_size // 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    coef = SIGMA_COEF[params.sigma_relation]
    fs = np.linspace(params.f_min, params.f_max, 25)
    norms = []
    for f0v in fs:
        sx = coef / f0v
        sy = sx / params.aspect
        kk = np.exp(-0.5 * (xx ** 2 / sx ** 2 + yy ** 2 / sy ** 2)) \
            * np.sin(2 * np.pi * f0v * xx)
        norms.append(float(np.linalg.norm(kk)))
    norms = np.array(norms)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    ax = axes[0]
    ax.hist(rho_all, bins=120, color="#4477AA", edgecolor="none")
    ax.axvline(params.rho_center, color="crimson", linestyle="--",
               linewidth=1.3,
               label=f"sigmoid centre = {params.rho_center}")
    ax.axvline(float(np.median(rho_all)), color="black", linestyle=":",
               linewidth=1.3,
               label=f"median = {np.median(rho_all):.3f}")
    ax.set_xlabel(r"$\rho_L$")
    ax.set_ylabel("Pixel count")
    ax.set_title("(a) Gradient activity and sigmoid centre")
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.hist(f0_all, bins=120, color="#228833", edgecolor="none")
    ax.axvline(params.f_min, color="crimson", linestyle="--", linewidth=1.3,
               label=r"$f_{min}$")
    ax.axvline(params.f_max, color="black", linestyle="--", linewidth=1.3,
               label=r"$f_{max}$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$f_{0,adapt}$ (cycles/pixel)")
    ax.set_ylabel("Pixel count (log)")
    ax.set_title(f"(b) Adaptive frequency: {pct:.1f}% within 1% of "
                 r"$f_{min}$")
    ax.legend(loc="upper right")

    ax = axes[2]
    ax.plot(fs, norms, "o-", color="#EE6677", markersize=4)
    ax.set_yscale("log")
    ax.set_xlabel(r"$f_0$ (cycles/pixel)")
    ax.set_ylabel(r"Kernel $L_2$ norm (log)")
    ax.set_title(f"(c) Kernel norm varies by {norms[0] / norms[-1]:.0f}x")
    ax.grid(alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    out = figures_dir / "fig9_adaptation_diagnostic.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white",
                pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] {out}  ({pct:.2f}% pixel pada f_min)")
    return True


# ============================================================================
# Figure 10: peta parameter (poin 9 dan 15), tanpa entropi
# ============================================================================
def fig10_parameter_maps(data_root, output_root, figures_dir,
                         samples=None):
    samples = samples or PARAM_MAP_SAMPLES
    params = AGWiParams()
    coef = SIGMA_COEF[params.sigma_relation]
    s_lo, s_hi = coef / params.f_max, coef / params.f_min
    col_labels = [
        "Original",
        r"$\rho_L$" + "\n[0, 1]",
        r"$f_{0,adapt}$" + f"\n[{params.f_min:.2f}, {params.f_max:.2f}]",
        r"$\sigma_{adapt}$" + f"\n[{s_lo:.2f}, {s_hi:.2f}]",
        "GWi (static)", "A-GWi (ours)"]
    n_rows, n_cols = len(samples), len(col_labels)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.2 * n_cols, 1.9 * n_rows),
                             squeeze=False)

    for r, (dataset, image_id) in enumerate(samples):
        layout = DATASET_LAYOUT[dataset]
        img_path = find_image_file(data_root / layout["img_dir"], image_id)
        if img_path is None:
            print(f"  [WARN] fig10: {dataset}/{image_id} tidak ditemukan")
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue
        gray, gray_u8 = preprocess(img_path)
        rho = estimate_density_sobel(gray)
        f0, sigma = compute_f0_sigma_maps(rho, params)

        axes[r, 0].imshow(gray_u8, cmap="gray", vmin=0, vmax=255,
                          aspect="auto")
        axes[r, 1].imshow(rho, cmap="jet", vmin=0, vmax=1, aspect="auto")
        axes[r, 2].imshow(f0, cmap="viridis", vmin=params.f_min,
                          vmax=params.f_max, aspect="auto")
        axes[r, 3].imshow(sigma, cmap="magma", vmin=s_lo, vmax=s_hi,
                          aspect="auto")

        for c, method in [(4, "GWi"), (5, "AGWi")]:
            e = load_edge_map(output_root, dataset, method, image_id)
            if e is not None:
                axes[r, c].imshow(e, cmap="gray_r", vmin=0, vmax=255,
                                  aspect="auto")

        axes[r, 0].set_ylabel(f"{dataset}\n{image_id}", rotation=0,
                              fontsize=7, labelpad=30, ha="right",
                              va="center")
        for c in range(n_cols):
            clean_axis(axes[r, c])

    for c, label in enumerate(col_labels):
        ours = "ours" in label
        axes[0, c].set_title(label, fontsize=8.5,
                             color=METHOD_COLORS["AGWi"] if ours else "black",
                             weight="bold" if ours else "normal")

    fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.01,
                        wspace=0.03, hspace=0.03)
    out = figures_dir / "fig10_parameter_maps.png"
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
    plt.close(fig)
    print(f"[OK] {out}")
    return True


# ============================================================================
# Main
# ============================================================================
ALL_FIGURES = ["fig3", "fig4", "fig5", "fig6", "fig7", "fig8", "fig9",
               "fig10"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--eval-results", type=Path,
                    default=Path("./eval_results_v2"))
    ap.add_argument("--eval-grid", type=Path, default=Path("./eval_grid"))
    ap.add_argument("--figures-dir", type=Path, default=Path("./figures"))
    ap.add_argument("--figures", nargs="+", default=["all"],
                    help="'all' atau daftar: " + " ".join(ALL_FIGURES))
    ap.add_argument("--max-images", type=int, default=None,
                    help="Batasi jumlah citra untuk fig7 dan fig8")
    args = ap.parse_args()

    setup_matplotlib()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    sel = ALL_FIGURES if args.figures == ["all"] else args.figures
    bad = [f for f in sel if f not in ALL_FIGURES]
    if bad:
        raise SystemExit(f"Figure tidak dikenal: {bad}")

    print("=" * 72)
    print("Figure generator, naskah revisi JESA 44740")
    print("=" * 72)
    print(f"  Figure: {sel}")

    if "fig3" in sel:
        print("\n[fig3] parameter grid (poin 3)")
        fig3_parameter_grid(args.eval_grid, args.figures_dir)
    if "fig4" in sel:
        print("\n[fig4] PR curves (poin 1, 13)")
        fig4_pr_curves(args.eval_results, args.figures_dir)
    if "fig5" in sel:
        print("\n[fig5] qualitative BSDS500")
        fig_qualitative("BSDS500", args.data_root, args.output_root,
                        args.figures_dir,
                        QUALITATIVE_SAMPLES["BSDS500"], 5)
    if "fig6" in sel:
        print("\n[fig6] qualitative UDED")
        fig_qualitative("UDED", args.data_root, args.output_root,
                        args.figures_dir, QUALITATIVE_SAMPLES["UDED"], 6)
    if "fig7" in sel:
        print("\n[fig7] activity distribution (poin 4)")
        fig7_activity_distribution(args.data_root, args.figures_dir,
                                   args.max_images)
    if "fig8" in sel:
        print("\n[fig8] activity validation (poin 4)")
        fig8_activity_validation(args.data_root, args.figures_dir,
                                 max_images=args.max_images)
    if "fig9" in sel:
        print("\n[fig9] adaptation diagnostic (poin 7)")
        fig9_adaptation_diagnostic(args.data_root, args.figures_dir)
    if "fig10" in sel:
        print("\n[fig10] parameter maps (poin 9, 15)")
        fig10_parameter_maps(args.data_root, args.output_root,
                             args.figures_dir)

    print("\nSelesai. Figure 1 dan Figure 2 adalah diagram, dibuat manual.")
    print("Figure 2 masih perlu diperbaiki: Stage 3 menyebut kernel size k")
    print("dimodulasi oleh rho_L, padahal k tetap 7 x 7.")


if __name__ == "__main__":
    main()
