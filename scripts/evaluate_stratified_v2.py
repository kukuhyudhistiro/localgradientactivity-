"""
evaluate_stratified_v2.py - Analisis terstratifikasi yang diperbaiki
Author: Kukuh Yudhistiro, 2026

Menjawab R1-8 dan R2-8. Dua cacat pada versi lama:

  (a) Matching dilakukan secara global lalu hasilnya dipecah per stratum.
      Piksel prediksi di stratum LOW bisa dipasangkan dengan piksel GT di
      stratum HIGH, sehingga presisi dan recall per stratum dihitung pada
      populasi yang berbeda dan nilainya tidak sebanding.
      Versi ini MEMBATASI prediksi DAN ground truth ke masker stratum yang
      sama sebelum matching.

  (b) Ambang tertil dihitung dari test set masing-masing dataset, sehingga
      memakai statistik test set dan membuat perbandingan lintas dataset
      tidak sah. Versi ini menghitung tertil sekali dari BSDS500 train,
      menyimpannya ke JSON, dan menerapkan ambang absolut yang sama ke
      BSDS500 test maupun UDED.

Catatan penamaan: nilai per stratum BUKAN ODS dalam arti protokol Berkeley.
Laporkan sebagai F_s (stratum F-measure) dan nyatakan bahwa nilainya tidak
sebanding dengan ODS agregat.

Pemakaian:
  # langkah 1: hitung tertil dari train split, sekali saja
  python evaluate_stratified_v2.py --data-root ./data \
      --compute-tertiles --tertile-source BSDS500_train \
      --tertile-file tertiles.json

  # langkah 2: evaluasi dengan ambang tetap
  python evaluate_stratified_v2.py --data-root ./data --output-root ./output \
      --results-dir ./eval_results_v2 --datasets BSDS500 UDED \
      --methods AGWi GWi GWC --tertile-file tertiles.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from scipy.spatial import KDTree
from skimage.morphology import thin as sk_thin

from methods_v2 import preprocess, estimate_density_sobel

EPS = 1e-10
STRATA = ["LOW", "MID", "HIGH"]

LAYOUTS = {
    "BSDS500": {"img_dir": "BSDS500/images/test",
                "gt_dir": "BSDS500/groundTruth/test",
                "gt_format": "mat", "img_ext": ".jpg"},
    "BSDS500_train": {"img_dir": "BSDS500/images/train",
                      "gt_dir": "BSDS500/groundTruth/train",
                      "gt_format": "mat", "img_ext": ".jpg"},
    "UDED": {"img_dir": "UDED/imgs", "gt_dir": "UDED/gt",
             "gt_format": "png", "img_ext": ".jpg"},
}


def load_gt(gt_path, fmt):
    if fmt == "mat":
        mat = loadmat(str(gt_path))
        gt = mat["groundTruth"]
        return [(gt[0, i]["Boundaries"][0, 0] > 0).astype(bool)
                for i in range(gt.shape[1])]
    arr = np.array(Image.open(str(gt_path)).convert("L"))
    return [(arr > 127).astype(bool)]


def load_pred(png_path):
    arr = np.array(Image.open(str(png_path)).convert("L"))
    return arr.astype(np.float32) / 255.0


def find_image(img_dir, stem, ext):
    p = img_dir / f"{stem}{ext}"
    if p.exists():
        return p
    for e in [".jpg", ".png", ".jpeg", ".JPG"]:
        q = img_dir / f"{stem}{e}"
        if q.exists():
            return q
    raise FileNotFoundError(f"{stem} tidak ditemukan di {img_dir}")


# ============================================================================
# Tertil dari train split
# ============================================================================
def compute_tertiles(data_root, source, max_images=None, subsample=11):
    lay = LAYOUTS[source]
    img_dir = data_root / lay["img_dir"]
    paths = sorted(img_dir.glob("*" + lay["img_ext"]))
    if not paths:
        for e in ["*.png", "*.jpeg"]:
            paths.extend(sorted(img_dir.glob(e)))
    paths = sorted(paths)[:max_images] if max_images else sorted(paths)
    vals = []
    for i, p in enumerate(paths, 1):
        gray, _ = preprocess(p)
        vals.append(estimate_density_sobel(gray).ravel()[::subsample])
        if i % 25 == 0:
            print(f"  tertil {i}/{len(paths)}")
    pooled = np.concatenate(vals)
    return {
        "source": source,
        "n_images": len(paths),
        "t33": float(np.percentile(pooled, 100.0 / 3.0)),
        "t66": float(np.percentile(pooled, 200.0 / 3.0)),
        "median": float(np.median(pooled)),
        "mean": float(pooled.mean()),
    }


# ============================================================================
# Matching dibatasi masker stratum
# ============================================================================
def match_masked(pred_pts, gt_pts, max_dist_px):
    n_pred, n_gt = len(pred_pts), len(gt_pts)
    if n_pred == 0 or n_gt == 0:
        return 0, 0
    tree = KDTree(gt_pts)
    cand = tree.query_ball_point(pred_pts, r=max_dist_px)
    pi, gi, cost = [], [], []
    for p_i, cl in enumerate(cand):
        for g_i in cl:
            pi.append(p_i)
            gi.append(g_i)
            cost.append(np.linalg.norm(pred_pts[p_i] - gt_pts[g_i]))
    if not cost:
        return 0, 0
    order = np.argsort(cost)
    pm = np.zeros(n_pred, dtype=bool)
    gm = np.zeros(n_gt, dtype=bool)
    for idx in order:
        p, g = pi[idx], gi[idx]
        if not pm[p] and not gm[g]:
            pm[p] = True
            gm[g] = True
    return int(pm.sum()), int(gm.sum())


def evaluate_image_stratified(pred, gts, rho, t33, t66, thresholds,
                              max_dist_frac=0.0075):
    h, w = pred.shape
    max_dist_px = max_dist_frac * np.sqrt(h * h + w * w)
    strat = np.where(rho < t33, 0, np.where(rho < t66, 1, 2))

    gt_cache = []          # per stratum: list of gt_pts per annotator
    sumR = [0, 0, 0]
    for gt in gts:
        gt_pts_all = np.argwhere(sk_thin(gt))
        per_s = []
        for s in range(3):
            if len(gt_pts_all) == 0:
                per_s.append(np.empty((0, 2), dtype=int))
                continue
            m = strat[gt_pts_all[:, 0], gt_pts_all[:, 1]] == s
            pts = gt_pts_all[m]
            per_s.append(pts)
            sumR[s] += len(pts)
        gt_cache.append(per_s)

    counts = {s: [] for s in range(3)}
    for t in thresholds:
        pred_pts_all = np.argwhere(sk_thin(pred >= t))
        for s in range(3):
            if len(pred_pts_all) == 0:
                counts[s].append((0, sumR[s], 0, 0))
                continue
            m = strat[pred_pts_all[:, 0], pred_pts_all[:, 1]] == s
            pred_pts = pred_pts_all[m]
            n_pred = len(pred_pts)
            if n_pred == 0:
                counts[s].append((0, sumR[s], 0, 0))
                continue
            matched_any = np.zeros(n_pred, dtype=bool)
            gt_matched = 0
            for per_s in gt_cache:
                gt_pts = per_s[s]
                if len(gt_pts) == 0:
                    continue
                # matching dijalankan hanya di dalam stratum
                tree = KDTree(gt_pts)
                cand = tree.query_ball_point(pred_pts, r=max_dist_px)
                pi, gi, cost = [], [], []
                for p_i, cl in enumerate(cand):
                    for g_i in cl:
                        pi.append(p_i)
                        gi.append(g_i)
                        cost.append(np.linalg.norm(pred_pts[p_i] - gt_pts[g_i]))
                if not cost:
                    continue
                order = np.argsort(cost)
                gm = np.zeros(len(gt_pts), dtype=bool)
                pm = np.zeros(n_pred, dtype=bool)
                for idx in order:
                    p, g = pi[idx], gi[idx]
                    if not pm[p] and not gm[g]:
                        pm[p] = True
                        gm[g] = True
                matched_any |= pm
                gt_matched += int(gm.sum())
            counts[s].append((gt_matched, sumR[s],
                              int(matched_any.sum()), n_pred))
    return counts


@dataclass
class StratScore:
    dataset: str
    method: str
    stratum: str
    n_images: int
    t33: float
    t66: float
    pct_gt_in_stratum: float
    f_s: float
    precision_at_best: float
    recall_at_best: float


def aggregate_stratum(per_image, dataset, method, stratum, t33, t66,
                      pct_gt):
    arr = np.asarray(per_image, dtype=np.float64)
    pooled = arr.sum(axis=0)
    r = pooled[:, 0] / np.maximum(pooled[:, 1], EPS)
    p = pooled[:, 2] / np.maximum(pooled[:, 3], EPS)
    f = 2 * p * r / np.maximum(p + r, EPS)
    i = int(np.argmax(f))
    return StratScore(dataset, method, stratum, arr.shape[0], t33, t66,
                      pct_gt, float(f[i]), float(p[i]), float(r[i]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--results-dir", type=Path, default=Path("."))
    ap.add_argument("--datasets", nargs="+", default=["BSDS500", "UDED"])
    ap.add_argument("--methods", nargs="+", default=["AGWi", "GWi", "GWC"])
    ap.add_argument("--n-thresholds", type=int, default=99)
    ap.add_argument("--max-dist", type=float, default=0.0075)
    ap.add_argument("--tertile-file", type=Path, default=Path("tertiles.json"))
    ap.add_argument("--compute-tertiles", action="store_true")
    ap.add_argument("--tertile-source", default="BSDS500_train")
    args = ap.parse_args()

    if args.compute_tertiles:
        print(f"[INFO] Menghitung tertil dari {args.tertile_source}")
        tert = compute_tertiles(args.data_root, args.tertile_source)
        args.tertile_file.write_text(json.dumps(tert, indent=2))
        print(json.dumps(tert, indent=2))
        print(f"[OK] {args.tertile_file}")
        return

    if not args.tertile_file.exists():
        raise SystemExit(
            f"{args.tertile_file} tidak ada. Jalankan --compute-tertiles dulu.")
    tert = json.loads(args.tertile_file.read_text())
    t33, t66 = tert["t33"], tert["t66"]
    print(f"[INFO] Ambang tetap dari {tert['source']}: "
          f"t33={t33:.4f}, t66={t66:.4f}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(1.0 / (args.n_thresholds + 1),
                             1.0 - 1.0 / (args.n_thresholds + 1),
                             args.n_thresholds)
    all_scores = []

    for dataset in args.datasets:
        lay = LAYOUTS[dataset]
        img_dir = args.data_root / lay["img_dir"]
        gt_dir = args.data_root / lay["gt_dir"]
        pred_root = args.output_root / dataset
        first = pred_root / args.methods[0]
        if not first.exists():
            print(f"[WARN] {first} tidak ada, {dataset} dilewati")
            continue
        stems = [p.stem for p in sorted(first.glob("*.png"))]
        print(f"\n{'=' * 72}\n  {dataset}: {len(stems)} citra\n{'=' * 72}")

        cache = {}
        for stem in stems:
            gray, _ = preprocess(find_image(img_dir, stem, lay["img_ext"]))
            gt_path = (gt_dir / f"{stem}.mat" if lay["gt_format"] == "mat"
                       else gt_dir / f"{stem}.png")
            cache[stem] = (estimate_density_sobel(gray),
                           load_gt(gt_path, lay["gt_format"]))

        # proporsi piksel GT per stratum, sama untuk semua metode
        gt_counts = [0, 0, 0]
        for stem in stems:
            rho, gts = cache[stem]
            strat = np.where(rho < t33, 0, np.where(rho < t66, 1, 2))
            for gt in gts:
                pts = np.argwhere(sk_thin(gt))
                if len(pts):
                    s = strat[pts[:, 0], pts[:, 1]]
                    for k in range(3):
                        gt_counts[k] += int((s == k).sum())
        tot = max(sum(gt_counts), 1)
        pct = [100.0 * c / tot for c in gt_counts]
        print(f"  Persentase piksel GT per stratum: "
              f"LOW {pct[0]:.1f}, MID {pct[1]:.1f}, HIGH {pct[2]:.1f}")

        for method in args.methods:
            mdir = pred_root / method
            if not mdir.exists():
                print(f"  [WARN] {method} tidak ada")
                continue
            print(f"\n  [{method}]")
            per_stratum = {s: [] for s in range(3)}
            for i, stem in enumerate(stems, 1):
                pp = mdir / f"{stem}.png"
                if not pp.exists():
                    continue
                rho, gts = cache[stem]
                c = evaluate_image_stratified(load_pred(pp), gts, rho,
                                              t33, t66, thresholds,
                                              args.max_dist)
                for s in range(3):
                    per_stratum[s].append(c[s])
                if i % 25 == 0:
                    print(f"    {i}/{len(stems)}")
            for s in range(3):
                if per_stratum[s]:
                    sc = aggregate_stratum(per_stratum[s], dataset, method,
                                           STRATA[s], t33, t66, pct[s])
                    all_scores.append(sc)
                    print(f"    {STRATA[s]:5s} F_s={sc.f_s:.4f} "
                          f"(P={sc.precision_at_best:.3f}, "
                          f"R={sc.recall_at_best:.3f})")

    if all_scores:
        df = pd.DataFrame([asdict(s) for s in all_scores])
        out = args.results_dir / "stratified_v2.csv"
        df.to_csv(out, index=False)
        print(f"\n[OK] {out}")
        piv = df.pivot_table(index=["dataset", "method"], columns="stratum",
                             values="f_s")
        print(piv.to_string())


if __name__ == "__main__":
    main()
