"""
error_metrics.py - Pengganti analisis entropi Shannon
Author: Kukuh Yudhistiro, 2026

Menjawab R1-9 dan R2-9 (nilai entropi 15.88 sampai 16.51 mustahil untuk
peta 8-bit) sekaligus R1-15 dan R2-15 (klaim kausal tanpa pengukuran).

Sumber bug entropi: gwc_vs_gwi_comparison.py memanggil
skimage.measure.shannon_entropy pada array float yang sudah dinormalkan.
Fungsi itu menghitung entropi atas nilai unik. Pada array float, hampir
setiap piksel adalah simbol unik, sehingga entropi mendekati
log2(jumlah piksel). Untuk citra 481x321 batasnya log2(154401) = 17.24 bit,
dan nilai 16.26 sampai 16.51 jatuh persis di bawah batas itu. Angka tersebut
mengukur keragaman nilai floating point, bukan kandungan informasi tepi.

Metrik pengganti yang punya arti langsung:
  1. precision_at_recall  : presisi pada recall tetap (default 0.5).
     Mengukur klaim "over-detection berkurang" secara kuantitatif.
  2. localization_error   : rata-rata jarak piksel prediksi tercocokkan ke
     piksel GT pasangannya, pada ambang ODS, dalam piksel.
  3. edge_width           : rasio piksel di atas ambang terhadap piksel
     setelah penipisan. Nilai mendekati 1 berarti respons sudah tipis.

Pemakaian:
  python error_metrics.py --data-root ./data --output-root ./output \
      --results-dir ./eval_results_v2 --dataset BSDS500 \
      --methods AGWi GWi GWC --target-recall 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from scipy.spatial import KDTree
from skimage.morphology import thin as sk_thin

EPS = 1e-10

LAYOUTS = {
    "BSDS500": {"gt_dir": "BSDS500/groundTruth/test", "gt_format": "mat"},
    "UDED": {"gt_dir": "UDED/gt", "gt_format": "png"},
}


def load_gt(gt_path, fmt):
    if fmt == "mat":
        mat = loadmat(str(gt_path))
        gt = mat["groundTruth"]
        return [(gt[0, i]["Boundaries"][0, 0] > 0).astype(bool)
                for i in range(gt.shape[1])]
    arr = np.array(Image.open(str(gt_path)).convert("L"))
    return [(arr > 127).astype(bool)]


def load_pred(p):
    return np.array(Image.open(str(p)).convert("L")).astype(np.float32) / 255.0


def match_with_distance(pred_pts, gt_pts, max_dist_px):
    """Returns (n_pred_matched, n_gt_matched, mean_distance_of_matches)."""
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0, 0, np.nan
    tree = KDTree(gt_pts)
    cand = tree.query_ball_point(pred_pts, r=max_dist_px)
    pi, gi, cost = [], [], []
    for p_i, cl in enumerate(cand):
        for g_i in cl:
            pi.append(p_i)
            gi.append(g_i)
            cost.append(float(np.linalg.norm(pred_pts[p_i] - gt_pts[g_i])))
    if not cost:
        return 0, 0, np.nan
    order = np.argsort(cost)
    pm = np.zeros(len(pred_pts), dtype=bool)
    gm = np.zeros(len(gt_pts), dtype=bool)
    dists = []
    for idx in order:
        p, g = pi[idx], gi[idx]
        if not pm[p] and not gm[g]:
            pm[p] = True
            gm[g] = True
            dists.append(cost[idx])
    return int(pm.sum()), int(gm.sum()), (float(np.mean(dists))
                                          if dists else np.nan)


def image_metrics(pred, gts, thresholds, target_recall, max_dist_frac=0.0075):
    h, w = pred.shape
    max_dist_px = max_dist_frac * np.sqrt(h * h + w * w)
    gt_pts_list = [np.argwhere(sk_thin(g)) for g in gts]
    total_gt = sum(len(g) for g in gt_pts_list)

    recs, precs, dists, widths = [], [], [], []
    for t in thresholds:
        bmap = pred >= t
        n_raw = int(bmap.sum())
        thin = sk_thin(bmap)
        pred_pts = np.argwhere(thin)
        n_pred = len(pred_pts)
        if n_pred == 0:
            recs.append(0.0)
            precs.append(0.0)
            dists.append(np.nan)
            widths.append(np.nan)
            continue
        matched_any = np.zeros(n_pred, dtype=bool)
        gt_matched, dsum, dn = 0, 0.0, 0
        for gt_pts in gt_pts_list:
            if len(gt_pts) == 0:
                continue
            tree = KDTree(gt_pts)
            cand = tree.query_ball_point(pred_pts, r=max_dist_px)
            pi, gi, cost = [], [], []
            for p_i, cl in enumerate(cand):
                for g_i in cl:
                    pi.append(p_i)
                    gi.append(g_i)
                    cost.append(float(np.linalg.norm(pred_pts[p_i]
                                                     - gt_pts[g_i])))
            if not cost:
                continue
            order = np.argsort(cost)
            pm = np.zeros(n_pred, dtype=bool)
            gm = np.zeros(len(gt_pts), dtype=bool)
            for idx in order:
                p, g = pi[idx], gi[idx]
                if not pm[p] and not gm[g]:
                    pm[p] = True
                    gm[g] = True
                    dsum += cost[idx]
                    dn += 1
            matched_any |= pm
            gt_matched += int(gm.sum())
        recs.append(gt_matched / max(total_gt, 1))
        precs.append(int(matched_any.sum()) / n_pred)
        dists.append(dsum / dn if dn else np.nan)
        widths.append(n_raw / max(n_pred, 1))

    recs = np.array(recs)
    precs = np.array(precs)
    # presisi pada recall target, interpolasi linear pada kurva
    order = np.argsort(recs)
    p_at_r = (float(np.interp(target_recall, recs[order], precs[order]))
              if recs.max() >= target_recall else np.nan)
    f = 2 * precs * recs / np.maximum(precs + recs, EPS)
    b = int(np.argmax(f))
    return {
        "precision_at_recall": p_at_r,
        "localization_error_px": dists[b],
        "edge_width_ratio": widths[b],
        "best_f": float(f[b]),
        "best_threshold": float(thresholds[b]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, default=Path("."))
    ap.add_argument("--dataset", default="BSDS500")
    ap.add_argument("--methods", nargs="+", default=["AGWi", "GWi", "GWC"])
    ap.add_argument("--n-thresholds", type=int, default=49)
    ap.add_argument("--target-recall", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    lay = LAYOUTS[args.dataset]
    gt_dir = args.data_root / lay["gt_dir"]
    thresholds = np.linspace(1.0 / (args.n_thresholds + 1),
                             1.0 - 1.0 / (args.n_thresholds + 1),
                             args.n_thresholds)
    rows = []
    for method in args.methods:
        mdir = args.output_root / args.dataset / method
        if not mdir.exists():
            print(f"[WARN] {mdir} tidak ada")
            continue
        preds = sorted(mdir.glob("*.png"))
        if args.max_images:
            preds = preds[:args.max_images]
        print(f"\n[{method}] {len(preds)} citra")
        for i, pp in enumerate(preds, 1):
            gt_path = (gt_dir / f"{pp.stem}.mat" if lay["gt_format"] == "mat"
                       else gt_dir / f"{pp.stem}.png")
            if not gt_path.exists():
                continue
            m = image_metrics(load_pred(pp), load_gt(gt_path, lay["gt_format"]),
                              thresholds, args.target_recall)
            m.update({"dataset": args.dataset, "method": method,
                      "image": pp.stem})
            rows.append(m)
            if i % 25 == 0:
                print(f"  {i}/{len(preds)}")

    df = pd.DataFrame(rows)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / f"error_metrics_{args.dataset}.csv"
    df.to_csv(out, index=False)

    summ = df.groupby("method").agg(
        precision_at_R=("precision_at_recall", "mean"),
        precision_at_R_std=("precision_at_recall", "std"),
        localization_px=("localization_error_px", "mean"),
        edge_width=("edge_width_ratio", "mean"),
    ).round(4)
    print("\n" + "=" * 72)
    print(f"METRIK KESALAHAN ({args.dataset}, "
          f"presisi diukur pada recall = {args.target_recall})")
    print("=" * 72)
    print(summ.to_string())
    print(f"\n[OK] {out}")
    print("\nPakai tabel ini untuk menggantikan paragraf entropi pada Figure 7.")
    print("Presisi yang lebih tinggi pada recall tetap adalah bukti langsung")
    print("untuk klaim pengurangan over-detection.")


if __name__ == "__main__":
    main()
