"""
evaluate_v2.py - Evaluasi Berkeley yang diperbaiki untuk revisi JESA 44740
Author: Kukuh Yudhistiro, 2026

Perubahan terhadap evaluate.py lama:

  [R1-1, R2-1] OIS dihitung dengan skema agregasi yang SAMA dengan ODS.
      Lama : OIS = rata-rata F terbaik per citra (mean-of-F).
      Baru : OIS = F tunggal dari hitungan yang diakumulasi melintasi citra
             pada ambang terbaik masing-masing citra (skema Arbelaez).
      Mencampur pooled-ODS dengan mean-of-F OIS adalah penyebab
      Canny UDED memberi ODS 0.704 > OIS 0.683.
      Kolom ois_f_mean tetap dilaporkan sebagai pembanding transparansi.

  [R1-6, R2-6] AP dihitung dari kurva PR tingkat dataset (bukan rata-rata AP
      per citra), sesuai definisi resmi. Kolom ap_mean_per_image dipertahankan.

  [R1-13, R2-13] Hitungan per citra per ambang diekspor ke .npz sehingga
      bootstrap dan uji berpasangan dapat dijalankan tanpa evaluasi ulang.

  [R1-6, R2-6] Mode --calibrate mencetak perbandingan terhadap nilai rujukan
      Canny pada BSDS500 (ODS 0.600, OIS 0.640, AP 0.580).

Pemakaian:
  python evaluate_v2.py --data-root ./data --output-root ./output \
      --results-dir ./eval_results_v2 --datasets BSDS500 UDED \
      --methods AGWi GWi GWC Canny Sobel PC --n-thresholds 99
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from scipy.spatial import KDTree
from skimage.morphology import thin as sk_thin

EPS = 1e-10
CANNY_REFERENCE = {"ods": 0.600, "ois": 0.640, "ap": 0.580}


# ============================================================================
# Loader
# ============================================================================
def load_gt_bsds_mat(mat_path):
    mat = loadmat(str(mat_path))
    gt = mat["groundTruth"]
    return [(gt[0, i]["Boundaries"][0, 0] > 0).astype(bool)
            for i in range(gt.shape[1])]


def load_gt_png(png_path):
    arr = np.array(Image.open(str(png_path)).convert("L"))
    return [(arr > 127).astype(bool)]


def load_pred_png(png_path):
    arr = np.array(Image.open(str(png_path)).convert("L"))
    return arr.astype(np.float32) / 255.0


# ============================================================================
# Matching (tidak diubah, sudah sesuai Arbelaez per-annotator)
# ============================================================================
def match_single_annotator(pred_pts, gt_pts, tree_gt, max_dist_px):
    n_pred, n_gt = len(pred_pts), len(gt_pts)
    pred_matched = np.zeros(n_pred, dtype=bool)
    if n_pred == 0 or n_gt == 0 or tree_gt is None:
        return pred_matched, 0

    candidates = tree_gt.query_ball_point(pred_pts, r=max_dist_px)
    pred_idx, gt_idx, cost = [], [], []
    for p_i, cand in enumerate(candidates):
        for g_i in cand:
            d = np.linalg.norm(pred_pts[p_i] - gt_pts[g_i])
            pred_idx.append(p_i)
            gt_idx.append(g_i)
            cost.append(d)
    if not cost:
        return pred_matched, 0

    order = np.argsort(cost)
    gt_matched = np.zeros(n_gt, dtype=bool)
    for idx in order:
        p, g = pred_idx[idx], gt_idx[idx]
        if not pred_matched[p] and not gt_matched[g]:
            pred_matched[p] = True
            gt_matched[g] = True
    return pred_matched, int(gt_matched.sum())


def evaluate_threshold(pred_bool, gt_cache, total_gt, max_dist_px):
    pred_thin = sk_thin(pred_bool)
    pred_pts = np.argwhere(pred_thin)
    n_pred = len(pred_pts)
    if n_pred == 0:
        return 0, total_gt, 0, 0
    pred_matched_any = np.zeros(n_pred, dtype=bool)
    total_gt_matched = 0
    for gt_pts, tree_gt in gt_cache:
        pm, n_gt_matched = match_single_annotator(pred_pts, gt_pts,
                                                  tree_gt, max_dist_px)
        pred_matched_any |= pm
        total_gt_matched += n_gt_matched
    return (int(total_gt_matched), total_gt,
            int(pred_matched_any.sum()), n_pred)


def evaluate_image(pred, gts, thresholds, max_dist_frac=0.0075):
    h, w = pred.shape
    max_dist_px = max_dist_frac * np.sqrt(h * h + w * w)
    gt_cache, total_gt = [], 0
    for gt in gts:
        gt_pts = np.argwhere(sk_thin(gt))
        tree = KDTree(gt_pts) if len(gt_pts) > 0 else None
        gt_cache.append((gt_pts, tree))
        total_gt += len(gt_pts)
    return [evaluate_threshold(pred >= t, gt_cache, total_gt, max_dist_px)
            for t in thresholds]


# ============================================================================
# Agregasi
# ============================================================================
def _prf(counts):
    """counts: array (..., 4) berisi cntR, sumR, cntP, sumP."""
    c = np.asarray(counts, dtype=np.float64)
    r = c[..., 0] / np.maximum(c[..., 1], EPS)
    p = c[..., 2] / np.maximum(c[..., 3], EPS)
    f = 2 * p * r / np.maximum(p + r, EPS)
    return p, r, f


def interpolated_ap(precision, recall):
    """AUC kurva PR tingkat dataset dengan presisi yang dimonotonkan."""
    order = np.argsort(recall)
    r = recall[order]
    p = precision[order]
    # monotonic envelope dari kanan (definisi interpolated precision)
    p_env = np.maximum.accumulate(p[::-1])[::-1]
    r_ext = np.concatenate(([0.0], r))
    return float(np.sum(np.diff(r_ext) * p_env))


@dataclass
class Score:
    dataset: str
    method: str
    n_images: int
    n_thresholds: int
    ods_threshold: float
    ods_precision: float
    ods_recall: float
    ods_f: float
    ois_precision: float
    ois_recall: float
    ois_f: float          # skema Arbelaez, akumulasi hitungan
    ois_f_mean: float     # skema lama, rata-rata F per citra (transparansi)
    ap: float             # AUC kurva PR tingkat dataset
    ap_mean_per_image: float
    f_at_ods_mean: float
    f_at_ods_std: float


def aggregate(per_image_counts, thresholds, dataset, method):
    arr = np.asarray(per_image_counts, dtype=np.float64)  # (n_img, n_thr, 4)
    n_img, n_thr, _ = arr.shape

    # --- ODS: akumulasi hitungan melintasi citra pada setiap ambang ---
    pooled = arr.sum(axis=0)                     # (n_thr, 4)
    p_t, r_t, f_t = _prf(pooled)
    ods_i = int(np.argmax(f_t))

    # --- OIS: ambang terbaik per citra, lalu akumulasi hitungannya ---
    p_i, r_i, f_i = _prf(arr)                    # (n_img, n_thr)
    best_t = np.argmax(f_i, axis=1)              # (n_img,)
    ois_counts = arr[np.arange(n_img), best_t].sum(axis=0)
    ois_p, ois_r, ois_f = _prf(ois_counts)
    ois_f_mean = float(np.mean(f_i[np.arange(n_img), best_t]))

    # --- AP ---
    ap_dataset = interpolated_ap(p_t, r_t)
    ap_per_image = [interpolated_ap(p_i[k], r_i[k]) for k in range(n_img)]

    f_at_ods = f_i[:, ods_i]

    score = Score(
        dataset=dataset, method=method, n_images=n_img, n_thresholds=n_thr,
        ods_threshold=float(thresholds[ods_i]),
        ods_precision=float(p_t[ods_i]), ods_recall=float(r_t[ods_i]),
        ods_f=float(f_t[ods_i]),
        ois_precision=float(ois_p), ois_recall=float(ois_r),
        ois_f=float(ois_f), ois_f_mean=ois_f_mean,
        ap=float(ap_dataset), ap_mean_per_image=float(np.mean(ap_per_image)),
        f_at_ods_mean=float(np.mean(f_at_ods)),
        f_at_ods_std=float(np.std(f_at_ods, ddof=1)) if n_img > 1 else 0.0,
    )
    pr_df = pd.DataFrame({"threshold": thresholds, "precision": p_t,
                          "recall": r_t, "f_measure": f_t})
    return score, pr_df, arr


# ============================================================================
# Driver
# ============================================================================
DATASET_LAYOUTS = {
    "BSDS500": {"gt_subdir": "groundTruth/test", "gt_format": "mat"},
    "BSDS500_train": {"gt_subdir": "groundTruth/train", "gt_format": "mat"},
    "BSDS500_val": {"gt_subdir": "groundTruth/val", "gt_format": "mat"},
    "UDED": {"gt_subdir": "gt", "gt_format": "png"},
    "BIPED": {"gt_subdir": "edge_maps/test", "gt_format": "png"},
}


def evaluate_method_dataset(pred_dir, gt_dir, dataset, method,
                            max_dist_frac, n_thresholds, gt_format):
    thresholds = np.linspace(1.0 / (n_thresholds + 1),
                             1.0 - 1.0 / (n_thresholds + 1), n_thresholds)
    pred_files = sorted(pred_dir.glob("*.png"))
    if not pred_files:
        raise FileNotFoundError(f"No predictions in {pred_dir}")

    per_image, stems, n_skip = [], [], 0
    for pred_path in pred_files:
        stem = pred_path.stem
        gt_path = (gt_dir / f"{stem}.mat" if gt_format == "mat"
                   else gt_dir / f"{stem}.png")
        if not gt_path.exists():
            n_skip += 1
            continue
        gts = (load_gt_bsds_mat(gt_path) if gt_format == "mat"
               else load_gt_png(gt_path))
        pred = load_pred_png(pred_path)
        per_image.append(evaluate_image(pred, gts, thresholds, max_dist_frac))
        stems.append(stem)
        if len(per_image) % 25 == 0:
            print(f"  [{dataset}/{method}] {len(per_image)}/{len(pred_files)}")

    print(f"  Processed {len(per_image)}, skipped {n_skip}")
    if not per_image:
        raise RuntimeError(f"No images evaluated for {dataset}/{method}")
    score, pr_df, arr = aggregate(per_image, thresholds, dataset, method)
    return score, pr_df, arr, stems, thresholds


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--data-root", type=Path, required=True)
    ap_.add_argument("--output-root", type=Path, required=True)
    ap_.add_argument("--results-dir", type=Path, required=True)
    ap_.add_argument("--datasets", nargs="+", default=["BSDS500", "UDED"])
    ap_.add_argument("--methods", nargs="+", default=None)
    ap_.add_argument("--n-thresholds", type=int, default=99)
    ap_.add_argument("--max-dist", type=float, default=0.0075)
    ap_.add_argument("--save-counts", action="store_true", default=True,
                     help="Simpan hitungan per citra ke .npz untuk statistik")
    args = ap_.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    counts_dir = args.results_dir / "counts"
    counts_dir.mkdir(exist_ok=True)
    all_scores = []

    for dataset in args.datasets:
        if dataset not in DATASET_LAYOUTS:
            print(f"[WARN] Unknown dataset {dataset}")
            continue
        layout = DATASET_LAYOUTS[dataset]
        gt_dir = args.data_root / dataset.split("_")[0] / layout["gt_subdir"]
        pred_root = args.output_root / dataset
        if not gt_dir.exists() or not pred_root.exists():
            print(f"[WARN] Skip {dataset}: {gt_dir} atau {pred_root} tidak ada")
            continue

        if args.methods is None:
            method_dirs = sorted(d for d in pred_root.iterdir() if d.is_dir())
        else:
            method_dirs = [pred_root / m for m in args.methods
                           if (pred_root / m).exists()]

        print(f"\n{'=' * 72}\n  Dataset: {dataset}\n{'=' * 72}")
        for method_dir in method_dirs:
            method = method_dir.name
            print(f"\n[INFO] {dataset}/{method}")
            try:
                score, pr_df, arr, stems, thr = evaluate_method_dataset(
                    method_dir, gt_dir, dataset, method,
                    args.max_dist, args.n_thresholds, layout["gt_format"])
            except Exception as e:
                print(f"  [FAIL] {e}")
                continue
            all_scores.append(score)
            pr_df.to_csv(args.results_dir /
                         f"pr_curve_{dataset}_{method}.csv", index=False)
            if args.save_counts:
                np.savez_compressed(
                    counts_dir / f"{dataset}_{method}.npz",
                    counts=arr, thresholds=thr, stems=np.array(stems))
            print(f"  ODS={score.ods_f:.4f}  OIS={score.ois_f:.4f}  "
                  f"AP={score.ap:.4f}   (OIS_mean_lama={score.ois_f_mean:.4f})")
            if score.ois_f < score.ods_f - 1e-9:
                print("  [ALERT] OIS < ODS. Periksa agregasi atau data.")
            if method.lower().startswith("canny") and dataset == "BSDS500":
                d = score.ods_f - CANNY_REFERENCE["ods"]
                print(f"  [KALIBRASI] Canny BSDS500 ODS={score.ods_f:.4f} "
                      f"vs rujukan 0.600, selisih {d:+.4f}")

    if all_scores:
        df = pd.DataFrame([asdict(s) for s in all_scores])
        out = args.results_dir / "ods_summary_v2.csv"
        df.to_csv(out, index=False)
        print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
        print(df[["dataset", "method", "n_images", "ods_f", "ois_f", "ap",
                  "f_at_ods_std"]].to_string(index=False))
        bad = df[df.ois_f < df.ods_f - 1e-9]
        if len(bad):
            print("\n[ALERT] Baris dengan OIS < ODS:")
            print(bad[["dataset", "method", "ods_f", "ois_f"]].to_string(index=False))
        else:
            print("\n[OK] Tidak ada baris dengan OIS < ODS.")
        print(f"[OK] {out}")


if __name__ == "__main__":
    main()
