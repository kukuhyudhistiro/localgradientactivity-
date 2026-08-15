"""
stats_analysis.py - Ketidakpastian dan uji berpasangan
Author: Kukuh Yudhistiro, 2026

Menjawab R1-13 dan R2-13: hasil dilaporkan sebagai nilai tunggal tanpa
sebaran per citra dan tanpa uji statistik, sementara selisih OIS antara
A-GWi dan Canny pada UDED hanya 0.001 pada 30 citra.

Masukan: file .npz yang ditulis evaluate_v2.py di <results-dir>/counts/.

Keluaran:
  bootstrap_ci.csv   ODS, OIS, AP dengan interval kepercayaan 95 persen
  paired_tests.csv   Wilcoxon signed-rank + Cliff's delta pada F per citra

Pemakaian:
  python stats_analysis.py --counts-dir ./eval_results_v2/counts \
      --dataset BSDS500 --reference AGWi --n-boot 2000
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

EPS = 1e-10


def prf(counts):
    c = np.asarray(counts, dtype=np.float64)
    r = c[..., 0] / np.maximum(c[..., 1], EPS)
    p = c[..., 2] / np.maximum(c[..., 3], EPS)
    f = 2 * p * r / np.maximum(p + r, EPS)
    return p, r, f


def interpolated_ap(p, r):
    o = np.argsort(r)
    r_, p_ = r[o], p[o]
    p_env = np.maximum.accumulate(p_[::-1])[::-1]
    r_ext = np.concatenate(([0.0], r_))
    return float(np.sum(np.diff(r_ext) * p_env))


def metrics_from_counts(arr):
    """arr: (n_img, n_thr, 4) -> (ods, ois, ap)"""
    n_img = arr.shape[0]
    pooled = arr.sum(axis=0)
    p_t, r_t, f_t = prf(pooled)
    ods = float(f_t.max())
    p_i, r_i, f_i = prf(arr)
    best = np.argmax(f_i, axis=1)
    ois_counts = arr[np.arange(n_img), best].sum(axis=0)
    _, _, ois = prf(ois_counts)
    return ods, float(ois), interpolated_ap(p_t, r_t)


def bootstrap(arr, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    out = np.empty((n_boot, 3))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        out[b] = metrics_from_counts(arr[idx])
    lo = np.percentile(out, 2.5, axis=0)
    hi = np.percentile(out, 97.5, axis=0)
    return lo, hi


def cliffs_delta(a, b):
    """Effect size non-parametrik. |d|: <0.147 negligible, <0.33 small,
    <0.474 medium, else large."""
    a = np.asarray(a)
    b = np.asarray(b)
    gt = sum((a[:, None] > b[None, :]).sum(axis=1))
    lt = sum((a[:, None] < b[None, :]).sum(axis=1))
    return float((gt - lt) / (len(a) * len(b)))


def f_at_ods_per_image(arr):
    """F per citra pada ambang ODS tingkat dataset."""
    pooled = arr.sum(axis=0)
    _, _, f_t = prf(pooled)
    ods_i = int(np.argmax(f_t))
    _, _, f_i = prf(arr)
    return f_i[:, ods_i]


def best_f_per_image(arr):
    _, _, f_i = prf(arr)
    return f_i.max(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts-dir", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--reference", default="AGWi",
                    help="Metode pembanding untuk uji berpasangan")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    files = sorted(args.counts_dir.glob(f"{args.dataset}_*.npz"))
    if not files:
        raise SystemExit(f"Tidak ada .npz untuk {args.dataset} di {args.counts_dir}")

    data, stems_ref = {}, None
    for f in files:
        method = f.stem[len(args.dataset) + 1:]
        if args.methods and method not in args.methods:
            continue
        z = np.load(f, allow_pickle=True)
        data[method] = z["counts"]
        stems = list(z["stems"])
        if stems_ref is None:
            stems_ref = stems
        elif stems != stems_ref:
            print(f"[WARN] {method}: urutan citra berbeda, "
                  f"uji berpasangan bisa salah pasangan.")

    # --- bootstrap ---
    rows = []
    for m, arr in data.items():
        ods, ois, ap_ = metrics_from_counts(arr)
        lo, hi = bootstrap(arr, args.n_boot)
        f_ods = f_at_ods_per_image(arr)
        rows.append({
            "dataset": args.dataset, "method": m, "n_images": arr.shape[0],
            "ODS": ods, "ODS_lo": lo[0], "ODS_hi": hi[0],
            "OIS": ois, "OIS_lo": lo[1], "OIS_hi": hi[1],
            "AP": ap_, "AP_lo": lo[2], "AP_hi": hi[2],
            "F_at_ODS_mean": float(f_ods.mean()),
            "F_at_ODS_std": float(f_ods.std(ddof=1)),
        })
    boot = pd.DataFrame(rows).sort_values("ODS", ascending=False)
    boot.to_csv(args.out_dir / f"bootstrap_ci_{args.dataset}.csv", index=False)

    print("=" * 92)
    print(f"BOOTSTRAP 95% CI  ({args.dataset}, {args.n_boot} resampling citra)")
    print("=" * 92)
    for _, r in boot.iterrows():
        print(f"  {r.method:18s} ODS={r.ODS:.4f} [{r.ODS_lo:.4f}, {r.ODS_hi:.4f}]  "
              f"OIS={r.OIS:.4f} [{r.OIS_lo:.4f}, {r.OIS_hi:.4f}]  "
              f"AP={r.AP:.4f} [{r.AP_lo:.4f}, {r.AP_hi:.4f}]")

    # --- uji berpasangan terhadap referensi ---
    if args.reference not in data:
        print(f"\n[WARN] Referensi {args.reference} tidak ada, uji dilewati.")
        return
    ref_f = f_at_ods_per_image(data[args.reference])
    ref_best = best_f_per_image(data[args.reference])

    trows = []
    for m, arr in data.items():
        if m == args.reference:
            continue
        oth_f = f_at_ods_per_image(arr)
        oth_best = best_f_per_image(arr)
        if len(oth_f) != len(ref_f):
            print(f"[WARN] {m}: jumlah citra beda, dilewati")
            continue
        try:
            _, p_ods = wilcoxon(ref_f, oth_f)
        except ValueError:
            p_ods = float("nan")
        try:
            _, p_ois = wilcoxon(ref_best, oth_best)
        except ValueError:
            p_ois = float("nan")
        trows.append({
            "dataset": args.dataset,
            "comparison": f"{args.reference} vs {m}",
            "median_diff_F_at_ODS": float(np.median(ref_f - oth_f)),
            "p_wilcoxon_F_at_ODS": p_ods,
            "cliffs_delta_F_at_ODS": cliffs_delta(ref_f, oth_f),
            "median_diff_best_F": float(np.median(ref_best - oth_best)),
            "p_wilcoxon_best_F": p_ois,
            "n_images": len(ref_f),
        })
    tests = pd.DataFrame(trows)
    tests.to_csv(args.out_dir / f"paired_tests_{args.dataset}.csv", index=False)

    print("\n" + "=" * 92)
    print(f"UJI BERPASANGAN (Wilcoxon signed-rank, referensi = {args.reference})")
    print("=" * 92)
    for _, r in tests.iterrows():
        sig = "signifikan" if r.p_wilcoxon_F_at_ODS < 0.05 else "TIDAK signifikan"
        print(f"  {r.comparison:26s} median dF={r.median_diff_F_at_ODS:+.4f}  "
              f"p={r.p_wilcoxon_F_at_ODS:.2e}  delta={r.cliffs_delta_F_at_ODS:+.3f}  {sig}")
    print("\nCatatan untuk naskah: laporkan p dan effect size, bukan hanya selisih rata-rata.")
    print("Selisih yang intervalnya bertumpang tindih tidak boleh dipakai untuk menyusun peringkat.")


if __name__ == "__main__":
    main()
