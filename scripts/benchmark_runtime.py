#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_runtime.py - Isolated timing run for Table 12.

Run this alone on an idle machine, with no other applications open. Load from
other processes inflates the measurement: in the ablation and grid runs the
mean drifted by a factor of two within a single variant on identical images,
while a run on a quiet machine gave a standard deviation of about 3 percent.

Each image is timed several times and the minimum is retained, which limits
the influence of transient system load.

Usage:
  python benchmark_runtime.py --data-root ./data --repeats 3
"""

import os
for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[_k] = "1"

import argparse
from pathlib import Path

import cv2
cv2.setNumThreads(1)
import numpy as np
import pandas as pd

from methods_v2 import preprocess, run_method, AGWiParams, warmup_agwi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--img-dir", default="BSDS500/images/test")
    ap.add_argument("--methods", nargs="+",
                    default=["Sobel", "Canny", "GWi", "GWC", "PC", "AGWi"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("runtime_logs/table12.csv"))
    args = ap.parse_args()

    paths = sorted((args.data_root / args.img_dir).glob("*.jpg"))
    if args.max_images:
        paths = paths[:args.max_images]
    if not paths:
        raise SystemExit(f"No images under {args.data_root / args.img_dir}")
    print(f"[INFO] {len(paths)} images, {args.repeats} repeats per image")
    warmup_agwi(AGWiParams())

    rows = []
    for m in args.methods:
        per_image = []
        for i, p in enumerate(paths, 1):
            gf, gu = preprocess(p)
            best = min(run_method(m, gf, gu, AGWiParams())[1]
                       for _ in range(args.repeats))
            per_image.append(best * 1000.0)
            if i % 50 == 0:
                print(f"  {m}: {i}/{len(paths)}")
        arr = np.array(per_image)
        rows.append({"method": m, "mean_ms": arr.mean(),
                     "std_ms": arr.std(ddof=1), "min_ms": arr.min(),
                     "max_ms": arr.max()})
        print(f"[{m:18s}] mean={arr.mean():8.2f} ms  "
              f"std={arr.std(ddof=1):6.2f}  min={arr.min():8.2f}  "
              f"max={arr.max():8.2f}")

    df = pd.DataFrame(rows)
    ref = df.loc[df.method == "AGWi", "mean_ms"]
    if len(ref):
        df["speed_vs_agwi"] = ref.iloc[0] / df.mean_ms
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n[OK] {args.out}")
    print(df.round(2).to_string(index=False))
    agwi = df[df.method == "AGWi"]
    if len(agwi):
        pct = 100.0 * agwi.std_ms.iloc[0] / agwi.mean_ms.iloc[0]
        print(f"\nA-GWi standard deviation is {pct:.2f} percent of the mean.")
        if pct > 10.0:
            print("[WARNING] Above 10 percent indicates background load. "
                  "Close other applications and repeat the run.")


if __name__ == "__main__":
    main()
