"""
run_experiment.py — Self-contained experiment runner for the A-GWi paper
Author: Kukuh Yudhistiro, 2026

Generates edge maps for all 6 methods on BSDS500 and UDED, in a layout
the bundled evaluate.py understands. No dependency on Paper 1.

Output:
    output/<dataset>/<method>/<image_id>.png   (8-bit magnitude)
    runtime_logs/runtime.csv

Workflow:
    python run_experiment.py --data-root ./data --output-root ./output \
        --datasets BSDS500 UDED --methods AGWi GWi GWC Canny Sobel PC

    python evaluate.py --data-root ./data --output-root ./output \
        --results-dir ./eval_results --datasets BSDS500 UDED \
        --methods AGWi GWi GWC Canny Sobel PC --n-thresholds 99

Single-threaded for fair runtime comparison.
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import argparse
import csv
import time
from pathlib import Path

import cv2
cv2.setNumThreads(1)
import numpy as np
from PIL import Image

from methods_v2 import (
    preprocess, run_method, warmup_agwi, AGWiParams,
    NUMBA_AVAILABLE, HAS_PHASEPACK,
)


DATASET_LAYOUT = {
    "BSDS500": {"img_dir": "BSDS500/images/test", "img_ext": "*.jpg"},
    "BSDS500_val": {"img_dir": "BSDS500/images/val", "img_ext": "*.jpg"},
    "UDED": {"img_dir": "UDED/imgs", "img_ext": "*.jpg"},
    "BIPED": {"img_dir": "BIPED/edges/imgs/test", "img_ext": "*.jpg"},
}

ALL_METHODS = ["AGWi", "GWi", "GWC", "Canny", "Sobel", "PC"]


def save_magnitude_png(magnitude, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mn, mx = magnitude.min(), magnitude.max()
    norm = (magnitude - mn) / (mx - mn) if mx > mn else np.zeros_like(magnitude)
    Image.fromarray((norm * 255).astype(np.uint8)).save(str(out_path))


def find_images(data_root, dataset, max_images=None):
    layout = DATASET_LAYOUT[dataset]
    img_dir = data_root / layout["img_dir"]
    if not img_dir.exists():
        return []
    paths = sorted(img_dir.glob(layout["img_ext"]))
    if not paths:
        for ext in ["*.png", "*.jpeg", "*.JPG"]:
            paths.extend(sorted(img_dir.glob(ext)))
        seen, uniq = set(), []
        for p in paths:
            if p.stem not in seen:
                seen.add(p.stem)
                uniq.append(p)
        paths = sorted(uniq)
    if max_images:
        paths = paths[:max_images]
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["BSDS500", "UDED"])
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--runtime-csv", type=Path,
                        default=Path("runtime_logs/runtime.csv"))
    # A-GWi ablation knobs
    parser.add_argument("--f-min", type=float, default=0.05)
    parser.add_argument("--f-max", type=float, default=0.45)
    parser.add_argument("--k-steepness", type=float, default=25.0)
    parser.add_argument("--density", choices=["sobel", "variance"],
                        default="sobel")
    parser.add_argument("--agwi-name", default="AGWi",
                        help="Output folder for A-GWi (e.g. AGWi_lmax10 for ablation)")
    args = parser.parse_args()

    agwi_params = AGWiParams(
        f_min=args.f_min,
        f_max=args.f_max,
        k_steepness=args.k_steepness,
    )

    print("=" * 72)
    print("A-GWi Paper: Edge Map Generation")
    print("=" * 72)
    print(f"  Numba: {NUMBA_AVAILABLE} | phasepack: {HAS_PHASEPACK}")
    print(f"  Methods: {args.methods}")
    print(f"  A-GWi: f0=[{args.f_min},{args.f_max}], "
          f"k_s={args.k_steepness}, density={args.density}")
    if not NUMBA_AVAILABLE:
        print("  [WARNING] Numba missing -> A-GWi very slow. pip install numba")
    if "PC" in args.methods and not HAS_PHASEPACK:
        print("  [WARNING] phasepack missing -> PC will be skipped")
        args.methods = [m for m in args.methods if m != "PC"]

    print("\nWarming up Numba JIT...")
    warmup_agwi(agwi_params)

    args.runtime_csv.parent.mkdir(parents=True, exist_ok=True)
    rt_rows = []

    for dataset in args.datasets:
        if dataset not in DATASET_LAYOUT:
            print(f"[WARN] Unknown dataset {dataset}")
            continue
        paths = find_images(args.data_root, dataset, args.max_images)
        if not paths:
            print(f"[WARN] No images for {dataset}")
            continue
        print(f"\n[{dataset}] {len(paths)} images")

        # Preprocess once per image, reuse for all methods
        for method in args.methods:
            folder = args.agwi_name if method == "AGWi" else method
            out_dir = args.output_root / dataset / folder
            times = []
            ds_t0 = time.perf_counter()

            for i, img_path in enumerate(paths, 1):
                gray_f, gray_u8 = preprocess(img_path)
                try:
                    mag, rt = run_method(
                        method, gray_f, gray_u8,
                        agwi_params=agwi_params,
                        density_method=args.density
                    )
                except Exception as e:
                    print(f"  [ERR] {method}/{img_path.stem}: {e}")
                    continue
                save_magnitude_png(mag, out_dir / f"{img_path.stem}.png")
                times.append(rt)
                rt_rows.append({
                    "dataset": dataset, "method": folder,
                    "image_id": img_path.stem,
                    "height": gray_f.shape[0], "width": gray_f.shape[1],
                    "runtime_s": f"{rt:.6f}", "status": "OK",
                })
                if i % 25 == 0 or i == len(paths):
                    el = time.perf_counter() - ds_t0
                    rate = i / el if el > 0 else 0
                    eta = (len(paths) - i) / rate if rate > 0 else 0
                    print(f"  {method}: {i}/{len(paths)} "
                          f"(mean {np.mean(times)*1000:.0f} ms, "
                          f"ETA {eta/60:.1f} min)")

            if times:
                print(f"  [{method}] mean={np.mean(times)*1000:.1f} ms, "
                      f"total={np.sum(times):.1f} s")

    with open(args.runtime_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["dataset", "method", "image_id", "height", "width",
                  "runtime_s", "status"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rt_rows)
    print(f"\n[OK] Runtime log: {args.runtime_csv}")
    print("\nNext: run evaluate.py to compute ODS/OIS/AP")


if __name__ == "__main__":
    main()
