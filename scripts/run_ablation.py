"""
run_ablation.py - Matriks ablasi untuk revisi JESA 44740
Author: Kukuh Yudhistiro, 2026

Menjawab R1-7 dan R2-7 (ablasi tidak memadai) serta R1-15 dan R2-15
(klaim kausal melampaui bukti).

Setiap varian menghasilkan folder peta tepi tersendiri di
output/<dataset>/<nama_varian>/ sehingga evaluate_v2.py dapat menilainya
tanpa perubahan apa pun.

Varian:
  A0_GWi_static     GWi statis, lambda = 4              (baseline)
  A1_AGWi_full      A-GWi penuh, setelan naskah         (referensi)
  A2_freq_only      f0 adaptif, sigma dikunci 2.24      pisahkan efek frekuensi
  A3_scale_only     sigma adaptif, f0 dikunci 0.25      pisahkan efek skala
  A4_noHE           A-GWi tanpa histogram equalization  efek praproses
  A5_ks_<v>         k_s alternatif                      sensitivitas kecuraman
  A6_ksize_<v>      ukuran kernel 5 dan 9               efek pemotongan jendela
  A7_rho_shuffle    rho_L dipermutasi spasial           KONTROL NEGATIF
  A8_rho_variance   rho_L dari varians lokal            ketergantungan Sobel
  A9_rho_edgemap    rho_L langsung sebagai peta tepi    KONTROL TERPENTING
  A10_octave        relasi sigma dengan faktor oktaf    perbaiki Eq. (8)
  A11_l2norm        kernel dinormalkan L2               bias amplitudo
  A12_static_fmin   Gabor statis pada f0 = f_min        UJI PALING KRITIS
  A13_cdf           rho_L ditransformasi CDF            aktifkan adaptasi

A9 dan A12 menentukan apakah klaim inti naskah bertahan:
  - Jika A9 (rho_L saja) mendekati A1, keunggulan berasal dari prior Sobel.
  - Jika A12 (statis f_min) mendekati A1, adaptasi tidak menyumbang apa pun.

Pemakaian:
  python run_ablation.py --data-root ./data --output-root ./output \
      --datasets BSDS500 UDED --variants all
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
from dataclasses import replace
from pathlib import Path

import cv2
cv2.setNumThreads(1)
import numpy as np
from PIL import Image

from methods_v2 import (
    preprocess, AGWiParams, run_agwi, run_gwi, run_gwc,
    run_rho_as_edgemap, warmup_agwi, NUMBA_AVAILABLE,
)

DATASET_LAYOUT = {
    "BSDS500": {"img_dir": "BSDS500/images/test", "img_ext": "*.jpg"},
    "BSDS500_train": {"img_dir": "BSDS500/images/train", "img_ext": "*.jpg"},
    "UDED": {"img_dir": "UDED/imgs", "img_ext": "*.jpg"},
    "BSDS500_val": {"img_dir": "BSDS500/images/val", "img_ext": "*.jpg"},
}

BASE = AGWiParams()


def build_variants():
    """name -> (kind, params, kwargs, use_he)"""
    v = {}
    v["A0_GWi_static"] = ("gwi", BASE, {}, True)
    v["A1_AGWi_full"] = ("agwi", BASE, {}, True)
    v["A2_freq_only"] = ("agwi", replace(BASE, adapt_scale=False,
                                         sigma_static=2.24), {}, True)
    v["A3_scale_only"] = ("agwi", replace(BASE, adapt_freq=False,
                                          f_static=0.25), {}, True)
    v["A4_noHE"] = ("agwi", BASE, {}, False)
    for ks in [5.0, 10.0, 40.0]:
        v[f"A5_ks_{int(ks)}"] = ("agwi", replace(BASE, k_steepness=ks), {}, True)
    for k in [5, 9]:
        v[f"A6_ksize_{k}"] = ("agwi", replace(BASE, kernel_size=k), {}, True)
    v["A7_rho_shuffle"] = ("agwi", BASE, {"rho_transform_mode": "shuffle"}, True)
    v["A8_rho_variance"] = ("agwi", BASE, {"density_method": "variance"}, True)
    v["A9_rho_edgemap"] = ("rho", BASE, {}, True)
    v["A10_octave"] = ("agwi", replace(BASE, sigma_relation="octave"), {}, True)
    v["A11_l2norm"] = ("agwi", replace(BASE, l2_normalize=True), {}, True)
    v["A12_static_fmin"] = ("agwi", replace(BASE, adapt_freq=False,
                                            adapt_scale=False,
                                            f_static=BASE.f_min,
                                            sigma_static=0.1873906 / BASE.f_min),
                            {}, True)
    v["A13_cdf"] = ("agwi", BASE, {"rho_transform_mode": "cdf"}, True)
    v["A14_center_med"] = ("agwi", replace(BASE, rho_center=0.0906,
                                           k_steepness=8.0), {}, True)
    v["A4b_GWi_noHE"] = ("gwi", BASE, {}, False)
    v["A4c_GWC_noHE"] = ("gwc", BASE, {}, False)
    # Parameter verification grid, run on the BSDS500 validation split only
    for fmin in [0.03, 0.05, 0.08, 0.12]:
        for ks in [5.0, 25.0]:
            tag = f"G_fmin{fmin:.2f}_ks{int(ks)}".replace(".", "")
            v[tag] = ("agwi", replace(BASE, f_min=fmin, k_steepness=ks),
                      {}, True)
    return v


CRITICAL = ["A0_GWi_static", "A1_AGWi_full", "A7_rho_shuffle",
            "A9_rho_edgemap", "A12_static_fmin", "A13_cdf"]


def save_png(mag, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    mn, mx = mag.min(), mag.max()
    norm = (mag - mn) / (mx - mn) if mx > mn else np.zeros_like(mag)
    Image.fromarray((norm * 255).astype(np.uint8)).save(str(path))


def find_images(data_root, dataset, max_images=None):
    lay = DATASET_LAYOUT[dataset]
    d = data_root / lay["img_dir"]
    if not d.exists():
        return []
    paths = sorted(d.glob(lay["img_ext"]))
    if not paths:
        for ext in ["*.png", "*.jpeg", "*.JPG"]:
            paths.extend(sorted(d.glob(ext)))
        seen, uniq = set(), []
        for p in sorted(paths):
            if p.stem not in seen:
                seen.add(p.stem)
                uniq.append(p)
        paths = uniq
    return paths[:max_images] if max_images else paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["BSDS500", "UDED"])
    ap.add_argument("--variants", nargs="+", default=["critical"],
                    help="'all', 'critical', atau daftar nama varian")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--runtime-csv", type=Path,
                    default=Path("runtime_logs/runtime_ablation.csv"))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    variants = build_variants()
    if args.list:
        for k, (kind, p, kw, he) in variants.items():
            print(f"{k:22s} kind={kind:5s} he={he} kwargs={kw}")
        return

    if args.variants == ["all"]:
        selected = list(variants)
    elif args.variants == ["critical"]:
        selected = CRITICAL
    else:
        selected = args.variants
    missing = [s for s in selected if s not in variants]
    if missing:
        raise SystemExit(f"Varian tidak dikenal: {missing}")

    print("=" * 72)
    print("A-GWi ablation runner")
    print("=" * 72)
    print(f"  Numba: {NUMBA_AVAILABLE}")
    print(f"  Varian: {selected}")
    if not NUMBA_AVAILABLE:
        print("  [WARNING] Numba tidak ada, A-GWi akan sangat lambat.")
    warmup_agwi(BASE)

    args.runtime_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for dataset in args.datasets:
        paths = find_images(args.data_root, dataset, args.max_images)
        if not paths:
            print(f"[WARN] Tidak ada citra untuk {dataset}")
            continue
        print(f"\n[{dataset}] {len(paths)} citra")

        for name in selected:
            kind, params, kwargs, use_he = variants[name]
            out_dir = args.output_root / dataset / name
            times = []
            t_start = time.perf_counter()
            for i, img_path in enumerate(paths, 1):
                gray_f, _ = preprocess(img_path, use_he=use_he)
                try:
                    if kind == "agwi":
                        mag, rt = run_agwi(gray_f, params, **kwargs)
                    elif kind == "gwi":
                        mag, rt = run_gwi(gray_f)
                    elif kind == "gwc":
                        mag, rt = run_gwc(gray_f)
                    elif kind == "rho":
                        mag, rt = run_rho_as_edgemap(gray_f, params)
                    else:
                        raise ValueError(kind)
                except Exception as e:
                    print(f"  [ERR] {name}/{img_path.stem}: {e}")
                    continue
                save_png(mag, out_dir / f"{img_path.stem}.png")
                times.append(rt)
                rows.append({"dataset": dataset, "variant": name,
                             "image_id": img_path.stem,
                             "runtime_s": f"{rt:.6f}"})
                if i % 25 == 0 or i == len(paths):
                    el = time.perf_counter() - t_start
                    eta = (len(paths) - i) / max(i / el, 1e-9)
                    print(f"  {name}: {i}/{len(paths)} "
                          f"(mean {np.mean(times) * 1000:.0f} ms, "
                          f"ETA {eta / 60:.1f} min)")
            if times:
                print(f"  [{name}] mean={np.mean(times) * 1000:.1f} ms")

    with open(args.runtime_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "variant",
                                          "image_id", "runtime_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {args.runtime_csv}")
    print("\nLangkah berikutnya:")
    print("  python evaluate_v2.py --data-root ./data --output-root ./output \\")
    print("      --results-dir ./eval_results_ablation --datasets BSDS500 UDED")


if __name__ == "__main__":
    main()
