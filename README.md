# Local Gradient Activity

Per-pixel Gabor kernel modulation for edge-based boundary detection.

This repository contains the code, result tables, and revised manuscript for
the Adaptive Imaginary Gabor Wavelet (A-GWi), a handcrafted method in which the
centre frequency and Gaussian scale of an imaginary-only Gabor kernel are set at
every pixel in closed form from a local gradient activity estimate, without
training and without a graphics processing unit.

Everything needed to reproduce every number and every figure in the manuscript
is here. The datasets and the generated edge maps are not tracked, because of
their size; the instructions below explain how to place them.

---

## What the experiments show

| Method | ODS (BSDS500) | ODS (UDED) | Runtime |
|---|---|---|---|
| Sobel | 0.464 | 0.637 | 1.66 ms |
| Canny (gradient magnitude) | 0.533 | 0.670 | 2.28 ms |
| Phase Congruency | 0.524 | 0.568 | 402.16 ms |
| Complex Gabor wavelet (GWC) | 0.397 | 0.507 | 33.70 ms |
| Imaginary-only Gabor wavelet (GWi) | 0.417 | 0.580 | 15.54 ms |
| **A-GWi (this work)** | **0.534** | **0.668** | 604.41 ms |

ODS is the optimal dataset scale F-measure under the Berkeley protocol with 99
thresholds, a matching tolerance of 0.0075 times the image diagonal, and
dataset-level count accumulation.

Three findings are worth stating plainly, because they qualify what the method
contributes.

The margin over both static Gabor baselines is statistically significant on
both datasets, with medium to large effect sizes. This is the controlled
comparison, since the three methods share the kernel structure, the orientation
count, the window size, and the preprocessing.

The margin over the gradient-magnitude Canny baseline is **not** significant on
either dataset, at p = 0.82 on both, while A-GWi is 265 times slower.

The ablation shows that a static kernel at the lower frequency bound reaches
the same accuracy as the full per-pixel formulation, and the paired difference
is not significant at p = 0.35. The improvement over the static baseline at a
wavelength of 4 pixels therefore comes from the change in effective centre
frequency rather than from per-pixel modulation. At the reported setting, 92.8
percent of pixels receive a frequency within 1 percent of the lower bound, and
the output correlates at 0.857 with a static bank at that frequency. When the
modulation is made active across the full range, accuracy falls to 0.361.

The per-pixel mechanism reaches an effective operating point without a manually
selected centre frequency. It does not improve on the best static
configuration, and the repository is arranged so that a reader can verify this
directly.

---

## Repository layout

```
scripts/      All experiment code
results/      Result tables, statistics, and diagnostics as CSV
figures/      Generated figures (created by the figure script)
data/         Datasets (not tracked, see below)
output/       Generated edge maps (not tracked)
```

### Scripts

| File | Purpose |
|---|---|
| `methods_v2.py` | All six detectors, the activity estimate, and the ablation switches |
| `run_experiment.py` | Generates edge maps for the six methods |
| `run_ablation.py` | Generates edge maps for the ablation variants and the parameter grid |
| `evaluate_v2.py` | Berkeley protocol evaluation, ODS, OIS, and AP |
| `evaluate_stratified_v2.py` | Activity-stratified evaluation with fixed tertiles |
| `stats_analysis.py` | Bootstrap confidence intervals and paired Wilcoxon tests |
| `diagnose_adaptation.py` | Measures whether the adaptive mechanism is active |
| `validate_rho.py` | Correlates the activity estimate against ground-truth boundary density |
| `error_metrics.py` | Precision at fixed recall, localisation error, edge width |
| `benchmark_runtime.py` | Isolated timing run |
| `generate_final_figures.py` | All data-derived figures |

---

## Setup

```bash
git clone https://github.com/kukuhyudhistiro/localgradientactivity-.git
cd localgradientactivity-
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux and macOS
source .venv/bin/activate
pip install -r requirements.txt
```

Numba is required. Without it the per-pixel convolution falls back to
interpreted Python and becomes impractically slow.

### Datasets

Place the datasets under `data/` in this layout:

```
data/
  BSDS500/
    images/train/      200 images, used only for the stratum tertiles
    images/val/        100 images, used only for parameter verification
    images/test/       200 images
    groundTruth/train/ .mat files
    groundTruth/val/   .mat files
    groundTruth/test/  .mat files
  UDED/
    imgs/              30 images
    gt/                30 PNG boundary maps
```

No test image is used at any stage of parameter selection.

---

## Reproducing the results

The steps are ordered so that each one consumes the output of the previous one.
Every step is also available as a VS Code task; press `Ctrl+Shift+P`, choose
`Tasks: Run Task`, and pick the numbered entry.

### Step 0. Check whether the adaptive mechanism is active

```bash
python scripts/diagnose_adaptation.py --data-root ./data --dataset BSDS500 \
    --max-images 50 --out-dir ./results/diagnostics
```

Read the line reporting the fraction of pixels at the lower frequency bound. At
the setting reported in the manuscript this is 92.78 percent, which is the
reason the ablation in Step 3 matters.

### Step 1. Generate edge maps

```bash
python scripts/run_experiment.py --data-root ./data --output-root ./output \
    --datasets BSDS500 UDED --methods AGWi GWi GWC Canny Sobel PC
```

All six methods receive identical preprocessing and identical border handling.
Canny is evaluated as the Gaussian-smoothed gradient magnitude before
non-maximum suppression, so that every method produces a continuous map that
the threshold sweep traverses in the same way.

### Step 2. Evaluate

```bash
python scripts/evaluate_v2.py --data-root ./data --output-root ./output \
    --results-dir ./results/eval --datasets BSDS500 UDED --n-thresholds 99
```

The script prints a calibration line for Canny on BSDS500 and raises an alert
if any row has an optimal image scale value below its optimal dataset scale
value, which would indicate an aggregation error.

### Step 3. Ablation

```bash
python scripts/run_ablation.py --data-root ./data --output-root ./output \
    --datasets BSDS500 --variants critical

python scripts/evaluate_v2.py --data-root ./data --output-root ./output \
    --results-dir ./results/eval_ablation --datasets BSDS500 --n-thresholds 99
```

`python scripts/run_ablation.py --list` prints every variant. The four that
determine how the results should be read are:

| Variant | Question it answers |
|---|---|
| `A12_static_fmin` | Does the per-pixel mechanism contribute anything beyond its operating point? |
| `A9_rho_edgemap` | How much of the accuracy is already present in the driving signal? |
| `A7_rho_shuffle` | Does the spatial structure of the driving signal matter? |
| `A13_cdf` | What happens when the modulation is made active across the full range? |

### Step 4. Statistics

```bash
python scripts/stats_analysis.py --counts-dir ./results/eval/counts \
    --dataset BSDS500 --reference AGWi --n-boot 2000 --out-dir ./results/stats
python scripts/stats_analysis.py --counts-dir ./results/eval/counts \
    --dataset UDED --reference AGWi --n-boot 2000 --out-dir ./results/stats
```

Reports bootstrap confidence intervals over images together with paired
Wilcoxon tests and Cliff delta effect sizes. Differences whose intervals
overlap are not used to establish a ranking.

### Step 5. Validate the activity estimate

```bash
python scripts/validate_rho.py --data-root ./data --dataset BSDS500 \
    --window 15 --out-dir ./results/diagnostics
```

Reports the Spearman correlation of the estimate against the local ground-truth
boundary count and against local intensity contrast. The second correlation is
the stronger of the two, which is why the quantity is called gradient activity
rather than density.

### Step 6. Stratified evaluation

```bash
python scripts/evaluate_stratified_v2.py --data-root ./data --compute-tertiles \
    --tertile-source BSDS500_train --tertile-file tertiles.json

python scripts/evaluate_stratified_v2.py --data-root ./data \
    --output-root ./output --results-dir ./results/eval \
    --datasets BSDS500 UDED --methods AGWi GWi GWC Canny \
    --tertile-file tertiles.json
```

Stratum boundaries come from the training split and are applied as absolute
thresholds to both evaluation sets. Predicted and ground-truth pixels outside a
stratum mask are both discarded before matching, so precision and recall are
computed on the same restricted population.

### Step 7. Error characteristics

```bash
python scripts/error_metrics.py --data-root ./data --output-root ./output \
    --results-dir ./results/eval --dataset BSDS500 \
    --methods AGWi GWi GWC Canny --target-recall 0.5
```

### Step 8. Runtime

```bash
python scripts/benchmark_runtime.py --data-root ./data --repeats 3 \
    --out ./results/tables/runtime_table12.csv
```

Run this alone on an idle machine. The script warns if the standard deviation
exceeds 10 percent of the mean, which indicates background load.

### Step 9. Figures

```bash
python scripts/generate_final_figures.py --data-root ./data \
    --output-root ./output --eval-results ./results/eval \
    --eval-grid ./results/eval_grid --figures-dir ./figures --figures all
```

Figures 1 and 2 in the manuscript are diagrams and are not produced by this
script.

---

## Result files

| File | Contents |
|---|---|
| `results/tables/main_results_ods_summary.csv` | ODS, OIS, and AP for the six methods on both datasets |
| `results/tables/ablation_ods_summary_bsds500.csv` | Ablation variants on the test split |
| `results/tables/fmax_sweep_bsds500.csv` | Sensitivity to the upper frequency bound |
| `results/tables/stratified_v2.csv` | Activity-stratified F-measure |
| `results/tables/runtime_table12.csv` | Per-image runtime |
| `results/stats/bootstrap_ci_BSDS500.csv` | Bootstrap confidence intervals |
| `results/stats/paired_tests_BSDS500.csv` | Paired Wilcoxon tests and effect sizes |
| `results/diagnostics/runtime_grid_raw.csv` | Raw per-image timings from the parameter grid, retained to document the load drift that made those timings unusable |

---

## Notes on the evaluation protocol

Both the optimal dataset scale and optimal image scale metrics use the same
count accumulation across the test set. Mixing pooled counts for one metric
with a mean of per-image values for the other does not guarantee an ordering
between them, and an earlier version of this work reported an inconsistent
ordering for that reason.

Matching is performed against each annotator separately. A predicted pixel
counts as correct if it matches at least one annotator, while recall counts are
summed across all annotators. Assignment within each annotator is one to one,
ordered by increasing distance.

Average precision is the area under the dataset-level precision-recall curve
computed from the same accumulated counts, with the precision made monotone,
rather than a mean of per-image values.

---

## Related repositories

| Repository | Content |
|---|---|
| `kukuhyudhistiro/gwi-odps` | Imaginary-only Gabor wavelet with orientation-aware double-peak suppression (submit to Digital Signal Processing - Under Review 4 June 2026|
| `kukuhyudhistiro/agwi` | Earlier version of the present work, retained so that the submitted results remain reproducible |

The Canny definition used here matches the one in `gwi-odps`, and the two
implementations agree to within 0.001 ODS on BSDS500.

---

## Citation

See `CITATION.cff`. The manuscript is under review at Journal Europeen des
Systemes Automatises.

## License

MIT. See `LICENSE`.
