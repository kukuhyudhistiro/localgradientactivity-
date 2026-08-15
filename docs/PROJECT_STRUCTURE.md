# Project Structure

```
localgradientactivity-/
├── README.md                  Overview and reproduction steps
├── LICENSE                    MIT
├── CITATION.cff               Citation metadata
├── requirements.txt           Python dependencies
├── .gitignore                 Excludes datasets, edge maps, and caches
├── .vscode/
│   ├── tasks.json             One task per reproduction step
│   └── settings.json          Editor and analysis paths
├── scripts/
│   ├── methods_v2.py          Detectors, activity estimate, ablation switches
│   ├── run_experiment.py      Edge map generation for the six methods
│   ├── run_ablation.py        Ablation variants and the parameter grid
│   ├── evaluate_v2.py         Berkeley protocol evaluation
│   ├── evaluate_stratified_v2.py  Activity-stratified evaluation
│   ├── stats_analysis.py      Bootstrap intervals and paired tests
│   ├── diagnose_adaptation.py Saturation diagnostic
│   ├── validate_rho.py        Validation of the activity estimate
│   ├── error_metrics.py       Precision at fixed recall and related metrics
│   ├── benchmark_runtime.py   Isolated timing run
│   └── generate_final_figures.py  All data-derived figures
├── results/
│   ├── tables/                Result tables as CSV
│   ├── stats/                 Confidence intervals and significance tests
│   └── diagnostics/           Diagnostic outputs
├── figures/                   Generated figures
├── manuscript/                Revised manuscript
├── docs/
│   ├── REPRODUCIBILITY.md     Environment, corrections, known limitations
│   ├── PROJECT_STRUCTURE.md   This file
│   ├── revision_log_id.md     Point-by-point revision log (Indonesian)
│   └── manuscript_notes_id.md Manuscript preparation notes (Indonesian)
├── data/                      Datasets, not tracked
└── output/                    Generated edge maps, not tracked
```

## Mapping from manuscript elements to code

| Manuscript element | Produced by |
|---|---|
| Table 4, Figure 3 | `run_ablation.py` grid variants, then `evaluate_v2.py` |
| Tables 5 and 6, Figure 4 | `evaluate_v2.py`, `stats_analysis.py`, `generate_final_figures.py` |
| Table 7 | `stats_analysis.py` |
| Table 8, Figure 9 | `run_ablation.py`, `evaluate_v2.py`, `diagnose_adaptation.py` |
| Tables 9 and 10 | `evaluate_stratified_v2.py` |
| Table 11 | `error_metrics.py` |
| Table 12 | `benchmark_runtime.py` |
| Figures 5, 6, 10 | `generate_final_figures.py` |
| Figures 7 and 8 | `generate_final_figures.py`, `validate_rho.py` |

Figures 1 and 2 are diagrams and are prepared manually.

## Two documents in Indonesian

`docs/revision_log_id.md` and `docs/manuscript_notes_id.md` are working
documents written in Indonesian for the authors. They record the point-by-point
response to the review and the remaining manuscript preparation steps. They are
included for transparency and are not required to run any experiment.
