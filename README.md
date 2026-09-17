# Holstein and SideView Cattle Re-Identification

Auditable code and numerical results for the Chinese working manuscript on Holstein full-candidate recognition and exploratory SideViewCows2026 transfer. No models were retrained or rescored to prepare this release.

## Main Results

| Holstein method | Rank-1 | Rank-5 |
| --- | ---: | ---: |
| B-selected | 83.67% | 98.78% |
| GAP | 74.10% | 96.74% |
| SupCon-in | 65.52% | 90.72% |

These are equal-model means across 15 frozen models per method, with K=1,3,5 mixed. Each model uses its fold's full 64/65-candidate set.

| SideView gray-background method | K=1 | K=3 | K=5 |
| --- | ---: | ---: | ---: |
| B | 50.10% | 58.35% | 61.82% |
| GAP | 39.31% | 47.16% | 51.12% |
| SupCon-in | 28.42% | 32.86% | 35.21% |

SideView is a **reused public dataset**, not newly collected data. The external preprocessing comparisons are exploratory and image-level: capture-event independence is not verified. They are not an untouched confirmatory test.

## Reproduce the Archive

```bash
python scripts/reproduce_results.py
python -m unittest discover -s tests
pip install -r requirements.txt
python scripts/identity_sensitivity.py
python scripts/plot_paper_figures.py
```

The first command runs offline with the Python standard library. No GPU, network or original images are required for count-level verification. The optional figure command requires a Chinese font for the manuscript labels.

## Contents

- `results/sideview/`: four byte-preserved historical result JSON files, 2700 records in total.
- `tables/`: Holstein model-level/stratified results, SideView verified summaries and eligibility/split metadata.
- `protocols/`: scientific split subset, training configuration, source hashes and the post-hoc sensitivity plan.
- `scripts/`: executable result checks, sensitivity analysis and Python vector-figure reproduction.
- `figures/`: manuscript SVG/PDF figures, without original animal photographs.
- `reference_training/`: hash-verified historical source excerpts, not a standalone training launcher.
- `MANIFEST.json`: SHA256 checksums for release files.

Read [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for estimands, metadata caveats and unsupported reproduction claims, and [DATA_SOURCES.md](DATA_SOURCES.md) for original-data attribution and release scope. Weights and original images are not included. This release does not provide end-to-end retraining or new-image inference reproduction.
