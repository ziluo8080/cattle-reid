# Reproducibility and Evidence Boundaries

## Supported Reproduction

1. Run `python scripts/reproduce_results.py` with Python 3.11 or later. This uses only the standard library, verifies file checksums, checks all 2700 SideView scoring records and their 145800 identity-level counts, and recreates 36 micro/macro result pairs.
2. Run `python -m unittest discover -s tests` to test valid inputs and rejection of corrupted counts and duplicate task keys.
3. Install `requirements.txt` and run `python scripts/identity_sensitivity.py` for the explicitly post-hoc, conditional identity-resampling analysis.
4. Run `python scripts/plot_paper_figures.py` to rebuild the six manuscript diagrams/plots as SVG/PDF. A Chinese font such as Microsoft YaHei is needed for Chinese labels. The public plotting adapter removes local absolute dependency paths and the private skill alignment checker, without changing the plot definitions. PNG files generated under `derived/` are only inspection previews.

## Data and Estimands

Holstein uses 324 eligible identities, five folds and three seeds. The headline results are equal-model means over 15 models per method and mix K=1,3,5. `holstein_per_model.csv` retains the paired fold/seed values. Candidate/K stratification is a different, scoring-entry-weighted summary; 88.00% is the 65-candidate/K=5 B stratum, not the headline result.

SideView uses the public snapshots subset: 607 input images from 63 identities, with 54 identities and 577 images retained by the six-image minimum. Five nested support draws reserve five images per identity, leaving 307 queries per draw. Four preprocessing variants each contain 675 records (45 models x 5 draws x 3 K). Repeated model/draw observations are not independent animals.

The numeric identifiers in the protocol are image-array indices, not globally meaningful filenames. The corresponding `protocols/sideview_manifest.jsonl` is now included. `scripts/prepare_sideview.py` can rebuild the four input arrays from the official snapshots ZIP; `scripts/score_sideview.py` can run frozen inference after all 45 original checkpoints are supplied. See the bilingual README for commands. These portable adapters do not retroactively establish missing historical execution receipts, and do not provide complete Holstein retraining.

## Provenance Caveat

The historical neutral128 result JSON incorrectly retained the literal preprocessing name `sideview2026-snapshots-fullbody-letterbox-v1`. It is preserved byte-for-byte. The variant is resolved from the recorded input upload and environment-variable launch, and matches the later historical comparison. `result_provenance.json` records this distinction. This is not a claim that a runtime input-byte receipt exists for the older baseline/neutral128 jobs.

Only model/draw/K and per-identity correct/total counts are present for baseline and neutral128. Individual query predictions, score matrices, original images and trained weights are not distributed here. Full retraining and independent inference reproduction therefore require additional artifacts.

## Statistical Scope

The identity resampling analysis averages correct counts across the 15 fixed models and 5 fixed support draws, then samples 54 identity clusters with replacement using the same indices for all comparisons. It does not rebuild the candidate gallery, resample training, or repair unverified event independence. Percentile bounds are conditional, exploratory sensitivity intervals, not confirmation of generalization to new farms or identities. Nine pointwise comparisons are reported with no familywise significance claim or P values. The plan was written after the original performance results were known.

All preprocessing choices were compared using external results. SideView is an external public data source, but this is not an untouched confirmatory external test. Official identity labels are reused; this archive does not claim new manual labeling by the authors.
