# Historical preprocessing and scoring sources

These files preserve the original input transformations and scoring logic. They retain historical absolute paths, mounts and execution guards; they are references, not the supported portable commands.

Use `scripts/prepare_sideview.py` and `scripts/score_sideview.py` as described in the bilingual README. The portable scorer directly imports `validate_tasks` and `score_embeddings` from the historical `score_sideview_variants.py`; importing it does not launch its guarded cloud main routine. The numerical scoring rule is therefore shared rather than reimplemented.

All four rebuilt arrays match the preserved local input cache SHA256 values in `protocols/preprocessing_verification.json`. This verifies preprocessing equivalence, not missing historical runtime input receipts. The portable encoder follows the historical DenseRGB inference structure, but the new full 45-model CUDA run has not been executed during archive packaging.
