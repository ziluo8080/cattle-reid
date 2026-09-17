# Historical Training Source Excerpts

These seven source files are byte-preserved excerpts recovered from the historical cloud runtime. Their hashes were checked against training release records. They document the actual network, three objectives, optimizer calls and learning-rate schedule.

This reference directory is **not itself the standalone training distribution**: its original imports and cloud authorizations are intentionally preserved. Use the separate portable workflow described below instead of directly launching these historical scripts. This archive does not grant a new historical cloud authorization.

For executable count verification, use `scripts/reproduce_results.py`. For standalone retraining and frozen scoring, use `scripts/holstein.py` and `scripts/score_sideview.py`, backed by `src/cattle_reid_repro/`. All 45 original checkpoints and required raw data archives are Release assets. See `docs/END_TO_END.md`; full retraining was not repeated during packaging. These historical files remain preserved references, not the portable launch commands.

`controlled_adapter_training.py` contains older adapter helpers as well as the shared learning-rate function. Only its `learning_rate` function is used by these DenseNet training entries. It must not be read as evidence that the present experiment uses the adapter architecture.
