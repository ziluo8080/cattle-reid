# Reproduce from a clean checkout

**English** | [简体中文](END_TO_END.zh-CN.md)

This workflow packages the original data inputs, 45 frozen checkpoints, exact task definitions, reference score matrices and standalone scientific computation code. Repository access is required while private. It does not guarantee bitwise retraining equivalence on different hardware, and it does not represent a new independent external-validation experiment.

## 1. Environment and assets

Use Python 3.11.15. Historical GPU software is PyTorch 2.7.1+cu126 / torchvision 0.22.1+cu126, NumPy 2.2.6 and Pillow 11.3.0. Install the official CUDA wheels as shown in the README, then:

```bash
python -m pip install -r requirements-inference.txt
# GitHub CLI must be installed separately and authenticated normally:
gh auth login
python scripts/fetch_assets.py --output data/assets --extract
python scripts/fetch_initialization.py --output data/assets/densenet121-a639ec97.pth
```

Alternatively download each attachment from the `reproducibility-v1.2.0` release using your signed-in browser, place it in `data/assets`, then run:

```bash
python scripts/fetch_assets.py --output data/assets --verify-only --extract
```

No tokens belong in files or commands. The fetch script invokes the official GitHub CLI, verifies file size/SHA256 and does not silently overwrite corrupt assets. For a private repository, anonymous download links will not work. The original ImageNet initialization is separately downloaded from the official PyTorch host and checked against the complete archived SHA256.

After extraction:

```text
data/assets/
  holstein-Raw.zip
  holstein-timestamp.xlsx
  sideview-snapshots.zip
  cattle-reid-checkpoints-v1.zip
  holstein-reference-scores.zip
  checkpoints/                         # 45 state dictionaries
  holstein-reference-scores/            # 45 NPY score matrices
  densenet121-a639ec97.pth               # initialization, not a trained model
```

The assets total roughly 3.3 GB compressed; allow at least 10 GB working storage for download, extracted weights, prepared arrays and new inference output. This is a planning allowance, not a measured training disk peak. Training retains selected weights, the last optimizer/RNG state and validation arrays rather than all 31 historical checkpoint snapshots. Allow additional storage for 45 jobs. Never use `/app` for cloud experiment output; use your mounted persistent storage for results and `/localdisk-tmp` for ephemeral caches.

## 2. Verify without model inference

```bash
python scripts/reproduce_results.py
python -m unittest discover -s tests
python scripts/holstein.py verify-protocol
python scripts/holstein.py replay --scores data/assets/holstein-reference-scores --output derived/holstein-reference-replay
```

The complete tests require PyTorch/torchvision; the original count verifier alone needs only standard Python. Protocol verification reconstructs all 450 fold/seed/epoch training plans and five validation plans and compares their original hashes. Matrix replay uses all 45 archived score arrays to recover the Holstein headline:

| Method | Rank-1 (%) | Rank-5 (%) |
| --- | ---: | ---: |
| B | 83.6676095217 | 98.7769693904 |
| GAP | 74.1038583543 | 96.7412518589 |
| SupCon-in | 65.5235049605 | 90.7228432982 |

The headline first averages tasks within each model, then weights the 15 models equally. Do not replace it with an identity/day-macro estimand.

## 3. Rebuild both datasets' inputs

```bash
python scripts/holstein.py prepare --raw-zip data/assets/holstein-Raw.zip --output derived/holstein-input
python scripts/prepare_sideview.py --zip data/assets/sideview-snapshots.zip --variant neutral128 --output derived/inputs/neutral128
```

Holstein checks original file hashes, every transformed image and complete train/test NPY bytes against the original contract. SideView checks the original 607-image list; all four variants have verified historical array hashes in `protocols/preprocessing_verification.json`. Repeat SideView with `baseline`, `geomalign_v3` or `blacktrim_v1` for other preprocessing results. Do not use the official Holstein Preprocessed.zip instead of reconstructing the recorded short-edge-256/center-224 recipe.

## 4. Rescore the frozen original models

```bash
python scripts/holstein.py score --input derived/holstein-input --weights data/assets/checkpoints --output derived/holstein-new-scores
python scripts/score_sideview.py --input derived/inputs/neutral128 --weights data/assets/checkpoints --output derived/sideview-neutral128-new-scores
```

Both commands load the fixed original checkpoints and perform no training. Holstein uses historical single-image encoding and CPU pairwise scoring; SideView uses batches of 64 and its own fixed mean-cosine rule. Do not unify the two numerical reductions merely because both use normalized embeddings. Store new results separately; historical results are not overwritten. Compare generated Holstein `summary.json` with the reference replay and SideView `results.json` with `results/sideview/neutral128.json` by `(method,fold,seed,draw,k)`, not JSON file bytes. Different runtime metadata/serialization is expected; near-tied scores can be hardware-sensitive.

A narrow Holstein integration check can use `--method B --fold 0 --seed 17`. Without filters the command scores all 45 models. A filtered run is not validation of the complete matrix.

## 5. Retrain from ImageNet initialization

First perform the non-training preflight:

```bash
python scripts/holstein.py train --input derived/holstein-input --init-weight data/assets/densenet121-a639ec97.pth --method B --fold 0 --seed 17 --output derived/training/B-fold0-seed17 --check-only
```

Then remove `--check-only` to train this one model through all 30 epochs and select from epoch 0..30 by the original validation criterion. For all models (Bash):

```bash
for method in B GAP SupCon-in; do
  for fold in 0 1 2 3 4; do
    for seed in 17 29 43; do
      python scripts/holstein.py train --input derived/holstein-input --init-weight data/assets/densenet121-a639ec97.pth --method "$method" --fold "$fold" --seed "$seed" --output "derived/training/$method-fold$fold-seed$seed" || exit 1
    done
  done
done
python scripts/holstein.py score --input derived/holstein-input --trained-root derived/training --output derived/holstein-retrained-scores
python scripts/score_sideview.py --input derived/inputs/neutral128 --trained-root derived/training --output derived/sideview-retrained-scores
```

Output directories must be new. This launcher does not support automatic resume of interrupted training; inspect retained state and restart into a new directory. It never reuses a partial job as complete. `selected.pt`, `last-state.pt`, `history.json`, epoch validation matrices and `completion.json` record each new job. The last state includes optimizer and RNG state; retaining it is not a claim that automatic exact recovery has been implemented.

The three losses, DenseRGB encoder and hash-based sampling use preserved scientific functions. Their source-function hashes are in `protocols/scientific_function_sources.json`. The historical design object contains older adapter configuration fields, but only its original sample pools/fold/task fields are used here; the actual DenseNet settings come from the current training implementation and `protocols/training_config.json`. Original cloud permission scripts remain untouched in the evidence sources; this is a separate user-invoked reproduction workflow, not a forged historical authorization.

## 6. What has actually been tested?

See [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md). Array reconstruction, exact task reconstruction, all 45 score-matrix replay, weight hashes and synthetic loss/gradient tests have been checked. Full 45-model retraining has not been rerun during repository preparation. Do not translate runnable commands into an unperformed experimental validation claim. Single-model inference checks, when recorded, apply only to the named model and environment.

For the original labels, source licenses and transformations, retain [DATA_LICENSES.md](../DATA_LICENSES.md). SideView remains exploratory and image-level, not a newly independent confirmation dataset.
