# Holstein and SideView Cattle Re-Identification

**English** | [简体中文](README.zh-CN.md)

## Manuscript v9 Update

The [Chinese working manuscript v9](docs/manuscript/manuscript-v9.zh-CN.docx) merges the citation-reviewed draft with the formula/evidence audit. See the [revision record and bibliography](docs/MANUSCRIPT_V9.md). This is a working draft, not a published article.

**Display-name mapping:** restricted **SupCon-out** (short label: SupCon-out) = archived `B` / `B-selected`. This is the existing query-anchor/support-comparison instance of SupCon-out, not a new loss. Commands, checkpoint names, CSV/JSON keys and equation superscript B are unchanged; continue using `--method B`.

The six files in `figures/` now match v9. Figure 6 uses a teal-gray scale and highlights the maximum in each K column: **50.10%, 58.35%, 61.82%**. Emphasis indicates point-estimate maxima, not statistical significance. The portable plotting script reproduces the values and labels in simplified, photograph-free layouts; it is not a pixel-identical reconstruction of the final manuscript artwork. No models were retrained or rescored for this update.

Code, protocols and historical results for Holstein full-candidate recognition and exploratory frozen-model transfer to SideViewCows2026. This is a research reproducibility archive, not a pretrained application. Repository visibility is controlled by its owner; the repository is currently private.

**New: datasets and all 45 frozen weights are now attached to the [reproduction asset release](https://github.com/ziluo8080/cattle-reid/releases/tag/reproducibility-v1.2.0). Start with the [clean-checkout, end-to-end instructions](docs/END_TO_END.md): download, checksum, reconstruct inputs, replay reference scores, run frozen inference, or retrain.**

## Contents

[Project structure](#project-structure) | [Quick start](#quick-start) | [Reproduction scope](#1-start-here-what-can-be-reproduced) | [Environment](#2-software-and-compute) | [Tables and figures](#3-reproduce-published-tables-and-figures-offline) | [Datasets](#4-data-acquisition-and-layout) | [Preprocessing](#5-image-preprocessing) | [Inference and training](#6-frozen-model-inference) | [Protocol](#7-experimental-protocol-and-training-configuration) | [Results](#8-reference-results) | [Limitations](#9-scientific-limitations) | [Troubleshooting](#10-troubleshooting) | [Citation and contributions](#11-citation-and-contributions)

## Project Structure

The tree below describes the checked-in repository; related files are grouped with `*` for readability. Downloaded assets and generated outputs are shown separately. Run supported commands from the repository root, not from `scripts/`.

```text
cattle-reid/
|-- README.md / README.zh-CN.md       # English / Chinese entry points
|-- LICENSE                          # Code license
|-- DATA_SOURCES.md                   # Original publishers and dataset scope
|-- DATA_LICENSES.md                  # Dataset licenses and attribution
|-- REPRODUCIBILITY.md                # Evidence and analysis conventions
|-- MANIFEST.json                     # Checksums for archived repository files
|-- requirements.txt                 # Result analysis and plotting dependencies
|-- requirements-inference.txt       # Pinned NumPy/Pillow; install torch separately
|-- data/
|   `-- README.md                    # Asset download and local storage instructions
|-- docs/
|   |-- END_TO_END.md                 # Complete English reproduction workflow
|   |-- END_TO_END.zh-CN.md           # Equivalent Chinese workflow
|   `-- REPRODUCTION_STATUS.md        # Tested steps and remaining verification limits
|-- scripts/                         # Supported runnable entry points
|   |-- fetch_assets.py              # Download, checksum and extract Release assets
|   |-- fetch_initialization.py      # Obtain original ImageNet initialization
|   |-- holstein.py                  # Prepare / verify-protocol / replay / train / score
|   |-- prepare_sideview.py           # Rebuild four SideView input variants
|   |-- score_sideview.py             # Frozen inference with fixed tasks
|   |-- reproduce_results.py          # Verify archived counts, tables and checksums
|   |-- identity_sensitivity.py       # Conditional paired identity resampling
|   `-- plot_paper_figures.py         # Recreate six paper figures
|-- src/cattle_reid_repro/             # Scientific implementation used by launchers
|   |-- densenet_pairwise.py          # Pairwise objective computations
|   |-- strong_rgb_densenet.py        # RGB encoder implementation
|   |-- supcon_in.py                  # SupCon-in objective
|   |-- legacy_math.py               # Preserved numerical helper functions
|   `-- evaluation_artifacts.py      # Load and validate evaluation artifacts
|-- protocols/                       # Fixed inputs to reproduction, not run outputs
|   |-- holstein/                    # Design, contract, image indices and task files
|   |-- holstein_manifest.csv        # Original RGB paths and hashes
|   |-- holstein_reference_scores.json
|   |-- sideview_manifest.jsonl      # Ordered list of 607 source images
|   |-- sideview_protocol.json       # Identity eligibility, supports and queries
|   |-- models.json                  # 45 checkpoint names, hashes and selected epochs
|   |-- training_config.json         # Actual historical training configuration
|   |-- environment.json             # Recorded software and hardware
|   |-- release_assets.json          # Attachment filenames, sizes and SHA256
|   |-- result_provenance.json        # Historical result-to-preprocessing mapping
|   |-- sensitivity_plan.json        # Fixed statistical analysis settings
|   `-- *verification.json / scientific_function_sources.json
|-- results/sideview/                 # Unchanged historical result counts
|   `-- baseline.json / neutral128.json / geomalign_v3.json / blacktrim_v1.json
|-- tables/                          # Machine-readable evidence and paper tables
|   |-- holstein_main_reported.csv / holstein_per_model.csv
|   |-- holstein_candidate_k.csv / holstein_date_k.csv
|   |-- holstein_cohort_383.csv / holstein_actual_split_1620.csv
|   |-- holstein_fold_counts.csv / model_lineage_45.csv
|   |-- sideview_verified.csv / sideview_all_available.csv
|   `-- sideview_identity_sensitivity.csv / training_consumption_45.csv
|-- figures/                         # Six archived figure sets, each SVG + PDF
|-- docs/manuscript/                 # Current Chinese Word working manuscript
|-- docs/MANUSCRIPT_V9.md             # Revision record and merged bibliography
|-- reference_pipeline/              # Historical preprocessing/scoring evidence
|-- reference_training/              # Historical training source evidence
`-- tests/                           # Four unittest modules, including protocol checks
```

After downloading assets and running the commands below, the **local, Git-ignored** layout is:

```text
data/assets/
|-- holstein-Raw.zip                  # Original Holstein archive; read directly
|-- holstein-timestamp.xlsx           # Original timestamp metadata
|-- sideview-snapshots.zip            # Experiment's images and official masks
|-- cattle-reid-checkpoints-v1.zip
|-- holstein-reference-scores.zip
|-- checkpoints/                     # 45 original trained state dictionaries
|-- holstein-reference-scores/        # 45 archived NPY score matrices
`-- densenet121-a639ec97.pth           # Separate download, only needed for retraining
derived/
|-- holstein-input/                  # Reconstructed Holstein arrays and receipt
|-- inputs/neutral128/               # SideView uint8 array and processing receipt
|-- holstein-reference-replay/        # Recomputed summary from historical matrices
|-- holstein-new-scores/              # New frozen Holstein inference output
|-- sideview-neutral128-new-scores/   # New frozen SideView inference output
|-- training/                        # New training jobs, never historical weights
|-- figures/                         # Regenerated SVG/PDF and PNG previews
`-- identity_sensitivity.csv         # Regenerated sensitivity estimates
```

`data/` therefore exists in Git, but its large assets are delivered through Release. Cloning is not downloading the datasets. Do not move new predictions into `results/`, change the fixed image order, or use historical reference scripts as portable launchers.

## Quick Start

Choose the path matching your goal. These are separate levels of reproduction, not equivalent evidence:

| Goal | Steps | GPU / large downloads |
| --- | --- | --- |
| Check the reported numbers | Clone; run `python scripts/reproduce_results.py` | Neither; standard Python only |
| Recreate tables and figures | Install analysis dependencies; follow section 3 | No GPU; no image archives |
| Replay original Holstein predictions | Install inference dependencies; fetch assets; run `holstein.py replay` | No GPU inference; reference matrices required |
| Recompute predictions from images | Install CUDA environment; fetch assets; prepare inputs; score | GPU recommended for Holstein and required by the SideView entry point |
| Retrain the study models | Complete input setup; fetch initialization; run training and rescoring | 45 training jobs for the full study; see section 6 |

The minimum first check, after cloning and changing into this repository, is:

```bash
python scripts/reproduce_results.py
```

For the full workflow, first install the environment in section 2, then authenticate GitHub CLI and obtain the assets:

```bash
gh auth login
python scripts/fetch_assets.py --output data/assets --extract
python scripts/holstein.py verify-protocol
python scripts/holstein.py replay --scores data/assets/holstein-reference-scores --output derived/holstein-reference-replay
```

Plan for at least 10 GB of working storage for downloads, extracted checkpoints, inputs and inference outputs, and additional storage for retraining. This is a planning allowance, not a measured peak. For a command-by-command walkthrough, read [END_TO_END.md](docs/END_TO_END.md).

## 1. Start Here: What Can Be Reproduced?

### Where are the datasets?

Open **[data/README.md](data/README.md)** for the dataset entry point. Original data and weights are in this repository's **[Release attachments](https://github.com/ziluo8080/cattle-reid/releases/tag/reproducibility-v1.2.0)**, not individual Git files. They include Holstein Raw/timestamps and the SideView snapshots images/masks used in this experiment. Cloning alone does not download them; after authenticating the GitHub CLI, run from the repository root:

```bash
python scripts/fetch_assets.py --output data/assets --extract
```

Files are downloaded to `data/assets/`, with weights in `data/assets/checkpoints/` and reference scores in `data/assets/holstein-reference-scores/`. The five attachments total approximately 3.31 GB. Only the data README is tracked; downloaded files are ignored. Private-repository access is required. Manual download and offline verification instructions are provided in the data README.

| Level | Available now | Additional requirements |
| --- | --- | --- |
| Historical result verification | All four SideView count files; Holstein model-level summaries; checksum verifier | Python 3.11+; CPU only |
| Conditional sensitivity analysis and six vector figures | Executable scripts, fixed analysis plan and source tables | NumPy, plotting dependencies and a Chinese font |
| SideView image preprocessing | Portable entry point for all four variants; original image manifest | Official `snapshots.zip`, including masks for gray background and geomalign |
| Frozen SideView inference | Portable scorer, fixed task indices and 45 original checkpoints in Release | Download/extract assets; CUDA environment |
| Holstein retraining and rescoring | Standalone launcher, actual task definitions, initialization downloader, data, weights and original score matrices | Follow the end-to-end guide; full 45-model retraining was not rerun during packaging |

Do not interpret count verification as end-to-end experimental reproduction. Portable adapters are newly packaged code; they do not replace historical execution receipts. One B/fold0/seed17 frozen Holstein inference check was run during packaging and reproduced its historical Rank-1/Rank-5. Full 45-model retraining was not performed.

## 2. Software and Compute

### Recorded experimental environment

| Component | Historical value |
| --- | --- |
| Platform | Linux cloud runtime |
| GPU | One NVIDIA GeForce RTX 4090 |
| Python | 3.11.15 |
| PyTorch | 2.7.1+cu126 |
| torchvision | 0.22.1+cu126 |
| NumPy | 2.2.6 |
| Pillow | 11.3.0 |
| Training arithmetic | FP32; no AMP; TF32 disabled |
| Frozen inference batch size | Holstein: 1; SideView: 64 |

Exact CPU model, host RAM, NVIDIA driver, Linux distribution, peak GPU memory and full-study wall-clock/GPU-hours have not been recovered as a complete verified record. They are not invented here. The `cu126` wheel build is not a claim that a separately installed CUDA toolkit had the same version.

Training uses 630 updates/model, 60 image inputs/update, 15 models/method and 3 methods: **28,350 updates and 1,701,000 training image presentations**, excluding validation, checkpoint verification and replay. These are operation counts, not unique photographs or measured GPU-hours. See [per-model consumption](tables/training_consumption_45.csv).

One SideView variant embeds 607 images with each of 45 models: 27,315 image forward passes. Five draws and three K values reuse these embeddings. Four variants correspond to 109,260 image passes if run once each; historical retries/replays are not included in that arithmetic. A uint8 input cache occupies approximately 91.4 MB. All 45 checkpoint files total approximately 1.28 GB before compression; they are stored as a Release attachment rather than ordinary Git history.

### Installation

Clone this repository with your normal GitHub access. While private, collaborators need repository permission. Never put an access token into a command or README.

```bash
git clone https://github.com/ziluo8080/cattle-reid.git
cd cattle-reid
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For preprocessing only, `requirements-inference.txt` supplies NumPy/Pillow. For GPU inference, use a separate environment with Python 3.11.15 and install the matching wheels:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-inference.txt
python -c "import torch, torchvision, numpy, PIL; print(torch.__version__, torchvision.__version__, numpy.__version__, PIL.__version__); print(torch.cuda.is_available())"
```

The historical versions above are recorded experimental versions. Plotting version ranges in `requirements.txt` are utility dependencies, not a recovered full training lockfile. Use Microsoft YaHei or another supported Chinese font for the Chinese figure labels; missing fonts can produce empty glyph boxes.

## 3. Reproduce Published Tables and Figures Offline

From the repository root (the full test suite now requires the PyTorch environment):

```bash
python scripts/reproduce_results.py
python -m unittest discover -s tests
python scripts/identity_sensitivity.py
python scripts/plot_paper_figures.py
```

- The verifier needs only the Python standard library. It checks `MANIFEST.json`, 2,700 SideView records, 145,800 identity correct/total entries, 36 micro/macro result pairs and the six Holstein headline numbers against model-level tables. It does not regenerate Holstein predictions from images.
- Sensitivity output: `derived/identity_sensitivity.csv`. It uses 10,000 paired identity-cluster resamples, NumPy PCG64 seed 20260917 and percentile bounds. The gallery, models and support draws remain fixed. These are exploratory conditional intervals, not new-farm generalization confidence intervals.
- Figure outputs: `derived/figures/`; SVG and PDF are publication assets, PNG is an inspection preview. Archived figure originals stay in `figures/`.
- Generated outputs are ignored by Git. Do not overwrite historical `results/` files with a new run.

## 4. Data Acquisition and Layout

Use the repository's Release mirrors or the original publishers, following [DATA_SOURCES.md](DATA_SOURCES.md) and [DATA_LICENSES.md](DATA_LICENSES.md). Mirrored archive bytes were checked against publisher checksums:

- Holstein: *Recognition of Holstein Cattle with Thermal and RGB images*, DOI `10.34894/7M108F`. This manuscript uses RGB, not a thermal/RGB fusion experiment.
- SideViewCows2026: *SideViewCows2026 - Dairy Cow Re-Identification Dataset*, version DOI `10.5281/zenodo.21605650`. Only `snapshots.zip` is needed for the archived experiment, not a newly selected parlor-to-snapshots protocol.

Expected entries inside the SideView ZIP:

```text
snapshots/images/<identity>/<image>.jpg
snapshots/masks/<identity>/<image>.png
```

`protocols/sideview_manifest.jsonl` binds every index to an image filename, identity and source SHA256. Keep its order: the task protocol uses integer array indices. `protocols/holstein_manifest.csv` contains original RGB path/hash metadata; `tables/holstein_actual_split_1620.csv` records the actual fold assignments. Source images/masks are distributed in the licensed original Release archives, not as individual Git files.

## 5. Image Preprocessing

### Holstein training input

RGB; resize the short edge to 256 with bilinear interpolation; center crop 224 x 224; scale to [0,1]; ImageNet mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`. No random augmentation in the verified training configuration. Do not apply SideView letterboxing to reproduce Holstein training.

### SideView input

All variants preserve the original 607-image manifest and produce `uint8 [607,3,224,224]` arrays. EXIF orientation is applied before RGB conversion. Do not center-crop these arrays again. ImageNet normalization occurs once, inside inference.

| Variant | Exact processing rule |
| --- | --- |
| `baseline` | Whole-image aspect-preserving LANCZOS resize into 224 x 224; centered RGB(128,128,128) padding; no crop |
| `neutral128` | Official mask > 0 is foreground; replace outside-mask pixels with RGB(128,128,128), then baseline letterbox; no bounding-box crop |
| `geomalign_v3` | Mask bounding box plus max(8 pixels, round(15% of longest bbox side)) margin, clipped to image; retain original background pixels; resize longest crop side to 200; center on 224 gray canvas |
| `blacktrim_v1` | Trim consecutive edge rows/columns with at least 95% pixels having max RGB <= 8; historical 40%-per-edge loop cap; baseline letterbox afterwards |

```bash
python scripts/prepare_sideview.py --zip /path/to/snapshots.zip --variant baseline --output derived/inputs/baseline
python scripts/prepare_sideview.py --zip /path/to/snapshots.zip --variant neutral128 --output derived/inputs/neutral128
python scripts/prepare_sideview.py --zip /path/to/snapshots.zip --variant geomalign_v3 --output derived/inputs/geomalign_v3
python scripts/prepare_sideview.py --zip /path/to/snapshots.zip --variant blacktrim_v1 --output derived/inputs/blacktrim_v1
```

Replace `/path/to/snapshots.zip` with your file. Output directories must not exist. Each run verifies all 607 original image hashes and writes `images_uint8.npy` plus `processing_receipt.json`. Original mask hashes are not separately frozen in this portable adapter; obtain the exact official ZIP version. The adapter is checked against existing locally processed arrays, not used to select a new preprocessing policy.

## 6. Frozen Model Inference

The methods use a DenseNet-121 feature extractor, spatial mean pooling and a 1024-dimensional L2-normalized embedding. Their trained weights differ. BatchNorm uses stored running statistics at inference. The classifier is absent.

Download and extract `cattle-reid-checkpoints-v1.zip` from Release. The **original raw state-dict** checkpoints are named exactly as `protocols/models.json`, for example `B-fold0-seed17.pt`, `GAP-fold0-seed17.pt`, `SupCon-in-fold0-seed17.pt`. The roster contains 45 SHA256 hashes and selected epochs. Do not substitute ImageNet weights or guess the format; `strict=True` loading and hashes reject substitutions.

```bash
python scripts/score_sideview.py --input derived/inputs/neutral128 --weights /path/to/checkpoints --output derived/inference/neutral128 --check-only
# Run only after supplying all 45 original checkpoints and the CUDA environment:
python scripts/score_sideview.py --input derived/inputs/neutral128 --weights /path/to/checkpoints --output derived/inference/neutral128
```

The second command is a new inference job, not a table-only check. It performs no training. It saves per-model score matrices and combined `results.json` under the new output directory. The archive preparation validated code/inputs, but did not run this new 45-model CUDA job. Exact bitwise output equivalence on other GPU/software environments is not claimed.

**Scoring rule:** normalize each image embedding; average the K support embeddings for an identity; take the dot product with the normalized query. **Do not normalize the mean support vector again.** This equals mean query-to-support cosine similarity. Candidate order is fixed; NumPy argmax selects the first maximum on a tie.

### Holstein: rebuild inputs and rescore

After asset extraction, use these paths directly:

```bash
python scripts/holstein.py prepare --raw-zip data/assets/holstein-Raw.zip --output derived/holstein-input
# Small integration run: only B / fold 0 / seed 17, not all headline results.
python scripts/holstein.py score --input derived/holstein-input --weights data/assets/checkpoints --method B --fold 0 --seed 17 --output derived/holstein-B-fold0-seed17
# Full frozen evaluation: 3 methods x 5 folds x 3 seeds.
python scripts/holstein.py score --input derived/holstein-input --weights data/assets/checkpoints --output derived/holstein-new-scores
```

The full run writes `summary.json`; compare it against the reference replay, not a single model's result. Holstein uses single-image encoding and CPU pairwise scoring. SideView uses batched encoding and the mean-cosine rule above; do not interchange their numerical reduction paths.

### Retraining is optional, not part of external inference

To reproduce training, first download the exact ImageNet initialization and run a preflight. **Do not train on SideView.** Its role here is frozen-model transfer.

```bash
python scripts/fetch_initialization.py --output data/assets/densenet121-a639ec97.pth
python scripts/holstein.py train --input derived/holstein-input --init-weight data/assets/densenet121-a639ec97.pth --method B --fold 0 --seed 17 --output derived/training/B-fold0-seed17 --check-only
# Remove --check-only to actually train this model.
```

Repeat training for methods `B`, `GAP`, `SupCon-in`, folds `0..4` and seeds `17,29,43`, keeping output names `<method>-fold<fold>-seed<seed>`. The complete Bash loop is in [the end-to-end guide](docs/END_TO_END.md). Once all 45 jobs complete:

```bash
python scripts/holstein.py score --input derived/holstein-input --trained-root derived/training --output derived/holstein-retrained-scores
python scripts/score_sideview.py --input derived/inputs/neutral128 --trained-root derived/training --output derived/sideview-retrained-scores
```

Each training job records `selected.pt`, `last-state.pt`, `history.json`, epoch validation matrices and `completion.json`. Use `--weights` for archived original checkpoints and `--trained-root` for newly trained job directories; they are mutually exclusive. Training does not implement automatic resume. A saved optimizer/RNG state is not a promise of automatic recovery. Full retraining may differ across environments and has not been rerun as part of packaging.

## 7. Experimental Protocol and Training Configuration

Holstein retains 324 eligible identities from the 383-identity audit. Use the archived actual splits, not a fresh random split. Five folds x seeds 17/29/43 give 15 frozen models per method. A task uses all 64 or 65 test-fold identities; the headline mixes K=1,3,5 and averages model-level values equally. The 15,735-task workload must not be interpreted as 15,735 independent animals. The 88.00% B result belongs to the 65-candidate/K=5 stratum, not the main pooled endpoint.

SideView has 607 images/63 identities before eligibility filtering, 54 identities/577 images after requiring >=6 images per identity. Five fixed draws reserve five supports per identity; K=1 and K=3 use prefixes of K=5; the remaining 307 images are queries in each draw. Unused supports do not become queries. Each preprocessing result contains 45 models x 5 draws x 3 K = 675 records.

| Training setting | Value |
| --- | --- |
| Initialization | DenseNet-121 ImageNet IMAGENET1K_V1; hash in `protocols/training_config.json` |
| Trainable parameters | All feature-extractor convolution and BN affine parameters; BN running statistics fixed |
| Optimizer | AdamW; LR 1e-4; weight decay 1e-4; betas (0.9,0.999); eps 1e-8 |
| Other optimizer settings | amsgrad/foreach/fused false; gradient norm clip 1 |
| Schedule | 30 epochs x 21 updates; 42-update linear warm-up, then per-update cosine decay to zero at update 630 |
| Episode | 10 identities; 10 queries + 50 supports; five supports/identity |
| Objective configuration | Temperature 0.1; denominator 10; support gradients enabled; method-specific losses in source files |
| Checkpoint selection | Highest validation identity-macro Rank-1 at K=5 among epochs 0..30; earliest tie |
| Selection boundary | Validation 10-way tasks; post-selection full-validation check does not reselect |

Read the actual source for B/GAP/SupCon-in rather than treating their labels as interchangeable loss definitions. `reference_training/` preserves evidence; `src/cattle_reid_repro/` and `scripts/holstein.py` provide the standalone computation. See [end-to-end commands](docs/END_TO_END.md) and [verification boundaries](docs/REPRODUCTION_STATUS.md).

## 8. Reference Results

| Holstein method | Rank-1 | Rank-5 |
| --- | ---: | ---: |
| SupCon-out (archive: B-selected) | 83.67% | 98.78% |
| GAP | 74.10% | 96.74% |
| SupCon-in | 65.52% | 90.72% |

| SideView gray-background method | K=1 | K=3 | K=5 |
| --- | ---: | ---: | ---: |
| SupCon-out (archive: B) | 50.10% | 58.35% | 61.82% |
| GAP | 39.31% | 47.16% | 51.12% |
| SupCon-in | 28.42% | 32.86% | 35.21% |

SideView table values are query micro-accuracy averaged over the 15 models and five draws. Identity-macro accuracy is separately reported in `tables/sideview_verified.csv`. Never mix these estimands, or compare different K/candidate counts as a controlled improvement.

### Result and figure navigation

| Question | Archived evidence |
| --- | --- |
| Where do Holstein headline values come from? | [Per-model values](tables/holstein_per_model.csv), [main table](tables/holstein_main_reported.csv), and Release score matrices |
| Why is there also an 88% result? | [Candidate-count / K strata](tables/holstein_candidate_k.csv); not the pooled endpoint |
| Which animals/images belong to each fold? | [Actual split](tables/holstein_actual_split_1620.csv), [fold counts](tables/holstein_fold_counts.csv) |
| Where are all SideView preprocessing comparisons? | [Verified micro/macro table](tables/sideview_verified.csv), [available comparisons](tables/sideview_all_available.csv), and `results/sideview/` |
| How do checkpoints map to training runs? | [Model lineage](tables/model_lineage_45.csv), [checkpoint roster](protocols/models.json) |
| Which steps were actually checked? | [Verification status](docs/REPRODUCTION_STATUS.md) |

The six SVG/PDF sets cover study protocol, model/scoring structure, Holstein main results, support-count results, SideView transfer and preprocessing comparison. [Figure 1](figures/Fig1_study_protocol.svg) and [Figure 2](figures/Fig2_model_scoring.svg) provide the workflow and model diagrams. These archived paper figures use Chinese labels; the executable workflow is documented in both languages.

## 9. Scientific Limitations

Historical pipeline files contain original workstation/cloud paths and execution guards. Do not directly run them after replacing paths or removing safety gates. Use the portable commands above for the specifically supported steps.

The historical neutral128 JSON mistakenly says letterbox in one metadata field; original bytes are retained and `protocols/result_provenance.json` resolves the variant using historical launch evidence. Baseline/neutral128 per-query score matrices and original runtime input-byte receipts remain unavailable. Source preprocessing outputs can be rebuilt, but that does not retroactively create a historical receipt.

SideView is a **publicly sourced, reused dataset**, not data collected by this repository's author. Its preprocessing was repeatedly compared using external outcomes. Capture-event independence is unverified. Results are exploratory image-level transfer, not untouched confirmatory external validation. Official identity labels are reused; no independent manual labeling study is claimed.

The original `paper-evidence-v1.0.0` tag remains unchanged. Cite the exact commit of the version you use. Repository code retains MIT licensing; original datasets and pretrained weights retain their own terms. No paper DOI or permanent archive DOI is invented. Repository privacy will be changed only by its owner.

## 10. Troubleshooting

| Symptom | Check or action |
| --- | --- |
| Only README exists in `data/` after cloning | Expected: download Release assets; image archives and weights are not ordinary Git files |
| Private Release returns 404 or an access error | Confirm repository permission and run `gh auth login` normally; never put tokens in files |
| GitHub CLI is unavailable | Download all five attachments in your signed-in browser into `data/assets/`, then verify offline below |
| Size, SHA256 or input checks fail | Stop inference; check the asset version, download completeness and modified images; do not disable validation |
| Output directory already exists | Use a fresh directory to preserve evidence; partial training jobs are not automatically resumed |
| `torch.cuda.is_available()` is false | Check NVIDIA driver, active Python environment and CUDA wheels; count verification needs no GPU |
| One model differs from the headline | The headline averages 15 models; check method/fold/seed, K, candidates and micro/macro definitions |
| Figures show missing glyphs | Install a supported Chinese font and rerun the figure script |

```bash
python scripts/fetch_assets.py --output data/assets --verify-only --extract
python scripts/holstein.py --help
python scripts/holstein.py train --help
python scripts/score_sideview.py --help
```

Offline verification does not download missing attachments. The `--help` commands do not train or infer. Individual commands do not require Linux continuation syntax; the complete training loop in the detailed guide uses Bash. Windows users can use Bash or invoke each single-model command separately.

## 11. Citation and Contributions

Code licensing is in [LICENSE](LICENSE); datasets retain the CC0/CC BY 4.0 terms documented in [DATA_LICENSES.md](DATA_LICENSES.md), not the repository's MIT license. Cite original dataset authors, titles and DOIs. For this archive, record the repository URL, actual commit and asset release `reproducibility-v1.2.0`. No published manuscript DOI is available here; no invented BibTeX record is supplied.

```bash
git rev-parse HEAD
```

When reporting reproduction issues, include the commit, method/fold/seed/preprocessing, command, Python/torch/torchvision versions, GPU and error log. Exclude credentials and personal access tokens. For code changes, run `python -m unittest discover -s tests`; never edit historical results to make tests pass. Changes to archived tracked files require an explicitly documented revision and corresponding `MANIFEST.json` updates, not substitution of new results for original evidence.

Documentation organization draws on [FastReID](https://github.com/JDAI-CV/fast-reid)'s installation/quick-start/model navigation and [Torchreid](https://github.com/KaiyangZhou/deep-person-reid)'s environment/training/cross-domain evaluation structure. This does not introduce their models, dependencies or experimental claims into this project.

---
[Back to top](#holstein-and-sideview-cattle-re-identification) | [切换至中文说明](README.zh-CN.md)
