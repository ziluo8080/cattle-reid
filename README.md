# Holstein and SideView Cattle Re-Identification

**English** | [简体中文](README.zh-CN.md)

Code, protocols and historical results for Holstein full-candidate recognition and exploratory frozen-model transfer to SideViewCows2026. This is a research reproducibility archive, not a pretrained application. Repository visibility is controlled by its owner; the repository is currently private.

**New: datasets and all 45 frozen weights are now attached to the [reproduction asset release](https://github.com/ziluo8080/cattle-reid/releases/tag/reproducibility-v1.2.0). Start with the [clean-checkout, end-to-end instructions](docs/END_TO_END.md): download, checksum, reconstruct inputs, replay reference scores, run frozen inference, or retrain.**

## 1. Start Here: What Can Be Reproduced?

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
| Frozen inference batch size | 64 |

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
| B-selected | 83.67% | 98.78% |
| GAP | 74.10% | 96.74% |
| SupCon-in | 65.52% | 90.72% |

| SideView gray-background method | K=1 | K=3 | K=5 |
| --- | ---: | ---: | ---: |
| B | 50.10% | 58.35% | 61.82% |
| GAP | 39.31% | 47.16% | 51.12% |
| SupCon-in | 28.42% | 32.86% | 35.21% |

SideView table values are query micro-accuracy averaged over the 15 models and five draws. Identity-macro accuracy is separately reported in `tables/sideview_verified.csv`. Never mix these estimands, or compare different K/candidate counts as a controlled improvement.

## 9. Repository Map and Scientific Limitations

```text
protocols/            image indices, fixed tasks, configuration, checkpoint roster
results/sideview/     four unchanged historical JSON result files
tables/              source tables, cohort/split and checkpoint lineage metadata
scripts/             supported portable CLI entry points
reference_pipeline/  historical preprocessing/scoring sources, not portable launchers
reference_training/  verified historical training source excerpts
figures/             six archived SVG/PDF figure sets
tests/               verification and preprocessing tests
derived/             ignored, locally regenerated outputs
MANIFEST.json        SHA256 inventory (does not hash itself)
```

Historical pipeline files contain original workstation/cloud paths and execution guards. Do not directly run them after replacing paths or removing safety gates. Use the portable commands above for the specifically supported steps.

The historical neutral128 JSON mistakenly says letterbox in one metadata field; original bytes are retained and `protocols/result_provenance.json` resolves the variant using historical launch evidence. Baseline/neutral128 per-query score matrices and original runtime input-byte receipts remain unavailable. Source preprocessing outputs can be rebuilt, but that does not retroactively create a historical receipt.

SideView is a **publicly sourced, reused dataset**, not data collected by this repository's author. Its preprocessing was repeatedly compared using external outcomes. Capture-event independence is unverified. Results are exploratory image-level transfer, not untouched confirmatory external validation. Official identity labels are reused; no independent manual labeling study is claimed.

The original `paper-evidence-v1.0.0` tag remains unchanged. Cite the exact commit of the version you use. Repository code retains MIT licensing; original datasets and pretrained weights retain their own terms. No paper DOI or permanent archive DOI is invented. Repository privacy will be changed only by its owner.

---
[Back to top](#holstein-and-sideview-cattle-re-identification) | [切换至中文说明](README.zh-CN.md)
