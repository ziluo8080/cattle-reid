# Reproduction status (2026-09-17, dataset/weight completion)

English: [end-to-end workflow](END_TO_END.md). 中文：[完整复现流程](END_TO_END.zh-CN.md).

## Delivered

- Private GitHub Release `reproducibility-v1.2.0`: original Holstein Raw.zip and timestamp.xlsx, original SideView snapshots.zip, 45 frozen checkpoints, and 45 Holstein reference score matrices. Five attachments total 3,310,854,086 bytes. Each GitHub SHA256 matches the local upload source; original dataset archives also match official publisher checksums.
- License/attribution records, downloadable asset manifest, verification/extraction utility and official ImageNet initialization downloader.
- Standalone Holstein input reconstruction, exact task reconstruction, three-method training, frozen inference, retrained-model inference and historical matrix replay. Historical cloud permission scripts are not modified.

## Verified

- Four SideView variants rebuilt from the original local ZIP; all output NPY SHA256 values equal preserved local preprocessing arrays.
- Holstein train/test arrays rebuilt from original Raw.zip; every image and complete NPY checksum equals the archived preparation contract.
- 450 training epoch task hashes and five validation task plans exactly recovered.
- All 45 checkpoint hashes verified; training initialization, inputs and complete task dependency preflight passed without training.
- Fourteen tests passed, including three loss/gradient comparisons with preserved source and preprocessing/scoring rules.
- All 45 Holstein archived score matrices replayed: the six headline values match the original report before rounding.
- Actual new frozen inference for B/fold0/seed17 on the local RTX 3060 Laptop GPU produced Rank-1 0.8310679611650486 and Rank-5 0.9902912621359223. Its complete 3090 x 65 score matrix also matched the archived matrix exactly (maximum absolute difference 0). This is one-model integration evidence, not a repeat of all 45 model experiments or a hardware-independent equality guarantee.

## Boundaries that remain

- Full 45-model retraining and complete new 45-model GPU rescoring were not repeated during repository packaging. Commands and dependencies are provided; exact hardware-independent equality is not claimed.
- Portable training saves selected weights and final optimizer/RNG state but does not implement automated exact resume or preserve every full epoch checkpoint as the historical cloud workflow did.
- SideView historical baseline/neutral128 per-query matrices and runtime input-byte receipts remain absent. Newly computing them cannot retroactively create original evidence.
- Capture-event independence and missing historical host/driver/peak-memory/runtime records remain scientific/reporting limitations, not solved by file distribution.
- The repository is private at the owner's request. Other researchers require permission until the owner changes visibility. No permanent archive DOI is claimed.
