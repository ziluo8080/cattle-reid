# Reproduction status (2026-09-17)

English: [README](../README.md). 中文：[中文说明](../README.zh-CN.md).

## Verified in this packaging run

- Historical count/table verification, without new training or model scoring.
- All four portable preprocessing commands ran against the official local snapshots ZIP; all 607 image hashes checked for each variant, and output NPY files matched the preserved local baseline/neutral128/geomalign/blacktrim files byte-for-byte.
- All 45 original local checkpoint files matched the sanitized roster hashes. The portable inference `--check-only` path passed input and task validation.
- Synthetic preprocessing/scoring tests cover padding, masking, empty-mask rejection, black-edge trimming, mean-cosine equivalence, support/query overlap rejection and tie ordering.

## Still required for complete reproduction

1. Checkpoint distribution: 45 original files were found locally, totaling 1,278,632,235 bytes, and checked. Release API access failed with a network connection timeout before creating a release or uploading an asset. The local ZIP is prepared separately; this repository contains hashes, not a working weight-download link. Do not imply publication has succeeded.
2. Full Holstein training/evaluation packaging: the preserved training snippets still depend on internal materialization/validation modules and historical task artifacts. A new training launcher has not been substituted for those dependencies. No end-to-end retraining command is advertised.
3. Full 45-model portable CUDA inference: not executed during this packaging run. Input reconstruction/checkpoint verification and synthetic scoring tests do not establish real GPU inference equivalence by themselves.
4. Missing historical evidence: baseline/neutral128 per-query score matrices and runtime input-byte receipts; independent capture-event evidence; complete host/driver/peak-memory/runtime records.

Repository privacy remains unchanged. The original evidence tag remains immutable. No weights, images or temporary credentials are committed to ordinary Git history.
