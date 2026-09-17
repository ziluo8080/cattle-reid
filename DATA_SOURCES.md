# Third-Party Data and Release Scope

The study reuses public datasets; the repository owner did not independently collect SideView images.

- Holstein: *Recognition of Holstein Cattle with Thermal and RGB images*, DOI `10.34894/7M108F`. Obtain original data and applicable terms from its publisher.
- SideViewCows2026: Sebastian Möller, *SideViewCows2026 - Dairy Cow Re-Identification Dataset*, version DOI `10.5281/zenodo.21605650`. The recorded dataset license is CC BY 4.0. Only its snapshots subset is analyzed.

The repository's `reproducibility-v1.2.0` Release now mirrors the original Holstein Raw.zip/timestamp.xlsx (CC0-1.0), SideView snapshots.zip including masks (CC BY 4.0), 45 trained checkpoints and 45 Holstein reference score matrices. See `DATA_LICENSES.md` for attribution and `protocols/release_assets.json` for exact checksums. These large files are Release assets, not ordinary Git history. ImageNet initialization is fetched separately from the official PyTorch distribution. Copyrighted paper PDFs, author-placeholder manuscripts, access tokens, private session logs and authentication files are not distributed.

The repository's existing MIT license is retained for repository-authored code and documentation. It does not change the licenses of third-party datasets, pretrained weights or cited publications. The source DOI records remain the authoritative locations for original images and terms.

This GitHub repository is a version-controlled research archive, currently private at the owner's request, not a claim of DOI-backed permanent preservation. No Zenodo DOI has been minted for this code/result release. Cite the exact release tag and commit; do not invent an archive DOI or paper authorship.
