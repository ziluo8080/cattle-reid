# Dataset attribution and redistribution

The official metadata endpoints were checked on 2026-09-17. Repository privacy does not replace source licensing requirements.

## Holstein

- Title: Recognition of Holstein Cattle with Thermal and RGB images.
- Dataset identifier: https://doi.org/10.34894/7M108F ; version 2.0.
- Authoritative metadata: https://dataverse.nl/api/datasets/:persistentId/?persistentId=doi:10.34894/7M108F
- License: CC0-1.0, https://creativecommons.org/publicdomain/zero/1.0/ .
- Original files mirrored: `Raw.zip` and `timestamp.xlsx`. Release filenames have a `holstein-` prefix only; file bytes are unchanged.
- This experiment uses the `Raw/RGB (640 x 480)` channel. The unmodified original Raw archive also contains other channels. Preprocessed.zip, CORF3D.zip and Temperature.zip are not required by this RGB experiment and are not mirrored.

## SideViewCows2026

- Creator: Sebastian Möller, Hochschule Osnabrück; ORCID 0009-0008-5364-6506.
- Title: SideViewCows2026 – Dairy Cow Re-Identification Dataset.
- Dataset/version identifier: https://doi.org/10.5281/zenodo.21605650 .
- Authoritative metadata: https://zenodo.org/api/records/21605650 .
- License: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/ .
- Original file mirrored: `snapshots.zip`, renamed `sideview-snapshots.zip` for the release; archive bytes and contained images/masks are unchanged.
- The repository's scripts create derived crops, masked backgrounds and resized arrays; these are research transformations, not original photographs. Retain this attribution when redistributing them.
- Barn/parlor images are not used by this archived experiment and are not mirrored.

## Models and code

The repository's MIT license applies to repository-authored software, not to a replacement license on source datasets. The 45 supplied model checkpoints are outputs of this research and are provided with the repository materials. ImageNet initialization is obtained separately from the official PyTorch distribution and checksum-verified; it is not re-licensed as an original dataset of this project.

Publisher checksums, mirrored file SHA256 values and asset sizes are recorded in `protocols/release_assets.json`. A GitHub release is not an invented permanent archival DOI.
