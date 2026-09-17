# 数据集与权重 / Datasets and Checkpoints

[中文总说明](../README.zh-CN.md) | [English README](../README.md)

## 数据在哪里？

本目录是数据入口。大文件存放在**同一私有仓库的 [v1.2.0 Release 附件](https://github.com/ziluo8080/cattle-reid/releases/tag/reproducibility-v1.2.0)**，不是遗漏上传。克隆Git代码不会自动下载附件。下载后，数据保存在本地 `data/assets/`；该目录内容被Git忽略，只有本说明文件进入版本控制。

| Release附件 | 内容 |
| --- | --- |
| `holstein-Raw.zip` | 官方Holstein原始包；本实验使用其中的`Raw/RGB (640 x 480)` |
| `holstein-timestamp.xlsx` | Holstein时间戳元数据 |
| `sideview-snapshots.zip` | 本实验使用的SideView snapshots原图及掩码，不包含未使用的barn/parlor子集 |
| `cattle-reid-checkpoints-v1.zip` | 45个原始冻结模型权重 |
| `holstein-reference-scores.zip` | 45份Holstein历史逐任务评分矩阵 |

附件合计约3.31 GB。数据来源、署名和再分发条件见[数据许可](../DATA_LICENSES.md)，大小与SHA256见[附件清单](../protocols/release_assets.json)。

## 下载与校验

在仓库根目录执行，先安装GitHub官方CLI并正常登录。私有仓库需要访问权限，不要把令牌写进文件或命令。

```bash
gh auth login
python scripts/fetch_assets.py --output data/assets --extract
```

也可在已登录的浏览器中下载Release全部附件到`data/assets/`，再离线校验并解压权重和评分矩阵：

```bash
python scripts/fetch_assets.py --output data/assets --verify-only --extract
```

完成后目录为：

```text
data/
├── README.md
└── assets/
    ├── holstein-Raw.zip
    ├── holstein-timestamp.xlsx
    ├── sideview-snapshots.zip
    ├── cattle-reid-checkpoints-v1.zip
    ├── holstein-reference-scores.zip
    ├── checkpoints/
    └── holstein-reference-scores/
```

原始图像ZIP不需要手动解压，预处理脚本直接读取。处理后的数组默认写入`derived/`，具体命令见[完整中文复现流程](../docs/END_TO_END.zh-CN.md)。ImageNet初始化不包含在上述五附件中，仅重新训练时通过`scripts/fetch_initialization.py`另行从官方源下载并校验。

## English

This directory is the visible data entry point. Large files are hosted as assets of this repository's [v1.2.0 Release](https://github.com/ziluo8080/cattle-reid/releases/tag/reproducibility-v1.2.0), not in ordinary Git history. Cloning the repository does not download these assets. While the repository is private, readers need repository access.

The five attachments contain the original Holstein Raw archive and timestamps, the SideView snapshots images/masks used by this experiment, 45 frozen checkpoints, and 45 Holstein reference score matrices. Unused SideView barn/parlor subsets are not mirrored. Download into `data/assets/` using the commands above, or download manually and use `--verify-only --extract`. The tool verifies sizes and SHA256, and extracts only checkpoint/reference-score archives; preprocessing reads image ZIPs directly.

Only `data/README.md` is tracked. Downloaded assets remain ignored to avoid committing multi-gigabyte binaries. See the [end-to-end guide](../docs/END_TO_END.md), [license/attribution record](../DATA_LICENSES.md) and [asset checksums](../protocols/release_assets.json). Official ImageNet initialization is a separate download for retraining.
