# Holstein 与 SideView 牛只重识别实验

[English](README.md) | **简体中文**

本仓库整理 Holstein 完整候选识别及 SideViewCows2026 冻结模型迁移实验的代码、协议和历史结果。它是科研复现材料库，不是开箱即用的识别产品。目前仓库为 **private**，可见性由所有者自行管理。

**新增：实际使用的数据包和全部45个冻结权重已上传至[复现附件Release](https://github.com/ziluo8080/cattle-reid/releases/tag/reproducibility-v1.2.0)。请优先阅读[从全新克隆到训练/评分的完整操作说明](docs/END_TO_END.zh-CN.md)，包含下载、校验、输入重建、历史矩阵重放、冻结评分及重新训练命令。**

## 内容导航

[项目结构](#项目结构) | [快速开始](#快速开始) | [复现范围](#1-先看复现范围) | [环境安装](#2-算力和软件版本) | [表格与图片](#3-离线复算表格和重画论文图) | [数据集](#4-数据获取与目录) | [预处理](#5-图像预处理) | [评分与训练](#6-使用冻结权重评分) | [实验协议](#7-实验协议与训练参数) | [对照结果](#8-对照结果) | [科学限制](#9-科学限制) | [常见问题](#10-常见问题) | [引用与贡献](#11-引用与贡献)

## 项目结构

下列是已经提交到Git的实际结构，为便于阅读将同类文件合并展示，`*`代表一组文件。下载附件及运行后生成的目录在下一棵目录树中单独说明。所有命令均从仓库根目录执行，不要先进入`scripts/`。

```text
cattle-reid/
|-- README.md / README.zh-CN.md       # 英文/中文总入口，可相互切换
|-- LICENSE                          # 代码许可
|-- DATA_SOURCES.md                   # 原发布者、数据版本与实际使用范围
|-- DATA_LICENSES.md                  # 数据许可及署名要求
|-- REPRODUCIBILITY.md                # 证据与分析约定
|-- MANIFEST.json                     # 已归档文件的SHA256校验清单
|-- requirements.txt                 # 结果分析与绘图依赖
|-- requirements-inference.txt       # 固定NumPy/Pillow；torch另行安装
|-- data/
|   `-- README.md                    # 附件下载、校验与本地存放说明
|-- docs/
|   |-- END_TO_END.md                 # 英文完整复现流程
|   |-- END_TO_END.zh-CN.md           # 中文完整复现流程
|   `-- REPRODUCTION_STATUS.md        # 实测通过项与未完成的验证边界
|-- scripts/                         # 当前支持的命令行入口
|   |-- fetch_assets.py              # 下载、校验、解压Release附件
|   |-- fetch_initialization.py      # 下载原始ImageNet初始化权重
|   |-- holstein.py                  # prepare/verify-protocol/replay/train/score
|   |-- prepare_sideview.py           # 重建四种SideView输入
|   |-- score_sideview.py             # 固定任务的冻结模型评分
|   |-- reproduce_results.py          # 历史计数、表格及哈希复算
|   |-- identity_sensitivity.py       # 条件配对身份重抽样
|   `-- plot_paper_figures.py         # 重画六幅论文图
|-- src/cattle_reid_repro/             # 运行入口实际调用的科研实现
|   |-- densenet_pairwise.py          # 成对目标计算
|   |-- strong_rgb_densenet.py        # RGB编码器
|   |-- supcon_in.py                  # SupCon-in目标
|   |-- legacy_math.py               # 保留的数值辅助函数
|   `-- evaluation_artifacts.py      # 加载与校验评估工件
|-- protocols/                       # 固定复现输入，不存新运行结果
|   |-- holstein/                    # design、契约、图像索引与任务文件
|   |-- holstein_manifest.csv        # 原始RGB路径与哈希
|   |-- holstein_reference_scores.json
|   |-- sideview_manifest.jsonl      # 固定顺序的607张原图清单
|   |-- sideview_protocol.json       # 身份资格、支持图和查询图索引
|   |-- models.json                  # 45权重名称、哈希及selected epoch
|   |-- training_config.json         # 实际历史训练配置
|   |-- environment.json             # 软件与硬件记录
|   |-- release_assets.json          # 附件名称、字节数与SHA256
|   |-- result_provenance.json        # 历史结果与真实预处理版本的对应关系
|   |-- sensitivity_plan.json        # 固定统计分析设置
|   `-- *verification.json / scientific_function_sources.json
|-- results/sideview/                 # 原样保留的历史结果计数
|   `-- baseline.json / neutral128.json / geomalign_v3.json / blacktrim_v1.json
|-- tables/                          # 可机器读取的证据与论文表格
|   |-- holstein_main_reported.csv / holstein_per_model.csv
|   |-- holstein_candidate_k.csv / holstein_date_k.csv
|   |-- holstein_cohort_383.csv / holstein_actual_split_1620.csv
|   |-- holstein_fold_counts.csv / model_lineage_45.csv
|   |-- sideview_verified.csv / sideview_all_available.csv
|   `-- sideview_identity_sensitivity.csv / training_consumption_45.csv
|-- figures/                         # 六组归档论文图，每组SVG与PDF各一份
|-- reference_pipeline/              # 历史预处理/评分源码证据
|-- reference_training/              # 历史训练源码证据
`-- tests/                           # 四个unittest模块，含协议核验
```

下载附件并执行相应命令后，**本地生成且不进入Git**的目录如下：

```text
data/assets/
|-- holstein-Raw.zip                  # Holstein原始包，预处理直接读取
|-- holstein-timestamp.xlsx           # 原始时间戳元数据
|-- sideview-snapshots.zip            # 本实验图像与官方掩码
|-- cattle-reid-checkpoints-v1.zip
|-- holstein-reference-scores.zip
|-- checkpoints/                     # 45个历史训练后的state dict
|-- holstein-reference-scores/        # 45份历史NPY评分矩阵
`-- densenet121-a639ec97.pth           # 单独下载，仅重训需要的初始化
derived/
|-- holstein-input/                  # 重建的Holstein数组及回执
|-- inputs/neutral128/               # SideView uint8数组及处理回执
|-- holstein-reference-replay/        # 根据历史矩阵复算的汇总
|-- holstein-new-scores/              # 新Holstein冻结推理结果
|-- sideview-neutral128-new-scores/   # 新SideView冻结推理结果
|-- training/                        # 新训练任务，不是历史权重
|-- figures/                         # 重画的SVG/PDF及PNG预览
`-- identity_sensitivity.csv         # 重新计算的敏感性分析
```

因此，Git中有`data/`入口，大文件本体通过Release分发，克隆代码不等于下载数据。不要将新预测移入`results/`，不要改变固定图像顺序，也不要把历史参考源码当作可移植运行入口。

## 快速开始

按目标选择路径，不同路径代表不同程度的复现证据：

| 目标 | 操作顺序 | GPU/大文件需求 |
| --- | --- | --- |
| 先确认已报告指标 | 克隆后运行`python scripts/reproduce_results.py` | 无需GPU或图像包，只需标准Python |
| 重画论文图与复算统计 | 安装分析依赖，执行第3节 | 无需GPU、无需原图包 |
| 重放Holstein历史预测 | 安装推理依赖、下载附件、运行`holstein.py replay` | 不做GPU推理，需要参考矩阵 |
| 从原图重新评分 | 安装CUDA环境、下载、预处理、评分 | Holstein建议GPU；SideView入口要求GPU |
| 从初始化重新训练 | 完成输入准备、下载初始化、训练并重评分 | 完整研究包含45个训练任务，见第6节 |

克隆并进入仓库后，最小检查命令为：

```bash
python scripts/reproduce_results.py
```

完整流程应先按第2节安装环境，再正常登录GitHub CLI并获取附件：

```bash
gh auth login
python scripts/fetch_assets.py --output data/assets --extract
python scripts/holstein.py verify-protocol
python scripts/holstein.py replay --scores data/assets/holstein-reference-scores --output derived/holstein-reference-replay
```

下载、解压权重、输入数组和评分输出建议预留至少10 GB工作空间，重训还需额外空间。这是容量规划建议，不是实测峰值。逐步操作说明见[END_TO_END.zh-CN.md](docs/END_TO_END.zh-CN.md)。

## 1. 先看复现范围

### 数据集在哪里？

请进入 **[data/README.md 数据入口](data/README.md)**。原始数据与权重存放在本仓库的 **[Release附件](https://github.com/ziluo8080/cattle-reid/releases/tag/reproducibility-v1.2.0)** 中，包括Holstein原始包/时间戳及本实验使用的SideView snapshots图像/掩码，不逐个写入Git代码目录。仅克隆仓库不会自动下载；正常登录GitHub官方CLI后，在仓库根目录执行：

```bash
python scripts/fetch_assets.py --output data/assets --extract
```

数据会下载到`data/assets/`，权重解压到`data/assets/checkpoints/`，参考分数解压到`data/assets/holstein-reference-scores/`。五个附件合计约3.31 GB。仓库只跟踪数据说明文件，下载的大文件仍被Git忽略。私有仓库需要访问权限；手动下载及离线校验方法见数据入口说明。

| 复现层级 | 已提供 | 还需要什么 |
| --- | --- | --- |
| 历史结果核验 | SideView四种预处理计数、Holstein逐模型汇总、文件哈希与复算脚本 | Python 3.11以上，普通CPU即可 |
| 敏感性分析与论文图 | 固定分析方案、统计脚本、六幅矢量图的绘图源码与数据 | NumPy、绘图依赖、中文字体 |
| SideView原图预处理 | 四种处理的可移植命令行入口、607张原图清单 | 官方`snapshots.zip`；灰背景和geomalign需要其中的掩码 |
| SideView冻结模型评分 | 推理入口、固定索引、Release中的45个原始权重 | 下载并解压附件；CUDA环境 |
| Holstein重训与重评分 | 独立运行入口、实际任务、初始化下载器、数据、权重及原始评分矩阵 | 按完整操作说明执行；本次未再次运行45模型全量重训 |

“指标可以复算”不等于“完整重复实验已被验证”。新包装的可移植入口不冒充历史原始运行记录。本次对B/fold0/seed17进行了冻结推理检查，其Rank-1/Rank-5与历史结果一致，未替换历史结果，也未运行45模型全量重训。

## 2. 算力和软件版本

### 已有记录确认的实验环境

| 项目 | 版本或配置 |
| --- | --- |
| 操作平台 | Linux云端环境 |
| GPU | 单张NVIDIA GeForce RTX 4090 |
| Python | 3.11.15 |
| PyTorch | 2.7.1+cu126 |
| torchvision | 0.22.1+cu126 |
| NumPy | 2.2.6 |
| Pillow | 11.3.0 |
| 训练精度 | FP32；不使用AMP；关闭TF32 |
| 冻结推理批大小 | Holstein：1；SideView：64 |

CPU型号、主机内存、驱动版本、Linux发行版、峰值显存和完整研究的GPU小时数尚未形成完整可核验记录，不能猜测填写。`cu126`表示PyTorch构建版本，不等于已经证明独立安装的CUDA toolkit也是相同版本。

训练预算：每个模型630次更新，每次60张图像输入，每种方法15个模型，共三种方法，即 **28,350次更新、1,701,000次训练图像呈现**。不包括验证、检查点核验及重放；不能将其理解成独立图像数量或实测GPU小时数。逐模型记录见[训练消费表](tables/training_consumption_45.csv)。

SideView每种预处理包含607张图像，每个模型只提取一遍embedding。45个模型合计27,315次图像前向，五个draw和三个K复用这些embedding。四种处理各跑一次对应109,260次图像前向，不包括历史重试与重放。单份uint8缓存约91.4 MB。45个权重解压后约1.28 GB，已作为Release附件提供，不写入普通Git历史。

### 安装环境

私有仓库的克隆需要GitHub访问权限，使用正常的Git认证，不要把令牌写进命令、配置或README。

```bash
git clone https://github.com/ziluo8080/cattle-reid.git
cd cattle-reid
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell请改用：
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

只做预处理时，安装`requirements-inference.txt`中的NumPy和Pillow即可。需要GPU推理时，建议单独建立Python 3.11.15环境：

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-inference.txt
python -c "import torch, torchvision, numpy, PIL; print(torch.__version__, torchvision.__version__, numpy.__version__, PIL.__version__); print(torch.cuda.is_available())"
```

上述表格记录的是历史实验版本；`requirements.txt`中的绘图库版本范围仅用于结果工具，不冒充完整训练环境锁文件。中文图件需要Microsoft YaHei或绘图配置支持的中文字体，否则可能出现方框或缺字。

## 3. 离线复算表格和重画论文图

在仓库根目录执行（完整测试现需安装PyTorch环境）：

```bash
python scripts/reproduce_results.py
python -m unittest discover -s tests
python scripts/identity_sensitivity.py
python scripts/plot_paper_figures.py
```

- 第一条命令仅使用Python标准库：核验`MANIFEST.json`、2700条SideView记录、145800条身份correct/total计数、36组micro/macro结果，以及基于逐模型汇总的6个Holstein主表数值。它不从原图重新生成Holstein预测。
- 敏感性输出为`derived/identity_sensitivity.csv`。使用NumPy PCG64、种子20260917、10000次配对身份簇重抽样及百分位区间。图库、模型和支持抽样固定，属于事后条件敏感性分析，不是对新牧场泛化能力的确认性置信区间。
- 图件输出到`derived/figures/`。SVG/PDF用于论文；PNG只用于目视检查。原归档图位于`figures/`，不覆盖。
- `derived/`不进入版本控制。新评分不得覆盖历史`results/`中的文件。

## 4. 数据获取与目录

可以从本仓库Release镜像或原发布者获取，遵守[数据署名与许可](DATA_LICENSES.md)和[来源说明](DATA_SOURCES.md)。镜像包均与发布者提供的校验值核对一致。

- Holstein：*Recognition of Holstein Cattle with Thermal and RGB images*，DOI `10.34894/7M108F`。本稿使用RGB，不是热红外/RGB融合实验。
- SideViewCows2026：*SideViewCows2026 - Dairy Cow Re-Identification Dataset*，版本DOI `10.5281/zenodo.21605650`。本归档实验仅用`snapshots.zip`，不要换成另一个parlor到snapshots的新协议。

SideView压缩包内部应包含：

```text
snapshots/images/<identity>/<image>.jpg
snapshots/masks/<identity>/<image>.png
```

`protocols/sideview_manifest.jsonl`将每个数组索引绑定到图像文件名、身份和原始文件SHA256。评分协议保存的是数组索引，**不能重新排序清单**。`protocols/holstein_manifest.csv`包含Holstein原始RGB路径和哈希；`tables/holstein_actual_split_1620.csv`记录实际fold划分。原图和掩码按许可保留在Release原始压缩包中，不逐图写入Git历史。

## 5. 图像预处理

### Holstein模型输入

转RGB；短边双线性缩放至256；中心裁剪224×224；像素缩放至[0,1]；使用ImageNet均值`[0.485,0.456,0.406]`和标准差`[0.229,0.224,0.225]`归一化。已核验训练配置没有随机增强。复现Holstein训练时不能替换为SideView的letterbox输入。

### SideView模型输入

所有版本保留同一607张图像清单，输出`uint8 [607,3,224,224]`。先处理EXIF方向，再转换RGB。**不得再做中心裁剪**；ImageNet归一化在推理入口中只执行一次。

| 参数值 | 精确处理规则 |
| --- | --- |
| `baseline` | 完整图像等比例LANCZOS缩放到224画布内，居中填充RGB(128,128,128)，不裁剪 |
| `neutral128` | 官方掩码>0视为前景，掩码外像素替换为RGB(128,128,128)，然后采用baseline的letterbox；不裁外接框 |
| `geomalign_v3` | 掩码外接框外扩max(8像素,round(最长框边×15%))并截断至图像范围，保留原背景；裁剪图最长边缩至200，放入224灰画布 |
| `blacktrim_v1` | 仅裁连续黑边：行/列中至少95%像素的RGB最大通道≤8，沿用历史每边40%循环限制；然后letterbox |

```bash
python scripts/prepare_sideview.py --zip /path/to/snapshots.zip --variant baseline --output derived/inputs/baseline
python scripts/prepare_sideview.py --zip /path/to/snapshots.zip --variant neutral128 --output derived/inputs/neutral128
python scripts/prepare_sideview.py --zip /path/to/snapshots.zip --variant geomalign_v3 --output derived/inputs/geomalign_v3
python scripts/prepare_sideview.py --zip /path/to/snapshots.zip --variant blacktrim_v1 --output derived/inputs/blacktrim_v1
```

将`/path/to/snapshots.zip`替换为实际文件路径。输出目录必须尚不存在。脚本核验607个原图SHA256，输出`images_uint8.npy`和`processing_receipt.json`。可移植入口没有单独冻结每张mask的SHA256，因此必须使用同一官方ZIP版本。该入口与已有本地处理数组核对，不用于重新选择预处理策略。

## 6. 使用冻结权重评分

三方法使用DenseNet-121卷积特征、空间均值池化和1024维L2归一化embedding，但训练权重不同。推理时使用BatchNorm已保存的统计量，不包含分类器。

下载并解压Release中的`cattle-reid-checkpoints-v1.zip`，其中45个**原始state-dict权重**已经按照`protocols/models.json`命名，例如`B-fold0-seed17.pt`、`GAP-fold0-seed17.pt`、`SupCon-in-fold0-seed17.pt`。清单包含SHA256和selected epoch。不允许换成ImageNet初始化或猜测格式；入口严格校验哈希并使用`strict=True`加载。单独克隆Git代码不会自动下载附件，请先执行附件下载步骤。

```bash
python scripts/score_sideview.py --input derived/inputs/neutral128 --weights /path/to/checkpoints --output derived/inference/neutral128 --check-only
# 配齐45个原始权重及CUDA环境后，才执行真实推理：
python scripts/score_sideview.py --input derived/inputs/neutral128 --weights /path/to/checkpoints --output derived/inference/neutral128
```

第二条命令是新的模型评分，不是指标复算，不进行训练。输出各模型分数矩阵和合并`results.json`，均写入新目录。本次材料整理未执行这个45模型CUDA评分任务，不声称已经证明跨硬件/软件的逐位一致。

**评分定义：**每张图的embedding先做L2归一化；对同一身份的K张支持图embedding求均值；与查询embedding点积。**支持均值向量不能再归一化**。此规则等价于查询与每张支持图的余弦相似度平均。候选顺序固定，平分时NumPy argmax选择第一个最大值。

### Holstein：重建输入与重新评分

完成附件解压后，可以直接使用以下路径：

```bash
python scripts/holstein.py prepare --raw-zip data/assets/holstein-Raw.zip --output derived/holstein-input
# 小规模集成检查：仅B/fold0/seed17，不代表完整主表。
python scripts/holstein.py score --input derived/holstein-input --weights data/assets/checkpoints --method B --fold 0 --seed 17 --output derived/holstein-B-fold0-seed17
# 完整冻结评分：3方法 x 5fold x 3seed。
python scripts/holstein.py score --input derived/holstein-input --weights data/assets/checkpoints --output derived/holstein-new-scores
```

全量运行输出`summary.json`，应与历史矩阵重放的汇总比较，不能拿单模型结果代替主表。Holstein使用单图编码和CPU成对评分；SideView使用批量编码及上文的平均余弦规则，两者数值归约流程不能随意互换。

### 重新训练是可选项，不是外部评分的前提

需要复现训练时，先下载精确匹配的ImageNet初始化并做预检查。**不要在SideView上训练**，本项目对它执行冻结模型迁移。

```bash
python scripts/fetch_initialization.py --output data/assets/densenet121-a639ec97.pth
python scripts/holstein.py train --input derived/holstein-input --init-weight data/assets/densenet121-a639ec97.pth --method B --fold 0 --seed 17 --output derived/training/B-fold0-seed17 --check-only
# 去掉 --check-only 才会真正训练该模型。
```

全量训练需要遍历方法`B/GAP/SupCon-in`、fold `0..4`和seed `17/29/43`，输出目录保持`<method>-fold<fold>-seed<seed>`命名。[完整操作说明](docs/END_TO_END.zh-CN.md)提供Bash循环。45个任务全部完成后：

```bash
python scripts/holstein.py score --input derived/holstein-input --trained-root derived/training --output derived/holstein-retrained-scores
python scripts/score_sideview.py --input derived/inputs/neutral128 --trained-root derived/training --output derived/sideview-retrained-scores
```

每个训练任务保存`selected.pt`、`last-state.pt`、`history.json`、各epoch验证矩阵及`completion.json`。原始归档权重使用`--weights`，新训练目录使用`--trained-root`，两者互斥。当前启动器不支持自动断点续训；保存优化器/RNG状态不等于已经实现自动恢复。跨环境重训可能产生差异，本次归档整理没有重新运行全量训练。

## 7. 实验协议与训练参数

Holstein从383个身份审计中保留324个合格身份。必须采用已归档的实际划分，不能重新随机划分。五个fold、种子17/29/43，每方法15个冻结模型。每个任务使用测试fold全部64或65个候选；主指标混合K=1/3/5后对模型等权平均。15,735个任务不等于15,735个独立个体。此前88.00%的B结果属于65候选/K=5子层，不是主表结果。

SideView初始607图/63身份，要求每身份至少6图后保留577图/54身份。每个draw预留5张支持图，K=1/3取K=5的前缀，剩余307张为查询。未使用的支持图不能转成查询。五次固定draw，每种处理形成45模型×5draw×3K=675条结果。

| 训练项 | 固定设置 |
| --- | --- |
| 初始化 | DenseNet-121 IMAGENET1K_V1；哈希见`protocols/training_config.json` |
| 可训练部分 | 全卷积特征和BN仿射参数；固定BN运行统计量 |
| 优化器 | AdamW，学习率1e-4，weight decay 1e-4，betas=(0.9,0.999)，eps=1e-8 |
| 其他设置 | amsgrad/foreach/fused均false；梯度范数裁剪1 |
| 日程 | 30epoch，每epoch21步，共630步；42步线性warm-up后逐步余弦衰减，末步降到0 |
| Episode | 10个身份，10张query+50张support，每身份5支持 |
| 目标配置 | 温度0.1，分母10，支持分支参与梯度；各方法损失以源码为准 |
| 选模 | epoch0到30中验证K=5身份宏Rank-1最高者，同分取最早 |
| 选模边界 | 验证使用10-way任务；选模后的完整验证检查不重新选模 |

不要把B/GAP/SupCon-in当作可互换的损失名。`reference_training/`保留历史证据，`src/cattle_reid_repro/`与`scripts/holstein.py`提供独立运行的实际计算。见[完整运行命令](docs/END_TO_END.zh-CN.md)和[验证边界](docs/REPRODUCTION_STATUS.md)。

## 8. 对照结果

| Holstein方法 | Rank-1 | Rank-5 |
| --- | ---: | ---: |
| B-selected | 83.67% | 98.78% |
| GAP | 74.10% | 96.74% |
| SupCon-in | 65.52% | 90.72% |

| SideView灰背景方法 | K=1 | K=3 | K=5 |
| --- | ---: | ---: | ---: |
| B | 50.10% | 58.35% | 61.82% |
| GAP | 39.31% | 47.16% | 51.12% |
| SupCon-in | 28.42% | 32.86% | 35.21% |

SideView表中数值是查询micro准确率，再对15模型×5draw平均。身份macro准确率单列于`tables/sideview_verified.csv`。不能混淆micro/macro，也不能把不同候选数或K的结果比较成受控提升。

### 结果与论文图导航

| 要查什么 | 归档证据 |
| --- | --- |
| Holstein主表从哪里得到？ | [逐模型值](tables/holstein_per_model.csv)、[主表](tables/holstein_main_reported.csv)及Release评分矩阵 |
| 为什么还出现过88%？ | [候选数/K分层表](tables/holstein_candidate_k.csv)，不能代替混合K主指标 |
| 每个fold有哪些身份和图片？ | [实际划分](tables/holstein_actual_split_1620.csv)、[fold计数](tables/holstein_fold_counts.csv) |
| SideView各预处理结果在哪里？ | [micro/macro核验表](tables/sideview_verified.csv)、[全部可用比较](tables/sideview_all_available.csv)及`results/sideview/` |
| 权重如何对应到训练？ | [模型来源表](tables/model_lineage_45.csv)、[权重清单](protocols/models.json) |
| 哪些步骤实际验证过？ | [复现验证状态](docs/REPRODUCTION_STATUS.md) |

六组SVG/PDF分别对应研究协议、模型评分结构、Holstein主结果、支持数量结果、SideView迁移和预处理比较。[图1](figures/Fig1_study_protocol.svg)与[图2](figures/Fig2_model_scoring.svg)可用于查看实验流程和模型结构。归档论文图使用中文标注，运行说明提供中英文两版。

## 9. 科学限制

历史源码保留当时的本机/云端路径和执行门禁，不要直接替换路径、删除门禁后启动。对于已支持的步骤，使用上面的可移植入口。

灰背景原JSON的一个预处理字段误写为letterbox，原文件保持不变；`protocols/result_provenance.json`依据历史启动证据说明真实版本。baseline/neutral128逐query完整分数及原始运行时输入字节回执仍缺失。今天重建数组不能替代过去缺失的运行回执。

SideView是**已有公开数据集**，不是本研究自行采集。由于根据外部结果比较过多种预处理，而且采集事件独立性未验证，应解释为探索性图像级迁移，不能写成从未使用过的独立确认性外部测试。本研究沿用官方身份标签，不宣称完成了独立人工标注研究。

历史标签`paper-evidence-v1.0.0`保持不变；引用时记录实际使用的commit。代码沿用MIT许可，数据和预训练权重仍遵守其原有条款。不虚构论文DOI或永久归档DOI。仓库可见性仅由所有者调整。

## 10. 常见问题

| 现象 | 检查与处理 |
| --- | --- |
| 克隆后`data/`只有README | 正常；按数据入口下载Release，图像包和权重不进入普通Git历史 |
| 私有Release返回404或权限错误 | 确认当前账号有仓库权限，并正常执行`gh auth login`；不要把token写入文件 |
| 没有安装`gh` | 用已登录的浏览器下载五个附件至`data/assets/`，再运行下方离线校验命令 |
| 文件大小、SHA256或输入哈希不匹配 | 停止评分，核查附件版本、下载完整性及是否改动原图；不要删除校验逻辑 |
| 输出目录已存在 | 指定新的输出目录，避免覆盖历史结果；启动器不自动恢复部分训练任务 |
| `torch.cuda.is_available()`为false | 核对NVIDIA驱动、当前Python环境及CUDA wheel；计数复算不依赖GPU |
| 单模型准确率与主表不同 | 主表平均15个模型；核对method/fold/seed、K、候选数和micro/macro口径 |
| 图中文字显示方框 | 安装绘图配置支持的中文字体，重新运行绘图脚本 |

```bash
python scripts/fetch_assets.py --output data/assets --verify-only --extract
python scripts/holstein.py --help
python scripts/holstein.py train --help
python scripts/score_sideview.py --help
```

离线校验不会下载缺失附件。以上`--help`不执行训练或推理。文档中的命令本身不依赖Linux续行符；完整训练循环是Bash语法，Windows用户可以使用Bash或逐个执行单模型命令。

## 11. 引用与贡献

代码许可见[LICENSE](LICENSE)，数据分别适用[DATA_LICENSES.md](DATA_LICENSES.md)中的CC0/CC BY 4.0条款，不因放进同一仓库就全部改成MIT。引用数据时保留原作者、数据集名称和对应DOI；引用本复现材料时记录仓库地址、实际commit和所用附件版本`reproducibility-v1.2.0`。目前没有可填写的正式论文DOI，不提供虚构BibTeX条目。

```bash
git rev-parse HEAD
```

报告复现问题时，请附上commit、方法/fold/seed/预处理、完整命令、Python/torch/torchvision版本、GPU及错误日志，勿包含凭据或个人访问令牌。修改代码时运行`python -m unittest discover -s tests`；不要修改历史结果以让测试通过。若修改了归档跟踪文件，需明确记录修改并相应更新`MANIFEST.json`，不可将新结果冒充原始证据。

文档组织参考了[FastReID](https://github.com/JDAI-CV/fast-reid)的安装/快速开始/模型导航，以及[Torchreid](https://github.com/KaiyangZhou/deep-person-reid)的环境/训练/跨域评估分层。本项目没有据此引入它们的模型、依赖或实验结论。

---
[返回顶部](#holstein-与-sideview-牛只重识别实验) | [Switch to English](README.md)
