# 从全新克隆开始复现实验

[English](END_TO_END.md) | **简体中文**

本流程提供实际使用的数据包、45个冻结权重、精确任务定义、历史分数矩阵及独立运行的计算代码。仓库为私有时，复现者必须具有访问权限。可运行不等于已经证明不同硬件上的重训逐位一致，也不意味着新增独立外部确认实验。

## 1. 环境与附件

采用Python 3.11.15、PyTorch 2.7.1+cu126、torchvision 0.22.1+cu126、NumPy 2.2.6、Pillow 11.3.0。按README安装CUDA对应版本后执行：

```bash
python -m pip install -r requirements-inference.txt
# 单独安装GitHub官方CLI，并按正常流程登录：
gh auth login
python scripts/fetch_assets.py --output data/assets --extract
python scripts/fetch_initialization.py --output data/assets/densenet121-a639ec97.pth
```

也可登录GitHub后，在`reproducibility-v1.2.0`的Release页面下载全部附件到`data/assets`，再执行：

```bash
python scripts/fetch_assets.py --output data/assets --verify-only --extract
```

脚本不处理或保存令牌，调用GitHub官方CLI并核验大小及SHA256，不自动覆盖损坏文件。私有仓库的附件不能匿名下载。ImageNet初始化另从PyTorch官方地址获取，核验完整SHA256，不将它混同于45个已训练模型。

完成后目录为：

```text
data/assets/
  holstein-Raw.zip
  holstein-timestamp.xlsx
  sideview-snapshots.zip
  cattle-reid-checkpoints-v1.zip
  holstein-reference-scores.zip
  checkpoints/                         # 45个原始state dict
  holstein-reference-scores/            # 45份逐任务分数矩阵
  densenet121-a639ec97.pth               # ImageNet初始化
```

压缩附件合计约3.3 GB。下载、解压权重、重建输入和新评分建议至少预留10 GB，这是规划余量，不是实测训练磁盘峰值。45次训练还需要额外空间。新训练入口保留选中权重、最后优化器/RNG状态和验证数组，不保存历史流程的全部31个完整检查点。云端禁止写`/app`，结果应放持久盘，临时缓存放`/localdisk-tmp`。

## 2. 不运行模型也能核对全部主表

```bash
python scripts/reproduce_results.py
python -m unittest discover -s tests
python scripts/holstein.py verify-protocol
python scripts/holstein.py replay --scores data/assets/holstein-reference-scores --output derived/holstein-reference-replay
```

完整测试需要PyTorch/torchvision，第一条历史计数复算命令仍只需Python标准库。协议核验重建450个训练epoch和5份验证任务，对比原始哈希。矩阵重放从45份原始分数恢复Holstein结果：

| 方法 | Rank-1 (%) | Rank-5 (%) |
| --- | ---: | ---: |
| B | 83.6676095217 | 98.7769693904 |
| GAP | 74.1038583543 | 96.7412518589 |
| SupCon-in | 65.5235049605 | 90.7228432982 |

主表是单模型任务微平均，然后15模型等权平均，不应替换为身份/日期宏平均。

## 3. 从原图重建两个数据集的输入

```bash
python scripts/holstein.py prepare --raw-zip data/assets/holstein-Raw.zip --output derived/holstein-input
python scripts/prepare_sideview.py --zip data/assets/sideview-snapshots.zip --variant neutral128 --output derived/inputs/neutral128
```

Holstein核验原图文件哈希、每张处理图的像素哈希及最终train/test数组完整字节哈希。SideView沿用原607张图清单，四种版本的历史数组SHA256在`protocols/preprocessing_verification.json`。其他版本将`neutral128`替换为`baseline`、`geomalign_v3`或`blacktrim_v1`，同时更换输出目录。不能用官方Preprocessed.zip直接替代本实验的短边256/中心224处理。

## 4. 冻结原模型重新评分

```bash
python scripts/holstein.py score --input derived/holstein-input --weights data/assets/checkpoints --output derived/holstein-new-scores
python scripts/score_sideview.py --input derived/inputs/neutral128 --weights data/assets/checkpoints --output derived/sideview-neutral128-new-scores
```

两条命令均不训练。Holstein严格使用单图编码和历史CPU逐对评分；SideView使用64批大小与自身的均值余弦实现。不能因为都使用L2归一化，就任意统一浮点求和顺序。新结果写新目录，不覆盖论文历史结果。

Holstein输出`summary.json`，与矩阵重放结果核对。SideView输出`results.json`，按`(method,fold,seed,draw,k)`与`results/sideview/neutral128.json`比较，不比较整个JSON的文件字节，因为运行元数据和序列化可以不同。接近并列的分数对硬件数值差异敏感。

Holstein可增加`--method B --fold 0 --seed 17`做单模型集成检查；不加筛选则跑全部45模型。单模型通过不等于全部模型都已重新验证。

## 5. 从ImageNet初始化重新训练

先进行不训练的预检：

```bash
python scripts/holstein.py train --input derived/holstein-input --init-weight data/assets/densenet121-a639ec97.pth --method B --fold 0 --seed 17 --output derived/training/B-fold0-seed17 --check-only
```

去掉`--check-only`才会启动完整30epoch训练，并在epoch0到30按原验证标准选模。Linux/Bash下完整45模型命令：

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

每个输出目录必须新建。入口暂不支持中断训练的自动续跑；请保留失败现场，检查后改用新目录重启，不会把半完成任务当作成功。每次训练输出`selected.pt`、`last-state.pt`、`history.json`、逐epoch验证分数和`completion.json`。最后状态含优化器和RNG，但保存了状态不等于实现了自动精确恢复。

三种损失、DenseRGB和哈希任务抽样使用保留的原科学函数，来源见`protocols/scientific_function_sources.json`。历史design含其他adapter字段，本入口只消费实际样本池、fold及任务字段；DenseNet实际配置由当前训练代码和`protocols/training_config.json`明确限定。历史云端许可代码未被改写，新入口是用户主动运行的独立复现流程，不伪造旧许可或旧运行回执。

## 6. 验证范围

详见[复现状态](REPRODUCTION_STATUS.md)。已核验原图重建、全部任务重建、45份历史矩阵重放、权重哈希及损失/梯度回归。本次仓库整理没有再次完成45模型全量重训，不能把提供了运行命令写成已经做过完整重复实验。单模型真实推理检查仅适用于所注明的模型和环境。

再分发请保留[数据署名与许可](../DATA_LICENSES.md)。SideView仍是探索性图像级外部迁移结果，不因材料补齐而升级为独立确认性实验。
