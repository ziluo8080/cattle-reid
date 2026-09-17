"""固定 adapter 检索训练计算与本地合成恢复工具；不包含真实载荷授权入口。"""

from collections import defaultdict
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

from .evaluation_artifacts import EvaluationError, digest, publish_once, signed, validate_signed
from .rgb_token_adapter import RGBTokenAdapter


def initialize(seed):
    """合成验收固定 CPU 数值设置并返回共享 epoch 0 的模型及优化器。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    model = RGBTokenAdapter().float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4,
        betas=(0.9, 0.999), eps=1e-8, amsgrad=False, foreach=False, fused=False)
    return model, optimizer


def representation(model, blocks):
    """保留 adapter 的逐槽位归一化，再按固定槽位顺序拼接与整体归一化。"""
    slots = model(blocks)
    norms = torch.linalg.vector_norm(slots, dim=-1)
    if slots.dtype != torch.float32 or not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise EvaluationError('nonfinite or zero slot norm')
    return F.normalize(slots.flatten(1), dim=-1, eps=1e-12)


def pair_scores(query, supports):
    """显式逐对点积后 K 均值，与最终检索规则一致并保留支持分支梯度。"""
    if (query.ndim != 2 or supports.ndim != 3 or supports.shape[0] != 10
            or supports.shape[1] not in (1, 3, 5) or query.shape[-1] != supports.shape[-1]):
        raise EvaluationError('invalid query/support shape')
    return (query[:, None, None, :] * supports[None, :, :, :]).sum(-1).mean(-1)


def episode_loss(embedding, active):
    """输入按十个 query 后五十个支持排列，尾部依然除以十。"""
    if embedding.shape != (60, 1280) or embedding.dtype != torch.float32:
        raise EvaluationError('expected FP32 [60,1280] embeddings')
    if active.dtype != torch.bool or active.shape != (10,) or not bool(active.any()):
        raise EvaluationError('invalid active query mask')
    scores = pair_scores(embedding[:10], embedding[10:].reshape(10, 5, 1280))
    per_query = F.cross_entropy(scores / 0.1, torch.arange(10, device=embedding.device), reduction='none')
    loss = (per_query * active).sum() / 10
    if not bool(torch.isfinite(loss)):
        raise EvaluationError('nonfinite loss')
    return loss, scores


def learning_rate(step, updates_per_epoch):
    """一步一调度，编号从一开始，恢复时由累计更新数定位。"""
    total, warm = 30 * updates_per_epoch, 2 * updates_per_epoch
    if updates_per_epoch < 1 or not 1 <= step <= total:
        raise EvaluationError('scheduler step out of range')
    if step == total:
        return 0.0
    if step <= warm:
        return 1e-4 * step / warm
    return 1e-4 * 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))


def update(model, optimizer, blocks, active, step, updates_per_epoch):
    """统一更新入口：有限损失/梯度检查、全局范数裁剪以及固定 AdamW 更新。"""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, scores = episode_loss(representation(model, blocks), active)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    lr = learning_rate(step, updates_per_epoch)
    for group in optimizer.param_groups:
        group['lr'] = lr
    optimizer.step()
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise EvaluationError('nonfinite updated parameter')
    return {'loss': float(loss.detach()), 'gradient_norm': float(norm), 'lr': lr,
            'scores': scores.detach()}


def macro_rank1(identities, correct):
    """整数计数生成精确有理数宏平均，避免浮点并列误选 checkpoint。"""
    if len(identities) != len(correct) or not identities:
        raise EvaluationError('invalid validation observations')
    counts = defaultdict(lambda: [0, 0])
    for identity, hit in zip(identities, correct):
        if type(hit) is not bool:
            raise EvaluationError('correctness must be boolean')
        counts[identity][0] += int(hit)
        counts[identity][1] += 1
    return sum((Fraction(a, b) for a, b in counts.values()), Fraction()) / len(counts)


def select_epoch(metrics):
    """按 0 开始的完整轮次扫描；精确并列保留早轮，不强迫训练后选模。"""
    if not metrics or any(not isinstance(m, Fraction) or not 0 <= m <= 1 for m in metrics):
        raise EvaluationError('exact epoch metrics required')
    return max(range(len(metrics)), key=lambda epoch: metrics[epoch])


def save_synthetic_checkpoint(path, model, optimizer, step, binding):
    """原子发布合成 checkpoint 与全文件清单；独立保存 RNG，不授予真实恢复许可。"""
    path = Path(path)
    if path.exists() or next(model.parameters()).device.type != 'cpu':
        raise EvaluationError('fresh CPU synthetic checkpoint required')
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_rng = np.random.get_state()
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step,
             'python_rng': random.getstate(), 'numpy_rng': (numpy_rng[0], numpy_rng[1].tolist(),
                numpy_rng[2], numpy_rng[3], numpy_rng[4]), 'torch_rng': torch.get_rng_state()}
    with tempfile.TemporaryDirectory(dir=path.parent, prefix='adapter-checkpoint-') as temporary:
        staged = Path(temporary) / 'published'
        staged.mkdir()
        torch.save(state, staged / 'state.pt')
        manifest = signed({'artifact_kind': 'controlled-adapter-synthetic-checkpoint-v1',
                           'binding': binding, 'step': step,
                           'files': {'state.pt': {'sha256': digest(staged / 'state.pt'),
                                      'bytes': (staged / 'state.pt').stat().st_size}}})
        publish_once(staged / 'manifest.json', manifest)
        os.rename(staged, path)
    return manifest['canonical_sha256']


def restore_synthetic_checkpoint(path, pin, binding):
    """先验证工件种类、来源与字节，再安全加载 CPU 状态并恢复全部 RNG。"""
    path = Path(path)
    if path.is_symlink() or {p.name for p in path.iterdir()} != {'manifest.json', 'state.pt'}:
        raise EvaluationError('unexpected checkpoint files')
    if any((path / name).is_symlink() for name in ('manifest.json', 'state.pt')):
        raise EvaluationError('symlink checkpoint file')
    manifest = validate_signed(json.loads((path / 'manifest.json').read_bytes()))
    if (manifest['canonical_sha256'] != pin or manifest['binding'] != binding
            or manifest['artifact_kind'] != 'controlled-adapter-synthetic-checkpoint-v1'):
        raise EvaluationError('checkpoint binding mismatch')
    entry = manifest['files']['state.pt']
    if digest(path / 'state.pt') != entry['sha256'] or (path / 'state.pt').stat().st_size != entry['bytes']:
        raise EvaluationError('checkpoint bytes changed')
    state = torch.load(path / 'state.pt', map_location='cpu', weights_only=True)
    if state['step'] != manifest['step']:
        raise EvaluationError('checkpoint step mismatch')
    model, optimizer = initialize(0)
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    random.setstate(state['python_rng'])
    name, keys, pos, gaussian, cached = state['numpy_rng']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, gaussian, cached))
    torch.set_rng_state(state['torch_rng'])
    return model, optimizer, state['step']
