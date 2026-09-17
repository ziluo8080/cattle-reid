"""同 query/support 角色的 SupCon-in 普通目标，不改变身份均值推理。"""
import math
import torch


def supcon_in_loss(embedding, active):
    """log 内平均五个正例概率；尾批保持固定除以十。"""
    if embedding.ndim != 2 or embedding.shape[0] != 60 or embedding.dtype != torch.float32:
        raise ValueError('expected 60 FP32 representations')
    if active.shape != (10,) or active.dtype != torch.bool or not bool(active.any()):
        raise ValueError('invalid active query mask')
    support = embedding[10:].reshape(10, 5, -1)
    similarities = (embedding[:10, None, None] * support[None]).sum(-1)
    z = similarities / .1
    idx = torch.arange(10, device=embedding.device)
    losses = z.flatten(1).logsumexp(1) - z[idx, idx].logsumexp(1) + math.log(5)
    loss = (losses * active).sum() / 10
    if not bool(torch.isfinite(loss)):
        raise ValueError('nonfinite SupCon-in loss')
    return loss, similarities.mean(-1)
