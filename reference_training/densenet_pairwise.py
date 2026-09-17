"""原query/support角色下的样本级多正例对比目标，推理仍用身份均值评分。"""

import torch
from torch.nn import functional as F

from .evaluation_artifacts import EvaluationError


def pairwise_loss(embedding, active):
    """分母含全部50个support、正例外平均；active尾部仍除以固定10。"""
    if embedding.ndim != 2 or embedding.shape[0] != 60 or embedding.dtype != torch.float32:
        raise EvaluationError('expected 60 FP32 representations')
    if active.shape != (10,) or active.dtype != torch.bool or not bool(active.any()):
        raise EvaluationError('invalid active query mask')
    support = embedding[10:].reshape(10, 5, -1)
    similarities = (embedding[:10, None, None] * support[None]).sum(-1)
    log_probability = F.log_softmax(similarities.flatten(1) / .1, dim=-1).reshape(10, 10, 5)
    indices = torch.arange(10, device=embedding.device)
    losses = -log_probability[indices, indices].mean(-1)
    loss = (losses * active).sum() / 10
    if not bool(torch.isfinite(loss)):
        raise EvaluationError('nonfinite DenseNet pairwise objective')
    return loss, similarities.mean(-1)
