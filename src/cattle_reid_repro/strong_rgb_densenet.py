"""固定统计量的 DenseNet RGB 表示及原 episode 目标，不修改 C 的冻结源码。"""

import re

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import densenet121

from .evaluation_artifacts import EvaluationError


class DenseRGB(nn.Module):
    """只使用卷积特征；固定 BN 统计量使每张图表示不依赖同批图像。"""

    def __init__(self, state=None):
        super().__init__()
        network = densenet121(weights=None, drop_rate=0)
        if state is not None:
            # 官方早期权重的 denselayer 键包含点，按 torchvision 官方兼容规则转换。
            pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            translated = {pattern.sub(r'\1\2', key): value for key, value in state.items()}
            network.load_state_dict(translated, strict=True)
        self.features = network.features
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, value):
        value = F.adaptive_avg_pool2d(F.relu(self.features(value)), (1, 1)).flatten(1)
        if not bool(torch.isfinite(value).all()) or not bool((value.norm(dim=-1) > 0).all()):
            raise EvaluationError('invalid DenseNet representation')
        return F.normalize(value, dim=-1, eps=1e-12)


def retrieval_loss(embedding, active):
    """只放宽表示维度；十query、五十support、温度及尾部分母沿用C。"""
    if embedding.ndim != 2 or embedding.shape[0] != 60 or embedding.dtype != torch.float32:
        raise EvaluationError('expected 60 FP32 representations')
    if active.shape != (10,) or active.dtype != torch.bool or not bool(active.any()):
        raise EvaluationError('invalid active query mask')
    support = embedding[10:].reshape(10, 5, -1)
    scores = (embedding[:10, None, None] * support[None]).sum(-1).mean(-1)
    losses = F.cross_entropy(scores / .1, torch.arange(10, device=embedding.device), reduction='none')
    loss = (losses * active).sum() / 10
    if not bool(torch.isfinite(loss)):
        raise EvaluationError('nonfinite DenseNet objective')
    return loss, scores
