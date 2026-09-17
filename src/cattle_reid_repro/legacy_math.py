"""历史科学函数原文提取；仅移除对云端运维模块的导入依赖。"""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import math
import numpy as np
import torch
from .evaluation_artifacts import EvaluationError, canonical_bytes, signed, validate_signed

def require(condition, message):
    if not condition:
        raise EvaluationError(message)

def ordered(values, *namespace):
    """独立命名空间的哈希排序，避免输入行顺序和随机数调用次数改变划分。"""
    return sorted(values, key=lambda value: (
        hashlib.sha256(canonical_bytes([*namespace, value])).hexdigest(), value))

def _pool(design, identity, role, namespace, count):
    keys = [row['sample_key'] for row in design['sample_pools'][identity][role]]
    if len(keys) != len(set(keys)) or len(keys) < count:
        raise EvaluationError('invalid sample pool')
    return ordered(keys, *namespace)[:count]

def training_epoch(design, fold, seed, epoch):
    """每身份一次有效 query，尾部补足不同支持身份，所有 query 都记录掩码。"""
    validate_signed(design)
    if seed not in design['config']['seeds'] or not 1 <= epoch <= 30 or not 0 <= fold < len(design['folds']):
        raise EvaluationError('invalid training seed or epoch')
    spec = design['folds'][fold]
    identities = ordered(spec['train'], seed, 'train-identities', fold, epoch)
    tasks = []
    for update, start in enumerate(range(0, len(identities), 10)):
        active = identities[start:start + 10]
        fillers = ordered(set(identities) - set(active), seed, 'tail-fill', fold, epoch)[:10 - len(active)]
        roster = ordered(active + fillers, seed, 'train-roster', fold, epoch, update)
        task = {'update': update, 'roster': roster,
                'active': [identity in active for identity in roster], 'loss_denominator': 10,
                'query_keys': [_pool(design, i, 'query',
                    [seed, 'train-query', fold, epoch, update, i], 1)[0] for i in roster],
                'support_keys': [_pool(design, i, 'support',
                    [seed, 'train-support', fold, epoch, update, i], 5) for i in roster]}
        if len(roster) != 10 or len(set(roster)) != 10:
            raise EvaluationError('ten distinct identities required')
        if set(task['query_keys']) & {k for group in task['support_keys'] for k in group}:
            raise EvaluationError('query/support overlap')
        tasks.append(task)
    counts = Counter(i for t in tasks for i, active in zip(t['roster'], t['active']) if active)
    if counts != Counter({i: 1 for i in spec['train']}):
        raise EvaluationError('active query coverage mismatch')
    return signed({'artifact_kind': 'controlled-adapter-train-epoch-v1',
        'design_sha256': design['canonical_sha256'], 'fold': fold, 'seed': seed, 'epoch': epoch,
        'tasks': tasks, 'active_queries': len(identities), 'physical_samples': len(tasks) * 60,
        'actual_consumption_verified': False})

def validation_tasks(design, fold):
    """验证任务不包含训练 seed 或 epoch，候选仅来自本折验证身份。"""
    validate_signed(design)
    if not 0 <= fold < len(design['folds']):
        raise EvaluationError('invalid validation fold')
    identities = design['folds'][fold]['validation']
    fixed_seed = design['config']['split']['seed']
    tasks = []
    for identity in sorted(identities):
        for draw in range(5):
            negatives = ordered(set(identities) - {identity}, fixed_seed,
                                'val-negatives', fold, identity, draw)[:9]
            roster = ordered([identity, *negatives], fixed_seed, 'val-roster', fold, identity, draw)
            supports = [_pool(design, i, 'support',
                [fixed_seed, 'val-support', fold, identity, draw, i], 5) for i in roster]
            for row in sorted(design['sample_pools'][identity]['query'], key=lambda r: r['sample_key']):
                tasks.append({'query_identity': identity, 'query_key': row['sample_key'],
                    'draw': draw, 'roster': roster, 'support_keys': supports})
    if len(tasks) != design['folds'][fold]['validation_tasks']:
        raise EvaluationError('validation task count mismatch')
    return signed({'artifact_kind': 'controlled-adapter-validation-tasks-v1',
                   'design_sha256': design['canonical_sha256'], 'fold': fold, 'tasks': tasks})

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

def validation_summary(plan, scores):
    """从完整分数重算身份宏平均，保留原候选次序和精确最早并列规则。"""
    validate_signed(plan)
    tasks = plan['tasks']
    if scores.shape != (len(tasks), 10) or scores.dtype != np.float32 or not np.isfinite(scores).all():
        raise EvaluationError('invalid validation scores')
    correct = [bool(int(pred) == t['roster'].index(t['query_identity']))
               for pred, t in zip(scores.argmax(-1), tasks)]
    metric = macro_rank1([t['query_identity'] for t in tasks], correct)
    return {'numerator': metric.numerator, 'denominator': metric.denominator,
            'tasks': len(tasks), 'correct': correct}

def score_tasks(embeddings, tasks, numpy_reduction=False):
    """逐图对余弦再均值，不把支持原型重新归一化；兼容10及完整候选集合。"""
    require(bool(tasks), 'empty tasks')
    keys = sorted(embeddings)
    positions = {key: i for i, key in enumerate(keys)}
    z = torch.stack([embeddings[key] for key in keys])
    result = np.empty((len(tasks), len(tasks[0]['roster'])), dtype=np.float32)
    # 将相同query和support组的计算复用，保持逐对点积的求和顺序。
    pair_cache = {}
    for n, task in enumerate(tasks):
        require(len(task['roster']) == result.shape[1], 'mixed candidate dimensions')
        query = task['query_key']
        if query not in pair_cache:
            pair_cache[query] = (np.sum(z.numpy() * embeddings[query].numpy(), axis=-1, dtype=np.float32)
                                 if numpy_reduction else (z * embeddings[query]).sum(-1))
        indices = torch.tensor([[positions[key] for key in group] for group in task['support_keys']])
        result[n] = (np.mean(pair_cache[query][indices.numpy()], axis=-1, dtype=np.float32)
                     if numpy_reduction else pair_cache[query][indices].mean(-1).detach().cpu().numpy())
    require(np.isfinite(result).all(), 'nonfinite scores')
    return result

def rank_metrics(scores, tasks):
    """先日期内任务等权，再有效日期等权；稳定排序沿用原候选并列规则。"""
    require(scores.shape == (len(tasks), len(tasks[0]['roster'])) and np.isfinite(scores).all(),
            'invalid score grid')
    ranks = np.argsort(-scores, axis=-1, kind='stable')
    groups = defaultdict(list)
    for index, task in enumerate(tasks):
        target = task['roster'].index(task['query_identity'])
        rank = int(np.flatnonzero(ranks[index] == target)[0]) + 1
        groups[task['fold'], task['query_identity'], task['k'], task['query_day']].append((rank == 1, rank <= 5))
    by_identity = defaultdict(list)
    for (fold, identity, k, day), hits in sorted(groups.items()):
        by_identity[fold, identity, k].append(np.mean(hits, axis=0))
    return [dict(fold=f, identity=i, k=k, rank1=float(np.mean(values, axis=0)[0]),
                 rank5=float(np.mean(values, axis=0)[1])) for (f, i, k), values in sorted(by_identity.items())]

