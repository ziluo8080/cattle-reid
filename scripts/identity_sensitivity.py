"""固定模型、支持抽样与候选库的事后身份重抽样敏感性分析。"""
import csv
from pathlib import Path
import numpy as np
from reproduce_results import ROOT, load, verify_result, METHODS


def identity_counts(rows, method, k, identities):
    """先平均模型和draw的计数；所有身份成组保留跨方法配对。"""
    selected = [r for r in rows if r['method'] == method and r['k'] == k]
    correct = np.array([[r['per_identity'][i]['correct'] for i in identities] for r in selected], dtype=float)
    total = np.array([[r['per_identity'][i]['total'] for i in identities] for r in selected], dtype=float)
    assert correct.shape == total.shape == (75, 54)
    assert np.all(total == total[0])
    return correct.mean(0), total[0]


def main():
    """不重建候选库、不重训、不估计新牧场性能，不报告P值。"""
    protocol = load(ROOT / 'protocols/sideview_protocol.json')
    identities = protocol['candidate_ids']
    plan = load(ROOT / 'protocols/sensitivity_plan.json')
    rng = np.random.Generator(np.random.PCG64(plan['seed']))
    indices = rng.integers(0, 54, size=(plan['replicates'], 54))
    sources = {v: verify_result(load(ROOT / f'results/sideview/{v}.json'), protocol)
               for v in ('neutral128', 'baseline')}
    output = []
    for k in (1, 3, 5):
        comparisons = [('B_minus_GAP', 'neutral128', 'B', 'neutral128', 'GAP'),
                       ('B_minus_SupCon-in', 'neutral128', 'B', 'neutral128', 'SupCon-in'),
                       ('B_neutral_minus_baseline', 'neutral128', 'B', 'baseline', 'B')]
        for name, va, ma, vb, mb in comparisons:
            ca, na = identity_counts(sources[va], ma, k, identities)
            cb, nb = identity_counts(sources[vb], mb, k, identities)
            assert np.array_equal(na, nb)
            point = (ca.sum() - cb.sum()) / na.sum() * 100
            values = (ca[indices].sum(1) - cb[indices].sum(1)) / na[indices].sum(1) * 100
            low, high = np.quantile(values, [0.025, 0.975], method='linear')
            output.append(dict(comparison=name, K=k, difference_pp=point,
                               lower_percentile_2_5=low, upper_percentile_97_5=high))
    dest = ROOT / 'derived'
    dest.mkdir(exist_ok=True)
    with (dest / 'identity_sensitivity.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    for row in output:
        print(row)


if __name__ == '__main__':
    main()
