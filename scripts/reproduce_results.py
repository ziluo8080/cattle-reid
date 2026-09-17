"""核验归档计数并复算点估计；不加载模型或重新推理。"""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('B', 'GAP', 'SupCon-in')
VARIANTS = ('baseline', 'neutral128', 'geomalign_v3', 'blacktrim_v1')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_result(data, protocol):
    """每个模型/抽样/K唯一，身份计数必须匹配固定查询协议。"""
    rows = data['results']
    assert len(rows) == 675 and data['models'] == 45 and data['training'] is False
    keys = {(r['method'], r['fold'], r['seed'], r['draw'], r['k']) for r in rows}
    expected = {(m, f, s, d, k) for m in METHODS for f in range(5)
                for s in (17, 29, 43) for d in range(5) for k in (1, 3, 5)}
    assert keys == expected
    ids = set(protocol['candidate_ids'])
    for r in rows:
        assert r['model'] == f"{r['method']}-fold{r['fold']}-seed{r['seed']}"
        assert r['total'] == 307 and r['accuracy'] == r['correct'] / r['total']
        assert set(r['per_identity']) == ids
        assert sum(v['correct'] for v in r['per_identity'].values()) == r['correct']
        assert sum(v['total'] for v in r['per_identity'].values()) == r['total']
        for identity, v in r['per_identity'].items():
            assert isinstance(v['correct'], int) and 0 <= v['correct'] <= v['total']
            assert v['total'] == len(protocol['draws'][r['draw']]['by_identity'][identity]['query'])
    return rows


def summarize(rows):
    """先计算单次查询微平均和身份宏平均，再等权平均75条记录。"""
    output = []
    for method in METHODS:
        for k in (1, 3, 5):
            selected = [r for r in rows if r['method'] == method and r['k'] == k]
            assert len(selected) == 75
            micro = math.fsum(r['accuracy'] for r in selected) / 75 * 100
            macro = math.fsum(math.fsum(v['correct'] / v['total'] for v in r['per_identity'].values())
                              / 54 for r in selected) / 75 * 100
            output.append(dict(method=method, K=k, records=75,
                               micro_accuracy_pct=micro, macro_accuracy_pct=macro))
    return output


def main():
    """离线复核文件哈希与全部2700条历史评分。"""
    manifest = load(ROOT / 'MANIFEST.json')
    for name, checksum in manifest['sha256'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == checksum, name
    protocol = load(ROOT / 'protocols/sideview_protocol.json')
    output = []
    for variant in VARIANTS:
        data = load(ROOT / f'results/sideview/{variant}.json')
        assert data['protocol_sha256'] == manifest['sideview_protocol_sha256']
        rows = verify_result(data, protocol)
        output.extend(dict(preprocessing=variant, **r) for r in summarize(rows))
    with (ROOT / 'tables/sideview_verified.csv').open(encoding='utf-8-sig', newline='') as f:
        expected = list(csv.DictReader(f))
    assert len(expected) == len(output) == 36
    for actual, old in zip(output, expected):
        assert actual['preprocessing'] == old['preprocessing'] and actual['method'] == old['method']
        assert actual['K'] == int(old['K'])
        for key in ('micro_accuracy_pct', 'macro_accuracy_pct'):
            assert abs(actual[key] - float(old[key])) < 1e-10
    with (ROOT / 'tables/holstein_per_model.csv').open(encoding='utf-8-sig', newline='') as f:
        holstein = list(csv.DictReader(f))
    assert {(int(r['fold']), int(r['seed'])) for r in holstein} == {(f, s) for f in range(5) for s in (17, 29, 43)}
    assert len(holstein) == 15
    with (ROOT / 'tables/holstein_main_reported.csv').open(encoding='utf-8-sig', newline='') as f:
        reported = {r['method']: r for r in csv.DictReader(f)}
    for method, prefix in [('B-selected', 'B'), ('GAP', 'GAP'), ('SupCon-in', 'SupCon_in')]:
        for rank in (1, 5):
            value = math.fsum(float(r[f'{prefix}_R{rank}']) for r in holstein) / 15 * 100
            assert round(value, 2) == float(reported[method][f'Rank{rank}_pct'])
    dest = ROOT / 'derived'
    dest.mkdir(exist_ok=True)
    with (dest / 'sideview_recomputed.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print('PASS: manifest, 2700 scoring records, 145800 identity counts, 36 micro/macro pairs, 6 Holstein headline values')


if __name__ == '__main__':
    main()
