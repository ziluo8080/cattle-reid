"""SideView 三方法冻结输入对照；不调用旧双方法 Gate 流水线。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    """原子发布进度；最终结果由全新输出目录保证不覆盖历史结果。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_tasks(rows, protocol):
    """复核原身份顺序和图像级隔离；不会把它解释为独立事件。"""
    require([r['index'] for r in rows] == list(range(len(rows))), 'manifest order')
    ids = protocol['candidate_ids']
    require(len(ids) == len(set(ids)), 'duplicate candidates')
    for draw in protocol['draws']:
        require(set(draw['by_identity']) == set(ids), 'candidate membership')
        query, support = [], []
        for identity in ids:
            item = draw['by_identity'][identity]
            require(bool(item['query']), 'empty query set')
            for k in (1, 3, 5):
                values = item['support'][str(k)]
                require(len(values) == k and len(set(values)) == k, 'support count')
                require(values == item['support']['5'][:k], 'non-nested support')
            values = item['query'] + item['support']['5']
            require(all(isinstance(i, int) and 0 <= i < len(rows) for i in values), 'index out of range')
            require(all(rows[i]['identity'] == identity for i in values), 'identity mismatch')
            query.extend(item['query'])
            support.extend(item['support']['5'])
        require(len(query) == len(set(query)), 'duplicate queries')
        require(not set(query) & set(support), 'support/query overlap')


def score_embeddings(emb, protocol):
    """保持旧脚本逐候选点积及均值原型不再归一化的规则，保存逐查询证据。"""
    require(np.isfinite(emb).all(), 'nonfinite embeddings')
    require(np.allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5), 'embedding norms')
    ids = protocol['candidate_ids']
    records, score_blocks, query_blocks, target_blocks = [], [], [], []
    for draw in protocol['draws']:
        d = draw['by_identity']
        queries = [q for identity in ids for q in d[identity]['query']]
        targets = np.array([i for i, identity in enumerate(ids) for _ in d[identity]['query']])
        query_blocks.append(queries)
        target_blocks.append(targets)
        k_blocks = []
        for k in (1, 3, 5):
            gallery = [emb[d[identity]['support'][str(k)]].mean(0) for identity in ids]
            scores = np.array([[float(g @ emb[q]) for g in gallery] for q in queries], dtype=np.float32)
            require(np.isfinite(scores).all(), 'nonfinite scores')
            correct = scores.argmax(1) == targets
            per = {identity: {'correct': int(correct[targets == i].sum()), 'total': int((targets == i).sum())}
                   for i, identity in enumerate(ids)}
            records.append({'draw': draw['draw'], 'k': k, 'correct': int(correct.sum()),
                            'total': len(queries), 'accuracy': float(correct.mean()),
                            'macro_accuracy': float(np.mean([v['correct']/v['total'] for v in per.values()])),
                            'per_identity': per})
            k_blocks.append(scores)
        score_blocks.append(k_blocks)
    return records, {'scores': np.array(score_blocks), 'query_indices': np.array(query_blocks),
                     'targets': np.array(target_blocks), 'candidate_ids': np.array(ids)}


def state_digest(model):
    """前后比对参数和 BN 缓冲区，确认纯推理没有改变模型。"""
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    """先绑定输入/权重/源码与用户本轮授权，再执行全新的一次评分。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args()
    require(digest(args.request) == args.request_sha256, 'request hash mismatch')
    request = json.loads(args.request.read_text(encoding='utf-8'))
    require(request['operation'].startswith('score_') and request['operation'].endswith('_frozen') and request['user_authorized'] is True, 'operation not authorized')
    require(digest(__file__) == request['runner_sha256'], 'runner source changed')
    root = Path(request['input_root']).resolve()
    stage = Path(request['staging_root']).resolve()
    output = Path(request['output_root']).resolve()
    base = Path(request['model_base']).resolve()
    require(Path('/localdisk-tmp').is_mount() and Path('/q1').is_mount(), 'required mounts missing')
    require(root.is_relative_to('/q1') and output.is_relative_to('/q1') and stage.is_relative_to('/localdisk-tmp'), 'invalid cloud paths')
    require(shutil.disk_usage('/localdisk-tmp').free > 1024**3, 'insufficient temporary space')
    require(not output.exists(), 'fresh output directory required')
    for name, expected in request['input_sha256'].items():
        require(digest(root/name) == expected, 'input hash mismatch: ' + name)
    for name, expected in request['runtime_source_sha256'].items():
        require(digest(base/'src/cattle_reid_gate0'/name) == expected, 'runtime source mismatch: ' + name)
    roster_path = base/'locks/cows2021-fullcandidate-threeway-v1/models.json'
    require(digest(roster_path) == request['roster_sha256'], 'roster mismatch')
    models = json.loads(roster_path.read_text())
    expected = {(m, f, s) for m in ('B', 'GAP', 'SupCon-in') for f in range(5) for s in (17, 29, 43)}
    require(len(models) == 45 and {(s['method'], s['fold'], s['seed']) for s in models} == expected, 'model roster membership')
    for spec in models:
        require(digest(spec['path']) == spec['sha256'], 'weight hash mismatch: ' + spec['path'])
    protocol = json.loads((root/'protocol_index.json').read_text())
    variant = json.loads((root/'protocol.json').read_text())
    receipt = json.loads((root/'processing_receipt.json').read_text())
    require(variant['name'] == request['variant_name'], 'wrong variant')
    require(receipt['array_sha256'] == request['input_sha256']['images_uint8.npy'] and
            receipt['manifest_sha256'] == request['input_sha256']['manifest.jsonl'], 'receipt mismatch')
    rows = [json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()]
    reference = [json.loads(s) for s in (root/'reference-manifest.jsonl').read_text().splitlines()]
    fields = ('index', 'identity', 'sample_key', 'relative_path', 'source_sha256')
    require([[r[f] for f in fields] for r in rows] == [[r[f] for f in fields] for r in reference], 'variant changed image membership')
    validate_tasks(rows, protocol)
    require(len(rows) == 607 and len(protocol['candidate_ids']) == 54 and len(protocol['draws']) == 5, 'unexpected cohort')
    for draw in protocol['draws']:
        require(sum(len(v['query']) for v in draw['by_identity'].values()) == 307, 'query count')
    import torch
    import torchvision
    import PIL
    runtime = {'python': sys.version.split()[0], 'torch': torch.__version__, 'torchvision': torchvision.__version__,
               'numpy': np.__version__, 'pillow': PIL.__version__}
    require(runtime == request['runtime_versions'], 'runtime versions changed')
    require(torch.cuda.is_available(), 'CUDA required; no silent CPU fallback')
    sys.path.insert(0, str(base/'src'))
    from cattle_reid_gate0.strong_rgb_densenet import DenseRGB
    stage.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root/'images_uint8.npy', stage/'images_uint8.npy')
    require(digest(stage/'images_uint8.npy') == receipt['array_sha256'], 'staging hash mismatch')
    arr = np.load(stage/'images_uint8.npy', mmap_mode='r', allow_pickle=False)
    require(arr.shape == (607, 3, 224, 224) and arr.dtype == np.uint8, 'input shape/dtype')
    output.mkdir()
    (output/'models').mkdir()
    shutil.copyfile(args.request, output/'run-request.json')
    shutil.copyfile(roster_path, output/'models.json')
    started = time.time()
    state = {'status': 'running', 'pid': os.getpid(), 'completed': 0, 'total': 45, 'failed': 0,
             'attempts': 1, 'request_sha256': args.request_sha256, 'runtime': runtime,
             'gpu': torch.cuda.get_device_name(0), 'started_unix': started}
    write_json(output/'state.json', state)
    mean = torch.tensor([.485, .456, .406], device='cuda')[None,:,None,None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None,:,None,None]
    results, catalog = [], []
    try:
        for mi, spec in enumerate(models, 1):
            name = f"{spec['method']}-fold{spec['fold']}-seed{spec['seed']}"
            model = DenseRGB().eval().cuda().requires_grad_(False)
            model.load_state_dict(torch.load(spec['path'], map_location='cpu', weights_only=True), strict=True)
            before = state_digest(model)
            chunks = []
            with torch.inference_mode():
                for start in range(0, len(arr), 64):
                    batch = torch.from_numpy(np.array(arr[start:start+64], copy=True)).float().cuda()
                    batch = (batch/255 - mean)/std
                    chunks.append(torch.nn.functional.normalize(model(batch), dim=1).cpu().numpy())
            emb = np.concatenate(chunks)
            require(state_digest(model) == before, 'model state mutated')
            records, raw = score_embeddings(emb, protocol)
            if mi == 1:
                # 与逐支持余弦平均独立核对，数值规则保持原协议不变。
                d = protocol['draws'][0]['by_identity']
                q = raw['query_indices'][0]
                for ki, k in enumerate([1, 3, 5]):
                    ref = np.stack([(emb[q] @ emb[d[c]['support'][str(k)]].T).mean(1) for c in protocol['candidate_ids']], axis=1)
                    require(np.allclose(ref, raw['scores'][0,ki], atol=2e-6, rtol=0), 'canary cosine mismatch')
                    require(np.array_equal(ref.argmax(1), raw['scores'][0,ki].argmax(1)), 'canary predictions mismatch')
                write_json(output/'canary.json', {'passed': True, 'model': name, 'finite': True,
                           'state_unchanged': True, 'mean_cosine_equivalent': True, 'training': False})
            for r in records:
                r.update(model=name, method=spec['method'], fold=spec['fold'], seed=spec['seed'])
            np.save(output/'models'/f'{name}-embeddings.npy', emb, allow_pickle=False)
            np.savez(output/'models'/f'{name}-scores.npz', **raw)
            write_json(output/'models'/f'{name}.json', {'weight_sha256': spec['sha256'], 'state_sha256': before, 'results': records})
            results.extend(records)
            catalog.append({'model': name, 'files': {p.name: digest(p) for p in (output/'models').glob(name + '*')}})
            state.update(completed=mi, updated_unix=time.time())
            write_json(output/'state.json', state)
            print(f'{mi}/45 {name}', flush=True)
            del model
        payload = {'preprocessing': variant['name'], 'protocol_sha256': request['input_sha256']['protocol_index.json'],
                   'input_sha256': receipt['array_sha256'], 'request_sha256': args.request_sha256,
                   'models': 45, 'results': results, 'training': False, 'device': 'cuda',
                   'elapsed_seconds': time.time()-started, 'scope': 'posthoc_image_holdout_preprocessing_ablation'}
        write_json(output/'results.json', payload)
        write_json(output/'artifact-manifest.json', {'models': catalog, 'results_sha256': digest(output/'results.json')})
        state.update(status='complete', updated_unix=time.time())
        write_json(output/'state.json', state)
        print('DONE', len(results), flush=True)
    except Exception as error:
        state.update(status='failed', failed=1, error=str(error), updated_unix=time.time())
        write_json(output/'state.json', state)
        raise


if __name__ == '__main__':
    main()
