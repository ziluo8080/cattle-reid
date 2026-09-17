"""生成独立于旧 Gate 流水线的冻结 RGB 图像级协议；本脚本不评分。"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image, ImageOps


def digest(path):
    """分块校验父工件，不因缓存大小占用额外内存。"""
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build_protocol(rows, seed):
    """所有 K 使用相同查询，支持集嵌套；不同 draw 不是独立观测。"""
    if [r['index'] for r in rows] != list(range(len(rows))):
        raise ValueError('Manifest indices must match array order')
    by_id = defaultdict(list)
    for row in rows:
        by_id[row['identity']].append(row['index'])
    candidates = sorted((i for i, indices in by_id.items() if len(indices) >= 6), key=int)
    if len(candidates) < 2:
        raise ValueError('At least two eligible identities required')
    draws = []
    for draw in range(5):
        selected = {}
        for identity in candidates:
            indices = by_id[identity].copy()
            random.Random(f'{seed}:{draw}:{identity}').shuffle(indices)
            selected[identity] = {
                'support': {str(k): indices[:k] for k in (1, 3, 5)},
                'query': indices[5:],
            }
        draws.append({'draw': draw, 'by_identity': selected})
    return {
        'name': 'sideview2026-snapshots-image-holdout-54id-v1',
        'evaluation_scope': 'exploratory_within_identity_image_holdout',
        'independent_events_verified': False,
        'near_duplicate_audit_complete': False,
        'external_source_independence_verified': False,
        'scoring_authorized_by_this_artifact': False,
        'input_policy': 'Existing fullbody-letterbox-v1 unchanged; no additional crop or resize',
        'seed': seed, 'k_values': [1, 3, 5], 'candidate_ids': candidates,
        'excluded_id_counts': {i: len(v) for i, v in by_id.items() if len(v) < 6},
        'included_images': sum(len(by_id[i]) for i in candidates),
        'queries_per_draw': sum(len(by_id[i]) - 5 for i in candidates),
        'support_policy': 'Reserve five per ID; K=1/3 use prefixes; unused supports are not queries',
        'scoring_rule': 'Mean cosine to L2-normalized support embeddings; no prototype renormalization',
        'methods': ['B', 'GAP', 'SupCon-in'],
        'weights_policy': 'Original frozen source weights; model roster hashes required before scoring',
        'metrics': ['micro_top1', 'macro_identity_top1'],
        'uncertainty_policy': 'Do not treat repeated queries, draws, or models as independent samples',
        'draws': draws,
    }


def main():
    """先验证来源校验和与逐像素重建，失败时不签发协议。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.input
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    receipt = json.loads((root / 'processing_receipt.json').read_text(encoding='utf-8'))
    hashes = {name: digest(root / name) for name in ('manifest.jsonl', 'images_uint8.npy', 'protocol.json')}
    if hashes['manifest.jsonl'] != receipt['manifest_sha256'] or hashes['images_uint8.npy'] != receipt['array_sha256']:
        raise ValueError('Parent receipt mismatch')
    rows = [json.loads(line) for line in (root / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    data = np.load(root / 'images_uint8.npy', mmap_mode='r', allow_pickle=False)
    if data.shape != (len(rows), 3, 224, 224) or data.dtype != np.uint8:
        raise ValueError('Unexpected cache shape or dtype')
    sums = {}
    for line in (root.parent / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        value, name = line.split(maxsplit=1)
        sums[name.lstrip('*')] = value
    source_hashes, pixel_hashes = [], []
    with zipfile.ZipFile(root.parent / 'snapshots.zip') as archive:
        names = sorted(n for n in archive.namelist() if n.startswith('snapshots/images/') and n.lower().endswith('.jpg'))
        expected = ['snapshots/images/' + r['relative_path'] for r in rows]
        if names != expected:
            raise ValueError('Source archive and manifest membership/order mismatch')
        for row, name in zip(rows, expected):
            raw = archive.read(name)
            sha = hashlib.sha256(raw).hexdigest()
            if sha != row['source_sha256'] or sha != sums.get(name):
                raise ValueError(f'Source checksum mismatch: {name}')
            with Image.open(io.BytesIO(raw)) as image:
                image = ImageOps.exif_transpose(image).convert('RGB')
                scale = min(224 / image.width, 224 / image.height)
                size = tuple(max(1, round(v * scale)) for v in image.size)
                canvas = Image.new('RGB', (224, 224), (128, 128, 128))
                canvas.paste(image.resize(size, Image.Resampling.LANCZOS), ((224-size[0])//2, (224-size[1])//2))
                if not np.array_equal(np.asarray(canvas).transpose(2, 0, 1), data[row['index']]):
                    raise ValueError(f'Cache reconstruction mismatch: {name}')
            source_hashes.append(sha)
            pixel_hashes.append(hashlib.sha256(data[row['index']].tobytes()).hexdigest())
    if len(set(source_hashes)) != len(rows) or len(set(pixel_hashes)) != len(rows):
        raise ValueError('Exact duplicates require grouping before protocol generation')
    protocol = build_protocol(rows, 20260916)
    protocol['input_root'] = str(root.resolve())
    protocol['parent_sha256'] = hashes
    protocol['builder_sha256'] = digest(Path(__file__))
    protocol['validation'] = {
        'source_checksums_passed': len(rows), 'pixel_reconstruction_passed': len(rows),
        'source_exact_duplicates': 0, 'cache_exact_duplicates': 0,
        'identity_counts': dict(sorted(Counter(r['identity'] for r in rows).items(), key=lambda item: int(item[0]))),
        'limitations': ['No capture timestamp found in inspected EXIF', 'No event metadata available locally',
                        'Masks exist but are unused', 'Parlor quality was not systematically audited',
                        'Snapshot inputs may retain original black borders, occlusion, and truncated bodies'],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({'output': str(args.output), 'sha256': digest(args.output),
                      'candidates': len(protocol['candidate_ids']), 'included_images': protocol['included_images'],
                      'queries_per_draw': protocol['queries_per_draw']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
