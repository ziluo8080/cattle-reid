"""可移植的冻结权重推理入口；需要45个原始权重，不训练、不改变历史结果。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'reference_pipeline'))
from score_sideview_variants import score_embeddings, validate_tasks


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory')
    roster = json.loads((ROOT/'protocols/models.json').read_text())
    for item in roster:
        path = args.weights/item['file']
        if digest(path) != item['sha256']:
            raise ValueError('Weight SHA256 mismatch: ' + item['file'])
    receipt = json.loads((args.input/'processing_receipt.json').read_text())
    array_path = args.input/'images_uint8.npy'
    if digest(array_path) != receipt['array_sha256']:
        raise ValueError('Input receipt mismatch')
    verified = json.loads((ROOT/'protocols/preprocessing_verification.json').read_text())
    expected = {r['variant']: r['sha256'] for r in verified['arrays']}
    if expected.get(receipt['variant']) != receipt['array_sha256']:
        raise ValueError('Input differs from the verified historical preprocessing array')
    arr = np.load(array_path, mmap_mode='r', allow_pickle=False)
    if arr.shape != (607, 3, 224, 224) or arr.dtype != np.uint8:
        raise ValueError('Expected 607 x 3 x 224 x 224 uint8')
    protocol = json.loads((ROOT/'protocols/sideview_protocol.json').read_text())
    rows = [json.loads(s) for s in (ROOT/'protocols/sideview_manifest.jsonl').read_text().splitlines()]
    validate_tasks(rows, protocol)
    if args.check_only:
        print('PASS: 45 checkpoint hashes, input receipt and support/query protocol')
        return
    import torch
    from torchvision.models import densenet121
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')

    class Encoder(torch.nn.Module):
        """与历史DenseRGB推理结构一致；严格加载features键，禁止自动下载权重。"""
        def __init__(self):
            super().__init__()
            self.features = densenet121(weights=None, drop_rate=0).features

        def forward(self, x):
            x = torch.nn.functional.adaptive_avg_pool2d(torch.relu(self.features(x)), 1).flatten(1)
            return torch.nn.functional.normalize(x, dim=1)

    args.output.mkdir(parents=True)
    mean = torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
    results = []
    for item in roster:
        model = Encoder().eval().cuda().requires_grad_(False)
        model.load_state_dict(torch.load(args.weights/item['file'], map_location='cpu', weights_only=True), strict=True)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(arr), 64):
                batch = torch.from_numpy(np.array(arr[start:start+64], copy=True)).float().cuda()
                chunks.append(torch.nn.functional.normalize(model((batch/255-mean)/std), dim=1).cpu().numpy())
        emb = np.concatenate(chunks)
        records, scores = score_embeddings(emb, protocol)
        name = f"{item['method']}-fold{item['fold']}-seed{item['seed']}"
        for row in records:
            row.update(model=name, method=item['method'], fold=item['fold'], seed=item['seed'])
        results.extend(records)
        np.savez(args.output/(name+'-scores.npz'), **scores)
        del model
        print(name, flush=True)
    payload = {'preprocessing': receipt['variant'], 'models': 45, 'training': False,
               'protocol_sha256': '25e97cb13bc6a88a69d8dc7914049bb384eff898de5e499a8128d7e6c6f1dab3',
               'input_sha256': receipt['array_sha256'], 'results': results,
               'runner': 'portable adapter; not the original runtime receipt',
               'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0)}
    (args.output/'results.json').write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
