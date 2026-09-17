"""独立复现入口：重建原输入、核验任务、训练或冻结评分；不修改历史实验。"""
import argparse
import csv
from fractions import Fraction
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF, InterpolationMode
from cattle_reid_repro.evaluation_artifacts import digest, validate_signed
from cattle_reid_repro.legacy_math import training_epoch, validation_tasks, score_tasks, validation_summary, learning_rate
from cattle_reid_repro.strong_rgb_densenet import DenseRGB, retrieval_loss
from cattle_reid_repro.densenet_pairwise import pairwise_loss
from cattle_reid_repro.supcon_in import supcon_in_loss

PROTOCOL = ROOT/'protocols/holstein'


def load(name):
    """读取逐字节绑定的历史协议；gzip仅为传输压缩。"""
    path = PROTOCOL/name
    data = path.read_bytes() if path.exists() else gzip.decompress((PROTOCOL/(name+'.gz')).read_bytes())
    if name != 'contract.json':
        expected = json.loads((PROTOCOL/'contract.json').read_text())['files'][name]
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Protocol hash mismatch: '+name)
    return validate_signed(json.loads(data))


def fresh(path):
    path = path.resolve()
    if path == Path('/app') or Path('/app') in path.parents:
        raise ValueError('Experiment output under /app is forbidden')
    if path.exists():
        raise FileExistsError('Use a fresh output directory')
    path.mkdir(parents=True)


def verify_protocol():
    """重建450个训练epoch和五个验证协议，逐项核对历史任务哈希。"""
    contract, design = load('contract.json'), load('design.json')
    for fold in range(5):
        if validation_tasks(design, fold) != load(f'validation-{fold}.json'):
            raise ValueError('Validation protocol mismatch')
        for seed in (17, 29, 43):
            for epoch in range(1, 31):
                task = training_epoch(design, fold, seed, epoch)
                if task['canonical_sha256'] != contract['tasks'][f'{fold}-{seed}'][epoch-1]:
                    raise ValueError('Training protocol mismatch')
    print('PASS: 450 training epoch hashes and 5 validation protocols', flush=True)


def prepare(args):
    """从官方Raw.zip按原始索引重建uint8输入，核验文件及逐像素哈希。"""
    contract = load('contract.json')
    indices = {role: load(role+'-index.json') for role in ('train', 'test')}
    records = {k: v for index in indices.values() for k, v in index['records'].items()}
    values = {}
    with zipfile.ZipFile(args.raw_zip) as archive:
        for number, (key, record) in enumerate(sorted(records.items()), 1):
            name = record['relative_path'].split('holstein/extracted/', 1)[1]
            data = archive.read(name)
            if hashlib.sha256(data).hexdigest() != record['file_sha256']:
                raise ValueError('Source image mismatch: '+name)
            with Image.open(io.BytesIO(data)) as image:
                image = TF.center_crop(TF.resize(image.convert('RGB'), 256, InterpolationMode.BILINEAR), 224)
                value = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).copy()
            if hashlib.sha256(value.tobytes()).hexdigest() != record['transformed_sha256']:
                raise ValueError('Decoded/transformed pixels mismatch: '+key)
            values[key] = value
            if number % 500 == 0:
                print('Verified images', number, flush=True)
    fresh(args.output)
    for role, index in indices.items():
        target = args.output/(role+'-images.npy')
        np.save(target, np.stack([values[k] for k in index['keys']]), allow_pickle=False)
        if digest(target) != contract['files'][target.name]:
            raise ValueError('Array bytes mismatch: '+target.name)
    print('PASS: original train/test arrays reproduced byte-for-byte', flush=True)


def configure(device):
    """沿用历史确定性设置；CPU仅允许只读矩阵重放和协议测试。"""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this historical GPU computation')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(17)
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(17)


def inputs(root, role):
    path = root/(role+'-images.npy')
    if digest(path) != load('contract.json')['files'][path.name]:
        raise ValueError('Historical input array mismatch')
    index = load(role+'-index.json')
    return np.load(path, mmap_mode='r', allow_pickle=False), {k: i for i, k in enumerate(index['keys'])}


def normalize(images, positions, device):
    value = torch.from_numpy(np.array(images[positions], copy=True)).to(device).float().div_(255)
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    return (value-mean)/std


def evaluate(model, images, positions, tasks, device, batch_size):
    """训练验证采用60并填充尾部；Holstein正式评分采用单图编码。"""
    keys = sorted({k for t in tasks for k in [t['query_key'], *sum(t['support_keys'], [])]})
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(keys), batch_size):
            group = keys[start:start+batch_size]
            padded = group+[group[-1]]*(batch_size-len(group))
            z = model(normalize(images, [positions[k] for k in padded], device)).cpu()
            embeddings.update({k: z[i] for i, k in enumerate(group)})
    return score_tasks(embeddings, tasks)


def train(args):
    """独立新运行使用原科学函数；不修改或伪造历史云端许可。"""
    verify_protocol()
    config = json.loads((ROOT/'protocols/training_config.json').read_text())
    if digest(args.init_weight) != config['pretrained_sha256']:
        raise ValueError('ImageNet initialization hash mismatch')
    images, positions = inputs(args.input, 'train')
    if args.check_only:
        print('PASS: training dependencies, all tasks, initialization and input hashes')
        return
    configure(args.device)
    fresh(args.output)
    torch.manual_seed(args.seed)
    model = DenseRGB(torch.load(args.init_weight, map_location='cpu', weights_only=True)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, foreach=False, fused=False)
    objective = {'B': pairwise_loss, 'GAP': retrieval_loss, 'SupCon-in': supcon_in_loss}[args.method]
    design, plan = load('design.json'), load(f'validation-{args.fold}.json')
    history, best, selected = [], None, None
    for epoch in range(31):
        if epoch:
            tasks = training_epoch(design, args.fold, args.seed, epoch)
            for task in tasks['tasks']:
                keys = task['query_keys']+sum(task['support_keys'], [])
                model.train()
                optimizer.zero_grad(set_to_none=True)
                loss, _ = objective(model(normalize(images, [positions[k] for k in keys], args.device)),
                                    torch.tensor(task['active'], device=args.device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
                optimizer.param_groups[0]['lr'] = learning_rate((epoch-1)*21+task['update']+1, 21)
                optimizer.step()
                if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                    raise ValueError('nonfinite parameters')
        scores = evaluate(model, images, positions, plan['tasks'], args.device, 60)
        metric = validation_summary(plan, scores)
        value = Fraction(metric['numerator'], metric['denominator'])
        history.append({'epoch': epoch, 'numerator': value.numerator, 'denominator': value.denominator})
        np.save(args.output/f'epoch-{epoch}-validation.npy', scores, allow_pickle=False)
        if best is None or value > best:
            best, selected = value, epoch
            torch.save(model.state_dict(), args.output/'selected.pt')
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(args.device),
                    'epoch': epoch}, args.output/'last-state.pt')
        (args.output/'history.json').write_text(json.dumps(history, indent=2)+'\n', encoding='utf-8')
        print(args.method, args.fold, args.seed, epoch, str(value), flush=True)
    result = {'method': args.method, 'fold': args.fold, 'seed': args.seed, 'selected_epoch': selected,
              'updates': 630, 'selected_sha256': digest(args.output/'selected.pt'),
              'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(args.device),
              'scope': 'new portable reproduction, not a historical execution receipt'}
    (args.output/'completion.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


def metrics(scores, tasks):
    """主表按单模型任务微平均，再模型等权；不替换成身份/日期宏平均。"""
    targets = np.array([t['roster'].index(t['query_identity']) for t in tasks])
    ranks = np.argsort(-scores, axis=1, kind='stable')
    return {'Rank1': float((ranks[:, 0] == targets).mean()),
            'Rank5': float((ranks[:, :5] == targets[:, None]).any(1).mean())}


def score(args):
    """冻结原权重重算，或直接重放45个已归档评分矩阵。"""
    tasks = load('full-tasks.json')['tasks']
    roster = json.loads((ROOT/'protocols/models.json').read_text())
    selected = [r for r in roster if (args.method is None or r['method'] == args.method)
                and (args.fold is None or r['fold'] == args.fold) and (args.seed is None or r['seed'] == args.seed)]
    if args.command == 'score':
        for row in selected:
            if args.trained_root is not None:
                job = args.trained_root/f"{row['method']}-fold{row['fold']}-seed{row['seed']}"
                completion = json.loads((job/'completion.json').read_text())
                if any(completion[k] != row[k] for k in ('method','fold','seed')) or completion['updates'] != 630:
                    raise ValueError('Retraining completion mismatch')
                row['checkpoint'] = job/'selected.pt'
                expected_sha = completion['selected_sha256']
            else:
                row['checkpoint'] = args.weights/row['file']
                expected_sha = row['sha256']
            if digest(row['checkpoint']) != expected_sha:
                raise ValueError('Checkpoint mismatch')
        images, positions = inputs(args.input, 'test')
        configure(args.device)
        model = DenseRGB().to(args.device).eval().requires_grad_(False)
    references = {r['file']: r['sha256'] for r in json.loads((ROOT/'protocols/holstein_reference_scores.json').read_text())}
    fresh(args.output)
    rows = []
    for row in selected:
        subset = [t for t in tasks if t['fold'] == row['fold']]
        name = f"{row['method']}-fold{row['fold']}-seed{row['seed']}-full.npy"
        if args.command == 'replay':
            source = args.scores/name
            if digest(source) != references[name]:
                raise ValueError('Archived score hash mismatch')
            scores = np.load(source, allow_pickle=False)
        else:
            model.load_state_dict(torch.load(row['checkpoint'], map_location='cpu', weights_only=True), strict=True)
            scores = evaluate(model, images, positions, subset, args.device, 1)
            np.save(args.output/name, scores, allow_pickle=False)
        result = {'method': row['method'], 'fold': row['fold'], 'seed': row['seed'], **metrics(scores, subset)}
        rows.append(result)
        print(result, flush=True)
    summary = {method: {rank: float(np.mean([r[rank] for r in rows if r['method'] == method]))
                        for rank in ('Rank1', 'Rank5')} for method in sorted({r['method'] for r in rows})}
    (args.output/'summary.json').write_text(json.dumps({'rows': rows, 'summary': summary}, indent=2)+'\n', encoding='utf-8')
    print(summary, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify-protocol')
    p = commands.add_parser('prepare')
    p.add_argument('--raw-zip', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('train')
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--init-weight', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method', choices=['B', 'GAP', 'SupCon-in'], required=True)
    p.add_argument('--fold', type=int, choices=range(5), required=True)
    p.add_argument('--seed', type=int, choices=[17,29,43], required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--check-only', action='store_true')
    for command in ('score', 'replay'):
        p = commands.add_parser(command)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--method', choices=['B','GAP','SupCon-in'])
        p.add_argument('--fold', type=int, choices=range(5))
        p.add_argument('--seed', type=int, choices=[17,29,43])
        if command == 'score':
            p.add_argument('--input', type=Path, required=True)
            group = p.add_mutually_exclusive_group(required=True)
            group.add_argument('--weights', type=Path)
            group.add_argument('--trained-root', type=Path)
            p.add_argument('--device', default='cuda:0')
        else:
            p.add_argument('--scores', type=Path, required=True)
    args = parser.parse_args()
    {'verify-protocol': lambda _: verify_protocol(), 'prepare': prepare, 'train': train,
     'score': score, 'replay': score}[args.command](args)


if __name__ == '__main__':
    main()
