"""独立RGB基线训练与逐epoch恢复，真实运行必须提供当前实例许可。"""

import argparse
from fractions import Fraction
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch

from cattle_reid_gate0.animals_supplement import score_tasks
from cattle_reid_gate0.controlled_adapter_issuer import load_signed, require
from cattle_reid_gate0.controlled_adapter_tasks import training_epoch
from cattle_reid_gate0.controlled_adapter_training import learning_rate
from cattle_reid_gate0.controlled_adapter_epochs import validation_summary
from cattle_reid_gate0.evaluation_artifacts import digest, publish_once, signed
from cattle_reid_gate0.strong_rgb_densenet import DenseRGB, retrieval_loss
from validate_strong_rgb import configure, capture, equal


def source_binding():
    """许可绑定所有会影响训练、选模及恢复的本地源码。"""
    project = Path(__file__).resolve().parents[1]
    files = list((project / 'src/cattle_reid_gate0').glob('*.py'))
    files += [Path(__file__), Path(__file__).with_name('validate_strong_rgb.py')]
    return {str(p.relative_to(project)).replace('\\', '/'): digest(p) for p in sorted(files)}


def normalize(images, positions, device):
    """固定uint8原图变换结果进入ImageNet归一化，不使用测试统计量。"""
    value = torch.from_numpy(np.array(images[positions], copy=True)).to(device).float().div_(255)
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    return (value - mean) / std


def update(model, optimizer, images, positions, task, step, device):
    """真实消费回执紧随实际索引生成，任务顺序和尾部权重不得变更。"""
    keys = task['query_keys'] + sum(task['support_keys'], [])
    batch = normalize(images, [positions[k] for k in keys], device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = retrieval_loss(model(batch), torch.tensor(task['active'], device=device))
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    optimizer.param_groups[0]['lr'] = learning_rate(step, 21)
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'nonfinite parameters')
    return dict(step=step, sample_keys=keys, query_mask=task['active'], loss_denominator=10,
                task_sha256=signed(task)['canonical_sha256'], loss=float(loss.detach()), gradient_norm=float(norm))


def evaluate(model, images, positions, plan, device):
    """验证只编码固定验证身份，固定60 batch和尾部重复规则。"""
    keys = sorted({k for t in plan['tasks'] for k in [t['query_key'], *sum(t['support_keys'], [])]})
    embedding = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(keys), 60):
            group = keys[start:start + 60]
            padded = group + [group[-1]] * (60 - len(group))
            z = model(normalize(images, [positions[k] for k in padded], device)).cpu()
            embedding.update({key: z[i] for i, key in enumerate(group)})
    scores = score_tasks(embedding, plan['tasks'])
    summary = validation_summary(plan, scores)
    return scores, summary


def main():
    """一个worker仅处理一个模型；无有效恢复回执的模型不能进入最终评分。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--fold', type=int, required=True, choices=range(5))
    parser.add_argument('--seed', type=int, required=True, choices=(17, 29, 43))
    parser.add_argument('--device', required=True, choices=('cuda:0', 'cuda:1'))
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--canary', action='store_true')
    args = parser.parse_args()
    configure(args.device)
    release = load_signed(args.release, args.pin)
    require(release['operation'] == 'strong-rgb-densenet-train-v1' and release['training_allowed'] is True,
            'wrong training permission')
    require(release['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'stale instance')
    require(release['source_sha256'] == source_binding(), 'training source changed')
    require(release['issued_unix'] <= time.time() <= release['expires_unix'], 'expired training release')
    epochs = 1 if args.canary else 30
    require(release['epochs'] == epochs and release['updates_per_model'] == epochs * 21,
            'training budget permission mismatch')
    require([args.fold, args.seed, args.device] in release['models'], 'model/device not authorized')
    require(os.path.ismount('/localdisk-tmp') and str(args.input.resolve()).startswith('/localdisk-tmp/'), 'ephemeral input required')
    require(str(args.output.resolve()).startswith('/q1/cattle-reid/strong-rgb-densenet-v1/'), 'persistent output required')
    contract = load_signed(args.input / 'contract.json', release['contract_sha256'])
    for name in ('train-images.npy', 'train-index.json', 'design.json', f'validation-{args.fold}.json', 'densenet121-a639ec97.pth'):
        require(digest(args.input / name) == contract['files'][name], 'training input changed')
    design = load_signed(args.input / 'design.json')
    index = load_signed(args.input / 'train-index.json')
    images = np.load(args.input / 'train-images.npy', mmap_mode='r', allow_pickle=False)
    positions = {key: i for i, key in enumerate(index['keys'])}
    plan = load_signed(args.input / f'validation-{args.fold}.json')
    torch.manual_seed(args.seed)
    model = DenseRGB(torch.load(args.input / 'densenet121-a639ec97.pth', weights_only=True)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, foreach=False, fused=False)
    job = args.output / f'fold-{args.fold}-seed-{args.seed}'
    if not args.verify:
        job.mkdir(parents=True, exist_ok=False)
    else:
        completion = load_signed(job / 'completion.json')
        require(completion['release_sha256'] == args.pin, 'recovery release mismatch')
        for name, sha in completion['files'].items():
            require(digest(job / name) == sha, 'published state changed')
    history = []
    started = time.perf_counter()
    for epoch in range(epochs + 1):
        receipts = []
        if args.verify:
            previous = max(0, epoch - 1)
            state = torch.load(job / f'epoch-{previous}.pt', weights_only=True, map_location=args.device)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['cpu_rng'].cpu())
            torch.cuda.set_rng_state(state['cuda_rng'].cpu(), args.device)
        if epoch:
            tasks = training_epoch(design, args.fold, args.seed, epoch)
            require(tasks['canonical_sha256'] == contract['tasks'][f'{args.fold}-{args.seed}'][epoch - 1], 'training plan changed')
            for task in tasks['tasks']:
                receipts.append(update(model, optimizer, images, positions, task,
                    (epoch - 1) * 21 + task['update'] + 1, args.device))
        scores, metric = evaluate(model, images, positions, plan, args.device)
        record = dict(epoch=epoch, numerator=metric['numerator'], denominator=metric['denominator'],
                      consumption=receipts, validation_sha256=plan['canonical_sha256'])
        history.append(record)
        if args.verify:
            expected = torch.load(job / f'epoch-{epoch}.pt', weights_only=True, map_location=args.device)
            require(equal(capture(model, optimizer, args.device), expected), 'epoch state restore mismatch')
            require(np.array_equal(scores, np.load(job / f'epoch-{epoch}-validation.npy')), 'validation restore mismatch')
            require(record == completion['history'][epoch], 'consumption restore mismatch')
        else:
            torch.save(capture(model, optimizer, args.device), job / f'epoch-{epoch}.pt')
            np.save(job / f'epoch-{epoch}-validation.npy', scores, allow_pickle=False)
        print(f'fold={args.fold} seed={args.seed} verify={args.verify} epoch={epoch} elapsed={time.perf_counter()-started:.1f}', flush=True)
    selected = max(range(epochs + 1), key=lambda e: Fraction(history[e]['numerator'], history[e]['denominator']))
    if args.verify:
        require(selected == completion['selected_epoch'], 'selection restore mismatch')
        publish_once(job / 'recovery.json', signed(dict(completion_sha256=completion['canonical_sha256'],
            restored_epochs=epochs + 1, replayed_updates=epochs * 21, exact_state_scores_consumption=True)))
    else:
        publish_once(job / 'completion.json', signed(dict(release_sha256=args.pin,
            contract_sha256=contract['canonical_sha256'], fold=args.fold, seed=args.seed,
            history=history, selected_epoch=selected, seconds=time.perf_counter()-started,
            files={p.name: digest(p) for p in job.iterdir()})))


if __name__ == '__main__':
    main()
