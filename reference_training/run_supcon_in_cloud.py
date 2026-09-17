"""云端 SupCon-in 普通目标配对训练；使用独立冻结许可和输出目录。"""

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
from cattle_reid_gate0.controlled_adapter_tasks import training_epoch, validation_tasks
from cattle_reid_gate0.densenet_pooling_probe import expand_validation_tasks
from supcon_in import supcon_in_loss as pairwise_loss
from cattle_reid_gate0.controlled_adapter_training import learning_rate
from cattle_reid_gate0.evaluation_artifacts import digest, publish_once, signed
from cattle_reid_gate0.strong_rgb_densenet import DenseRGB
from run_strong_rgb import evaluate, normalize
from run_densenet_pooling_probe import task_result
from validate_strong_rgb import configure, capture, equal


def update(model, optimizer, images, positions, task, step, device):
    """消费、优化器及裁剪沿用原入口，只替换训练目标。"""
    keys = task['query_keys'] + sum(task['support_keys'], [])
    batch = normalize(images, [positions[k] for k in keys], device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, _ = pairwise_loss(model(batch), torch.tensor(task['active'], device=device))
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
    optimizer.param_groups[0]['lr'] = learning_rate(step, 21)
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'nonfinite parameters')
    return dict(step=step, sample_keys=keys, query_mask=task['active'], loss_denominator=10,
                task_sha256=signed(task)['canonical_sha256'], loss=float(loss.detach()), gradient_norm=float(norm))


def source_binding():
    """只对本轮运行树签发许可，不修改原DenseNet运行树或旧许可。"""
    project = Path(__file__).resolve().parents[1]
    files = list((project / 'src/cattle_reid_gate0').glob('*.py'))
    files += [project / 'scripts' / name for name in ('run_supcon_in_cloud.py', 'supcon_in.py',
        'run_densenet_pooling_probe.py', 'run_strong_rgb.py', 'validate_strong_rgb.py')]
    return {str(p.relative_to(project)).replace('\\', '/'): digest(p) for p in sorted(files)}


def evaluate_full(model, images, positions, plan, device):
    """全验证候选只作选模完成后的开发门核对，不改变逐epoch选模。"""
    keys = sorted({k for t in plan['tasks'] for k in [t['query_key'], *sum(t['support_keys'], [])]})
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for start in range(0, len(keys), 60):
            group = keys[start:start + 60]
            padded = group + [group[-1]] * (60 - len(group))
            z = model(normalize(images, [positions[k] for k in padded], device)).cpu()
            embeddings.update({k: z[n] for n, k in enumerate(group)})
    scores = score_tasks(embeddings, plan['tasks'])
    metric, _, _ = task_result(scores, plan['tasks'])
    return scores, str(metric)


def main():
    """从官方初态完整训练，恢复进程逐轮重放；验证失败保留现场而不放宽规则。"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'output', 'release', 'baseline'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pin', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    release = load_signed(args.release, args.pin)
    require(release['operation'] == 'supcon-in-cloud-v1' and release['training_allowed'] is True,
            'wrong training permission')
    require(release['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'stale instance')
    require(release['source_sha256'] == source_binding(), 'training source changed')
    require(release['issued_unix'] <= time.time() <= release['expires_unix'], 'expired release')
    require([args.fold, args.seed, args.device] in release['models'], 'model/device not permitted')
    require(release['epochs'] == 30 and release['updates_per_model'] == 630, 'budget changed')
    require(os.path.ismount('/localdisk-tmp') and str(args.input.resolve()).startswith('/localdisk-tmp/'), 'ephemeral input required')
    require(str(args.output.resolve()).startswith('/q1/cattle-reid/supcon-in-v1/'), 'persistent output required')
    contract = load_signed(args.input / 'contract.json', release['contract_sha256'])
    for name in ('train-images.npy', 'train-index.json', 'design.json', f'validation-{args.fold}.json', 'densenet121-a639ec97.pth'):
        require(digest(args.input / name) == contract['files'][name], f'training input changed: {name}')
    for name, checksum in release['baseline_files'].items():
        require(digest(args.baseline / name) == checksum, f'baseline file changed: {name}')
    design = load_signed(args.input / 'design.json')
    plan = load_signed(args.input / f'validation-{args.fold}.json')
    require(validation_tasks(design, args.fold) == plan, 'validation task mismatch')
    full = expand_validation_tasks(design, plan)
    index = load_signed(args.input / 'train-index.json')
    images = np.load(args.input / 'train-images.npy', mmap_mode='r', allow_pickle=False)
    positions = {k: n for n, k in enumerate(index['keys'])}
    configure(args.device)
    torch.manual_seed(args.seed)
    model = DenseRGB(torch.load(args.input / 'densenet121-a639ec97.pth', weights_only=True)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, foreach=False, fused=False)
    job = args.output / f'fold-{args.fold}-seed-{args.seed}'
    if not args.verify:
        job.mkdir(parents=True, exist_ok=False)
    else:
        completion = load_signed(job / 'completion.json')
        require(completion['release_sha256'] == args.pin, 'recovery release mismatch')
        for name, checksum in completion['files'].items():
            require(digest(job / name) == checksum, f'published state changed: {name}')
    history = []
    started = time.perf_counter()
    for epoch in range(31):
        receipts = []
        if args.verify:
            state = torch.load(job / f'epoch-{max(0, epoch - 1)}.pt', map_location=args.device, weights_only=True)
            model.load_state_dict(state['model'], strict=True)
            optimizer.load_state_dict(state['optimizer'])
            torch.set_rng_state(state['cpu_rng'].cpu())
            torch.cuda.set_rng_state(state['cuda_rng'].cpu(), args.device)
            del state
        if epoch:
            tasks = training_epoch(design, args.fold, args.seed, epoch)
            require(tasks['canonical_sha256'] == contract['tasks'][f'{args.fold}-{args.seed}'][epoch - 1], 'training task changed')
            for task in tasks['tasks']:
                receipts.append(update(model, optimizer, images, positions, task,
                    (epoch - 1) * 21 + task['update'] + 1, args.device))
        scores, metric = evaluate(model, images, positions, plan, args.device)
        record = dict(epoch=epoch, numerator=metric['numerator'], denominator=metric['denominator'],
                      consumption=receipts, validation_sha256=plan['canonical_sha256'])
        history.append(record)
        if args.verify:
            expected = torch.load(job / f'epoch-{epoch}.pt', map_location=args.device, weights_only=True)
            require(equal(capture(model, optimizer, args.device), expected), 'epoch state restore mismatch')
            require(np.array_equal(scores, np.load(job / f'epoch-{epoch}-validation.npy')), 'validation restore mismatch')
            require(record == completion['history'][epoch], 'consumption restore mismatch')
            del expected
        else:
            torch.save(capture(model, optimizer, args.device), job / f'epoch-{epoch}.pt')
            np.save(job / f'epoch-{epoch}-validation.npy', scores, allow_pickle=False)
        publish_once(job / f'{"recovery" if args.verify else "train"}-epoch-{epoch}.json', signed(dict(
            epoch=epoch, metric=f'{metric["numerator"]}/{metric["denominator"]}', completed_updates=epoch * 21)))
        print(f'fold={args.fold} seed={args.seed} verify={args.verify} epoch={epoch} elapsed={time.perf_counter()-started:.1f}', flush=True)
    selected = max(range(31), key=lambda e: Fraction(history[e]['numerator'], history[e]['denominator']))
    if args.verify:
        require(selected == completion['selected_epoch'], 'selection restore mismatch')
        state = torch.load(job / f'epoch-{selected}.pt', map_location=args.device, weights_only=True)
        model.load_state_dict(state['model'], strict=True)
        rebuilt_full, rebuilt_metric = evaluate_full(model, images, positions, full, args.device)
        require(np.array_equal(rebuilt_full, np.load(job / 'selected-full-validation.npy'))
                and rebuilt_metric == completion['full_validation_metric'], 'full validation restore mismatch')
        publish_once(job / 'recovery.json', signed(dict(completion_sha256=completion['canonical_sha256'],
            restored_epochs=31, replayed_updates=630, exact_state_scores_consumption=True)))
        return
    state = torch.load(job / f'epoch-{selected}.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    full_scores, full_metric = evaluate_full(model, images, positions, full, args.device)
    np.save(job / 'selected-full-validation.npy', full_scores, allow_pickle=False)
    baseline_prefix = f'fold{args.fold}-seed{args.seed}'
    baseline = DenseRGB().to(args.device)
    baseline.load_state_dict(torch.load(args.baseline / f'{baseline_prefix}-D_selected.pt', map_location=args.device, weights_only=True))
    baseline_scores, baseline_metric = evaluate(baseline, images, positions, plan, args.device)
    reference_scores = np.load(args.baseline / f'{baseline_prefix}-validation.npy', allow_pickle=False)
    np.testing.assert_allclose(baseline_scores, reference_scores, atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(baseline_scores.argmax(-1), reference_scores.argmax(-1))
    original = load_signed(args.baseline / f'{baseline_prefix}-completion.json')
    original_metric = original['history'][original['selected_epoch']]
    require(Fraction(baseline_metric['numerator'], baseline_metric['denominator']) ==
            Fraction(original_metric['numerator'], original_metric['denominator']), 'baseline metric reconstruction failed')
    baseline_full, baseline_full_metric = evaluate_full(baseline, images, positions, full, args.device)
    np.save(job / 'baseline-validation.npy', baseline_scores, allow_pickle=False)
    np.save(job / 'baseline-full-validation.npy', baseline_full, allow_pickle=False)
    publish_once(job / 'completion.json', signed(dict(release_sha256=args.pin,
        contract_sha256=contract['canonical_sha256'], fold=args.fold, seed=args.seed,
        history=history, selected_epoch=selected, seconds=time.perf_counter()-started,
        full_validation_metric=full_metric, baseline_full_validation_metric=baseline_full_metric,
        baseline_validation_metric=str(Fraction(baseline_metric['numerator'], baseline_metric['denominator'])),
        pool_type='gap', objective='query-support-supcon-in-v1', temperature=0.1,
        files={p.name: digest(p) for p in job.iterdir()})))


if __name__ == '__main__':
    main()
