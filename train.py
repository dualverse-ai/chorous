"""Train Chorus and its baselines on the paper's packed token streams."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch


def lr_multiplier(config, step):
    """Return the learning-rate multiplier for a zero-based training step."""
    duration = config['warmdown_updates']
    index = step + 1 if config['backbone'] == 'nanochat' else step
    return min(1., max(0., (config['steps'] - index) / duration))


def apply_schedule(optimizer, config, step):
    scale = lr_multiplier(config, step)
    for group in optimizer.param_groups:
        warmup = group.get('lr_warmup_steps', 0)
        group['lr'] = group['initial_lr'] * scale
        if warmup:
            group['lr'] *= min((step + 1) / warmup, 1.)
        if group['kind'] == 'muon':
            group['momentum'] = .85 + .10 * min(step / 500., 1.)
            if config['backbone'] == 'nanochat':
                group['weight_decay'] = config['muon_weight_decay'] * max(
                    0., 1. - (step + 1) / config['steps'])
            elif group.get('optimizer_family') != 'native_golf_muon':
                group.setdefault('base_weight_decay', group['weight_decay'])
                group['weight_decay'] = group['base_weight_decay'] * max(
                    0., 1. - (step + 1) / config['steps'])
    return scale


def autocast(device):
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16)


def configure_runtime(config, device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_float32_matmul_precision('high')
    if torch.device(device).type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if config.get('deterministic_inductor', False):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        import torch._inductor.config as inductor
        if hasattr(inductor, 'deterministic'):
            inductor.deterministic = True
        if hasattr(inductor, 'batch_invariant'):
            inductor.batch_invariant = False
        inductor.triton.autotune_pointwise = False
    if config.get('compile', True):
        import torch._dynamo.config as dynamo
        if hasattr(dynamo, 'recompile_limit'):
            dynamo.recompile_limit = 128
        else:
            dynamo.cache_size_limit = 128


def make_model(config, device):
    from chorus import build_model, initialize_model
    with torch.device('meta'):
        model = build_model(config)
    model.to_empty(device=device)
    initialize_model(model, config['seed'])
    return model


def load_model(checkpoint, device='cuda'):
    """Load a final model or a resumable training checkpoint."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    configure_runtime(config, device)
    model = make_model(config, device)
    model.load_state_dict(saved['model'], strict=True)
    if hasattr(model, 'set_step'):
        model.set_step(saved['step'])
    return model, config


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def ember_buffers(model):
    return {name: value for name, value in model.named_buffers()
            if name.rsplit('.', 1)[-1].startswith('ember_')}


def save_checkpoint(path, model, optimizer, config, step):
    """Save all state needed to resume at the next update, including Ember."""
    payload = dict(config=config, step=step, model=cpu_tree(model.state_dict()))
    if optimizer is not None:
        device = next(model.parameters()).device
        payload.update(
            optimizer=cpu_tree(optimizer.state_dict()),
            ember=cpu_tree(ember_buffers(model)),
            extra_groups=[{key: cpu_tree(value) for key, value in group.items()
                           if key != 'params'} for group in optimizer.param_groups
                          if group['kind'] == 'oe_mfree'],
            rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                     torch=torch.get_rng_state(),
                     cuda=torch.cuda.get_rng_state(device)
                     if device.type == 'cuda' else None))
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, config):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    # Batch size and compilation may change when moving to another GPU.
    operational = {'microbatch', 'eval_microbatch', 'compile', 'checkpoint_every'}
    if ({k: v for k, v in config.items() if k not in operational}
            != {k: v for k, v in saved['config'].items() if k not in operational}):
        raise ValueError('Resume configuration differs from the checkpoint')
    model.load_state_dict(saved['model'], strict=True)
    buffers = ember_buffers(model)
    if set(buffers) != set(saved['ember']):
        raise ValueError('Checkpoint Ember state does not match the model')
    with torch.no_grad():
        for name, value in saved['ember'].items():
            buffers[name].copy_(value)
    optimizer.load_state_dict(saved['optimizer'])
    groups = [g for g in optimizer.param_groups if g['kind'] == 'oe_mfree']
    for group, extra in zip(groups, saved['extra_groups'], strict=True):
        group.update(extra)
    random.setstate(saved['rng']['python'])
    np.random.set_state(saved['rng']['numpy'])
    torch.set_rng_state(saved['rng']['torch'])
    device = next(model.parameters()).device
    if saved['rng']['cuda'] is not None and device.type == 'cuda':
        torch.cuda.set_rng_state(saved['rng']['cuda'], device)
    model.zero_grad(set_to_none=True)
    return saved['step']


def train_update(model, forward, optimizer, data, config, step, counts, device):
    model.train()
    if hasattr(model, 'set_step'):
        model.set_step(step + 1)
    rows = config['tokens_per_update'] // config['context']
    accumulation = rows // config['microbatch']
    objective = torch.zeros((), device=device)
    for offset in range(0, rows, config['microbatch']):
        x, y = data.batch_numpy(step, offset, config['microbatch'])
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        with autocast(device):
            loss = forward(x, y)
        # FactorEngram's forward includes its coefficient L1 penalty in train mode.
        objective += loss.detach() / accumulation
        (loss / accumulation).backward()
        if counts:
            model.oe_count(x, counts, [])
    objective = float(objective.item())
    if not math.isfinite(objective):
        raise FloatingPointError(f'Nonfinite loss at step {step + 1}')
    scale = apply_schedule(optimizer, config, step)
    optimizer.step()
    if counts:
        group = next(g for g in optimizer.param_groups if g['kind'] == 'oe_mfree')
        model.oe_mfree_step(group['lr'], counts, [], step + 1)
    model.zero_grad(set_to_none=True)
    return objective, scale


def parse_size(value):
    value = value.strip().lower()
    multiplier = {'m': 1_000_000, 'b': 1_000_000_000}.get(value[-1], 1)
    return int(float(value[:-1] if multiplier != 1 else value) * multiplier)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--table-size', type=parse_size,
                        help='Added parameter budget, e.g. 76443648 or 1.2B')
    parser.add_argument('--microbatch', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-compile', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--stop-after', type=int,
                        help='Save and stop after this many total updates')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ('seed', 'microbatch'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.table_size is not None:
        config['memory_budget'] = args.table_size
        config.pop('lookup_parameters', None)
    if args.no_compile:
        config['compile'] = False
    rows = config['tokens_per_update'] // config['context']
    if config['microbatch'] <= 0 or rows % config['microbatch']:
        raise ValueError('Microbatch must divide the number of sequences per update')
    end = config['steps'] if args.stop_after is None else args.stop_after
    if not 0 < end <= config['steps']:
        raise ValueError('--stop-after must be between 1 and the configured steps')
    if args.out.exists() and any(args.out.iterdir()) and args.resume is None:
        raise ValueError('Output directory is not empty; use --resume or a new directory')
    args.out.mkdir(parents=True, exist_ok=True)
    configure_runtime(config, args.device)
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    from chorus import setup_optimizer
    from data import Dataset, evaluate
    data = Dataset(args.data_dir, dataset=config['dataset'], steps=config['steps'])
    model = make_model(config, args.device)
    optimizer = setup_optimizer(model, config)
    completed = restore_checkpoint(args.resume, model, optimizer, config) if args.resume else 0
    if completed > end:
        raise ValueError('Checkpoint is already past --stop-after')
    forward = (torch.compile(model, dynamic=False,
                             fullgraph=config['backbone'] == 'golf')
               if config.get('compile', True) else model)
    groups = [g for g in optimizer.param_groups if g['kind'] == 'oe_mfree']
    if len(groups) > 1:
        raise ValueError('Expected one shared Ember learning-rate group')
    counts = [torch.zeros(t.weight.shape[0], device=t.weight.device)
              for t in model.oe_tables] if groups else []
    model.zero_grad(set_to_none=True)
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True
        print('Saving and stopping after the current update.', flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    with (args.out / 'metrics.jsonl').open('a', buffering=1) as log:
        for step in range(completed, end):
            start = time.perf_counter()
            objective, scale = train_update(model, forward, optimizer, data, config,
                                            step, counts, args.device)
            if torch.device(args.device).type == 'cuda':
                torch.cuda.synchronize(args.device)
            completed = step + 1
            record = dict(step=completed, tokens=completed * config['tokens_per_update'],
                          loss=objective, lr_multiplier=scale,
                          seconds=time.perf_counter() - start)
            if completed % config['val_every'] == 0 and completed < config['steps']:
                record['validation'] = evaluate(
                    forward, data, args.device, rows=config['intermediate_rows'],
                    batch_size=config['eval_microbatch'])
            log.write(json.dumps(record) + '\n')
            if step < 3 or completed % 100 == 0:
                print(json.dumps(record), flush=True)
            if (completed % config['checkpoint_every'] == 0
                    or completed == end or stopping):
                save_checkpoint(args.out / 'last.pt', model, optimizer, config, completed)
            if stopping:
                break
    if completed == config['steps']:
        result = evaluate(forward, data, args.device, rows=config['evaluation_rows'],
                          batch_size=config['eval_microbatch'])
        (args.out / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
        save_checkpoint(args.out / 'model.pt', model, None, config, completed)
        print(json.dumps(result), flush=True)
    print(f'Saved {args.out / "last.pt"} at update {completed}.', flush=True)


if __name__ == '__main__':
    main()
