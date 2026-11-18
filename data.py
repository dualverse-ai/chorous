"""Data loaders and bits-per-byte evaluation for the paper's two corpora."""
from __future__ import annotations
import base64
import bisect
from contextlib import nullcontext
import json
import math
from pathlib import Path
import numpy as np
import torch

CONTEXT = 1024
SEQUENCES_PER_STEP = 512
FINEWEB_TOKENS_PER_STEP = 524296


def climbmix_tokenizer(path=None):
    import tiktoken
    path = Path(path) if path else Path(__file__).parent / 'tokenizer' / 'climbmix.json'
    obj = json.loads(path.read_text())
    return tiktoken.Encoding(name='climbmix8192', pat_str=obj['pattern'],
                            mergeable_ranks={base64.b64decode(k): v for k, v in obj['ranks']},
                            special_tokens=obj['special_tokens'])


class PackedRows:
    """Split each packed 2049-token row into two overlapping 1025-token rows."""
    def __init__(self, path):
        path = Path(path)
        if not path.stat().st_size or path.stat().st_size % (2049 * 2):
            raise ValueError(f'Invalid packed data: {path}')
        self.source = np.memmap(path, dtype='<u2', mode='r').reshape(-1, 2049)

    def __len__(self):
        return 2 * len(self.source)

    def batch_numpy(self, start, count):
        if start < 0 or count <= 0 or start + count > len(self):
            raise IndexError('Packed dataset exhausted')
        rows = np.empty((count, CONTEXT + 1), dtype=np.int64)
        for half in (0, 1):
            offset = (half - start) % 2
            if offset < count:
                size = (count - offset + 1) // 2
                original = (start + offset) // 2
                left = half * CONTEXT
                rows[offset::2] = self.source[original:original + size, left:left + CONTEXT + 1]
        return rows[:, :-1].copy(), rows[:, 1:].copy()


class ShardStream:
    """Read consecutive tokens across Parameter Golf binary shards."""
    def __init__(self, paths):
        self.arrays, self.ends = [], []
        self.total = 0
        for path in paths:
            path = Path(path)
            header = np.fromfile(path, dtype='<i4', count=256)
            if header.size != 256 or tuple(header[:2]) != (20240520, 1):
                raise ValueError(f'Invalid shard header: {path}')
            count = int(header[2])
            if count <= 0 or path.stat().st_size != 1024 + 2 * count:
                raise ValueError(f'Invalid shard size: {path}')
            self.arrays.append(np.memmap(path, mode='r', dtype='<u2', offset=1024, shape=(count,)))
            self.total += count
            self.ends.append(self.total)
        if not self.arrays:
            raise ValueError('No data shards found')

    def take(self, start, count):
        if start < 0 or count <= 0 or start + count > self.total:
            raise IndexError('Token stream exhausted')
        pieces = []
        while count:
            index = bisect.bisect_right(self.ends, start)
            base = self.ends[index - 1] if index else 0
            size = min(count, self.ends[index] - start)
            pieces.append(self.arrays[index][start - base:start - base + size])
            start += size
            count -= size
        return np.array(pieces[0] if len(pieces) == 1 else np.concatenate(pieces), dtype=np.int64)


def sentencepiece_luts(path):
    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor(model_file=str(path))
    base = np.zeros(tokenizer.vocab_size(), dtype=np.int16)
    spaces = np.zeros(tokenizer.vocab_size(), dtype=bool)
    boundary = np.ones(tokenizer.vocab_size(), dtype=bool)
    for i in range(tokenizer.vocab_size()):
        if tokenizer.is_control(i) or tokenizer.is_unknown(i) or tokenizer.is_unused(i):
            continue
        boundary[i] = False
        if tokenizer.is_byte(i):
            base[i] = 1
            continue
        piece = tokenizer.id_to_piece(i)
        if piece.startswith('▁'):
            spaces[i] = True
            piece = piece[1:]
        base[i] = len(piece.encode('utf8'))
    return tokenizer, base, spaces, boundary


def target_byte_counts(previous, targets, luts):
    _, base, spaces, boundary = luts
    return base[targets] + (spaces[targets] & ~boundary[previous]).astype(np.int16)


class Dataset:
    def __init__(self, root, dataset='climbmix', steps=6000, training=True):
        self.root, self.name, self.steps = Path(root), dataset, steps
        self.training = None
        if dataset == 'climbmix':
            if training:
                self.training = PackedRows(self.root / 'packed/train_fresh.uint16.bin')
                if len(self.training) < steps * SEQUENCES_PER_STEP:
                    raise ValueError('Training corpus is too short for the requested steps')
            self.validation = PackedRows(self.root / 'packed/validation.uint16.bin')
            self.validation_rows = min(20480, len(self.validation))
            self.tokenizer = climbmix_tokenizer()
            self.byte_lengths = np.array([
                0 if i in self.tokenizer._special_tokens.values()
                else len(self.tokenizer.decode([i]).encode('utf8'))
                for i in range(self.tokenizer.n_vocab)], dtype=np.int16)
        elif dataset == 'fineweb':
            folder = self.root / 'datasets/fineweb10B_sp1024'
            if training:
                self.training = ShardStream(sorted(folder.glob('fineweb_train_*.bin')))
                if self.training.total < steps * FINEWEB_TOKENS_PER_STEP:
                    raise ValueError('Download enough FineWeb shards for the requested steps')
            self.validation = ShardStream([folder / 'fineweb_val_000000.bin'])
            self.validation_rows = (self.validation.total - 1) // CONTEXT
            self.luts = sentencepiece_luts(self.root / 'tokenizers/fineweb_1024_bpe.model')
            self.tokenizer = self.luts[0]
        else:
            raise ValueError(f'Unknown dataset: {dataset}')

    def batch_numpy(self, step, start_sequence=0, sequences=16):
        if not 0 <= step < self.steps or start_sequence < 0 or sequences <= 0 or start_sequence + sequences > SEQUENCES_PER_STEP:
            raise ValueError('Invalid training step or microbatch')
        if self.training is None:
            raise ValueError('Dataset was opened for evaluation only')
        if self.name == 'climbmix':
            return self.training.batch_numpy(step * SEQUENCES_PER_STEP + start_sequence, sequences)
        rows = []
        for sequence in range(start_sequence, start_sequence + sequences):
            logical, within = divmod(sequence, 64)
            pos = step * FINEWEB_TOKENS_PER_STEP + logical * 65537 + within * CONTEXT
            rows.append(self.training.take(pos, CONTEXT + 1))
        rows = np.stack(rows)
        return rows[:, :-1].copy(), rows[:, 1:].copy()

    def validation_batch(self, start_sequence, sequences):
        if start_sequence < 0 or sequences <= 0 or start_sequence + sequences > self.validation_rows:
            raise ValueError('Validation batch exceeds the evaluation split')
        if self.name == 'climbmix':
            return self.validation.batch_numpy(start_sequence, sequences)
        chunk = self.validation.take(start_sequence * CONTEXT, sequences * CONTEXT + 1)
        return chunk[:-1].reshape(sequences, CONTEXT), chunk[1:].reshape(sequences, CONTEXT)


@torch.no_grad()
def evaluate(model, dataset, device, rows=None, batch_size=16):
    """Use each corpus's original byte normalization and special-token policy."""
    device = torch.device(device)
    rows = dataset.validation_rows if rows is None else rows
    if not 0 < rows <= dataset.validation_rows or batch_size <= 0:
        raise ValueError('Invalid validation size')
    was_training = model.training
    model.eval()
    nats = 0.0
    nbytes = ntokens = 0
    try:
        for start in range(0, rows, batch_size):
            x, y = dataset.validation_batch(start, min(batch_size, rows - start))
            tx, ty = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            context = torch.autocast(device.type, dtype=torch.bfloat16) if device.type in ('cuda', 'cpu') else nullcontext()
            with context:
                loss = model(tx, ty, reduction='none').reshape(ty.shape)
            losses = loss.float().cpu().numpy().astype(np.float64)
            if not np.isfinite(losses).all():
                raise FloatingPointError('Nonfinite validation loss')
            if dataset.name == 'climbmix':
                lengths = dataset.byte_lengths[y]
                mask = lengths > 0
                nats += (losses * mask).sum()
                ntokens += int(mask.sum())
            else:
                lengths = target_byte_counts(x, y, dataset.luts)
                nats += losses.sum()
                ntokens += y.size
            nbytes += int(lengths.sum())
    finally:
        model.train(was_training)
    if not nbytes:
        raise ValueError('Validation split has no scored bytes')
    return dict(bpb=float(nats / (math.log(2) * nbytes)), ce=float(nats / ntokens),
                nats=float(nats), bytes=nbytes, tokens=ntokens, rows=rows)
