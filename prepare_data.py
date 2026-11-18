"""Prepare ClimbMix, FineWeb, or the downstream evaluation bundle."""
from __future__ import annotations
import argparse
import bisect
from collections import defaultdict, deque
import hashlib
import itertools
from pathlib import Path
import shutil
import urllib.request

import numpy as np

from data import climbmix_tokenizer

CLIMBMIX_REVISION = '915333b4f8b8684f39aeaafea600fea6f43fb703'
FINEWEB_REVISION = 'a85b0e6035c3c94bc23685a07c81a8f3bf89db80'


def download(repo, filename, revision, root):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo, repo_type='dataset', filename=filename,
                               revision=revision, local_dir=root))


def text_groups(root, shards):
    import pyarrow.parquet as pq
    for shard in shards:
        parquet = pq.ParquetFile(root / f'shard_{shard:05d}.parquet')
        for group in range(parquet.num_row_groups):
            yield parquet.read_row_group(group, columns=['text']).column('text').to_pylist()


def document_key(text):
    return hashlib.blake2b(text.encode('utf8'), digest_size=32).digest()


def token_batches(root, shards, tokenizer, *, exclude=None, deduplicate=False, repeat=False):
    seen = set()
    bos = tokenizer.encode_single_token('<|reserved_0|>')
    while True:
        for texts in text_groups(root, shards):
            if exclude or deduplicate:
                accepted = []
                for text in texts:
                    key = document_key(text)
                    if exclude and key in exclude:
                        continue
                    if deduplicate and key in seen:
                        continue
                    seen.add(key)
                    accepted.append(text)
                texts = accepted
            for start in range(0, len(texts), 128):
                encoded = tokenizer.encode_ordinary_batch(texts[start:start + 128], num_threads=8)
                yield [np.asarray([bos] + tokens, dtype='<u2') for tokens in encoded]
        if not repeat:
            return


def packed_rows(batches, capacity=2049, buffer_size=1000):
    """Best-fit packing with first-occurrence tie breaking and cropped tails."""
    buckets, lengths = defaultdict(deque), []
    size, exhausted = 0, False
    batches = iter(batches)
    while True:
        row = np.empty(capacity, dtype='<u2')
        pos = 0
        while pos < capacity:
            while size < buffer_size and not exhausted:
                try:
                    batch = next(batches)
                except StopIteration:
                    exhausted = True
                    break
                for doc in batch:
                    length = len(doc)
                    if not buckets[length]:
                        bisect.insort(lengths, length)
                    buckets[length].append(doc)
                    size += 1
            if not size:
                return
            index = max(0, bisect.bisect_right(lengths, capacity - pos) - 1)
            length = lengths[index]
            doc = buckets[length].popleft()
            if not buckets[length]:
                del buckets[length]
                lengths.pop(index)
            size -= 1
            count = min(length, capacity - pos)
            row[pos:pos + count] = doc[:count]
            pos += count
        yield row


def write_rows(path, batches, count):
    if path.exists() and path.stat().st_size == count * 2049 * 2:
        return
    temporary = path.with_suffix(path.suffix + '.partial')
    written = 0
    with temporary.open('wb') as stream:
        for row in itertools.islice(packed_rows(batches), count):
            row.tofile(stream)
            written += 1
            if written % 100000 == 0:
                print(f'{path.name}: {written:,}/{count:,} rows', flush=True)
    if written != count:
        raise ValueError(f'Not enough data: {written} rows, expected {count}')
    temporary.replace(path)


def prepare_climbmix(root, steps):
    raw = root / 'raw'
    packed = root / 'packed'
    packed.mkdir(parents=True, exist_ok=True)
    # Shard 9 is held out; shard 6542 is the original validation split.
    for shard in [*range(74), 6542]:
        download('karpathy/climbmix-400b-shuffle', f'shard_{shard:05d}.parquet', CLIMBMIX_REVISION, raw)
    tokenizer = climbmix_tokenizer()
    exclude = {document_key(text) for texts in text_groups(raw, [9, 6542]) for text in texts}
    batches = token_batches(raw, [*range(9), *range(10, 74)], tokenizer,
                            exclude=exclude, deduplicate=True)
    write_rows(packed / 'train_fresh.uint16.bin', batches, steps * 256)
    batches = token_batches(raw, [6542], tokenizer, repeat=True)
    write_rows(packed / 'validation.uint16.bin', batches, 10240)


def prepare_fineweb(root, steps):
    shards = (steps * 524296 + 99_999_999) // 100_000_000
    names = [f'datasets/fineweb10B_sp1024/fineweb_train_{i:06d}.bin' for i in range(shards)]
    names += ['datasets/fineweb10B_sp1024/fineweb_val_000000.bin',
              'tokenizers/fineweb_1024_bpe.model', 'tokenizers/fineweb_1024_bpe.vocab']
    for name in names:
        target = root / name
        if target.exists():
            continue
        downloaded = download('willdepueoai/parameter-golf', 'datasets/' + name, FINEWEB_REVISION, root / '.download')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(downloaded), target)


def prepare_downstream(root):
    url = 'https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip'
    archive = root / 'eval_bundle.zip'
    root.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix('.partial')
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(archive)
    import zipfile
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            destination = (root / member.filename).resolve()
            if not destination.is_relative_to(root.resolve()):
                raise ValueError('Invalid path in evaluation bundle')
        bundle.extractall(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['climbmix', 'fineweb', 'downstream'])
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=6000, help='Training updates; use 10000 for FineWeb18')
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be positive')
    if args.dataset == 'climbmix':
        prepare_climbmix(args.data_dir, args.steps)
    elif args.dataset == 'fineweb':
        prepare_fineweb(args.data_dir, args.steps)
    else:
        prepare_downstream(args.data_dir)


if __name__ == '__main__':
    main()
