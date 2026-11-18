"""The paper's 22-task DCLM evaluation, using nanochat's prompt templates."""
from contextlib import nullcontext
import json
from pathlib import Path
import random
import torch
from . import prompts


class Tokenizer:
    def __init__(self, path):
        import sentencepiece as spm
        self.encoder = spm.SentencePieceProcessor(model_file=str(path))

    def __call__(self, texts, prepend):
        return [[prepend] + self.encoder.encode(text, out_type=int, add_bos=False, add_eos=False)
                for text in texts]

    def get_bos_token_id(self):
        return self.encoder.bos_id()


def requests_for(index, data, task, tokenizer):
    item = data[index]
    sampled = random.Random(1234 + index).sample(range(len(data) - 1), task['shots'])
    demos = [data[i if i < index else i + 1] for i in sampled]
    suffix = {'multiple_choice': 'mc', 'schema': 'schema', 'language_modeling': 'lm'}[task['task_type']]
    texts = getattr(prompts, 'render_prompts_' + suffix)(item, task['delimiter'], demos)
    tokens, starts, ends = getattr(prompts, 'batch_sequences_' + suffix)(tokenizer, texts)
    if task['task_type'] == 'multiple_choice' and any(s >= e for s, e in zip(starts, ends)):
        prefixes = [text[:-len(choice)].rstrip() if choice else text.rstrip()
                    for text, choice in zip(texts, item['choices'])]
        if not all(item['choices']):
            raise ValueError('Empty answer choice')
        starts = [prompts.find_common_length([tokenizer([prefix], prepend=tokenizer.get_bos_token_id())[0], seq])
                  for prefix, seq in zip(prefixes, tokens)]
    requests = []
    for seq, start, end in zip(tokens, starts, ends):
        crop = max(0, len(seq) - 1024)
        seq, start, end = seq[crop:], start - crop, end - crop
        if not 1 <= start < end <= len(seq) <= 1024:
            raise ValueError(f'Invalid continuation in {task["id"]}, example {index}')
        requests.append((seq[:end], start))
    return requests


@torch.no_grad()
def score_requests(model, requests, device, batch_size, bos=1):
    scores = []
    device = torch.device(device)
    for start in range(0, len(requests), batch_size):
        batch = requests[start:start + batch_size]
        inputs = torch.full((batch_size, 1024), bos, dtype=torch.long, device=device)
        for i, (sequence, _) in enumerate(batch):
            inputs[i, :len(sequence) - 1] = torch.tensor(sequence[:-1], device=device)
        context = torch.autocast(device.type, dtype=torch.bfloat16) if device.type in ('cuda', 'cpu') else nullcontext()
        with context:
            logits = model(inputs)
        for i, (sequence, begin) in enumerate(batch):
            selected = logits[i, begin - 1:len(sequence) - 1].float()
            gold = torch.tensor(sequence[begin:], device=device)
            if not torch.isfinite(selected).all():
                raise FloatingPointError('Nonfinite downstream logits')
            logp = selected.log_softmax(-1).gather(-1, gold[:, None]).squeeze(-1)
            scores.append(dict(loglikelihood=logp.double().sum().item(), target_tokens=len(gold),
                               greedy_correct=torch.equal(selected.argmax(-1), gold)))
    return scores


def evaluate_downstream(model, bundle, tokenizer_path, device, batch_size=4, tasks=None):
    definitions = json.loads(Path(__file__).with_name('tasks.json').read_text())
    if tasks and not set(tasks) <= {t['id'] for t in definitions}:
        raise ValueError('Unknown downstream task')
    selected_tasks = [t for t in definitions if not tasks or t['id'] in tasks]
    tokenizer = Tokenizer(tokenizer_path)
    bundle = Path(bundle)
    if not (bundle / 'eval_data').exists():
        bundle = bundle / 'eval_bundle'
    was_training = model.training
    model.eval()
    results = {}
    try:
        for task in selected_tasks:
            source = bundle / 'eval_data' / task['dataset_uri']
            data = [json.loads(line) for line in source.read_text().splitlines()]
            if len(data) != task['examples']:
                raise ValueError(f'Unexpected example count for {task["id"]}')
            random.Random(1337).shuffle(data)
            correct = 0
            for first in range(0, len(data), 256):
                requests, sizes = [], []
                for index in range(first, min(first + 256, len(data))):
                    req = requests_for(index, data, task, tokenizer)
                    requests.extend(req)
                    sizes.append(len(req))
                scores = score_requests(model, requests, device, batch_size, tokenizer.get_bos_token_id())
                offset = 0
                for i, size in enumerate(sizes):
                    values = scores[offset:offset + size]
                    offset += size
                    if task['task_type'] == 'language_modeling':
                        hit = values[0]['greedy_correct']
                    else:
                        losses = [-v['loglikelihood'] / v['target_tokens'] for v in values]
                        hit = min(range(len(losses)), key=losses.__getitem__) == data[first + i]['gold']
                    correct += int(hit)
            accuracy = correct / len(data)
            results[task['id']] = dict(accuracy=accuracy, correct=correct, examples=len(data),
                                       normalized_accuracy=(accuracy - task['chance']) / (1 - task['chance']))
            print(json.dumps({'task': task['id'], **results[task['id']]}), flush=True)
    finally:
        model.train(was_training)
    result = {'tasks': results}
    if len(results) == len(definitions):
        result['overall_mean'] = sum(r['normalized_accuracy'] for r in results.values()) / len(results)
    return result
