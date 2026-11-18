"""Evaluate a trained checkpoint on validation BPB or the 22 downstream tasks."""
import argparse
import json
import torch
from pathlib import Path

from data import Dataset, evaluate
from train import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--no-compile', action='store_true', help='Run without torch.compile')
    parser.add_argument('--rows', type=int, help='Validation rows; defaults to the full split')
    parser.add_argument('--downstream', type=Path, help='Directory containing the downloaded evaluation bundle')
    parser.add_argument('--tasks', nargs='+', help='Optional subset of downstream task IDs')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    model, config = load_model(args.checkpoint, args.device)
    if not args.no_compile and config.get('compile', True) and torch.device(args.device).type == 'cuda':
        model = torch.compile(model, fullgraph=config['backbone'] == 'golf', dynamic=False)
    if args.downstream:
        if config['dataset'] != 'fineweb':
            parser.error('The paper\'s 22-task protocol uses the FineWeb SentencePiece tokenizer')
        from evaluation.downstream import evaluate_downstream
        result = evaluate_downstream(model, args.downstream,
                                     args.data_dir / 'tokenizers/fineweb_1024_bpe.model',
                                     args.device, args.batch_size, args.tasks)
    else:
        dataset = Dataset(args.data_dir, dataset=config['dataset'], training=False)
        result = evaluate(model, dataset, args.device, args.rows, args.batch_size)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == '__main__':
    main()
