"""Run the paper's main comparisons or table-size scaling sweep."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys


SCALING_BUDGETS = [20_443_648, 39_443_456, 76_443_648, 152_754_176,
                   302_754_304, 602_754_560, 1_202_754_560, 2_402_754_560,
                   4_802_754_560, 9_602_754_560]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=['climbmix9', 'fineweb9', 'fineweb18'],
                        default='climbmix9')
    parser.add_argument('--methods', nargs='+', default=['chorus', 'engram', 'factorengram', 'plain'],
                        choices=['chorus', 'engram', 'factorengram', 'plain'])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--scaling', action='store_true')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--microbatch', type=int)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.scaling and args.setting != 'climbmix9':
        parser.error('The paper scaling sweep uses --setting climbmix9')
    root = Path(__file__).resolve().parent
    seeds = args.seeds or ([1337] if args.scaling else [1337, 2027, 3141])
    for method in args.methods:
        if args.scaling and method == 'plain':
            continue
        for budget in SCALING_BUDGETS if args.scaling else [None]:
            for seed in seeds:
                name = f'{args.setting}_{method}_s{seed}'
                if budget is not None:
                    name += f'_table{budget}'
                output = args.out / name
                config = root / 'configs' / f'{args.setting}_{method}.json'
                command = [sys.executable, str(root / 'train.py'), '--config', str(config),
                           '--data-dir', str(args.data_dir), '--out', str(output),
                           '--seed', str(seed)]
                if budget is not None:
                    command += ['--table-size', str(budget)]
                if args.microbatch is not None:
                    command += ['--microbatch', str(args.microbatch)]
                if (output / 'validation.json').exists():
                    continue
                if (output / 'last.pt').exists():
                    command += ['--resume', str(output / 'last.pt')]
                print(shlex.join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
