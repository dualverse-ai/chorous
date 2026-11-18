# Experiments

Each configuration specifies the model, table size, optimizer settings and training
schedule used for that comparison. The methods are `chorus`, `engram`,
`factorengram` and `plain`.

| Setting | Training updates | Training tokens | Seeds |
|---|---:|---:|---|
| ClimbMix, 9 layers | 6,000 | 3.15B | 1337, 2027, 3141 |
| FineWeb, 9 layers | 6,000 | 3.15B | 1337, 2027, 3141 |
| FineWeb, 18 layers | 10,000 | 5.24B | 1337, 2027, 3141 |

All settings use context length 1,024 and 524,288 prediction targets per update.
The two nine-layer settings use a 1,200-update learning-rate decay. For the
18-layer setting, Chorus and Engram use a 2,000-update decay; FactorEngram and the
plain model use a 1,200-update decay.

## Comparisons

Prepare the datasets as described in the README, then run the three seeds for
all four methods:

```bash
python run_experiments.py --setting climbmix9 \
  --data-dir data/climbmix --out runs/climbmix9
python run_experiments.py --setting fineweb9 \
  --data-dir data/fineweb --out runs/fineweb9
python run_experiments.py --setting fineweb18 \
  --data-dir data/fineweb --out runs/fineweb18
```

Runs execute sequentially. Use `--methods chorus engram` or `--seeds 1337` to
select a subset. Add `--dry-run` to print the commands. The runner skips finished
runs and resumes those with a saved checkpoint.

The nine-layer ClimbMix backbone is based on nanochat; the FineWeb models use
Parameter Golf. Chorus uses Ember for its shared token and n-gram tables. Engram
and FactorEngram use AdamW for their tables. The backbone matrices use Muon.

## Table-size scaling

The scaling sweep compares Chorus, Engram and FactorEngram at ten table sizes,
from approximately 20M to 9.6B added parameters, using seed 1337:

```bash
python run_experiments.py --setting climbmix9 --scaling \
  --data-dir data/climbmix --out runs/scaling
```

The runner uses the exact parameter budgets from the experiments. For a custom
budget, call `train.py` with `--table-size`, for example `--table-size 1.2B`.
The public trainer runs on one device, so choose a GPU with sufficient memory
for the requested table size. Reducing `--microbatch` reduces activation memory
while preserving the total batch size.

## Outputs and resuming

Each run saves `config.json`, `metrics.jsonl`, `validation.json`, a final
`model.pt`, and a `last.pt` training checkpoint. `last.pt` includes the optimizer
state, Ember accumulators, completed update count and random state. Resume with
the same configuration and `--resume runs/NAME/last.pt`.

To stop at an update boundary and save state, use `--stop-after 100` or interrupt
the trainer with Ctrl-C. Resume without `--stop-after` to continue the full
schedule. `--microbatch` may be changed when moving to a different GPU.
