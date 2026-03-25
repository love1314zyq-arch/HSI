# Unified HSI Benchmark Runner

This directory contains a lightweight adapter-based benchmark framework for
running multiple class-incremental algorithms under one scenario definition.

## What is supported now

- `ours`: fully integrated with isolated benchmark configs and outputs.
- `feica_cil`: integrated with isolated benchmark outputs.
- `hyperkd`, `ssre`, `lwf`, `gfr_il`, `fetril`: registered as explicit
  placeholders and reported as incompatible until they are ported to the same
  HSI class-incremental protocol.

## Scenario file

See `benchmarks/scenarios/` for examples. A scenario specifies:

- dataset
- task split
- seeds
- algorithms
- device

## Run

```bash
python -m benchmarks.runner --scenarios benchmarks/scenarios/paviau_split8_1.yaml
```

Or construct a scenario directly from the command line:

```bash
python -m benchmarks.runner \
  --dataset paviau \
  --task-split 8 1 \
  --seeds 1993 2025 \
  --algorithms ours feica_cil
```

Outputs are written to `benchmark_runs/` by default and will not touch your
existing `outputs/`, `logs/`, or `checkpoints/` directories.
