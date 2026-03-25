import argparse
import os
import sys
from typing import List

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.common import Scenario, load_scenario
from benchmarks.registry import build_registry
from benchmarks.reporting import write_scenario_report
from utils_hsi import ensure_dir

DEFAULT_ALGORITHMS = ["ours", "feica_cil", "hyperkd", "ssre", "lwf", "fetril", "gfr_il"]


def parse_args():
    parser = argparse.ArgumentParser(description="Unified benchmark runner for HSI class-incremental baselines.")
    parser.add_argument("--scenarios", nargs="+", default=None, help="Scenario yaml files.")
    parser.add_argument("--dataset", default=None, help="Dataset name for direct CLI mode, e.g. paviau.")
    parser.add_argument("--task-split", type=int, nargs="+", default=None, help="Task split for direct CLI mode, e.g. --task-split 8 1.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Seeds for direct CLI mode, e.g. --seeds 1993 2025.")
    parser.add_argument("--algorithms", nargs="+", default=None, help="Optional algorithm whitelist.")
    parser.add_argument("--device", default="cuda:0", help="Device string for direct CLI mode.")
    parser.add_argument("--name", default=None, help="Optional scenario name for direct CLI mode.")
    parser.add_argument("--output-root", default="benchmark_runs", help="Root directory for isolated benchmark artifacts.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for sub-runs.")
    args = parser.parse_args()
    if not args.scenarios and not (args.dataset and args.task_split):
        parser.error("Provide either --scenarios or direct CLI arguments: --dataset and --task-split.")
    return args


def _select_algorithms(scenario: Scenario, cli_algorithms: List[str] = None) -> List[str]:
    return cli_algorithms if cli_algorithms else scenario.algorithms


def _scenario_from_cli(args) -> Scenario:
    if args.name:
        name = args.name
    else:
        base_name = f"{args.dataset}_split{'-'.join(str(x) for x in args.task_split)}"
        cli_seeds = [int(x) for x in (args.seeds or [1993])]
        if len(cli_seeds) == 1:
            name = f"{base_name}_seed{cli_seeds[0]}"
        else:
            name = base_name
    return Scenario(
        name=name,
        dataset=str(args.dataset),
        task_split=[int(x) for x in args.task_split],
        seeds=[int(x) for x in (args.seeds or [1993])],
        algorithms=[str(x) for x in (args.algorithms or DEFAULT_ALGORITHMS)],
        device=str(args.device),
        notes="Generated from direct CLI arguments.",
        algorithm_options={},
    )


def main():
    args = parse_args()
    registry = build_registry()
    output_root = os.path.abspath(args.output_root)
    ensure_dir(output_root)

    scenarios = [load_scenario(path) for path in args.scenarios] if args.scenarios else [_scenario_from_cli(args)]

    for scenario in scenarios:
        print(f"[benchmark] scenario={scenario.name} dataset={scenario.dataset} split={scenario.task_split}")
        scenario_results = []
        for algorithm in _select_algorithms(scenario, args.algorithms):
            adapter = registry.get(algorithm)
            if adapter is None:
                print(f"[skip] unknown algorithm: {algorithm}")
                continue
            for seed in scenario.seeds:
                print(f"[run] algorithm={algorithm} seed={seed}")
                result = adapter.run(scenario, int(seed), output_root, args.python)
                scenario_results.append(result)
                print(f"[done] algorithm={algorithm} seed={seed} status={result['status']}")

        write_scenario_report(scenario_results, scenario, output_root)
        print(f"[report] saved to {os.path.join(output_root, 'reports', scenario.name)}")


if __name__ == "__main__":
    main()
