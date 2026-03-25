import json
import os
from typing import Any, Dict

from benchmarks.adapters.base import BenchmarkAdapter, unsupported_result
from benchmarks.common import Scenario, dataset_key, ensure_dir, run_command


class HyperKDAdapter(BenchmarkAdapter):
    name = "hyperkd"

    SUPPORTED_DATASETS = {"paviau", "salinas", "houston", "indianpines"}

    def run(self, scenario: Scenario, seed: int, output_root: str, python_exec: str) -> Dict[str, Any]:
        dataset = dataset_key(scenario.dataset)
        if dataset not in self.SUPPORTED_DATASETS:
            return unsupported_result(self.name, scenario, seed, f"HyperKD benchmark wrapper has no dataset mapping for {dataset}.")

        scenario_root = os.path.join(output_root, self.name, scenario.name, f"seed_{seed}")
        ensure_dir(scenario_root)
        cmd = [
            python_exec,
            "benchmark_hsi.py",
            "--dataset",
            dataset,
            "--task-split",
            *[str(x) for x in scenario.task_split],
            "--seed",
            str(seed),
            "--device",
            str(scenario.device),
            "--output-root",
            os.path.abspath(scenario_root),
        ]
        try:
            run_command(cmd, cwd=os.path.join(os.getcwd(), "hyperkd"))
        except Exception as exc:
            return {
                "algorithm": self.name,
                "dataset": scenario.dataset,
                "scenario": scenario.name,
                "task_split": scenario.task_split,
                "seed": int(seed),
                "status": "failed",
                "reason": str(exc),
                "seen_metrics": [],
                "final_oa": None,
                "final_aa": None,
                "final_kappa": None,
                "average_forgetting": None,
                "raw_result_path": None,
            }

        result_path = os.path.join(scenario_root, "result.json")
        if not os.path.exists(result_path):
            return {
                "algorithm": self.name,
                "dataset": scenario.dataset,
                "scenario": scenario.name,
                "task_split": scenario.task_split,
                "seed": int(seed),
                "status": "failed",
                "reason": f"No HyperKD result.json found under {scenario_root}.",
                "seen_metrics": [],
                "final_oa": None,
                "final_aa": None,
                "final_kappa": None,
                "average_forgetting": None,
                "raw_result_path": scenario_root,
            }

        with open(result_path, "r", encoding="utf-8") as f:
            result = json.load(f)

        final = result["seen_metrics"][-1] if result.get("seen_metrics") else {"oa": None, "aa": None, "kappa": None}
        return {
            "algorithm": self.name,
            "dataset": scenario.dataset,
            "scenario": scenario.name,
            "task_split": scenario.task_split,
            "seed": int(seed),
            "status": "success",
            "reason": "",
            "seen_metrics": result.get("seen_metrics", []),
            "final_oa": final.get("oa"),
            "final_aa": final.get("aa"),
            "final_kappa": final.get("kappa"),
            "average_forgetting": result.get("average_forgetting"),
            "raw_result_path": result_path,
        }
