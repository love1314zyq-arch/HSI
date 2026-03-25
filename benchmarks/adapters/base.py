from abc import ABC, abstractmethod
from typing import Any, Dict

from benchmarks.common import Scenario


class BenchmarkAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, scenario: Scenario, seed: int, output_root: str, python_exec: str) -> Dict[str, Any]:
        raise NotImplementedError


def unsupported_result(
    algorithm: str,
    scenario: Scenario,
    seed: int,
    reason: str,
    status: str = "unsupported",
) -> Dict[str, Any]:
    return {
        "algorithm": algorithm,
        "dataset": scenario.dataset,
        "scenario": scenario.name,
        "task_split": scenario.task_split,
        "seed": int(seed),
        "status": status,
        "reason": reason,
        "seen_metrics": [],
        "final_oa": None,
        "final_aa": None,
        "final_kappa": None,
        "average_forgetting": None,
        "raw_result_path": None,
    }
