from typing import Any, Dict

from benchmarks.adapters.base import BenchmarkAdapter, unsupported_result
from benchmarks.common import Scenario


class PlaceholderAdapter(BenchmarkAdapter):
    def __init__(self, name: str, reason: str, status: str = "unsupported"):
        self.name = name
        self.reason = reason
        self.status = status

    def run(self, scenario: Scenario, seed: int, output_root: str, python_exec: str) -> Dict[str, Any]:
        return unsupported_result(self.name, scenario, seed, self.reason, status=self.status)
