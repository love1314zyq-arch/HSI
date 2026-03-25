import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from utils_hsi import ensure_dir, load_yaml


@dataclass
class Scenario:
    name: str
    dataset: str
    task_split: List[int]
    seeds: List[int]
    algorithms: List[str]
    device: str = "cuda:0"
    notes: str = ""
    algorithm_options: Dict[str, Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def dataset_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    aliases = {
        "paviau": "paviau",
        "paviau": "paviau",
        "salinas": "salinas",
        "houston": "houston",
        "indianpines": "indianpines",
        "indianpine": "indianpines",
        "indian": "indianpines",
    }
    return aliases.get(key, key)


def normalize_scenario(raw: Dict[str, Any], path: str) -> Scenario:
    name = raw.get("name") or os.path.splitext(os.path.basename(path))[0]
    dataset = dataset_key(raw["dataset"])
    task_split = [int(x) for x in raw["task_split"]]
    seeds = [int(x) for x in raw.get("seeds", [1993])]
    algorithms = [str(x) for x in raw.get("algorithms", ["ours", "feica_cil"])]
    device = str(raw.get("device", "cuda:0"))
    notes = str(raw.get("notes", ""))
    algorithm_options = raw.get("algorithm_options", {}) or {}
    return Scenario(
        name=name,
        dataset=dataset,
        task_split=task_split,
        seeds=seeds,
        algorithms=algorithms,
        device=device,
        notes=notes,
        algorithm_options=algorithm_options,
    )


def load_scenario(path: str) -> Scenario:
    return normalize_scenario(load_yaml(path), path)


def cumulative_seen_classes(task_split: List[int]) -> List[int]:
    seen = []
    total = 0
    for size in task_split:
        total += int(size)
        seen.append(total)
    return seen


def scenario_slug(scenario: Scenario) -> str:
    return f"{scenario.dataset}_split{'-'.join(str(x) for x in scenario.task_split)}"


def write_yaml(path: str, data: Dict[str, Any]) -> None:
    import yaml

    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def run_command(
    cmd: List[str],
    cwd: str,
    env: Optional[Dict[str, str]] = None,
    quiet: bool = False,
) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.STDOUT if quiet else None
    return subprocess.run(cmd, cwd=cwd, env=merged_env, check=True, stdout=stdout, stderr=stderr)


def python_bin() -> str:
    return sys.executable
