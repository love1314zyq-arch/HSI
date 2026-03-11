import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


# 创建目录（若已存在则忽略），用于统一管理输出路径。
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# 读取 YAML 配置文件，返回 Python 字典。
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 将字典保存为 JSON 文件（含缩进，便于阅读）。
def save_json(path: str, data: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_experiment_log(
    path: str,
    config_path: str,
    exp_name: str,
    cfg: Dict[str, Any],
    task_split: List[int],
    seen_classes: List[int],
    task_metrics: Dict[str, Any],
    average_forgetting: float,
) -> None:
    ensure_dir(os.path.dirname(path) or ".")

    lines = []
    lines.append("=" * 96)
    lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Experiment: {exp_name}")
    lines.append(f"Config: {config_path}")
    lines.append(f"Seed: {cfg['seed']}")
    lines.append("-" * 96)
    lines.append(f"{'Task':<8}{'Classes':<10}{'OA':<12}{'AA':<12}{'Kappa':<12}")
    for task_id, class_count in enumerate(task_split):
        metrics = task_metrics.get(f"task_{task_id}", {})
        lines.append(
            f"{task_id:<8}{class_count:<10}{float(metrics.get('oa', 0.0)):<12.4f}"
            f"{float(metrics.get('aa', 0.0)):<12.4f}{float(metrics.get('kappa', 0.0)):<12.4f}"
        )
    lines.append("")

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


# 固定随机种子，保证实验可复现。
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 根据配置字符串返回 torch.device；当 CUDA 不可用时自动回退 CPU。
def get_device(device_name: str) -> torch.device:
    if "cuda" in device_name and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)
