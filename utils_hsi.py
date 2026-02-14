import json
import os
import random
from typing import Any, Dict

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
