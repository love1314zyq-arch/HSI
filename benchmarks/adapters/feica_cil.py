import glob
import io
import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from utils_hsi import load_yaml

from benchmarks.adapters.base import BenchmarkAdapter, unsupported_result
from benchmarks.common import Scenario, cumulative_seen_classes, dataset_key, ensure_dir, run_command, write_yaml


class FEICACILAdapter(BenchmarkAdapter):
    name = "feica_cil"

    DATA_OPTIONS = {
        "paviau": "options/data/PAU.yaml",
        "salinas": "options/data/SalinasA.yaml",
        "houston": "options/data/Houston.yaml",
        "indianpines": "options/data/InP.yaml",
    }

    MODEL_OPTIONS = {
        "paviau": "options/LSC/lsc_PAU.yaml",
        "salinas": "options/LSC/lsc_SalinasA.yaml",
        "houston": "options/LSC/lsc_Houston_benchmark.yaml",
        "indianpines": "options/LSC/lsc_InP_benchmark.yaml",
    }

    MODEL_DATASET_NAMES = {
        "paviau": "HSI_PAU",
        "salinas": "HSI_SaN",
        "houston": "HSI_HOU",
        "indianpines": "HSI_INP",
    }

    RAW_DATA_ROOTS = {
        "paviau": os.path.join("data", "PaviaU", "raw"),
        "salinas": os.path.join("data", "Salinas", "raw"),
        "houston": os.path.join("data", "Houston", "raw"),
        "indianpines": os.path.join("data", "IndianPines", "raw"),
    }

    def _uniform_increment(self, task_split: List[int]) -> bool:
        inc = task_split[1:]
        return len(inc) <= 1 or len(set(int(x) for x in inc)) == 1

    def _gpu_index(self, device: str) -> int:
        if device.startswith("cuda:"):
            return int(device.split(":", 1)[1])
        if device == "cpu":
            return -1
        return 0

    def _label(self, scenario: Scenario, seed: int) -> str:
        return f"benchmark_{scenario.name}_seed{seed}"

    def _load_prediction_dump(self, path: str):
        old_loader = torch.storage._load_from_bytes
        torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        try:
            with open(path, "rb") as f:
                preds, labels = pickle.load(f)
        finally:
            torch.storage._load_from_bytes = old_loader
        if hasattr(preds, "cpu"):
            preds = preds.cpu().numpy()
        if hasattr(labels, "cpu"):
            labels = labels.cpu().numpy()
        return np.asarray(preds).reshape(-1), np.asarray(labels).reshape(-1)

    def _compute_metrics(self, preds: np.ndarray, labels: np.ndarray):
        oa = float(np.mean(preds == labels)) if len(labels) else 0.0
        label_order = sorted(int(x) for x in np.unique(labels))
        cm = confusion_matrix(labels, preds, labels=label_order)
        per_class = []
        for idx in range(len(label_order)):
            denom = cm[idx, :].sum()
            per_class.append(float(cm[idx, idx] / denom) if denom > 0 else 0.0)
        aa = float(np.mean(per_class)) if per_class else 0.0
        try:
            kappa = float(cohen_kappa_score(labels, preds, labels=label_order))
        except Exception:
            kappa = 0.0
        if np.isnan(kappa):
            kappa = 0.0
        return oa, aa, kappa

    def _normalize_results(self, result_json: Dict[str, Any], task_split: List[int], prediction_dir: str = None) -> List[Dict[str, Any]]:
        seen = cumulative_seen_classes(task_split)
        prediction_files = sorted(glob.glob(os.path.join(prediction_dir, "*.pkl"))) if prediction_dir and os.path.isdir(prediction_dir) else []
        normalized = []
        for idx, task in enumerate(result_json.get("results", [])):
            aa = None
            kappa = None
            oa = float(task["accuracy"]["total"])
            if idx < len(prediction_files):
                preds, labels = self._load_prediction_dump(prediction_files[idx])
                oa, aa, kappa = self._compute_metrics(preds, labels)
            normalized.append(
                {
                    "task_id": idx,
                    "seen_classes": seen[idx] if idx < len(seen) else None,
                    "oa": oa,
                    "aa": aa,
                    "kappa": kappa,
                }
            )
        return normalized

    def run(self, scenario: Scenario, seed: int, output_root: str, python_exec: str) -> Dict[str, Any]:
        dataset = dataset_key(scenario.dataset)
        if dataset not in self.DATA_OPTIONS:
            return unsupported_result(self.name, scenario, seed, f"FEICA-CIL has no dataset mapping for {dataset}.")
        if not self._uniform_increment(scenario.task_split):
            return unsupported_result(
                self.name,
                scenario,
                seed,
                "FEICA-CIL benchmark adapter currently supports only uniform post-base increments.",
                status="incompatible",
            )

        results_root = os.path.join(output_root, self.name, scenario.name, "runs")
        config_root = os.path.join(output_root, self.name, scenario.name, "configs")
        ensure_dir(results_root)
        ensure_dir(config_root)
        label = self._label(scenario, seed)
        initial_increment = int(scenario.task_split[0])
        increment = int(scenario.task_split[1]) if len(scenario.task_split) > 1 else int(scenario.task_split[0])

        data_option = load_yaml(os.path.join("FEICA-CIL", self.DATA_OPTIONS[dataset]))
        data_root = os.path.abspath(os.path.join(os.getcwd(), self.RAW_DATA_ROOTS[dataset]))
        if not data_root.endswith(os.sep):
            data_root = data_root + os.sep
        data_option["data_path"] = data_root
        data_option["split_rate"] = 0.2
        data_option["seed"] = [int(seed)]
        temp_data_option = os.path.join(config_root, f"{label}_data.yaml")
        write_yaml(temp_data_option, data_option)

        model_option = load_yaml(os.path.join("FEICA-CIL", self.MODEL_OPTIONS[dataset]))
        model_option["dataset"] = self.MODEL_DATASET_NAMES[dataset]
        temp_model_option = os.path.join(config_root, f"{label}_model.yaml")
        write_yaml(temp_model_option, model_option)

        cmd = [
            python_exec,
            "-m",
            "hylearn",
            "--options",
            os.path.abspath(temp_data_option),
            os.path.abspath(temp_model_option),
            "--initial-increment",
            str(initial_increment),
            "--increment",
            str(increment),
            "--device",
            str(self._gpu_index(scenario.device)),
            "--seed",
            str(seed),
            "--label",
            label,
            "--results-root",
            os.path.abspath(results_root),
            "--dump-predictions",
            "--save-model",
            "never",
        ]
        try:
            run_command(cmd, cwd=os.path.join(os.getcwd(), "FEICA-CIL"))
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

        run_files = sorted(glob.glob(os.path.join(results_root, label, "run_*_.json")))
        if not run_files:
            return {
                "algorithm": self.name,
                "dataset": scenario.dataset,
                "scenario": scenario.name,
                "task_split": scenario.task_split,
                "seed": int(seed),
                "status": "failed",
                "reason": f"No FEICA-CIL result json found under {os.path.join(results_root, label)}.",
                "seen_metrics": [],
                "final_oa": None,
                "final_aa": None,
                "final_kappa": None,
                "average_forgetting": None,
                "raw_result_path": os.path.join(results_root, label),
            }
        raw_path = run_files[0]
        with open(raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prediction_dir = os.path.join(results_root, label, "predictions_0")
        normalized = self._normalize_results(data, scenario.task_split, prediction_dir=prediction_dir)
        final = normalized[-1] if normalized else {"oa": None, "aa": None, "kappa": None}
        last_forgetting = None
        if data.get("results"):
            last_forgetting = float(data["results"][-1].get("forgetting", 0.0))
        return {
            "algorithm": self.name,
            "dataset": scenario.dataset,
            "scenario": scenario.name,
            "task_split": scenario.task_split,
            "seed": int(seed),
            "status": "success",
            "reason": "",
            "seen_metrics": normalized,
            "final_oa": final["oa"],
            "final_aa": final["aa"],
            "final_kappa": final["kappa"],
            "average_forgetting": last_forgetting,
            "raw_result_path": raw_path,
        }
