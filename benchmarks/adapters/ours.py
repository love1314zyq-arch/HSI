import copy
import json
import os
from typing import Any, Dict, List

from preprocess_hsi import infer_dataset_name
from utils_hsi import load_yaml

from benchmarks.adapters.base import BenchmarkAdapter, unsupported_result
from benchmarks.common import Scenario, cumulative_seen_classes, dataset_key, ensure_dir, run_command, write_yaml


class OursAdapter(BenchmarkAdapter):
    name = "ours"

    TEMPLATE_CONFIGS = {
        "paviau": {
            "spectral3": "configs/paviau/paviau_planA_step4b_spectral3_rawreplay_herding_merged_bestselect_protoaug_hsi_budget480_fullreplay10_min5.yaml",
            "ssma": "configs/paviau/paviau_planA_step4b_ssma_rawreplay_herding_merged_bestselect_protoaug_hsi_budget480_fullreplay10_min5.yaml",
        },
        "salinas": {
            "spectral3": "configs/salinas/salinas_planA_step4b_spectral3_rawreplay_herding_merged_bestselect_protoaug_hsi_budget480_fullreplay10_min5.yaml",
            "ssma": "configs/salinas/salinas_planA_step4b_ssma_rawreplay_herding_merged_bestselect_protoaug_hsi_budget480_fullreplay10_min5.yaml",
        },
        "houston": {
            "spectral3": "configs/houston/step4b_spectral3_rawreplay_herding_merged_bestselect_protoaug_hsi_budget480_fullreplay10_min5.yaml",
            "ssma": "configs/houston/step4b_ssma_rawreplay_herding_merged_bestselect_protoaug_hsi_budget480_fullreplay10_min5.yaml",
        },
        "indianpines": {
            "spectral3": "configs/indianpines/indianpines_planA_step4b_spectral3_rawreplay_herding_merged_bestselect_protoaug_hsi_budget480_fullreplay10.yaml",
            "ssma": "configs/indianpines/indianpines_planA_step4b_ssma_rawreplay_herding_merged_bestselect_protoaug_hsi_budget480_fullreplay10.yaml",
        },
    }

    def _ssl_mode(self, task_split: List[int]) -> str:
        inc = task_split[1:] if len(task_split) > 1 else task_split
        return "spectral3" if inc and all(int(x) == 1 for x in inc) else "ssma"

    def _exp_name(self, cfg: Dict[str, Any], seed: int, task_split: List[int]) -> str:
        split_tag = "split" + "-".join(str(x) for x in task_split)
        dataset_tag = infer_dataset_name(cfg["data"]["root"]).lower()
        ssl_tag = str(cfg["pass"]["ssl_mode"]).lower()
        return f"{dataset_tag}_{split_tag}_{ssl_tag}_pca{cfg['data']['pca_dim']}_seed{seed}"

    def _normalize_metrics(self, metrics: Dict[str, Any], task_split: List[int]) -> List[Dict[str, Any]]:
        seen = cumulative_seen_classes(task_split)
        normalized = []
        for idx in range(len(task_split)):
            cur = metrics[f"task_{idx}"]
            normalized.append(
                {
                    "task_id": idx,
                    "seen_classes": seen[idx],
                    "oa": float(cur["oa"]),
                    "aa": float(cur["aa"]),
                    "kappa": float(cur["kappa"]),
                }
            )
        return normalized

    def run(self, scenario: Scenario, seed: int, output_root: str, python_exec: str) -> Dict[str, Any]:
        dataset = dataset_key(scenario.dataset)
        ssl_mode = self._ssl_mode(scenario.task_split)
        template = self.TEMPLATE_CONFIGS.get(dataset, {}).get(ssl_mode)
        if template is None or not os.path.exists(template):
            return unsupported_result(self.name, scenario, seed, f"No benchmark template config for {dataset}/{ssl_mode}.")

        cfg = copy.deepcopy(load_yaml(template))
        scenario_root = os.path.join(output_root, self.name, scenario.name)
        artifact_root = os.path.join(scenario_root, f"seed_{seed}")
        cfg["seed"] = int(seed)
        cfg["device"] = scenario.device
        cfg["save_path"] = os.path.join(artifact_root, "checkpoints")
        cfg["log_path"] = os.path.join(artifact_root, "logs")
        cfg["output_path"] = os.path.join(artifact_root, "outputs")
        cfg.setdefault("benchmark", {})
        cfg["benchmark"]["disable_experiment_log"] = True
        cfg["pass"]["ssl_mode"] = ssl_mode

        temp_cfg = os.path.join(output_root, self.name, "configs", f"{scenario.name}_seed{seed}_{ssl_mode}.yaml")
        write_yaml(temp_cfg, cfg)

        cmd = [
            python_exec,
            "main_hsi.py",
            "--config",
            temp_cfg,
            "--seed",
            str(seed),
            "--task_split",
            *[str(x) for x in scenario.task_split],
        ]
        try:
            run_command(cmd, cwd=os.getcwd())
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

        exp_name = self._exp_name(cfg, int(seed), scenario.task_split)
        exp_dir = os.path.join(cfg["output_path"], exp_name)
        seen_path = os.path.join(exp_dir, "seen_metrics.json")
        forget_path = os.path.join(exp_dir, "forgetting.json")
        if not (os.path.exists(seen_path) and os.path.exists(forget_path)):
            return {
                "algorithm": self.name,
                "dataset": scenario.dataset,
                "scenario": scenario.name,
                "task_split": scenario.task_split,
                "seed": int(seed),
                "status": "failed",
                "reason": f"Missing expected outputs under {exp_dir}.",
                "seen_metrics": [],
                "final_oa": None,
                "final_aa": None,
                "final_kappa": None,
                "average_forgetting": None,
                "raw_result_path": exp_dir,
            }

        with open(seen_path, "r", encoding="utf-8") as f:
            seen_metrics = json.load(f)
        with open(forget_path, "r", encoding="utf-8") as f:
            forgetting = json.load(f)

        normalized = self._normalize_metrics(seen_metrics, scenario.task_split)
        final = normalized[-1]
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
            "average_forgetting": float(forgetting["average_forgetting"]),
            "raw_result_path": exp_dir,
        }
