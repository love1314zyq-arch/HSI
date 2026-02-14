import argparse
import json
import os
import statistics
import subprocess
import sys
from typing import Dict, List

from utils_hsi import ensure_dir, load_yaml, save_json


def _exp_name(cfg: Dict, seed: int) -> str:
    return (
        f"paviau_base{cfg['incremental']['base_classes']}_"
        f"inc{cfg['incremental']['task_num']}_"
        f"pca{cfg['data']['pca_dim']}_seed{seed}"
    )


def _mean_std(vals: List[float]) -> Dict[str, float]:
    if len(vals) == 0:
        return {"mean": 0.0, "std": 0.0}
    if len(vals) == 1:
        return {"mean": float(vals[0]), "std": 0.0}
    return {"mean": float(statistics.mean(vals)), "std": float(statistics.stdev(vals))}


def parse_args():
    parser = argparse.ArgumentParser(description="Run PASS-HSI on multiple random seeds")
    parser.add_argument("--config", type=str, default="configs/paviau_default.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1993, 2025, 3407])
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to run main_hsi.py")
    parser.add_argument("--skip_train", action="store_true", help="仅聚合已有结果，不重新训练")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    if not args.skip_train:
        for seed in args.seeds:
            cmd = [args.python, "main_hsi.py", "--config", args.config, "--seed", str(seed)]
            print("[run]", " ".join(cmd))
            subprocess.run(cmd, check=True)

    # 聚合每个 seed 的结果。
    seed_results = {}
    for seed in args.seeds:
        exp = _exp_name(cfg, seed)
        exp_dir = os.path.join(cfg["output_path"], exp)

        seen_path = os.path.join(exp_dir, "seen_metrics.json")
        forget_path = os.path.join(exp_dir, "forgetting.json")
        if not os.path.exists(seen_path):
            raise FileNotFoundError(f"Missing metrics file: {seen_path}")
        if not os.path.exists(forget_path):
            raise FileNotFoundError(f"Missing forgetting file: {forget_path}")

        with open(seen_path, "r", encoding="utf-8") as f:
            seen_metrics = json.load(f)
        with open(forget_path, "r", encoding="utf-8") as f:
            forgetting = json.load(f)

        seed_results[str(seed)] = {
            "seen_metrics": seen_metrics,
            "average_forgetting": forgetting["average_forgetting"],
        }

    # 按任务聚合 OA/AA/Kappa。
    task_count = len(next(iter(seed_results.values()))["seen_metrics"])
    summary = {"per_task": {}, "average_forgetting": {}}

    for t in range(task_count):
        oa_vals, aa_vals, kp_vals = [], [], []
        for seed in args.seeds:
            m = seed_results[str(seed)]["seen_metrics"][f"task_{t}"]
            oa_vals.append(float(m["oa"]))
            aa_vals.append(float(m["aa"]))
            kp_vals.append(float(m["kappa"]))

        summary["per_task"][f"task_{t}"] = {
            "oa": _mean_std(oa_vals),
            "aa": _mean_std(aa_vals),
            "kappa": _mean_std(kp_vals),
        }

    forget_vals = [float(seed_results[str(seed)]["average_forgetting"]) for seed in args.seeds]
    summary["average_forgetting"] = _mean_std(forget_vals)

    out_dir = os.path.join(cfg["output_path"], "multi_seed_summary")
    ensure_dir(out_dir)

    save_json(os.path.join(out_dir, "multi_seed_raw.json"), seed_results)
    save_json(os.path.join(out_dir, "multi_seed_summary.json"), summary)

    # 同时导出一个易读 CSV。
    csv_path = os.path.join(out_dir, "multi_seed_summary.csv")
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("item,mean,std\n")
        for t in range(task_count):
            cur = summary["per_task"][f"task_{t}"]
            f.write(f"task_{t}_oa,{cur['oa']['mean']},{cur['oa']['std']}\n")
            f.write(f"task_{t}_aa,{cur['aa']['mean']},{cur['aa']['std']}\n")
            f.write(f"task_{t}_kappa,{cur['kappa']['mean']},{cur['kappa']['std']}\n")
        f.write(
            f"average_forgetting,{summary['average_forgetting']['mean']},{summary['average_forgetting']['std']}\n"
        )

    print("Multi-seed summary saved:")
    print(os.path.join(out_dir, "multi_seed_raw.json"))
    print(os.path.join(out_dir, "multi_seed_summary.json"))
    print(csv_path)


if __name__ == "__main__":
    main()

