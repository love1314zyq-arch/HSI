import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from utils_hsi import ensure_dir, save_json


# 计算每个阶段的遗忘率与平均遗忘率。
# A[t, k] 表示训练到第 t 个任务后，在第 k 个任务测试集上的准确率。
def compute_forgetting(taskwise_matrix: List[List[float]]) -> Dict[str, List[float]]:
    arr = np.asarray(taskwise_matrix, dtype=np.float32)
    task_count = arr.shape[0]

    forgetting_per_task = [0.0] * task_count
    for t in range(1, task_count):
        vals = []
        for k in range(t):
            hist = arr[k:t, k]
            if hist.size == 0:
                continue
            vals.append(float(np.max(hist) - arr[t, k]))
        forgetting_per_task[t] = float(np.mean(vals)) if vals else 0.0

    avg_forgetting = float(np.mean(forgetting_per_task[1:])) if task_count > 1 else 0.0
    return {
        "forgetting_per_task": forgetting_per_task,
        "average_forgetting": avg_forgetting,
    }


# 导出任务矩阵为 CSV，便于论文插表或 Excel 二次处理。
def export_taskwise_matrix_csv(taskwise_matrix: List[List[float]], seen_classes: List[int], out_csv: str):
    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        header = ["trained_task", "seen_classes"] + [f"eval_task_{i}" for i in range(len(taskwise_matrix))]
        writer.writerow(header)
        for t, row in enumerate(taskwise_matrix):
            writer.writerow([t, seen_classes[t], *row])


# 导出按任务汇总表（OA/AA/Kappa/遗忘率）。
def export_summary_csv(seen_metrics: Dict, seen_classes: List[int], forgetting: Dict, out_csv: str):
    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "seen_classes", "oa", "aa", "kappa", "forgetting"])
        for t in range(len(seen_classes)):
            m = seen_metrics.get(f"task_{t}", {})
            writer.writerow([
                t,
                seen_classes[t],
                m.get("oa", 0.0),
                m.get("aa", 0.0),
                m.get("kappa", 0.0),
                forgetting["forgetting_per_task"][t],
            ])


# 生成单算法增量曲线图：
# 横轴是已学习类别数量（例如 5, 7, 9），纵轴是该阶段整体准确率。
def plot_incremental_curves(seen_metrics: Dict, seen_classes: List[int], out_png: str, algo_name: str = "PASS"):
    ensure_dir(os.path.dirname(out_png))
    xs = seen_classes
    ys = []
    for t in range(len(seen_classes)):
        ys.append(float(seen_metrics.get(f"task_{t}", {}).get("oa", 0.0)) * 100.0)

    plt.figure(figsize=(7.5, 5.2))
    plt.plot(xs, ys, marker="o", linewidth=2.5, label=algo_name, color="#d62728")

    plt.xlabel("Number of classes", fontsize=12)
    plt.ylabel("Top-1 Accuracy (%)", fontsize=12)
    plt.title("HSI Incremental Learning (PaviaU)", fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# 生成任务矩阵热力图。
def plot_taskwise_heatmap(taskwise_matrix: List[List[float]], out_png: str):
    ensure_dir(os.path.dirname(out_png))
    arr = np.asarray(taskwise_matrix, dtype=np.float32) * 100.0

    plt.figure(figsize=(6, 5))
    im = plt.imshow(arr, cmap="YlGnBu", vmin=0, vmax=100)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Accuracy (%)")
    plt.title("Task-wise OA Matrix", fontsize=14)
    plt.xlabel("Eval Task")
    plt.ylabel("Trained Task")

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def generate_reports(exp_dir: str, seen_metrics: Dict, taskwise_matrix: List[List[float]], seen_classes: List[int]):
    forgetting = compute_forgetting(taskwise_matrix)

    save_json(os.path.join(exp_dir, "forgetting.json"), forgetting)
    export_taskwise_matrix_csv(taskwise_matrix, seen_classes, os.path.join(exp_dir, "taskwise_oa_matrix.csv"))
    export_summary_csv(seen_metrics, seen_classes, forgetting, os.path.join(exp_dir, "summary_metrics.csv"))

    plot_incremental_curves(seen_metrics, seen_classes, os.path.join(exp_dir, "incremental_curves.png"))
    plot_taskwise_heatmap(taskwise_matrix, os.path.join(exp_dir, "taskwise_heatmap.png"))

    return forgetting
