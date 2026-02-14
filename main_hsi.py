import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from PASS_hsi import ProtoAugSSLHSI
from ResNet_hsi import resnet18_hsi
from hybrid_hsi import hybrid_hsi_lite
from dataset_paviau import PaviaUDataManager
from metrics_hsi import evaluate_all
from myNetwork_hsi import NetworkHSI
from preprocess_hsi import download_paviau, prepare_processed_data
from report_hsi import generate_reports
from utils_hsi import ensure_dir, get_device, load_yaml, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="PASS for HSI incremental classification (PaviaU)")
    parser.add_argument("--config", type=str, default="configs/paviau_default.yaml")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Optional: override random seed in config")
    return parser.parse_args()


def ensure_data_ready(cfg: Dict):
    data_root = cfg["data"]["root"]
    pca_dim = int(cfg["data"]["pca_dim"])
    processed_dir = os.path.join(data_root, "processed")
    processed = os.path.join(processed_dir, f"pca{pca_dim}_cube.npy")
    legacy_processed = os.path.join(processed_dir, "pca30_cube.npy")
    gt_path = os.path.join(processed_dir, "gt.npy")
    train_mask = os.path.join(processed_dir, f"train_mask_seed{cfg['seed']}.npy")

    if (os.path.exists(processed) or (pca_dim == 30 and os.path.exists(legacy_processed))) and os.path.exists(gt_path) and os.path.exists(train_mask):
        return

    download_paviau(data_root)
    prepare_processed_data(
        data_root,
        pca_dim=pca_dim,
        train_ratio=float(cfg["data"]["train_ratio"]),
        seed=int(cfg["seed"]),
    )


def _build_backbone(cfg: Dict):
    in_channels = int(cfg["data"]["pca_dim"])
    feature_dim = int(cfg.get("model", {}).get("feature_dim", 512))
    name = cfg.get("model", {}).get("backbone", "resnet18_hsi")
    if name == "hybrid_hsi_lite":
        return hybrid_hsi_lite(in_channels=in_channels, feature_dim=feature_dim)
    return resnet18_hsi(in_channels=in_channels, feature_dim=feature_dim)


def _build_model(cfg: Dict, num_classes: int):
    feature_dim = int(cfg.get("model", {}).get("feature_dim", 512))
    classifier_cfg = cfg.get("classifier", {})
    classifier_type = classifier_cfg.get("type", "linear")
    cosine_scale = float(classifier_cfg.get("cosine_scale", 16.0))
    backbone = _build_backbone(cfg)
    return NetworkHSI(
        num_classes=num_classes,
        feature_extractor=backbone,
        feature_dim=feature_dim,
        classifier_type=classifier_type,
        cosine_scale=cosine_scale,
    )


def evaluate_taskwise_matrix(trainer, data_manager, cfg, exp_name: str):
    task_count = len(data_manager.tasks)
    result_matrix: List[List[float]] = []

    for current_task in range(task_count):
        seen_count = data_manager.get_seen_class_count(current_task)
        ckpt = os.path.join(cfg["save_path"], exp_name, f"{seen_count}_model.pth")
        state = torch.load(ckpt, map_location=trainer.device)

        eval_model = _build_model(cfg, num_classes=seen_count * 4).to(trainer.device)
        eval_model.load_state_dict(state["model_state"])
        eval_model.eval()

        row = []
        for eval_task in range(task_count):
            if eval_task > current_task:
                row.append(0.0)
                continue

            ds = data_manager.get_taskwise_test_dataset(current_task, eval_task)
            dl = DataLoader(
                ds,
                batch_size=int(cfg["batch_size"]),
                shuffle=False,
                drop_last=False,
                num_workers=int(cfg["num_workers"]),
            )

            y_true_list = []
            y_pred_list = []
            with torch.no_grad():
                for images, labels, _, _ in dl:
                    images = images.to(trainer.device)
                    logits, _ = eval_model(images)
                    logits = logits[:, : seen_count * 4]
                    logits = logits[:, ::4]
                    pred = torch.argmax(logits, dim=1)
                    y_true_list.append(labels.numpy())
                    y_pred_list.append(pred.cpu().numpy())

            y_true = np.array([], dtype=np.int64) if len(y_true_list) == 0 else np.concatenate(y_true_list)
            y_pred = np.array([], dtype=np.int64) if len(y_pred_list) == 0 else np.concatenate(y_pred_list)
            metrics = evaluate_all(y_true, y_pred, num_classes=seen_count)
            row.append(metrics["oa"])

        result_matrix.append(row)

    out_path = os.path.join(cfg["output_path"], exp_name, "taskwise_oa_matrix.json")
    ensure_dir(os.path.dirname(out_path))
    save_json(out_path, {"taskwise_oa_matrix": result_matrix})
    return result_matrix


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    ensure_dir(cfg["save_path"])
    ensure_dir(cfg["log_path"])
    ensure_dir(cfg["output_path"])

    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])

    ensure_data_ready(cfg)
    if args.prepare_only:
        print("Data prepared only. Exiting.")
        return

    data_manager = PaviaUDataManager(
        root=cfg["data"]["root"],
        seed=int(cfg["seed"]),
        patch_size=int(cfg["data"]["patch_size"]),
        base_classes=int(cfg["incremental"]["base_classes"]),
        task_num=int(cfg["incremental"]["task_num"]),
        pca_dim=int(cfg["data"]["pca_dim"]),
    )

    base_classes = int(cfg["incremental"]["base_classes"])
    model = _build_model(cfg, num_classes=base_classes * 4)

    trainer = ProtoAugSSLHSI(cfg=cfg, data_manager=data_manager, model=model, device=device)

    exp_name = (
        f"paviau_base{cfg['incremental']['base_classes']}_"
        f"inc{cfg['incremental']['task_num']}_"
        f"pca{cfg['data']['pca_dim']}_seed{cfg['seed']}"
    )
    exp_dir = os.path.join(cfg["output_path"], exp_name)

    task_metrics = {}
    for task_id in range(len(data_manager.tasks)):
        trainer.before_train(task_id)
        trainer.train_task()
        seen_metrics = trainer.evaluate_seen()
        task_metrics[f"task_{task_id}"] = seen_metrics
        trainer.save_task_visualization(exp_dir=exp_dir, task_id=task_id)
        trainer.after_train(exp_name)
        print(
            f"[Task {task_id}] OA={seen_metrics['oa']:.4f}, "
            f"AA={seen_metrics['aa']:.4f}, Kappa={seen_metrics['kappa']:.4f}"
        )

    out_metrics = os.path.join(cfg["output_path"], exp_name, "seen_metrics.json")
    ensure_dir(os.path.dirname(out_metrics))
    save_json(out_metrics, task_metrics)

    taskwise_matrix = evaluate_taskwise_matrix(trainer, data_manager, cfg, exp_name)
    seen_classes = [data_manager.get_seen_class_count(t) for t in range(len(data_manager.tasks))]
    forgetting = generate_reports(exp_dir, task_metrics, taskwise_matrix, seen_classes)
    print(f"Average Forgetting: {forgetting['average_forgetting']:.4f}")
    print("Training and evaluation completed.")


if __name__ == "__main__":
    main()

