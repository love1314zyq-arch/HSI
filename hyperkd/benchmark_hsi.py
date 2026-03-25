import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from scipy.io import loadmat, savemat
from torch.utils.data import ConcatDataset, DataLoader

from benchmarks.visualizations import save_task_prediction_artifact

RAW_CUBE_FILES = {
    "paviau": "data/PaviaU/raw/PaviaU.mat",
    "salinas": "data/Salinas/raw/Salinas_corrected.mat",
    "houston": "data/Houston/raw/Houston.mat",
    "indianpines": "data/IndianPines/raw/Indian_pines_corrected.mat",
}

PROCESSED_FILES = {
    "paviau": {
        "gt": "data/PaviaU/processed/gt.npy",
        "train_mask": "data/PaviaU/processed/train_mask_seed{seed}.npy",
        "test_mask": "data/PaviaU/processed/test_mask_seed{seed}.npy",
    },
    "salinas": {
        "gt": "data/Salinas/processed/gt.npy",
        "train_mask": "data/Salinas/processed/train_mask_seed{seed}.npy",
        "test_mask": "data/Salinas/processed/test_mask_seed{seed}.npy",
    },
    "houston": {
        "gt": "data/Houston/processed/gt.npy",
        "train_mask": "data/Houston/processed/train_mask_seed{seed}.npy",
        "test_mask": "data/Houston/processed/test_mask_seed{seed}.npy",
    },
    "indianpines": {
        "gt": "data/IndianPines/processed/gt.npy",
        "train_mask": "data/IndianPines/processed/train_mask_seed{seed}.npy",
        "test_mask": "data/IndianPines/processed/test_mask_seed{seed}.npy",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark wrapper for HyperKD on single-dataset HSI class-incremental scenarios.")
    parser.add_argument("--dataset", required=True, choices=sorted(RAW_CUBE_FILES.keys()))
    parser.add_argument("--task-split", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patches", type=int, default=7)
    parser.add_argument("--band-patches", type=int, default=3)
    parser.add_argument("--nepochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num-exemplars-per-class", type=int, default=5)
    parser.add_argument("--exemplar-selection", type=str, default="ssgd")
    parser.add_argument("--eval-on-train", action="store_true")
    parser.add_argument("--results-name", default="hyperkd_benchmark")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def first_mat_key(path: str):
    data = loadmat(path)
    for key in data.keys():
        if not key.startswith("__"):
            return data[key]
    raise KeyError(f"No matlab data key found in {path}")


def task_classes_from_split(task_split):
    classes = []
    start = 0
    for size in task_split:
        stop = start + int(size)
        classes.append(list(range(start, stop)))
        start = stop
    return classes


def build_task_mat(cube, gt, train_mask, test_mask, class_ids, out_path):
    tr = np.zeros(gt.shape, dtype=np.int32)
    te = np.zeros(gt.shape, dtype=np.int32)
    class_to_local = {cls_id: idx + 1 for idx, cls_id in enumerate(class_ids)}

    for cls_id, local_label in class_to_local.items():
        cls_mask = gt == cls_id
        tr[np.logical_and(cls_mask, train_mask)] = local_label
        te[np.logical_and(cls_mask, test_mask)] = local_label

    savemat(out_path, {"input": cube, "TR": tr, "TE": te})
    return int(len(class_ids))


def task_test_coords(gt, test_mask, class_ids):
    coords = []
    for cls_id in class_ids:
        cls_coords = np.argwhere(np.logical_and(gt == cls_id, test_mask))
        coords.append(cls_coords.astype(np.int64))
    if not coords:
        return np.empty((0, 2), dtype=np.int64)
    return np.concatenate(coords, axis=0)


def weighted_stage_metrics(stage_results):
    total = sum(item["n_test"] for item in stage_results)
    if total <= 0:
        return {"oa": None, "aa": None, "kappa": None}
    return {
        "oa": float(sum(item["oa"] * item["n_test"] for item in stage_results) / total),
        "aa": float(sum(item["aa"] * item["n_test"] for item in stage_results) / total),
        "kappa": float(sum(item["kappa"] * item["n_test"] for item in stage_results) / total),
    }


def compute_average_forgetting(acc_matrix):
    if len(acc_matrix) <= 1:
        return 0.0
    last = len(acc_matrix) - 1
    forgetting = []
    for task_id in range(last):
        best_before_last = max(acc_matrix[t][task_id] for t in range(task_id, last))
        forgetting.append(best_before_last - acc_matrix[last][task_id])
    return float(np.mean(forgetting)) if forgetting else 0.0


def build_seen_loader(loaders, upto_task, batch_size):
    class SeenConcatDataset(ConcatDataset):
        def __init__(self, datasets):
            super().__init__(datasets)
            self.transform = getattr(datasets[0], "transform", None) if datasets else None

    datasets = [loaders[idx].dataset for idx in range(upto_task + 1)]
    merged_dataset = SeenConcatDataset(datasets)
    return DataLoader(
        merged_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def main():
    args = parse_args()
    from approach.our import Appr
    from datasets.exemplars_dataset import ExemplarsDataset
    from datasets.my_dataset import get_loader
    from networks.LSGA_ViT import LSGAVIT
    from networks.network import LLL_Net

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).resolve()
    temp_root = output_root / "temp_mats"
    ensure_dir(str(output_root))
    ensure_dir(str(temp_root))

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    print(f"[hyperkd_benchmark] dataset={args.dataset} split={args.task_split} seed={args.seed} device={device}")

    cube = first_mat_key(str(repo_root / RAW_CUBE_FILES[args.dataset]))
    paths = PROCESSED_FILES[args.dataset]
    gt = np.load(repo_root / paths["gt"])
    train_mask = np.load(repo_root / paths["train_mask"].format(seed=args.seed)).astype(bool)
    test_mask = np.load(repo_root / paths["test_mask"].format(seed=args.seed)).astype(bool)

    task_classes = task_classes_from_split(args.task_split)
    trn_loader = []
    tst_loader = []
    taskcla = []
    bands = []
    test_coords = []
    label_offset = 0

    for task_id, class_ids in enumerate(task_classes):
        mat_path = temp_root / f"task_{task_id}.mat"
        num_classes = build_task_mat(cube, gt, train_mask, test_mask, class_ids, str(mat_path))
        train_loader, test_loader, _, band = get_loader(
            str(mat_path),
            args.batch_size,
            args.patches,
            args.band_patches,
            is_shuffle=True,
            tsk_offset=label_offset,
        )
        test_loader = DataLoader(
            test_loader.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        trn_loader.append(train_loader)
        tst_loader.append(test_loader)
        taskcla.append(num_classes)
        bands.append(band)
        test_coords.append(task_test_coords(gt, test_mask, class_ids))
        label_offset += num_classes
        print(f"[hyperkd_benchmark] task={task_id} classes={class_ids} train={len(train_loader.dataset)} test={len(test_loader.dataset)}")

    init_model = LSGAVIT(
        img_size=args.patches,
        patch_size=3,
        in_chans=36,
        num_classes=taskcla[0],
        embed_dim=120,
        depths=[2],
        num_heads=[12, 12, 12, 24],
    )
    model = LLL_Net(init_model)
    appr_kwargs = dict(
        nepochs=args.nepochs,
        lr=args.lr,
        lr_min=1e-8,
        lr_factor=1 / 0.9,
        lr_patience=5,
        clipgrad=10000,
        momentum=0.0,
        wd=0.0,
        multi_softmax=False,
        wu_nepochs=0,
        wu_lr_factor=1.0,
        fix_bn=False,
        eval_on_train=args.eval_on_train,
        logger=None,
        exemplars_dataset=ExemplarsDataset(
            None,
            None,
            num_exemplars=0,
            num_exemplars_per_class=args.num_exemplars_per_class,
            exemplar_selection=args.exemplar_selection,
        ),
    )
    appr = Appr(model, device, **appr_kwargs)
    appr.save_model_dir = str(output_root / "models")
    ensure_dir(appr.save_model_dir)

    max_task = len(taskcla)
    acc_tag = np.zeros((max_task, max_task), dtype=np.float32)
    stage_metrics = []

    for t, ncla in enumerate(taskcla):
        print("*" * 80)
        print(f"[hyperkd_benchmark] task={t}/{max_task-1} num_classes={ncla}")
        print("*" * 80)
        appr.model.add_head(taskcla[t])
        appr.model.to(device)
        seen_val_loader = build_seen_loader(tst_loader, t, args.batch_size)
        appr.train(t, trn_loader[t], seen_val_loader)

        current_stage = []
        stage_rows = []
        stage_cols = []
        stage_preds = []
        stage_labels = []
        for u in range(t + 1):
            test_loss, acc_taw_u, acc_tag_u, oa_tar, aa_tar, kappa_tar, aa_each_tar, oa_tag, aa_tag, kappa_tag, aa_each_tag = appr.eval(u, tst_loader[u])
            acc_tag[t, u] = float(acc_tag_u)
            coords_u = test_coords[u]
            preds_u = np.asarray(getattr(appr, "_last_eval_preds_tag", np.asarray([], dtype=np.int64)), dtype=np.int64)
            labels_u = np.asarray(getattr(appr, "_last_eval_targets", np.asarray([], dtype=np.int64)), dtype=np.int64)
            if len(coords_u) == len(preds_u):
                stage_rows.append(coords_u[:, 0])
                stage_cols.append(coords_u[:, 1])
                stage_preds.append(preds_u)
                stage_labels.append(labels_u)
            current_stage.append(
                {
                    "task_id": u,
                    "n_test": int(len(tst_loader[u].dataset)),
                    "oa": float(oa_tag),
                    "aa": float(aa_tag),
                    "kappa": float(kappa_tag),
                    "acc_tag": float(acc_tag_u),
                    "loss": float(test_loss),
                }
            )
            print(
                f"[hyperkd_benchmark] eval_task={u} "
                f"loss={test_loss:.4f} acc_tag={acc_tag_u:.4f} oa={oa_tag:.4f} aa={aa_tag:.4f} kappa={kappa_tag:.4f}"
            )

        seen_test_loss, seen_acc_taw, seen_acc_tag, seen_oa_tar, seen_aa_tar, seen_kappa_tar, _, seen_oa_tag, seen_aa_tag, seen_kappa_tag, _ = appr.eval(t, seen_val_loader)
        stage_metrics.append(
            {
                "task_id": t,
                "seen_classes": int(sum(args.task_split[: t + 1])),
                "n_test": int(len(seen_val_loader.dataset)),
                "oa": float(seen_oa_tag),
                "aa": float(seen_aa_tag),
                "kappa": None if np.isnan(seen_kappa_tag) else float(seen_kappa_tag),
                "acc_tag": float(seen_acc_tag),
                "loss": float(seen_test_loss),
                "per_task": current_stage,
            }
        )
        if stage_rows:
            save_task_prediction_artifact(
                str(output_root),
                t,
                list(range(sum(args.task_split[: t + 1]))),
                np.concatenate(stage_rows, axis=0),
                np.concatenate(stage_cols, axis=0),
                np.concatenate(stage_preds, axis=0),
                np.concatenate(stage_labels, axis=0),
            )
        print(
            f"[hyperkd_benchmark] seen_eval "
            f"task={t} loss={seen_test_loss:.4f} acc_tag={seen_acc_tag:.4f} "
            f"oa={seen_oa_tag:.4f} aa={seen_aa_tag:.4f} "
            f"kappa={'nan' if np.isnan(seen_kappa_tag) else f'{seen_kappa_tag:.4f}'}"
        )

    result = {
        "algorithm": "hyperkd",
        "dataset": args.dataset,
        "seed": int(args.seed),
        "task_split": [int(x) for x in args.task_split],
        "seen_metrics": [
            {
                "task_id": item["task_id"],
                "seen_classes": item["seen_classes"],
                "oa": item["oa"],
                "aa": item["aa"],
                "kappa": item["kappa"],
            }
            for item in stage_metrics
        ],
        "average_forgetting": compute_average_forgetting(acc_tag.tolist()),
        "raw_stage_metrics": stage_metrics,
    }
    with open(output_root / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[hyperkd_benchmark] result saved to {output_root / 'result.json'}")


if __name__ == "__main__":
    main()
