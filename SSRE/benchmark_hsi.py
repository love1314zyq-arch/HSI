import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from utils.Trainer_cvpr import Trainer


DATASET_NAMES = {"paviau": "PaviaU", "salinas": "Salinas", "houston": "Houston", "indianpines": "IndianPines"}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark wrapper for SSRE on HSI class-incremental scenarios.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_NAMES.keys()))
    parser.add_argument("--task-split", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--patches", type=int, default=1)
    parser.add_argument("--pca-dim", type=int, default=30)
    parser.add_argument("--base-epochs", type=int, default=60)
    parser.add_argument("--new-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batch-size-test", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-new", type=float, default=0.0002)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    return parser.parse_args()


def build_args(cli_args):
    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(cli_args.output_root).resolve()
    log_root = output_root / "logs"
    os.makedirs(log_root, exist_ok=True)

    args = argparse.Namespace()
    args.data_dir = str(repo_root / "data")
    args.dataset_name = DATASET_NAMES[cli_args.dataset]
    args.dataset_type = "hsi_cl"
    args.loader_type = "normal"
    args.base_class = int(cli_args.task_split[0])
    args.way = int(cli_args.task_split[1]) if len(cli_args.task_split) > 1 else int(cli_args.task_split[0])
    args.shot = 0
    args.orders = None
    args.orders_path = ""
    args.no_order = True
    args.model_name = "my"
    args.model_path = None
    args.pretrained = False
    args.backbone_name = "resnet18_no1"
    args.init_fic = "None"
    args.batch_task = 3
    args.embedding = 64
    args.latent_dim = 512
    args.classifier = "fc_IL_base"
    args.mode = "normal"
    args.base_train = "base"
    args.no_cuda = cli_args.device == "cpu"
    args.cudnn = True
    args.gpu_ids = cli_args.device.split(":", 1)[1] if cli_args.device.startswith("cuda:") else "0"
    args.loss_type = "ce"
    args.loss_weight = 1
    args.seed = int(cli_args.seed)
    args.session = len(cli_args.task_split)
    args.base_epochs = int(cli_args.base_epochs)
    args.new_epochs = int(cli_args.new_epochs)
    args.batch_size = int(cli_args.batch_size)
    args.batch_size_test = int(cli_args.batch_size_test)
    args.lr = float(cli_args.lr)
    args.lr_new = float(cli_args.lr_new)
    args.lr_scheduler = "step"
    args.no_lr_scheduler = False
    args.lr_coefficient = [1, 1, 1, 1]
    args.momentum = 0.9
    args.weight_decay = float(cli_args.weight_decay)
    args.nesterov = False
    args.opt = "opt1"
    args.optim = "adam"
    args.eval_interval = 1
    args.val = False
    args.checkname = "ssre_hsi"
    args.alpha = 0.75
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.data_path = os.path.join(args.data_dir, args.dataset_name)
    args.tasks = args.session - 1
    args.all_class = sum(int(x) for x in cli_args.task_split)
    args.dir_name = str(log_root)
    args.output_root = str(output_root)
    args.now_time = ""
    args.task_sizes = [int(x) for x in cli_args.task_split]
    args.patch_size = int(cli_args.patches)
    args.patches = int(cli_args.patches)
    args.pca_dim = int(cli_args.pca_dim)
    args.num_workers = 0
    args.hsi_input_mode = "pixel"
    return args


def main():
    args_cli = parse_args()
    print("[ssre_benchmark] parse args finished")
    args = build_args(args_cli)
    output_root = Path(args_cli.output_root).resolve()
    os.makedirs(output_root, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print(
        f"[ssre_benchmark] dataset={args_cli.dataset} | task_split={args.task_sizes} | "
        f"seed={args.seed} | device={args_cli.device} | output_root={output_root}"
    )
    print(
        f"[ssre_benchmark] backbone={args.backbone_name} | classifier={args.classifier} | "
        f"patches={args.patches} | pca_dim={args.pca_dim} | input_mode={args.hsi_input_mode} | "
        f"base_epochs={args.base_epochs} | new_epochs={args.new_epochs}"
    )

    print("[ssre_benchmark] building trainer")
    trainer = Trainer(args)
    print("[ssre_benchmark] trainer ready")
    for session in range(args.session):
        print(f"[ssre_benchmark] ===== start session {session + 1}/{args.session} =====")
        trainer.training(session)
        trainer.validation(session, print=True)
        print(f"[ssre_benchmark] ===== end session {session + 1}/{args.session} =====")

    result = {
        "algorithm": "ssre",
        "dataset": args_cli.dataset,
        "seed": int(args.seed),
        "task_split": args.task_sizes,
        "seen_metrics": trainer.stage_metrics,
        "average_forgetting": trainer.average_forgetting(),
    }
    with open(output_root / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[ssre_benchmark] result saved to {output_root / 'result.json'}")


if __name__ == "__main__":
    main()
