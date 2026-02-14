import argparse
import subprocess
import sys


STEP_CONFIGS = [
    "configs/paviau_planA_step0.yaml",
    "configs/paviau_planA_step1.yaml",
    "configs/paviau_planA_step2.yaml",
    "configs/paviau_planA_step3.yaml",
    "configs/paviau_planA_step4.yaml",
    "configs/paviau_planA_step5.yaml",
    "configs/paviau_planA_step6.yaml",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run controlled-variable Plan A experiments")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--steps", type=int, nargs="+", default=list(range(0, 7)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1993, 2025])
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for step in args.steps:
        cfg = STEP_CONFIGS[step]
        for seed in args.seeds:
            cmd = [args.python, "main_hsi.py", "--config", cfg, "--seed", str(seed)]
            if args.prepare_only:
                cmd.append("--prepare_only")
            print("[run]", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
