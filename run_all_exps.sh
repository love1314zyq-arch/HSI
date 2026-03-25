#!/bin/bash

echo "========================================================="
echo " 开始执行 HSI 实验队列 (共 19 组) - $(date)"
echo "========================================================="

echo "[1/19] 运行 dataset: paviau | task-split: 8 1"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset paviau --task-split 8 1 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[2/19] 运行 dataset: paviau | task-split: 7 1 1"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset paviau --task-split 7 1 1 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[3/19] 运行 dataset: salinas | task-split: 15 1"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset salinas --task-split 15 1 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[4/19] 运行 dataset: salinas | task-split: 14 1 1"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset salinas --task-split 14 1 1 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[5/19] 运行 dataset: indianpines | task-split: 15 1"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset indianpines --task-split 15 1 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[6/19] 运行 dataset: indianpines | task-split: 14 1 1"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset indianpines --task-split 14 1 1 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[7/19] 运行 dataset: houston | task-split: 14 1"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset houston --task-split 14 1 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[8/19] 运行 dataset: houston | task-split: 13 1 1"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset houston --task-split 13 1 1 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[9/19] 运行 dataset: houston | task-split: 7 4 4"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset houston --task-split 7 4 4 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[10/19] 运行 dataset: houston | task-split: 7 2 2 2 2"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset houston --task-split 7 2 2 2 2 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[11/19] 运行 dataset: houston | task-split: 3 4 4 4"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset houston --task-split 3 4 4 4 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[12/19] 运行 dataset: paviau | task-split: 5 2 2"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset paviau --task-split 5 2 2 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[13/19] 运行 dataset: paviau | task-split: 3 3 3"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset paviau --task-split 3 3 3 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[14/19] 运行 dataset: salinas | task-split: 8 4 4"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset salinas --task-split 8 4 4 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[15/19] 运行 dataset: salinas | task-split: 8 2 2 2 2"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset salinas --task-split 8 2 2 2 2 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[16/19] 运行 dataset: salinas | task-split: 4 4 4 4"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset salinas --task-split 4 4 4 4 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[17/19] 运行 dataset: indianpines | task-split: 8 4 4"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset indianpines --task-split 8 4 4 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[18/19] 运行 dataset: indianpines | task-split: 8 2 2 2 2"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset indianpines --task-split 8 2 2 2 2 --seeds 1993 2025 3407
echo "---------------------------------------------------------"

echo "[19/19] 运行 dataset: indianpines | task-split: 4 4 4 4"
/home/zyq/miniconda3/envs/hsi/bin/python -m benchmarks.runner --dataset indianpines --task-split 4 4 4 4 --seeds 1993 2025 3407
echo "========================================================="
echo " 所有实验组执行完毕！辛苦了！ - $(date)"
echo "========================================================="

	
