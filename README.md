# PASS-HSI 项目说明

## 项目目标
- 任务：高光谱图像类增量学习
- 当前主线方法：
  - 任务感知辅助学习：`spectral3 / ssma`
  - HSI 专属原型增强：`protoaug_hsi`
  - 最新回放机制：`merged raw replay + herding + fixed-budget replay`
- 辅助实现细节：
  - `hybrid_hsi_lite`
  - `cosine classifier`
  - `bias correction`
  - `feature distillation`

## 当前支持数据集
- `PaviaU`
- `Salinas`
- `IndianPines`
- `Houston`

## 当前推荐 replay 配置
- `memory_budget: 480`
- `full_replay_below_train_count: 10`
- `min_memory_per_class: 5`

## 目录结构
```text
./
  main_hsi.py
  preprocess_hsi.py
  dataset_paviau.py
  dataset_replay_hsi.py
  PASS_hsi.py
  replay_selection.py
  configs/
  data/
    PaviaU/
    Salinas/
    IndianPines/
    Houston/
  checkpoints/
  logs/
  outputs/
  experiment_results.txt
```

## 环境安装
```bash
pip install -r requirements.txt
```

## 当前推荐运行命令
```bash
# PaviaU：多类增量主线
python main_hsi.py --config configs/paviau/paviau_planA_step4b_ssma_rawreplay_herding_merged_bestselect_protoaug_hsi_budget480_fullreplay10_min5.yaml --seed 2025 --task_split 5 2 2

# PaviaU：单类增量主线
python main_hsi.py --config configs/paviau/paviau_planA_step4b_spectral3_rawreplay_herding_merged_bestselect_protoaug_hsi_budget480_fullreplay10_min5.yaml --seed 2025 --task_split 7 1 1

# Salinas：多类增量主线
python main_hsi.py --config configs/salinas/salinas_planA_step4b_ssma_rawreplay_herding_merged_bestselect_protoaug_hsi_budget480_fullreplay10_min5.yaml --seed 2025 --task_split 8 2 2 2 2

# IndianPines：多类增量主线
python main_hsi.py --config configs/indianpines/indianpines_planA_step4b_ssma_rawreplay_herding_merged_bestselect_protoaug_hsi_budget480_fullreplay10.yaml --seed 2025 --task_split 8 4 4

# Houston：多类增量主线
python main_hsi.py --config configs/houston/step4b_ssma_rawreplay_herding_merged_bestselect_protoaug_hsi_budget480_fullreplay10_min5.yaml --seed 2025 --task_split 7 2 2 2 2
```

## 关键配置
- `data.root`：数据集目录
- `data.pca_dim`：PCA 维度
- `data.patch_size`：patch 大小
- `incremental.base_classes + incremental.task_num`：默认任务划分
- `--task_split`：运行时覆盖任务划分
- `pass.ssl_mode`
  - `ssma`
  - `spectral3`
- `replay.strategy`
  - `merged`
- `replay.selection`
  - `herding`
- `replay.fixed_budget`
- `replay.memory_budget`
- `replay.full_replay_below_train_count`
- `replay.min_memory_per_class`

## 输出文件
单次实验输出目录：`outputs/<实验名>/`
- `seen_metrics.json`
- `taskwise_oa_matrix.json`
- `summary_metrics.csv`
- `incremental_curves.png`
- `taskwise_heatmap.png`
- `forgetting.json`

## 实验日志
- 根目录统一维护：`experiment_results.txt`
- 每次实验结束后自动追加：
  - 实验时间
  - 配置路径
  - 随机种子
  - 各 task `OA / AA / Kappa`

## 当前代表性结果
- PaviaU `5+2+2 + ssma`
  - Task0 `OA=0.9990`
  - Task1 `OA=0.9874`
  - Task2 `OA=0.9672`
- Salinas `8+2+2+2+2 + ssma`
  - Task0 `OA=1.0000`
  - Task1 `OA=0.9955`
  - Task2 `OA=0.9916`
  - Task3 `OA=0.9907`
  - Task4 `OA=0.9920`
- IndianPines `8+4+4 + ssma`
  - Task0 `OA=0.9991`
  - Task1 `OA=0.9506`
  - Task2 `OA=0.9423`
- Houston `7+2+2+2+2 + ssma`
  - Task0 `OA=0.9985`
  - Task1 `OA=0.9975`
  - Task2 `OA=0.9750`
  - Task3 `OA=0.9276`
  - Task4 `OA=0.9102`
