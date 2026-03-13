# PASS-HSI 项目说明

## 1. 项目目标
- 任务：高光谱图像增量分类
- 方法：PASS（原型增强 + 自监督 + 特征蒸馏）
- 当前支持数据集：
  - `PaviaU`
  - `Salinas`
- 主要指标：`OA / AA / Kappa / Average Forgetting`

## 2. 当前主线
- PaviaU 论文主线配置：`configs/paviau_planA_step4b.yaml`
- PaviaU 3 光谱版本：`configs/paviau_planA_step4b_spectral3.yaml`
- Salinas 基线配置：`configs/salinas_planA_step4b.yaml`
- Salinas 3 光谱版本：`configs/salinas_planA_step4b_spectral3.yaml`

当前更稳的主线仍是 PaviaU `5+2+2`；Salinas 已完成接入并已有首轮结果，但还处在补实验阶段。

## 3. 目录结构
```text
./
  main_hsi.py
  run_multi_seed.py
  preprocess_hsi.py
  dataset_paviau.py
  PASS_hsi.py
  report_hsi.py
  task_visualize_hsi.py
  configs/
  data/
    PaviaU/
    Salinas/
  checkpoints/
  logs/
  outputs/
  experiment_results.txt
```

## 4. 环境安装
```bash
pip install -r requirements.txt
```

## 5. 运行命令
```bash
# 仅准备数据
python main_hsi.py --prepare_only

# PaviaU：默认主线 step4b（5+2+2）
python main_hsi.py --config configs/paviau_planA_step4b.yaml --seed 1993

# PaviaU：3 光谱版本 + 自定义任务划分
python main_hsi.py --config configs/paviau_planA_step4b_spectral3.yaml --seed 2025 --task_split 8 1

# Salinas：rotation4 版本
python main_hsi.py --config configs/salinas_planA_step4b.yaml --seed 2025 --task_split 15 1

# Salinas：3 光谱版本
python main_hsi.py --config configs/salinas_planA_step4b_spectral3.yaml --seed 2025 --task_split 14 1 1

# 多随机种子汇总
python run_multi_seed.py --config configs/paviau_planA_step4b.yaml --seeds 1993 2025 3407 4242 6666
```

说明：
- `--task_split` 后面的数字必须和命令写在同一条 shell 命令里。
- 如果要换行，上一行末尾必须写 `\`，且 `\` 后不能有空格。

## 6. 关键配置
- `data.root`：数据集根目录，决定使用 `PaviaU` 还是 `Salinas`
- `data.pca_dim`：PCA 维度，对应 `processed/pca{pca_dim}_cube.npy`
- `data.patch_size`：输入 patch 大小
- `incremental.base_classes + incremental.task_num`：默认任务划分
- `--task_split`：运行时显式覆盖任务划分
- `pass.ssl_mode`
  - 空 / `use_rotation_ssl: true`：rotation4
  - `spectral3`：3 光谱增强版本

## 7. 输出说明
单次实验输出目录：`outputs/<实验名>/`
- `seen_metrics.json`
- `taskwise_oa_matrix.json`
- `taskwise_oa_matrix.csv`
- `summary_metrics.csv`
- `incremental_curves.png`
- `taskwise_heatmap.png`
- `forgetting.json`
- `task_visualizations/`（当前主要用于 PaviaU）

## 8. 固定实验日志
项目根目录下维护一个持续追加的汇总文本：
- `experiment_results.txt`

每次运行结束后会自动追加：
- 实验时间
- `exp_name`
- 配置文件路径
- `seed`
- 每个 task 的类别数
- 每个 task 最终测试结果：`OA / AA / Kappa`

这个文件是跨实验累计记录，不属于某次单独实验目录。
