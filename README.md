# PASS-HSI 项目说明（PaviaU）

## 1. 项目目标
- 任务：高光谱图像增量分类
- 数据集：University of Pavia (PaviaU)
- 方法：PASS（原型增强 + 自监督 + 特征蒸馏）
- 指标：OA / AA / Kappa

说明：当前仓库根目录即项目根目录，不再使用旧的 `HSI2/` 前缀。

## 2. 目录结构
```text
./
  main_hsi.py
  run_multi_seed.py
  preprocess_hsi.py
  dataset_paviau.py
  PASS_hsi.py
  configs/
    paviau_default.yaml
  data/
    PaviaU/
      raw/
      processed/
      splits/
      metadata/
  checkpoints/
  logs/
  outputs/
```

## 3. 环境安装
```bash
pip install -r requirements.txt
```

## 4. 运行命令
```bash
# 仅准备数据
python main_hsi.py --prepare_only

# 训练 + 评估
python main_hsi.py

# 多随机种子实验（默认 1993 2025 3407）
python run_multi_seed.py
```

## 5. 关键配置
配置文件：`configs/paviau_default.yaml`
- `save_path: checkpoints`
- `log_path: logs`
- `output_path: outputs`
- `data.root: data/PaviaU`
- `data.pca_dim`: PCA 维度（对应处理后文件 `pca{pca_dim}_cube.npy`）

## 6. 输出说明
单次实验输出目录：`outputs/<实验名>/`
- `seen_metrics.json`
- `taskwise_oa_matrix.json`
- `taskwise_oa_matrix.csv`
- `summary_metrics.csv`
- `incremental_curves.png`
- `taskwise_heatmap.png`
- `forgetting.json`

多种子汇总输出：`outputs/multi_seed_summary/`
- `multi_seed_raw.json`
- `multi_seed_summary.json`
- `multi_seed_summary.csv`