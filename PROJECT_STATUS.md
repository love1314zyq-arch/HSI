# 项目状态快照（用于论文写作）

更新时间：2026-03-10

## 一句话现状
- 项目已从“仅支持 PaviaU 的 PASS-HSI 增量分类实验仓库”扩展为“支持多高光谱数据集的统一训练/预处理框架”。
- 当前论文主线结论仍然主要来自 PaviaU；新增的 Salinas 已完成数据接入、配置接入和主流程适配，但仓库内尚未看到 Salinas 的完整训练结果沉淀。

## 当前项目主线
- 主任务仍是高光谱图像增量分类（PASS: prototype augmentation + self-supervision + distillation）。
- 当前最稳定、已有结果支撑的主线仍是 PaviaU 上的 `step4b`。
- 当前代码主流程已不再把数据集写死为 PaviaU，而是根据 `data.root` 自动识别数据集并复用同一套训练/评估逻辑。

## 数据集支持现状

### 1) PaviaU（主实验数据集）
- 状态：完整可用，已有预处理结果、split 文件、checkpoint 和输出结果。
- 当前主线协议：`5+2+2`
- 当前主线 PCA：`30`
- 现有主要配置：
  - `configs/paviau_default.yaml`
  - `configs/paviau_planA_step4b.yaml`
- 仓库中已存在多组 PaviaU 结果，包括：
  - `outputs/planA_step4b/` 下的多 seed 结果
  - `checkpoints/planA_step4b/` 下的对应模型
- 额外已做过的探索：
  - 不同 PCA 维度：20 / 25 / 30 / 35 / 40 / 60
  - 不同任务划分：`5+2+2`、`7+2`、`8+1`

### 2) Salinas（新接入数据集）
- 状态：已接入代码主流程。
- 当前已落地内容：
  - 原始数据已放入 `data/Salinas/raw/`
  - 已新增专用配置：`configs/salinas_planA_step4b.yaml`
  - 预处理脚本已支持 Salinas 下载、读取、标签映射、PCA、train/test split、metadata 生成
  - 训练入口 `main_hsi.py` 已可基于 `data.root: data/Salinas` 运行
  - 多 seed 聚合脚本 `run_multi_seed.py` 已兼容 Salinas 实验命名
- 当前配置协议：
  - 总类别数：16
  - 任务划分：`8+4+4`
  - PCA：`30`
  - backbone：`hybrid_hsi_lite`
  - classifier：`cosine`
  - bias correction：开启
- 当前边界：
  - 仓库中尚未看到 Salinas 的 `processed/`、`splits/`、`metadata/` 成果文件
  - 仓库中尚未看到 Salinas 的完整训练输出或 checkpoint
  - 目前仅看到一个数据集总览图：`outputs/visualizations/salinas_dataset_overview.png`

## 代码结构更新（相对旧版 PROJECT_STATUS）

### 1) 预处理已泛化为多数据集
- `preprocess_hsi.py` 现在通过 `DATASET_SPECS` 统一管理数据集信息，而不是只写死 PaviaU。
- 当前已内置两个数据集规格：
  - `PaviaU`
  - `Salinas`
- 已支持的泛化能力：
  - 根据根目录自动推断数据集名
  - 按数据集选择 `.mat` 文件名和 key
  - 自动读取类别名
  - 按真实类别数生成 metadata，而不是固定 9 类

### 2) 训练入口已按数据集自动命名与适配
- `main_hsi.py` 已改为调用 `download_dataset()` 与 `infer_dataset_name()`。
- 实验命名已加入数据集前缀，便于后续并行维护 PaviaU / Salinas 结果。
- 可视化输出目前仍只对 PaviaU 启用：
  - `if dataset_name == "PaviaU": trainer.save_task_visualization(...)`
- 含义：主训练与评估流程已支持 Salinas，但 task 级可视化仍保留为 PaviaU 专用逻辑。

### 3) DataManager 不再隐含固定 `5+2+2`
- `dataset_paviau.py` 中的数据管理器虽然类名仍叫 `PaviaUDataManager`，但实际已支持通用任务划分。
- 当前支持两种方式：
  - 通过 `base_classes + task_num` 自动均分剩余类别
  - 通过 `task_split` 显式指定划分
- 这也是 Salinas 能直接使用 `8+4+4` 的基础。

### 4) 多 seed 汇总已兼容多数据集
- `run_multi_seed.py` 现已根据 `data.root` 推断数据集名生成实验目录。
- 这意味着后续可以分别对 PaviaU 和 Salinas 做独立的多 seed 聚合，而不会再共享旧的 `paviau_*` 命名。

## PaviaU 现有实验结论（仍然有效）

### 1) Step1b vs Step0（通过）
- Step1b 相比 Step0：
  - Task1 指标不降，Task2 明显提升
  - forgetting 下降
- 结论：Step1b 作为后续基线合理

### 2) Step2b（PCA60）vs Step1b（淘汰）
- 观察：Task1/Task2 的 OA、AA 同时下降，forgetting 变差
- 结论：在当前 pipeline 下，`PCA60` 不适合主线
- 决策：PaviaU 主线固定 `PCA30`

### 3) Step3b（Hybrid Backbone）vs Step1b（强通过）
- 两个 seed（1993, 2025）均显示：
  - Task1 OA/AA 大幅上升
  - Task2 OA/AA 显著上升
  - forgetting 进一步下降
- Task2 OA 两 seed 均值已超过 80%

### 4) Step4b（Cosine Classifier + Bias Correction）vs Step3b（通过，当前最优）
- 在 step3b 基础上增加 `cosine classifier + bias correction` 后：
  - seed=1993：Task2 `OA=0.8883`，`AA=0.8815`，Average Forgetting `0.0840`
  - seed=2025：Task2 `OA=0.8995`，`AA=0.9035`，Average Forgetting `0.0743`
- 结论：Step4b 仍是当前论文主线与后续升级的对照基线

### 5) Step5b（Replay 初版）vs Step4b（暂不通过）
- 初步观察：在 step4b 已较强的情况下，Replay 若参数不匹配可能导致中间 task 指标下降、forgetting 变差
- 结论：暂不作为主线，需要单变量调参或改进 memory 策略后再评估

## 当前版本建议
- PaviaU 论文主线继续使用：`configs/paviau_planA_step4b.yaml`
- PaviaU 主推荐协议：`5+2+2`, `PCA30`
- Salinas 当前建议定位：先作为“代码已适配完成、待系统实验验证”的第二数据集分支，不要在论文文字里写成“已经完成主结果复现”
- 若下一步要补论文现状，优先顺序建议：
  - 先补 Salinas 的完整预处理产物与首轮单 seed 结果
  - 再跑 Salinas 多 seed 汇总
  - 最后再决定是否把 Replay / memory 策略一并迁移到 Salinas

## 可视化与输出能力
- PaviaU 已支持每个 task 输出 GT vs Pred 对比图：
  - `task_X_gt_seen.png`
  - `task_X_pred.png`
  - `task_X_gt_pred_compare.png`
- PaviaU 已支持测试像素对齐版对比图：
  - `task_X_gt_test.png`
  - `task_X_pred_test.png`
  - `task_X_gt_test_pred_test_compare.png`
- 输出目录：`outputs/<实验名>/task_visualizations/`
- 当前说明：
  - 这套 task 可视化逻辑目前只在 PaviaU 训练流程中自动调用
  - Salinas 如需同等可视化，还需要单独确认颜色映射与显示逻辑是否泛化完成

## 当前需要注意的问题
- `PROJECT_STATUS.md` 已更新，但 `README.md` 仍明显偏向 “PaviaU-only” 叙述，后续建议一并更新。
- `dataset_paviau.py` 的类名仍为 `PaviaUDataManager`，与其实际“多数据集通用”职责不完全一致；这不影响功能，但会影响代码语义清晰度。
- 当前仓库里历史输出目录同时存在旧命名（如 `paviau_base5_inc2_*`）和新命名逻辑（代码现已按 `dataset + split` 命名），写论文或整理结果时要注意区分历史结果与当前命名规则。
