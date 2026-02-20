# 项目状态快照（用于论文写作）

更新时间：2026-02-20

## 当前实验主线
- 协议：PaviaU，`5+2+2`
- 目标：提升增量后期任务（最后一个 task）并降低 Average Forgetting
- 当前最佳候选：`configs/paviau_planA_step4b.yaml`

## 关键实验结论

### 1) Step1b vs Step0（通过）
- Step1b 相比 Step0：
  - Task1 指标不降，Task2 明显提升
  - forgetting 下降
- 结论：Step1b 作为后续基线合理

### 2) Step2b（PCA60）vs Step1b（淘汰）
- 观察：Task1/Task2 的 OA、AA 同时下降，forgetting 变差
- 结论：在当前 pipeline 下，`PCA60` 不适合主线
- 决策：主线固定 `PCA30`

### 3) Step3b（Hybrid Backbone）vs Step1b（强通过）
- 两个 seed（1993, 2025）均显示：
  - Task1 OA/AA 大幅上升
  - Task2 OA/AA 显著上升
  - forgetting 进一步下降
- Task2 OA 两 seed 均值已超过 80%

### 4) Step4b（Cosine Classifier + Bias Correction）vs Step3b（通过，当前最优）
- 仅在 step3b 基础上增加 `cosine classifier + bias correction` 后：
  - seed=1993：Task2 `OA=0.8883`，`AA=0.8815`，Average Forgetting `0.0840`
  - seed=2025：Task2 `OA=0.8995`，`AA=0.9035`，Average Forgetting `0.0743`
- 结论：Step4b 作为当前论文主线与后续升级的对照基线

### 5) Step5b（Replay 初版）vs Step4b（暂不通过）
- 初步观察：在 step4b 已较强的情况下，Replay 若参数不匹配可能导致中间 task 指标下降、forgetting 变差。
- 结论：暂不作为主线，需要单变量调参或改进 memory 策略后再评估

## 当前版本建议
- 主线配置：`step4b`
- 任务划分：保持 `5+2+2`
- 维度：保持 `PCA30`
- 下一步：先跑 `step4b` 多种子统计（mean±std），再决定是否继续 Replay（step5b+）

## 训练可视化功能
- 已支持每个 task 输出 GT vs Pred 对比图（原版：GT 为全图已见类别，Pred 为测试像素预测）：
  - `task_X_gt_seen.png`
  - `task_X_pred.png`
  - `task_X_gt_pred_compare.png`
- 已新增对齐版（GT(Test) 与 Pred(Test) 使用同一批测试像素点，便于论文对比）：
  - `task_X_gt_test.png`
  - `task_X_pred_test.png`
  - `task_X_gt_test_pred_test_compare.png`
- 输出目录：`outputs/<实验名>/task_visualizations/`

## 服务器操作提示（网络不稳时）
- 优先拉取：
  - `git fetch origin && git reset --hard origin/main`
- 若 GitHub 超时，可临时手动同步配置文件后继续实验
