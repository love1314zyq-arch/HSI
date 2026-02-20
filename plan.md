# Plan A 当前执行进度（2026-02-20）

## 1. 执行原则
- 控制变量：每一步只改一类因素。
- 对照固定：始终保留 `step0` 作为基线对照。
- 验收维度：同时看 Task1/Task2 的 OA、AA 和 Average Forgetting。

## 2. 各 Step 定义
- `step0`：baseline（PCA30 + ResNet + linear + Adam/Step，无回放）
- `step1`：原始训练配方升级版（已弃用）
- `step1b`：当前训练配方通过版（AdamW+Cosine, lr=5e-4, epoch=60）
- `step2b`：基于 step1b 只改 `pca_dim: 30 -> 60`
- `step3b`：基于 step1b 只改 `backbone: resnet18_hsi -> hybrid_hsi_lite`
- `step4b`：基于 step3b，仅加 `cosine classifier + bias correction`（已通过，当前最优）
- `step5b`：基于 step4b，加入 Replay 初版（已试跑，暂不通过，需要调参/改 memory）
- `step6`：计划中（更强回放/增强/去偏置的系统性升级）

## 3. 已完成结果判定

### 3.1 step1（淘汰）
- 现象：Task2 局部有提升，但 Task1 稳定性与 forgetting 不理想。
- 结论：不通过，改为 step1b。

### 3.2 step1b（通过）
- 与 step0 比较：Task1/Task2 OA 与 AA 均改善，forgetting 明显下降。
- 结论：通过，作为后续新起点。

### 3.3 step2b（淘汰）
- 仅改 PCA60 后，Task1/Task2 OA、AA 均值均下降，forgetting 恶化。
- 结论：不通过，主线恢复 PCA30。

### 3.4 step3b（强通过）
- 仅改 backbone 为 hybrid_hsi_lite 后：
  - Task1 和 Task2 指标显著提升
  - Task2 OA 两 seed 均值已超过 80%
  - forgetting 进一步下降
- 结论：强通过，进入下一步 step4b。

### 3.5 step4b（通过，当前最优）
- 在 step3b 上增加 `cosine classifier + bias correction` 后：
  - 最后一个 task（Task2）显著提升
  - Average Forgetting 进一步下降
- 结论：通过，作为当前论文主线与后续升级对照基线。

### 3.6 step5b（暂不通过）
- 在 step4b 上加入 Replay 初版后：
  - 观察到部分 seed 上中间 task 指标可能下降，forgetting 可能变差
- 结论：不纳入主线，后续以“单变量”方式调参或改 memory 策略后再评估。

## 4. 当前主线结论
- 任务划分主线保持 `5+2+2`。
- 输入维度主线保持 `PCA30`。
- 当前最优版本：`step4b`。

## 5. 下一步
1. 跑 `step4b` 多种子统计（mean±std），确认稳定性
2. 若最终 task 仍需提升，再对 `step5b` 做“单变量”Replay 调参（先调 `lambda_replay`、`memory_per_class`，再考虑 memory 选择策略）

## 6. 可视化功能进度
- 已新增 `visualize_paviau_gt.py`：输出三联图（伪彩色、真值、图例）。
- 已接入主程序：每个 task 评估后自动生成
  - `task_X_gt_seen.png`
  - `task_X_pred.png`
  - `task_X_gt_pred_compare.png`
  - `task_X_gt_test.png`（对齐版：仅测试像素）
  - `task_X_pred_test.png`（对齐版：仅测试像素）
  - `task_X_gt_test_pred_test_compare.png`（对齐版对比图）
  保存目录：`outputs/<实验名>/task_visualizations/`
