# Plan A 分步说明（当前执行版）

## 总原则
- 控制变量：每一步只改一类因素。
- 对照基线：始终保留 `step0` 作为对照。
- 验收维度：同时看 `Task1/Task2` 的 `OA + AA`，并看 `forgetting`。

## Step0（基线对照）
- 配置文件：`configs/paviau_planA_step0.yaml`
- 任务划分：`5+2+2`
- 作用：作为后续所有改动的参照，不做算法升级。
- 关键特征：
  - `PCA30`
  - `ResNet`
  - `linear classifier`
  - `Adam + StepLR`
  - 无回放、无偏置校准

## Step1（原始版，已被 step1b 替代）
- 配置文件：`configs/paviau_planA_step1.yaml`
- 作用：只改训练配方（优化器与调度、训练轮数）。
- 当前状态：你的结果显示不够稳定，已不作为主推进版本。

## Step1b（当前通过版）
- 配置文件：`configs/paviau_planA_step1b.yaml`
- 相对 Step0 的唯一目标：优化训练配方并提升稳定性。
- 关键改动：
  - `AdamW + Cosine`
  - `learning_rate=5e-4`
  - `epochs_base/inc=60`
- 你当前结果结论：
  - `Task2 OA` 明显提升
  - `Task1 OA/AA` 未恶化并有提升
  - `forgetting` 显著下降
- 当前结论：作为后续步骤的新起点。

## Step2（原始版）
- 配置文件：`configs/paviau_planA_step2.yaml`
- 作用：只改输入维度 `PCA30 -> PCA60`，其他保持 Step1。
- 当前状态：由于主线切到 Step1b，建议改用 Step2b。

## Step2b（当前建议版）
- 配置文件：`configs/paviau_planA_step2b.yaml`
- 相对 Step1b 的唯一改动：`pca_dim: 30 -> 60`
- 作用：验证更高光谱维度是否提升增量表现。

## Step3
- 配置文件：`configs/paviau_planA_step3.yaml`
- 相对前一步主要改动：`ResNet -> Hybrid 3D-2D backbone`
- 作用：验证谱空联合建模是否带来收益。

## Step4
- 配置文件：`configs/paviau_planA_step4.yaml`
- 相对前一步主要改动：
  - `linear -> cosine classifier`
  - 启用 `bias_correction`
- 作用：缓解增量阶段新旧类偏置问题。

## Step5
- 配置文件：`configs/paviau_planA_step5.yaml`
- 相对前一步主要改动：
  - 启用特征回放（不存原图）
  - `memory_per_class=20`
- 作用：首次验证 Plan A 核心机制（feature replay）是否有效。

## Step6
- 配置文件：`configs/paviau_planA_step6.yaml`
- 相对前一步主要改动：
  - `memory_per_class=40`
  - `lambda_align=0.1`
  - 启用 `balanced_finetune`
- 作用：在 Step5 可行基础上冲击更高上限。

## 额外对照：任务划分敏感性实验
- 配置文件：`configs/paviau_planA_step1b_task5211.yaml`
- 任务划分：`5+2+1+1`
- 作用：对比 `5+2+2` 与更细粒度任务切分的影响。
- 你当前结果结论：
  - 最终任务 `OA/AA` 明显劣于 `5+2+2`
  - `forgetting` 接近，但最终性能不占优
- 当前建议：论文主线继续使用 `5+2+2`，`5+2+1+1` 作为补充分析。

## 当前推荐推进顺序
1. `step0`（基线）
2. `step1b`（已通过）
3. `step2b`
4. 后续再决定是否生成 `step3b/step4b`（基于 step1b 继续控制变量）

