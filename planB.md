# 【Plan B】PASS-HSI升级方案（纯无回放增强）

## 标记信息
- 方案编号：Plan B
- 方案名称：纯无回放PASS增强
- 论文定位：严格rehearsal-free
- 目标：尽量冲击Task2 OA 80%，平均OA争取80%以上

## 1. 目标与风险
- 不使用任何旧类样本回放（不存原图，不存旧类样本级特征）。
- 方法更纯，但达到最终任务80%+难度高于Plan A。

## 2. 总体路线
在无回放约束下，重点增强三部分：
- 原型建模能力
- 跨任务知识保持能力
- 新旧类边界稳定能力

## 3. 技术改造
### 3.1 数据维度
做同样的三组对比：
- PCA30
- PCA60（主推）
- Full-band（补充）

### 3.2 Backbone升级
- 使用轻量3D-2D谱空联合网络。
- 理由：无回放条件下更依赖强表征。

### 3.3 原型机制升级
- 从点原型升级为分布原型。
- 每类维护均值+方差（或低秩协方差）。
- protoAug按分布采样，提高类内多样性覆盖。

### 3.4 蒸馏升级
- logit distillation
- feature distillation
- relational distillation（保持样本关系结构）

### 3.5 边界与偏置控制
- margin regularization抑制旧类边界被挤压
- cosine classifier + weight alignment
- 任务后无数据后校准（仅基于原型统计采样）

### 3.6 HSI增强
- 光谱平滑扰动
- 光谱mixup
- 空间CutMix（轻量）
- 多尺度patch
- 可选：光谱重建辅助头（低权重）

## 4. 训练配方
- Optimizer：AdamW
- LR：3e-4 + cosine
- WD：5e-4
- Batch：128
- Epoch：base=120，inc=100
- Label smoothing：0.05
- Seeds：5个固定种子

## 5. 工程实施清单
建议新增：
- models/hybrid_hsi.py
- cil/losses.py
- augment/hsi_aug.py
- trainer/pass_rf_v2.py

配置新增：
- data.spectral_mode
- model.backbone
- cil.classifier
- proto.distributional
- loss.lambda_rel_kd
- loss.lambda_margin
- post_calibration.enable

约束说明：
- 禁止保存旧类原图样本
- 禁止保存旧类样本级特征库
- 允许保存每类统计量（均值、方差等摘要）

## 6. 实验设计
主表：
- baseline PASS vs Plan B
- Task0/1/2 OA、AA、Kappa
- forgetting与稳定性统计

消融：
- 点原型 vs 分布原型
- 单一KD vs 多重KD
- 有无margin约束
- 有无后校准
- PCA30/60/full-band

## 7. 风险与回退
- Task2不足80：增加模型容量并强化KD
- 遗忘仍高：提高关系蒸馏和边界约束权重
- Full-band不稳：以PCA60为主结果

## 8. 预期结果
- Task2 OA：78%-83%（优秀配置可破80）
- 平均OA：80%-85%

## 9. 论文贡献表述建议
- 提出严格无回放的PASS分布原型增强框架
- 结合谱空增强、多重蒸馏和边界约束
- 在无回放前提下降低遗忘并提升后期任务性能