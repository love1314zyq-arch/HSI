# 【Plan A】PASS-HSI升级方案（特征回放，不存原图）

## 标记信息
- 方案编号：Plan A
- 方案名称：PASS + 特征回放
- 论文定位：稳妥可复现优先
- 目标：平均OA 80%-90%，最后任务OA尽量80%以上

## 1. 目标与现状
- 当前baseline后期遗忘明显，最后任务精度不足。
- 主目标是提高Task2并降低forgetting。
- 使用5个随机种子报告mean±std。

## 2. 总体路线
保留PASS主框架，加入只基于特征的回放机制：
- 不存旧任务原图patch
- 存每类少量特征向量与统计量
- 增量训练时与新类样本联合优化

## 3. 技术改造
### 3.1 数据维度策略
对比三种设置：
- PCA30
- PCA60（主推）
- Full-band（PaviaU约103维）

### 3.2 Backbone升级
- 从2D ResNet升级为轻量3D-2D谱空联合网络（HybridSN-lite思路）。
- 目的：增强光谱和空间联合建模能力。

### 3.3 损失与分类头升级
总损失：
- 分类损失L_ce
- 蒸馏损失L_kd（logit+feature）
- 原型增强损失L_proto
- 特征回放损失L_freplay（新增）
- 分布对齐损失L_align（新增）

分类头：
- 使用cosine classifier
- 每个任务后做weight alignment和logit calibration

### 3.4 特征记忆库设计
- 每类存K条特征，建议K=40（消融20/40/80）
- 存储：feature(fp16)、类均值、方差
- 采样：按类均衡采样，与新类batch混合训练

### 3.5 HSI专用增强
在现有增强上增加：
- 光谱平滑扰动
- 光谱mixup（同类优先）
- 空间CutMix（轻量）
- 多尺度patch（11/15）

## 4. 训练配方
- Optimizer：AdamW
- LR：3e-4 + cosine
- WD：5e-4
- Batch：128
- Epoch：base=100，inc=80
- 任务后平衡微调：5-10 epoch
- Seeds：1993, 2025, 3407, 4242, 6666

## 5. 工程实施清单
建议新增：
- models/hybrid_hsi.py
- cil/memory_bank.py
- cil/losses.py
- augment/hsi_aug.py
- trainer/pass_v2.py

配置新增：
- data.spectral_mode: pca|full
- data.pca_dim: 30|60
- model.backbone
- cil.classifier
- replay.enable/replay.type/replay.memory_per_class
- loss.lambda_freplay/loss.lambda_align
- bias_correction.enable

## 6. 实验设计
主表：
- Task0/1/2的OA、AA、Kappa
- Average Forgetting
- 5 seeds mean±std

消融：
- PCA30 vs PCA60 vs Full-band
- ResNet vs Hybrid
- 无回放PASS vs Plan A
- memory_per_class消融
- 有无bias correction

## 7. 风险与回退
- Full-band不稳：退回PCA60主线
- 回放收益不足：提高K或加强对齐损失
- 新类过拟合：提高KD权重并启用平衡微调

## 8. 预期结果
- Task2 OA：80%-86%
- 平均OA：82%-88%

## 9. 论文贡献表述建议
- 提出无原图特征回放的PASS升级框架
- 系统比较PCA与全谱段在增量HSI中的表现
- 通过谱空增强与偏置校准显著降低遗忘