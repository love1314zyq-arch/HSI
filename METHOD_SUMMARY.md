# TAPIR-HSI 方法整合草案

更新时间：2026-03-15

## 1. 当前项目方法主线

基于当前项目的代码实现与实验结果，现阶段最合理的方法整合方式可以概括为三部分：

1. `任务感知辅助学习（Task-Aware Auxiliary Learning）`
2. `任务感知原型增强（Task-Aware Prototype Augmentation）`
3. `融合式原始样本回放（Integrated / Merged Raw Replay）`

在此基础上，再结合：
- `cosine classifier`
- `bias correction`

形成一个统一的高光谱图像类增量学习框架。

---

## 2. 三个核心模块

### 2.1 任务感知辅助学习

当前项目最有辨识度的结论不是固定使用某一种辅助学习方式，而是：

- 多类增量阶段使用 `SSMA`
- 单类增量阶段使用 `spectral3`

这部分应统一表述为：

- 中文：`任务感知辅助学习`
- 英文：`Task-Aware Auxiliary Learning`

核心思想是：

- 不同增量任务结构对应不同更合适的辅助学习形式
- 固定单一辅助目标并不能同时适配所有增量阶段

当前项目中：

- `spectral3` 更适合单类增量阶段
- `SSMA` 更适合多类增量阶段

---

### 2.2 任务感知原型增强

当前项目中的原型增强主干来源于 PASS 风格的 prototype augmentation：

- 先提取每个旧类的 prototype
- 在特征空间对 prototype 加入高斯扰动
- 生成旧类伪特征
- 用这些伪特征对分类头施加约束

但在方法整合层面，可以进一步将其表述为：

- 中文：`任务感知原型增强`
- 英文：`Task-Aware Prototype Augmentation`

其方法定位可描述为：

- 原型增强不是独立、固定地作用于所有增量阶段
- 而是嵌入在整个任务结构感知框架中
- 单类增量阶段更强调旧类稳定
- 多类增量阶段更强调旧类边界覆盖与判别支撑

即使当前代码尚未完全显式写出 task-aware 的调制项，这一表述依然与当前实验规律保持一致，并可以作为方法思想的统一描述。

---

### 2.3 融合式原始样本回放

当前项目中的 `replay-merged` 与常规 replay 的区别在于：

- 常规 replay：
  - 每次训练迭代中额外采样 memory 样本
  - 通过附加 replay loss 参与训练

- 当前 merged replay：
  - 直接将 memory 中的旧类 raw patch 拼接到当前 task 的训练集中
  - 由统一 dataloader 一起训练

因此，这部分应当被视为一个具有独立方法意义的模块，而不仅仅是参数技巧。

推荐表述：

- 中文：`融合式原始样本回放`
- 英文：`Integrated Raw Replay` 或 `Merged Raw Replay`

该模块的作用是：

- 更直接地把旧类信息融入当前增量训练过程
- 降低新旧样本训练机制割裂的问题
- 在单类和多类增量场景中都表现出明显增益

---

## 3. 方法总结构

因此，当前项目可以统一表述为：

> 一个面向高光谱图像类增量学习的统一框架，由任务感知辅助学习、任务感知原型增强和融合式原始样本回放三部分构成，并结合 cosine classifier 与 bias correction 实现稳定的增量分类。

英文可表述为：

> We propose a unified framework for hyperspectral class-incremental learning, which integrates task-aware auxiliary learning, task-aware prototype augmentation, and merged raw replay, together with cosine classifier calibration and bias correction.

---

## 4. 方法命名建议

基于当前项目的结构，推荐方法名：

## `TAPIR-HSI`

可解释为：

`Task-Aware Prototype Augmentation with Integrated Replay for Hyperspectral Class-Incremental Learning`

对应中文可写为：

`面向高光谱增量分类的任务感知原型增强与融合回放框架`

或更简洁地写成：

`任务感知原型增强与融合回放高光谱增量学习框架`

---

## 5. TAPIR-HSI 的核心组成

### 5.1 Task-Aware Auxiliary Learning
- 单类增量：`spectral3`
- 多类增量：`SSMA`

### 5.2 Task-Aware Prototype Augmentation
- 基于 prototype augmentation
- 在任务结构感知框架内发挥旧类稳定与边界支撑作用

### 5.3 Integrated Replay
- 使用 merged raw replay
- 将旧类 memory 与当前任务数据统一融合训练

### 5.4 Classifier Calibration
- `cosine classifier`
- `bias correction`

---

## 6. 当前最合理的论文叙事

论文中不应再将项目描述为：

- `PASS + spectral3`
- `PASS + SSMA`
- `PASS + replay`

而应统一写成：

1. 高光谱类增量学习中，不同增量任务结构具有不同学习难点
2. 因此需要一个任务结构感知的统一框架
3. 我们采用任务感知辅助学习以匹配不同任务类型
4. 同时结合原型增强稳定旧类知识
5. 再通过融合式原始样本回放进一步强化新旧类联合学习
6. 最终配合分类器校准实现稳定增量分类

在这种叙事下，整个方法更像一个完整的新框架，而不是对 PASS 的简单追加实验。

---

## 7. 当前写作时需要注意的边界

为了与当前代码实现保持一致，写作时建议注意以下边界：

- `task-aware auxiliary learning` 是当前已有明确实验支持的核心结论
- `merged raw replay` 也是当前已有代码和实验支持的明确模块
- `task-aware prototype augmentation` 可以作为方法整合层面的统一描述
- 但如果后续论文需要把该部分写成明确新增算法模块，最好进一步补一个轻量 task-aware 调制实现，以增强“方法创新点”的落地性

---

## 8. 当前最精炼的方法一句话

> TAPIR-HSI 是一个面向高光谱图像类增量学习的统一框架，通过任务感知辅助学习、任务感知原型增强与融合式原始样本回放，实现对不同任务结构的稳定增量分类。
