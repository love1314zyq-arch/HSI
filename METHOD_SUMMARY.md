# TAPIR-HSI 方法整合草案

更新时间：2026-03-16

## 方法总览
当前项目最合理的方法整合方式已经固定为三大主干模块加一组辅助实现细节。

### 主干模块
1. `任务感知辅助学习`
2. `HSI 专属原型增强`
3. `最新回放机制`

### 辅助实现细节
- `hybrid_hsi_lite`
- `cosine classifier`
- `bias correction`
- `feature distillation`

这些辅助细节对结果稳定很重要，但不建议在论文中与三大主干模块并列宣称为核心创新。

## 1. 任务感知辅助学习

当前项目的辅助学习模块不是固定单一目标，而是根据任务结构选择：

- 多类增量阶段：`ssma`
- 单类或极小类增量阶段：`spectral3`

推荐英文表述：
- `Task-Aware Auxiliary Learning`

核心思想：
- 辅助学习目标不应在所有增量阶段保持固定
- 不同任务结构应选择不同更合适的辅助目标

## 2. HSI 专属原型增强

当前项目中的原型增强已经发展成 HSI 主线专用的模块，而不是直接照搬 PASS 原始表述。

推荐中文表述：
- `HSI 专属原型增强`

推荐英文表述：
- `HSI-Specific Prototype Augmentation`

其作用是：
- 稳定旧类特征分布
- 增强旧类边界约束
- 为增量阶段提供类级表征支撑

## 3. 最新回放机制

当前项目中的回放机制应统一写成一个完整模块，而不是零散写成几个技巧。

组成包括：
- `merged raw replay`
- `herding exemplar selection`
- `fixed-budget memory allocation`

推荐英文表述：
- `Fixed-Budget Merged Raw Replay`

其核心规则是：
- 使用原始 patch 级 replay
- 旧类 memory 直接并入当前任务训练集
- exemplar 通过 herding 选择
- memory 总量固定
- 极小训练样本类优先全量回放
- 其余类按训练样本规模比例分配
- 普通类保留最小 exemplar 下限

## 方法命名建议

推荐继续使用：

## `TAPIR-HSI`

推荐解释：

`Task-Aware Prototype Augmentation with Integrated Replay for Hyperspectral Class-Incremental Learning`

如果你想让名字更贴近当前最终实现，也可以在正文里补一句：

> TAPIR-HSI integrates task-aware auxiliary learning, HSI-specific prototype augmentation, and fixed-budget merged raw replay for hyperspectral class-incremental learning.

## 一句话方法描述

> TAPIR-HSI is a hyperspectral class-incremental learning framework that combines task-aware auxiliary learning, HSI-specific prototype augmentation, and fixed-budget merged raw replay, while using cosine classification and bias correction as stabilizing implementation details.
