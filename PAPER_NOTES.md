# 论文写作备忘

更新时间：2026-03-16

## 当前论文主线
- 当前论文主线已经固定，不再围绕早期 `step4b / rotation4` 展开。
- 论文应统一围绕以下四部分组织方法：
  - 任务感知辅助学习
  - HSI 专属原型增强
  - 最新回放机制
  - 分类与表征稳定化细节

## 方法应如何表述
- 主干算法：
  - `spectral3 / ssma` 的任务感知辅助学习
  - `protoaug_hsi` 风格的 HSI 专属原型增强
  - 最新回放机制：
    - `merged raw replay`
    - `herding exemplar selection`
    - `fixed-budget replay allocation`
- 辅助实现细节，不宜作为主创新点单列：
  - `hybrid_hsi_lite`
  - `cosine classifier`
  - `bias correction`
  - `feature distillation`

## 四数据集支撑
- 当前论文不再是双数据集结构，而是四数据集共同支撑：
  - `PaviaU`
  - `Salinas`
  - `IndianPines`
  - `Houston`

## 当前最重要的实验事实
- 多类增量阶段：
  - `ssma` 是当前更稳的主用辅助学习方式
- 单类或极小类增量阶段：
  - `spectral3` 更适合作为主用辅助学习方式
- `protoaug_hsi` 已经不是可选项，而是当前主线方法的一部分
- 最新回放机制在四个数据集上都已得到正向验证

## 最新回放机制应如何写
- 当前不应再写成“简单 raw replay”
- 更准确的表述是：
  - merged raw replay
  - herding-based exemplar selection
  - fixed-budget memory allocation

当前主用 memory 规则：
- `memory_budget: 480`
- 极小训练样本类优先全量回放
- 其余类按样本规模比例分配 replay 预算
- 同时保留最小 exemplar 保底

## 论文中推荐的统一叙事
1. 高光谱增量分类中，不同任务结构对应不同学习难点。
2. 因此，辅助学习目标不应固定，而应根据任务结构进行选择。
3. 在此基础上，引入 HSI 专属原型增强以稳定旧类表征与边界。
4. 再通过 merged raw replay 与 fixed-budget memory allocation 强化新旧类联合训练。
5. 最终配合分类器与表示稳定化细节，实现稳定增量分类。

## 当前方法定位
- 不要再写成：
  - `PASS + spectral3`
  - `PASS + SSMA`
  - `PASS + replay`
- 应统一写成：
  - 一个面向 HSI 增量分类的完整框架
  - 其中包含 task-aware auxiliary learning、HSI-specific prototype augmentation 和 fixed-budget merged replay

## 当前建议的方法名口径
- 中文：
  - 任务感知辅助学习 + HSI 专属原型增强 + 固定总量融合回放框架
- 英文建议继续使用：
  - `TAPIR-HSI`

## 当前写作优先级
1. 主方法图
2. 主结果表
3. 对比方法表
4. 消融实验
5. 类别级误差分析
