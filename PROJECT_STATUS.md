# 项目状态快照

更新时间：2026-03-16

## 一句话现状
- 当前主线方法已经固定为四模块组合：
  - 任务感知辅助学习：`spectral3 / ssma`
  - HSI 专属原型增强：`protoaug_hsi`
  - 最新回放机制：`merged raw replay + herding + fixed-budget replay`
  - 辅助实现细节：`hybrid_hsi_lite + cosine classifier + bias correction`
- 代码主流程已稳定支持四个数据集：
  - `PaviaU`
  - `Salinas`
  - `IndianPines`
  - `Houston`
- 最新配置已经在四个数据集上取得稳定正向结果，当前主要工作已从“主线方法探索”转入“对比方法与论文整理”。

## 当前方法主线
- 主干算法：
  - 任务感知辅助学习
  - HSI 专属原型增强
  - 最新回放机制
- 不作为主干算法宣称、但稳定使用的实现细节：
  - `hybrid_hsi_lite`
  - `cosine classifier`
  - `bias correction`
  - `feature distillation`

## 当前回放主线
- 组织方式：`merged raw replay`
- exemplar 选择：`herding`
- 记忆分配：`fixed-budget replay`
- 当前主用配置字段：
  - `memory_budget: 480`
  - `full_replay_below_train_count: 10`
  - `min_memory_per_class: 5`

## 四数据集代表性结果
- PaviaU `5+2+2 + ssma`
  - Task0 `OA=0.9990`
  - Task1 `OA=0.9874`
  - Task2 `OA=0.9672`
- PaviaU `7+1+1 + spectral3`
  - Task0 `OA=0.9998`
  - Task1 `OA=0.9826`
  - Task2 `OA=0.9819`
- Salinas `15+1 + ssma`
  - Task0 `OA=0.9997`
  - Task1 `OA=0.9929`
- Salinas `8+2+2+2+2 + ssma`
  - Task0 `OA=1.0000`
  - Task1 `OA=0.9955`
  - Task2 `OA=0.9916`
  - Task3 `OA=0.9907`
  - Task4 `OA=0.9920`
- IndianPines `8+4+4 + ssma`
  - Task0 `OA=0.9991`
  - Task1 `OA=0.9506`
  - Task2 `OA=0.9423`
- IndianPines `8+2+2+2+2 + ssma`
  - Task0 `OA=0.9991`
  - Task1 `OA=0.9696`
  - Task2 `OA=0.9495`
  - Task3 `OA=0.9490`
  - Task4 `OA=0.9425`
- Houston `7+2+2+2+2 + ssma`
  - Task0 `OA=0.9985`
  - Task1 `OA=0.9975`
  - Task2 `OA=0.9750`
  - Task3 `OA=0.9276`
  - Task4 `OA=0.9102`
- Houston `9+2+2+2 + ssma`
  - Task0 `OA=0.9976`
  - Task1 `OA=0.9837`
  - Task2 `OA=0.9522`
  - Task3 `OA=0.9288`
- Houston `13+1+1 + spectral3`
  - Task0 `OA=0.9984`
  - Task1 `OA=0.9900`
  - Task2 `OA=0.9804`

## 当前稳定结论
- 多类增量阶段，`ssma` 是当前主用辅助学习方式。
- 单类或极小类增量阶段，`spectral3` 更适合当前主线。
- `protoaug_hsi` 已成为主线方法的一部分，不再视为附加实验。
- `merged raw replay` 明显优于仅附加 replay loss 的旧组织方式。
- `herding + fixed-budget replay` 在 `IndianPines` 和 `Houston` 上带来了尤其明显的提升，同时在 `PaviaU` 和 `Salinas` 上也保持了强结果。

## 当前代码能力
- 统一训练入口：`main_hsi.py`
- 统一预处理入口：`preprocess_hsi.py`
- 统一多数据集数据管理：`dataset_paviau.py`
- merged replay 数据集封装：`dataset_replay_hsi.py`
- 原型增强与回放主逻辑：`PASS_hsi.py`
- replay exemplar 选择：`replay_selection.py`
- 固定实验日志：`experiment_results.txt`

## 当前工作重点
1. 继续完善外部 baseline 对比。
2. 补齐论文中的消融实验。
3. 统一论文中的方法命名、方法图和主结果表。
