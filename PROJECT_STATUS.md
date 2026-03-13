# 项目状态快照

更新时间：2026-03-12

## 一句话现状
- 代码主流程已稳定支持 `PaviaU` 和 `Salinas` 两个数据集。
- PaviaU 主线仍是 `step4b`，默认协议 `5+2+2`。
- Salinas 已完成接入，并有多组单 seed 探索结果；当前看 `8+4+4` 明显比极端拆分更稳定。
- 项目已接入统一实验汇总日志：`experiment_results.txt`。

## 当前代码能力
- 统一训练入口：`main_hsi.py`
- 统一预处理入口：`preprocess_hsi.py`
- 支持两种 SSL 方式：
  - `rotation4`
  - `spectral3`
- 支持两种任务划分方式：
  - 配置内 `base_classes + task_num`
  - 命令行显式 `--task_split`
- 每次实验结束后自动写入固定汇总文本：
  - `experiment_results.txt`

## PaviaU 当前结论

### 主线
- 主配置：`configs/paviau_planA_step4b.yaml`
- 默认协议：`5+2+2`
- 默认 PCA：`30`
- 当前最稳结论仍然是：
  - `hybrid_hsi_lite`
  - `cosine classifier`
  - `bias correction`

### 已有代表性结果
- `5+2+2 + rotation4 + seed=2025`
  - Task0 `OA=0.9998`
  - Task1 `OA=0.9696`
  - Task2 `OA=0.8903`
- `5+2+2 + spectral3 + seed=2025`
  - Task0 `OA=0.9998`
  - Task1 `OA=0.8940`
  - Task2 `OA=0.7672`

### 当前判断
- 在当前实现和已记录结果下，PaviaU 上 `rotation4` 仍优于 `spectral3`。
- 极端拆分会明显放大最后一个 task 的不稳定性：
  - `7+1+1`
  - `8+1`
- 原因是最后一个 task 只有 1 个新类时，当前无 replay 的设置更容易出现新类偏置和遗忘。
- 但需要补充一个更细的阶段性观察：
  - 当“初始任务之后的第一个增量阶段”只有 `1` 个新类时，`spectral3` 明显优于 `rotation4`
  - 代表性例子：`8+1`
    - rotation4：Task1 `OA=0.4949`
    - spectral3：Task1 `OA=0.8937`
  - 说明 `spectral3` 对“首个单类增量任务”更友好，但这还不能直接推出它在整个增量序列上全局更优

## Salinas 当前结论

### 已完成内容
- 数据接入、预处理、训练主流程、配置文件均已就绪。
- 已有可运行配置：
  - `configs/salinas_planA_step4b.yaml`
  - `configs/salinas_planA_step4b_spectral3.yaml`

### 已记录的代表性结果
- `8+4+4 + rotation4 + seed=2025`
  - Task0 `OA=1.0000`
  - Task1 `OA=0.8827`
  - Task2 `OA=0.9203`
- `8+4+4 + spectral3 + seed=2025`
  - Task0 `OA=0.9996`
  - Task1 `OA=0.9691`
  - Task2 `OA=0.9296`
- `15+1 + rotation4 + seed=2025`
  - Task1 `OA=0.7490`
- `15+1 + spectral3 + seed=2025`
  - Task1 `OA=0.8050`
- `14+1+1 + rotation4 + seed=2025`
  - Task1 `OA=0.9420`
  - Task2 `OA=0.7765`
- `14+1+1 + spectral3 + seed=2025`
  - Task1 `OA=0.9905`
  - Task2 `OA=0.7272`

### 当前判断
- Salinas 上 `8+4+4` 是目前最平衡、最值得继续扩展的协议。
- `spectral3` 在 Salinas 的 `8+4+4` 上表现优于 rotation4。
- 但在极端单类尾任务拆分（如 `14+1+1`、`15+1`）下，最后一个 task 仍明显掉点，说明问题不是数据集接入，而是当前训练目标对“最后单类增量”不够稳。
- 同时也观察到：
  - 当“初始任务后的第一个增量阶段”只有 `1` 个新类时，`spectral3` 往往优于 `rotation4`
  - 代表性例子：`15+1`
    - rotation4：Task1 `OA=0.7490`
    - spectral3：Task1 `OA=0.8050`
  - 这支持一个更统一的判断：`spectral3` 对“首个单类增量阶段”更有帮助，但对后续阶段是否继续收益，要按任务结构单独验证

## 实验记录机制
- 根目录维护固定汇总文件：`experiment_results.txt`
- 当前记录字段：
  - 实验时间
  - `exp_name`
  - 配置文件路径
  - `seed`
  - 每个 task 的类别数
  - 每个 task 最终测试指标：`OA / AA / Kappa`

## 当前主要风险
- `experiment_results.txt` 中夹杂了少量手工补写行，例如：
  - `oa:0.66 aa:0.775 k:0.592`
  - `oa：0.857 aa:0.858 k:0.841`
  这些行不会影响训练，但后续如果要自动解析日志，建议清洗格式或单独迁出。
- 类名 `PaviaUDataManager` 仍沿用旧命名，但职责实际上已经覆盖多数据集。
- 当前 task 可视化逻辑主要仍围绕 PaviaU 使用场景组织，Salinas 虽可运行，但可视化部分还不是项目重点。

## 下一步建议
1. PaviaU 继续以 `5+2+2 + step4b` 作为论文主线，不再把 `7+1+1`、`8+1` 当主结果。
2. Salinas 先固定比较 `8+4+4` 下的 `rotation4` 与 `spectral3`，再决定是否做多 seed。
3. 若要继续研究极端拆分尾任务掉点，优先尝试：
   - 减少 `epochs_inc`
   - 开启 replay
   - 在 task 开始前增加一次预评估，单独观察“初始保留能力”与“训练后退化”
4. 若论文要把 `spectral3` 融入统一方法，建议把它表述为“任务感知的 SSL 策略选择”，重点验证：
   - 常规多类增量：rotation4 更稳
   - 首个单类增量：spectral3 更优
