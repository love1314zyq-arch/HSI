# 论文写作备忘

更新时间：2026-03-12

## 1. 当前是否已经可以开始写论文
- 可以开始写，而且应该开始写。
- 当前项目已经具备论文主体所需的基本要素：
  - 明确任务：高光谱图像增量分类
  - 明确方法主干：`hybrid_hsi_lite + PASS + cosine classifier + bias correction`
  - 双数据集：`PaviaU`、`Salinas`
  - 双 SSL 策略：`rotation4`、`spectral3`
  - 多种任务划分实验
  - 一个稳定主线：PaviaU `5+2+2 + step4b`

结论：
- 主框架已经定型，可以开始论文主体写作。
- 当前尚未完全定型的是：如何把 `spectral3` 的条件性优势融入统一方法叙述中。

## 2. 当前最稳的论文主线
- 主线方法仍然应以 `step4b` 为核心：
  - `hybrid_hsi_lite`
  - PASS（prototype augmentation + distillation）
  - cosine classifier
  - bias correction
- 主线实验建议继续以 PaviaU `5+2+2` 为核心结果。

原因：
- 这一条线目前最稳定。
- 已有结果最完整。
- 最适合承担论文的“主结果表”和“主方法有效性论证”。

## 3. 对 spectral3 的正确定位
- 当前不适合写成：`spectral3` 全面优于 `rotation4`
- 当前更准确的结论是：
  - `spectral3` 在“初始任务之后的第一个增量阶段且只有 1 个新类”时，明显优于 `rotation4`
  - 但这并不意味着它在整个增量序列或所有任务划分上都更优

当前已有支持例子：
- PaviaU `8+1`
  - rotation4：Task1 `OA=0.4949`
  - spectral3：Task1 `OA=0.8937`
- Salinas `15+1`
  - rotation4：Task1 `OA=0.7490`
  - spectral3：Task1 `OA=0.8050`

同时已有反例或边界：
- PaviaU `5+2+2` 下，rotation4 整体优于 spectral3
- Salinas `14+1+1` 下，spectral3 在 Task1 很强，但 Task2 继续退化

因此论文表达不能写成“spectral3 更强”，而应该写成“spectral3 对特定任务结构更有效”。

## 4. 如何把它融进统一框架，而不是看起来像拼凑

### 不推荐的写法
- 主方法是一个固定框架
- 后面再额外补一个 spectral3 小技巧

这样会让读者感觉：
- 主方法和 spectral3 是分离的
- spectral3 更像实验中途添加的补丁

### 推荐的写法
把整个方法统一表述为：
- 一个面向 HSI 增量分类的统一框架
- 其中自监督模块本身就是“可适配任务结构的辅助目标”

推荐命名方向：
- `task-aware self-supervision`
- `task-adaptive auxiliary self-supervision`
- `structure-aware self-supervised regularization`

也就是说：
- 你的整体方法不是“PASS + rotation4”或“PASS + spectral3”
- 而是“PASS-HSI with task-aware self-supervision”

其中：
- `rotation4` 是多类常规增量阶段更稳的 SSL 实例
- `spectral3` 是极小类增量阶段，尤其首个单类增量阶段更合适的 SSL 实例

这样在逻辑上就是一个框架里的不同实例化，而不是两个拼起来的方法。

## 5. 方法部分建议怎么写

### 方法总框架
方法部分应先讲统一框架，不先分 rotation4 / spectral3：
- backbone feature extraction
- prototype augmentation
- feature distillation
- classifier calibration / bias correction
- auxiliary self-supervised objective

然后把 SSL 写成一个统一模块：
- 它的目的是缓解增量学习中的过拟合和表征退化
- 但不同任务结构下，合适的代理任务并不相同

### 两种 SSL 实例
再把 rotation4 和 spectral3 作为同一模块下的两种实例：
- spatial transformation based SSL
- spectral transformation based SSL

最后给出你的关键论点：
- 当增量阶段包含多个新类时，空间变换类 SSL 更稳
- 当增量阶段极小，尤其只包含 1 个新类时，光谱变换类 SSL 更有利于学习该新类的判别表示

## 6. 最重要的统一表达
论文中建议反复坚持这个核心表述：

> 不应将自监督目标固定为单一形式，而应根据增量任务的结构选择更合适的辅助目标。

中文可写成：
- 自监督辅助目标不应在所有增量阶段保持固定，而应随任务结构自适应选择。

这句话很重要，因为它决定你的 spectral3 不是“补充实验”，而是“方法思想的一部分”。

## 7. 实验部分建议怎么组织

### 主结果
- 仍然使用最稳主线承担论文核心结果：
  - PaviaU `5+2+2 + step4b`

### 扩展实验
把不同 SSL 的比较放进“结构敏感性分析”里：
- `5+2+2`
- `8+1`
- `15+1`
- `8+4+4`

目标不是证明某一个 SSL 全局最强，而是证明：
- 不同任务结构对应不同更优的 SSL 形式

### 推荐的小节标题
- Effect of Task Structure on Auxiliary Self-Supervision
- Task-Aware Self-Supervision for Incremental HSI Classification
- Choosing SSL Objectives Under Different Incremental Regimes

## 8. 如何让论文看起来是一个完整方法
最关键的方式不是“文字上解释”，而是补一个非常简单但清晰的机制：

### 推荐机制：规则式选择器
例如：
- 若当前增量 task 新类数 `<= 1`，使用 `spectral3`
- 否则使用 `rotation4`

这个机制的优点：
- 极简单
- 可解释
- 和你的实验观察直接一致
- 能把“经验现象”升级成“方法设计”

一旦加入这个选择器，你的论文就可以写成：
- 我们提出一个 task-aware SSL framework
- 它根据当前增量阶段的新类规模选择更合适的辅助任务

这样别人看到的是一个统一策略，而不是几块实验拼起来。

## 9. 当前最合理的论文叙事

### 论文故事线
1. HSI 增量分类存在严重遗忘与表征退化
2. PASS 提供了 prototype augmentation + distillation 的基础框架
3. 但固定单一的辅助自监督目标不足以适应不同增量结构
4. 因此，引入 task-aware SSL 机制
5. 在常规多类增量下使用更稳的空间型 SSL
6. 在极小类增量下使用更有效的光谱型 SSL
7. 最终形成统一、可适配任务结构的 HSI 增量学习框架

这个叙事是连贯的，也更像一个完整方法。

## 10. 现阶段不建议在论文里怎么写
- 不要写：`spectral3` 全面优于 `rotation4`
- 不要写：我们额外尝试了另一个 SSL，发现有时更好
- 不要把 spectral3 单独作为一个孤立 tricks 小节

这些写法都会削弱整体性。

## 11. 现阶段建议马上补的实验
为了让上述写法更稳，建议补最关键的一组对照：
- 固定 backbone / classifier / bias correction
- 只比较：
  - rotation4
  - spectral3
  - task-aware selection
- 在以下划分上验证：
  - PaviaU `5+2+2`
  - PaviaU `8+1`
  - Salinas `8+4+4`
  - Salinas `15+1`

如果 task-aware selection 在这些设置下整体最好或最平衡，这篇论文的整体性会明显增强。

## 12. 当前结论
- 可以开始写论文。
- 主线已经足够成熟。
- `spectral3` 最适合作为“task-aware SSL”中的一个条件性策略，而不是独立平行方法。
- 只要再补一组简洁但关键的策略选择实验，就可以把当前项目从“稳定工程实现 + 有趣现象”提升成“有统一叙事的方法论文”。

## 13. 截至 2026 的 HSI 增量分类科研进展判断

### 总体判断
- 到 2026 年，HSI 增量分类仍然是一个相对小众但正在升温的方向。
- 和通用视觉 CIL 相比，HSI 领域的专用方法数量明显更少。
- 当前更清晰的研究脉络主要有三条：
  - 基于蒸馏与偏置校正的常规 CIL
  - 基于 prototype / analytic learning 的无回放 CIL
  - 面向极低样本增量场景的 few-shot CIL / training-free CIL

### 时间线式梳理
- `2022`：
  - HSI 领域已经出现较明确的 class-incremental 论文，核心思想是
    - knowledge distillation
    - channel attention
    - linear correction / bias correction
  - 这说明“蒸馏 + 分类器校正”是 HSI 增量分类的早期主线之一

- `2024`：
  - 出现了面向 HSI 的 analytic-learning 路线（HSI-CIL）
  - 这类方法强调把增量学习尽量转化为解析式/轻训练更新，以降低灾难遗忘和重复训练成本
  - 说明领域开始从“普通深度微调 + distillation”向“更轻量、可快速增量”的方向扩展

- `2025`：
  - HSI-CIL 继续向“空间-光谱联合增强 + 特征空间扩展/压缩”演进，代表性工作是 FEICA-CIL
  - 同时期，通用 CIL 领域中 rehearsal-free 方法也继续增强，特别是 PASS++ 这类从 representation bias 和 classifier bias 同时下手的工作
  - 说明对“偏置控制”“特征空间结构设计”“不依赖回放样本”的关注明显增强

- `2026`：
  - HSI 领域进一步出现了 few-shot class-incremental / training-free prototype 路线，代表是 PSCEN
  - 这表明研究热点开始向“极少样本”“更低训练成本”“原型驱动增量更新”倾斜

### 对你项目最重要的结论
- 截至 2026，HSI 增量分类领域还没有形成一个像 ImageNet-CIL 那样非常统一、卷得很深的 benchmark 生态。
- 对你有利的一点是：
  - 你不需要面对特别多的 HSI 专用强 baseline
  - 只要把“HSI 专用无回放方法 + 通用无回放强基线”组织好，论文对比就可以很完整

## 14. 论文里建议对比哪些算法

### 总原则
因为你的项目目标是强调“无回放设置下算法表现优秀”，所以主表建议优先放：
- HSI 专用、无回放或近似无回放方法
- 通用视觉领域的强无回放 CIL 方法

不要把主表重心放在 exemplar-based 方法上，否则会稀释你的论文定位。

### 第一层：必须对比
这些方法最适合放在主结果表中。

#### 1) HSI class-incremental + knowledge distillation + linear correction（2022）
- 这篇工作几乎可以看作 HSI 增量分类的早期直接基线。
- 和你当前项目最相关，因为它同样涉及：
  - distillation
  - bias / linear correction
  - HSI 专用 backbone 设计
- 如果你的方法明显超过它，论文叙事会很顺。

建议定位：
- “HSI 专用早期 CIL 基线”

#### 2) HSI-CIL: analytic learning for hyperspectral image classification（2024）
- 这是更近的 HSI 专用无回放路线。
- 它强调 analytic learning / 轻训练增量更新，是典型的“不要依赖回放”的思路。

建议定位：
- “HSI 专用无回放代表方法”

#### 3) FEICA-CIL（2025）
- 这是到 2025 年相当值得关注的 HSI 专用方法。
- 关键词本身就和你的项目很相关：
  - spatial-spectral augmentation
  - feature space expansion / integration / compression
- 这篇是你现在最应该重点盯住的同领域对手之一。

建议定位：
- “截至 2025 的 HSI 专用较新强基线”

#### 4) PASS（2021）
- 你当前项目本身就是在 PASS 框架上演化，所以 PASS 必须保留。
- 否则读者会不清楚你到底比基础方法强了多少。

建议定位：
- “方法祖线 / direct ancestor baseline”

#### 5) PASS++（2025）
- 这是通用视觉里非常适合你借来对比的 non-exemplar 强基线。
- 它直接抓 representation bias 和 classifier bias，这和你论文里“bias correction + task-aware SSL”的叙事天然同频。

建议定位：
- “通用 rehearsal-free CIL 强基线”

### 第二层：建议对比
这些方法不一定都要放主表，但很适合放补充实验或附录。

#### 6) FeTrIL（2023）
- exemplar-free
- 用 pseudo-features / feature translation 维持旧类稳定性
- 和你的 prototype / feature-space 思路有对照价值

建议定位：
- “特征空间生成型 rehearsal-free baseline”

#### 7) Class-Incremental Learning With Strong Pre-Trained Models（2022）
- 这类方法强调强预训练表示 + 轻量增量适配
- 如果你后面想把“强 backbone + 轻量增量更新”写进讨论，这个很适合引用

建议定位：
- “强预训练表征路线对照”

#### 8) Class-Incremental Learning With Generative Classifiers（2021）
- 这是一条非 exemplar、偏 generative classifier 的路线
- 和你的方法差别较大，但适合作为“另一类 rehearsal-free 思路”出现

建议定位：
- “非回放、生成式分类路线基线”

### 第三层：按论文定位决定要不要对比

#### 9) PSCEN（2026）
- 这是 few-shot class-incremental for HSI
- 如果你论文明确强调：
  - 单类增量
  - 极低样本增量
  - training-efficient incremental learning
  那它非常值得加入
- 但如果你的主任务仍然是常规 class-incremental，而不是 few-shot incremental，就不建议把它放主表中心位置

建议定位：
- “few-shot / training-free HSI incremental 扩展对比”

## 15. 你的论文应该怎么组织对比表

### 推荐的主表结构
主表只放无回放方法，形成一个非常清晰的对比：
- HSI KD + linear correction（2022）
- HSI-CIL（2024）
- FEICA-CIL（2025）
- PASS（2021）
- PASS++（2025）
- Ours

这样读者一眼就会明白：
- 你比的是“同赛道”的无回放方法
- 不是拿自己去和需要 memory buffer 的方法硬混比

### 推荐的补充表
补充表可以再分两张：

#### 补充表 A：task-aware SSL 消融
- PASS + rotation4
- PASS + spectral3
- PASS + task-aware selection
- Ours full

#### 补充表 B：极端单类增量 / low-shot 增量分析
- 重点放：
  - `8+1`
  - `15+1`
  - `14+1+1`
- 如果论文最终强调“单类首增量阶段 spectral3 更强”，这里就是关键支撑

## 16. 如何在论文里表述“为什么只和无回放方法比”
- 可以明确说明：
  - 你的目标场景受到 memory / privacy / storage 限制
  - 因此论文主对比聚焦 rehearsal-free / non-exemplar 方法
- 不要写成“因为 exemplar-based 太强所以不比”
- 应该写成：
  - “为了公平评估在无回放约束下的方法能力，主实验选择同为 rehearsal-free 的代表方法作为主要比较对象”

如果担心审稿人质疑，可以在附录或补充实验里放 1 到 2 个经典 exemplar-based 方法作为参考上界，但不要让它们主导主表。

## 17. 当前最推荐的对比名单

### 主表推荐名单
- PASS (2021)
- HSI incremental classification with KD + linear correction (2022)
- HSI-CIL analytic learning (2024)
- FEICA-CIL (2025)
- PASS++ (2025)
- Ours

### 可选补充名单
- FeTrIL (2023)
- Generative Classifiers for CIL (2021)
- PSCEN (2026, 如果你强调 few-shot / single-class incremental)

## 18. 文献线索（后续写 Related Work 时可直接回看）
- PASS, CVPR 2021  
  https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.html

- HSI incremental classification with knowledge distillation, channel attention, and linear correction, Remote Sensing 2022  
  https://www.mdpi.com/2072-4292/14/11/2556

- HSI-CIL, analytic learning for hyperspectral image classification, 2024  
  https://www.sciencedirect.com/science/article/pii/S0016003224007063

- FEICA-CIL, Pattern Recognition 2025  
  https://linkinghub.elsevier.com/retrieve/pii/S003132032500490X

- PASS++, TPAMI 2025  
  https://pubmed.ncbi.nlm.nih.gov/40354219/

- FeTrIL, WACV 2023  
  https://openaccess.thecvf.com/content/WACV2023/html/Petit_FeTrIL_Feature_Translation_for_Exemplar-Free_Class-Incremental_Learning_WACV_2023_paper.html

- Class-Incremental Learning With Strong Pre-Trained Models, CVPR 2022  
  https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Class-Incremental_Learning_With_Strong_Pre-Trained_Models_CVPR_2022_paper.html

- Class-Incremental Learning With Generative Classifiers, CVPRW 2021  
  https://openaccess.thecvf.com/content/CVPR2021W/CLVision/html/van_de_Ven_Class-Incremental_Learning_With_Generative_Classifiers_CVPRW_2021_paper.html

- PSCEN, Expert Systems with Applications 2026  
  https://www.sciencedirect.com/science/article/pii/S0957417425045269

- CIL Survey, TPAMI 2024  
  https://pubmed.ncbi.nlm.nih.gov/39012754/

## 19. 对比论文代码可用性（截至 2026-03-13）

### 已确认有公开代码
- PASS, CVPR 2021
  - 论文：https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.html
  - 代码：https://github.com/Impression2805/CVPR21_PASS

- FEICA-CIL, Pattern Recognition 2025
  - 论文：https://linkinghub.elsevier.com/retrieve/pii/S003132032500490X
  - 代码：https://github.com/knockshot/FEICA-CIL

- FeTrIL, WACV 2023
  - 论文：https://openaccess.thecvf.com/content/WACV2023/html/Petit_FeTrIL_Feature_Translation_for_Exemplar-Free_Class-Incremental_Learning_WACV_2023_paper.html
  - 代码：https://github.com/GregoirePetit/FeTrIL

- Class-Incremental Learning With Strong Pre-Trained Models, CVPR 2022
  - 论文：https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Class-Incremental_Learning_With_Strong_Pre-Trained_Models_CVPR_2022_paper.html
  - Amazon Science 页面：https://www.amazon.science/publications/class-incremental-learning-with-strong-pre-trained-models
  - 代码：https://github.com/amazon-science/sp-cil

- Class-Incremental Learning With Generative Classifiers, CVPRW 2021
  - 论文：https://openaccess.thecvf.com/content/CVPR2021W/CLVision/html/van_de_Ven_Class-Incremental_Learning_With_Generative_Classifiers_CVPRW_2021_paper.html
  - 代码：https://github.com/GMvandeVen/class-incremental-learning

### 目前未确认到公开官方代码
- HSI incremental classification with knowledge distillation, channel attention, and linear correction, Remote Sensing 2022
  - 论文：https://www.mdpi.com/2072-4292/14/11/2556
  - 状态：截至 `2026-03-13` 未确认到公开官方代码仓库

- HSI-CIL, analytic learning for hyperspectral image classification, 2024
  - 论文：https://www.sciencedirect.com/science/article/pii/S0016003224007063
  - 状态：截至 `2026-03-13` 未确认到公开官方代码仓库

- PASS++, TPAMI 2025
  - 论文：https://pubmed.ncbi.nlm.nih.gov/40354219/
  - 状态：截至 `2026-03-13` 未确认到公开官方代码仓库

- PSCEN, Expert Systems with Applications 2026
  - 论文：https://www.sciencedirect.com/science/article/pii/S0957417425045269
  - 状态：截至 `2026-03-13` 未确认到公开官方代码仓库

### 写论文时的使用建议
- 主表里优先放“有论文 + 最好有代码”的方法，这样复现和公平对比都更容易解释。
- 对于没有公开代码的 HSI 专用方法：
  - 可以继续保留在 Related Work 中
  - 若无法完整复现，主表中应明确标注“结果引自原论文”或不做主表数值对比
- 对于 FEICA-CIL 和 PASS：
  - 这两个最值得优先复现
  - 一个是 HSI 专用较新方法，一个是你方法的直接祖线
# Paper Notes

## Auto SSL 的理论动机

### 核心结论

`spectral3` 并不是在所有增量场景下都优于 `rotation4`。更合理的结论是：两种自监督辅助任务对任务结构敏感，在不同的新类规模下作用机制不同，因此最优选择也不同。

### 现象总结

现有实验结果表明：

- 当初始任务之后的第一个增量阶段只包含 `1` 个新类时，`spectral3` 往往明显优于 `rotation4`
- 当增量阶段包含多个新类时，`rotation4` 通常更稳定

因此，辅助自监督目标不应被固定为单一形式，而应根据任务结构进行选择。

### 机制解释

#### 1. 单类增量阶段更容易出现监督不足和分类器偏置

在 PASS 框架中，增量阶段同时承担三项任务：

- 学习当前新类
- 通过蒸馏保持旧类知识
- 通过辅助自监督约束特征表示

当第一个增量 task 只有 `1` 个新类时，监督信号会出现两个问题：

- 当前 task 几乎没有新类之间的类间对比，监督信号主要只是在把这个新类与旧类拉开
- 当前阶段的真实训练样本几乎全部来自最新类，优化过程容易导致分类器向最新类偏置

因此，此时辅助自监督不再只是常规正则项，而是在补偿当前 task 缺失的判别结构。

#### 2. rotation4 更偏空间结构建模，在单类小增量场景下不够对口

`rotation4` 的本质是通过四种空间旋转构造自监督视图，让模型学习空间变换下的判别能力。它更依赖：

- patch 中存在足够稳定的空间纹理
- 当前 task 本身具有一定类别多样性
- 模型能够同时稳定学习类别判别和空间变换识别

但在“首个增量 task 只有 1 个新类”的场景中：

- 当前 task 的类别多样性很低
- 旋转带来的变化主要集中在空间排列
- 对高光谱分类真正关键的新类光谱边界补充不够直接

因此，`rotation4` 在该场景下提供的是空间辅助信息，而不是最直接的新类光谱建模约束。

#### 3. spectral3 直接作用于光谱表征，因此更适合单类新类建模

与 `rotation4` 不同，`spectral3` 是沿光谱维构造多视图。即使当前实现仍然属于启发式设计，它仍具有一个重要特性：

- 它直接约束模型学习同一类别在不同光谱视图下的稳定表示

当增量阶段只有一个新类时，`spectral3` 等价于人为为该新类补充多个光谱子视图，从而：

- 增强类内光谱结构建模
- 缓解单类监督过于单薄的问题
- 降低模型快速向单一新类过拟合的风险

因此，在 `8+1`、`15+1` 这类单类首增量任务下，`spectral3` 更容易带来性能优势。

#### 4. 多类增量阶段下，rotation4 往往更稳

当增量阶段包含多个新类时，例如 `5+2+2` 或 `8+4+4`：

- 当前 task 本身已经提供了足够的新类间监督信号
- SSL 的主要作用从“补判别结构”转为“增强泛化与稳定性”

此时 `rotation4` 更像一种温和的空间结构正则化：

- 它不会直接改动类别敏感的光谱分布
- 对多类场景中的整体特征稳定性更有利

相比之下，`spectral3` 直接改动光谱视图，在多类增量时可能对已有类别边界引入额外扰动，因此不一定更优。

### 自动切换逻辑的论文表达

可以将自动切换策略概括为：

当增量阶段的新类数极少时，模型缺乏足够的类间监督约束，此时需要更直接作用于光谱表示的辅助任务来增强新类建模；当增量阶段包含多个新类时，监督信号已经足够提供类间判别，此时更温和的空间自监督更有利于保持整体稳定性。

### 方法设计表述

建议在论文中表述为：

- 当第一个增量阶段只包含极少新类时，采用 spectral self-supervision
- 当增量阶段具有常规多类规模时，采用 rotation-based self-supervision

这应被写成“基于任务结构的设计选择”，而不是“某一种 SSL 在所有场景下普遍更优”。

### 推荐英文表达

#### Observation

In our experiments, spectral self-supervision consistently performs better when the first incremental stage contains only one new class, while rotation-based self-supervision is more stable in regular multi-class incremental settings.

#### Hypothesis

We hypothesize that single-class incremental stages suffer from insufficient inter-class supervision, making spectral-view based auxiliary tasks more effective for enriching intra-class spectral structure.

#### Design

Based on this observation, we adopt a task-aware SSL selection strategy that chooses spectral SSL for extremely low-class incremental stages and rotation SSL otherwise.

### 写作注意事项

论文中应避免使用如下绝对化表述：

- `spectral3 is universally better than rotation4`
- `spectral3 is theoretically superior`

更稳妥的写法应为：

- `the results suggest`
- `we hypothesize`
- `the observations indicate`

这样能把当前结论表述为“有实验支撑的机制假设”，而不是未经证明的理论定理。

## 可直接用于论文的方法动机表述

### 中文版本

在高光谱增量分类中，辅助自监督任务的有效性与任务结构密切相关。我们观察到，当初始任务后的第一个增量阶段仅包含极少新类，尤其是单一新类时，常规监督信号难以提供充分的类间约束，模型更容易出现新类表示过窄以及分类器向最新类偏置的问题。在这种情况下，沿光谱维构造多视图的自监督任务能够直接作用于新类的光谱表示学习，通过增强类内光谱结构建模来缓解监督不足带来的不稳定性。相对地，当增量阶段包含多个新类时，任务本身已经具备较充分的类间判别信息，此时辅助自监督的主要作用转变为提升整体表示的稳定性与泛化能力。基于这一点，空间旋转类自监督由于不会直接扰动类别敏感的光谱分布，通常能够在常规多类增量场景下提供更稳健的优化效果。基于上述分析，本文采用一种面向任务结构的自监督选择策略：当首个增量阶段仅包含极少新类时，采用光谱自监督；否则采用旋转自监督。该设计并非假设某一种自监督形式在所有场景下都更优，而是强调辅助任务应与增量阶段的类别结构相匹配。

### 精简中文版本

我们认为，增量阶段的新类规模会直接影响辅助自监督任务的最优形式。当首个增量阶段仅包含极少新类时，监督信号缺乏足够的类间约束，模型更容易对最新类产生过拟合，此时更直接作用于光谱表示的自监督任务能够有效增强类内结构建模；而在常规多类增量场景下，监督信息已足以支撑新类边界学习，更温和的空间旋转自监督通常更有利于保持整体表示稳定性。因此，本文采用基于任务结构的自监督选择策略，在极少新类增量场景下使用 spectral self-supervision，在常规多类增量场景下使用 rotation-based self-supervision。

### English Version

In hyperspectral class-incremental learning, the effectiveness of an auxiliary self-supervised objective is closely related to the task structure. We observe that when the first incremental stage after the base session contains very few new classes, especially only one new class, the supervised signal becomes insufficient to provide reliable inter-class constraints, making the model more vulnerable to narrow new-class representations and classifier bias toward the latest class. In this case, spectral-view based self-supervision directly regularizes the spectral representation of the new class and enriches its intra-class structure, thereby alleviating the instability caused by weak supervision. By contrast, when the incremental stage contains multiple new classes, the task itself already provides stronger inter-class discrimination, and the role of self-supervision shifts from compensating for missing class structure to improving feature stability and generalization. Under this regime, rotation-based self-supervision is usually more stable because it regularizes spatial structure without directly perturbing class-sensitive spectral distributions. Based on this observation, we adopt a task-aware SSL selection strategy: spectral self-supervision is used for extremely low-class incremental stages, while rotation-based self-supervision is used otherwise. This design does not assume that one SSL objective is universally superior; instead, it emphasizes that the auxiliary objective should match the class structure of the incremental task.
