# 近 5 年且有代码可复现的 baseline 清单（2021-2026）

更新时间：2026-03-13

说明：
- 这个版本已经按你的要求重新筛过。
- 只保留：
  - `2021-2026` 范围内的方法
  - 已确认有公开代码的方法
- 所有“只有论文、未确认代码”的方法都已经删除，不再放进这份文档。

## 1. 最推荐的对比名单

### 主表优先推荐
- PASS (2021)
- SSRE (2022)
- Strong Pre-Trained Models for CIL (2022)
- FeTrIL (2023)
- FEICA-CIL (2025)
- Ours

### 可选补充
- Generative Classifiers for CIL (2021)
- DER-ClassIL / Dark Experience Replay 系列实现（2021 后持续可用）

说明：
- 如果你论文主打“无回放”，主表建议优先放：
  - PASS
  - SSRE
  - FeTrIL
  - FEICA-CIL
  - Strong Pre-Trained Models
  - Ours
- `Generative Classifiers` 和 `DER` 更适合补充实验或附录。

## 2. 具体方法、论文和代码

### 2.1 PASS (2021)
- 论文：
  - Kai Zhu et al.
  - *Prototype Augmentation and Self-Supervision for Incremental Learning*
  - CVPR 2021
  - 论文链接：
    https://openaccess.thecvf.com/content/CVPR2021/html/Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.html
- 代码：
  - https://github.com/Impression2805/CVPR21_PASS
- 为什么值得比：
  - 这是你当前项目的直接祖线
  - 必须保留，不然读者看不出你的改进到底来自哪里

### 2.2 Generative Classifiers for CIL (2021)
- 论文：
  - Gido M. van de Ven et al.
  - *Class-Incremental Learning With Generative Classifiers*
  - CVPRW 2021
  - 论文链接：
    https://openaccess.thecvf.com/content/CVPR2021W/CLVision/html/van_de_Ven_Class-Incremental_Learning_With_Generative_Classifiers_CVPRW_2021_paper.html
- 代码：
  - https://github.com/GMvandeVen/class-incremental-learning
- 为什么值得比：
  - 属于 non-exemplar / generative classifier 路线
  - 可以作为与你当前 prototype / distillation 思路差异较大的补充对照
- 建议用法：
  - 放补充实验即可，不建议压过 HSI 专用方法

### 2.3 SSRE (2022)
- 论文：
  - Kai Zhu et al.
  - *Self-Sustaining Representation Expansion for Non-Exemplar Class-Incremental Learning*
  - CVPR 2022
  - 论文链接：
    https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Self-Sustaining_Representation_Expansion_for_Non-Exemplar_Class-Incremental_Learning_CVPR_2022_paper.html
- 代码：
  - https://github.com/zhukaii/SSRE
- 为什么值得比：
  - 是近 5 年内标准的 rehearsal-free 强基线
  - 在无回放设定下很有代表性

### 2.4 Strong Pre-Trained Models for CIL (2022)
- 论文：
  - Yue Wu et al.
  - *Class-Incremental Learning With Strong Pre-Trained Models*
  - CVPR 2022
  - 论文链接：
    https://openaccess.thecvf.com/content/CVPR2022/html/Wu_Class-Incremental_Learning_With_Strong_Pre-Trained_Models_CVPR_2022_paper.html
  - Amazon Science 页面：
    https://www.amazon.science/publications/class-incremental-learning-with-strong-pre-trained-models
- 代码：
  - https://github.com/amazon-science/sp-cil
- 为什么值得比：
  - 代表“强表征 + 轻量增量适配”路线
  - 如果你论文里会讨论 backbone / representation quality，这个非常合适
- 建议用法：
  - 可进主表，也可放补充表，取决于你主表容量

### 2.5 FeTrIL (2023)
- 论文：
  - Gregoire Petit et al.
  - *FeTrIL: Feature Translation for Exemplar-Free Class-Incremental Learning*
  - WACV 2023
  - 论文链接：
    https://openaccess.thecvf.com/content/WACV2023/html/Petit_FeTrIL_Feature_Translation_for_Exemplar-Free_Class-Incremental_Learning_WACV_2023_paper.html
- 代码：
  - https://github.com/GregoirePetit/FeTrIL
- 为什么值得比：
  - exemplar-free
  - 侧重 feature translation / pseudo-features
  - 和你当前的 prototype / feature-space 叙事有较强可比性

### 2.6 FEICA-CIL (2025)
- 论文：
  - Ruihang Wu et al.
  - *Feature space expansion and compression with spatial-spectral augmentation for hyperspectral image class-incremental learning*
  - Pattern Recognition, 2025
  - 论文链接：
    https://linkinghub.elsevier.com/retrieve/pii/S003132032500490X
- 代码：
  - https://github.com/knockshot/FEICA-CIL
- 为什么值得比：
  - 这是目前最关键的 HSI 专用较新 baseline 之一
  - 它和你当前做的 spatial-spectral / task-aware SSL 方向最接近
  - 如果只能复现一个 HSI 专用新方法，优先就是它

### 2.7 DER-ClassIL / Dark Experience Replay 相关实现
- 两种常见对应方式：

#### A. DER: Dynamically Expandable Representation for Class Incremental Learning (CVPR 2021)
- 论文：
  - *DER: Dynamically Expandable Representation for Class Incremental Learning*
  - CVPR 2021
  - 论文链接：
    https://openaccess.thecvf.com/content/CVPR2021/html/Yan_DER_Dynamically_Expandable_Representation_for_Class_Incremental_Learning_CVPR_2021_paper.html
- 代码：
  - https://github.com/Rhyssiyan/DER-ClassIL.pytorch

#### B. Dark Experience Replay 系列实现
- 代码平台：
  - https://github.com/aimagelab/mammoth

- 为什么值得比：
  - 如果你想给审稿人一个“补充的常见 CIL 参考实现”，DER 系列很常见
- 建议用法：
  - 由于它更偏 replay，不建议放在你“无回放主表”的中心位置

## 3. 最终推荐怎么排

### 最稳的主表版本
- PASS (2021)
- SSRE (2022)
- Strong Pre-Trained Models (2022)
- FeTrIL (2023)
- FEICA-CIL (2025)
- Ours

### 如果主表想更聚焦“无回放 + 更贴你方法”
- PASS
- SSRE
- FeTrIL
- FEICA-CIL
- Ours

这版会更干净。

### 补充实验
- Generative Classifiers for CIL
- DER / Mammoth 中的对应实现

## 4. 复现优先级建议

### 第一优先级
- PASS
- FEICA-CIL

原因：
- PASS 是直接祖线
- FEICA-CIL 是 HSI 专用较新强方法

### 第二优先级
- SSRE
- FeTrIL

原因：
- 都是标准 rehearsal-free / exemplar-free 方法
- 都有代码，适合作为无回放主表补强

### 第三优先级
- Strong Pre-Trained Models
- Generative Classifiers
- DER 系列

原因：
- 都有代码
- 但与你当前 HSI 主线的贴合度略弱，更适合做补充参考
