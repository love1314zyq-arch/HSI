# 师姐论文中提到的方法对应论文与代码

更新时间：2026-03-13

说明：
- 本文档对应你截图中提到的方法：
  - `ResNetAM`
  - `LwF`
  - `GFR`
  - `SSRE`
  - `iCaRL`
  - `DER`
- 其中 `ResNetAM` 和 `DER` 都存在同名/缩写歧义，这里优先给出和你截图语境最匹配的版本，并附上说明。

## 1. ResNetAM

### 最可能对应的论文
- **HResNetAM: Hierarchical Residual Network With Attention Mechanism for Hyperspectral Image Classification**
- Zhixiang Xue, Xuchu Yu, Bing Liu, Xiong Tan, Xiangpo Wei
- IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2021
- 论文链接：
  - https://doi.org/10.1109/JSTARS.2021.3065987
  - DBLP 页面：
    https://dblp.org/rec/journals/staeors/LiuY21

### 代码
- 截至 `2026-03-13`，未确认到作者公开的官方代码仓库
- 可找到的非官方实现：
  - https://github.com/AryanJ11/Hyperspectral-Image-classification

### 备注
- 你师姐文中写的是“带有注意机制的 ResNet (ResNetAM)”，在 HSI 分类语境下，最可能就是这篇 `HResNetAM`
- 如果你们组内原始参考文献 [71] 不是这篇，需要再按参考文献表核对一次

## 2. LwF

### 对应论文
- **Learning without Forgetting**
- Zhizhong Li, Derek Hoiem
- ECCV 2016
- 论文链接：
  - https://link.springer.com/chapter/10.1007/978-3-319-46493-0_37
  - arXiv:
    https://arxiv.org/abs/1606.09282

### 代码
- 官方代码：
  - https://github.com/lizhitwo/LearningWithoutForgetting

## 3. GFR

### 对应论文
- **Generative Feature Replay for Class-Incremental Learning**
- Xialei Liu, Chenshen Wu, Mikel Menta, Luis Herranz, Bogdan Raducanu, Andrew D. Bagdanov, Shangling Jui, Joost van de Weijer
- CVPR Workshops 2020
- 论文链接：
  - https://openaccess.thecvf.com/content_CVPRW_2020/html/w15/Liu_Generative_Feature_Replay_for_Class-Incremental_Learning_CVPRW_2020_paper.html
  - DeepAI 摘要页：
    https://deepai.org/publication/generative-feature-replay-for-class-incremental-learning

### 代码
- 官方代码：
  - https://github.com/xialeiliu/GFR-IL

## 4. SSRE

### 对应论文
- **Self-Sustaining Representation Expansion for Non-Exemplar Class-Incremental Learning**
- Kai Zhu, Wei Zhai, Yang Cao, Jiebo Luo, Zheng-Jun Zha
- CVPR 2022
- 论文链接：
  - https://openaccess.thecvf.com/content/CVPR2022/html/Zhu_Self-Sustaining_Representation_Expansion_for_Non-Exemplar_Class-Incremental_Learning_CVPR_2022_paper.html
  - arXiv:
    https://arxiv.org/abs/2203.06359

### 代码
- 官方代码：
  - https://github.com/zhukaii/SSRE

## 5. iCaRL

### 对应论文
- **iCaRL: Incremental Classifier and Representation Learning**
- Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg Sperl, Christoph H. Lampert
- CVPR 2017
- 论文链接：
  - https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html
  - DOI 页面：
    https://doi.org/10.1109/CVPR.2017.587

### 代码
- 官方代码：
  - https://github.com/srebuffi/iCaRL
- 备选实现：
  - https://github.com/RuggieroFrancavilla/Incremental-Learning-iCaRL

## 6. DER

### 最可能对应的论文
- **Dark Experience for General Continual Learning: a Strong, Simple Baseline**
- Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara
- NeurIPS 2020
- 论文链接：
  - https://proceedings.neurips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html
  - arXiv:
    https://arxiv.org/abs/2004.07211

### 代码
- 官方代码后续主要并入 Mammoth：
  - https://github.com/aimagelab/mammoth

### 备注
- 你截图里把 `DER` 放在“有范例的增量方法”里，并描述为“利用保存的旧类范例进行重放”，这更符合 `Dark Experience Replay`
- 但 `DER` 在增量学习文献里还有另一个常见含义：
  - **DER: Dynamically Expandable Representation for Class Incremental Learning**
  - CVPR 2021
  - 论文：https://openaccess.thecvf.com/content/CVPR2021/html/Yan_DER_Dynamically_Expandable_Representation_for_Class_Incremental_Learning_CVPR_2021_paper.html
  - 官方代码：https://github.com/Rhyssiyan/DER-ClassIL.pytorch
- 如果你师姐论文里的 `[44]` 实际上引用的是 CVPR 2021 这篇，那么需要把 DER 条目切换成上面这个版本

## 7. 建议你论文里怎么写

### 如果你要完全复现师姐这套对比
- 非增量分类方法：
  - ResNetAM（建议写全称 `HResNetAM`）
- 无回放增量方法：
  - LwF
  - SSRE
- 回放类增量方法：
  - GFR
  - iCaRL
  - DER

### 更稳妥的写法
- 在正文里首次出现时直接写：
  - `HResNetAM`
  - `LwF`
  - `GFR`
  - `SSRE`
  - `iCaRL`
  - `DER (Dark Experience Replay)`
- 这样可以避免后面读者对 `ResNetAM` 和 `DER` 的歧义
