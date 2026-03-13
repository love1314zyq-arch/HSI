# Zhu_Self-Sustaining_Representation_Expansion_for_Non-Exemplar_Class-Incremental_Learning_CVPR_2022_paper

Source: `Zhu_Self-Sustaining_Representation_Expansion_for_Non-Exemplar_Class-Incremental_Learning_CVPR_2022_paper.pdf`

Self-Sustaining Representation Expansion for
                              Non-Exemplar Class-Incremental Learning

               Kai Zhu1        Wei Zhai1         Yang Cao1,3,†   Jiebo Luo2       Zheng-Jun Zha1
                1                                                       2
                  University of Science and Technology of China           University of Rochester
              3
                Institute of Artificial Intelligence, Hefei Comprehensive National Science Center
 {zkzy@mail., wzhai056@mail., forrest@}ustc.edu.cn                        jluo@cs.rochester.edu               zhazj@ustc.edu.cn

                           Abstract

   Non-exemplar class-incremental learning is to recog-
nize both the old and new classes when old class sam-
ples cannot be saved. It is a challenging task since rep-
resentation optimization and feature retention can only be
achieved under supervision from new classes. To address
this problem, we propose a novel self-sustaining representa-
tion expansion scheme. Our scheme consists of a structure
reorganization strategy that fuses main-branch expansion
and side-branch updating to maintain the old features, and
a main-branch distillation scheme to transfer the invari-
ant knowledge. Furthermore, a prototype selection mecha-
nism is proposed to enhance the discrimination between the
old and new classes by selectively incorporating new sam-
ples into the distillation process. Extensive experiments on
three benchmarks demonstrate significant incremental per-           Figure 1. The t-SNE visualization. Compared to the baseline in
formance, outperforming the state-of-the-art methods by a           Section 4.1, (1) the representations of the old classes in our method
margin of 3%, 3% and 6%, respectively.                              are better maintained (circular area), (2) and the novel class is
                                                                    more discriminating from the old classes (rectangular area).

1. Introduction
                                                                    the entire representation and the classifier become biased
    Since deep neural networks have made great advances             toward the new class, resulting in a sharp drop in the perfor-
in fully supervised conditions, research attention is increas-      mance for the old class. To deal with it, recent CIL methods
ingly turning to other aspects of learning. An important as-        maintain the past knowledge by preserving some represen-
pect is the ability to continuously learn new tasks as the          tative samples (i.e., exemplars [27]) and introducing various
input stream is updated, which is often the case in real ap-        distillation losses [7], and correct the bias caused by number
plications. In recent years, class-incremental learning (CIL)       imbalance by calibrating the classifier [11].
[11, 27], a difficult type in continual learning, has attracted
                                                                        However, most of the existing methods [21, 30] assume
much attention, which aims to recognize new classes with-
                                                                    that a certain number (e.g., 2000) of exemplars can be stored
out forgetting the old ones that have been learned.
                                                                    in memory, which is usually difficult to satisfy in practice
    In this case, re-training the old and new class samples
                                                                    due to user privacy or device limitations. This fact poses
jointly in each phase is time-consuming and laborious, not
                                                                    great difficulties to incremental learning, because the opti-
to mention that the old class samples may not be fully avail-
                                                                    mization of the representation and the correction of the clas-
able. A simple alternative is to fine-tune the network using
                                                                    sifier will degenerate directly from the imbalance between
the new class, however, it will cause the catastrophic forget-
                                                                    the old and new classes. To this end, this paper focuses
ting problem [8]. That is, during the optimization process,
                                                                    on this ability of incrementally learning new classes where
  † Corresponding Author                                            old class samples cannot be preserved, which is called non-

                                                                  9296

---

Figure 2. Motivation of our method. In NECIL, the rehearsal-based and structure-based methods suffer from the unreliability of distillation
in the absence of exemplars and continuously expanding structure, respectively. DSR is proposed to drive network to expand from a
structurally recoverable direction, thus maintaining the discrimination during the new optimization process. On this foundation, we utilize
MBD to exploit the ability of distillation-based methods to balance old and new class knowledge.

exemplar class-incremental learning (NECIL [44]).                       class may directly overlap with the old cluster, resulting in
    A natural idea for this problem is to directly transfer the         serious confusion to the subsequent optimization process.
existing CIL framework (i.e., rehearsal-based and structure-                To address this problem, we propose a self-sustaining
based methods in Section 2.1) to NECIL, but the exper-                  representation expansion scheme to learn a structure-cyclic
imental results show that this way leads to performance                 representation, promoting the optimization from the ex-
degradation and parameter explosion. On one hand, in                    panded direction while integrating the overall structure at
rehearsal-based methods, due to the lack of old class sam-              the end of each phase. As shown in Fig. 2, the preservation
ples, the distillation that the new class samples participate in        of the old classes is reflected in both the structure and fea-
is the only one that can help maintain the representation of            ture aspects. First, we adopt a dynamic structure reorgani-
old classes. However, for new samples, it is impractical to             zation (DSR) strategy, which leaves structured space for the
provide the same complete old class distribution as the ex-             learning of new class while stably preserving the old class
emplars, so it is difficult to effectively promote the knowl-           space through maintaining heritage at the main-branch and
edge transfer in the distillation process. Consequently, rep-           fusing update at the side-branch. Second, on the basis of the
resentative features learned in the old phase are lost phase            expandable structure, we employ a main-branch distillation
by phase with the decrease of relevance to the new class.               (MBD) to maintain the discrimination of the new network
    On the other hand, the idea of structure-based methods              with respect to the old features by aligning the invariant dis-
is to leave the old model for inference and expand a new                tribution knowledge on the old classes.
model for training at each new phase [28,35]. Although this                 Specifically, we insert a residual adapter in each block
strategy maintains the performance of the old class com-                of the old feature extractor to map the old representation to
pletely, demonstrating strong performance [35], the net-                a high-dimensional embedding space, forcing the optimiza-
work parameters that increase linearly with phase (i.e., 5, 10          tion flow to only pass through the expanding branches unre-
and 20 in this paper) during training are discouraging. Be-             lated to the old class. After the optimization, we adopt the
sides, although a large amount of data can be used to learn             structural re-parameterization technique to fuse the old and
the discriminative features among new classes, it is easy to            new features and map them back to the initial space loss-
confuse with similar ones from the old distribution. The                lessly. Furthermore, to reduce the confusion between the
augmentation of prototypes [44] can only improve the se-                newly incremental classes and the original classes, we add
lection of the optimal boundary for the classifier, but cannot          a prototype selection mechanism (PSM) during the distilla-
essentially improve the discrimination of the old and new               tion process. The normalized cosine is first used to measure
classes in the feature representation. As shown in Fig. 1, the          the similarity between the new representation and the old
representations of old classes obtained by the standard CIL             prototype. Then samples similar to the old classes are used
method are more confused compared to the initial phase, be-             for distillation, maintaining the old knowledge with a soft
cause they may gradually overlap with similar classes due to            label that retains the old class statistical information, while
the lack of effective supervision. At the same time, the new            those samples dissimilar to the old classes are used for new

                                                                     9297

---

class training. This mechanism improves the performance            tic drift of incremental features and compensates the proto-
of forward transfer and mitigates the lack of joint optimiza-      types in each test phase. [44] adopts prototype augmentation
tion to some extent. Our main contributions are as follows:        to maintain the decision boundary of previous tasks, and
   1) A self-sustaining representation expansion scheme is         employ self-supervised learning to learn more transferable
proposed for non-exemplar incremental learning, in which           features for future tasks. We follow their NECIL settings.
a cyclically expanding optimization is accomplished by a           However, different from their work considering generaliz-
dynamic structure reorganization strategy, resulting in a          able features and augmented prototypes, we mainly con-
structure-invariant representation.                                sider the adjustment for joint representation learning and
   2) A prototype selection mechanism is proposed, which           distillation process in the absence of exemplars.
combinatorially co-uses the preserved invariant knowledge
and the incoming new supervision to reduce the feature con-
                                                                   2.2. Residual Block
fusion among the similar classes.                                      Residual block has been widely used in convolutional
   3) Extensive experiments are performed on bench-                neural network as the basic structure of ResNet [10], which
marks including CIFAR-100, TinyImageNet and ImageNet-              improves the network depth and prevents vanishing gradi-
Subset, and the results demonstrate the superiority of our         ent. Further improvements have been investigated for su-
method over the state of the art.                                  perior dynamic performance [19] and inference efficiency
                                                                   [5, 6, 9] recently. In domain adaptation [20, 43], residual
2. Related Work                                                    adapter [18, 25, 26] is proposed to learn style information
                                                                   related to new domains, thus improving the overall general-
2.1. Incremental Learning                                          ization performance of the network. In these efforts, resid-
    As deep learning research advances [34, 41], there is a        ual block is used to improve joint optimization performance
growing demand for continual learning of neural networks,          or statiscal domain information. Instead, we consider dy-
which requires the network to learn new tasks without for-         namically incremental residual blocks to learn new knowl-
getting the old knowledge to achieve the stability-plasticity      edge efficiently while maintaining old features.
trade-off. CIL [32, 45] is the most difficult scenery in con-
tinual learning and has received more attention recently.          3. Problem Description
Current methods can broadly divided into the following                 The NECIL problem is defined as follows. Here we de-
three classes. Regularization-based methods [15, 39] esti-         note X, Y and Z as the training set, the label set and the
mate the importance of the network parameters learned in           test set, respectively. Our task is to train the model from a
the past tasks and constrain their optimization accordingly.       continuous data stream, i.e., training sets X 1 , X 2 , · · · X n ,
Rehearsal-based methods [1, 7, 12, 22, 33] preserve exem-          where samples of a set X i (1 ≤ i ≤ n) are from the label
plars of fixed memory size to maintain the distribution of         set Y i , and n represents the incremental phase. It should be
old classes in the incremental phases, and adopt the distil-       mentioned that all the incremental classes are disjoint, that
lation skills to retain the discriminative features of the old     is, Y i ∩ Y j = ∅(i ̸= j). Except that there are sufficient
task. [11] incorporates three components, cosine normal-           samples in the current phase X i , no old samples are avail-
ization, less-forget constraint, and inter-class separation,       able in memory for old classes. To measure the performance
to address the imbalance between previous and new data.            of models in NECIL task, we calculate the classification ac-
The techniques on exemplar and distillation in rehearsal-          curacy on the test set Z i at each phase i. Different from the
based methods are widely used in class-incremental learn-          training set, the classes             set Z i are from all the
ing. Structure-based methods [13,29] select and expand dif-                           1
                                                                                        S 1 of theS test
                                                                                                      i
                                                                   seen label sets Y      Y ··· Y .
ferent sub-network structures involved in the optimization
process of the incremental tasks. [24] progressively chooses       4. Methodology
optimal paths for the new tasks while encouraging param-
eter sharing, which promotes the forward knowledge trans-              First of all, we demonstrate the paradigms of standard
fer. [35] freezes the previously learned representation and        CIL and how we adapt it to the NECIL setting as the base-
augment it with additional feature dimensions from a new           line. Then we analyze the optimization flow of the over-
mask-based feature extractor. The structure-based methods          all pipeline and explain why it doesn’t work well. Finally,
are often mixed with other techniques such as exemplar and         two proposed core components dynamic structure reorgani-
distillation, and have achieved good results.                      zation and prototype selection are introduced.
    Recently, some works [36, 38, 44] focus on a challeng-
                                                                   4.1. Standard NECIL Paradigm
ing but practical non-exemplar class-incremental learning
problem, where no past data can be stored due to equip-                [27] first proposed a practical strategy for decoupling
ment limits or privacy security. [38] estimates the seman-         representation and classifiers learning in the CIL setting,

                                                                 9298

---

                                             (Eq. 8)

                                             (Eq. 9)

Figure 3. Our proposed self-sustaining representation expansion scheme for NECIL: (a) overview of our scheme, (b) dynamic structure
reorganization, and (c) main-branch distillation and prototype balance. The source code will be made available to the public.

which is followed by most subsequent work. The three                sentation to the label space,
main components are representation learning using knowl-
                                                                                snq = gcn (rqn ; θcn ), Lce = Fce (snq , yqn ),   (2)
edge distillation and prototype rehearsal, prioritized exem-
plar selection, and classification by a balance calibration.           Fce represents the standard cross-entropy loss. Finally,
   Incremental Representation Learning. As the exem-                to maintain the useful information learned with the old
plar cannot be saved in the NECIL setting, the representa-          classes, the knowledge distillation is used to measure the
tion learning will be slightly different, mainly in terms of        similarity between the obtained representation and that of
cross-entropy loss and distillation loss. At the first phase,       the previous model fen−1 ,
a standard classification model fθ1 consisting of the fea-               rqn−1 = fen−1 (Q; θen−1 ), Lkd = Fkd (rqn , rqn−1 ),     (3)
ture extractor fe1 and classifier gc1 is optimized under the
                                                                    Fkd represents Euclidean distance the same as [44].
full supervision, i.e., X 1 and Y 1 . At the incremental phase
                                                                       Incremental Classifier Calibration. A common way
(n > 1), the input of the current model is only the predicted
                                                                    to overcome the imbalance between the exemplars and new
images Q from X n without old samples. A base feature ex-
                                                                    samples in CIL is to under-sample a balanced subset for
tractor fen such as VGG [31] or ResNet [10] parameterized
                                                                    fine-tuning. As there is no exemplars in NECIL, we memo-
by θen is utilized to learn the corresponding representation:
                                                                    rize one prototype in the deep feature space for each class,
                                                                    which is consistent with PASS [44]. Different from PASS
                      rqn = fen (Q; θen ).                 (1)
                                                                    augmenting the prototypes via Gaussian noise, we choose
                                                                    to over-sample (i.e., U pB ) prototypes to the batch size (i.e.,
Then, to learn the discriminative features among the novel          B), achieving the calibration of the classifier, which is the
classes, the obtained representation is optimized under the         simplest way adopted in the long-tail recognition [2],
supervision of the class label ynq from Y n . We adopt the
fully connected layer as the classifier gcn to map the repre-           pB = U pB (P rototype), Lproto = Fce (pB , yB ),          (4)

                                                                 9299

---

yB is the over-sampled label set of the initial prototypes.           where ˆ represents the fixed parameters, and ⊕ represents
The final loss for current model is their addition:                   the structural expansion operation. After training, we use
                                                                      the structural re-parameterization [6] to integrate the side-
                L = Lce + λLkd + γLproto ,                    (5)     branch information into the main branch losslessly, ensur-
λ and γ are loss weights, and we set them to 10.                      ing that the number of network parameters does not increase
                                                                      at the end of each phase. Specifically, the parameters in
4.2. Optimization                                                     the residual structures are fused with the parameters of the
   Different from the previous work focusing on the effect            original convolution kernel and BatchNorm [14] through
on the classifier, this paper tries to analyze the representa-        the zero-padding operation and linear transformation, and
tion. In the CIL, Equation 2 can be turned into two parts:            finally the adapters are removed to keep the network struc-
                                                                      ture unchanged for the next update,
             Lce = Fce (snq , yqn ) + Fce (sne , yen ),       (6)
                                                                            fen−1 (Q; θ̂en−1 ⊕ ∆θen )
sne represents the saved exemplars, whose number is much                                                                               (9)
lower than that of snq . While this imbalance can bias the op-                              = fen (Q; θen ′ ⊕ 0) = fen (Q; θen ′ ).
timization process towards features that are more discrimi-               Prototype Selection. While new features are learned
native for the new class, the added distillation in Equation 3        based on the old structure, the old class features are main-
can alleviate this problem,                                           tained in coordination with the main-branch distillation. To
          Lkd = Fkd (rqn , rqn−1 ) + Fkd (ren , ren−1 ).      (7)     reduce the feature confusion in the distillation part, we
                                                                      adopt a prototype selection mechanism based on the ex-
ren represents the representation of exemplars. In this case,         pandable embedding space. In general, based on the sim-
the features that are significant for the old and new classes         ilarity between the representation of new samples and the
will be maintained. However, note that there is no exem-              old prototypes, dissimilar samples are involved in the up-
plars involved in the above NECIL setting. It means that              date of residual adapter to learn new features, and similar
the joint optimization on the old and new class representa-           samples are involved in the distillation to retain the old dis-
tions completely collapses into feature optimization that is          criminative features maintained in the main branch at pre-
relevant only to incremental classes. What is reflected in            vious phase. Specifically, after mapping all new samples to
the first part is that the cross-entropy loss will only focus on      the learned embedding space [3, 40], we compute the nor-
the features that facilitate the recognition of the new class,        malized cosine scores Si between them and all prototypes,
while in the second part, it will focus on the maintenance
of the features related to the new class, which both accel-                      Si = Cosine(N (rqn ), N or(P rototype)),             (10)
erate the forgetting of the representative features of the old        Nor represents the normalization operation. We then set a
class. Suppose the forgetting rate (Fr) of distillation part in       threshold value, and attach a mask to the corresponding po-
the initial phase is α. Note that the distillation loss is based      sition of its distillation loss (M askkd ) if greater than the
on the overall representation of the previous phase, and the          threshold σ, and add a mask to the corresponding part of its
error will accumulate exponentially with the phase, that is,          cross-entropy loss (M askce ) if less than the threshold. Fi-
F rn ≥ αn−1 . Therefore, it is necessary to correct this error        nally, the two loses are summed with the prototype balance
from the representational level.                                      loss as the final optimization function for the new phase.
4.3. Self-Sustaining Representation Expansion
                                                                           L = M askce (Lce ) + λM askkd (Lkd ) + γLproto .           (11)
    Dynamic Structure Reorganization. To retain the rep-
resentation [42] of the old class and guarantee the unbiased          5. Experiments
training of the new class, we propose a dynamic structure             5.1. Dataset and Settings
reorganization strategy. In general, as shown in Fig. 3, we
firstly adopt the structural expansion to add the side branch             Dataset. To evaluate the performance of our pro-
to the current model by block for the optimization of new             posed method, we conduct comprehensive experiments on
classes. Specifically, we insert a residual adapter to each           three datasets CIFAR-100 [16], TinyImageNet [17] and
convolution block of the fixed feature extractor from pre-            ImageNet-Subset. CIFAR-100 contains 60000 images of
vious phase. The optimized flow propagates only through               32 × 32 size from 100 classes, and each class includes 500
the adapter, updating the most discriminating position while          training images and 100 test images. TinyImageNet con-
maintaining the old features,                                         tains 200 classes, and each class contains 500 training im-
                                                                      ages, 50 validation images and 50 test images. It provides
 fen (Q; θen ) = Ftransf orm (fen−1 (Q; θen−1 ))                      more phases and incremental classes to compare the sensi-
                                                              (8)
                               = fen−1 (Q; θ̂en−1 ⊕ ∆θen ),           tivity of different methods. ImageNet-Subset is a 100-class

                                                                    9300

---

                                                                                  68
                                       CIFAR-100                                              5 phases
  DSR     MBD      PSM                                                                                                                65.88         65.45
                            5 phases   10 phases 20 phases                        66
                                                                                              10 phases               64.25

                                                                       Accuracy
                             61.11       57.08     51.04                                      20 phases
   √                                                                              64                           61.8                       65.04
                             64.86       63.25     54.09                                                                                            64.18
           √                                                                           61.5                                   63.63       61.7
                             62.70       62.60     58.57                          62
   √       √                                                                                                             60.16                      60.21
                             65.10       63.87     60.60                                  61.8       62.05
   √       √        √                                                             60
                             65.88       64.69     61.61                                 58.56        58.57

                                                                                  58
                                                                                          0.5            0.6              0.7                 0.8   0.85
     Table 1. Ablation study of our method on CIFAR-100.
                                                                                                                      Threshold

                                    CIFAR-100                          Figure 4. Illustration of the role of the selection mechanism.
      Method            5 phases    10 phases      20 phases
     3×3 conv            64.28        63.47          60.81
   1×1 conv + bn         65.88        64.84          60.72
     1×1 conv            65.87        65.12          61.60

   Table 2. Performance under different expanding structures.

                                                                      (a) Finetuning       (b) iCaRL            (c) Ours
subset of ImageNet-1k [4], which is much larger. For the          Figure 5. Confusion matrices of different methods on CIFAR-100.
order and division of all dataset classes in our experiments,
we followed exactly the settings in [44].
    Setting. As adopted in [44], we use ResNet-18 as the          dynamic representation and main-branch distillation sepa-
backbone network. The difference is that we use standard          rately bring a 4.3% and 4.8% improvement in overall per-
supervised training for the whole optimization process in-        formance. It demonstrates that the two parts are far more
stead of involving self-supervised learning. For a fair com-      useful than standard representation and common distillation
parison, we achieve the same accuracy as [44] at the first        respectively in NECIL. It is worth noting that the former
phase for all datasets. We use an Adam optimizer, in which        plays a greater role when there are fewer incremental phases
the initial learning rate is set to 0.001 and the attenuation     (i.e., 5 and 10 phases), while the latter shines more brightly
rate is set to 0.0005. The model stops training after 100         when there are more incremental phases (i.e., 20 phases).
epochs, and batch size is set to 128.                             It demonstrates that keeping the old class features helps to
                                                                  improve the overall performance of incremental learning in
    Evaluation Metrics. Following [44], we report average
                                                                  the short term. However, as analyzed in the introduction,
incremental accuracy and average forgetting, and our per-
                                                                  if the distilled network keep decaying or fixed, the errors
formance is evaluated on three different runs. Average in-
                                                                  will accumulate as the incremental phase increases. At this
cremental accuracy is computed as the average accuracy of
                                                                  point, how to reasonably correct the distillation loss is the
all the incremental phases (including the first phase), which
                                                                  key to ensure the long-term effect.
compares the overall incremental performance of different
methods fairly. Average forgetting is computed as the av-         5.3. Analysis
erage forgetting of different tasks throughout the incremen-
tal process, which directly measures the ability of different        The impact of the adapter structure. To explore the
methods to resist catastrophic forgetting.                        impact of the structure of residual adapter on expandable
                                                                  representation during training, we design the following ex-
5.2. Ablation Study                                               periments. We adopt three different convolution blocks to
                                                                  the residual part: 1×1 convolution only, 3×3 convolution
   To prove the effectiveness of our proposed method,             only and the combination of 1×1 convolution and Batch-
we conduct several ablation experiments on CIFAR-100.             Norm. As shown in Table 2, the results of 1×1 convolution
The performance of our scheme is mainly attributed to             and the combination are similar, and that of 3×3 convolu-
two prominent components: the dynamic structure reor-             tion is one point lower. It suggests that the 1×1 convolution
ganization strategy (DSR) and the main-branch distillation        structure is good enough to learn the representation of the
(MBD). To clarify the function of DSR, we replace the dy-         new class without needing more parameters.
namic representation with the structurally invariant repre-
sentation, which is adopted in most CIL methods [7, 44].             The impact of the threshold in prototype selection. To
To clarify the function of MBD, we replace the distillation       verify the role of the PSM, we conduct data statistics on the
process with the one that interacts with the continuously         incremental samples. As shown in Fig. 8, the new classes
optimized representation. As can be seen in Table 1, the          have a large difference in similarity. And the intra-class

                                                                9301

---

                                        (a)                                                             (b)
Figure 6. Effect of our scheme on the representation. (a) DSR maintains the discriminative features and inter-relations of old classes, thus
enhancing the clustering and separation of the distribution of old classes. (b) MBD results in a better distinction between similar classes.

                                              CIFAR-100                                TinyImageNet                ImageNet-Subset
                Methods
                                    P=5         P=10       P=20              P=5           P=10       P=20              P=10
               iCaRL-CNN∗          51.07        48.66      44.43            34.64          31.15      27.90             50.53
    (1) E=20

               iCaRL-NCM∗ [27]     58.56        54.19      50.51            45.86          43.29      38.04             60.79
               EEIL∗ [1]           60.37        56.05      52.34            47.12          45.01      40.50             63.34
               UCIR∗ [11]          63.78        62.39      59.07            49.15          48.52      42.83             66.16
               EWC∗ [15]           24.48        21.20      15.89            18.80          15.77      12.39             20.40
               LwF_MC∗ [27]        45.93        27.43      20.07            29.12          23.10      17.43             31.18
    (2) E=0

               MUC∗ [37]           49.42        30.19      21.27            32.58          26.61      21.95             35.07
               SDC [38]            56.77        57.00      58.90              -              -          -               61.12
               PASS [44]           63.47        61.84      58.09            49.55          47.29      42.07             61.80
               Ours              65.88+2.41   65.04+3.20 61.70+2.80       50.39+0.84     48.93+1.64 48.17+6.10        67.69+5.89
Table 3. Comparisons of the average incremental accuracy (%) with other methods on CIFAR-100, TinyImageNet, and ImageNet-Subset.
P represents the number of phases and E represents the number of exemplars. Models with an asterisk ∗ represent the reproduced results
in [44]. The red footnotes in the last row represent the relative improvement compared with the results of SOTA.

fluctuations are also large, so different classes and sam-               between the old and new classes without favoring one side
ples involved in the optimization process will bring differ-             due to overfitting, which is a prerequisite for a good incre-
ent changes. Therefore it is important to reasonably place               mental learning system.
them in the two potentially conflicting processes of old fea-
ture distillation and new feature learning. To demonstrate               5.4. Visualization
the sensitivity of the threshold on the distillation effect, we
plot its fluctuation curve. As shown in Fig. 4, all curves rise              To better demonstrate the role of DSR and MBD during
to a peak at a threshold of 0.8, then gradually fall and lose            optimization, we show the visualization results with t-SNE
distillation effect. It suggests that in the absence of the ex-          [23]. As shown in Fig. 6 (a), although the old classes have
emplars, fine-grained optimization of the new samples can                slightly changed in the representation after multi-phase op-
better maintain the old features and learn the new features.             timization, their discrimination and relative relationship al-
                                                                         most do not decline with our DSR. As shown in Fig. 6 (b),
    Classification accuracy of old and novel classes. To                 newly incremental classes are easily confused with some of
evaluate performance of both old and new classes during                  the old classes. Owing to our MBD, the optimized features
training, we compare their accuracy at each phase. As                    are promoted to differentiate from the old class, thus im-
shown in Fig. 5, our method achieves similar performance                 proving the seperation of novel clusters.

                                                                     9302

---

         (a) 5 phases CIFAR-100          (b) 10 phases CIFAR-100          (c) 20 phases CIFAR-100

       (d) 5 phases TinyImagNet        (e) 10 phases TinyImagNet (f) 20 phases TinyImagNet (g) 10 phases ImagNet-Subset
      Figure 7. Classification accuracy on CIFAR-100, TinyImageNet and ImageNet-Subset, which contains the complete curves.

                                                                                                  CIFAR-100                TinyImageNet
                                                                              Method            5     10    20           5      10    20
                                                                              iCaRL-CNN       42.13 45.69 43.54       36.89 36.70 45.12
                                                                              iCaRL-NCM       24.90 28.32 35.53       27.15 28.89 37.40
                                                                              EEIL            23.36 26.65 32.40       25.56 25.91 35.04
                                                                              UCIR            21.00 25.12 28.65       20.61 22.25 33.74
                                                                              LwF_MC          44.23 50.47 55.46       54.26 54.37 63.54
                                                                              MUC             40.28 47.56 52.65       51.46 50.21 58.00
     (a) Mean of similarity     (b) Standard deviation of similarity          PASS            25.20 30.25 30.61       18.04 23.11 30.55
  Figure 8. Statistics of similarity on the incremental samples.              Ours            18.37 19.48 19.00        9.17 14.06 14.20

                                                                              Table 4. Results of average forgetting on 5, 10 and 20 phases.
5.5. Comparison with SOTA
   To better assess the overall performance of our scheme,               larger ImageNet-Subset dataset, our method has a notable
we compare it to the SOTA of NECIL (EWC∗ , LwF_MC∗ ,                     advantage, demonstrating its robustness.
MUC∗ , SDC and PASS) and some classical methods of
exemplar-based CIL (iCARL∗ , EEIL∗ and UCIR∗ ).
                                                                         6. Conclusion and Discussion
   Average accuracy and average forgetting. As shown in
Table 3, compared to the SOTA of non-exemplar methods                        In this paper, a novel self-sustaining representation ex-
(E=0), our method achieves average improvement of 3, 3                   pansion scheme is presented for the NECIL task. A dy-
and 6 points on CIFAR-100, TinyImageNet and ImageNet-                    namic structure reorganization strategy is first proposed to
Subset, respectively. The performance of our method                      optimize the newly incremental features in a side branch
is comparable to the classical exemplar-based methods                    while maintaining the old feature distribution from the
(E=20), which shows that our scheme further reduces the                  structurally expanded direction, and then the distillation
impact of exemplars on CIL models. To provide further in-                process is arranged in the main branch. In particular, a
sight into the behaviors of different methods, we compare                prototype selection mechanism is integrated into the joint
their average forgetting of all phases. As shown in Table 4,             training to enhance the distinction between the old and new
our method achieves much lower average forgetting, resist-               classes. Experimental results show that our method is su-
ing catastrophic forgetting well in the absence of exemplars.            perior in both performance and adaptability to the state-of-
   Trend of accuracy. To analyze the trend of different                  the-art methods, especially in the multi-phase process.
methods, we show the detailed accuracy curves on three                       Acknowledgments. Supported by National Key R&D
datasets. As shown in Fig. 7, our method is superior at                  Program of China under Grant 2020AAA0105700, Na-
almost all phases, striking a better stability-plasticity bal-           tional Natural Science Foundation of China (NSFC) under
ance. It can be seen that the difficulty increases as the num-           Grants 61872327 and U19B2038, Major Special Science
ber of incremental phases (P) increases. In this process, the            and Technology Project of Anhui (No. 012223665049), the
advantage of our method are even expanding, such as in                   University Synergy Innovation Program of Anhui Province
TinyImageNet. Whether in the smaller CIFAR-100 or the                    under Grants GXXT-2019-025.

                                                                       9303

---

References                                                                  ing, picking and growing for unforgetting continual learning.
                                                                            ArXiv, abs/1910.06562, 2019.
 [1] Francisco M Castro, Manuel J Marín-Jiménez, Nicolás Guil,
                                                                       [14] Sergey Ioffe and Christian Szegedy. Batch normalization:
     Cordelia Schmid, and Karteek Alahari. End-to-end incre-
                                                                            Accelerating deep network training by reducing internal co-
     mental learning. In Proceedings of the European conference
                                                                            variate shift. In International conference on machine learn-
     on computer vision (ECCV), pages 233–248, 2018.
                                                                            ing, pages 448–456. PMLR, 2015.
 [2] Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, and             [15] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel
     W Philip Kegelmeyer. Smote: synthetic minority over-                   Veness, Guillaume Desjardins, Andrei A Rusu, Kieran
     sampling technique. Journal of artificial intelligence re-             Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-
     search, 16:321–357, 2002.                                              Barwinska, et al. Overcoming catastrophic forgetting in neu-
 [3] Zhen Cheng, Zhiwei Xiong, Chang Chen, Dong Liu, and                    ral networks. Proceedings of the national academy of sci-
     Zheng-Jun Zha. Light field super-resolution with zero-shot             ences, 114(13):3521–3526, 2017.
     learning. In Proceedings of the IEEE/CVF Conference on            [16] A. Krizhevsky. Learning multiple layers of features from
     Computer Vision and Pattern Recognition, pages 10010–                  tiny images. 2009.
     10019, 2021.
                                                                       [17] Ya Le and Xuan Yang. Tiny imagenet visual recognition
 [4] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li,                 challenge. CS 231N, 7(7):3, 2015.
     and Li Fei-Fei. Imagenet: A large-scale hierarchical image        [18] Wei-Hong Li, Xialei Liu, and Hakan Bilen. Improving
     database. In 2009 IEEE conference on computer vision and               task adaptation for cross-domain few-shot learning. ArXiv,
     pattern recognition, pages 248–255. Ieee, 2009.                        abs/2107.00358, 2021.
 [5] Xiaohan Ding, Yuchen Guo, Guiguang Ding, and J. Han. Ac-          [19] Yunsheng Li, Yinpeng Chen, Xiyang Dai, Mengchen Liu,
     net: Strengthening the kernel skeletons for powerful cnn via           Dongdong Chen, Ye Yu, Lu Yuan, Zicheng Liu, Mei Chen,
     asymmetric convolution blocks. 2019 IEEE/CVF Interna-                  and Nuno Vasconcelos. Revisiting dynamic convolution via
     tional Conference on Computer Vision (ICCV), pages 1911–               matrix decomposition. arXiv preprint arXiv:2103.08756,
     1920, 2019.                                                            2021.
 [6] Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han,            [20] Jiawei Liu, Zheng-Jun Zha, Di Chen, Richang Hong, and
     Guiguang Ding, and Jian Sun. Repvgg: Making vgg-style                  Meng Wang. Adaptive transfer network for cross-domain
     convnets great again. In Proceedings of the IEEE/CVF Con-              person re-identification. In Proceedings of the IEEE/CVF
     ference on Computer Vision and Pattern Recognition, pages              Conference on Computer Vision and Pattern Recognition,
     13733–13742, 2021.                                                     pages 7202–7211, 2019.
 [7] Arthur Douillard, Matthieu Cord, Charles Ollion, Thomas           [21] Yaoyao Liu, Bernt Schiele, and Qianru Sun. Adaptive aggre-
     Robert, and Eduardo Valle. Podnet: Pooled outputs dis-                 gation networks for class-incremental learning. In Proceed-
     tillation for small-tasks incremental learning. In Computer            ings of the IEEE/CVF Conference on Computer Vision and
     Vision–ECCV 2020: 16th European Conference, Glasgow,                   Pattern Recognition, pages 2544–2553, 2021.
     UK, August 23–28, 2020, Proceedings, Part XX 16, pages            [22] Yaoyao Liu, Yuting Su, An-An Liu, Bernt Schiele, and
     86–102. Springer, 2020.                                                Qianru Sun. Mnemonics training: Multi-class incremental
 [8] Robert M French. Catastrophic forgetting in connectionist              learning without forgetting. In Proceedings of the IEEE/CVF
     networks. Trends in cognitive sciences, 3(4):128–135, 1999.            Conference on Computer Vision and Pattern Recognition,
 [9] Shuxuan Guo, José Manuel Álvarez, and Mathieu Salzmann.                pages 12245–12254, 2020.
     Expandnets: Linear over-parameterization to train compact         [23] Laurens van der Maaten and Geoffrey Hinton. Visualiz-
     convolutional networks. arXiv: Computer Vision and Pattern             ing data using t-sne. Journal of machine learning research,
     Recognition, 2020.                                                     9(Nov):2579–2605, 2008.
[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.            [24] Jathushan Rajasegaran, Munawar Hayat, Salman Khan, Fa-
     Deep residual learning for image recognition. In Proceed-              had Shahbaz Khan, and Ling Shao. Random path selection
     ings of the IEEE conference on computer vision and pattern             for incremental learning. Advances in Neural Information
     recognition, pages 770–778, 2016.                                      Processing Systems, 2019.
[11] Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, and           [25] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi.
     Dahua Lin. Learning a unified classifier incrementally via             Learning multiple visual domains with residual adapters.
     rebalancing. In Proceedings of the IEEE/CVF Conference on              arXiv preprint arXiv:1705.08045, 2017.
     Computer Vision and Pattern Recognition, pages 831–839,           [26] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi.
     2019.                                                                  Efficient parametrization of multi-domain deep neural net-
[12] Xinting Hu, Kaihua Tang, Chunyan Miao, Xiansheng Hua,                  works. 2018 IEEE/CVF Conference on Computer Vision and
     and Hanwang Zhang. Distilling causal effect of data in class-          Pattern Recognition, pages 8119–8127, 2018.
     incremental learning. 2021 IEEE/CVF Conference on Com-            [27] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg
     puter Vision and Pattern Recognition (CVPR), pages 3956–               Sperl, and Christoph H Lampert. icarl: Incremental classifier
     3965, 2021.                                                            and representation learning. In Proceedings of the IEEE con-
[13] Steven C. Y. Hung, Cheng-Hao Tu, Cheng-En Wu, Chien-                   ference on Computer Vision and Pattern Recognition, pages
     Hung Chen, Yi-Ming Chan, and Chu-Song Chen. Compact-                   2001–2010, 2017.

                                                                     9304

---

[28] Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins,                tive bias for image recognition and beyond. arXiv preprint
     Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Raz-               arXiv:2202.10108, 2022.
     van Pascanu, and Raia Hadsell. Progressive neural networks.       [42] Heliang Zheng, Jianlong Fu, Zheng-Jun Zha, and Jiebo Luo.
     arXiv preprint arXiv:1606.04671, 2016.                                 Learning deep bilinear transformation for fine-grained image
[29] Andrei A. Rusu, Neil C. Rabinowitz, Guillaume Desjardins,              representation. Advances in Neural Information Processing
     Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Raz-               Systems, 32, 2019.
     van Pascanu, and Raia Hadsell. Progressive neural networks.       [43] Kecheng Zheng, Wu Liu, Lingxiao He, Tao Mei, Jiebo Luo,
     ArXiv, abs/1606.04671, 2016.                                           and Zheng-Jun Zha. Group-aware label transfer for do-
[30] Christian Simon, Piotr Koniusz, and Mehrtash Harandi. On               main adaptive person re-identification. In Proceedings of
     learning the geodesic path for incremental learning. In Pro-           the IEEE/CVF Conference on Computer Vision and Pattern
     ceedings of the IEEE/CVF Conference on Computer Vision                 Recognition, pages 5310–5319, 2021.
     and Pattern Recognition, pages 1591–1600, 2021.                   [44] Fei Zhu, Xu-Yao Zhang, Chuang Wang, Fei Yin, and Cheng-
[31] K. Simonyan and Andrew Zisserman. Very deep convolu-                   Lin Liu. Prototype augmentation and self-supervision for
     tional networks for large-scale image recognition. CoRR,               incremental learning. In Proceedings of the IEEE/CVF Con-
     abs/1409.1556, 2015.                                                   ference on Computer Vision and Pattern Recognition, pages
[32] Gido M. van de Ven and A. Tolias. Three scenarios for con-             5871–5880, 2021.
     tinual learning. ArXiv, abs/1904.07734, 2019.                     [45] Kai Zhu, Yang Cao, Wei Zhai, Jie Cheng, and Zheng-Jun
[33] Yue Wu, Yinpeng Chen, Lijuan Wang, Yuancheng Ye,                       Zha. Self-promoted prototype refinement for few-shot class-
     Zicheng Liu, Yandong Guo, and Yun Fu. Large scale incre-               incremental learning. In Proceedings of the IEEE/CVF Con-
     mental learning. In Proceedings of the IEEE Conference on              ference on Computer Vision and Pattern Recognition, pages
     Computer Vision and Pattern Recognition, pages 374–382,                6801–6810, 2021.
     2019.
[34] Yufei Xu, Qiming Zhang, Jing Zhang, and Dacheng Tao. Vi-
     tae: Vision transformer advanced by exploring intrinsic in-
     ductive bias. Advances in Neural Information Processing
     Systems, 34, 2021.
[35] Shipeng Yan, Jiangwei Xie, and Xuming He. Der: Dynam-
     ically expandable representation for class incremental learn-
     ing. In Proceedings of the IEEE/CVF Conference on Com-
     puter Vision and Pattern Recognition, pages 3014–3023,
     2021.
[36] Hongxu Yin, Pavlo Molchanov, Jose M Alvarez, Zhizhong
     Li, Arun Mallya, Derek Hoiem, Niraj K Jha, and Jan Kautz.
     Dreaming to distill: Data-free knowledge transfer via deep-
     inversion. In Proceedings of the IEEE/CVF Conference
     on Computer Vision and Pattern Recognition, pages 8715–
     8724, 2020.
[37] L. Yu, S. Parisot, G. Slabaugh, J. Xu, and T. Tuytelaars.
     More classifiers, less forgetting: A generic multi-classifier
     paradigm for incremental learning. European Conference on
     Computer Vision, 2020.
[38] Lu Yu, Bartlomiej Twardowski, Xialei Liu, Luis Herranz,
     Kai Wang, Yongmei Cheng, Shangling Jui, and Joost van de
     Weijer. Semantic drift compensation for class-incremental
     learning. In Proceedings of the IEEE/CVF Conference
     on Computer Vision and Pattern Recognition, pages 6982–
     6991, 2020.
[39] Friedemann Zenke, Ben Poole, and Surya Ganguli. Contin-
     ual learning through synaptic intelligence. In International
     Conference on Machine Learning, pages 3987–3995. PMLR,
     2017.
[40] Hanwang Zhang, Zheng-Jun Zha, Yang Yang, Shuicheng
     Yan, and Tat-Seng Chua.          Robust (semi) nonnegative
     graph embedding. IEEE transactions on image processing,
     23(7):2996–3012, 2014.
[41] Qiming Zhang, Yufei Xu, Jing Zhang, and Dacheng Tao.
     Vitaev2: Vision transformer advanced by exploring induc-

                                                                     9305

---
