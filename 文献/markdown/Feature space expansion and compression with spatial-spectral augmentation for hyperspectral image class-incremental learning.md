# Feature space expansion and compression with spatial-spectral augmentation for hyperspectral image class-incremental learning

Source: `Feature space expansion and compression with spatial-spectral augmentation for hyperspectral image class-incremental learning.pdf`

Pattern Recognition 168 (2025) 111830

                                                                     Contents lists available at ScienceDirect

                                                                         Pattern Recognition
                                                               journal homepage: www.elsevier.com/locate/pr

Feature space expansion and compression with spatial–spectral
augmentation for hyperspectral image Class-Incremental Learning
Ran Wu a , Huanyu Liu a ,∗, Zongcheng Yue b , Chiu-Wing Sham b , Jun-Bao Li a
a
    School of Computer Science, Harbin Institute of Technology, Yikuang Street, Harbin, 150001, Heilongjiang Province, China
b
    School of Computer Science, The University of Auckland, Princes Street, Auckland, 1062, New Zealand

ARTICLE                  INFO                               ABSTRACT

Keywords:                                                   Hyperspectral image classification (HSIC) has gained significant attention because of its applications in
Hyperspectral image classification                          geographic detection and military surveillance. Numerous deep neural network-based approaches have been
Class incremental learning                                  proposed to enable efficient HSIC. However, changing landscapes and sequentially acquired scenes require
Spatial–spectral augmentation
                                                            HSIC methods that can continuously learn new classes. Updating models with new data often leads to
Feature space expansion
                                                            catastrophic forgetting of previously learned knowledge. Existing methods struggle to maintain consistent
Network compression
                                                            classification performance across all classes during the progressive updating process. To address this challenge,
                                                            we propose a learning framework named Feature space Expansion, Integration, and Compression with
                                                            spatial–spectral augmentation for Class-Incremental Learning (FEICA-CIL). This framework aims to enable the
                                                            retention of old knowledge and generalization of new knowledge simultaneously. The spatial–spectral mixed
                                                            augmentation technique encourages the model to explore robust representations in two dimensions. The CIL
                                                            training process is divided into two stages: initial and incremental. An online adverse distillation strategy is
                                                            introduced to optimize the capacity of the model in the initial stage. In the incremental stage, an expansion and
                                                            integration strategy is proposed to enlarge the feature space for new tasks while retaining old knowledge. A
                                                            compression approach is employed to simplify the expanded structures. The incremental stage can be repeated
                                                            to increase the capability of the model continuously. Extensive experiments were conducted on four datasets.
                                                            Our method exhibited novel performance compared with other IL frameworks on hyperspectral data, surpassing
                                                            comparative approaches, with the largest improvement being 1.65%. In addition, ablation studies demonstrated
                                                            the effectiveness of the proposed modules. The code is available at https://github.com/knockshot/FEICA-CIL.

1. Introduction                                                                                  spatial–spectral decision fusion strategy to weigh multi-scale superpixel
                                                                                                 maps. Functional data analysis has also been incorporated into HSIC to
    Hyperspectral image classification (HSIC) is an important topic in                           extract better spatial and spectral knowledge [6].
pattern recognition and has been widely studied in several fields, in-                               Inspired by the dominance of deep learning in other optical com-
cluding agricultural research, ocean exploration, military surveillance,                         puter vision tasks, many HSIC approaches based on deep neural net-
and mineral detection. The ultimate aim of HSIC is to assign every                               works have been developed to improve the classification performance
pixel in the hyperspectral image to a specific category based on its                             in hyperspectral data tasks. Nevertheless, they were based on the
spatial and spectral characteristics; thus, HSIC is a pixel-level image                          premise that the training dataset could cover all types of targets in
classification task. Traditional machine learning methods, such as sup-                          testing scenarios. However, the landscape is usually affected by hu-
port vector machines [1] and multinomial logistic regression [2], were                           man activities and seasonal changes, and it changes gradually over
initially adopted in HSIC. These approaches focus on utilizing the spec-                         time. Besides, different landscape types may be obtained sequentially
tral features of hyperspectral data. Dimension-reduction methods have                            during training. Hence, the models must continuously increase their
also been proposed for extracting more representative features from                              capacity to accommodate dynamic landscape changes. However, the
spectral channels, including principal component analysis (PCA) [3]                              catastrophic forgetting phenomenon [7] is unavoidable if models are
and independent component analysis [4]. Spatial–spectral classifica-                             updated with new incoming data. The previous decision boundary will
tion methodologies have been introduced to realize the synchronous
                                                                                                 change if the model is generalized to new image types.
utilization of spatial and spectral signatures. MSMIL [5] introduced a

     ∗ Corresponding author.
       E-mail address: liuhuanyu@hit.edu.cn (H. Liu).

https://doi.org/10.1016/j.patcog.2025.111830
Received 3 June 2024; Received in revised form 3 April 2025; Accepted 6 May 2025
Available online 22 May 2025
0031-3203/© 2025 Elsevier Ltd. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

---

R. Wu et al.                                                                                                               Pattern Recognition 168 (2025) 111830

    Class-incremental learning (CIL) is proposed to address catastrophic            ∙ Extensive experiments were conducted on four datasets. The
forgetting by enabling models to continuously learn new classes while                 results demonstrate the effectiveness of the proposed method.
retaining knowledge of previously learned ones. As the most chal-                     Specifically, our approach exceeds all other comparative objects
lenging scenario in incremental learning, CIL focuses on developing                   in all training settings, even when we migrated the same back-
a unified classifier that can effectively discriminate among all en-                  bone to other CIL methods. The largest margin was achieved on
countered classes, regardless of when they were introduced to the                     the Hanchuan dataset, with an improvement of 1.65%.
model. CIL approaches can be categorized into parameter regulariza-
tion, structural, rehearsal, and knowledge distillation methods. Prior              The remainder of this paper is organized as follows: Section 2 re-
works in this field include RWalk [8], which iteratively updates the             views HSIC mechanisms and recent CIL research. Section 3 presents the
fisher information matrix using moving average techniques to de-                 proposed FEICA-CIL framework. Section 4 details experimental setup
velop an approach that maintains stability regardless of task quantity;          and results on multiple hyperspectral datasets. Section 5 concludes the
KAN [9], which expands the feature extractor during the learning                 paper and discusses future work.
process and introduces an adaptive feature fusion module to dynam-
ically balance and integrate information from multiple branches; Tang            2. Related works
et al. [10] developed a learnable feature generator to create diverse
exemplars by combining semantic information from exemplars with                     In this section, related studies on HSIC and CIL are elaborated.
semantically-irrelevant information from unlabeled data; and K3D [11],
which employs knowledge fusion distillation to maintain clear decision           2.1. Hyperspectral image classification
boundaries and reduce feature confusion.
    While these approaches have advanced CIL in general domains,                     Deep learning has revolutionized HSIC by extracting hierarchical
they are not specifically designed to address the unique challenges              features, from low-level textures and edges in earlier layers to complex
of HSIC, which involves high-dimensional spectral data with complex              abstract features in deeper layers [12]. Common architectures include
spatial–spectral relationships. To overcome catastrophic forgetting in           stacked autoencoders (SAEs), deep belief networks (DBNs), recurrent
CIL specifically for HSIC applications, we propose Feature space Expan-          neural networks (RNNs), and convolutional neural networks (CNNs).
sion, Integration, and Compression with spatial–spectral Augmentation                The SAE consists of multiple concatenated autoencoders (AEs). The
(FEICA-CIL). The framework first introduces a spatial–spectral mixed             classification results can be obtained using a subsequent logistic regres-
augmentation technique that enhances the model robustness by extract-            sion classifier. LAutoAE [3] is an efficient lightweight AE that addresses
ing comprehensive representations from both the spatial and spectral             computational challenges. DBN employs the Restricted Boltzmann Ma-
dimensions, directly mitigating catastrophic forgetting through im-              chine (RBM) as its fundamental building block, providing an efficient
proved cross-class performance. The learning process is structured into          solution to mitigate the challenges of slow convergence and overfit-
two distinct stages, namely initial and incremental, with each stage             ting that are prevalent in conventional neural network architectures.
optimized through network quantization to reduce the computational               CG-based DBN [13] adopts the gradient descent algorithm to acceler-
overhead. During the initial stage, we implement an online adverse               ate the DBN convergence and avoid the ‘‘zig-zagging’’ problem. The
distillation mechanism where a student network learns simultaneously             RNN can capture the dynamic temporal characteristics of sequential
from ground truth labels and a teacher network, while the teacher                data and perform classification through a recurrent hidden state. Geo-
explores knowledge beyond the scope of the student. This distillation            DRNN [14] combines a U-Net with an RNN to mimic the human brain
strategy enables more abstract representations, allowing the model               and continuously refines the output classification map.
to learn robust knowledge and maintain strong performance on the                     The CNN is the most widely used deep learning method for HSIC.
first task, thereby mitigating catastrophic forgetting. In the incremental       CNNs can extract more robust representations through deep hierar-
stage, we address feature space limitations by introducing a new full-           chical structures to realize better classification performance. AAtt-
precision model that expands feature space. This expansion allows                CNN [15] achieves the automatic design and optimization of CNNs
clear decision boundaries for new classes while preserving old class             through a neural architecture search and channel-based attention mech-
representations through a model ensemble. The framework alleviates               anisms. GMHN [16] integrates global and local information and utilizes
catastrophic forgetting by supplementing, rather than modifying, the             superpixel features in HSIC. Spatial information was integrated into
feature extractor for previously learned data. Finally, we compress the          HSIC in [17]. The proposed model incorporates both spatial and
dual-model ensemble into an efficient single-branch network through              spectral classifiers.
knowledge distillation, optimizing both the performance and compu-
tational efficiency. The incremental stage can be repeated for further           2.2. Class-incremental learning
enlargement of the model’s capacity. The main contributions of this
study are summarized as follows:                                                     Most CIL approaches can be categorized into four types according
                                                                                 to their anti-forgetting techniques: parameter regularization, structural,
    ∙ This study proposes a CIL framework named FEICA-CIL, which                 rehearsal, and knowledge distillation methods.
      incrementally trains a model for HSIC. In particular, a spatial–               Parameter regularization methods condition the updating of weights
      spectral mixed augmentation technique is introduced to enable              using regularization constraints. EWC [18] calculates the Fisher in-
      the model to extract more robust features. The training process is         formation matrix to estimate the importance factors for each weight.
      divided into two stages according to the optimization objectives.          The renewal of the model in the current task is regularized by the
      Different optimization strategies are applied for different stages.        weights from the previous model with the importance factors as co-
    ∙ In FEICA-CIL, a full-precision network is introduced as a teacher          efficients. Structural approaches explore the network architecture to
      model to provide online supervision for training the lightweight           realize old knowledge reservations. Some methods, such as HAT [19],
      network in the initial stage. In the incremental step, a new full-         use gated parameters to maintain significant parameters for previous
      precision network is introduced to narrow the gap between the              tasks. Besides, some approaches, such as DNE [20], expand the model
      old and target feature spaces. By combining the old and new mod-           to integrate new knowledge. Rehearsal methods concentrate on select-
      els, the features of the previous classes are maintained, while the        ing representative samples to reproduce the domain distribution of the
      features of the new categories are well formulated. The ensemble           entire training dataset. For example, DER [21] incorporates a random
      of the two models is distilled into a single-branch model to reduce        selection strategy to ensure equal probability for each sample to be
      the growth of structures.                                                  selected. Knowledge distillation approaches follow the basic knowledge

                                                                             2

---

R. Wu et al.                                                                                                                    Pattern Recognition 168 (2025) 111830

distillation methodology. A model that is well-trained on previous tasks         Table 1
                                                                                 Symbols used in this section and their corresponding explanations. The difference
is considered as the teacher model to guide the optimization in the
                                                                                 between the feature extraction backbone and backbone network is whether feature
current task. PODNet [22] utilizes pooled intermediate features to               promotion layers are considered. The feature extraction backbone generates more
realize relaxed distillation constraints. FOSTER [23] introduces new             abstract and representative features aided by the feature promotion layers.
modules to narrow the residuals between targets and outputs and to                Symbol        Explanation
compress the inference block further using distillation strategies.               ⟨⋅⟩           cosine similarity function
    CIL has been studied increasingly in HSIC in recent years. LPILC [24]         [ ⋅]]         blocked gradient path
introduces a linear programming incremental learning classifier to                [⋅]+          activation function
                                                                                  MSE[⋅]        Mean Square Error loss function
adapt previous models to new datasets. Meng et al. [25] combined
                                                                                  𝜃             the feature extraction backbone in the initial stage
knowledge distillation and linear correction to balance the bias over             𝛩             the backbone network in the initial stage
the old and new classes in the outputs of incremental models. GS2 FIN-                         the feature promotion layer
CIL [26] consists of a new network architecture based on an attention                          negative cosine similarity operator
                                                                                  𝑥 and 𝑥̄      augmented samples
block to get spatiospectral features. A bias correction mechanism is
                                                                                  𝑦             the label of the sample
adopted along with the BCK classifier to correct the bias towards new             𝜏             cosine similarity operator
classes. These studies demonstrated the potential of CIL in continuously          𝜇             the parameter conditioning the new model’s output in the
learning knowledge from hyperspectral data and obtaining superior                               expansion stage
                                                                                  𝜂             the vector conditioning the old model’s output in the expansion
classification results.
                                                                                                stage
    Traditional class-incremental learning (CIL) enables models to ex-            𝛩𝑜            the backbone of the old model in the expansion stage
pand their classification capacity when sufficient new samples become             𝛩𝑛            the backbone of the new model in the expansion stage
available. In scenarios where exemplars from new classes are limited,             𝛩𝑒𝑛           the ensemble of 𝛩𝑜 and 𝛩𝑛
                                                                                  𝛷             overall feature extraction backbone including 𝛩𝑒𝑛 in the expansion
few-shot class-incremental learning (FSCIL) provides a solution for
                                                                                                stage
acquiring new knowledge. The primary challenges of few-shot incre-                𝜗             the feature extraction backbone of the new model in the
mental learning include model overfitting on new data and catastrophic                          compression stage
forgetting of previously learned information. TOPIC [27] addressed                𝛩𝑞            the backbone of the new model in the compression stage
these challenges by leveraging a neural gas network to preserve ex-
isting knowledge while strengthening the representation learning of
new categories. CPC [28] extended FSCIL to remote sensing imagery                models in all stages. The feature aggregation loss function is formulated
by introducing a prototype separability module that enhances inter-              as follows:
class distinguishability and generates more representative prototypes
                                                                                             1 ∑∑ (              ) (                 )
for new data. In summary, while both CIL and FSCIL address incre-                ℎ𝑎 (𝜃) =             1 𝑦𝑖 == 𝑦𝑗 ⋅ 𝜃(𝑥𝑖 ), 𝜃(𝑥̄ 𝑗 )                  (1)
                                                                                            2𝑘 𝑖 𝑗
mental learning challenges, they do so under different data availability           (               ) (   ⟨[[ (       )]]   ( (         ))⟩)
assumptions. Our proposed method operates within the CIL framework,               𝜃(𝑥𝑖 ), 𝜃(𝑥̄ 𝑗 ) = 1 − 1 𝛩(𝑥𝑖 ) , 2 1 𝛩(𝑥̄ 𝑗 )
                                                                                                     (   ⟨[[ (       )]]    ( (        ))⟩)            (2)
where adequate new samples are available for training.                                              + 1 − 1 𝛩(𝑥̄ 𝑗 ) , 2 1 𝛩(𝑥𝑖 )
                                                                                     The indicator function, denoted as 1, returns a value of 1 when
3. Methodology                                                                   its input satisfies the specified requirement. 𝜃 is the feature extraction
                                                                                 backbone of the model, including the backbone network and feature
    The details of the proposed FEICA-CIL method are fully discussed             promotion layers. 𝛩 represents the backbone network and 1 and 2 are
in this section. Specifically in the section. 3.1, the entire architecture       two feature promotion layers.  denotes a negative cosine similarity
of FEICA-CIL is illustrated and the training process is demonstrated. In         operator based on the representations after the feature promotion
addition, the novelties of FEICA-CIL are introduced. The augmentation            layers. The feature promotion layers project the features onto higher
technique assists the model in capturing more robust characteristics             dimensions and identify more typical representations. ⟨⋅⟩ denotes the
from both the spatial and spectral dimensions. The distillation strategy         cosine similarity function. [[⋅]] implies that the gradient path of the inner
used in the initial stage is explained, and the expansion and compres-           part is blocked. 𝑥 and 𝑥̄ represent the augmented samples, and 𝑦 is
sion processes in the incremental step are elaborated upon. Table 1              the sample label. 𝑘 is a variable calculated from accumulated squared
presents the symbols used in this section, along with explanations of            lengths of each class. This function aggregates the representations of
their functions.                                                                 samples from the same class in the hyper-feature space. These features
                                                                                 do not implicitly converge. The diversity of the embeddings is re-
                                                                                 tained, which is beneficial for training the classifier. The local similarity
3.1. Framework
                                                                                 classifier is formulated as follows:
                                                                                          exp ⟨𝝂 𝑐,𝑘 , 𝐡⟩             ∑
    Fig. 1 depicts the framework of the proposed FEICA-CIL. The process          𝑠𝑐,𝑘 = ∑                    , 𝐲̂ 𝑐 =   𝑠𝑐,𝑘 ⟨𝝂 𝑐,𝑘 , 𝐡⟩                   (3)
begins with extracting small data cubes from hyperspectral images                          𝑖 exp ⟨𝝂 𝑐,𝑖 , 𝐡⟩          𝑘
using a sliding window, followed by applying spatial–spectral mixed                  𝝂 𝑐,𝑘 represents the 𝑘th normalized class agent for the 𝑐th class. The
augmentation to these selected samples. The framework operates in two            model output is denoted as 𝐡, and 𝐲̂ 𝑐 represents the score of 𝐡 for the
stages: In the initial stage, a quantized student model is guided by a           𝑐th class. The outputs of the local similarity classifier can be formulated
full-precision teacher network. During the incremental stage, when new           as follows:
classes are introduced into the training dataset, the previously trained                    [         ( (        )) ]
                                                                                                  exp 𝜎 𝐲̂ 𝑦 − 𝛿
model is preserved and combined with a new full-precision network                𝐿𝑆𝐶 = − log ∑             ( )                                          (4)
                                                                                                               ̂𝑖
                                                                                                     𝑖≠𝑦 exp 𝜎 𝐲
to form an ensemble. This ensemble integrates the feature extraction                                                  +
capabilities of both models. To address the increasing complexity of                 The parameter 𝜎 is a learnable scaling factor, and 𝛿 is a constant
the parameters, a new lightweight model is trained from scratch to               used to induce a stronger inter-class separation. The activation function
replicate the inductive characteristics of the ensemble. This incremental        is denoted by [⋅]+ . The basic loss module in the proposed method is as
stage can be repeated iteratively based on the requirements for expand-          follows:
ing the training samples. Besides, the hyper-feature aggregation loss
                                                                                 𝑏 (𝜃) = 𝐿𝑆𝐶 (𝜃) + ℎ𝑎 (𝜃)                                                     (5)
functions [29] and local similarity classifier [22] are adopted for the

                                                                             3

---

R. Wu et al.                                                                                                                                Pattern Recognition 168 (2025) 111830

Fig. 1. Illustration of learning phases in FEICA-CIL. Hyperspectral data is divided into small cubes to match the requirements of the model’s input. Spatial–spectral mixed
augmentation is introduced to improve the model’s representation extraction ability. The initial stage is designed for the initial state of the CIL when no old knowledge must be
reserved. The incremental stage enables the model to enlarge its classification capacity for new classes while remembering knowledge from old classes.

Fig. 2. Visualization of image augmentation process. The upper row shows images of different channels from a single sample. The middle row demonstrates the effect of the
CenterCropResize and the Horizonflip. The bottom row exhibits the impact of SSMA. The augmentations of the same sample have diverse patterns after SSMA. Aggregating these
exemplars induces the model to find robust representations from both spatial and spectral dimensions.

                                                                                            models can be updated simultaneously so that our method can be easily
                                                                                            migrated to diverse hyperspectral scenarios. Through the adversarial
                                                                                            relationship established between their outputs, the teacher model is
                                                                                            compelled to emphasize differences from the student model during
                                                                                            training, enabling it to explore knowledge beyond the student’s current
                                                                                            understanding and provide meaningful supervision. This knowledge is
                                                                                            then transferred to the student model through conventional distillation
                                                                                            constraints. In this way, the teacher model can effectively guide the stu-
                                                                                            dent’s learning without requiring pre-training. We use the correlation
Fig. 3. Process of computing trainable 𝜂 and 𝜇 using internal features. Features from
                                                                                            relation between samples to capture the feature space of the model.
the old model determine the value of 𝜂, conditioning the impact of the old model.                                                        {                            }
Representations from the new model control the effect of the new model with 𝜇.              Specifically, assuming that  = (𝑥1 , 𝑦), (𝑥2 , 𝑦), (𝑥3 , 𝑦), … , (𝑥𝑛 , 𝑦) is a
                                                                                            dataset including all samples belonging to category 𝑦, the outputs of
                                                                                            the student model can be organized as:
3.1.1. Spatial–spectral mixed augmentation                                                           [                                         ]
                                                                                            𝜃𝑠 () = 𝜃𝑠 (𝑥1 ), 𝜃𝑠 (𝑥2 ), 𝜃𝑠 (𝑥3 ), … , 𝜃𝑠 (𝑥𝑛 )                         (6)
    We propose spatial–spectral mixed augmentation (SSMA), which
performs random permutations across both spatial and spectral dimen-                           Here, 𝜃𝑠 represents the feature extraction backbone of the student
sions by altering values at selected points within randomly chosen                          model. Then, the correlation relations of this specific class can be
bands. Combined with hyper-feature aggregation loss, SSMA encour-                           calculated:
ages learning of stable, representative features. We also incorporate                             { [       ] [           ]       [              ]}
CenterCropResize and Horizonflip techniques. As shown in Fig. 2, SSMA                       𝑠 = 𝜏 𝜃𝑠1 , 𝜃𝑠2 , 𝜏 𝜃𝑠1 , 𝜃𝑠3 , … , 𝜏 𝜃𝑠𝑛 , 𝜃𝑠(𝑛−1)              (7)
changes the randomly selected channels in different augmentations,
                                                                                               𝜏 is the cosine similarity operator. 𝜃𝑠1 denotes 𝜃𝑠 (𝑥1 ). Similarly, the
forcing the model to extract robust representations from both the
                                                                                            correlation relations generated by the teacher model are as follows:
spatial and spectral dimensions.
                                                                                                  { [        ] [          ]       [              ]}
                                                                                            𝑡 = 𝜏 𝜃𝑡1 , 𝜃𝑡2 , 𝜏 𝜃𝑡1 , 𝜃𝑡3 , … , 𝜏 𝜃𝑡𝑛 , 𝜃𝑡(𝑛−1)                     (8)
3.1.2. The initial stage optimization
    In the initial stage, the model is trained to undertake a common                            Let 𝜃𝑡 represent the feature extraction backbone of the teacher
classification task. Hence, a distillation strategy is employed to improve                  model. The difference between 𝑠 and 𝑡 is computed as 𝑃𝑠𝑡 =
the classification performance. The common distillation strategy only                       |𝑠 − 𝑡 |. While this calculation initially focuses on specific classes,
                                                                                            |          |
transfers knowledge from the teacher to the student, requiring a well                       the actual training process involves computing the intra-class corre-
updated teacher model. Inspired by adversarial distillation, we propose                     lation matrices independently for each class to obtain the difference
constructing a one-stage framework where the teacher and student                            matrices. We preserve the top one-third values in 𝑃𝑠𝑡 as coefficients

                                                                                        4

---

R. Wu et al.                                                                                                                    Pattern Recognition 168 (2025) 111830

for the teacher model’s hyper-feature aggregation loss function. The            Table 2
                                                                                The number of training and testing samples of each dataset.
modified loss function is formulated as follows:
            1 ∑∑ (           )       (                    )                      Dataset                  Class         Train           Test               Total
𝑡 (𝜃𝑡 ) =        1 𝑦𝑖 == 𝑦𝑗 ⋅𝑃𝑠𝑡 ⋅  𝜃𝑡 (𝑥𝑖 ), 𝜃𝑡 (𝑥̄ 𝑗 )        (9)            SA                       16            1076            53053              54129
           2𝑘 𝑖 𝑗
                                                                                 PU                       9             1277            41499              42776
   This loss serves as the adversarial distillation loss, highlighting           Longkou                  9             1019            203523             204542
the differences between the teacher and student models. Through this             Hanchuan                 16            1279            256251             257530

mechanism, the teacher model explores knowledge beyond the current
understanding of the student model. Simultaneously, we introduce
a distillation loss function to facilitate knowledge transfer from the              Setting overall feature extraction backbone including 𝛩𝑒𝑛 as 𝛷, the
teacher to the student model:                                                   overall loss function of the feature expansion and integration step is as
                  [             ]                                               follows:
𝑑𝑖𝑠𝑡 (𝜃𝑠 ) = 𝑀𝑆𝐸 𝛩𝑡 (𝑥), 𝛩𝑠 (𝑥) + ℎ𝑑 (𝜃𝑡 , 𝜃𝑠 )                 (10)
                                                                                𝑖𝑛𝑡𝑒𝑔 = 𝑏 (𝛷) = 𝐿𝑆𝐶 (𝛷) + ℎ𝑎 (𝛷)                                           (16)
                    1 ∑∑ (          ) (               )
ℎ𝑑 (𝜃𝑡 , 𝜃𝑠 ) =          1 𝑦𝑖 == 𝑦𝑗 ⋅ 𝜃(𝑥𝑖 ), 𝜃(𝑥𝑖 )              (11)
                   2𝑘 𝑖 𝑗
    𝛩 denotes the backbone of the network, whereas 𝜃 represents the             3.1.4. Feature space compression
feature extraction backbone of the network where the feature promo-                 While the model ensemble achieves impressive classification re-
tion layers are included. 𝑀𝑆𝐸 indicates that the mean square error loss         sults, it introduces significant computational overhead during infer-
function is adopted. 𝑋 includes the augmented samples. ℎ𝑑 aggregates           ence. Since a single model can typically handle more classes, parameter
the representations of the student and teacher models in the higher             redundancy likely exists. To address this, we propose a compression
dimension while 𝑀𝑆𝐸 constrains the outputs in the lower dimension.              strategy using knowledge distillation. A new quantized model is trained
The overall loss function in the initial stage is as follows:                   under the supervision of the previously trained model ensemble, em-
                                                                                ploying multi-level feature constraints similar to the initial stage. The
𝑖𝑛𝑡 = 𝑏 (𝜃𝑡 ) + 𝑏 (𝜃𝑠 ) + 𝑡 (𝜃𝑡 ) + 𝑑𝑖𝑠𝑡 (𝜃𝑠 )                 (12)        optimization objective is formulated as follows:

                                                                                𝑠𝑖𝑚𝑝 = 𝑏 (𝜗) + ℎ𝑑 (𝜗, 𝛷) + 𝑓 𝑒 (𝜗)                                         (17)
3.1.3. Feature expansion and integration
    Although the old model trained in the last stage can well represent                      ∑       [⟨               ⟩ ⟨             ⟩]
                                                                                𝑓 𝑒 (𝜗) =       −     𝛩𝑞 (𝑥), 𝛩𝑒𝑛 (𝑥) + 𝛩𝑞 (𝑥),
                                                                                                                             ̄ 𝛩𝑒𝑛 (𝑥)
                                                                                                                                    ̄                          (18)
the characteristics of the samples from the old classes. New classes are
beyond its sight. Hence, a new model is introduced to supplement the               Here, 𝜗 represents the feature extraction backbone of the new
representation capability of the old model. Let (𝑥, 𝑦) denote the input         model, which consists of the feature promotion backbone and the
and its label, with 𝛩𝑜 and 𝛩𝑛 representing the backbones of the old             backbone network 𝛩𝑞 . ⟨⋅⟩ represents the cosine similarity function.
and new models respectively. The combined representation of these               Unlike the loss in the initial stage, we choose cosine similarity loss
backbones is denoted as 𝛩𝑒𝑛 (𝑥). The training process can be represented        rather than the means square error loss to restrict shallow features
as:                                                                             output by 𝛩𝑞 .

𝛩𝑒𝑛 (𝑥) = 𝛩𝑜 (𝑥) + 𝑎𝑟𝑔𝑚𝑖𝑛𝐸(𝑥,𝑦) [𝑙(𝑦, 𝜑(𝛩𝑜 (𝑥) + 𝛩𝑛 (𝑥)))]          (13)
                            𝛩𝑛
                                                                                4. Experiments
     Here, 𝜑(⋅) denotes the normalized logits of the model ensemble, and
                                                                                    The performance of the proposed FEICA-CIL method is evaluated on
𝑙(⋅) represents the loss function. Then, the ideal training result should
                                                                                several datasets. These results demonstrate the novelty of the proposed
be:
                                                                                method. Ablation studies were conducted to analyze the impacts of
𝑦 = 𝜑(𝛩𝑜 (𝑥) + 𝛩𝑛 (𝑥)) = 𝑆𝑜𝑓 𝑡𝑚𝑎𝑥(𝜏(𝝂 𝑛+𝑚 , 𝛩𝑜 (𝑥) + 𝛩𝑛 (𝑥)))       (14)        different modules in the proposed architecture.

    Let 𝜏 denote the cosine similarity function, and 𝝂 𝑛+𝑚 represent the
                                                                                4.1. Experimental settings
growing class agents in the local similarity classifier. While the new
model reduces the disparity between old and ideal features, simply
                                                                                   This section explains the data settings in the incremental learn-
accumulating outputs from both models constrains the update capa-
                                                                                ing framework and the parameter settings for training. The running
bility of Eq. (14). This limitation occurs because only 𝛩𝑛 can be
                                                                                environment is also described.
updated during incremental training, while the impact of 𝛩𝑜 remains
static. For new tasks, the model ensemble is enhanced to accommodate
                                                                                4.1.1. Data description
additional classes. We implement a trainable factor 𝜂 to independently
                                                                                    Four classical hyperspectral datasets were adopted in the evaluation
regulate each channel of the old model’s output. 𝜂 is implemented as
                                                                                experiments: SA (Salinas), PU (University of Pavia), Longkou (WHU-Hi-
a factor rather than a parameter because the disproportionate volume
                                                                                LongKou) and Hanchuan (WHU-Hi-HanChuan). The SA dataset com-
of new class data compared to old class data can drive a trainable
                                                                                prises 512 × 217 pixels with 224 spectral bands, containing 16 distinct
parameter toward trivial values during training. For old classes, feature
                                                                                classes in its ground truth. The PU dataset contains 610 × 340 pix-
representations are effectively captured by the old model’s outputs. The
                                                                                els with 103 spectral bands, with its land cover categorized into 9
model ensemble is trained to reconstruct feature patterns using selected
                                                                                classes. The Longkou dataset comprises 550 × 400 pixels with 270
samples. Therefore, we introduce a trainable factor 𝜇 that modulates
                                                                                spectral bands, containing 9 land-cover classes. The Hanchuan dataset
the influence of 𝛩𝑛 . The final optimization objective is formulated as
                                                                                contains 1217 × 303 pixels with 274 spectral bands, comprising 16
follows:
                                                                                target classes. Dataset-specific splitting ratios were used based on their
𝛩𝑛 = 𝑎𝑟𝑔𝑚𝑖𝑛[𝑦, 𝑆𝑜𝑓 𝑡𝑚𝑎𝑥(𝜏(𝝂 𝑛+𝑚 , 𝜇 ⋅ 𝛩𝑜 (𝑥) + 𝜂 ⋅ 𝛩𝑛 (𝑥)))]        (15)        respective sizes. Specifically, we selected 2% of the SA dataset, 3% of
          𝛩𝑛
                                                                                the PU dataset, and 0.5% of the Longkou [30] and Hanchuan [30]
    Fig. 3 demonstrates how these two factors are obtained during the           datasets, respectively, as training sets. Table 2 lists the numbers of
training. 𝜇 acts as a global attention coefficient for the outputs of the       training and testing samples. The PCA approach was adopted to reduce
new model, while 𝜂 is a channel attention parameter for the outputs of          the spectral dimensions to eight. The entire data block was divided into
the old network.                                                                3D cubes of size 8 × 27 × 27.

                                                                            5

---

R. Wu et al.                                                                                                                         Pattern Recognition 168 (2025) 111830

               Table 3
               Classification results of FEICA-CIL on the SA and Hanchuan datasets (%).
                Dataset            SA                                                              Hanchuan
                Class              Task 1          Task 2         Task 3          Task 4           Task 1         Task 2        Task 3         Task 4
                1                  100.00          100.00         100.00          100.00           99.96          99.53         99.82          99.33
                2                  99.94           100.00         100.00          100.00           99.84          97.86         98.87          98.13
                3                  100.00          100.00         100.00          100.00           99.03          97.33         97.59          97.27
                4                  100.00          100.00         99.78           99.85            99.72          99.91         99.91          99.72
                5                                  99.77          99.73           98.13                           96.73         95.73          87.27
                6                                  100.00         100.00          100.00                          79.16         74.71          70.23
                7                                  99.86          99.97           100.00                          96.94         93.85          92.32
                8                                  100.00         99.97           99.46                           93.56         92.26          91.44
                9                                                 100.00          100.00                                        94.08          82.55
                10                                                99.78           99.94                                         99.39          98.87
                11                                                99.81           98.95                                         98.26          98.69
                12                                                100.00          100.00                                        99.78          90.55
                13                                                                100.00                                                       86.27
                14                                                                100.00                                                       97.15
                15                                                                99.83                                                        54.73
                16                                                                100.00                                                       99.96
                OA                 99.98           99.96          99.95           99.74            99.79          97.07         97.08          96.44
                AA                 99.99           99.95          99.92           99.76            99.64          95.13         95.35          90.28
                Kappa              99.97           99.95          99.94           99.71            99.66          96.15         96.56          95.82
                Forgetting         0.00            −0.01          0.02            0.25             0.00           0.98          1.55           4.19

4.1.2. Training settings                                                                  4.3. Comparative experiments
    We employed Adam optimizer with a learning rate of 0.001 and
weight decay of 0.005. Training used a batch size of 256 and a memory                         Methods marked with ∗ share our memory management strategy,
size of 480, with per-class memory allocated proportionally to training                   and we included GS2 FIN-CIL, the current state-of-the-art method for
samples. Following [31], we used the herding approach for sample                          hyperspectral data classification. We conducted experiments with three
selection. The initial stage required 200 epochs for all datasets except                  random seeds (1, 2048, and 3407) and reported means and standard
                                                                                          deviations of OAs, adopting the comparative criterion of reporting
SA (50 epochs), while incremental stages including the integration and
                                                                                          statistical results as in [33]. Tables 5–6 present detailed results with the
compression processes used 200 epochs each. Our CNN architecture
                                                                                          best performances highlighted. On the SA dataset, FEICA-CIL consis-
follows [32], comprising four convolutional layers and three pooling
                                                                                          tently outperformed all competitors across all tasks, achieving a notable
layers.
                                                                                          performance advantage of 0.54% in task 4 compared to the second-best
    All experiments are operated on the Ubutnu 20.04 system. The GPU                      method (Bic∗ at 98.97%). For the PU dataset, our approach demon-
used in training is Nvidia RTX 4090. The model is built with pytorch                      strated superior performance with accuracies of 100.00%, 99.71%,
1.11.0, and torchvision 0.12.0.                                                           and 99.10% across tasks, surpassing the best competitor FORSTER∗ .
                                                                                          Experiments on the Longkou dataset further confirmed our method’s ef-
4.2. Evaluation results                                                                   fectiveness, with FEICA-CIL achieving a 0.33% accuracy improvement
                                                                                          in the final task over Bic∗ , which achieved the second highest per-
                                                                                          formance. Notably, on the Hanchuan dataset, FEICA-CIL consistently
    We evaluated FEICA-CIL on four datasets with random seed 1.                           demonstrated superior performance across all tasks, with the most
Tables 3–4 present classification results and forgetting rates across                     substantial margin of 1.65% in the final task compared to Bic∗ , which
incremental stages. Performance metrics include overall accuracy (OA),                    ranked as the second most effective approach.
average accuracy (AA), and Kappa coefficient, along with per-class                            Across all datasets, our method not only achieved higher mean accu-
accuracies. To quantify catastrophic forgetting, we calculated the for-                   racies but also maintained competitive standard deviations comparable
getting rate as the difference between historical best and current per-                   to or better than competing approaches, demonstrating both superior
formance for each class. A positive forgetting rate indicates knowledge                   performance and robust stability across different random initializations.
loss in previously learned classes, while a negative rate signifies an                    This consistency in performance underscores the reliability of our pro-
improved performance in these classes.                                                    posed approach in diverse experimental settings. It should be noted that
    On the SA dataset, our method maintained high OA (99.98%–                             results for GS2 FIN-CIL were directly obtained from the original paper
99.74%) across tasks 1–4, with maximum degradation and increas-                           as no source code was publicly available, and therefore means and
                                                                                          standard deviations from multiple experimental runs are not available
ing forgetting rate in the final task. Despite rapid forgetting rate in-
                                                                                          for this method. For all other methods, we conducted three independent
creased on the Hanchuan dataset, the model achieved 96.44% final
                                                                                          experiments with different random seeds to ensure fair and robust
OA, showing only a 3.35% decrease across all tasks. The Longkou
                                                                                          comparison. Figs. 4–5 visualize classification maps comparing BiC,
dataset exhibited gradual OA decline with increasing forgetting rates,
                                                                                          PODNet, FORSTER, DRC, and FEICA-CIL, demonstrating our method’s
yet limited the final performance drop to 0.37% with a maximum
                                                                                          consistent superiority across CIL tasks.
forgetting rate of 1.23%. Similarly, the PU dataset showed a mere                             We compared computational costs for both training and inference
0.76% accuracy decrease with a maximum forgetting rate of 1.43%.                          across methods, as shown in Table 7. For inference, we evaluated
Overall, FEICA-CIL effectively balanced knowledge retention and acqui-                    model size (parameters) and computation complexity (FLOPs). While
sition, achieving state-of-the-art results despite persistent catastrophic                most methods maintain identical core architectures, our method and
forgetting. Additionally, the four datasets encompassed diverse land                      PODNet use a local similarity classifier, increasing model size to 182.25
covers and environmental conditions. Experimental results across these                    KB compared to 164.25 KB in other methods. However, this only
varied scenarios proved the effectiveness of our method under different                   marginally increased computational requirements to 24.1 MFLOPs, just
environmental factors.                                                                    0.2 MFLOPs above other methods. Regarding training costs, while our

                                                                                    6

---

R. Wu et al.                                                                                                                                        Pattern Recognition 168 (2025) 111830

                          Table 4
                          Classification results of FEICA-CIL on the Longkou and PU datasets (%).
                           Dataset                Longkou                                                     PU
                           Class                  Task1             Task2              Task3                  Task 1               Task 2            Task 3
                           1                      99.87             99.59              99.85                  100.00               99.84             99.94
                           2                      100.00            100.00             100.00                 100.00               99.99             99.96
                           3                      96.98             96.82              90.98                  100.00               99.95             97.69
                           4                                        99.79              99.75                                       98.05             92.67
                           5                                        98.47              97.07                                       100.00            99.23
                           6                                        99.72              99.81                                       100.00            99.98
                           7                                                           99.99                                                         99.92
                           8                                                           98.87                                                         99.83
                           9                                                           87.81                                                         97.93
                           OA                     99.71             99.63              99.34                  100.00               99.80             99.24
                           AA                     98.95             99.07              97.13                  100.00               99.64             98.57
                           Kappa                  99.26             99.43              99.13                  100.00               99.71             99.00
                           Forgetting             0.00              0.15               1.23                   0.00                 0.07              1.43

               Table 5
               Comparative results of different methods on the SA and PU datasets (%).
                Dataset                 SA                                                                         PU
                Method                  Task 1            Task 2         Task 3                Task 4              Task 1            Task 2            Task 3
                iCaRL [31]              99.98 ± 0.01      84.30 ± 4.08   67.44 ± 2.70          60.97 ± 4.14        99.94 ± 0.03      89.86 ± 3.34      85.46 ± 4.69
                iCaRL∗ [31]             99.98 ± 0.01      82.61 ± 3.45   63.18 ± 2.15          57.29 ± 5.46        99.94 ± 0.03      89.66 ± 0.44      83.09 ± 0.62
                Bic [34]                99.98 ± 0.02      99.85 ± 0.12   99.16 ± 1.15          89.79 ± 8.37        99.94 ± 0.05      91.00 ± 7.62      89.81 ± 4.54
                Bic∗ [34]               99.98 ± 0.02      99.90 ± 0.06   99.79 ± 0.12          98.97 ± 0.24        99.94 ± 0.05      99.53 ± 0.08      98.77 ± 0.28
                PODNet [22]             99.97 ± 0.03      93.38 ± 3.59   92.86 ± 1.27          84.81 ± 1.69        99.99 ± 0.01      80.00 ± 1.74      72.92 ± 1.12
                PODNet∗ [22]            99.97 ± 0.03      98.34 ± 0.39   96.87 ± 1.03          92.31 ± 1.19        99.99 ± 0.01      94.65 ± 1.91      88.83 ± 4.19
                FORSTER [23]            99.91 ± 0.15      99.83 ± 0.06   99.58 ± 0.09          96.92 ± 0.36        99.79 ± 0.03      98.90 ± 0.27      98.30 ± 0.11
                FORSTER∗ [23]           99.91 ± 0.15      99.74 ± 0.04   99.35 ± 0.14          94.86 ± 3.42        99.79 ± 0.03      99.63 ± 0.03      99.09 ± 0.17
                DRC [35]                99.97 ± 0.04      96.15 ± 5.56   92.56 ± 5.52          93.89 ± 1.09        99.84 ± 0.06      94.54 ± 0.82      97.62 ± 0.89
                DRC∗ [35]               99.97 ± 0.04      95.61 ± 6.53   94.69 ± 7.86          89.78 ± 7.56        99.84 ± 0.06      98.00 ± 0.92      97.87 ± 0.42
                GS2 FIN-CIL [26]        98.80             98.16          98.25                 96.66               98.82             90.04             90.06
                Ours                    99.99 ± 0.01      99.97 ± 0.01   99.95 ± 0.01          99.51 ± 0.23        100.00 ± 0.01     99.71 ± 0.08      99.10 ± 0.12

               Table 6
               Comparative results of different methods on the Longkou and Hanchuan datasets (%).
                Dataset                 Longkou                                                Hanchuan
                Method                  Task 1            Task 2            Task 3             Task 1              Task 2            Task 3            Task 4
                iCaRL [31]              99.71 ± 0.05      82.18 ± 2.18      80.71 ± 6.45       99.55 ± 0.12        89.63 ± 1.42      66.79 ± 1.67      72.08 ± 0.81
                iCaRL∗ [31]             99.71 ± 0.05      82.56 ± 0.75      70.86 ± 2.11       99.55 ± 0.12        89.08 ± 1.39      77.87 ± 0.86      71.20 ± 2.11
                Bic [34]                99.63 ± 0.07      99.32 ± 0.16      97.78 ± 0.36       99.51 ± 0.07        97.08 ± 0.15      95.32 ± 0.36      92.70 ± 0.64
                Bic∗ [34]               99.63 ± 0.07      99.40 ± 0.05      98.94 ± 0.06       99.61 ± 0.11        96.97 ± 0.16      96.24 ± 0.28      94.46 ± 0.23
                PODNet [22]             99.70 ± 0.05      82.38 ± 4.56      86.05 ± 1.95       99.66 ± 0.05        87.93 ± 1.06      80.43 ± 2.84      72.21 ± 1.48
                PODNet∗ [22]            99.70 ± 0.05      90.97 ± 1.89      92.45 ± 1.47       99.66 ± 0.05        93.14 ± 0.39      87.17 ± 2.04      83.18 ± 2.78
                FORSTER [23]            99.57 ± 0.09      99.02 ± 0.20      98.40 ± 0.27       99.49 ± 0.08        88.80 ± 1.03      92.91 ± 0.72      93.37 ± 0.05
                FORSTER∗ [23]           99.57 ± 0.09      98.91 ± 0.72      89.40 ± 8.25       99.49 ± 0.08        96.02 ± 0.17      64.52 ± 7.44      32.23 ± 1.01
                DRC [35]                99.74 ± 0.10      98.99 ± 0.08      95.84 ± 2.31       99.57 ± 0.10        89.44 ± 0.22      87.26 ± 2.02      83.28 ± 2.49
                DRC∗ [35]               99.74 ± 0.10      99.13 ± 0.10      95.95 ± 2.42       99.57 ± 0.10        96.16 ± 0.49      92.58 ± 2.69      85.66 ± 0.81
                Ours                    99.79 ± 0.07      99.67 ± 0.04      99.27 ± 0.12       99.75 ± 0.04        97.32 ± 0.27      97.13 ± 0.06      96.11 ± 0.32

method required higher GPU memory due to feature space expan-                                  the impact of correlation difference distillation by comparing models
sion and integration, it achieved faster training times compared to                            trained with and without this strategy.
similar methods with growing architectures, such as FORSTER and
DRC. Although iCaRL, BiC, and PODNet showed quicker optimiza-                                  4.4.1. Impacts of SSMA
tion, our method delivered superior classification accuracy. For a fair                            To evaluate the impact of SSMA, we conducted experiments re-
comparison, all methods were implemented with the same memory                                  taining only CentreCropResize and HorizontalFlip techniques in data
management strategy as FEICA-CIL. The results are based on experi-                             preprocessing, while removing the SSMA strategy. Table 8 presents
ments on the SA dataset. The FLOPs value represents the number of
                                                                                               the Overall Accuracies (OAs) for all tasks across all datasets. Notable
floating-point operations performed by the model after completing the
                                                                                               performance degradation was observed in the final task across all
final task.
                                                                                               datasets, with the SA dataset showing the most significant decline of
                                                                                               1.62%. These results demonstrated that the combination of SSMA with
4.4. Ablation study
                                                                                               other data augmentation techniques contributes significantly to the
    This section presents a comprehensive analysis of several key com-                         effectiveness of the proposed FEICA-CIL.
ponents of our method. We first examined the impact of SSMA and
investigated how varying memory sizes affect performance. We then                              4.4.2. Impact of memory size
compared against a full-precision network to assess accuracy degrada-                             The effectiveness of our IL approach depends on the number of
tion caused by quantization. Additionally, we evaluated the effective-                         samples retained from previous tasks, as these samples are crucial for
ness of trainable parameters in feature integration. Finally, we analyzed                      maintaining the classification capabilities on the old data. To evaluate

                                                                                           7

---

R. Wu et al.                                                                                                                        Pattern Recognition 168 (2025) 111830

                      Table 7
                      Illustration of the training and inference consumption of different methods.
                       Method        Param (KB)      FLOPs (M)     Training time (s)                     Memory consumption (MB)
                                                                   Task1     Task2       Task3   Task4   Task1   Task2     Task3    Task4
                       iCaRL∗        164.25          23.9          2         13          14      13      2232    2530      2534     2534
                       Bic∗          164.28          23.9          2         11          13      12      2216    2550      2554     2554
                       PODNet∗       182.25          24.1          2         16          17      16      2238    2551      2549     2589
                       FORSTER∗      164.25          23.9          18        153         154     156     2228    2520      2538     2566
                       DRC∗          164.25          23.9          18        157         158     157     2234    2518      2590     2614
                       FEICA-CIL     182.25          24.1          13        84          83      82      2707    2921      2931     2941

                   Fig. 4. Classification maps of comparative methods on the SA dataset. (a) Bic, (b) PODNet, (c) FORSTER, (d) DRC, (e) FEICA-CIL.

                Fig. 5. Classification maps of comparative methods on the Longkou dataset. (a) Bic, (b) PODNet, (c) FORSTER, (d) DRC, (e) FEICA-CIL.

this dependency, we conducted extensive experiments with varying                         the framework’s resilience and practical applicability across diverse
memory sizes across multiple datasets. The results presented in Table                    memory requirements.
9 reveal that increasing the memory size generally benefited model
performance in most cases, while limited memory conditions typically                     4.4.3. Impacts of the quantization strategy
led to lower classification accuracy. Notably, even under extreme mem-                       We adopted Deep Projection (DP) [36] as our quantization strat-
ory constraints, our method demonstrated robust resistance to catas-                     egy, which employs projection layers and normalization functions to
trophic forgetting, maintaining acceptable classification performance                    generate flexible gradients for quantized weights. To evaluate the
throughout the incremental learning process. These results validate                      effectiveness of quantization, we conducted a comparative analysis

                                                                                     8

---

R. Wu et al.                                                                                                                                         Pattern Recognition 168 (2025) 111830

Table 8                                                                                          Table 11
Comparison of the performance between FEICA-CIL trained with or without SSMA (%).                Comparison of FEICA-CIL with different settings of trainable factors (%).
 Dataset            SSMA             Task1              Task2            Task3      Task4         Dataset      𝜇                                       𝜂
                    with             99.98              99.96            99.95      99.74                      0            0.1     1        2         0           0.1     1       2
 SA
                    w/o              100.00             99.94            99.94      98.12
                                                                                                  SA           99.17        99.07   99.74    99.72     98.59       98.79   99.74   98.95
                    with             100.00             99.80            99.24                    PU           99.10        99.32   99.24    98.81     99.04       98.87   99.24   99.18
 PU
                    w/o              99.98              99.66            99.19                    Longkou      99.23        99.15   99.34    99.14     99.26       99.17   99.34   99.28
                                                                                                  Hanchuan     96.07        96.51   96.44    96.31     96.05       95.93   96.44   95.99
                    with             99.71              99.63            99.34
 Longkou
                    w/o              99.69              99.54            98.85
                    with             99.79              97.07            97.08      96.44        Table 12
 Hanchuan
                    w/o              99.72              97.03            96.84      96.23        Comparison of the model’s performance with and without the correlation distillation
                                                                                                 strategy (%).
                                                                                                  Method               SA               PU                 Longkou             Hanchuan
Table 9
Comparison of the performance between FEICA-CIL trained with different memory sizes               with                 99.67            99.66              99.48               97.06
(%).                                                                                              w/o                  98.99            99.64              99.46               96.99
 Memory Size               120                  240                480              960
 SA                        91.07                97.20              99.74            99.57
 PU                        94.91                97.58              99.24            99.55
 Longkou                   97.58                98.86              99.34            99.28
                                                                                                 5. Conclusion
 Hanchuan                  90.07                93.74              96.44            97.17
                                                                                                     In this study, a novel CIL framework based on hyperspectral data
                                                                                                 named FEICA-CIL has been developed. To address the challenges posed
Table 10
                                                                                                 by changing land covers and sequential data, the proposed method
Comparison of the performance between FEICA-CIL structures based on the quantized
and full precision networks (%).                                                                 expands the feature space by supplementing a new model to learn new
 Dataset       Network             Param (KB)     Task 1        Task 2     Task 3   Task 4       classes. The old knowledge is preserved by freezing the old model and
                                                                                                 integrating its outputs with the new model. A compression strategy is
               Quantized           182.25         99.98         99.96      99.95    99.74
 SA                                                                                              introduced to transfer all acquired knowledge to a single-branch ob-
               Full precision      729.00         100.00        99.83      99.88    98.14
               Full precision      173.50         100.00        99.80      99.24
                                                                                                 jective model to mitigate the growing model parameters. Furthermore,
 PU                                                                                              a spatial–spectral mixed augmentation technique is introduced to help
               Full precision      694.00         100.00        99.55      98.53
               Quantized           173.50         99.71         99.63      99.34
                                                                                                 the models to learn more representative characteristics from spatiospec-
 Longkou                                                                                         tral information. While our approach adapts structural modification
               Full precision      694.00         99.82         99.71      99.20
               Quantized           182.25         99.79         97.07      97.08    96.44        principles and knowledge distillation concepts from conventional CIL
 Hanchuan
               Full precision      729.00         99.83         97.72      97.27    96.42        methods, it incorporates tailored designs specifically for HSIC chal-
                                                                                                 lenges, resulting in state-of-the-art performance across various datasets.
                                                                                                 In future work, we plan to explore further methods to decrease the
                                                                                                 number of new network parameters. We aim to leverage the abundant
between the quantized and full-precision implementations of FEICA-                               parameters from the old model to simplify the IL stage.
CIL, with the distillation function removed in the initial stage for
the latter. Table 10 presents the experimental results. The quantized                            CRediT authorship contribution statement
model achieved comparable or superior performance while maintaining
a reduced parameter count. Notably, it outperformed the full-precision                               Ran Wu: Writing – original draft, Visualization, Valida-
model in the final task across four datasets, with the most substantial                          tion, Software, Methodology, Investigation, Formal analysis, Data
improvement of 1.60% observed on the SA dataset. The parameter                                   curation, Conceptualization. Huanyu Liu: Supervision, Project
counts for both full-precision and quantized models after the final task                         administration. Zongcheng Yue: Visualization, Data curation. Chiu-
are documented in the table.                                                                     Wing Sham: Validation, Supervision. Jun-Bao Li: Supervision,
                                                                                                 Project administration, Funding acquisition.
4.4.4. Impacts of the trainable factors
                                                                                                 Declaration of Generative AI and AI-assisted technologies in the
    We examined the influence of trainable factors 𝜂 and 𝜇, which
                                                                                                 writing process
control the relative contributions of previous and new models in the
final feature representations. Through experiments with varying scale
                                                                                                    During the preparation of this work the author(s) used Claude
coefficients (where zero coefficients represent direct output accumu-
                                                                                                 in order to improve language and readability. After using this
lation), we analyzed their impact. Table 11 presents our results, with
                                                                                                 tool/service, the author(s) reviewed and edited the content as
default parameters in bold. The findings reveal that FEICA-CIL demon-
                                                                                                 needed and take(s) full responsibility for the content of the
strated robust performance across various combinations of 𝜂 and 𝜇,
                                                                                                 publication.
while omitting these factors results in notable performance degrada-
tion. These trainable parameters enhanced incremental learning out-
                                                                                                 Declaration of competing interest
comes by enabling dynamic weighting of previous and new model
outputs.
                                                                                                     The authors declare that they have no known competing financial
                                                                                                 interests or personal relationships that could have appeared to
4.4.5. Effect of the correlation difference distillation strategy                                influence the work reported in this paper.
    The proposed correlation difference distillation strategy aims to en-
hance the performance of the quantized model using an online teacher                             Acknowledgments
model. Experiments on the complete datasets are conducted to compare
the results with and without the proposed strategy. As shown in Table                               Our work is supported by the National Natural Science Foundation
12, the distillation approach benefited the precision of the model across                        of China (Grant No. 62271166) and the Interdisciplinary Research
all datasets, with the largest improvement of 0.68% on the SA dataset.                           Foundation of HIT, No. IR2021104.

                                                                                             9

---

R. Wu et al.                                                                                                                                       Pattern Recognition 168 (2025) 111830

Data availability                                                                                 [23] F.-Y. Wang, D.-W. Zhou, H.-J. Ye, D.-C. Zhan, Foster: Feature boosting and com-
                                                                                                       pression for class-incremental learning, in: European Conference on Computer
                                                                                                       Vision, Springer, 2022, pp. 398–414.
    Data will be made available on request.
                                                                                                  [24] J. Bai, A. Yuan, Z. Xiao, H. Zhou, D. Wang, H. Jiang, L. Jiao, Class incremental
                                                                                                       learning with few-shots based on linear programming for hyperspectral image
                                                                                                       classification, IEEE Trans. Cybern. 52 (6) (2022) 5474–5485, http://dx.doi.org/
References                                                                                             10.1109/TCYB.2020.3032958.
                                                                                                  [25] M. Xu, Y. Zhao, Y. Liang, X. Ma, Hyperspectral image classification based on
 [1] Z. He, K. Xia, J. Zhang, S. Wang, Z. Yin, An enhanced semi-supervised support                     class-incremental learning with knowledge distillation, Remote. Sens. 14 (11)
     vector machine algorithm for spectral-spatial hyperspectral image classification,                 (2022) 2556.
     Pattern Recognit. Image Anal. 34 (1) (2024) 199–211.                                         [26] J. Bai, R. Liu, H. Zhao, Z. Xiao, Z. Chen, W. Shi, Y. Xiong, L. Jiao, Hyperspectral
 [2] M. Khodadadzadeh, P. Ghamisi, C. Contreras, R. Gloaguen, Subspace multinomial                     image classification using geometric spatial–spectral feature integration: A class
     logistic regression ensemble for classification of hyperspectral images, in: IGARSS               incremental learning approach, IEEE Trans. Geosci. Remote Sens. 61 (2023)
     2018-2018 IEEE International Geoscience and Remote Sensing Symposium, IEEE,                       1–15, http://dx.doi.org/10.1109/TGRS.2023.3333005.
     2018, pp. 5740–5743.                                                                         [27] X. Tao, X. Hong, X. Chang, S. Dong, X. Wei, Y. Gong, Few-shot class-incremental
 [3] V.C. Gogineni, K. Müller, M. Orlandic, S. Werner, Lightweight autonomous                          learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and
     autoencoders for timely hyperspectral anomaly detection, IEEE Geosci. Remote.                     Pattern Recognition, 2020, pp. 12183–12192.
     Sens. Lett. 21 (2024) 1–5, http://dx.doi.org/10.1109/LGRS.2024.3355471.                      [28] Z. Zhu, P. Wang, W. Diao, J. Yang, H. Wang, X. Sun, Few-shot incremental
 [4] Q. Lu, Y. Xie, L. Wei, Z. Wei, S. Tian, H. Liu, L. Cao, Extended attribute                        learning with continual prototype calibration for remote sensing image fine-
     profiles for precise crop classification in UAV-Borne hyperspectral imagery, IEEE                 grained classification, ISPRS J. Photogramm. Remote Sens. 196 (2023) 210–227,
     Geosci. Remote. Sens. Lett. 21 (2024) 1–5, http://dx.doi.org/10.1109/LGRS.                        http://dx.doi.org/10.1016/j.isprsjprs.2022.12.024.
     2023.3348462.                                                                                [29] R. Wu, H. Liu, Z. Yue, J.-B. Li, C.-W. Sham, Hyper-feature aggregation and
 [5] S. Huang, Z. Liu, W. Jin, Y. Mu, Superpixel-based multi-scale multi-instance                      relaxed distillation for class incremental learning, Pattern Recognit. 152 (2024)
     learning for hyperspectral image classification, Pattern Recognit. 149 (2024)                     110440, http://dx.doi.org/10.1016/j.patcog.2024.110440.
     110257.                                                                                      [30] Y. Zhong, X. Hu, C. Luo, X. Wang, J. Zhao, L. Zhang, WHU-Hi: UAV-borne
 [6] F. Xue, F. Tan, Z. Ye, J. Chen, Y. Wei, Spectral-spatial classification of hyperspec-             hyperspectral with high spatial resolution (H2) benchmark datasets and classifier
     tral image using improved functional principal component analysis, IEEE Geosci.                   for precise crop identification based on deep convolutional neural network with
     Remote. Sens. Lett. 19 (2021) 1–5.                                                                CRF, Remote Sens. Environ. 250 (2020) 112012.
 [7] X. Song, K. Shu, S. Dong, J. Cheng, X. Wei, Y. Gong, Overcoming catastrophic                 [31] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, C.H. Lampert, icarl: Incremental classifier
     forgetting for multi-label class-incremental learning, in: Proceedings of the                     and representation learning, in: Proceedings of the IEEE Conference on Computer
     IEEE/CVF Winter Conference on Applications of Computer Vision, 2024, pp.                          Vision and Pattern Recognition, 2017, pp. 2001–2010.
     2389–2398.                                                                                   [32] L. Huang, Y. Chen, X. He, P. Ghamisi, Supervised contrastive learning-based
 [8] A. Chaudhry, P.K. Dokania, T. Ajanthan, P.H. Torr, Riemannian walk for                            classification for hyperspectral image, Remote. Sens. 14 (21) (2022) 5530.
     incremental learning: Understanding forgetting and intransigence, in: Proceedings            [33] C. Chen, Y. Wan, A. Ma, L. Zhang, Y. Zhong, A decomposition-based multiob-
     of the European Conference on Computer Vision, ECCV, 2018, pp. 532–547.                           jective clonal selection algorithm for hyperspectral image feature selection, IEEE
 [9] Z. Fu, Z. Wang, X. Xu, D. Li, H. Yang, Knowledge aggregation networks for class                   Trans. Geosci. Remote Sens. 60 (2022) 1–16.
     incremental learning, Pattern Recognit. 137 (2023) 109310.                                   [34] Y. Wu, Y. Chen, L. Wang, Y. Ye, Z. Liu, Y. Guo, Y. Fu, Large scale incremental
[10] Y.-M. Tang, Y.-X. Peng, W.-S. Zheng, Learning to imagine: Diversify memory                        learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and
     for incremental learning using unlabeled data, in: Proceedings of the IEEE/CVF                    Pattern Recognition, 2019, pp. 374–382.
     Conference on Computer Vision and Pattern Recognition, 2022, pp. 9549–9558.                  [35] X. Chen, X. Chang, Dynamic residual classifier for class incremental learning,
[11] L. Xiong, X. Guan, H. Xiong, K. Zhu, F. Zhang, Knowledge fusion distillation and                  in: Proceedings of the IEEE/CVF International Conference on Computer Vision,
     gradient-based data distillation for class-incremental learning, Neurocomputing                   2023, pp. 18743–18752.
     622 (2025) 129286, http://dx.doi.org/10.1016/j.neucom.2024.129286.                           [36] R. Wu, H. Liu, J.-B. Li, Adaptive gradients and weight projection based on
[12] S. Li, W. Song, L. Fang, Y. Chen, P. Ghamisi, J.A. Benediktsson, Deep learning                    quantized neural networks for efficient image classification, Comput. Vis. Image
     for hyperspectral image classification: An overview, IEEE Trans. Geosci. Remote                   Underst. 223 (2022) 103516, http://dx.doi.org/10.1016/j.cviu.2022.103516.
     Sens. 57 (9) (2019) 6690–6709.
[13] C. Chen, Y. Ma, G. Ren, Hyperspectral classification using deep belief networks
     based on conjugate gradient update and pixel-centric spectral block features,
     IEEE J. Sel. Top. Appl. Earth Obs. Remote. Sens. 13 (2020) 4060–4069, http:                                             Ran Wu is currently working toward the Doctor’s degree
     //dx.doi.org/10.1109/JSTARS.2020.3008825.                                                                               in Computer Science through Successive Postgraduate and
[14] S. Hao, W. Wang, M. Salzmann, Geometry-aware deep recurrent neural networks                                             Doctoral Program at Harbin Institute of Technology in
     for hyperspectral image classification, IEEE Trans. Geosci. Remote Sens. 59 (3)                                         China. He is currently a visiting Ph.D. student in The Uni-
     (2021) 2448–2460, http://dx.doi.org/10.1109/TGRS.2020.3005623.                                                          versity of Auckland. His primary research interests include
[15] M.E. Paoletti, S. Moreno-Álvarez, Y. Xue, J.M. Haut, A. Plaza, AAtt-CNN:                                                neural networks, edge computing, self-supervised learning
     Automatic attention-based convolutional neural networks for hyperspectral image                                         and incremental learning.
     classification, IEEE Trans. Geosci. Remote Sens. 61 (2023) 1–18, http://dx.doi.
     org/10.1109/TGRS.2023.3272639.
[16] A. Zhao, C. Wang, X. Li, A global+ multiscale hybrid network for hyperspectral
     image classification, Remote. Sens. Lett. 14 (9) (2023) 1002–1010.
[17] R. Confalonieri, P.P. Htun, B. Sun, T. Tillo, An end-to-end framework for the                                           Huanyu Liu received his Ph.D. degree from Harbin Institute
     classification of hyperspectral images in the wood domain, IEEE Access 12 (2024)                                        of Technology, Harbin, China, in 2022. He is currently a
     38908–38916, http://dx.doi.org/10.1109/ACCESS.2024.3376258.                                                             lecturer in Harbin Institute of Technology. He has published
                                                                                                                             18 papers, and authorized/accepted 9 invention patents.
[18] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A.A. Rusu,
                                                                                                                             His research interests are incremental learning, intelli-
     K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, et al., Overcoming
                                                                                                                             gent perception assessment, reinforcement learning and its
     catastrophic forgetting in neural networks, Proc. Natl. Acad. Sci. 114 (13) (2017)
                                                                                                                             application.
     3521–3526.
[19] J. Serra, D. Suris, M. Miron, A. Karatzoglou, Overcoming catastrophic forgetting
     with hard attention to the task, in: International Conference on Machine
     Learning, PMLR, 2018, pp. 4548–4557.
[20] Z. Hu, Y. Li, J. Lyu, D. Gao, N. Vasconcelos, Dense network expansion for class                                         Zongcheng Yue received his B.Eng. degree from Nangchang
     incremental learning, in: Proceedings of the IEEE/CVF Conference on Computer                                            Hangkong University, Nangchang, China, in 2018, and the
     Vision and Pattern Recognition, 2023, pp. 11858–11867.                                                                  M.Eng. degree from China University of Petroleum (East
[21] P. Buzzega, M. Boschini, A. Porrello, D. Abati, S. Calderara, Dark experience for                                       China), Qingdao, China, in 2021. He is currently pursuing
     general continual learning: a strong, simple baseline, Adv. Neural Inf. Process.                                        the Ph.D. degree in Computer Science from The University
     Syst. 33 (2020) 15920–15930.                                                                                            of Auckland, Auckland, New Zealand. His current research
[22] A. Douillard, M. Cord, C. Ollion, T. Robert, E. Valle, Podnet: Pooled outputs                                           interests include machine learning and embedded systems,
     distillation for small-tasks incremental learning, in: Computer Vision–ECCV 2020:                                       hardware architecture.
     16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part
     XX 16, Springer, 2020, pp. 86–102.

                                                                                             10

---

R. Wu et al.                                                                                               Pattern Recognition 168 (2025) 111830

               Chiu-Wing Sham (Senior Member, IEEE) received the bach-               Jun-Bao Li received the Ph.D. degree from Harbin Institute
               elor’s degree in computer engineering and the M.Phil. and             of Technology in 2008. He is currently a professor at School
               Ph.D. degrees from The Chinese University of Hong Kong,               of Electronics and Information Engineering, Harbin Institute
               in 2000, 2002, and 2006, respectively. During his years               of Technology, Harbin 150001. His research interests are
               with The Hong Kong Polytechnic University, he engaged                 image processing and pattern recognition. He is the reviewer
               in various university projects for the commercialization              of many fund projects such as national natural fund, natural
               of technology, particularly a few optical communication               science fund of Heilongjiang province, natural science fund
               projects in collaboration with Huawei. He is currently with           of Hebei province, and deputy editor of 4 international
               the University of Auckland, as a Senior Lecturer. His interest        journals.
               focus on computer hardware and edge applications. He was
               the Associate Editor of IEEE TRANSACTIONS ON CIRCUITS
               AND SYSTEMS—II: EXPRESS BRIEFS since 2017.

                                                                                11

---
