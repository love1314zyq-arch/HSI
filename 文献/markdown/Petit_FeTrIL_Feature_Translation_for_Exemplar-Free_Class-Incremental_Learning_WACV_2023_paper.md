# Petit_FeTrIL_Feature_Translation_for_Exemplar-Free_Class-Incremental_Learning_WACV_2023_paper

Source: `Petit_FeTrIL_Feature_Translation_for_Exemplar-Free_Class-Incremental_Learning_WACV_2023_paper.pdf`

FeTrIL: Feature Translation for Exemplar-Free Class-Incremental Learning

         Grégoire Petit1,2 , Adrian Popescu1 , Hugo Schindler1 , David Picard2 , Bertrand Delezoide3
                      1
                        Université Paris-Saclay, CEA, LIST, F-91120, Palaiseau, France
              2
                LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, Marne-la-Vallée, France
                    3
                      Amanda, 34 Avenue Des Champs Elysées, F-75008, Paris, France
                    {gregoire.petit, adrian.popescu}@cea.fr,hugo-schindler@orange.fr
                              david.picard@enpc.fr,bertrand.delezoide@amanda.com

                         Abstract                                   optimal performance only if trained with all data at once
                                                                    whenever new classes are learned. This is an important
    Exemplar-free class-incremental learning is very chal-          limitation because data often occur in sequences [17] and
lenging due to the negative effect of catastrophic forget-          their storage is costly. Also, iterative retraining to integrate
ting. A balance between stability and plasticity of the in-         new data is computationally costly and difficult in time-
cremental process is needed in order to obtain good accu-           or computation-constrained applications [9, 32]. Incremen-
racy for past as well as new classes. Existing exemplar-free        tal learning [36] was introduced to reduce the memory and
class-incremental methods focus either on successive fine           computational costs of machine learning algorithms. The
tuning of the model, thus favoring plasticity, or on using          main problem faced by class-incremental learning (CIL)
a feature extractor fixed after the initial incremental state,      methods is catastrophic forgetting [14, 25], the tendency
thus favoring stability. We introduce a method which com-           of neural nets to underfit past classes when ingesting new
bines a fixed feature extractor and a pseudo-features gen-          data. Many recent solutions [4, 13, 33, 44, 46], based
erator to improve the stability-plasticity balance. The gen-        on deep nets, use replay from a bounded memory of the
erator uses a simple yet effective geometric translation of         past to reduce forgetting. However, replay-based methods
new class features to create representations of past classes,       make a strong assumption because past data are often un-
made of pseudo-features. The translation of features only           available [41]. Also, the footprint of the image memory
requires the storage of the centroid representations of past        can be problematic for memory-constrained devices [32].
classes to produce their pseudo-features. Actual features           Exemplar-free class-incremental learning (EFCIL) methods
of new classes and pseudo-features of past classes are fed          recently gained momentum [45, 38, 47, 48]. Most of them
into a linear classifier which is trained incrementally to dis-     use distillation [12] to preserve past knowledge, and gener-
criminate between all classes. The incremental process is           ally favor plasticity. New classes are well predicted since
much faster with the proposed method compared to main-              models are learned with all new data and only a representa-
stream ones which update the entire deep model. Experi-             tion of past data [24, 31, 49]. A few EFCIL methods [1, 6]
ments are performed with three challenging datasets, and            are inspired by transfer learning [37, 39]. They learn a fea-
different incremental settings. A comparison with ten exist-        ture extractor in the initial state, and use it as such later to
ing methods shows that our method outperforms the oth-              train new classifiers. In this case, stability is favored over
ers in most cases. FeTrIL code is available at https:               plasticity since the model is frozen [24].
//github.com/GregoirePetit/FeTrIL.                                     We introduce FeTrIL, a new EFCIL method which com-
                                                                    bines a frozen feature extractor and a pseudo-feature gen-
                                                                    erator to improve incremental performance. New classes
1. Introduction                                                     are represented by their image features obtained from the
                                                                    feature extractor. Past classes are represented by pseudo-
   Deep learning [8] has dramatically improved the qual-            features which are derived from features of new classes
ity of automatic visual recognition, both in terms of ac-           by using a geometric translation process. This translation
curacy and scale. Current models discriminate between               moves features toward a region of the features space which
thousands of classes with an accuracy often close to that           is relevant for past classes. The proposed pseudo-feature
of human recognition, assuming that sufficient training ex-         generation is adapted for EFCIL since it is simple, fast and
amples are provided. Unlike humans, algorithms reach                only requires the storage of the centroids for past classes.

                                                                  3911

---

             Initial state                   Incremental state 1                Incremental state 2                  Actual features

                                                f 1(C1)     f 1(C3)            f 2(C1)   f 2(C3)      f(C5)        f(C1)    f(C3)      f(C5)
         f(C1)    f(C2)      f(C3)              f 1(C2)     f(C4)              f 2(C2)   f 2(C4)                   f(C2)    f(C4)
                  (a)                                 (b)                                (c)                               (d)
Figure 1. Illustration of the proposed pseudo-feature generation procedure. This toy example includes an initial state (3 classes) and two IL
states (1 new class per state) in subfigures (a), (b) and (c). Subfigure (d) provides the actual features of all classes that would be available
for a classical learning. The illustration uses a 2D projection of actual features. Pseudo-features of past classes are generated by geometric
translation of features of the new class added in each state with the difference between the centroids of the target past class and of the new
class. While imperfect, the pseudo-feature generator produces a usable representation of past classes. Best viewed in color.

FeTrIL is illustrated with a toy example in Figure 1. We                   by iCaRL [33], itself inspired by learning without forget-
run experiments with a standard EFCIL setting [13, 47, 48],                ting (LwF) [19]. Distillation was later refined and comple-
which consists of a larger initial state, followed by smaller              mented with other components to improve the plasticity-
states which include the same number of classes. Results                   stability compromise. LUCIR [13] applies distillation on
show that the proposed approach has better behavior com-                   features instead of raw classification scores to preserve the
pared to ten existing methods, including very recent ones.                 geometry of past classes, and an inter-class separation to
                                                                           maximize the distances between past and new classes. The
2. Related Work                                                            problem was partially addressed by adding specific class
                                                                           separability components in [7, 13]. Distillation-based
    CIL algorithms are needed when data arrive sequentially
                                                                           methods need to store the current and the preceding model
and/or computational constraints are important [9, 17, 24,
                                                                           for incremental updates. Their memory footprint is larger
29]. Their objective is to ensure a good balance between
                                                                           compared to methods which do not use distillation [24].
plasticity, i.e. integration of new information, and stability,
i.e. preservation of knowledge about past classes [27]. This                   Another important problem in CIL is the semantic drift
is challenging because the lack of past data leads to catas-               between incremental states. Auxiliary classifiers were intro-
trophic forgetting, i.e. the tendency of neural networks to                duced in [20] to reduce the effect of forgetting. ABD [38]
focus on newly learned data at the expense of past knowl-                  uses image inversion to produce pseudo-samples of past
edge [25]. Recent reviews of CIL [2, 24] show that a ma-                   classes. The method is interesting but image inversion is
jority of methods replay samples of past classes to mitigate               difficult for complex datasets. Another interesting solution
forgetting [4, 13, 33, 46]. One advantage here is that the                 is proposed in [45], where the features drift between incre-
network architecture remains constant throughout the incre-                mental steps is estimated from that of new classes. Recent
mental process. However, these methods have two major                      EFCIL approaches [47, 48, 49] use past class prototypes in
drawbacks: (1) First, the assumption that past samples are                 conjunction with distillation to improve performance. Pro-
available is strong since in many cases past data cannot be                totype augmentation is proposed in PASS [48] to improve
stored due, for instance, to privacy restrictions [41] and (2)             the discrimination of classes learned in different incremen-
the memory footprint of the stored images is high.                         tal states. Feature generation for past classes is introduced
    Here, we investigate exemplar-free CIL, with focus on                  in IL2A [47] by leveraging information about the class dis-
methods which keep the network size constant. This setting                 tribution. This approach is difficult to scale-up because a
is very challenging since it imposes strong constraints on                 covariance matrix needs to be stored for each class. A pro-
both memory and computational costs. A majority of ex-                     totype selection mechanism is introduced in SSRE [49] to
isting methods use regularization to update the deep model                 better discriminate past from new classes. FeTrIL shares the
for each incremental step [24], and adapt distillation [12] to             idea of using class prototypes with [45, 47, 48, 49]. An im-
preserve past knowledge by penalizing variations for past                  portant difference is that we freeze the model after the initial
classes during model updates. Note that, while some of the                 state, while the other methods deploy more sophisticated
distillation-based methods were introduced in an exemplar-                 mechanisms to integrate prototypes in a knowledge distilla-
based CIL (EBCIL) setting, many of them are also appli-                    tion process. Past comparative studies [2, 24] found that,
cable to EFCIL. This approach to CIL was popularized                       while appealing in theory, distillation-based methods un-

                                                                       3912

---

derperform in EFCIL, particularly for large-scale datasets.          tween different states are not known at test time.
Second, since the representation space is fixed, a simple                The global functioning of FeTrIL is illustrated in Fig-
geometric translation of actual features of new classes is           ure 2. It uses a feature extractor, a pseudo-feature generator
sufficient to produce usable pseudo-features. In contrast,           based on geometric translation, and an external classifica-
IL2A [47], the work which is closest to ours, needs to store a       tion layer in order to address EFCIL. Inspired by transfer-
covariance matrix per class to obtain optimal performance.           learning based CIL [1, 33], the feature extractor F is frozen
Third, the use of a fixed extractor simplifies the training pro-     after the initial state. This ensures a stable representation
cess since only the final linear layer is trained, compared          space through the entire CIL process. Given that images
to a fine tuning of the backbone model required by recent            of past classes cannot be stored in EFCIL, a generator G is
methods which use prototypes and feature generation.                 used to produce pseudo-features of past classes (fˆt (Cp )). G
    Another line of work takes inspiration from transfer             takes features of new classes (f (Cn )) and prototypes of past
learning [28, 37] to tackle EFCIL. A feature extractor is            and new classes (µ(Cp ), µ(Cn )) as inputs. A linear classi-
trained in the initial non-incremental state and fixed after-        fier L combines features and pseudo-features to jointly train
wards. Then, an external classification layer is updated in          classifiers for all seen classes (past and new). The pseudo-
each incremental state to integrate new classes. The nearest         features generation is crucial since it enables class discrim-
class mean (NCM) [26] was used in [33], linear SVMs [30]             ination across all incremental states. The hypotheses made
were used in [1] and extreme value machines [34] were re-            here are that: (1) while imperfect, the pseudo-features still
cently tested by [6]. The advantages of transfer-learning            produce effective representations of past classes, and (2) us-
methods are their simplicity, since only the classification          ing a frozen extractor in combination with a generator in
layer is updated, and their lower memory requirement, since          EFCIL is preferable to mainstream distillation-based meth-
they need a single deep model to function. These meth-               ods [45, 47, 48, 49]. These hypotheses are tested through
ods give competitive performance compared to distillation-           the extensive experiments in Section 4. We present the main
based ones in EFCIL, particularly at scale [2]. However,             components of FeTrIL in the next subsections.
features are not updated, and they are sensitive to large do-
main shifts between incremental tasks [17]. Equally, exist-          3.1. Generation of pseudo-features
ing transfer-learning inspired works do not sufficiently ad-
dress inter-class separability, which is in focus here.                 The pseudo-feature generator, illustrated in Figure 1,
    Class prototypes creation was studied in other learning          produces effective representations of past classes. Exist-
settings than CIL. A very interesting method focused on              ing approaches which generate past data rely on methods
few-shot learning was proposed in [5]. A distance-based              such as generative adversarial networks [10], image inver-
classifier which uses an approximation of the Mahalanobis            sion [38], or covariance-based past class models [47]. We
distance is proposed. The means and variances of new                 propose a much simpler alternative which is defined as:
classes are predicted using two supplementary neural net-
works. While adapted for few-shot learning, such an ap-                          fˆt (cp ) = f (cn ) + µ(Cp ) − µ(Cn )         (1)
proach is not fully adapted in CIL. First, the supplementary
neural networks require a large number of supplementary              with: Cp - target past class for which pseudo-features are
parameters. This is a disadvantage here, since CIL methods           needed; Cn - new class for which images b are available;
are needed in computationally-constrained environments.              f (cn ) - features of a sample cn of class Cn extracted with
Second, we do not focus on few-shot learning and the means           F; µ(Cp ), µ(Cn ) - mean features of classes Cp and Cn ex-
of past classes are well-placed in the representation space.
                                                                     tracted with F; fˆt (cp ) - pseudo-feature vector of a pseudo-
                                                                     sample cp of class Cp produced in the tth incremental state.
3. Proposed Method
                                                                         Eq. 1 translates the value of each dimension with the dif-
   The objective of CIL is to learn a total of N classes             ference between the values of the corresponding dimension
which appear sequentially during training. This process              of µ(Cp ) and µ(Cn ). It creates a pseudo-feature vector sit-
includes an initial state (0) and T incremental ones. New            uated in the region of the representation space associated
classes need to be recognized alongside past classes which           to target class Cp based on actual features of a new class
were learned in previous states. We focus on the exemplar-           f (Cn ). The computational cost of generation is very small
free CIL setting [33, 38, 45, 49], which assumes that no             since it only involves additions and subtractions. µ(Cp ) is
past images can be stored. This scenario is more challeng-           needed to drive the geometric translation toward a region of
ing than exemplar-based CIL since catastrophic forgetting            the representation space which is relevant for Cp . Centroids
needs to be tackled without resorting to replay [24]. There          are computed when classes occur for the first time and then
is no intersection between the classes learned in different          stored. Their reuse is possible because F is fixed after the
incremental states. Unlike task IL [40], the boundaries be-          initial step and its associated features do not evolve.

                                                                   3913

---

Figure 2. FeTrIL overview for a toy example with an initial state (3 classes) and two incremental states (1 class per state). The feature
extractor F is trained in the initial state, using sets of data X1 , X2 , X3 , and then frozen afterwards. The generator G uses features f (Cn )
of the new class extracted with F and prototypes of past classes µ(Cp ) to generate pseudo-features of past classes fˆt (Cp ) in the tth state.
Prototypes (µ(Ci )) are the centroids of all classes (past and new). They are learned when classes are first seen and then stored throughout
the IL process. A linear classifier L is used to learn classification weights w(Ci ) for all seen classes (past and new).

3.2. Selection of pseudo-features                                          around the centroid of the target past class is needed.

   Eq. 1 translates the features for a single sample. If each              3.3. Linear classification layer training
class is represented by s samples, the generation process                     We assume that the CIL process is in the tth CIL state,
needs to be repeated s times. The overview of FeTrIL (Fig-                 which includes P past classes and N new classes. The com-
ure 2) and of the pseudo-feature generation (Figure 1) use                 bination of the feature generator (Subsection 3.1) and se-
a minimal example which adds a single class per IL state.                  lection (Subsection 3.2) provides a set fˆt (Cp ) of s pseudo-
When CIL states include several classes Cn , the s pseudo-                 features for each class Cp . The objective is to train a linear
features of each class Cp can be obtained using different                  classifier for all P + N seen classes which takes pseudo
strategies, depending on how features of new classes are                   features of past classes and actual features of new classes as
used. We deploy the following strategies:                                  inputs. This linear layer is defined as:
• FeTrILk : s features are transferred from the k th similar
  new class of each past class Cp . Similarities between the                  W t = {wt (C1 ), ..., wt (CP ), wt (CP +1 ), ..., wt (CP +N )}   (2)
  target Cp and the Cn available in the current state is com-
  puted using the cosine similarity between the centroids of               with: wt - the weight of known classes in the tth CIL state.
  each pair of classes. Experiments are run with different                    W t can be implemented using different classifiers, and
  values of k to assess if a variable class similarity has a sig-          we instantiate two versions in Section 4: (1) FeTrIL using
  nificant effect on EFCIL performance. Since translation                  LinearSVCs [30] as external classifiers, and (2) FeTrILf c
  is based on a single new class, the distribution of pseudo-              using a fully-connected layer to enable end-to-end training.
  features will be similar to that of features of Cn , but in
  the region of the representation space around µ(Cp ).                    4. Evaluation
• FeTrILrand : s features are randomly selected from all                       We evaluate FeTrIL by using a comprehensive EFCIL
  new classes. This strategy assesses whether a more di-                   evaluation scenario [47, 48, 49]. This setting includes four
  versified source of features from different Cn produces                  datasets and CIL states of different size.
  an effective representation of class Cp .                                    Datasets. We use four public datasets: (1) CIFAR-
• FeTrILherd : s features are selected from any new class                  100 [16] - 100 classes, 32x32 pixels images, 500 and 100
  based on a herding algorithm [43]. It assumes that sam-                  images/class for training and test; (2) TinyImageNet [18] -
  pling should include features which produce a good ap-                   200 leaf clases from ImageNet, 64x64 pixels images, 500
  proximation of the past class. Herding was introduced                    and 50 for training and test; (3) ImageNet-Subset - 100
  in exemplar-based CIL in order to obtain an accurate                     classes subset of ImageNet LSVRC dataset [35], 1300 and
  approximation of each class by using only a few sam-                     50 for training and test; (4) ILSVRC - full dataset from [35].
  ples [33] and its usefulness was later confirmed [2, 13,                     Incremental setting. We use a classical EFCIL protocol
  44]. It is adapted here to obtain a good approximation of                from [47, 48, 49]. The number of classes in the initial state
  the sample distribution of Cp with s pseudo-features.                    is larger, and the rest of the classes are evenly distributed
   The comparison of these different strategies will allow us              between incremental states. CIFAR-100 and ImageNet-
to determine whether the geometric translation of features                 Subset are tested with: (1) 50 initial classes and 5 IL states
is prevalent, or if a particular configuration of the features             of 10 classes, (2) 50 initial classes and 10 IL states of 5

                                                                       3914

---

classes, (3) 40 initial classes and 20 states of 3 classes, and      procedure. In Subsection 4.2, we also test a one-vs-many
(4) 40 initial classes and 60 states of 1 class. Compared            strategy to accelerate incremental updates. The second vari-
to [47, 48, 49], configurations (1) and (3) for ImageNet-            ant, FeTrIL1f c , using a fully-connected layer as final layer,
Subset are added for more consistent evaluation. TinyIma-            and implements an end-to-end training strategy. FeTrIL1f c is
geNet is tested with 100 initial classes and the other classes       trained for 50 epochs with an initial learning rate of 0.1, 0.1
distributed as follows: (1) 5 states of 20 classes, (2) 10           decay, and 10 epochs patience.
states of 10 classes, (3) 20 states of 5 classes, and (4) 100            Evaluation metric. The average incremental accuracy,
states of 1 class. Configuration (4) is interesting since it en-     widely used in CIL [24, 33], is the main evaluation mea-
ables one class increments. It cannot be deployed for any            sure. For comparability with [47, 48, 49], it is computed
of the compared EFCIL methods since they require at least            as the average accuracy of all states, including the initial
two classes per increment to update models. ILSVRC is                one. We equally provide per-state accuracy curves to have
tested with 500 initial classes, and the other 500 split evenly      a more detailed view of the accuracy evolution during the
among T ∈ {5, 10, 20} states. This enables a comprehen-              CIL process. Following [49], we run each configuration of
sive comparison of the methods in varied EFCIL configura-            FeTrIL three times and report the averaged results.
tions. Naturally, task IDs are not available at test time.
    Compared methods.             We use the following EF-           4.1. Results
CIL methods in evaluation: EWC [15], LwF-MC [33],                        Comparison to existing EFCIL methods. The re-
DeeSIL [1], LUCIR [13], MUC [20], SDC [45], PASS [48],               sults from Table 1 show that FeTrIL1 outperforms all com-
ABD [38], IL2A [47], SSRE [49]. As we discussed in Sec-              pared methods in 11 tested configurations out of 12. It
tion 2, these methods cover a large variety of EFCIL ap-             is also close to the best in the remaining one. The sec-
proaches. The inclusion of recent works [47, 48, 49] is im-          ond best results are obtained with the very recent SSRE
portant to situate our contribution with respect to current          method [49]. FeTrIL1 and SSRE accuracies are close to
EFCIL trends. While focus is on EFCIL, we follow [49]                each other for CIFAR-100, with relative differences be-
and include a comparisonwith EBCIL methods. We test                  tween 0.4 and -0.2. The performance gain brought by
our method against the recent AANets approach [21], and              FeTrIL is of over 4 and 3 top-1 accuracy points for TinyIm-
against the EBCIL methods to which AANETS was added                  ageNet and ImageNet-Subset, respectively. PASS [48] and
(LUCIR [13], Mnemonics [22], PODNet [7]). Whenever                   IL2A [47], two other recent EFCIL methods, have lower av-
available, results of compared methods marked with ∗ are             erage performance. We note that EFCIL performance boost
reproduced either from their initial paper or from [49] for          was recently reported, with methods such as PASS, IL2A,
EFCIL or from [21] for EBCIL. The other results are re-              SSRE. These methods combine knowledge distillation and
computed using the original configurations of the methods.           sophisticated mechanisms for dealing with the stability-
    Implementation details. Following [33, 47, 48, 49], we           plasticity dilemma. In contrast, our method uses a fixed fea-
use ResNet-18 [11] in all experiments. FeTrIL initial train-         ture extractor and a lightweight pseudo-feature generator.
ing is done uniquely with images of initial classes to ensure        FeTrIL only optimizes a linear classification layer, while
comparability with existing methods. The feature extractor           compared recent methods use backpropagation of the en-
is trained in the initial state and then frozen for the reminder     tire model, and need much more computational resources
of the IL process. We implement a supervised training with           and time to perform the IL process. A more in-depth dis-
cross-entropy loss, SGD optimization, a batch size of 128,           cussion of complexity is proposed in Subsection 4.2. Per-
for a total of 160 epochs. The initial learning rate is 0.1, and     formance of the ILSVRC dataset is also very interesting.
it is decayed by 0.1 after every 50 epochs. To ensure com-           Direct comparison to PASS or SSRE is impossible since
parability, classes are assigned to IL states using the same         these methods were not tested at scale. However, we can
random seed as in the compared methods [13, 48, 47, 49].             safely assume that FeTrIL1 is better given PASS and SSRE
    We provide implementation details for the final layer            accuracy for the simpler ImageNet-Subset. ILSVRC results
(Eq. 2) introduced in Subsection 3.3. The hyperparameters            show that the simple method proposed here is effective for a
of the classification layers were optimized on a pool of 50          high range of classes. Interestingly, ILSVRC performance
classes selected randomly from ImageNet, but disjoint from           is stabler compared to smaller datasets since the pool of new
ILSVRC or ImageNet-Subset. L2-normalization is applied               classes available for pseudo-features generation is larger.
before the linear layer. The LinearSVC layer included in                 Comparison to a transfer-learning baseline.
FeTrIL1 uses 1.0 and 0.0001 for regularization and the tol-          DeeSIL [1] is a simple application of transfer learn-
erance parameters. The number of samples is higher than              ing to EFCIL. It has no class separability mechanism across
the dimensionality of the features, and we solve the pri-            different incremental states since classifiers are learned
mal rather than the dual optimization problem. The clas-             within each state. The need for global separability, included
sifiers are then trained using a standard one against the rest       in FeTrIL, is shown by the comparison of short and long

                                                                   3915

---

                                       CIFAR-100                  TinyImageNet                       ImageNet-Subset                     ImageNet
     CIL Method
                               T=5    T=10   T=20   T=60   T=5    T=10   T=20      T=100      T=5    T=10    T=20      T=60       T=5     T=10      T=20
     EWC∗ [15] (PNAS’17)       24.5   21.2   15.9    x     18.8   15.8    12.4       x         -      20.4    -         x          -       -         -
     LwF-MC∗ [33] (CVPR’17)    45.9   27.4   20.1    x     29.1   23.1    17.4       x         -      31.2    -         x          -       -         -
     DeeSIL [1] (ECCVW’18)     60.0   50.6   38.1    x     49.8   43.9    34.1       x        67.9    60.1   50.5       x         61.9    54.6      45.8
     LUCIR (CVPR’19)           51.2   41.1   25.2    x     41.7   28.1    18.9       x        56.8    41.4   28.5       x         47.4    37.2      26.6
     MUC∗ [20] (ECCV’20)       49.4   30.2   21.3    x     32.6   26.6    21.9       x         -      35.1    -         x          -       -         -
     SDC∗ [45] (CVPR’20)       56.8   57.0   58.9    x      -      -       -         x         -      61.2    -         x          -       -         -
     ABD∗ [38] (ICCV’21)       63.8   62.5   57.4    x      -      -       -         x         -       -      -         x          -       -         -
     PASS∗ [48] (CVPR’21)      63.5   61.8   58.1    x     49.6   47.3    42.1       x        64.4    61.8   51.3       x          -       -         -
     IL2A∗ [47] (NeurIPS’21)   66.0   60.3   57.9    x     47.3   44.7    40.0       x         -       -      -         x          -       -         -
     SSRE∗ [49] (CVPR’22)      65.9   65.0   61.7    x     50.4   48.9    48.2       x         -      67.7    -         x          -       -         -
     FeTrIL1f c                64.7   63.4   57.4   50.8   52.9   51.7    49.7      41.9      69.6    68.9   62.5      58.9       65.6    64.4      63.4
     FeTrIL1                   66.3   65.2   61.5   59.8   54.8   53.1    52.2      50.2      72.2    71.2   67.1      65.4       66.1    65.0      63.8

Table 1. Average top-1 incremental accuracy in EFCIL with different numbers of incremental steps. FeTrIL1 results are reported with
pseudo-features translated from the most similar new class. ”-” cells indicate that results were not available (see supp. material for details).
”x” cells indicate that the configuration is impossible for that method. Best results - in bold, second best - underlined.

CIL processes. DeeSIL [1] performance is good for T = 5                          CIL Method
                                                                                                                CIFAR-100                ImageNet-Subset
because each class is trained against enough other classes,                                                    T =5      T = 10          T =5       T = 10
but drops significantly for T = 20, when there are few                           LUCIR [13] (CVPR’19)          63.2       61.1           70.8        68.3
new classes. The important performance gain brought by                           +AAnets (CVPR’21)             66.7       65.3           72.6        69.2
FeTrIL highlights the importance of class separability.                          Mnemonics [23] (CVPR’20)      63.3       62.3           72.6        71.4
                                                                                 +AAnets (CVPR’21)             67.6       65.7           72.9        71.9
    Behavior for minimal incremental updates. Com-                               PODNet [7] (ECCV’20)          64.8       63.2           75.5        74.3
pared EFCIL methods can only be updated with a minimum                           +AAnets (CVPR’21)             66.3       64.3           77.0        75.6
of two classes per CIL state since they use discriminative                       FeTrIL1                       66.3       65.2           71.9        70.8
classifiers, which require both positive and negative sam-                 Table 2. Comparison of FeTrIL with the recent AANets
ples. In practice, it is interesting to enable updates once                method [21], applied on top of EBCIL baselines which store 20
each new class is available. This is possible with FeTrIL                  exemplars of past classes to mitigate catastrophic forgetting.
because pseudo-features can all originate from a single new
class. Results in the right columns of CIFAR-100, Tiny-                    mitigates catastrophic forgetting. Following [13, 21], a
ImageNet and ImageNet-Subset from Table 1 show that the                    memory of 20 images per class is allowed for all EBCIL
accuracy obtained in with one class increments is close to                 methods tested here. FeTrIL is better than all three base
that observed for T = 20. This highlights the robustness of                methods to which AANets is applied for CIFAR-100. For
FeTrIL with respect to frequent updates.                                   ImageNet-Subset, FeTrIL accuracy is better than LUCIR’s,
                                                                           slightly behind that of Mnemonics [22] and approximately
    Influence of the final classification layer.
                                                                           3.5 points lower than that of PODNet [7]. The performance
FeTrIL1 compares favorably with FeTrIL1f c . LinearSVC
                                                                           of FeTrIL remains close that of EBCIL methods in a ma-
gives better performance than a fully-connected layer,
                                                                           jority of cases even after the introduction of AANets. The
particularly for a large number of incremental steps.
                                                                           results from Table 2 indicate that, while still present, the gap
However, FeTrIL1f c is also competitive, and outperforms
                                                                           between EFCIL and EBCIL methods is narrowing.
existing methods in a majority of configurations.
    Detailed view of accuracy. We illustrate the evolution                 4.2. Method analysis
of accuracy across incremental states in Figure 3 to com-                     We present an analysis of: (1) the selection strategies, (2)
plement the averaged results from Table 1. These detailed                  the memory footprint of the methods, (3) the complexity of
results confirm the good behavior of the proposed method.                  model updates, and (4) the stability-plasticity balance.
The evolution of accuracy for FeTrIL and SSRE is very sim-                    Pseudo-feature selection comparison. FeTrIL can use
ilar for CIFAR-100, FeTrIL method is better throughout the                 any past-new classes combination for translation. In Ta-
process for TinyImageNet, and also better than SSRE for                    ble 3, we compare the selection strategies from Subsec-
the first incremental states for ImageNet-Subset. The per-                 tion 3.2. Accuracy varies in a relatively small range for all
formance gain with respect to the other compared methods                   strategies, indicating that FeTrIL is robust to the way fea-
is much larger for all incremental states.                                 tures of new classes are selected, and it can be successfully
    Comparison to exemplar-based CIL methods. This                         implemented with any of the strategies. FeTrIL1 is better
comparison is interesting because EFCIL is a much more                     than the other selection methods and this motivates its use
challenging task than EBCIL [2, 24], and an important per-                 in the main experiments. Class similarity matters, but re-
formance gap between the two was observed. This is intu-                   sults with FeTrIL10 remain interesting. FeTrILherd also has
itive since the storage of images of past classes in EBCIL                 interesting accuracy, but is slightly behind that of FeTrIL1 .

                                                                         3916

---

  100%         CIFAR-100, T = 10         100%   TinyImageNet, T = 10         100% ImageNet-Subset, T = 10
          PASS     MUC     SSRE   IL2A        PASS     MUC     SSRE   IL2A
          LUCIR    LwF-MC  DeeSIL FeTrIL      LUCIR    LwF-MC  DeeSIL FeTrIL
   80%                                    80%                                 80%
   60%                                    60%                                 60%
   40%                                    40%                                 40%
   20%                                    20%                                 20% PASS     MUC     SSRE   FeTrIL
                                                                                  LUCIR    LwF-MC  DeeSIL
    0% 0 1 2 3 4 5 6 7 8 9 10 0% 0 1 2 3 4 5 6 7 8 9 10 0% 0 1 2 3 4 5 6 7 8 9 10
               Incremental state                   Incremental state                   Incremental state
               Figure 3. Evolution of top-1 accuracy for an incremental process with T = 10 IL states. Best viewed in color.

                 CIFAR-100     TinyImageNet    ImageNet-Subset
                                                                                       100% ImageNet-Subset, T = 10
                                      T =5
                                                                                        80%
  FeTrIL1           66.3            54.8             72.2
  FeTrIL5           65.7            53.8             72.2                               60%
  FeTrIL10
  FeTrILherd
                    65.1
                    66.2
                                    53.8
                                    53.8
                                                     71.6
                                                     72.1
                                                                                        40%     ova - 71.2    r = 10 - 70.0
                                                                                                r = 25 - 70.8 r = 1 - 67.3
  FeTrILrand        65.1            51.5             70.3                               20% 0 1 2 3 4 5 6 7 8 9 10
Table 3. Average top-1 CIL accuracy obtained with the variants                                 Incremental state
of pseudo-feature selection from Subsection 3.2 for T = 5. We            Figure 4. Top-1 incremental accuracy of FeTrIL1 for approximate
set k = {1, 5, 10} for the similarity rank between the past and          training of the classification layer with different ratios for negative
new classes to test the effect of class similarities. There are 10       sampling. ova denotes a classical one-vs-all training procedure
(CIFAR-100 and ImageNet-Subset) and 20 (TinyImageNet) new                which is used to report the main results from Table 1 and Figure 3.
classes per state from which to select features translation.
                                                                         and 102.4 memory need for 100 and 200 classes, respec-
The results from Table 3 motivate the use of FeTrIL1 in                  tively. The class similarities needed for pseudo-feature se-
the main experiments. Overall, the geometric translation to-             lection (Subsection 3.2) can be computed sequentially and
ward the centroid of the past class is by far more important             the added memory cost of this step is negligible. PASS [48],
than the new classes features sampling policy. This finding              IL2A [47] and SSRE [49] also require the storage of a proto-
is also supported by the results obtained with a single new              type (mean representation) for each past class and their foot-
class per CIL state (Table 1).                                           print is equivalent to that of FeTrIL. IL2A [47] addition-
    Memory footprint. A low memory footprint is a de-                    ally stores a covariance matrix per past class (512x512 for
sirable property of incremental learning algorithms be-                  ResNet-18) for optimal functioning, which is prohibitive.
cause they are most useful in memory-constrained appli-                      Complexity of incremental updates. CIL is useful in
cations [24, 32, 33], and recommended for embedded de-                   resource-constrained environments, and the integration of
vices [9]. All EFCIL methods need to store a representa-                 new classes should be fast [9, 32]. Distillation-based meth-
tion of past classes to counter catastrophic forgetting. Nat-            ods retrain the full backbone model at each update. This is
urally, this representation should be as compact as possible.            is costly because backpropagation complexity depends on
Mainstream methods (such as LwF-MC [19], PASS [48],                      the network architecture, the number of samples and the
IL2A [48], and SSRE [49]) need to the previous and current               number of epochs [8]. Updates of transfer-based methods
deep models during CIL updates for distillation. ResNet-                 are simpler because they update only the final layer. DeeSIL
18 [11], the most frequent CIL backbone, has approxi-                    trains linear classifiers using a one-vs-all procedure within
mately 11.4M parameters. Consequently, distillation-based                each CIL state. The complexity of one training epoch for
methods require around 22.8M parameters. Transfer-based                  all classifiers in a CIL state is O(( Tn )2 sd) [3], with n - to-
methods, such as DeeSIL [1] and FeTrIL, use only the deep                tal number of classes in the dataset, d - dimensionality of
model learned in the initial state and frozen afterwards, and            features and s - samples per class. FeTrIL retrains all lin-
only need 11.4M parameters for the model. DeeSIL does                    ear classifiers, past and new, in each CIL state to improve
not need supplementary parameters during incremental up-                 global separability. Its complexity is O(n2 sd) in the last
dates. However, this comes at the cost of poor global dis-               incremental state, which includes all classes. However, the
crimination of classes, which is reflected in the final per-             one-versus-all training can be replaced with a one-versus-
formance. FeTrIL stores the class centroids of past classes              many training with negligible loss of accuracy. A sampling
in order to perform feature translation. Each class needs                of negative features is performed to respect a predefined
512 parameters, which leads to a supplementary 51.2K                     ratio r between negatives and positives used to train each

                                                                     3917

---

                 100%            TinyImageNet, T = 10                     classes because the deep model is learned with the initial
                                        SSRE                   FeTrIL
Top-1 accuracy

                                                                          classes (a subset of past classes) and then frozen. The accu-
                  75%                                                     racy gap between past and new classes is smaller for FeTrIL
                  50%                                                     compared to SSRE, except for state 4. There, low perfor-
                  25%      Past       Avg                                 mance on new classes is probably explained by a strong do-
                           New                                            main shift compared to the initial state. Globally, the pro-
                   0%0 1 2 3 4 5 6 7 8 9 10 0 1 2 3 4 5 6 7 8 9 10        posed method improves the stability-plasticity balance.
                         Incremental state      Incremental state
  Figure 5. Top-1 incremental accuracy per state for past and new
  classes for TinyImageNet, with T = 10 incremental states for
  FeTrIL1 and SSRE, the best compared method. An ideal method             5. Conclusion
  would provide high accuracy, but also similar performance for past
  and new classes. The accuracy of past and new classes is globally
  closer for FeTrIL1 , which indicates that our method provides a             We introduce FeTrIL, a new method which addresses
  better stability-plasticity balance than SSRE. Overall accuracy is      exemplar-free class-incremental learning. The proposed
  better for FeTrIL1 in Figure 3 because the contribution of new          combination of a frozen feature extractor and of a pseudo-
  classes in each state diminishes during the CIL process.                feature generator improves results compared to recent EF-
                                                                          CIL methods. The generation of pseudo-features is sim-
  classifier. This approximation has O(rnsd) complexity. It               ple, since it consists in a geometric translation, yet effec-
  is interesting since r < n, and is more and more useful as n            tive. Our proposal is advantageous from memory and speed
  grows during the IL process since r remains constant.                   perspectives compared to mainstream methods [13, 33, 38,
      In Figure 4, we present results with different r values for         42, 45, 47, 48, 49]. This is particularly important for edge
  ImageNet-Subset, T = 10. Accuracy drops when negative                   devices [9, 32], whose storage and computation capacities
  sampling is performed, but it is close to that of one-vs-all            are limited. FeTrIL performance is also close to that of
  training when r = 25 and r = 10. Performance drops                      exemplar-based methods, which need to store samples of
  more significantly for r = 1, when each linear classifier is            past classes to mitigate catastrophic forgetting. While a gap
  learned with an aggressive sampling of negatives. Similar               between exemplar-based and exemplar-free setting subsists,
  results for CIFAR-100 and TinyImageNet are provided in                  it becomes significantly narrower. The results reported here
  the suppl. material. Globally, Figure 4 indicates that FeTrIL           resonate with past works which show that simple methods
  increments can be accelerated with little accuracy loss.                can be highly effective in CIL [2, 24, 31]. They question the
      We measure the time needed for incremental training of              usefulness of the knowledge distillation component, used
  ImageNet-Subset, T = 10. The training of the initial model              by a majority of existing methods. The FeTrIL code will be
  is similar for all models and is thus discarded. FeTrIL train-          made public to enable reproducibility.
  ing is done on a single thread of an Intel E5-2620v4 CPU,
                                                                             The main limitations of the proposed method motivate
  and only takes 1 hour, 4 minutes and 16 seconds. If FeTrIL
                                                                          our future work. First, FeTrIL uses a frozen feature ex-
  is run with r = 10 ratio between positives and negatives,
                                                                          tractor learned on the initial state and tends to favor past
  training time is only 15 minutes and 3 seconds. In com-
                                                                          classes over new ones. We will investigate ways to combine
  parison, PASS [48] needs 11 hours, 8 minutes and 19 sec-
                                                                          the pseudo-feature generation mechanism and fine-tuning to
  onds on an NVIDIA V100 GPU, with 4 workers for data
                                                                          further improve global performance, as well as the stability-
  loading. While clearly favorable to FeTrIL, the comparison
                                                                          plasticity balance. Second, FeTrIL produces usable pseudo-
  is biased in favor of PASS since this method uses an en-
                                                                          features, but past class representations would be better if the
  tire GPU, in comparison to a single CPU thread for FeTrIL.
                                                                          pseudo-features would be more similar to the original fea-
  Further speed gains are possible for our method by using a
                                                                          tures of past classes. We will study methods that generate
  GPU implementation of the linear layer. Our method would
                                                                          more refined features, for instance by using the distribution
  run much faster with a GPU implementation of the linear
                                                                          of the initial features. Last but not least, the tested selection
  layer. Note that the running time of the other methods, such
                                                                          strategies are all effective. However, they could be further
  as LUCIR [13] and SSRE [49], which perform backpropa-
                                                                          improved by filtering out outliers based on the localization
  gation is similar to that of PASS [48].
                                                                          of pseudo-features in the representation space.
      Stability-plasticity balance. CIL should ideally ensure
  a similar accuracy level for past and new classes [24, 49].             Acknowledgements. This work was supported by the Eu-
  Figure 5 shows that the two methods have complementary                  ropean Commission under European Horizon 2020 Pro-
  behavior, which results from the way deep backbones are                 gramme, grant number 951911 - AI4Media. It was made
  used. SSRE is biased toward new classes since the model                 possible by the use of the FactoryIA supercomputer, finan-
  is fine tuned in each incremental state. FeTrIL favors past             cially supported by the Ile-de-France Regional Council.

                                                                        3918

---

References                                                                   Barwinska, et al. Overcoming catastrophic forgetting in neu-
                                                                             ral networks. Proceedings of the national academy of sci-
 [1] Eden Belouadah and Adrian Popescu. Deesil: Deep-shallow                 ences, 114(13):3521–3526, 2017.
     incremental learning. TaskCV Workshop @ ECCV 2018.,
                                                                        [16] Alex Krizhevsky. Learning multiple layers of features from
     2018.
                                                                             tiny images. Technical report, University of Toronto, 2009.
 [2] Eden Belouadah, Adrian Popescu, and Ioannis Kanellos.              [17] Matthias De Lange, Rahaf Aljundi, Marc Masana, Sarah
     A comprehensive study of class incremental learning algo-               Parisot, Xu Jia, Ales Leonardis, Gregory G. Slabaugh, and
     rithms for visual tasks. Neural Networks, 135:38–54, 2021.              Tinne Tuytelaars. Continual learning: A comparative study
 [3] Léon Bottou and Olivier Bousquet. The tradeoffs of large               on how to defy forgetting in classification tasks. CoRR,
     scale learning. Advances in neural information processing               abs/1909.08383, 2019.
     systems, 20, 2007.                                                 [18] Ya Le and Xuan Yang. Tiny imagenet visual recognition
 [4] Francisco M. Castro, Manuel J. Marı́n-Jiménez, Nicolás                challenge. CS 231N, 7(7):3, 2015.
     Guil, Cordelia Schmid, and Karteek Alahari. End-to-end in-         [19] Zhizhong Li and Derek Hoiem. Learning without forgetting.
     cremental learning. In Computer Vision - ECCV 2018 - 15th               In European Conference on Computer Vision, ECCV, 2016.
     European Conference, Munich, Germany, September 8-14,
                                                                        [20] Yu Liu, Sarah Parisot, Gregory Slabaugh, Xu Jia, Ales
     2018, Proceedings, Part XII, pages 241–257, 2018.
                                                                             Leonardis, and Tinne Tuytelaars. More classifiers, less for-
 [5] Debasmit Das and CS George Lee. A two-stage approach to                 getting: A generic multi-classifier paradigm for incremen-
     few-shot learning for image recognition. IEEE Transactions              tal learning. In European Conference on Computer Vision,
     on Image Processing, 29:3336–3350, 2019.                                pages 699–716. Springer, 2020.
 [6] Akshay Raj Dhamija, Touqeer Ahmad, Jonathan Schwan,                [21] Yaoyao Liu, Bernt Schiele, and Qianru Sun. Adaptive aggre-
     Mohsen Jafarzadeh, Chunchun Li, and Terrance E Boult.                   gation networks for class-incremental learning. In Confer-
     Self-supervised features improve open-world learning. arXiv             ence on Computer Vision and Pattern Recognition, CVPR,
     preprint arXiv:2102.07848, 2021.                                        2021.
 [7] Arthur Douillard, Matthieu Cord, Charles Ollion, Thomas            [22] Yaoyao Liu, Yuting Su, An-An Liu, Bernt Schiele, and
     Robert, and Eduardo Valle. Podnet: Pooled outputs dis-                  Qianru Sun. Mnemonics training: Multi-class incremental
     tillation for small-tasks incremental learning. In Com-                 learning without forgetting. In 2020 IEEE/CVF Conference
     puter vision-ECCV 2020-16th European conference, Glas-                  on Computer Vision and Pattern Recognition, CVPR 2020,
     gow, UK, August 23-28, 2020, Proceedings, Part XX, volume               Seattle, WA, USA, June 13-19, 2020, pages 12242–12251.
     12365, pages 86–102. Springer, 2020.                                    IEEE, 2020.
 [8] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep           [23] Yaoyao Liu, Yuting Su, An-An Liu, Bernt Schiele, and
     learning. MIT press, 2016.                                              Qianru Sun. Mnemonics training: Multi-class incremen-
 [9] Tyler L Hayes and Christopher Kanan.             Online con-            tal learning without forgetting. In The IEEE Conference on
     tinual learning for embedded devices.          arXiv preprint           Computer Vision and Pattern Recognition (CVPR), 06 2020.
     arXiv:2203.10681, 2022.                                            [24] Marc Masana, Xialei Liu, Bartlomiej Twardowski, Mikel
[10] Chen He, Ruiping Wang, Shiguang Shan, and Xilin Chen.                   Menta, Andrew D. Bagdanov, and Joost van de Weijer.
     Exemplar-supported generative reproduction for class in-                Class-incremental learning: survey and performance evalu-
     cremental learning. In British Machine Vision Conference                ation on image classification, 2021.
     2018, BMVC 2018, Northumbria University, Newcastle, UK,            [25] Michael Mccloskey and Neil J. Cohen. Catastrophic in-
     September 3-6, 2018, page 98, 2018.                                     terference in connectionist networks: The sequential learn-
[11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.                  ing problem. The Psychology of Learning and Motivation,
     Deep residual learning for image recognition. In Conference             24:104–169, 1989.
     on Computer Vision and Pattern Recognition, CVPR, 2016.            [26] Thomas Mensink, Jakob Verbeek, Florent Perronnin, and
[12] Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean.                    Gabriela Csurka. Distance-based image classification: Gen-
     Distilling the knowledge in a neural network. CoRR,                     eralizing to new classes at near-zero cost. IEEE transactions
     abs/1503.02531, 2015.                                                   on pattern analysis and machine intelligence, 35(11):2624–
[13] Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, and                 2637, 2013.
     Dahua Lin. Learning a unified classifier incrementally via re-     [27] M Mermillod, A Bugaiska, and P Bonin. The stability-
     balancing. In IEEE Conference on Computer Vision and Pat-               plasticity dilemma: investigating the continuum from catas-
     tern Recognition, CVPR 2019, Long Beach, CA, USA, June                  trophic forgetting to age-limited learning effects. Frontiers
     16-20, 2019, pages 831–839, 2019.                                       in Psychology, 4:504–504, 2013.
[14] Ronald Kemker, Marc McClure, Angelina Abitino, Tyler               [28] Behnam Neyshabur, Hanie Sedghi, and Chiyuan Zhang.
     Hayes, and Christopher Kanan. Measuring catastrophic for-               What is being transferred in transfer learning? arXiv preprint
     getting in neural networks. In Proceedings of the AAAI Con-             arXiv:2008.11687, 2020.
     ference on Artificial Intelligence, volume 32, 2018.               [29] German Ignacio Parisi, Ronald Kemker, Jose L. Part,
[15] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel                Christopher Kanan, and Stefan Wermter. Continual lifelong
     Veness, Guillaume Desjardins, Andrei A Rusu, Kieran                     learning with neural networks: A review. Neural Networks,
     Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-                     113, 2019.

                                                                      3919

---

[30] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort,              [43] Max Welling. Herding dynamical weights to learn. In Pro-
     Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu                ceedings of the 26th Annual International Conference on
     Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg,                 Machine Learning, ICML 2009, Montreal, Quebec, Canada,
     Jake VanderPlas, Alexandre Passos, David Cournapeau,                     June 14-18, 2009, pages 1121–1128, 2009.
     Matthieu Brucher, Matthieu Perrot, and Edouard Duches-              [44] Yue Wu, Yinpeng Chen, Lijuan Wang, Yuancheng Ye,
     nay. Scikit-learn: Machine learning in python. CoRR,                     Zicheng Liu, Yandong Guo, and Yun Fu. Large scale in-
     abs/1201.0490, 2012.                                                     cremental learning. In IEEE Conference on Computer Vi-
[31] Ameya Prabhu, Philip HS Torr, and Puneet K Dokania.                      sion and Pattern Recognition, CVPR 2019, Long Beach, CA,
     Gdumb: A simple approach that questions our progress in                  USA, June 16-20, 2019, pages 374–382, 2019.
     continual learning. In European Conference on Computer              [45] Lu Yu, Bartlomiej Twardowski, Xialei Liu, Luis Herranz,
     Vision, pages 524–540. Springer, 2020.                                   Kai Wang, Yongmei Cheng, Shangling Jui, and Joost van de
[32] Leonardo Ravaglia, Manuele Rusci, Davide Nadalini,                       Weijer. Semantic drift compensation for class-incremental
     Alessandro Capotondi, Francesco Conti, and Luca Benini. A                learning. In 2020 IEEE/CVF Conference on Computer Vision
     tinyml platform for on-device continual learning with quan-              and Pattern Recognition, CVPR 2020, Seattle, WA, USA,
     tized latent replays. IEEE Journal on Emerging and Selected              June 13-19, 2020, pages 6980–6989. IEEE, 2020.
     Topics in Circuits and Systems, 11(4):789–802, 2021.                [46] Bowen Zhao, Xi Xiao, Guojun Gan, Bin Zhang, and Shu-Tao
[33] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg                    Xia. Maintaining discrimination and fairness in class incre-
     Sperl, and Christoph H. Lampert. icarl: Incremental classi-              mental learning. In 2020 IEEE/CVF Conference on Com-
     fier and representation learning. In Conference on Computer              puter Vision and Pattern Recognition, CVPR 2020, Seattle,
     Vision and Pattern Recognition, CVPR, 2017.                              WA, USA, June 13-19, 2020, pages 13205–13214. IEEE,
[34] Ethan M Rudd, Lalit P Jain, Walter J Scheirer, and Ter-                  2020.
     rance E Boult.        The extreme value machine.         IEEE       [47] Fei Zhu, Zhen Cheng, Xu-yao Zhang, and Cheng-lin Liu.
     transactions on pattern analysis and machine intelligence,               Class-incremental learning via dual augmentation. Advances
     40(3):762–768, 2017.                                                     in Neural Information Processing Systems, 34, 2021.
[35] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, San-           [48] Fei Zhu, Xu-Yao Zhang, Chuang Wang, Fei Yin, and Cheng-
     jeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpa-                     Lin Liu. Prototype augmentation and self-supervision for
     thy, Aditya Khosla, Michael S. Bernstein, Alexander C.                   incremental learning. In Proceedings of the IEEE/CVF Con-
     Berg, and Fei-Fei Li. Imagenet large scale visual recogni-               ference on Computer Vision and Pattern Recognition, pages
     tion challenge. International Journal of Computer Vision,                5871–5880, 2021.
     115(3):211–252, 2015.                                               [49] Kai Zhu, Wei Zhai, Yang Cao, Jiebo Luo, and Zheng-
[36] Jeffrey C Schlimmer and Douglas Fisher. A case study of                  Jun Zha. Self-sustaining representation expansion for non-
     incremental concept induction. In AAAI, volume 86, pages                 exemplar class-incremental learning. In Proceedings of
     496–501, 1986.                                                           the IEEE/CVF Conference on Computer Vision and Pattern
[37] Ali Sharif Razavian, Hossein Azizpour, Josephine Sullivan,               Recognition, pages 9296–9305, 2022.
     and Stefan Carlsson. Cnn features off-the-shelf: an astound-
     ing baseline for recognition. In Proceedings of the IEEE con-
     ference on computer vision and pattern recognition work-
     shops, pages 806–813, 2014.
[38] James Smith, Yen-Chang Hsu, Jonathan Balloch, Yilin Shen,
     Hongxia Jin, and Zsolt Kira. Always be dreaming: A new
     approach for data-free class-incremental learning. arXiv
     preprint arXiv:2106.09701, 2021.
[39] Chuanqi Tan, Fuchun Sun, Tao Kong, Wenchang Zhang,
     Chao Yang, and Chunfang Liu. A survey on deep transfer
     learning. In International conference on artificial neural net-
     works, pages 270–279. Springer, 2018.
[40] Gido M Van de Ven and Andreas S Tolias. Three scenar-
     ios for continual learning. arXiv preprint arXiv:1904.07734,
     2019.
[41] Ragav Venkatesan, Hemanth Venkateswara, Sethuraman
     Panchanathan, and Baoxin Li.            A strategy for an
     uncompromising incremental learner.           arXiv preprint
     arXiv:1705.00744, 2017.
[42] Vinay Kumar Verma, Kevin J. Liang, Nikhil Mehta, Piyush
     Rai, and Lawrence Carin. Efficient feature transformations
     for discriminative and generative continual learning. CoRR,
     abs/2103.13558, 2021.

                                                                       3920

---
