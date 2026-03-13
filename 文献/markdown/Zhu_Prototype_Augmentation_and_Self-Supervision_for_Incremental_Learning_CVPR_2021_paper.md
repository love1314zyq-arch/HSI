# Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper

Source: `Zhu_Prototype_Augmentation_and_Self-Supervision_for_Incremental_Learning_CVPR_2021_paper.pdf`

Prototype Augmentation and Self-Supervision for Incremental Learning

             Fei Zhu1,2 , Xu-Yao Zhang1,2∗, Chuang Wang1,2 , Fei Yin1,2 , Cheng-Lin Liu1,2,3
         1
           NLPR, Institute of Automation, Chinese Academy of Sciences, Beijing 100190, China
 2
   School of Artificial Intelligence, University of Chinese Academy of Sciences, Beijing 100049, China
    3
      CAS Center for Excellence of Brain Science and Intelligence Technology, Beijing 100190, China
             zhufei2018@ia.ac.cn, {xyz, fyin, liucl}@nlpr.ia.ac.cn, wangchuang@ia.ac.cn

                            Abstract                                  and almost fully forget previously acquired knowledge. Mo-
                                                                      tivated by this, a multitude of works [28, 34, 43, 51, 42]
   Despite the impressive performance in many individual              have recently emerged that try to alleviate the catastrophic
tasks, deep neural networks suffer from catastrophic for-             forgetting [16, 37, 13] problem. In this paper, we consider
getting when learning new tasks incrementally. Recently,              a challenging scenario of class-incremental learning (CIL),
various incremental learning methods have been proposed,              in which each task in the sequence contains a set of classes
and some approaches achieved acceptable performance rely-             disjoint from the old tasks, and the model need to learn a
ing on stored data or complex generative models. However,             unified classifier that can classify all classes seen at different
storing data from previous tasks is limited by memory or pri-         stages without the task-identifier at inference time.
vacy issues, and generative models are usually unstable and               Intuitively, catastrophic forgetting is caused by overlap-
inefficient in training. In this paper, we propose a simple non-      ping or confusion between the representations of new and
exemplar based method named PASS, to address the catas-               old classes in the feature space. When learning new classes,
trophic forgetting problem in incremental learning. On the            the decision boundary for previous classes can be dramati-
one hand, we propose to memorize one class-representative             cally changed, and the unified classifier is severely biased.
prototype for each old class and adopt prototype augmen-              To address this issue and maintain previous knowledge, one
tation (protoAug) in the deep feature space to maintain the           can store a fraction of old data to jointly train the model with
decision boundary of previous tasks. On the other hand, we            current data [50, 43, 50, 6, 12]. However, storing data is
employ self-supervised learning (SSL) to learn more gen-              undesirable due to memory limits or privacy issues, in which
eralizable and transferable features for other tasks, which           the data are not allowed to be stored. An alternative way is
demonstrates the effectiveness of SSL in incremental learn-           to learn deep generative models to generate pseudo-samples
ing. Experimental results on benchmark datasets show that             of previous classes [46, 49, 51, 25]. Nevertheless, it is inef-
our approach significantly outperforms non-exemplar based             ficient to train big generative models such as GAN [17, 3]
methods, and achieves comparable performance compared                 and autoencoder [27, 25] for complex datasets (e.g., natural
to exemplar based approaches.                                         images). Moreover, the generative models also suffer from
                                                                      catastrophic forgetting. Another direction is to identify and
                                                                      penalize future changes to some important parameters of the
1. Introduction                                                       original model [28, 54]. These regularization strategies are
                                                                      effective in scenarios where multi-head classifiers are used
   Incremental learning (IL) enables humans to acquire
                                                                      and the task-identifier is available at inference. However, as
novel experience continually while maintaining existing
                                                                      noticed in some works [23, 48], those methods show poor
knowledge. In dynamic and open environment, it is crit-
                                                                      performance in CIL scenario.
ical for modern artificial intelligence to have the ability of IL
because training examples in real-world applications usually              Besides the catastrophic forgetting, another obstacle for
appear sequentially. For instance, a face recognition sys-            IL is the task-level overfitting phenomenon, which has been
tem may encounter new faces which need to be added and                ignored by previous works. Specifically, DNNs can easily
learned throughout its life without forgetting or re-learning         overfit to the training task when learning task continually. In-
the people already learned. However, deep neural networks             tuitively, the model may focus on capturing features that are
(DNNs) tend to adjust the learned parameters to new task              useful for current task, while discarding those less discrim-
                                                                      inative directions which could capture data characteristics
  ∗ Corresponding author.                                             for future tasks. This may not be a problem for common

                                                                    5871

---

Figure 1: Motivation of PASS. (a) When learning new task, the decision boundary of previous tasks could be dramatically
changed, resulting in catastrophic forgetting. ProtoAug is proposed to restrain the decision boundary, thus maintaining the
discrimination and balance between old and new classes. (b) If the learned features are task-specific in each stage, the model
trained on previous task might be a bad initialization for current task. We propose to leverage the benefit of SSL to learn richer
and more transferable features. Intuitively, different tasks would be closer in the parameter space, and it would be easier to
find a model to perform well on all tasks, thus improving both the stability and plasticity of the model.

single task learning scenario, but leads particles influence       ferent tasks would be closer in the parameter space, and
for IL since the model for current task is initialized with        the model trained on current task would be a better initial-
previous model. A recent study [42] found that a model             ization for learning the next task. In conclusion, our main
trained from scratch using samples stored can surprisingly         contributions are summarized as follows:
outperforms many recently proposed algorithms. This study
indicates that the previous model, which mainly carries task-           • We propose a simple and effective non-exemplar based
specific features, might be a bad initialization for current              method to overcome catastrophic forgetting problem in
task, as shown in Fig. 1(b). Consequently, the model would                CIL by memorizing and augmenting prototypes of old
need more updates to perform well on current task, which                  classes in the deep feature space.
increases the forgetting problem on the other hand.                     • We emphasize the task-level overfitting phenomenon
   Motivated by the above analysis, we propose to improve                 in IL, and adopt self-supervised learning to learn more
CIL performance by maintaining the decision boundary and                  generalizable and transferable features.
reducing task-level overfitting phenomenon, as shown in
Fig. 1. The proposed PASS mainly consists of Prototype                  • Our method significantly outperforms non-exemplar
Augmentation and Self-Supervision. On the one hand,                       based methods and obtains comparable results com-
prototype augmentation (protoAug) memorizes one class-                    pared to exemplar based methods in CIL scenario.
representative prototype (typically the class mean in the deep
feature space) for each old class, and augments the mem-
                                                                   2. Related Work
orized prototypes via Gaussian noise when learning new
classes. Then, the augmented prototypes and deep features          Incremental Learning. IL has been a long-standing re-
of new data are jointly classified to maintain the discrimi-       search topic. Several early approaches used nearest class
nation and balance between old and new classes. This is            mean classifier [38] or random forest [47] for IL based on
inspired by a recent work [35] in long-tailed recognition          fixed data representations. Recently, a variety of attempts
which expands the distribution of the tail classes by aug-         have been made to enable IL for DNNs. Regularization
menting the tail classes with certain disturbances. While          strategies such as elastic weight consolidation (EWC) [28],
[35] focuses on class-imbalance learning and learns the em-        synaptic intelligence (SI) [54], and memory aware synapses
bedding augmentation strategy from the head classes, in our        (MAS) [2] use different metrics to identify and penalize the
work, we focus on CIL and investigate the value of simple          changes of important parameters of the original network
Gaussian noise based augmentation.                                 when learning a new task. An alternative solution is to per-
   On the other hand, we take inspiration from self-               form implicit regularization by using knowledge distillation
supervised learning (SSL) to alleviate task-level overfitting      technique [34, 21]. Nevertheless, it is hard to design a rea-
phenomenon in IL. In particular, SSL aims to learn trans-          sonable metric to evaluate the importance of parameters of
ferable representations that would be useful for other tasks.      a model, and the performances of regularization strategies
Inspired by the natural connection between IL and SSL, we          based methods for CIL remain significantly inferior to those
propose to leverage the benefit of SSL to learn task-agnostic      obtained by joint training. Recently, Yu et al., [53] found that
and transferable representations. Intuitively, with SSL, dif-      embedding network suffers less forgetting for CIL. However,

                                                                 5872

---

Figure 2: Illustration of PASS for CIL. The classes of current task are augmented by rotation based transformation [32], and
the augmented data are fed to the feature extractor. In the deep feature space, we augment the memorized prototypes (one for
each classes) via Gaussian noise (right). Our method is non-exemplar based, simple and effective.

training embedding network with metric learning could often           in CIL, revealing surprising yet intriguing findings that SSL
be harder than softmax-based networks.                                can boost the performance of CIL significantly.
    Another direction is rehearsal strategies, which provide a
strong baseline for CIL by storing and replaying a fraction           3. Methodology
of samples from the old classes. With stored samples, some
works [50, 43, 12] use a distillation loss to prevent forgetting,
                                                                      3.1. Problem Statement and Analysis
while others [44, 8] only include classification loss and con-            The goal of CIL is to sequentially learn a unified model
struct each mini-batch with an equal amount of new data and           to classify the test samples of all classes that have been
the rehearsal data. More recently, the imbalance problem              learned so far. Specifically, the model consists of two parts:
between the previous and current tasks has been found to              the feature extractor Fθ and a unified classifier Gφ . Let
be constituting a key challenge for CIL, and several works,           D = {Dt }Tt=1 be a stream of data, where Dt = {Xt , Yt } =
such as EEIL [6], BiC [50], UCIR [22] and WA [56] were                {xt,j , yt,j }N t
                                                                                    j=1 is the dataset that the system receives at step
proposed to reduce the bias towards currents tasks. However,          t. Dataset Dt consists of Nt labeled samples for training,
those techniques may not be applicable without storing data.          and yt,j ∈ Ct , where Ct is the class set of task t and the
                                                                      class sets of different task are disjoint. At step t, the goal
Without directly storing raw data, a line of work [46, 49, 51]
                                                                      is to minimize a predefined loss function L on new dataset
sequentially constructs a separate generative model to gener-         Dt without interfering with and possibly improving on those
ate old samples. Nevertheless, those approaches rely heavily          that were learned previously [1]:
on the quality of the generative model. In this paper, we aim                                                                         X
to reduce catastrophic forgetting in CIL without storing old                   {θt , φt } = argmin Lt (G(F (Xt ; θt ); φt ), Yt ) +       ǫi
                                                                                            θt ,φt ,ǫ                                          ,
data or leveraging complex generative models.
                                                                           s.t. Lt (Xi , Yi ) − Li (Xi , Yi ) 6 ǫi , ǫi > 0; ∀i ∈ [1, t − 1]
Self-Supervised Learning. Recently, learning with self-                                                                                        (1)
supervision [24] has been demonstrated effective to learn              where Lt (Xi , Yi ) = L(G(F (Xi ; θt ); φt ), Yi ) is the loss
general representations, by learning some proxy tasks, e.g.           of the model at t on old data set Di and Li (Xi , Yi ) =
prediction rotations [15], patch permutation [40], image col-         L(G(F (Xi ; θi ); φi ), Yi ) is the loss of the previous model
orization [30] and clustering [4, 5]. More recently, con-             at i on old dataset Di . The last term ǫ = {ǫi } is a slack
trastive losses based SSL methods [9, 18] show great suc-             variable that tolerates a small increase in old dataset.
cess. By SSL, the model could learn features that are un-                There are mainly two obstacles in CIL: classifier bias and
necessary for current task but useful for other tasks, e.g.,          task-level overfitting. First, with only new data, the decision
semi-supervised learning [55], few-shot learning [14], and            boundary learned previously can be dramatically changed,
improving robustness [20]. In particular, it has been found           and the unified classifier is severely biased. Second, it is dif-
that self-supervised pretraining is a good choice to initialize       ficult to learn general features which could be generalizable
the model for class-imbalance learning [52]. Lee et al., [32]         well on other classes with data only for current classes. As
propose to augment original labels via self-supervision of            a result, the feature extractor is also biased and the param-
input transformation, and show that the supervised classifi-          eter space of model at different stages would be far, which
cation accuracy could be improved by this simple technique.           makes it difficult to find a model to perform well on all tasks.
Inspired by the natural connection between IL and SSL, we             Therefore, from a multi-task learning perspective, learning
employ the self-supervised method in [32] to investigate SSL          task-agnostic representations is important for CIL.

                                                                    5873

---

Overview of Framework. The framework of our method is               classes are augmented with soft variance, which represents
shown in Fig. 2. Specifically, for each old class, we do not        the confidence of reality of the features generated. During
store any old samples, but to memorize a class-representative       training with current data, the augmented features are feed
prototype in the deep feature space. Then, when learning            to classifier to maintain discrimination and balance among
new task, each old prototype is augmented with certain dis-         all classes that have been learned so far.
turbances and fed to the unified classifier for classification.
Consequentially, it alleviates the distortion of the learned        3.3. SSL based Label Augmentation
feature space and the classifier bias. In addition, to reduce          Inspried by [32], we simply learn a unified model by
the task-level overfitting, SSL is adopted to learn more gen-       augmenting the current class based on SSL. Specifically,
eral features for other (previous and future) tasks by using        for each class, we rotate its training data 90, 180, and 270
rotation-based label augmentation [32].                             degrees to generate 3 novel classes, extending the original
                                                                    K-class problem to a new 4K class problem:
3.2. Prototype Augmentation
   At stage t, only Dt is available for training, thus we                     X′ t = rotate(Xt , θ), θ ∈ {90, 180, 270},         (6)
can not directly optimize Eq. (1). To alleviate distortion of
                                                                    and the augmented sample is assigned a new label Y′ t .
the feature space when learning new task, we compute and
                                                                    Comparing the widely used 4-way self-supervised tasks,
memorize one prototype (class mean) for each classes:
                                                                    as demonstrated in [32], the above approach relaxes a cer-
                           1 X
                               Nt,k                                 tain invariant constraint during learning the original and
                µt,k =              F (Xt,k ; θt ).        (2)      self-supervised tasks simultaneously, which is beneficial to
                          Nt,k n=1
                                                                    learning richer features. As shown in our experiments, the
When learning new task, the prototype of each old class, e.g.       performance of CIL can be improved by this simple method.
class kold at stage told , is augmented as below (shown in          3.4. Integrated Objective of PASS
Fig. 2):
               Ftold ,kold = µtold ,kold + e ∗ r,        (3)           When learning new classes, the feature extractor would
                                                                    be updated continually. To alleviate the mismatch between
where e ∼ N (0, 1) is the derived Gaussion noise which has
                                                                    the saved old prototypes and the feature extractor, the well-
the same dimension as prototype. r is a scale to control
                                                                    known knowledge distillation (KD) [21, 22] is employed
the uncertainty of the augmented prototypes. In particular,
                                                                    to regularize the feature extractor. Specifically, we restrain
the scale r can be pre-defined, or computed as the average
                                                                    the feature extractor by matching the features of new data
variance of the class representations:
                                                                    extracted by current model with that of previous model:
               1                     X
                                     new   K
                                         Tr(Σt,k )
  rt2 =                        2
                      (Kold ∗ rt−1 +               ), (4)                     Lt,kd = kFt (X′ t ; θt ) − Ft−1 (X′ t ; θt−1 )k.   (7)
          Kold + Knew                       D
                                            k=1
                                                                    Combining the techniques presented above, we reach a total
where Kold and Knew represent the number of old classes             loss of PASS that comprised of three terms, given as:
and new classes at stage t, respectively. D is the dimension
of the deep feature space. Σt,k is the covariance matrix for                Lt,total = Lt,ce + λ ∗ Lt,protoAug + γ ∗ Lt,kd .     (8)
the features from class k at stage t, and the Tr operation
computes the trace of a matrix. We observed that the rt             L     = Lt,ce (G(F (X′ t ; θt ); φt ), Y′ t ), and Lt,protoAug =
                                                                    Pt,ce
                                                                      t−1
changes slightly at different stage in the course of a CIL            i=1 Lt,ce (G(Fi ; φt ), Yi ). λ and γ are loss weights, and
experiment. Therefore, one can only compute and use the             we use λ = γ = 10 in our experiments.
average variance of
                  PK the features in the first task as follows:
r2 = r12 = K11∗D k=1   1
                         Tr(Σ1,k ).
                                                                    3.5. Preliminary Experiments
   Then, the features of new classes and the augmented              3.5.1    2D Visualization of ProtoAug
prototypes are feed to the unified classifier. As a result,
Eq. (1) could be empirically approximated by Eq. (5):               To provide an illustration of protoAug, we conduct exper-
                                                                    iment on MNIST [31] with a 2-dimensional feature space
      {θt , φt } = argmin{Lt (G(F (Xt ; θt ); φt ), Yt )            which is suitable for visualization. SSL is not applied here
                    θt ,φt ,ǫ
                                                                    since the effect of protoAug is the focus in this experiment.
                    X
                    t−1                                    (5)      We start from a Resnet-18 model [19] trained on 4 classes
                +         L(G(Fi ; φt ), Yi )},                     and the remaining 6 classes are continually added in 3 phases.
                    i=1
                                                                    We compare our method with finetuning, LwF [34], and
where Fi represents the features augmented for old class set        LwF-MC (binary cross entropy based) [43]. As shown in
Ci . Intuitively, in the feature space, the prototypes of old       Fig. 3, the distribution of old classes is dramatically changed

                                                                  5874

---

Figure 3: Visualization of class representations in the feature space when learning MNIST [31] incrementally. The outputted
features are 2-dimensional which is suitable for visualization. Best viewed in color.

        Table 1: Results of zero-cost class incremental learning. The model is tested using nearest class mean classifier.
                        #classes              4 (base)      5        6       7         8        9       Final    Average
                                   Baseline      —        27.20    20.55   17.40     17.23    15.68     14.80    18.81
                          Novel
                                   + SSL         —        76.40    61.10   46.83     40.80    40.36     37.57    50.50+31.69
            CIFAR-10
                                   Baseline    94.55      79.26    68.00   59.65     52.88    48.97     46.46    64.25
                           All
                                   + SSL       95.35      87.26    79.22   70.04     64.05    61.18     58.36    73.63+9.38
                        #classes              40 (base)    50       60      70        80       90       Final    Average
                                   Baseline      —        43.50    33.10   30.43     27.45    25.20     23.58    30.54
                          Novel
                                   + SSL         —        55.70    44.85   42.67     38.37    34.70     32.15    41.46+10.92
            CIFAR-100
                                   Baseline    71.83      63.60    55.73   50.64     46.38    42.61     39.37    52.93
                           All
                                   + SSL       72.03      64.52    58.37   54.46     50.50    46.48     43.46    55.68+2.74

in finetuning, and there is an obvious overlap of distribution      classes of CIFAR-10, and surpasses the baseline model by
from different classes, resulting in catastrophic forgetting.       a large margin of 31.69%. Similarly, on CIFAR-100, SSL
Contrarily, our method can maintain the distribution of old         based model outperforms the baseline model by a margin of
classes when learning new classes, thus reduces the forget-         10.92%. Those results strongly demonstrate the suitability
ting phenomenon in the course of CIL.                               and effectiveness of SSL for CIL.

                                                                    Deep feature space anaysis. An intuitively explanation for
3.5.2    A Closer Look at SSL for CIL                               the effectiveness of SSL on the above experiments is that
Setup. We train ResNet-18 for classifying CIFAR-10 and              SSL improves the separation of the distribution of novel
CIFAR-100 [29]. Similar to [33, 39], we first train a classi-       classes. As shown in Fig. 4, the class representations of novel
fication model on some base classes. Then a nearest class           classes are much more separated with SSL, and the overlap
mean (NCM) classifier is built on the pre-trained feature           between base and novel classes is less, comparing with
extractor to classify both base and new classes incrementally.      baseline model. We further anaysis the deep feature space
For SSL based model, the based classes are augmented using          quantitatively. Specifically, we   P use average inter-class dis-
                                                                                                1
the label augmentation method in Section 3.3. We train all          tances πinter (F ) = Zinter          yl ,yk ,l6=k d(µ(Fyl ), µ(Fyk )),
the models for 120 epochs with batch size 64 and Adam [26]          and average
                                                                           P        P intra-class       distances       πintra (F )    =
                                                                       1
optimizer with 0.001 initial learning rate, and the learning        Zintra    yl ∈y   fi ,fj ∈Fyl ,i6=j d(fi , fj ) to measure the distri-
rate is multiplied by 0.1 after 50 and 100 epochs.                  bution of class representations. d(·; ·) is the cosine distance
                                                                    in our experiment. Fyl = {fi := fθ (xi )|xi ∈ X, yi = yl }
Results. For each learning stage, we report the test accu-          denotes the set of embedded samples of a class yl . µ(Fyl )
racy on novel classes that appeared so far. And we also             is their mean embedding. Zintra and Zinter are two
test on both base and novel classes that appeared so far. As        normalization constants.
shown in Table 1, the accuracy of novel classes can be signif-
icantly improved with SSL. For instance, SSL based model            As shown in Fig. 4, for unseen classes, SSL results in
achieves 50.50% average incremental task accuracy on novel          smaller intra distance on novel classes, which implies that

                                                                  5875

---

Figure 4: (a-b) SSL improves the separation of the distribution of novel classes, and reducing the the overlap between base and
novel classes. (c-b) SSL results in smaller intra distance on novel classes, and high feature space density.

          Figure 5: Results of classification accuracy on CIFAR-100, which contains 5, 10 and 20 sequential tasks.

the model learned with SSL generalizes better than base-           on CIFAR-100, we mainly train the model on half of classes
line on novel classes. While for training classes, baseline        for the first task, and equal classes in the rest phases.
has more compact feature distributions. This indicates that
representation learning for new class generalization may           Comparison Approaches. We compare our method (PASS)
be hurt by excessive feature compression. In particular,           with non-exemplar based methods such as EWC [28], LwF
Roth et al., [45] proposed a concept of feature space den-         [34], LwF-MC [43], LwM [11] and MUC [36]. We also com-
sity: πratio (F ) = πintra (F )/πinter (F ), and found that an     pare with several state-of-the-art exemplar-based approaches:
increased feature space density πratio is linked to stronger       iCaRL [43], EEIL [6], UCIR [22]. Note that our method is
generalization under considerable shifts between training          non-exemplar based since we do not save any old samples,
and testing distribution. Fig. 4(d) shows that SSL leads to a      but to memorize one prototype in the deep feature space
higher feature space density πratio , and the improvement on       for each class, which is very memory efficient and has no
generalization is consistent with the observation in [45].         privacy issues.

                                                                   Evaluation metrics. We report the standard metrics to
4. Experiments
                                                                   measure the quality of CIL: Accuracy [43] is computed
Datasets. We perform our experiments on CIFAR-100 [29],            as the average accuracy of all the classes that have already
TinyImageNet [41] and ImageNet-Subset [10]. The classes            been learned. Average forgetting [7] is defined to estimate
are arranged in a fixed random order. Except for one setting       the forgetting of previous tasks. The forgetting measure

                                                                 5876

---

        Figure 6: Results of classification accuracy on TinyImageNet, which contains 5, 10 and 20 sequential tasks.

fki of the i-th task after training k-th task is defined as        Table 2: Results of average forgetting on CIFAR-100 and
fki = max (at,i −ak,i ), ∀i < k, in which am,n is the ac-          TinyImageNet.
      t∈1,...,k−1
curacy of task n after training task m. The average forgetting                         CIFAR-100                   TinyImageNet
                                         1
                                           Pk−1 i                   Method    5 phases 10 phases 20 phases 5 phases 10 phases 20 phases
measure Fk is then defined as Fk = k−1        i=1 fk .              LwF_MC     44.23     50.47     55.46    54.26     54.37     63.54
                                                                    MUC        40.28     47.56     52.65    51.46     50.21     58.00
Implementation details.1 ResNet-18 [19] is used and                 PASS       25.20     30.25     30.61    18.04     23.11     30.55
trained from scratch in our experiments. We train all the           iCaRL-CNN 42.13      45.69     43.54    36.89     36.70     45.12
models with batch size 64 and Adam [26] optimizer with              iCaRL-NCM 24.90      28.32     35.53    27.15     28.89     37.40
0.001 initial learning rate. We train all the models for 100        EEIL       23.36     26.65     32.40    25.56     25.91     35.04
                                                                    UCIR       21.00     25.12     28.65    20.61     22.25     33.74
epochs, and the learning rate is multiplied by 0.1 after 45
and 90 epochs. All the experiments are repeated three times
and the average results are reported. We conduct different in-
cremental settings (5, 10 and 20 phases) for both CIFAR-100
and TinyImageNet. For ImageNet-Subset, we use the 10 in-
cremental phases evaluation protocol. After each phase, the
model is evaluated on all the learned classes so far. For the
exemplar-based approaches: iCaRL [43], EEIL [6], UCIR
[22], we use herd selection [43] to select and store 20 sam-
ples per old class, which is a common setting [43, 22].
4.1. Comparative Results                                           Figure 7: Results of classification accuracy on ImageNet-
                                                                   Subset, which contains 10 sequential tasks.
Results are shown in Fig. 5, Fig. 6 and Fig. 7. We ob-
serve that our method outperforms significantly better than
                                                                   suffers from less forgetting than iCaRL-NCM on CIFAR-
non-exemplar based methods, which confirms that PASS
                                                                   100. The results on TinyImageNet are also conclusive. In
can effectively address the catastrophic forgetting in CIL
                                                                   conclusion, PASS outperforms all the non-exemplar based
without storing old training samples. Take the results of 10
                                                                   methods and some exemplar based methods in terms of both
phases as an example, our method outperforms the best non-
                                                                   accuracy and average forgetting.
exemplar methods MUC [36] with a gap of 29.3% on CIFAR-
100 and with a gap of 25.2% on TinyImageNet. In addi-              The comparison of the confusion matrix. Fig. 8 shows the
tion, our method outperforms the strong baseline method,           comparison of confusion matrix by finetuning, iCaRL, and
iCaRL-NCM [43], by 3.7% on CIFAR-100 (10 phases), and              our approach. The diagonal entries represent the correction
achieves comparable accuracy with state-of-the-art exemplar-       predictions and off-diagonal entries represent the misclassi-
based approaches which are based on many saved samples             fication. Because of the severe imbalance between old and
overall. The observations on ImageNet-Subset are consistent        new classes, finetuning tends to classify the samples into
with those on CIFAR-100 and TinyImageNet.                          new classes (strong confusions on the last task), as shown
                                                                   in Fig. 8(a). PASS is capable to remove most of the bias
To compare the effectiveness of alleviating forgetting, we
                                                                   and achieves better overall performance without relying on
show the average forgetting results in Table 2. Our method
                                                                   stored data of old classes.
  1 Code available at https://github.com/Impression2805/

CVPR21_PASS.                                                       The comparison of weight in the FC Layer. For the ex-

                                                                 5877

---

                                                                                            Table 3: The effectiveness of each component in our method.
                                                 #dataset & classes                                                                                                      CIFAR-100                        TinyImageNet
                                                  Method            protoAug                                                               SSL        5 phases           10 phases 20 phases   5 phases    10 phases 20 phases
                                                    KD                 %                                                                    %          14.33               6.04      5.67        7.23         4.70      4.23
                                                  KD+SSL               %                                                                    X          17.15               8.46      8.57        9.71         6.53      6.60
                         Accuracy
                                                KD+protoAug             X                                                                   %          50.19               39.80     38.61      33.11        26.52     20.97
                                              KD+protoAug+SSL           X                                                                   X          55.67               49.03     48.48      41.58        39.28     32.78
                                                KD+protoAug             X                                                                   %          28.72               35.70     40.59      25.62        35.33     43.91
                         Forgetting
                                              KD+protoAug+SSL           X                                                                   X          25.20               30.25     30.61      18.04        23.12     30.55

                0                                                   0                                                   0

                                                                                                                                                            0.8
               20                                                  20                                                  20
true classes

                                                    true classes

                                                                                                        true classes

                                                                                                                                                            0.6
               40                                                  40                                                  40

               60                                                  60                                                  60                                   0.4

               80                                                  80                                                  80                                   0.2

               100                                                 100                                                 100                                  0.0
                     0    20   40   60   80   100                        0   20   40   60    80   100                        0   20   40    60   80   100
                          predicted classes                                   predicted classes                                   predicted classes
                            (a) finetuning                                       (b) iCaRL                                           (c) PASS                       Figure 9: Norms of the weight vectors in the fully con-
                                                                                                                                                                    nected (FC) layer after learning all classes incrementally.
       Figure 8: The comparison of confusion matrix of finetuning,                                                                                                  Our method can remove the bias and learn a balance weight.
       iCaRL and PASS.

                                                                                                                                                                        By employing SSL in CIL, the model could learn more
       periment on CIFAR-100 (5 phases), after the last step, we
                                                                                                                                                                    general and transferable features for other tasks (as demon-
       calculate the norms of the weight vectors and plot them in
                                                                                                                                                                    strated in Section 3.5.2), which can reduce the feature extrac-
       Fig. 9. As shown in Fig. 9(a), by finetuning, the norms of
                                                                                                                                                                    tor bais. Thus, it would be easier to find a model to perform
       the weight vectors of new classes are much larger than those
                                                                                                                                                                    well on all tasks, which improves both the stability and plas-
       of old classes. As a result, an input image can be easily
                                                                                                                                                                    ticity of the model. Therefore, we emphasize that the feature
       predicted to a new class. Moreover, the weight learned by
                                                                                                                                                                    extractor bias should be considered and more future effort
       iCaRL suffers less imbalance problem comparing with fine-
                                                                                                                                                                    should be put into task-agnostic representation learning for
       tuning, but the bias still exists in Fig. 9(b). It can be seen
                                                                                                                                                                    IL, especially for non-exemplar based CIL.
       from Fig. 9(c) that our method is capable to remove the bias
       of the weight vectors in the FC Layer.
                                                                                                                                                                    5. Conclusion
       4.2. Ablation Study                                                                                                                                             This paper proposes a simple and effective method of
           The proposed PASS is comprised of three components:                                                                                                      PASS for CIL. PASS is capable to alleviate the catastrophic
       protoAug, SSL, and KD, as shown in Fig. 2. Here we ana-                                                                                                      forgetting problem in CIL, and achieves significantly bet-
       lyze the effect of isolate individual aspects of the methods.                                                                                                ter classification results on several datasets without stor-
       From the results in Table 3, we can observe that: (1) Only                                                                                                   ing exemplar samples for old class or using complex gen-
       using KD (as that in LwF) is completely failed in CIL with-                                                                                                  erative models. In particular, we propose to introduce
       out protoAug and SSL. (2) SSL has a relatively small effect                                                                                                  self-supervised learning to incremental learning for better
       combining with KD since the imbalance problem of the                                                                                                         task generalizable features. Extensive experiments demon-
       classifier is severe. (3) ProtoAug successfully mitigates the                                                                                                strate that our approach outperforms non-exemplar based
       imbalance problem and achieves much better results than                                                                                                      methods by large margins, and achieves comparable perfor-
       KD, e.g., protoAug improves the performance of KD with                                                                                                       mance compared to several state-of-the-art exemplar-based
       a margin of 32.94% on CIFAR-100 (20 phases). (4) The                                                                                                         approaches under different settings.
       performance of protoAug could be significantly improved by
       combining with SSL, e.g., SSL improves the performance                                                                                                       Acknowledgements
       of KD+protoAug with a margin of 9.87% on CIFAR-100
                                                                                                                                                                       This work has been supported by the Major Project for
       (20 phases). Moreover, it can be seen that the effectiveness                                                                                                 New Generation of AI under Grant No. 2018AAA0100400,
       of SSL is more obvious with the help of protoAug, which                                                                                                      the National Natural Science Foundation of China (NSFC)
       indicates that SSL and protoAug could benefit from each                                                                                                      grants U20A20223, 61633021, 62076236, 61721004, the
       other. Particularly, we have experimentally observed that the                                                                                                Key Research Program of Frontier Sciences of CAS under
       performance will drop significantly without KD. As demon-                                                                                                    Grant ZDBS-LY-7004, and the Youth Innovation Promotion
       strated in Section 3.4, KD is critical for the success of PASS.                                                                                              Association of CAS under Grant 2019141.

                                                                                                                                                                  5878

---

References                                                               [18] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross
                                                                              Girshick. Momentum contrast for unsupervised visual repre-
 [1] Rahaf Aljundi. Continual learning in neural networks. arXiv              sentation learning. In CVPR, pages 9729–9738, 2020.
     preprint arXiv:1910.02718, 2019.
                                                                         [19] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
 [2] Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny,                    Deep residual learning for image recognition. In CVPR, pages
     Marcus Rohrbach, and Tinne Tuytelaars. Memory aware                      770–778, 2016.
     synapses: Learning what (not) to forget. In ECCV, pages
                                                                         [20] Dan Hendrycks, Mantas Mazeika, Saurav Kadavath, and
     139–154, 2018.
                                                                              Dawn Song.        Using self-supervised learning can im-
 [3] Martín Arjovsky, Soumith Chintala, and L. Bottou. Wasser-
                                                                              prove model robustness and uncertainty. arXiv preprint
     stein generative adversarial networks. In ICML, pages 214–
                                                                              arXiv:1906.12340, 2019.
     223, 2017.
                                                                         [21] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distill-
 [4] Mathilde Caron, Piotr Bojanowski, Armand Joulin, and
                                                                              ing the knowledge in a neural network. arXiv preprint
     Matthijs Douze. Deep clustering for unsupervised learning of
                                                                              arXiv:1503.02531, 2015.
     visual features. In ECCV, pages 139–156, 2018.
                                                                         [22] Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, and
 [5] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal,
                                                                              D. Lin. Learning a unified classifier incrementally via rebal-
     Piotr Bojanowski, and Armand Joulin. Unsupervised learn-
                                                                              ancing. In CVPR, pages 831–839, 2019.
     ing of visual features by contrasting cluster assignments. In
                                                                         [23] Yen-Chang Hsu, Yen-Cheng Liu, Anita Ramasamy, and Zsolt
     NeurIPS, 2020.
                                                                              Kira. Re-evaluating continual learning scenarios: A cat-
 [6] Francisco M Castro, Manuel J Marín-Jiménez, Nicolás Guil,
                                                                              egorization and case for strong baselines. arXiv preprint
     Cordelia Schmid, and Karteek Alahari. End-to-end incremen-
                                                                              arXiv:1810.12488, 2018.
     tal learning. In ECCV, pages 233–248, 2018.
                                                                         [24] Longlong Jing and Yingli Tian. Self-supervised visual feature
 [7] Arslan Chaudhry, P. Dokania, Thalaiyasingam Ajanthan, and
                                                                              learning with deep neural networks: A survey. IEEE Trans.
     P. Torr. Riemannian walk for incremental learning: Under-
                                                                              Pattern Anal. Mach. Intell., 2020.
     standing forgetting and intransigence. In ECCV, pages 532–
     547, 2018.                                                          [25] Ronald Kemker and Christopher Kanan. Fearnet: Brain-
 [8] Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny,                     inspired model for incremental learning. In ICLR, 2018.
     Thalaiyasingam Ajanthan, Puneet K Dokania, Philip HS Torr,          [26] Diederik P Kingma and Jimmy Ba. Adam: A method for
     and Marc’Aurelio Ranzato. Continual learning with tiny                   stochastic optimization. In ICLR, 2015.
     episodic memories. ICML Workshop: Multi-Task and Life-              [27] Diederik P. Kingma and Max Welling. An introduction to
     long Reinforcement Learning, 2019.                                       variational autoencoders. Found. Trends Mach. Learn., pages
 [9] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geof-                  307–392, 2019.
     frey Hinton. A simple framework for contrastive learning of         [28] J. Kirkpatrick, Razvan Pascanu, Neil C. Rabinowitz, J. Veness,
     visual representations. In ICML, 2020.                                   G. Desjardins, Andrei A. Rusu, K. Milan, John Quan, Tiago
[10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li            Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis,
     Fei-Fei. Imagenet: A large-scale hierarchical image database.            C. Clopath, D. Kumaran, and Raia Hadsell. Overcoming
     In CVPR, pages 248–255, 2009.                                            catastrophic forgetting in neural networks. Proceedings of the
[11] Prithviraj Dhar, Rajat Vikram Singh, Kuan-Chuan Peng,                    National Academy of Sciences, pages 3521 – 3526, 2017.
     Ziyan Wu, and Rama Chellappa. Learning without mem-                 [29] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple
     orizing. In CVPR, pages 5138–5146, 2019.                                 layers of features from tiny images. Technical report, 2009.
[12] Arthur Douillard, Matthieu Cord, Charles Ollion, Thomas             [30] Gustav Larsson, Michael Maire, and Gregory Shakhnarovich.
     Robert, and Eduardo Valle. Podnet: Pooled outputs distilla-              Learning representations for automatic colorization. In ECCV,
     tion for small-tasks incremental learning. In ECCV, pages                pages 577–593, 2016.
     86–102, 2020.                                                       [31] Yann LeCun and Corinna Cortes. The mnist database of
[13] Robert M French. Interactive tandem networks and the se-                 handwritten digits. 2005.
     quential learning problem. Citeseer.                                [32] Hankook Lee, Sung Ju Hwang, and Jinwoo Shin. Self-
[14] Spyros Gidaris, Andrei Bursuc, Nikos Komodakis, Patrick                  supervised label augmentation via input transformations. In
     Pérez, and Matthieu Cord. Boosting few-shot visual learning              ICML, 2020.
     with self-supervision. In ICCV, pages 8059–8068, 2019.              [33] Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A
[15] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsu-                simple unified framework for detecting out-of-distribution
     pervised representation learning by predicting image rotations.          samples and adversarial attacks. In NeurIPS, pages 7167–
     In ICLR, 2018.                                                           7177, 2018.
[16] Ian J. Goodfellow, M. Mirza, Xia Da, Aaron C. Courville, and        [34] Zhizhong Li and Derek Hoiem. Learning without forgetting.
     Yoshua Bengio. An empirical investigation of catastrophic                IEEE Trans. Pattern Anal. Mach. Intell., pages 2935–2947,
     forgeting in gradient-based neural networks. CoRR, 2014.                 2018.
[17] Ian J. Goodfellow, Jean Pouget-Abadie, M. Mirza, Bing Xu,           [35] Jialun Liu, Yifan Sun, Chuchu Han, Zhaopeng Dou, and
     David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and               Wenhui Li. Deep representation learning on long-tailed data:
     Yoshua Bengio. Generative adversarial nets. In NeurIPS,                  A learnable embedding augmentation perspective. In CVPR,
     2014.                                                                    pages 2967–2976, 2020.

                                                                       5879

---

[36] Yu Liu, Sarah Parisot, Gregory G. Slabaugh, Xu Jia, Ales           [53] Lu Yu, Bartlomiej Twardowski, X. Liu, L. Herranz, Kai Wang,
     Leonardis, and Tinne Tuytelaars. More classifiers, less for-            Yong mei Cheng, Shangling Jui, and Joost van de Weijer.
     getting: A generic multi-classifier paradigm for incremental            Semantic drift compensation for class-incremental learning.
     learning. In ECCV, pages 699–716, 2020.                                 In CVPR, pages 6980–6989, 2020.
[37] M. McCloskey and N. J. Cohen. Catastrophic interference            [54] Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual
     in connectionist networks: The sequential learning prob-                learning through synaptic intelligence. In ICML, pages 3987–
     lem. Psychology of Learning and Motivation, pages 109–165,              3995, 2017.
     1989.                                                              [55] Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, and Lu-
[38] Thomas Mensink, J. Verbeek, F. Perronnin, and G. Csurka.                cas Beyer. S4l: Self-supervised semi-supervised learning. In
     Distance-based image classification: Generalizing to new                ICCV, pages 1476–1485, 2019.
     classes at near-zero cost. IEEE Trans. Pattern Anal. Mach.         [56] Bowen Zhao, Xi Xiao, Guojun Gan, Bin Zhang, and Shu-
     Intell., pages 2624–2637, 2013.                                         Tao Xia. Maintaining discrimination and fairness in class
[39] Thomas Mensink, Jakob J. Verbeek, Florent Perronnin, and                incremental learning. In CVPR, pages 13205–13214, 2020.
     Gabriela Csurka. Distance-based image classification: Gener-
     alizing to new classes at near-zero cost. IEEE Trans. Pattern
     Anal. Mach. Intell., pages 2624–2637, 2013.
[40] M. Noroozi and P. Favaro. Unsupervised learning of visual
     representations by solving jigsaw puzzles. In ECCV, pages
     69–84, 2016.
[41] Hadi Pouransari and Saman Ghili. Tiny imagenet visual recog-
     nition challenge. CS231N course, Stanford Univ., Stanford,
     CA, USA, 2015.
[42] Ameya Prabhu, Philip HS Torr, and Puneet K Dokania.
     Gdumb: A simple approach that questions our progress in
     continual learning. In ECCV, pages 524–540, 2020.
[43] Sylvestre-Alvise Rebuffi, A. Kolesnikov, Georg Sperl, and
     Christoph H. Lampert. icarl: Incremental classifier and repre-
     sentation learning. In CVPR, pages 5533–5542, 2017.
[44] Matthew Riemer, Ignacio Cases, Robert Ajemian, Miao Liu,
     Irina Rish, Yuhai Tu, and Gerald Tesauro. Learning to learn
     without forgetting by maximizing transfer and minimizing
     interference. In ICLR, 2018.
[45] Karsten Roth, Timo Milbich, Samarth Sinha, Prateek Gupta,
     Björn Ommer, and Joseph Paul Cohen. Revisiting train-
     ing strategies and generalization performance in deep metric
     learning. In ICML, 2020.
[46] Hanul Shin, Jung Kwon Lee, Jaehong Kim, and Jiwon Kim.
     Continual learning with deep generative replay. In NeurIPS,
     pages 2994–3003, 2017.
[47] P Shrestha et al. Incremental learning strategies with random
     forest classifiers. In WIC Symposium on Information Theory
     in the Benelux, pages 1–6, 2011.
[48] Gido M Van de Ven and Andreas S Tolias. Three scenarios for
     continual learning. arXiv preprint arXiv:1904.07734, 2019.
[49] Chenshen Wu, L. Herranz, X. Liu, Y. Wang, Joost van de
     Weijer, and B. Raducanu. Memory replay gans: Learning
     to generate new categories without forgetting. In NeurIPS,
     pages 5962–5972, 2018.
[50] Y. Wu, Yan-Jia Chen, Lijuan Wang, Yuancheng Ye, Zicheng
     Liu, Yandong Guo, and Yun Fu. Large scale incremental
     learning. In CVPR, pages 374–382, 2019.
[51] Ye Xiang, Ying Fu, Pan Ji, and Hua Huang. Incremental
     learning using conditional adversarial networks. In ICCV,
     pages 6618–6627, 2019.
[52] Yuzhe Yang and Zhi Xu. Rethinking the value of labels for
     improving class-imbalanced learning. In NeurIPS, 2020.

                                                                      5880

---
