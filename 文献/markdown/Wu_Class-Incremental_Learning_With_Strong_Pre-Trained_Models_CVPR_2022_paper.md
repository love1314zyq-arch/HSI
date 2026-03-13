# Wu_Class-Incremental_Learning_With_Strong_Pre-Trained_Models_CVPR_2022_paper

Source: `Wu_Class-Incremental_Learning_With_Strong_Pre-Trained_Models_CVPR_2022_paper.pdf`

Class-Incremental Learning with Strong Pre-trained Models

                     Tz-Ying Wu1,2 Gurumurthy Swaminathan1 Zhizhong Li1
            Avinash Ravichandran1 Nuno Vasconcelos2 Rahul Bhotika1 Stefano Soatto1
                             1                                     2
                               AWS AI Labs                           UC San Diego
      {gurumurs,lzhizhon,ravinash,bhotikar,soattos}@amazon.com                                    {tzw001,nuno}@ucsd.edu

                        Abstract                                                                                            Base Class
                                                                                                               Y𝒃           Novel Class
                                                                                                                            Overlapping Class
    Class-incremental learning (CIL) has been widely stud-         Old Data
ied under the setting of starting from a small number of
classes (base classes). Instead, we explore an understud-                                                                                    Y𝒏
ied real-world setting of CIL that starts with a strong model
                                                                                                                              New Data
pre-trained on a large number of base classes. We hypoth-              Large Pretrained Classifier w/ Label Set Y𝒃         w/ Label Set Y𝒏
esize that a strong base model can provide a good repre-
sentation for novel classes and incremental learning can be
done with small adaptations. We propose a 2-stage train-
ing scheme, i) feature augmentation – cloning part of the
backbone and fine-tuning it on the novel data, and ii) fusion                            Updated Classifier w/ Label Set Y𝒃 ∪ Y𝒏

– combining the base and novel classifiers into a unified
                                                                  Figure 1. We study the problem of CIL in the setting where there
classifier. Experiments show that the proposed method sig-
                                                                  are a large number of base classes. Classes between splits can
nificantly outperforms state-of-the-art CIL methods on the
                                                                  have overlaps, and data can be sampled from the same or different
large-scale ImageNet dataset (e.g. +10% overall accuracy          distributions (e.g. different styles, poses).
than the best). We also propose and analyze understudied
practical CIL scenarios, such as base-novel overlap with
distribution shift. Our proposed method is robust and gen-
eralizes to all analyzed CIL settings.                            tion. For example, a strong model could have been devel-
                                                                  oped to identify different dog breeds, and a small set of ad-
                                                                  ditional breeds needs to be added for model update. More-
                                                                  over, base and novel classes may overlap but may distribute
1. Introduction                                                   differently, such as a guitar class present in both base and
   As deep classifiers become more popular for real-world         novel classes, with only acoustic guitars in base samples
applications, the need for incrementally learning novel           while electric guitars in novel ones.
classes (novel data) becomes more prevalent. Training                 Some CIL methods use a static model and typically
a classifier with both old and novel data is not optimal          fine-tune the existing parameters with some constraints im-
when old data can become unavailable over time [6, 13,            posed on parameter changes [3, 13, 32], gradients [17], fea-
15, 18, 19, 23, 24, 32]. Fewer old data leads to a high im-       tures [8, 11], or activations [6, 15, 23]. These methods mod-
balance between the old and novel data, and simply fine-          ify the well-trained network weights and risk performance
tuning the model causes catastrophic forgetting for the old       degradation. On the other hand, methods based on dynamic
classes [15].                                                     models learn separate parameters for novel tasks, either by
   Class-incremental learning (CIL) methods [5,6,8,11,23,         expanding the model [24], or introducing a parameter gat-
29,34] learn to categorize more and more classes over time.       ing function [18,19]. However, almost all dynamic methods
However, they typically start their incremental training with     focus on task-incremental learning (TIL). TIL assumes that
a small number of base classes (e.g. only 50), and add an         which task a sample belongs to is known at inference time,
equally small number of new classes at a time. In many            and different tasks are inferred individually. This assump-
practical scenarios, having a large number of base classes        tion is not realistic if the application needs to distinguish
can be a more useful starting point for building an applica-      between base and novel classes. A recent work DER [30]

                                                                9601

---

uses a dynamic model for CIL, where it duplicates the entire         dynamic model architecture. For the first group, the gen-
backbone for novel data and prunes the model.                        eral approach for incremental training is to impose con-
   Current CIL approaches may not be optimal when a large            straints on parameter changes [3, 13, 32], gradients [17],
number of base classes (e.g. 800) is used to pre-train a             features [8,11,25], or activations [2,6,10,15,23] when fine-
strong model. We hypothesize that the well-trained back-             tuning the network with novel samples. They can be either
bone is capable of extracting representative features for the        memory-free [3,13,15,32] which only rely on the novel data
novel data and freezing it partially while learning a small          and the original point of convergence in parameters, or be
adaptation branch for novel data works better than fine-             memory-based [2, 6, 8, 11, 13, 17, 23, 29, 34] which relaxes
tuning the whole backbone. We show in a preliminary study            the constraint and keep exemplars of old data for replay.
that fine-tuning fewer layer blocks outperforms full fine-           The exemplars can be selected in many ways, such as herd-
tuning when using a strong pre-trained model.                        ing [23], random sampling [10], uncertainty sampling [5].
   Hence, we propose a 2-stage training scheme for CIL                  The second line of work adopt dynamic models, where
starting with a large number of base classes: i) duplicat-           task-specific parameters are introduced to prevent interfer-
ing part of the backbone as the adaptation module and fine-          ence among tasks, which can be achieved by growing the
tuning it on the novel data, and ii) combining all the inde-         model [14,16,21,22,24,26,30,31,33], or introducing a gat-
pendently trained base and novel classifiers into a unified          ing function on the parameters [1, 18, 19]. However, most
classifier at each incremental step. Towards this, we pro-           works in this line either focus only on the performance of
pose a score fusion network that enables knowledge trans-            the target task [16, 21, 22, 26], or assume task label is avail-
fer between base and novel classes by combining the logits.          able at inference time [12, 14, 18, 19, 24, 31, 33]. The excep-
While the optimal adaptation module size may depend on               tions infer the task identity typically by learning a routing
the old and novel data discrepancy, we show that our score           classifier [1, 4], or estimating confidence scores [27]. How-
fusion generalizes to different adaptation module sizes.             ever, they do not produce a unified head for all the classes.
   Most CIL research [6, 8, 11, 23, 29, 34] only consider the           Our proposal is a two-stage method in the offline setting
scenario where the base and novel label sets are disjoint.           with a dynamic model. In stage-I, we expand the network
Rainbow Memory [5] (RM) is the only exception where                  by cloning and fine-tuning partial parameters from the orig-
label sets are identical among tasks but class frequencies           inal backbone. While DER [30] shares some commonalities
differ. In this work, we explore a more general setting              with our first stage, we differ in stage-II. We show the down-
where some novel classes can overlap with base classes, po-          side of retraining a linear classifier for all classes solely with
tentially with a different distribution (e.g., different styles,     the data in the memory as in [5, 30], and propose to com-
poses), as shown in Figure 1. We show how our score fu-              bine the logits of the old and new expert classifiers. We also
sion can handle overlapping classes by using a knowledge             investigate consolidating knowledge of overlapping classes,
pooler to combine their base and novel logits.                       which is understudied in the literature.
   In summary, we provide three contributions:                          Our method bears superficial similarity to transfer learn-
                                                                     ing and few-shot learning in that it uses a large number of
   • We propose a 2-stage CIL training strategy. In stage-I,         base classes. However, they focus only on the performance
     instead of tuning the base network and risking catas-           of novel classes. Such methods underperform on overall
     trophic forgetting, we replicate part of the network and        average performance in our experiments.
     fine-tune the extra branch on novel data. We show that
     as we start with a strong pre-trained base network, this        3. CIL with strong pre-trained models
     approach outperforms state-of-the-art CIL methods.
                                                                        In this section, we present the problem formulation and
   • We propose a new score fusion algorithm for stage-II,           the details of the proposed method.
     where we unify the classifiers for base and novel data
     into one by consolidating their output logits.                  3.1. Problem formulation
   • We generalize CIL to a broader and more challenging                Given a dataset D = {(xi , yi )}Ni=1 , where xi and yi are
     scenario: base and novel classes can partially overlap,         data and label respectively, the goal of a traditional classifi-
     and the overlap classes may have changed distributions          cation network is to learn a feature extractor h(·; Φ) ∈ Rk
     (e.g. new style). We show that our method is robust and         and a linear classifier W ∈ Rk×|Y| , where Y is the label
     generalizes to this new scenario.                               set of D. This is usually obtained by minimizing the cross-
                                                                     entropy loss,
2. Related work
                                                                                                    N
                                                                                                1 X
   Prior work in continual learning can be mainly catego-                            Lce = −          log p̂(yi ) (xi )            (1)
                                                                                                N i=1
rized into two streams: methods using a (1) static and (2)

                                                                   9602

---

where                                                               our method to leverage the powerful pre-trained model. We
                                                                    hypothesize that drastic changes in the existing network is
                  p̂(x) = σ(WT h(x; Φ)),                   (2)      unnecessary, and modifying only the top few layers will suf-
                                                                    fice. Before we introduce our method, we validate this hy-
σ(·) is the softmax function, and v(l) is the lth element of v.     pothesis and our strategy of minimal network changes with
    In class-incremental learning (CIL), we first focus on one      two motivational experiments.
incremental step, and generalize into multiple steps in Sec-            We first analyze when there are many base classes, how
tion 3.3. Given a base model Mb pre-trained on a label set          much (or little) benefit is in optimizing representations with
Yb using the base dataset Db , we augment Yb with another           the novel data. Starting from a pre-trained ResNet10 [9] as
label set Yn using dataset Dn , i.e., the new all-labels set        Mb , we train a classifier for Yn only using new data Dn ,
Ya = Yb ∪ Yn . Most prior works [6, 8, 11, 23, 29, 34] focus        with or without fine-tuning h(·; Φ). As shown in Figure 2
on the CIL scenario where the label sets are fully disjoint,        (left), the gap in novel class accuracy between fine-tuning
i.e. Yb ∩ Yn = ∅, while in practice, classes can overlap be-        and fixing the feature extractor is large when the number
tween base and novel splits [5], where |Ya | < |Yb | + |Yn |.       of base classes is small. The gap significantly reduces with
A full-overlapping extreme case is Yn ⊆ Yb , which means            increasing number of base classes, but does not disappear
the new label set Ya = Yb . Despite the overlap, base and           even with |Yb | = 800. This indicates that fully changing
novel samples can be very different in each of the overlap-         the representations for the novel data is still beneficial for
ping classes, such as having different poses or styles.             learning Dn , but with greatly reduced significance.
    An intuitive solution to learn the combined label set is            We then explore whether we need to optimize the en-
to use Db ∪ Dn to train a standard classifier with eq. (1).         tire pre-trained feature extractor using novel data. Fig-
However, as motivated in the CIL literature, Db often be-           ure 2 (right) presents the novel class accuracy of fine-
comes unavailable over time in practice. In this case, Dn           tuning smaller subsets of the four convolutional blocks in
dominates the training set and causes catastrophic forget-          ResNet10. With weaker pre-trained models (e.g. |Yb | =
ting, degrading base class performance significantly.               40, 100, 200), almost all the layers in the backbone need to
    Traditional CIL methods divide existing datasets’ classes       be fine-tuned to gain good performance in novel classes.
evenly into multiple splits, each with a small number of            However, for strong pre-trained models (i.e., |Yb | = 800),
classes (e.g. 50). Typically, the initial model is briefly          fine-tuning anything beyond the last convolutional block
trained only on one of the small splits. Instead, we explore        (i.e., layer4) degrades the strong performance.
a real-world scenario that is understudied in the literature,
where the pre-trained model has been sufficiently trained           3.3. Training pipeline
on a large number of classes (e.g. 800 classes). In following           Inspired by these observations, we propose a 2-stage in-
incremental steps |Yb |
 |Yn |, so the smaller novel splits        cremental training pipeline (Figure 3). We first formulate
bring comparably little additional information.                     using one incremental step (base+novel):
                                                                    Stage-I – Feature augmentation (FA): We split the pre-
3.2. Can minimal network modification work well?
                                                                    trained feature extractor h(·; Φ) into two sub-networks,
    If we freeze the pre-trained network, we avoid catas-           where the encoder with parameter Φs is followed by the en-
trophic forgetting. However, this representation might not          coder with parameter Φb , Φ = {Φs , Φb }. To optimize the
generalize to the novel classes, especially in traditional          features for Dn without forgetting the ones for Db , we ex-
CIL where the first-step model is pre-trained with a small          pand the features by cloning the sub-network of Φb to the
number of classes. To mitigate this issue, most prior               branch Φn as the adaptation module, and fine-tuning Φn
CIL work [6, 8, 11, 23, 29, 34] update the feature extractor        and the weights of the novel class classifier Wn ∈ Rk×|Yn |
with novel data. Regularization in the loss function (such          on Dn with the loss of eq. (1). The shared Φs is frozen. This
as penalizing changes in predictions [15, 23] or network            setup ensures no forgetting in the old representation, while
weights [13,32], or re-training on saved examplars [29,34])         enabling feature learning to accommodate new knowledge.
can reduce catastrophic forgetting to a certain extent, but it      While the optimal size of the frozen Φs depends on the data
still remains prevalent and unsolved.                               discrepancy between base and novel splits, which itself can
    In this paper, we propose to add extra capacity to              be a research topic to explore, we adopt the last convolu-
the network. We differ from prior network-growing ap-               tional block (i.e. layer4) as Φb as it suffices for most cases,
proaches [24, 26] in two ways. First, our goal is CIL               and Φs is the layer1-3 blocks.
rather than TIL, i.e., the network must avoid confusion be-         Stage-II – Fusion: After the first-stage training, we have
tween novel and base classes. This is not solved by merely          the base and novel classifiers, Mb (·; Φs , Φb , Wb ) and
freezing existing network weights, which by construct only          Mn (·; Φs , Φn , Wn ), optimized for Yb and Yn respec-
avoids forgetting in classifying among old classes. More            tively. We introduce in Section 4 our score fusion scheme
importantly, CIL with a large number of base classes allows         to combine the knowledge of the two networks and get a

                                                                  9603

---

                                                                                                         90
                       85
                                                                                                                  finetuning layer1-4 (all)
                                                                                                                  finetuning layer2-4
                                                                                                                                                                         h(x; Φs , Φd ), is frozen to preserve the information, and the
                                                                                                         88       finetuning layer3-4
                                                                                                                                                                         linear weights Wr ∈ R2k×2 are the only parameters for
novel class accuracy

                                                                                  novel class accuracy
                       80                                                                                         finetuning layer4
                                                                                                         86
                       75                                                                                                                                                learning the routing classifier.
                                                                                                         84
                       70                                                                                                                                                   The routing loss is a binary cross-entropy loss, i.e.
                       65                              fixed backbone+linear                             82
                                                       finetuning
                       60
                            100   200     300    400    500   600    700    800
                                                                                                         80
                                                                                                                  40             100                 200   800
                                                                                                                                                                              lrt (x, r) = −(1 − r) log r̂(0) (x) − r log r̂(1) (x) ,   (4)
                                        number of base classes                                                              number of base classes

                                                                                                                                                                         where r = 1[x∈Dn ] is the split label for x. However, there is
Figure 2. Left: The comparison between fine-tuning and freezing
                                                                                                                                                                         a large base-novel sample imbalance due to |Dn |
 |E|. To
representations. Right: The novel class accuracy of fine-tuning
                                                                                                                                                                         address this, we re-balance the class losses by re-weighting:
each pre-trained model with different numbers of layers.
                                                                                                                                                                                        1 X                    1    X
                                   Pretraining                                    Stage I – Clone & Finetune on novel                                                     Lrt−bal =         lrt (xi , ri ) +          lrt (xi , ri ).
                                                                                                                                                                                       2|E|                  2|Dn |
                                                                                         𝒙                    𝝓!           𝝓#                                                               xi ∈E                            xi ∈Dn
                                    𝒙           𝝓!        𝝓#
                                                                                                                                                 𝑦"$&                                                                                   (5)
                                                                                                                          𝝓$&
                                                                           𝑦"#
                                                                                                                                                                            While routing is a way to get the prediction for all classes
                                   Stage II - Fusion
                                                               𝝓#
                                                                                                                                                                         in Ya , these baselines cannot produce a unified probability
                                                                                                                                          𝑏𝑎𝑠𝑒
                                                                                                                                                                         distribution for all classes. Also, the routing function’s pre-
                                        𝒙        𝝓!            𝝓$%                                            ?                           𝑛𝑜𝑣𝑒𝑙                          diction error will propagate to the final class prediction.
                                                                    …

                                                                                                                                     …

                                                                                                                                          𝑠𝑡𝑒𝑝𝑠
                                                               𝝓$&                                                                        1. . . 𝑡

                                                                                                                                    𝑦""                                  4.2. General score fusion network
                                                Figure 3. General training pipeline.                                                                                         Our proposed score fusion network (Figure 4c) inte-
                                                                                                                                                                         grates the knowledge and output of all branches and gener-
                                                                                                                                                                         ates a unified probability distribution. Modifying the exist-
unified classifier of the label set Ya , where Ya = Yb ∪ Yn .
                                                                                                                                                                         ing well-trained classifiers with few samples leads to over-
Multiple incremental steps: We maintain one extra branch
                                                                                                                                                                         fitting while being susceptible to class imbalance in the lim-
per new step: {Φb , Wb , Φn1 , Wn1 , . . . , ΦnT , WnT }. In
                                                                                                                                                                         ited data, but freezing everything rules out knowledge trans-
each novel step t ∈ {1..T }, we initialize the new branch
                                                                                                                                                                         fer opportunities between branches. In addition, predictions
Φnt with Φb , fine-tune it, and re-fuse all t + 1 branches.
                                                                                                                                                                         regarding overlapping classes need unified scores from base
                                                                                                                                                                         and novel classifiers. We introduce knowledge-preserving
4. Fusion: unifying base and novel classifiers
                                                                                                                                                                         transfer, overlap knowledge integration schemes, and bal-
4.1. Baseline using routing                                                                                                                                              anced optimization to address these issues.
                                                                                                                                                                         Knowledge-preserving transfer. After the stage-I train-
    We analyze fusion baselines with only one incremental
                                                                                                                                                                         ing, we obtain expert models for each step {Φd , Wd } (for
step, although they can naturally extend to additional steps.
                                                                                                                                                                         d ∈ {b, n1, . . . , nt}), where the probability of the individ-
We first explore two intuitive baselines that rely on each
                                                                                                                                                                         ual classifier is computed by applying the softmax function
classifier’s prediction and a routing function r̂(·) ∈ {0, 1}
                                                                                                                                                                         on the logit score zd = WdT hd , Wd ∈ Rk×|Yd | . To pre-
to decide whether a sample is from Db or Dn . The final
                                                                                                                                                                         vent overfitting to the small dataset of E ∪ Dnt , we pro-
prediction ŷ is then assigned by ŷb if r̂(x) = 0 and ŷn if
                                                                                                                                                                         pose to also freeze all classifier weights (i.e. Wb , Wnτ ,
r̂(x) = 1, where ŷd = arg maxl p̂(l) (x; Φs , Φd , Wd ), for
                                                                                                                                                                         in addition to Φs , Φb , Φnτ , τ ∈ [1, t]) to preserve their
d ∈ {b, n}. We explore two routing functions:
                                                                                                                                                                         capability of distinguishing classes within the same split.
Confidence-based routing (Fig. 4a) uses the confidence
                                                                                                                                                                         To additionally enable the knowledge transfer between the
score of individual classifiers as the proxy where the routing
                                                                                                                                                                         two splits, we use E ∪ Dnt to learn Wdd0 ∈ Rk×|Yd | ,
function is defined as r̂(x) = 1[Confb <Confn ] , where 1[·] is
                                                                                                                                                                         d, d0 ∈ {b, n1, . . . , nt}, d 6= d0 (randomly initialized), con-
the indicator function, and the confidence score Confd =
                                                                                                                                                                         necting d0 branch’s features to d branch’s logits, which are
maxl p̂(l) (x; Φs , Φd , Wd ), for d ∈ {b, n}.
                                                                                                                                                                         used to learn the delta logits for knowledge transfer, to be
Learning-based routing (Fig. 4b) directly learns a rout-
                                                                                                                                                                         added to the final logits for the d branch:
ing classifier with available data. Following prior work in
CIL [23], we keep few-shot exemplars E from all past in-                                                                                                                                                 0
                                                                                                                                                                                                        dX 6=d
cremental steps in the memory to supplement the novel data                                                                                                                                                             T
                                                                                                                                                                                          4zd =                       Wdd 0 hd0 ,       (6)
Dn . The routing classifier is formulated as                                                                                                                                                        d0 =b,n1,...,nt

                                                     r̂(x) = σ(WrT (hb ⊕ hn )) ,                                                                                 (3)                        z̃d = zd + 4zd .                            (7)

where ⊕ denotes vector concatenation. For d ∈ {b, n},                                                                                                                    Overlap knowledge integration. To get a unified classi-
the feature from each feature extractor branch, hd =                                                                                                                     fier with probability distribution for all the classes, we can

                                                                                                                                                                       9604

---

                                                                                                                    𝒉!    Knowledge transferer   𝒛$ !              Knowledge pooler
                                                                                                                                                              𝒛%                      𝒛$ %
                                                                                                             𝝓!             𝐖!      𝒛!
                          𝑦ො𝑏                       concatenation
                                                                         𝑦ො𝑏                                                                                            pool
                                                                                                                                                         𝒛%
                   𝒉𝑏
                                                                    𝒉𝑏                                                      𝐖!"#
                                                                                                                                                                        pool
              𝝓𝑏                𝐶𝑜𝑛𝑓𝑏                                                                                       𝐖!"$    Δ𝒛!                                 pool
                                                               𝝓𝑏
                                                                                                                   𝒉"#
   𝒙    𝝓𝑠                          base/novel? 𝒙       𝝓𝑠                                                                                       𝒛$ "#
                                                                                   base/novel?   𝒙    𝝓$
                                                                                                                            𝐖"#     𝒛"#
              𝝓𝑛                𝐶𝑜𝑛𝑓𝑛                          𝝓𝑛                                            𝝓"#

                                                                                                                                                         …
                                                                                                                            𝐖"#!                               pool   pooling layer
                          𝑦ො𝑛                                                                                                       Δ𝒛"#
                   𝒉𝑛                                               𝒉𝑛                                                     𝐖"#"$                                      addition
                                                                         𝑦ො𝑛

                                                                                                                   …

                                                                                                                                                 …
                                                                                                                   𝒉"&
  (a) Confidence-based routing base- (b) Learning-based routing base-                                                                            𝒛$ "&                concatenation
                                                                                                                            𝐖"&     𝒛"&                               base class
  line.                                                                                                      𝝓"&
                                               line.                                                                        𝐖"$!                                      novel class
                                                                                                                           𝐖"$"#                                      overlapping class
                                                                                                                                    Δ𝒛"&
                                                                                                                       (c) General score fusion network.

                         Figure 4. Fusion methods. (a-b) Routing as a baseline. (c) Our proposed network for score fusion.

combine the logit scores of the base and novel branches                                                where Wr,aux ∈ R(t+1)×(t+1) is the linear routing classi-
by concatenation, i.e. za = z̃b ⊕ z̃n1 ⊕ · · · ⊕ z̃nt where                                            fier’s weights. The full loss then becomes
z̃d ∈ R|Yd | , d ∈ {b, n1, . . . , nt}, and obtain the posterior
                                                                                                                   Ltotal = (1 − α) · Lcls + α · Lrt−bal ,                                   (10)
probability with σ(za ). However, when overlapping classes
exist (Yd ∩ Yd0 6= ∅, ∃d, d0 ), they appear inPz̃a multiple                                            with Lcls from eq. (8), Lrt−bal from eq. (5) using r̂ from
times. We apply a knowledge pooler to za ∈ R d |Yd | to get                                            eq. (9), and α is a loss weight hyperparameter.
the final logit z̃a ∈ R|Ya | , as illustrated in Figure 4c (right),                                        Second, during training we normalize and scale hd0 for
that either max-pools or average-pools the multiple logit                                              base samples by a factor of β ∈ [0, 1] before feeding it into
scores for each overlapping class. Note that z̃a = za when                                             Wbd0 , d0 6= b. This prevents the training from drastically
Yd ∩ Yd0 = ∅, ∀d, d0 . From our experiments, max pooling                                               influencing base classes, but limits the knowledge transfer
performs better than average pooling, since the branches do                                            from novel branch features to base classes.
not always simultaneously output high responses to a sam-                                                  Like all imbalanced learning problems, there is a trade-
ple on its class, especially when the data of the same class                                           off between the performance of base and novel classes. Us-
are very different in base and novel splits.                                                           ing our two regularization mechanisms provides the flexi-
Balanced optimization. With the final logit score z̃a,i for                                            bility to optimize for the metric that matters more for each
each sample xi ∈ E ∪ Dnt , the parameters Wdd0 can be                                                  customized application. The behavior of these mechanisms
optimized with the loss of eq. (1). However, since |Dnt |
                                            will be discussed further in the experiment section.
|E|, the training will be dominated by the novel classes. To
balance the probability estimation, we sample a subset B ⊂                                             5. Experiments
E ∪ Dnt uniformly over all the classes, where each class in                                               In this section, we compare our results with baselines
B has equal number of samples. With this class-balanced                                                and state-of-the-art CIL methods. We also show ablation
sampling, the classification loss becomes                                                              studies on the effect of different components and analyze
                        |B|                                                                            our results in different practical settings. Code will be re-
               1 X                                                                                     leased on acceptance.
       Lcls =         − log p̂(yi ) (xi ) p̂(x) = σ(z̃a ).                               (8)
              |B| i=1
                                                                                                       5.1. Experimental setup
However, in addition to the sample imbalance, our number                                               Dataset. To study the behavior of strong pre-trained mod-
of classes is also highly imbalanced. Since |Yb |
 |Ynτ |,                                            els, we create several data splits from ImageNet [7] to sim-
base logits have more chances of being the largest than                                                ulate different practical scenarios. For disjoint-CIL with
novel logits, so eq. (8) will favor one of the base classes,                                           disjoint base and novel classes, we first perform our main
which may or may not be desired depending on the appli-                                                analysis with one incremental step. Unless otherwise noted,
cation. To control base and novel logit balance, we explore                                            we randomly select 800 base classes and 40 novel classes.
two regularization mechanisms.                                                                         Variations are tested in ablation studies. We then test on
    First, we explicitly train to balance the largest base logit                                       an existing 10-step asymmetric split scheme with 500 base
and the largest novel logit scores, by adding a routing aux-                                           classes and ten steps of 50 classes each [8], originally de-
iliary loss over the maximum score from each split. The                                                signed to test short but numerous incremental steps.
routing classifier can be defined as                                                                      For overlapping-CIL, where some or all novel classes
           T                            (l)                  (l)                       (l)             overlap with base classes, we experiment with three differ-
r̂(x) = σ(Wr,aux (max z̃b ⊕ max z̃n1 ⊕ · · · ⊕ max z̃nt )) ,                                           ent base-novel class splits (one incremental step). 1) ran-
                                l                   l                          l
                                                                                         (9)           dom class split: 800 base classes, 40 novel classes with

                                                                                                     9605

---

5 overlapping classes among them. 2) domain-changing                                             65

split: ImageNet classes are grouped into two categories,

                                                                                    Acc (base)
                                                                                                 60    score fusion, alpha=0.0
animate and inanimate. Inanimate and animate classes are                                               score fusion, alpha=0.4
                                                                                                       score fusion, alpha=1.0
                                                                                                 55
                                                                                                       learning-based routing
taken as base and novel classes respectively, with 5 overlap-                                          joint training
                                                                                                 50    beta=1
ping classes that are randomly selected. 3) style-changing                                            55     60      65          70   75   80   85
                                                                                                                       Acc (novel)
split: To simulate this scenario with currently available an-
notations of ImageNet, we select five pairs of classes that          Figure 5. Design space. Score fusion performs comparably to
are semantically similar to each other, merge each pair into         joint training and outperforms learning based routing. Optimal
a single class, and use them as five overlapping classes.            α and β values depend on the desired balance between base and
For example, “pembroke” and “cardigan” can be merged                 novel performance.
into “corgi”, and “electric guitar” and “acoustic guitar” can
be merged into “guitar”. Non-overlapping novel and base
classes are chosen randomly.                                         For fairness, all compared methods use the same memory
    For overlapping classes in (1) and (2), samples need to          constraint and dataset sizes. Dataset subsampling are also
be distributed between base and novel splits. We explore             identical except where results are from other papers.
splitting each class either randomly or by unsupervised
clustering using K-Means (K = 2) on features extracted               5.2. Disjoint-CIL
from a full-ImageNet trained ResNet10 penultimate layer.                 We first study the most common scenario in the litera-
After splitting all 1000 classes into two splits, we use one         ture, disjoint-CIL, where the set of base and novel classes
split for base classes and the other split for novel classes.        are disjoint. In the following, we present the comparison to
That is, the 800 base classes are trained on roughly half of         the baselines, along with extensive ablations and analysis of
the 800 classes’ data.                                               the proposed method.
Metrics. In prior work, the number of base and novel                 Design choices. We first study the balancing behavior of
classes are usually balanced (e.g. |Yb | = |Yn | = 50), so           the two regularization methods in score fusion, routing loss
it is natural to simply evaluate using the accuracy over all         and feature scaling. For demonstration, we show a hyper-
the classes, i.e. Accall = A(D
                           P test ), where Dtest is the test-        parameter sweep in Figure 5 with α ∈ {0, 0.4, 1}, β ∈
                             (x,y)∈S 1[y=arg maxk p̂(k) (x)] ei-
                         1
ing set and A(S) = |S|                                               {0, 0.2, 0.4, 0.6, 0.8, 1}. Actual hyperparameters are se-
ther at t = T or averaged over all incremental steps (iden-          lected on validation data instead. As we increase α (routing
tical to incremental accuracy). We follow the version im-            loss weight) and decrease β (scaling factor), our novel class
plemented in [8] that includes the base step t = 0 to facil-         performance is boosted significantly, while the drop in base
itate comparison. However, in our setting of strong pre-             class accuracy is comparably minor. The effects of α and
trained models, |Yb |
 |Yn |, e.g. 800 and 40. In this              β are similar, but changing both gives us the flexibility to
case, the overall accuracy will be dominated by the base             reach the full range of base-novel balance operating points.
class performance. A model with high overall accuracy                    The optimal hyperparameter selection depends on the
is not guaranteed to perform well for the novel classes.             specific design choices and desired metric, such as bal-
Hence, in addition to the overall accuracy, we also present          ancing between base and novel (best Accavg ) or prioritiz-
the accuracy in each split. When there is overlap, we                ing novel classes (best Accnovel ). In the following exper-
use Accbase = A({(x, y) : (x, y) ∈ Dtest , y ∈ Yb \                  iments, we will present the score fusion results optimizing
(Yb ∩ Yn )}), Accnovel = A({(x, y) : (x, y) ∈ Dtest , y ∈            for these practical scenarios: best-Accall , best-Accavg and
Yn \ (Yb ∩ Yn )}) and Accovlp = A({(x, y) : (x, y) ∈                 best-balanced (optimize
                                                                                                Accall +Accavg
                                                                                                               ).
                                                                                                       2
Dtest , y ∈ Yb ∩ Yn }). Note that the classification is still        Comparisons to CIL methods. In Table 1 (one novel step)
predicted among all classes. An aggregate metric to bal-             and Table 2 (multiple novel steps), we compare to state-
ance amongPthe per split accuracy is the average of them:            of-the-art class-incremental learning methods and the joint
                           Accd
Accavg = d∈{b,n1,...,nt}
                    t+1         either at the final t = T step       training oracle that uses all base data but unavailable to us.
or follow the 500+50x10 split implementation [8] by aver-            For score fusion, we report the results of three operating
                          t=0..10
aging over all steps (Accavg      ). When there is class over-       points, the one with the best Accall , the best Accavg , and
lap, we use Accavg = (Accbase + Accnovel + Accovlp )/3               the most balanced performance of these two metrics. Fine-
for partial overlap and Accavg = (Accbase +Accovlp )/2 for           tuning and LwF [15] are simple baselines that do not adopt
full overlap (Yn ⊆ Yb ).                                             image replay and perform especially badly. iCaRL [23],
Learning. For stage-II training, 10 exemplars per class are          BiC [29], PODNet [8], and WA [34] keep few-shot exam-
randomly selected to create a class-balanced split, and the          plars to mitigate forgetting, but these methods still under-
experiments are repeated with different random seeds. Hy-            perform on base classes since they use a small amount of
perparameters α and β are selected with a grid search con-           data to modify the well-trained representation layers. We
ducted with the validation set as per the metric of interest.        outperform other methods under the multiple novel step

                                                                   9606

---

Table 1. Comparison to SOTA class-incremental learning methods. Our method outperforms without additional parameters and pushes the
performance further with additional parameters.

                                                                           ResNet10                                                        ResNet18
                   Method                         # of params     Accall    Accbase     Accnovel      Accavg    # of params    Accall       Accbase     Accnovel    Accavg
                 fine-tuning                         4.9M          4.18          0.01    87.63         43.82        11.2M           4.25      0.00       89.37      44.68
                  LwF [15]                                        9.50           5.54    88.53         47.04                        9.50      5.46       90.30      47.88
                iCaRL [23]                                        16.26         13.91    63.40         38.66                       10.65      8.15       60.80      34.78
                   BiC [29]                                       30.30         27.55    85.20         56.38                       31.50      28.75      86.60      57.68
                   WA [34]                                        51.33         52.33    31.40         41.87                       54.79      55.17      47.20      51.19
               DER w/o P [30]                        9.8M         52.31         52.43    50.10         51.27          -               -         -          -          -
       score fusion (ours) best-Accall               8.6M         63.24         63.77    52.67         58.22        19.6M          69.45      70.01      58.13      64.07
      score fusion (ours) best-balanced                           62.15         61.49    75.37         68.43                       67.36      66.61      82.37      74.49
       score fusion (ours) best-Accavg                            58.90         57.73    82.40         70.06                       65.83      64.85      82.50      75.17
  score fusion (ours, fc-only) best-Accall           4.9M         62.65         63.56    44.53         54.05        11.2M          68.79      69.58      53.07      61.32
 score fusion (ours, fc-only) best-balanced                       61.01         60.81    65.07         62.94                       66.76      66.50      71.83      69.17
  score fusion (ours, fc-only) best-Accavg                        57.91         57.24    71.57         64.40                       65.89      65.49      73.77      69.63
            joint learning (oracle)                  4.9M         63.80         63.94    61.00         62.47        11.2M          70.32      70.43      68.20      69.32

Table 2. Multiple novel steps disjoint-CIL with ResNet10, 500                             Table 3. Disjoint-CIL analysis with ResNet10, 40 novel classes
base classes with ten 50-class novel increments, random class                             using random class split. (Results for ResNet18/ResNet50 archi-
split. For ResNet18, BIC and PODNet results are from [8], and                             tectures and for 200 novel classes are in supplemental)
we use their experimental setup and 20/class memory constraints.
                                                                                                           Method                    Accall   Accbase    Accnovel   Accavg
             Network                        ResNet10               ResNet18                      confidence-based routing            41.58     39.26      88.00      63.63
             Method                   inc. acc. Acct=0..10
                                                   avg       inc. acc. Acct=0..10
                                                                          avg               learning-based routing w/ Lrt−bal        58.69     57.97      73.17      65.57
            BiC [29]                     –          –         44.31         –                   oracle routing w/ Lrt−bal            58.57     57.48      80.50      68.99
           PODNet [8]                    –          –         64.13         –                          FeatCat+RT                    55.94     56.08      53.27      54.67
  score fusion (ours) best-Accall      62.70      61.45       67.48        65.66                       LogitCat+RT                   58.36     58.56      54.30      56.43
 score fusion (ours) best-balanced     61.03      65.34       65.95        69.40                       LogitCat+FT                   59.39     59.34      60.57      59.95
  score fusion (ours) best-Accavg      55.31      66.23       61.06        70.67                   FA (ours) + BiC [29]              63.21     63.66      54.20      58.93
                                                                                                   FA (ours) + WA [34]               62.77     64.19      34.30      49.25
                                                                                                score fusion (ours) best-Accall      63.24     63.77      52.67      58.22
                                                                                               score fusion (ours) best-balanced     62.15     61.49      75.37      68.43
benchmark as well, including PODNet [8] which proposed                                          score fusion (ours) best-Accavg      58.90     57.73      82.40      70.06
the benchmark. DER w/o P [30] is a recent approach that                                             joint learning (oracle)          63.80     63.94      61.00      62.47
freezes the feature extractor of the base network and creates
an identical branch to learn novel features. While similar
to the proposed method, the architecture of the additional
feature extractor branch and the training objectives are dif-                             the worst Accbase , and around half of the base samples are
ferent. In addition, they relearn the final linear classifier                             misclassified as novel. Learning-based routing improves
with limited data, which we show is less effective in our ab-                             the result for Accbase , and reaches a more balanced perfor-
lation studies (Table 3). Compared with existing CIL meth-                                mance. The oracle routing performance (trained on all base
ods with the same backbone, the proposed method achieves                                  and novel data rather than 10-shot) is also shown in the ta-
significantly better performance in both Accall and Accavg                                ble for reference, and it mainly improves in novel classes
and generalize well into multiple novel step scenarios.                                   and the gap in Accbase remains.
    For fair comparison, we also present our results branch-                                  We then compare the score fusion method with a set of
ing at the classifier layer (“fc-only”) to bring our number of                            baselines in the proposed framework in Table 3. FeatCat +
network parameters on par with most compared methods,                                     RT retrains a linear classifier with features hb and hn con-
i.e. Φs = Φ, Φb = Φn = ∅, and hb = hn . Our results                                       catenated. LogitCat concatenates logits zbase and znovel ,
still outperform all compared methods, and interestingly,                                 and retrains (RT) or fine-tunes (FT) the linear weights Wb
our ResNet10 results outperform others’ ResNet18 results                                  and Wn . Since the original linear weights were already
despite using fewer parameters.                                                           trained using a large amount of data, retraining them from
Comparisons to fusion baselines. In Table 3, we explore                                   scratch on 10-shot with feature or logit concatenation leads
two lines of fusion methods in stage-II training, routing,                                to inferior results than fine-tuning the linear weights, as
and score fusion. We show that our observations hold for                                  shown in the table. This supports our idea of preserving
different backbones (ResNet18/ResNet50) and more novel                                    the weights of the original linear classifiers.
classes in supplemental.                                                                      Next, we apply two score fusion baselines, BiC [29] and
    Among routing methods, confidence-based routing has                                   WA [34], to our network (FA) after stage-I training. Both

                                                                                        9607

---

Table 4. Disjoint-CIL with ResNet10, 40 novel classes using the                                   Table 5. Overlapping-CIL results of style-changing splits (merge
inanimate-animate split (wider base-novel gap). Score fusion can                                  similar classes into one). See supplemental for overlapping-CIL
generalize to fine-tuning more/less layers in stage-I if necessary.                               with random and domain-changing split.

     Trainable                    Method                 Accall   Accbase   Accnovel   Accavg                  Method                   Accall Accbase Accnovel Accovlp Accavg
       layer3,4      score fusion (ours) best-balanced   57.08     55.27     83.10     69.18
        layer4       score fusion (ours) best-balanced   57.12     55.50     80.37     67.93
                                                                                                        confidence-based routing        40.70      38.17     86.97     80.00     68.38
 layer4 (conv2 only) score fusion (ours) best-balanced   57.15     55.73     77.37     66.55       learning-based routing w/ Lrt−bal    59.14      58.63     72.91     51.73     61.09
                                                                                                        oracle routing w/ Lrt−bal       59.57      58.83     78.63     51.20     62.89
                                                                                                      logit concatenation (avg pool)    63.39      63.63     64.57     40.40     56.20
                                                                                                     logit concatenation (max pool)     63.50      63.38     64.46     69.60     65.81
methods estimate a scalar to balance between the base and                                           score fusion (ours) best-Accall 63.81          64.35     53.76     56.13     58.08
novel logits, and they estimate it by validation data and                                          score fusion (ours) best-balanced 61.02         60.02     79.58     75.60     71.73
                                                                                                    score fusion (ours) best-Accavg 57.58          56.18     83.77     77.73     72.56
weight norms respectively. Using these methods with our
                                                                                                        joint learning (oracle)         64.33      64.28     62.06     76.80     67.71
network preserves the base class performance better than
RT or FT methods, and FA+BiC on ResNet10 even per-
forms similarly to ours (but not on ResNet18). Although                                           Table 6. Fully-overlapping-CIL with all 40 novel classes overlap-
                                                                                                  ping with the 800 base classes, i.e. Yn ⊂ Yb .
we note that using BiC and WA directly performs poorly as
shown in Table 1.                                                                                  Backbone                Method                  Accall   Accbase   Accovlp   Accavg
Robustness. To test the robustness of the proposed method,                                         ResNet10            pre-trained model           57.69     58.03     51.20    54.62
we study score fusion with a larger number of novel classes                                                        confidence-based routing        35.43     33.01     81.50    57.25
                                                                                                              learning-based routing w/ Lrt−bal    52.82     52.12     66.10    59.11
and deeper networks. Results of 800 base and 200 novel                                                                     fine-tuning             3.86       0.00     77.20    38.6
classes instead of 40 and the results for different architec-                                                              iCaRL [23]              20.23     19.77     28.90    24.34
                                                                                                                             RM [5]                21.10     18.12     77.70    47.91
tures are in supplemental. The same trends in Table 3 hold.                                                     score fusion (ours) best-Accall    57.82     57.80     58.23    58.02
Generalization. When the base and novel data are very                                                          score fusion (ours) best-balanced   55.66     54.81     71.90    63.36
                                                                                                                score fusion (ours) best-Accavg    53.42     52.28     75.13    63.70
different, e.g. in Table 4 where we use the domain-changing                                                          joint learning (oracle)       47.63     47.32     53.60    50.46
split (inanimate as base classes and animate for novel), fine-                                     ResNet18                iCaRL [23]              18.12     17.04     38.60    27.82
                                                                                                                             RM [5]                21.89     28.78     80.90    49.84
tuning more layers in the first stage may produce higher ac-
curacy on the novel data. We show that our score fusion
generalizes to different ways of branching the network.
                                                                                                  izes to both scenarios, with overlapping class samples either
5.3. Overlapping-CIL                                                                              randomly split or clustered.
                                                                                                     Additionally, Table 6 shows an extreme full-overlap sce-
   In practice, base and novel classes may not be mutually
                                                                                                  nario where all novel classes are overlapping with base
exclusive. In this section, we study the overlapping-CIL
                                                                                                  classes (random base-novel split, split overlapping classes
scenario where a subset of base and novel classes overlap.
                                                                                                  by clustering). This simulates the practical scenario where
As a recap of Section 5.1, we analyze five different ways to
                                                                                                  one needs to adapt a portion of existing classes to a new
split base and novel classes and samples within each class.
                                                                                                  domain. We compare to prior incremental learning work.
Unless otherwise noted, we test on ResNet10 with 800 base,
                                                                                                  These methods underperform due to changing the back-
40 novel with 5 overlapping classes.
                                                                                                  bone network weights. Even Rainbow Memory [5] which
   Table 5 shows the result when the style of the overlap-
                                                                                                  explicitly tackles fully overlapping classes, underperforms,
ping classes changes from base to novel. We simulate this
                                                                                                  partially because it assumes the distribution of each over-
by merging pairs of similar classes that are siblings in the
                                                                                                  lapping class does not change. In comparison, our method
WordNet hierarchy [20, 28], and placing the two classes in
                                                                                                  generalizes well to this scenario.
each pair into the base and novel split respectively. Com-
pared to average pooling, applying max pooling to the logits                                      6. Conclusion and Limitations
of overlapping classes performs better in Accovlp with little
drawback in Accbase . Hence, in our score fusion results,                                            In this work, we investigated CIL in the context of a
we adopt max pooling to fuse the logits. Confidence-based                                         pre-trained model with large number of base classes. We
routing performs the worst for the base classes. Our score                                        showed how branching can be an effective solution for
fusion achieves the best performance for both splits even                                         learning novel data when using a strong pre-trained model
when the samples of overlapping classes are in a different                                        and how it can preserve the learning with the old data. Fur-
style from base to novel.                                                                         thermore, we discuss a novel score fusion method that uses
   Due to space constraints, we show in the supplemental                                          both feature and classifier information from old and novel
results for the random base-novel-overlap split scenario and                                      networks and generates a unified classifier. This approach
the domain change (inanimate and animate, respectively,                                           leads to state-of-the-art results for CIL with large number of
and 5 inanimate classes as overlapping classes) scenario.                                         base classes. Our method can further be improved by using
The conclusions are identical and our score fusion general-                                       distillation approach to reduce memory footprint.

                                                                                                9608

---

References                                                                  ral networks. Proceedings of the national academy of sci-
                                                                            ences, 114(13):3521–3526, 2017. 1, 2, 3
 [1] Davide Abati, Jakub Tomczak, Tijmen Blankevoort, Simone
                                                                       [14] Xilai Li, Yingbo Zhou, Tianfu Wu, Richard Socher, and
     Calderara, Rita Cucchiara, and Babak Ehteshami Bejnordi.
                                                                            Caiming Xiong. Learn to grow: A continual structure learn-
     Conditional channel gated networks for task-aware contin-
                                                                            ing framework for overcoming catastrophic forgetting. In In-
     ual learning. In Proceedings of the IEEE/CVF Conference
                                                                            ternational Conference on Machine Learning, pages 3925–
     on Computer Vision and Pattern Recognition, pages 3931–
                                                                            3934. PMLR, 2019. 2
     3940, 2020. 2
 [2] Hongjoon Ahn, Jihwan Kwak, Subin Lim, Hyeonsu Bang,               [15] Zhizhong Li and Derek Hoiem. Learning without forgetting.
     Hyojun Kim, and Taesup Moon. Ss-il: Separated softmax                  IEEE transactions on pattern analysis and machine intelli-
     for incremental learning. In Proceedings of the IEEE/CVF               gence, 40(12):2935–2947, 2017. 1, 2, 3, 6, 7
     International Conference on Computer Vision, pages 844–           [16] Mingsheng Long, Yue Cao, Jianmin Wang, and Michael Jor-
     853, 2021. 2                                                           dan. Learning transferable features with deep adaptation net-
 [3] Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny,                  works. In International conference on machine learning,
     Marcus Rohrbach, and Tinne Tuytelaars. Memory aware                    pages 97–105. PMLR, 2015. 2
     synapses: Learning what (not) to forget. In Proceedings           [17] David Lopez-Paz and Marc’Aurelio Ranzato. Gradient
     of the European Conference on Computer Vision (ECCV),                  episodic memory for continual learning. Advances in neu-
     pages 139–154, 2018. 1, 2                                              ral information processing systems, 30:6467–6476, 2017. 1,
 [4] Rahaf Aljundi, Punarjay Chakravarty, and Tinne Tuytelaars.             2
     Expert gate: Lifelong learning with a network of experts.         [18] Arun Mallya, Dillon Davis, and Svetlana Lazebnik. Piggy-
     In Proceedings of the IEEE Conference on Computer Vision               back: Adapting a single network to multiple tasks by learn-
     and Pattern Recognition, pages 3366–3375, 2017. 2                      ing to mask weights. In Proceedings of the European Con-
 [5] Jihwan Bang, Heesu Kim, YoungJoon Yoo, Jung-Woo Ha,                    ference on Computer Vision (ECCV), pages 67–82, 2018. 1,
     and Jonghyun Choi. Rainbow memory: Continual learn-                    2
     ing with a memory of diverse samples. In Proceedings of           [19] Arun Mallya and Svetlana Lazebnik. Packnet: Adding mul-
     the IEEE/CVF Conference on Computer Vision and Pattern                 tiple tasks to a single network by iterative pruning. In Pro-
     Recognition, pages 8218–8227, 2021. 1, 2, 3, 8                         ceedings of the IEEE conference on Computer Vision and
 [6] Francisco M Castro, Manuel J Marı́n-Jiménez, Nicolás Guil,           Pattern Recognition, pages 7765–7773, 2018. 1, 2
     Cordelia Schmid, and Karteek Alahari. End-to-end incre-           [20] George A. Miller. Wordnet: A lexical database for english.
     mental learning. In Proceedings of the European conference             Communications of the ACM, 38:39–41, 11 1995. 8
     on computer vision (ECCV), pages 233–248, 2018. 1, 2, 3           [21] Pedro Morgado and Nuno Vasconcelos. Nettailor: Tuning
 [7] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-              the architecture, not just the weights. In Proceedings of
     Fei. Imagenet: A large-scale hierarchical image database. In           the IEEE/CVF Conference on Computer Vision and Pattern
     IEEE Conference on Computer Vision and Pattern Recogni-                Recognition, pages 3044–3054, 2019. 2
     tion (CVPR), 2009. 5
                                                                       [22] Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi.
 [8] Arthur Douillard and et al. Podnet: Pooled outputs distilla-
                                                                            Learning multiple visual domains with residual adapters.
     tion for small-tasks incremental learning. In ECCV 2020. 1,
                                                                            arXiv preprint arXiv:1705.08045, 2017. 2
     2, 3, 5, 6, 7
                                                                       [23] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg
 [9] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
                                                                            Sperl, and Christoph H Lampert. icarl: Incremental classifier
     Deep residual learning for image recognition. In Proceed-
                                                                            and representation learning. In Proceedings of the IEEE con-
     ings of the IEEE conference on computer vision and pattern
                                                                            ference on Computer Vision and Pattern Recognition, pages
     recognition, pages 770–778, 2016. 3
                                                                            2001–2010, 2017. 1, 2, 3, 4, 6, 7, 8
[10] Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, and
     Dahua Lin. Lifelong learning via progressive distillation and     [24] Andrei A Rusu, Neil C Rabinowitz, Guillaume Desjardins,
     retrospection. In Proceedings of the European Conference               Hubert Soyer, James Kirkpatrick, Koray Kavukcuoglu, Raz-
     on Computer Vision (ECCV), pages 437–452, 2018. 2                      van Pascanu, and Raia Hadsell. Progressive neural networks.
[11] Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, and                arXiv preprint arXiv:1606.04671, 2016. 1, 2, 3
     Dahua Lin. Learning a unified classifier incrementally via        [25] Xiaoyu Tao, Xinyuan Chang, Xiaopeng Hong, Xing Wei,
     rebalancing. In Proceedings of the IEEE/CVF Conference on              and Yihong Gong. Topology-preserving class-incremental
     Computer Vision and Pattern Recognition, pages 831–839,                learning. In European Conference on Computer Vision,
     2019. 1, 2, 3                                                          pages 254–270. Springer, 2020. 2
[12] Steven CY Hung, Cheng-Hao Tu, Cheng-En Wu, Chien-                 [26] Yu-Xiong Wang, Deva Ramanan, and Martial Hebert. Grow-
     Hung Chen, Yi-Ming Chan, and Chu-Song Chen. Compact-                   ing a brain: Fine-tuning by increasing model capacity. In
     ing, picking and growing for unforgetting continual learning.          Proceedings of the IEEE Conference on Computer Vision
     arXiv preprint arXiv:1910.06562, 2019. 2                               and Pattern Recognition, pages 2471–2480, 2017. 2, 3
[13] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel          [27] Mitchell Wortsman, Vivek Ramanujan, Rosanne Liu,
     Veness, Guillaume Desjardins, Andrei A Rusu, Kieran                    Aniruddha Kembhavi, Mohammad Rastegari, Jason Yosin-
     Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-                    ski, and Ali Farhadi. Supermasks in superposition. arXiv
     Barwinska, et al. Overcoming catastrophic forgetting in neu-           preprint arXiv:2006.14769, 2020. 2

                                                                     9609

---

[28] Tz-Ying Wu, Pedro Morgado, Pei Wang, Chih-Hui Ho, and
     Nuno Vasconcelos. Solving long-tailed recognition with
     deep realistic taxonomic classifier. In European Conference
     on Computer Vision (ECCV), 2020. 8
[29] Yue Wu and et al. Large scale incremental learning. In CVPR
     2019. 1, 2, 3, 6, 7
[30] Shipeng Yan, Jiangwei Xie, and Xuming He. Der: Dynam-
     ically expandable representation for class incremental learn-
     ing. In Proceedings of the IEEE/CVF Conference on Com-
     puter Vision and Pattern Recognition, pages 3014–3023,
     2021. 1, 2, 7
[31] Jaehong Yoon, Eunho Yang, Jeongtae Lee, and Sung Ju
     Hwang. Lifelong learning with dynamically expandable net-
     works. In International Conference on Learning Represen-
     tations (ICLR), 2018. 2
[32] Friedemann Zenke, Ben Poole, and Surya Ganguli. Contin-
     ual learning through synaptic intelligence. In International
     Conference on Machine Learning, pages 3987–3995. PMLR,
     2017. 1, 2, 3
[33] Jeffrey O Zhang, Alexander Sax, Amir Zamir, Leonidas
     Guibas, and Jitendra Malik. Side-tuning: A baseline for net-
     work adaptation via additive side networks. In Computer
     Vision–ECCV 2020: 16th European Conference, Glasgow,
     UK, August 23–28, 2020, Proceedings, Part III 16, pages
     698–714. Springer, 2020. 2
[34] Bowen Zhao, Xi Xiao, Guojun Gan, Bin Zhang, and Shu-
     Tao Xia. Maintaining discrimination and fairness in class
     incremental learning. In Proceedings of the IEEE/CVF Con-
     ference on Computer Vision and Pattern Recognition, pages
     13208–13217, 2020. 1, 2, 3, 6, 7

                                                                     9610

---
