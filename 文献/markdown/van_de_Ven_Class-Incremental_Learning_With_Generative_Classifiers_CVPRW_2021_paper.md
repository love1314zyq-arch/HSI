# van_de_Ven_Class-Incremental_Learning_With_Generative_Classifiers_CVPRW_2021_paper

Source: `van_de_Ven_Class-Incremental_Learning_With_Generative_Classifiers_CVPRW_2021_paper.pdf`

Class-Incremental Learning with Generative Classifiers

                              Gido M. van de Ven1,2, *, Zhe Li1 & Andreas S. Tolias1,3
          1
              Center for Neuroscience and Artificial Intelligence, Baylor College of Medicine, Houston, Texas, USA
           2
               Computational and Biological Learning Lab, University of Cambridge, Cambridge, United Kingdom
                 3
                   Department of Electrical and Computer Engineering, Rice University, Houston, Texas, USA

                          Abstract                                data or using replay are not allowed. In the past few years
                                                                  several methods have been proposed that can do class-
    Incrementally training deep neural networks to recog-         incremental learning without replay or stored data [7, 19,
nize new classes is a challenging problem. Most exist-            31, 34]. However, those methods rely on protocols with
ing class-incremental learning methods store data or use          explicit task boundaries and/or their performance critically
generative replay, both of which have drawbacks, while            depends on the availability of a suitably pre-trained feature
‘rehearsal-free’ alternatives such as parameter regulariza-       extractor.
tion or bias-correction methods do not consistently achieve          In this paper, we put forward generative classification
high performance. Here, we put forward a new strat-               as a promising new strategy for class-incremental learn-
egy for class-incremental learning: generative classifica-        ing. Specifically, instead of training neural networks to di-
tion. Rather than directly learning the conditional distri-       rectly learn the conditional distribution p(y|x), we propose
bution p(y|x), our proposal is to learn the joint distribu-       to train them to learn the joint distribution p(x, y), factor-
tion p(x, y), factorized as p(x|y)p(y), and to perform clas-      ized as p(x|y)p(y), and then to perform classification using
sification using Bayes’ rule. As a proof-of-principle, here       Bayes’ rule. A key benefit of this strategy is that it rephrases
we implement this strategy by training a variational autoen-      a challenging class-incremental learning problem as a more
coder for each class to be learned and by using importance        easily addressable task-incremental learning problem (see
sampling to estimate the likelihoods p(x|y). This simple          Section 4.1).
approach performs very well on a diverse set of continual            To demonstrate the potential of generative classification
learning benchmarks, outperforming generative replay and          for class-incremental learning, as a proof-of-principle we
other existing baselines that do not store data.                  implement this strategy by training a variational autoen-
                                                                  coder model for each class to be learned and by using im-
                                                                  portance sampling to estimate the class-conditional like-
1. Introduction                                                   lihoods during inference. We find that such a straight-
                                                                  forward implementation of a generative classifier performs
   Deep neural networks excel in supervised learning tasks,       very well on a diverse range of class-incremental learn-
but only when all the classes to be learned are available at      ing problems, outperforming generative replay and existing
the same time. Incrementally training a deep neural net-          rehearsal-free methods. Moreover, this approach does not
work to distinguish between a gradually growing number            use replay, it does not store data, it can be applied to arbi-
of classes has turned out to be very challenging [12, 43,         trary class-incremental data streams (i.e. no need for task
48, 50]. Successful strategies for class-incremental learn-       boundaries) and it does not rely on pre-trained networks,
ing generally either rely on storing a subset of the past data    although if available those can be used effectively.
and/or on replaying (representations of) past data, both of
which have important disadvantages. Storing data is not           2. Problem formulation
always possible in practice (e.g. due to safety/privacy con-
cerns or because of limited storage capacity), while replay          In continual or incremental learning, an algorithm does
— or rehearsal — is computationally expensive as it in-           not have access to all data at the same time, but it encoun-
volves constant retraining on past data.                          ters the data in a sequence [13, 17, 41]. Recently, three
   These drawbacks have sparked recent interest in                different types, or ‘scenarios’, of continual learning have
‘rehearsal-free’ continual learning [32], in which storing        been described [52]: in task-incremental learning an al-
                                                                  gorithm must incrementally learn a set of clearly distinct
  * Corresponding author: ven@bcm.edu                             tasks, in domain-incremental learning an algorithm must

---

learn the same task but with changing contexts, and in class-
incremental learning an algorithm must incrementally learn
to distinguish between a growing number of classes. In
this paper, we focus on class-incremental learning, which
is generally considered to be the most challenging contin-
ual learning scenario [6, 35, 42].                                Figure 1. Schematic illustrating the distinction between (A) task-
                                                                  based and (B) task-free class-incremental learning.
2.1. Class-incremental learning
   There are various different ways in which a class-             separate generative model for each class — the actual class-
incremental learning problem can be set up. This makes            incremental sequence of the data stream does not matter.
direct comparisons between studies challenging, even when            Task-free continual learning has also been referred to
they use the same datasets. We therefore start by discussing      as ‘streaming’ or ‘online’ continual learning. In that case,
some important assumptions that vary between studies.             sometimes additional constraints are that each training sam-
                                                                  ple should only be presented once and that the mini-batch
2.1.1   Task-based vs. task-free                                  size should be one [2, 19]. However, it is worth pointing
                                                                  out that these constraints relate to the sample efficiency of
The goal of class-incremental learning is to learn, given a       an algorithm and its robustness to noisy updates and, al-
dataset D = {xi , yi }ni=1 , a classification rule that maps an   though they are topics worth studying, these are indepen-
input x ∈ X to a predicted label y ∈ Y. However, unlike           dent from the distinction between task-based and task-free
in classical machine learning, the algorithm that must learn      class-incremental learning. For one of the benchmarks re-
this mapping is not given access to the entire dataset at once.   ported in this paper, we follow this more strict definition of
Instead, the data is made available according to a particular     streaming learning.
class-incremental protocol.

Task-based class-incremental learning A commonly                  2.1.2   Other critical assumptions
used class-incremental learning protocol is to split up the       Data storage An important assumption made by many
dataset into distinct ‘tasks’ (or ‘episodes’), whereby each       class-incremental learning methods is that it is acceptable to
task contains a different subset of classes [e.g. 43, 48, 52].    store a limited amount of past samples in a memory buffer.
The algorithm is then sequentially given access to the data       The size of this memory buffer is typically one of the most
of each task (Figure 1A). Importantly, after transitioning        important determinants of a method’s performance [4, 42].
from one task to the next, the data from the previous task        In practice, storing data is not always possible (e.g. safety
is no longer available. During each task, the training data of    or privacy concerns), and in this study we do not allow data
that task could either be given to the algorithm all at once,     storage, a setting which has been referred to as memoryless
or it might be presented according to a fixed stream that is      class-incremental learning [7].
not controlled by the algorithm (see Appendix B in [1]).
                                                                  Pre-training Another assumption commonly made in the
Task-free class-incremental learning It has been argued           class-incremental learning literature, especially by studies
that task-based protocols are not representative of real-         that do not allow storing data, is that a suitably pre-trained
world problems, and that the community should shift its           network or feature extractor is available or that there is an
focus to ‘task-free’ continual learning [2, 3, 19, 56]. In a      extended, non-incremental initialization phase that can be
task-free protocol, the algorithm is presented with an arbi-      used for pre-training [e.g. 19, 30, 34, 50]. While the im-
trary stream of data, without any prior knowledge about the       portance of the assumption about data storage seems to be
structure of this stream (Figure 1B). Many existing methods       widely acknowledged, this assumption about pre-training
for class-incremental learning cannot deal with this setting,     has received less attention. Here we investigate the impor-
because they rely on the presence of ‘task boundaries’ (see       tance of pre-training by considering both benchmarks with
Table B.1 in [1] for an overview).                                pre-trained networks available (CIFAR-100 and CORe50)
   In general, benchmarks for task-free class-incremental         and benchmarks without (MNIST and CIFAR-10).
learning need to include a protocol for how the data stream
should be generated (i.e. they should specify when sam-           3. Existing class-incremental learning methods
ples from each class are presented). An open, largely un-
                                                                  3.1. Methods relying on stored data
addressed research question relates to the development of a
principled way to design such data streams. In this paper            Many class-incremental learning methods store a sub-
we side-step this question, because for the particular imple-     set of past data in a memory buffer. That data could be
mentation of generative classifier considered here — with a       replayed when training on new data [10, 33, 46], they

---

could be used as exemplars or prototypes to guide classi-             is similar to CWR+ except that it does not freeze the hidden
fication decisions [12, 43] or they could be used in other            layers but regularizes them using a modified version of SI.
ways [5, 21, 54]. Important questions when storing data are               There are also several bias-correcting algorithms that
which samples to store [37, 39] and in what format [9, 18].           rely on stored data from previously seen classes [e.g. 5, 54],
As discussed, we do not consider methods that store data.             but as discussed we do not consider those methods here.
                                                                          Related to these bias-correction algorithms, a trick to
3.2. Generative replay                                                prevent large differences in the magnitude of the output
   If it is not possible to store data, an alternative is to replay   weights between tasks in the first place, is to always only
generated ‘pseudo-data’ [45]. This strategy has been shown            train on the classes from the current task (i.e. only include
to be successful for toy problems with relatively simple in-          the output units of classes from the current task in the
puts [48, 52], but it struggles on problems with more com-            softmax-normalization, see Appendix A.1.5 in [1] for de-
plex inputs, such as natural images [2, 28]. Some recent              tails). Zeno et al. [56] called this the ‘labels trick’. A lim-
studies have shown competitive performance with genera-               itation of this trick is that there is no attempt to train the
tive replay on class-incremental learning problems with nat-          network to distinguish between classes from different tasks.
ural images [11, 30, 50], but the approaches in those studies
depend on pre-trained networks (or on an extensive, non-              3.5. Other methods
incremental initialization phase [30]).                                   Incremental linear discriminant analysis [23, 40] is a
   We include two generative replay methods in our com-               popular method in the data mining community that is suit-
parison: deep generative replay [DGR; 48], which re-                  able for class-incremental learning. Until recently, this
plays pixel-level representations, and brain-inspired replay          method had largely been ignored in the continual learn-
[BI-R; 50], which replays latent feature representations.             ing community, likely because it can only learn a linear
                                                                      classifier. However, a recent study applied this method —
3.3. Regularization-based methods
                                                                      now referred to as streaming linear discriminant analysis
   A popular strategy for continual learning is parameter             [SLDA; 19] — to the features extracted by a fixed, pre-
regularization, which aims to minimize changes to param-              trained deep neural network, which resulted in impressive
eters important for previously learned tasks. Examples of             performance on several class-incremental learning prob-
this strategy are elastic weight consolidation [EWC; 25]              lems. The main disadvantage of SLDA is that it is not ca-
and synaptic intelligence [SI; 55]. Although it is well-              pable of representation learning, which means that its per-
established that these parameter regularization methods by            formance will likely heavily depend on the availability of
themselves do not perform well in the class-incremental               suitably pre-trained networks. Here we test this: on the
learning scenario [15, 22, 51], we include them in our                benchmarks in this paper for which no pre-trained networks
comparison for completeness. Some regularization-based                are available, we apply SLDA directly on the input space.
methods can be interpreted as performing approximate
Bayesian inference on the parameters of the neural net-               4. Proposed strategy: generative classification
work [16, 25, 38] (i.e. Bayes’ rule is used to find p(θ|D),
with D the observed data). Note that this is different from           4.1. General framework & intuition
the generative classification strategy proposed in this paper,            In deep learning, the typical approach to classification is
which uses Bayes’ rule for the classification decision (i.e. to       to train a neural network to directly learn the conditional
find p(y|x)).                                                         distribution p(y|x) that we are interested in, for example
                                                                      by training a feed-forward classifier with a softmax output
3.4. Bias-correcting algorithms
                                                                      layer using cross-entropy loss. When all classes are avail-
   When a standard softmax-based classifier is trained on             able at the same time, this approach indeed works very well.
a class-incremental learning problem, it ends up predicting           In the incremental setting, however, this direct approach
only the most recently seen classes [29]. It has been ar-             breaks down. A softmax classifier trained in the standard
gued that this is due to a bias in the output layer [6, 54],          way heavily over fits to the most recently seen classes, a
and several recent class-incremental learning methods aim             phenomenon referred to as catastrophic forgetting. A rea-
to correct this bias by making the magnitude of the output            son for this catastrophic forgetting is that, based on the most
weights of all classes comparable. Examples of this strat-            recently seen data, the empirical version of p(y|x) — which
egy are ‘CopyWeights with Re-init’ [CWR; 31] and its im-              the softmax classifier aims to learn — is indeed heavily bi-
proved version CWR+ [34]. A disadvantage of these two                 ased towards the most recent classes. So far, as reviewed in
methods is that they freeze the parameters of all hidden lay-         Section 3, the dominant approach in the continual learning
ers after the first task, so representation learning is limited.      field has been to try to find methods and tricks to alleviate
To address this, the method AR1 was proposed [34], which              catastrophic forgetting.

---

    Here we propose a shift of gears. Breaking with the tra-               and CIFAR-10, a completely separate VAE model is learned
ditional deep learning approach of training classifiers dis-               for every class, while for the experiments on CIFAR-100
criminatively, we propose to tackle class-incremental learn-               and CORe50 the lower, pretrained layers are shared be-
ing with generative classifiers. Rather than training deep                 tween all models (see Section 4.3).
neural networks to directly learn the conditional distribu-                    A VAE model consists of an encoder qφ that maps an
tion p(y|x), we propose to train them to learn the joint dis-              input x to a posterior distribution qφ (z|x) in latent space,
tribution p(x, y) — factorized as p(x|y)p(y) — and to use                  a decoder pθ that maps a latent variable z back to a distri-
Bayes’ rule for classification. The key benefit of this pro-               bution pθ (x|z) in the input space and a prior distribution
posed strategy is that, in a class-incremental learning set-               pprior (z). For the VAE models used in this paper, these dis-
ting, based on the most recently seen data the empirical ver-              tributions are given by:
sion of p(x|y) should not have any particular bias. Only the
                                                                                                              (x) 2
                                                                                                                   
empirical version of p(y) is biased, but learning this distri-                                          (x)
                                                                                        qφ (z|x) = N z µφ , σ φ I                     (1)
bution without catastrophic forgetting is typically straight-                                                
                                                                                                        (z)
forward (e.g. the number of times each label is observed                                pθ (x|z) = N x µθ , I                         (2)
could be counted) or not needed (e.g. if it can be assumed
that all labels have the same prior probability).                                       pprior (z) = N (z | 0, I )                    (3)

Class-incremental problem becomes task-incremental                                    (x)        (x)
                                                                           whereby µφ and σ φ are the outputs of the encoder net-
Another way to describe the benefit of the proposed gen-                                                 (z)
                                                                           work when x is fed in, and µθ is the output of the decoder
erative classifier strategy is that it turns a challenging
                                                                           network when z is fed in. For both the encoder network
class-incremental learning problem into an easier task-
                                                                           and the decoder network, we use deep neural networks. See
incremental learning problem. This is the case because
                                                                           Appendix A.3 in [1] for full details on the architectures that
learning p(x|y) can be interpreted as a task-incremental
                                                                           are used for the different benchmarks. Importantly, for each
problem whereby each ‘task’ consists of learning a class-
                                                                           benchmark, the architecture of the VAE models is chosen so
conditional generative model for a specific label y. An im-
                                                                           that the total number of parameters of the generative classi-
portant advantage of task-incremental learning is that it
                                                                           fier is similar to the number of parameters used by genera-
is possible to train networks with task-specific compo-
                                                                           tive replay.
nents [e.g. 36, 47, 53], or even to use completely separate
                                                                               The VAE models are trained by optimizing   R a variational
networks for each task to be learned. This last insight is
                                                                           lower   bound   to the likelihood  p θ (x)  =     pθ (x, z)dz =
used for our proof-of-principle implementation of a gener-                 R
                                                                              pθ (x|z)pprior (z)dz. This lower bound, or ELBO, is given
ative classifier with a separate generative model for every
                                                                           by:
class. Note however that it should be possible to use other
task-incremental learning techniques to enable parameter                                                    
                                                                                                                   pθ (x, z)
                                                                                                                             
sharing between these models (see also the discussion).                      LELBO (θ, φ; x) = Eqφ (z|x) log
                                                                                                                   qφ (z|x)
4.2. Implementation: VAEs & importance sampling                                = Eqφ (z|x) [log pθ (x|z)] − DKL (qφ (z|x)||pprior (z))
   In this paper, to demonstrate the potential of the pro-                                                                           (4)
posed generative classification strategy, we implement a
                                                                           where DKL is the Kullback-Leibler divergence. Full details
generative classifier by training a variational autoencoder
                                                                           of the VAE training are given in Appendix A.2 in [1].
[VAE; 24] model for each class to be learned1 and by us-
ing importance sampling to estimate the likelihoods p(x|y).
For p(y) we use a uniform distribution over all possible                   4.2.2   Importance sampling
classes, as all benchmarks have an approximately equal
                                                                           To estimate the likelihoods p(x|y), we use importance sam-
amount of samples per class. In general, p(y) could be
                                                                           pling [8, 44]. This means that the likelihood of a test sam-
learned from the data as well, for example by counting the
                                                                           ple x under the VAE model of class y is estimated using:
number of times each class is observed in the training data.
                                                                                                S
                                                                                             1 X pθy x|z (s) pprior z (s)
                                                                                                                         
4.2.1    Variational autoencoder                                                    p(x|y) =                                         (5)
                                                                                             S s=1    qφy z (s) |x
To learn the distribution p(x|y), we train a VAE model for
each class to be learned. For the experiments on MNIST                     whereby θ y and φy are the parameters of the VAE model
   1 Note that this setup could also be described as a single VAE          of class y, S is the number of importance samples and z (s)
model with class-specific masks whereby for each class a different, non-   is the sth importance sample drawn from qφy (z|x). For the
overlapping subset of parameters is unmasked.                              results in Table 2, we use S = 10, 000 importance samples

---

Table 1. Overview of the benchmarks used in this paper. Each benchmark consists of an image dataset split up into a number of distinct
tasks, with all tasks containing an equal number of classes. Such a task-based design is not needed for our generative classifier, but it is
used to enable a comparison with other methods. Within each task, the training data is presented to the algorithm in a random, i.i.d stream,
with the number of iterations per task and the mini-batch size being part of the benchmark. Another important aspect of each benchmark
is whether pre-trained models are available. For all benchmarks considered in this paper, storing data is not allowed.

                                       Dataset Info                    Data-Stream Parameters                       Pretrained
                                  Classes    Image-type              Tasks   Iterations Batch size                   Models?

                  MNIST             10        28x28, grey             5           2000             128                  -
                  CIFAR-10          10        32x32, RGB              5           5000             256                  -
                  CIFAR-100         100       32x32, RGB              10          5000             256              ConvLayers
                  CORe50            10       128x128, RGB             5        single pass          1                ResNet18

for each likelihood estimation. The effect of reducing the                 data (see Appendix A.1 in [1] for technical details of all
number of importance samples is explored in Table 4.                       compared methods). As far as possible, we use the same
   Based on Bayes’ rule: p(y|x) ∝ p(x|y)p(y), classifica-                  “base network” architecture and the same training settings
tion is then done using:                                                   for all compared methods. Full details of the architec-
                                                                           tures and training settings used for each benchmark are pro-
       ŷ (x) = argmax p(x|y)p(y) = argmax p(x|y)              (6)         vided in Appendix A.3 in [1]. Documented code for all
                  y∈Y                       y∈Y
                                                                           experiments (including for all compared methods) is avail-
whereby ŷ (x) is the class label predicted by the generative              able online: https://github.com/GMvandeVen/
classifier for test sample x. Note that the last equality in               class-incremental-learning.
Eq. 6 holds because, in this paper, p(y) is modelled with a
uniform distribution over all possible classes.
                                                                           5.1. Benchmarks
                                                                              An overview of the benchmarks used in this paper is pro-
4.3. When pre-trained models are available:
                                                                           vided in Table 1. All benchmarks are set up as task-based,
      reconstruction loss in the feature space
                                                                           in order to be able to compare with current state-of-the-art
    The generative classifier approach described so far does               class-incremental learning methods, even though our gen-
not depend on the availability of pre-trained networks, as it              erative classifier can be applied to task-free protocols as
is possible to train the full generative models from scratch.              well.2 Important aspects of each benchmark are the number
If pre-trained models are available, however, there are var-               of tasks, the number of iterations per task, the mini-batch
ious ways in which they could be used. For example, sup-                   size and whether pre-trained models are available. For all
pose that pre-trained convolutional layers are available. One              benchmarks, within tasks the training data is always fed to
option would be to use these to initialize the convolutional               the network in an i.i.d. stream, although some of the com-
layers of the encoder networks of the VAE models, and then                 pared methods (EWC, and SLDA for the first task) addi-
to proceed with training in the standard way. Another op-                  tionally assume they can access a task’s training data in one
tion, which is the approach taken in this paper, is to use the             large batch (see Appendix B in [1]).
pre-trained convolutional layers as a fixed feature extractor,
and then to train the VAE models on the extracted features                 5.1.1    MNIST
rather than on the raw inputs. An advantage of this second
approach, which is reminiscent of recent studies that per-                 The first benchmark is based on the MNIST dataset [27],
formed generative replay in the feature space [30, 50], is                 which is split up into 5 tasks with 2 digits each. Following
that it appears to be easier to learn good generative mod-                 previous studies [22, 51], this benchmark has 2000 itera-
els for such extracted features, presumably because they are               tions per task and a mini-batch size of 128. The base net-
less complex than the raw inputs.                                          work for this benchmark is a fully-connected network with
                                                                           2 hidden layers of 400 ReLU units and a softmax output
5. Experiments                                                             layer. No pre-training is used.
   In this section we test the above implementation of the                     2 For the specific implementation of the generative classifier used in

proposed generative classification strategy on a diverse set               this paper, with a separate model for each class, the performance does not
                                                                           depend on the specific class-incremental sequence at all. The reason is that
of class-incremental learning benchmarks. On each bench-
                                                                           the class-specific VAE models are trained only on samples of their own
mark, we compare our generative classifier with the ap-                    class, and it therefore does not matter if those classes are intermingled in
plicable methods discussed in Section 3 that do not store                  certain ways.

---

Table 2. Final test accuracy (as %) of all compared methods on the different benchmarks. Evaluation is according to the “class-incremental
learning scenario” or the “single-headed setting” (i.e. the model has to chose between all classes). Only methods that do not store data are
included. All experiments were performed 10 times with different random seeds, reported are the means (± SEMs) over these runs.

               Strategy                Method          MNIST             CIFAR-10           CIFAR-100         CORe50

                                       None           19.92 (± 0.02)    18.74 (± 0.29)     7.96 (± 0.11)     18.65 (± 0.26)
               Baselines
                                       Joint          98.23 (± 0.04)    82.07 (± 0.15)    54.08 (± 0.27)     71.85 (± 0.30)
                                       DGR            91.30 (± 0.60)    17.21 (± 1.88)     9.22 (± 0.24)          -
               Generative Replay       BI-R                -                 -            21.51 (± 0.25)     60.40 (± 1.04)
                                       BI-R + SI           -                 -            34.38 (± 0.21)     62.68 (± 0.72)
                                       EWC            19.95 (± 0.05)    18.63 (± 0.29)      8.47 (± 0.09)    18.56 (± 0.31)
               Regularization
                                       SI             19.95 (± 0.11)    18.14 (± 0.36)      8.43 (± 0.08)    18.69 (± 0.26)
                                       CWR            32.48 (± 2.64)    18.37 (± 1.61)    21.90 (± 0.68)     40.28 (± 1.13)
                                       CWR+           37.20 (± 3.11)    22.32 (± 1.08)     9.34 (± 0.25)     40.12 (± 1.06)
               Bias-correction
                                       AR1            48.84 (± 2.55)    24.44 (± 1.08)    20.62 (± 0.45)     45.27 (± 1.02)
                                       Labels Trick   32.46 (± 1.95)    18.43 (± 1.31)    23.68 (± 0.26)     42.59 (± 1.03)
               Other                   SLDA           87.30 (± 0.02)    38.35 (± 0.03)    44.49 (± 0.00)     70.80 (± 0.00)
               Generative Classifier                  93.79 (± 0.08)    56.03 (± 0.04)    49.55 (± 0.06)     70.81 (± 0.11)

5.1.2   CIFAR-10 without pre-training                                    class. The dataset is split up into 5 tasks with 2 classes
                                                                         each. This benchmark follows the more strict definition
For this benchmark the CIFAR-10 dataset [26] is split up
                                                                         of streaming learning: each training image is presented by
into 5 tasks with 2 classes each. The number of iterations
                                                                         itself (i.e. mini-batch size of 1) and only once. Follow-
per task for this benchmark is 5000 and the mini-batch size
                                                                         ing [19], a standard ResNet18 pretrained on ImageNet is
is 256. Following previous studies [2, 12, 33], the base net-
                                                                         used as a fixed feature extractor. The base network on top
work is a small version of ResNet18 [20] with three times
                                                                         of this feature extractor consists of one fully connected layer
less feature maps across all layers. No pre-training is used.
                                                                         with 1024 ReLU units and a softmax output layer.

5.1.3   CIFAR-100 with pre-training on CIFAR-10                          5.2. Results
This benchmark is taken from the study that proposed                         Table 2 shows the performance of our generative classi-
BI-R [50]. The CIFAR-100 dataset [26] is split up into 10                fier on the four benchmarks described above, along with the
tasks with 10 classes each. There are 5000 iterations per                performance of the methods discussed in Section 3 that also
task with mini-batch size of 256. The base network is a con-             do not store data. The generative classifier performed very
volutional neural network with 5 pre-trained convolutional               strongly, comfortably outperforming all compared methods
layers followed by 2 randomly initialized fully-connected                on three out of four benchmarks. Of special note are the
layers with 2000 ReLU units and a softmax output layer.                  substantial gaps with the generative replay variants, while
The convolutional layers were pre-trained on CIFAR-10.                   these methods used similar number of parameters. Only
To enable a direct comparison, we use the exact same pre-                on the CORe50 benchmark, in which an extensively pre-
trained convolutional layers as in [50], which were made                 trained network was used and in which each sample was
publicly available by the authors.                                       presented only once, the performance of the generative clas-
                                                                         sifier was comparable to that of SLDA, while still substan-
                                                                         tially higher than that of the other compared methods.
5.1.4   CORe50 with pre-training on ImageNet
                                                                             An interesting result is that SLDA still performed com-
The final benchmark is based on the CORe50 dataset [31].                 petitively when it was applied directly on the raw inputs of
This dataset is made up of image-frames cropped from short               MNIST and CIFAR-10. Although its performance was well
15 second videos of moving objects. There are 10 differ-                 below that of our generative classifier, it outperformed al-
ent classes, with each class represented in the dataset by 5             most all other methods.
different objects that were each filmed in 11 different envi-                Another thing to note from these results is the modest
ronments. As in [14, 31], we use the images from eight of                performance of the bias-correction methods, especially on
these environments for training and the others for testing.              benchmarks where no pre-training was used. When pre-
This results in approximately 10, 500 training images per                trained networks were available the relative performances

---

        Figure 2. Samples randomly drawn from the VAE models of the generative classifier for (A) MNIST and (B) CIFAR-10.

Table 3. Comparison of the performance of the generative classifier with the performance of a softmax-based classifier discriminatively
trained on samples from the VAE models of the generative classifier. Shown is the test accuracy (as %) over all classes. All experiments
were performed 10 times with different random seeds, reported are the means (± SEMs) over these runs.

                                                              MNIST            CIFAR-10          CIFAR-100         CORe50

     Generative classifier                                    93.79 (± 0.08)   56.03 (± 0.04)    49.55 (± 0.06)    70.81 (± 0.11)
     Discriminative classifier trained on generated samples   85.93 (± 0.43)   13.71 (± 0.61)    33.84 (± 0.14)    47.86 (± 1.77)

of these methods improved, but they did not come close to              that, also when the same generative models are used, gener-
those of the best performing methods.                                  ative classification outperforms generative replay.
                                                                          The results in Table 3 also indicate that the quality of
5.3. Generative classification vs. generative replay                   the samples produced by the VAE models of our genera-
    An intriguing result from the above comparisons is that            tive classifiers was not so good. To check this, we visual-
our generative classifier consistently and sometimes sub-              ized samples drawn from the VAE models of the generative
stantially outperformed generative replay. This suggests               classifier for the MNIST and CIFAR-10 benchmarks (Fig-
that directly using generative models to perform classifica-           ure 2). While for MNIST the generated samples look rea-
tion might be a better strategy than using those models indi-          sonable, for CIFAR-10 they are indeed not great. This thus
rectly to generate replay for discriminatively training a clas-        indicates that competitive class-incremental learning per-
sifier. However, it could be argued that this conclusion is not        formance could be obtained by a generative classifier even
completely warranted by these results, as both strategies did          without high-quality generative models.
not use the exact same generative models (even though the
                                                                       6. Discussion
total number of parameters was similar). For generative re-
play one large generative model was incrementally trained                  Class-incremental learning is a challenging problem. So
on all classes, while for the generative classifier a series of        far the deep learning community has tackled this prob-
smaller, separate generative models was trained.                       lem by directly learning a discriminative classifier, which
    To more directly compare generative classification and             only seems to work in combination with tricks such as pre-
generative replay, we trained — in an i.i.d. manner — a                training, storing data or replay. Here we proposed an alter-
softmax-based classifier on samples generated by the VAE               native strategy — to learn a generative classifier — and we
models of the generative classifier (see Appendix A.4 in [1]           showed that it can outperform generative replay and exist-
for full details on this experiment). Another way to phrase            ing rehearsal-free methods.
this experiment is that a discriminative classifier was trained            An interesting finding from our comparison of class-
exclusively with ‘generative replay’ produced by the same              incremental learning methods was the strong performance
generative models as used by the generative classifier. For            of SLDA [19]. It outperformed generative replay variants
all benchmarks, we found that the generative classifier                on three out of four benchmarks, and it achieved competi-
substantially outperformed the discriminative classifier that          tive performance even when applied directly on the raw in-
was trained on its own samples (Table 3). This suggests                puts. We believe this strong performance can be explained

---

Table 4. Performance of generative classifier as function of number of importance samples used for inference. Shown is test accuracy (as
%) over all classes. Experiments were performed 10 times with different random seeds, reported are the means (± SEMs) over these runs.

                                S=1                S = 10            S = 100          S = 1, 000        S = 10, 000

                 MNIST          91.14 (± 0.08)     92.46 (± 0.09)    93.25 (± 0.09)   93.62 (± 0.10)    93.79 (± 0.08)
                 CIFAR-10       50.86 (± 0.10)     54.64 (± 0.09)    55.43 (± 0.10)   55.83 (± 0.09)    56.03 (± 0.04)
                 CIFAR-100      45.02 (± 0.10)     48.45 (± 0.10)    49.26 (± 0.10)   49.48 (± 0.08)    49.55 (± 0.06)
                 CORe50         61.00 (± 0.19)     69.09 (± 0.14)    70.33 (± 0.14)   70.62 (± 0.14)    70.81 (± 0.11)

because SLDA can be interpreted as a generative classifier.            erative models [37]) to inform the number of importance
SLDA learns a mean vector µy for each class y and a co-                samples to use, or the classification decision could be made
variance matrix Σ that is shared between all classes. The              hierarchical (e.g. first decide whether it is a cat or a dog,
generative model that SLDA implicitly assumes           for each      then decide on the specific breed).
class y is given by p(x|y) = N x µy , Σ . SLDA is how-                     Another disadvantage of our specific implementation of
ever “a generative classifier in disguise” because it does not         the generative classifier is that a completely new generative
explicitly compute the likelihoods during inference, since             model is learned for each new class. It could be questioned
with its assumptions the decision boundaries implied by the            how scalable this is. In this regard, we believe it is im-
underlying generative models can be computed analytically.             portant to point out three things. Firstly, to ensure a fair
    The main disadvantage of SLDA is that it can only learn            comparison between our generative classifier and genera-
linear classifiers. To further improve upon SLDA, it seems             tive replay, we controlled for the total number of parame-
necessary to find a way to do representation learning in a             ters. Secondly, as illustrated by SLDA, even using small
class-incremental way. This is exactly what our deep gen-              or minimal generative models for each class can result in
erative classifiers are able to do. Learning good represen-            competitive performance. Finally, and perhaps most impor-
tations is not easy, and it is not surprising that this abil-          tantly, the main point of this paper is to highlight the poten-
ity comes at a cost of increased sample complexity. How-               tial of generative classification for class-incremental learn-
ever, (complex) representation learning is not a necessary             ing: our implementation with independent VAE models is
component of our proposed strategy. When the amount of                 a proof-of-principle. For practical applications, the gener-
training data is small, or when the representations provided           ative models of the different classes should probably share
by a pre-trained network are already good, it is probably              substantial parts of their networks. Such sharing introduces
better to learn relatively simple generative models. Indeed,           the risk of interference, but it also opens up the possibility
SLDA’s performance can be seen as the minimal attainable               of positive transfer between the generative models. Impor-
performance for a generative classifier, upon which can be             tantly, as pointed out in Section 4.1, learning the different
improved when sufficient data is available.                            class-conditional generative models is a task-incremental
    Compared with generative replay, an important advan-               problem, which is an important simplification compared to
tage of generative classifiers is that training is less costly, as     the original class-incremental problem [52]. We therefore
replay is not necessary. On the flip-side, inference (i.e. mak-        expect the question of how to optimally share parts of the
ing a classification decision) with generative classifiers is          generative models to be a fruitful topic for further research.
relatively costly, as it involves computing/estimating the
likelihood of a test sample under the generative model of              Acknowledgments
each possible class. For our specific implementation, this
seems especially problematic because a large number of                     We thank Siddharth Swaroop and Martin Mundt for use-
importance samples tends to be needed for high precision               ful comments. This research project has been supported by
likelihood estimates with VAE models [49]. For the results             the Lifelong Learning Machines (L2M) program of the De-
reported in Table 2, we used 10, 000 importance samples                fence Advanced Research Projects Agency (DARPA) via
for each likelihood estimation. However, we found that the             contract number HR0011-18-2-0025 and by the Intelligence
number of importance samples could be lowered substan-                 Advanced Research Projects Activity (IARPA) via Depart-
tially without large drops in performance (Table 4). Even              ment of Interior/Interior Business Center (DoI/IBC) con-
using just a single importance sample resulted in state-of-            tract number D16PC00003. Disclaimer: The views and
the-art class-incremental learning performance on three out            conclusions contained herein are those of the authors and
of four benchmarks. Moreover, there are also other tricks              should not be interpreted as necessarily representing the of-
that could speed up inference: it might be possible to use             ficial policies or endorsements, either expressed or implied,
uncertainty estimates (which can be obtained from the gen-             of DARPA, IARPA, DoI/IBC, or the U.S. Government.

---

References                                                                Pascanu. Embracing change: Continual learning in deep
                                                                          neural networks. Trends in Cognitive Sciences, 2020.
 [1] Authors [...]. Supplementary material, 2021. Supplied as        [18] Tyler L Hayes, Kushal Kafle, Robik Shrestha, Manoj
     suppl material.pdf.                                                  Acharya, and Christopher Kanan. Remind your neural net-
 [2] Rahaf Aljundi, Eugene Belilovsky, Tinne Tuytelaars, Lau-             work to prevent catastrophic forgetting. In European Confer-
     rent Charlin, Massimo Caccia, Min Lin, and Lucas Page-               ence on Computer Vision, pages 466–483. Springer, 2020.
     Caccia. Online continual learning with maximal interfered       [19] Tyler L Hayes and Christopher Kanan. Lifelong machine
     retrieval. In Advances in Neural Information Processing Sys-         learning with deep streaming linear discriminant analysis.
     tems, pages 11849–11860, 2019.                                       In Proceedings of the IEEE/CVF Conference on Computer
 [3] Rahaf Aljundi, Klaas Kelchtermans, and Tinne Tuyte-                  Vision and Pattern Recognition Workshops, pages 220–221,
     laars. Task-free continual learning. In Proceedings of               2020.
     the IEEE/CVF Conference on Computer Vision and Pattern          [20] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
     Recognition, pages 11254–11263, 2019.                                Identity mappings in deep residual networks. In European
 [4] Yogesh Balaji, Mehrdad Farajtabar, Dong Yin, Alex Mott,              conference on computer vision, pages 630–645. Springer,
     and Ang Li. The effectiveness of memory replay in large              2016.
     scale continual learning. arXiv preprint arXiv:2010.02418,      [21] Saihui Hou, Xinyu Pan, Chen Change Loy, Zilei Wang, and
     2020.                                                                Dahua Lin. Learning a unified classifier incrementally via re-
 [5] Eden Belouadah and Adrian Popescu. Scail: Classifier                 balancing. In Proceedings of the IEEE Conference on Com-
     weights scaling for class incremental learning. In Proceed-          puter Vision and Pattern Recognition, pages 831–839, 2019.
     ings of the IEEE/CVF Winter Conference on Applications of       [22] Yen-Chang Hsu, Yen-Cheng Liu, and Zsolt Kira. Re-
     Computer Vision, pages 1266–1275, 2020.                              evaluating continual learning scenarios: A categoriza-
 [6] Eden Belouadah, Adrian Popescu, and Ioannis Kanellos.                tion and case for strong baselines.             arXiv preprint
     A comprehensive study of class incremental learning algo-            arXiv:1810.12488, 2018.
     rithms for visual tasks. Neural Networks, 2020.                 [23] Tae-Kyun Kim, Björn Stenger, Josef Kittler, and Roberto
 [7] Eden Belouadah, Adrian Popescu, and Ioannis Kanellos. Ini-           Cipolla. Incremental linear discriminant analysis using suf-
     tial classifier weights replay for memoryless class incremen-        ficient spanning sets and its applications. International Jour-
     tal learning. In British Machine Vision Conference, 2020.            nal of Computer Vision, 91(2):216–232, 2011.
 [8] Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov. Im-         [24] Diederik P Kingma and Max Welling. Auto-encoding varia-
     portance weighted autoencoders. In International Con-                tional bayes. arXiv preprint arXiv:1312.6114, 2013.
     ference on Learning Representations, 2016.                      [25] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel
 [9] Lucas Caccia, Eugene Belilovsky, Massimo Caccia, and                 Veness, Guillaume Desjardins, Andrei A Rusu, Kieran
     Joelle Pineau. Online learned continual compression with             Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-
     adaptive quantization modules. In International Conference           Barwinska, et al. Overcoming catastrophic forgetting in neu-
     on Machine Learning, pages 1240–1250. PMLR, 2020.                    ral networks. Proceedings of the National Academy of Sci-
[10] Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny,
                                                                          ences, page 201611835, 2017.
     Thalaiyasingam Ajanthan, Puneet K Dokania, Philip HS            [26] Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple
     Torr, and Marc’Aurelio Ranzato. On tiny episodic memo-               layers of features from tiny images. Technical report, Uni-
     ries in continual learning. arXiv preprint arXiv:1902.10486,         versity of Toronto, 2009.
     2019.                                                           [27] Yann LeCun, Léon Bottou, Yoshua Bengio, Patrick Haffner,
[11] Yulai Cong, Miaoyun Zhao, Jianqiao Li, Sijia Wang, and               et al. Gradient-based learning applied to document recogni-
     Lawrence Carin. GAN memory with no forgetting. In Ad-                tion. Proceedings of the IEEE, 86(11):2278–2324, 1998.
     vances in Neural Information Processing Systems, 2020.          [28] Timothée Lesort, Hugo Caselles-Dupré, Michael Garcia-
[12] Matthias De Lange and Tinne Tuytelaars. Continual pro-               Ortiz, Andrei Stoian, and David Filliat. Generative models
     totype evolution: Learning online from non-stationary data           from the perspective of continual learning. In International
     streams. arXiv preprint arXiv:2009.00919, 2020.                      Joint Conference on Neural Networks. IEEE, 2019.
[13] Matthias Delange, Rahaf Aljundi, Marc Masana, Sarah             [29] Shuang Li, Yilun Du, Gido M van de Ven, Antonio Tor-
     Parisot, Xu Jia, Ales Leonardis, Greg Slabaugh, and Tinne            ralba, and Igor Mordatch. Energy-based models for contin-
     Tuytelaars. A continual learning survey: Defying forgetting          ual learning. arXiv preprint arXiv:2011.12216, 2020.
     in classification tasks. IEEE Transactions on Pattern Analy-    [30] Xialei Liu, Chenshen Wu, Mikel Menta, Luis Herranz, Bog-
     sis and Machine Intelligence, 2021.                                  dan Raducanu, Andrew D Bagdanov, Shangling Jui, and
[14] Arthur Douillard and Timothée Lesort. Continuum: Simple             Joost van de Weijer. Generative feature replay for class-
     management of complex continual learning scenarios. arXiv            incremental learning. In Proceedings of the IEEE/CVF Con-
     preprint arXiv:2102.06253, 2021.                                     ference on Computer Vision and Pattern Recognition Work-
[15] Sebastian Farquhar and Yarin Gal.               Towards ro-
                                                                          shops, pages 226–227, 2020.
     bust evaluations of continual learning. arXiv preprint          [31] Vincenzo Lomonaco and Davide Maltoni. Core50: a new
     arXiv:1805.09733, 2018.                                              dataset and benchmark for continuous object recognition. In
[16] Sebastian Farquhar and Yarin Gal. A unifying bayesian view
                                                                          Conference on Robot Learning, pages 17–26. PMLR, 2017.
     of continual learning. arXiv preprint arXiv:1902.06494,         [32] Vincenzo Lomonaco, Davide Maltoni, and Lorenzo Pelle-
     2019.                                                                grini. Rehearsal-free continual learning over small non-iid
[17] Raia Hadsell, Dushyant Rao, Andrei A Rusu, and Razvan

---

     batches. In Proceedings of the IEEE/CVF Conference on                 attention to the task. In International Conference on Machine
     Computer Vision and Pattern Recognition Workshops, pages              Learning, pages 4548–4557. PMLR, 2018.
     989–998, 2020.                                                   [48] Hanul Shin, Jung Kwon Lee, Jaehong Kim, and Jiwon
[33] David Lopez-Paz and Marc’Aurelio Ranzato. Gradient                    Kim. Continual learning with deep generative replay. In
     episodic memory for continual learning. In Proceedings                Advances in Neural Information Processing Systems, pages
     of the 31st International Conference on Neural Information            2994–3003, 2017.
     Processing Systems, pages 6470–6479, 2017.                       [49] L Theis, A van den Oord, and M Bethge. A note on the
[34] Davide Maltoni and Vincenzo Lomonaco. Continuous learn-               evaluation of generative models. In International Conference
     ing in single-incremental-task scenarios. Neural Networks,            on Learning Representations, 2016.
     116:56–73, 2019.                                                 [50] Gido M van de Ven, Hava T Siegelmann, and Andreas S To-
[35] Marc Masana, Xialei Liu, Bartlomiej Twardowski, Mikel                 lias. Brain-inspired replay for continual learning with ar-
     Menta, Andrew D Bagdanov, and Joost van de Weijer. Class-             tificial neural networks. Nature Communications, 11:4069,
     incremental learning: survey and performance evaluation.              2020.
     arXiv preprint arXiv:2010.15277, 2020.                           [51] Gido M van de Ven and Andreas S Tolias. Generative replay
[36] Nicolas Y Masse, Gregory D Grant, and David J Freedman.               with feedback connections as a general strategy for continual
     Alleviating catastrophic forgetting using context-dependent           learning. arXiv preprint arXiv:1809.10635, 2018.
     gating and synaptic stabilization. Proceedings of the Na-        [52] Gido M van de Ven and Andreas S Tolias. Three scenar-
     tional Academy of Sciences, pages E10467–75, 2018.                    ios for continual learning. arXiv preprint arXiv:1904.07734,
[37] Martin Mundt, Yong Won Hong, Iuliia Pliushch, and Vis-                2019.
     vanathan Ramesh. A wholistic view of continual learn-            [53] Joshua T Vogelstein, Jayanta Dey, Hayden S Helm, Will
     ing with deep neural networks: Forgotten lessons and the              LeVine, Ronak D Mehta, Ali Geisa, Gido M van de Ven,
     bridge to active and open world learning. arXiv preprint              Emily Chang, Chenyu Gao, Weiwei Yang, Bryan Tower,
     arXiv:2009.01797, 2020.                                               Jonathan Larson, Christopher M White, and Carey E Priebe.
[38] Cuong V Nguyen, Yingzhen Li, Thang D Bui, and Richard E               Omnidirectional transfer for quasilinear lifelong learning.
     Turner. Variational continual learning. In International Con-         arXiv preprint arXiv:2004.12908, 2020.
     ference on Learning Representations, 2018.                       [54] Yue Wu, Yinpeng Chen, Lijuan Wang, Yuancheng Ye,
[39] Pingbo Pan, Siddharth Swaroop, Alexander Immer, Runa                  Zicheng Liu, Yandong Guo, and Yun Fu. Large scale in-
     Eschenhagen, Richard E Turner, and Mohammad Emtiyaz                   cremental learning. In Proceedings of the IEEE/CVF Con-
     Khan. Continual deep learning by functional regularisation            ference on Computer Vision and Pattern Recognition, pages
     of memorable past. In Advances in Neural Information Pro-             374–382, 2019.
     cessing Systems, 2020.                                           [55] Friedemann Zenke, Ben Poole, and Surya Ganguli. Contin-
[40] Shaoning Pang, Seiichi Ozawa, and Nikola Kasabov. Incre-              ual learning through synaptic intelligence. In Proceedings
     mental linear discriminant analysis for classification of data        of the 34th International Conference on Machine Learning,
     streams. IEEE transactions on Systems, Man, and Cybernet-             pages 3987–3995, 2017.
     ics, part B (Cybernetics), 35(5):905–914, 2005.                  [56] Chen Zeno, Itay Golan, Elad Hoffer, and Daniel Soudry.
[41] German I Parisi, Ronald Kemker, Jose L Part, Christopher              Task agnostic continual learning using online variational
     Kanan, and Stefan Wermter. Continual lifelong learning with           bayes. arXiv preprint arXiv:1803.10123v3, 2019.
     neural networks: A review. Neural Networks, 113:54–71,
     2019.
[42] Ameya Prabhu, Philip HS Torr, and Puneet K Dokania.
     GDumb: A simple approach that questions our progress in
     continual learning. In European Conference on Computer
     Vision, pages 524–540. Springer, 2020.
[43] Sylvestre-Alvise Rebuffi, Alexander Kolesnikov, Georg
     Sperl, and Christoph H Lampert. iCaRL: Incremental clas-
     sifier and representation learning. In Proceedings of the
     IEEE conference on Computer Vision and Pattern Recogni-
     tion, pages 2001–2010, 2017.
[44] Danilo Rezende and Shakir Mohamed. Variational inference
     with normalizing flows. In International Conference on Ma-
     chine Learning, pages 1530–1538. PMLR, 2015.
[45] Anthony Robins. Catastrophic forgetting, rehearsal and
     pseudorehearsal. Connection Science, 7(2):123–146, 1995.
[46] David Rolnick, Arun Ahuja, Jonathan Schwarz, Timothy P
     Lillicrap, and Greg Wayne. Experience replay for contin-
     ual learning. In Advances in Neural Information Processing
     Systems, 2019.
[47] Joan Serra, Didac Suris, Marius Miron, and Alexandros
     Karatzoglou. Overcoming catastrophic forgetting with hard

---
