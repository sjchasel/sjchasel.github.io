---
layout: post
title: 【论文阅读】Molecule Representation Learning
categories: 论文笔记
keywords: Bio
mathjax: true
---



[TOC]

# 动机

以研发新药为例，传统方法想要研发一种药物出来需要经过很多流程。挑选分子、逐步优化、在生物上做实验等等。这个过程非常漫长，可以达到十年以上。并且每个环节都面临着失败的风险，也非常昂贵。

深度学习能将所有事物进行特征化，用数据驱动的方式去学习一些复杂的过程。因此如果能应用在药物行业，相比于传统的湿实验，可以大大降低成本。除此之外，药学家通过经验人工设计药物，他能想到的分子也有限。而计算机可以帮我们去探索更广阔的化学空间。

如果能将蛋白质和分子特征化，我们还能以数据驱动的方式去做很多任务。这些任务如果用传统方法的话可能都需要大量成本。比如仅仅输入一个分子的序列/图的表示，就能快速得到一个衡量它的成药性、可合成性、可溶性这些性质的数值。


# 分子表示的不同形式

在Multimodal Molecule-Language Models里写了，这里就不写了。接下来就按分子表示的维度分类各个工作，从1D到3D介绍如下。

# 基于1D序列的工作



## X-MOL

Science Bulletin22 X-MOL- large-scale pre-training for molecular understanding and diverse molecular analysis


### Motivation

motivation非常简单粗暴，就是想知道用大数据、大模型、大算力，仅靠分子的SMILES能做到什么水平。


![](/images/blog/xmol.png)

### Method

#### pretrain


![](/images/blog/xmol-pre.png)


模型采用了类似UniLM的设计，用一个encoder来进行序列的生成。

#### finetune


性质预测就是加上分类头了，除了性质预测外，还做了分子优化的任务。

![](/images/blog/xmol-opt.png)

首先要处理出用来finetune的数据，这需要计算SMILES之间的相似度和计算每个分子的QED等要优化的值。将相似，并且QED不同的分子两两配对。那么输入给模型的就是QED比较小的，需要预测另一个更大的。因为这两个分子本来就是相似的，那么其实就相当于在原分子的基础上修改了一些结构，让它的QED提高。


#### knowledge embedding

探究了一下，模型已经能学好SMILES的规则了，如果加上SMILES以外的信息，能否提升模型的性能。这里设计了三种知识，通过类似position embedding的方式将知识融入SMILES中。

1. Link embedding
因为SMILES这种表示方式，有可能在分子中相邻的原子在字符串中相隔很远。因此作者想将原子之间的连接信息加入。那么link embedding就表示当前原子连接的前一个原子是谁，如果它是第一个原子，那么就指向自己。
2. Ring embedding
似乎表示当前原子是否为开环原子或闭环原子，如果是开环原子，那么它所对应的ring embedding的值就是闭环原子。
3. Type embedding 
在SMILES中，字母表示原子类型，-、=、#表示化学键，（、）、@表示结构，因此加入type embedding指示每个字符都是什么类型。

然而最后的结果是不管加上哪种知识，模型的效果都会降低。因此作者得出的结论是“SMILES is all you need”。

![](/images/blog/xmol-embed.png)

## MM-Deacon

ACL22 Multilingual Molecular Representation Learning via Contrastive Pre-training

### Motivation

之前介绍分子的1D表征只展示了SMILES的形式，但除了SMILES，还有其他一些使用没那么广泛的序列表示，比如SELFILES、IUPAC和InChI。这篇文章认为不同的序列用不同方式描述这个分子，可以类比成多种语言。因此想用多语言的学习方式来利用不同的序列表示学习这个分子。

它使用的是SMILES和IUPAC，如下图所示。SMILES表示的是原子和键级别的信息。而IUPAC包含了丰富的先验知识，因为它表示的是分子由哪些官能团组成。官能团就是这些具有一样颜色的部分，由几个相邻的原子组成，它们组合起来才具有了某些性质。比如这个黄色的就是我们熟悉的苯环、序列中直接就写出benzene/苯。

![](/images/blog/mmdeacon1.png)

### Method

#### Pretrain

首先要对两种序列分词，SMILES一般都是使用BPE去构建词表，对于IUPAC，用了前人工作中提出的正则表达式规则。模型非常简单，用两个transformer分别编码SMILES和IUPAC。两种语言的特征经过平均池化，投影到同一个空间中。用对比学习去拉近同一个分子的两种表示，推远不同分子的表示。没有自己构建正负样本对，只是用同一个batch中的N个互相配对，因此有N^2个样本，其中N个正样本。

预训练数据是来自PubChem的10M个分子。

---

其实之前也有一些工作是使用分子的不同语言的，但它们都是去互相翻译，比如：
+ Stout: Smiles to iupac names using neural machine translation
+ Translating the InChI adapting neural machine translation to predict IUPAC names from a chemical identifier
+ Transformer-based artificial neural networks for the conversion between chemical notations

#### Downstream

这个模型有两种应用场景，如下图：

1. MM-Deacon fine-tuning
也就是加上额外的分类/回归头在任务数据集上finetune。
2. MM-Deacon fingerprint
这个模型借助两种分子的表示可以提取出独特的特征，既包含了分子最原始的结构信息，又包含了官能团这样的先验知识。我们可以把这个模型提取出的分子表示叫做MM-Deacon指纹。可以对分子做一些聚类、相似性检索的任务。当然也可以加上任务相关的分类头finetune。

![](/images/blog/mmdeacon2.png)

### Experiment

这篇包含的实验有：
1. MoleculeNet

![](/images/blog/mmdeacon3.png)


MM-Deacon只用了SMILES的那边，没有用IUPAC。
其中MLM-CLS是ChemRoberta，不过用相同规模的SMILES预训练。

2. zero-shot cross-lingual retrieval

做了IUPAC和SMILES的双向检索。这个就是汇报了一下自己的Recall和一些例子，也没有什么baseline可以对比。

3. DDI

将配对药物的mm-deacon指纹拼接过mlp进行二分类。

![](/images/blog/mmdeacon-ddi.png)


# 基于2D图的工作

## GROVER

NIPS20 Self-Supervised Graph Transformer on Large-Scale Molecular Data

本文利用无标注的分子图数据，设计了两种自监督任务，涵盖了节点、边、图三个级别，来学习分子的结构和语义信息。为了设计更有表现力的模型，借鉴Transformer的设计提出了一种新的图神经网络。这个模型的大小是100 million parameters（也有12M和40+M的），数据规模是10 million unlabelled molecules。


### Method

#### 模型结构

GROVER是一个很对称的结构，如图：

![](/images/blog/grover.png)

它由两个GNN Transformer（GTransformer）组成，一个是graph中的node视图，一个是edge视图。接下来以node GNN transformer为例讲一下细节。  
模型结构上，作者主要有三个创新：
1. 将GNN与Transformer结合。
作者将这个模型称为bi-level（双层）信息提取框架。因为输入数据首先经过图神经网络编码，它擅长于捕捉图中的局部结构的信息。由于Transformer实际上可以看成是在全连接图上的GAT模型，所以从GNN出来的特征再经过Transformer的编码，能够捕捉全局的信息。
2. 使用长期残差连接。
有两个好处：
    1. 缓解梯度消失问题；
    2. 缓解图神经网络中的over-smoothing问题。
3. 设计了dyMPN。
传统的图神经网络中有两个超参数：（1）迭代次数/层数；（2）每层的跳（hop）数。跳数就是在进行消息传递的过程中，最多有几阶的邻居的信息可以传过来。它会影响模型的泛化性，因此作者不将这个参数预定义固定住，因此在每层都是从一个截断的正态分布中去采样。这样就实现了动态的消息传递——Dynamic Message Passing networks（dyMPN）。


#### 自监督任务

作者设计了两个自监督任务：
1. Contextual Property Prediction

node端和edge端都有，都是预测自己周围有什么东西。作者将周围的信息构建成字符串形式，然后作为一个类别，这个任务就成了分类任务。比如以红色的碳原子节点为预测目标时，需要预测的是这个原子周围的原子类型、连接这个原子的化学键、以及它们相应的数量。当只考虑1阶邻居的时候，目标就是“C_N-DOUBLE1_O-SINGLE1”，各项是按字母表顺序排列。有点像在聚类，有相似周围结构的原子特征被聚到一起。


2. Graph-level Motif Prediction


### Experiment 

在11个分子性质预测的数据集上做了实验

![](/images/blog/grover-exp.png)

## MolCLR

Nature Machine Intelligence22 Molecular Contrastive Learning of Representations via Graph Neural Networks

数据规模：大约10M




# 基于3D构象的工作

## 一些建模3D分子的方法

### 密度图

### 点云

### G-SchNet

## Uni-Mol

### Motivation

描述一个分子，知道原子类型和相应的位置就是这个分子所包含的所有信息了。因此我们可以认为分子的性质就是由原子类型和其位置决定。使用原子类型+原子坐标两种特征来建模分子，这样避免引入官能团等人为定义的概念，也就不会引入其中的偏差。并且能够捕捉到除了化学键以外的更多原子间的作用力。

![](/images/blog/unimol.png)

这是第一篇将分子完全用直接的3D信息来表示的预训练模型。这个通用的框架可以泛化到多种任务上。

### Method

![](/images/blog/unimol-model.png)

#### 原子对表示

我们知道，Transformer中的绝对位置编码，是不具有平移不变性的。但相对位置编码捕捉的是两两位置之间的绝对距离，因此整个序列平移了，特征也不会改变。受到相对位置编码的启发，这里建模的也是原子对的表示。我们输入的信息是每个原子的三维坐标，由此可以计算原子对之间的相对距离。但跟序列用整数表示位置和距离不同，原子坐标是3维空间中的连续变量。所以我们用欧式距离来表示原子对之间的相对距离，然后通过仿射变换和高斯核函数将它们映射到特征空间中。

放射变换的作用就是，在对原子对距离进行变换的时候，根据原子对类型的不同有不同的变换。


#### 编码

对于一个分子，输入的数据有原子类型和原子坐标。模型会将其编码成原子类型表示和原子对表示。模型内部对原子对表示编码的过程就是在变换坐标的特征，能够保持分子的旋转平移不变性。

为了更新坐标的表示，也就是需要更新原子对的特征，我们将信息从atom传到pair。

$$
\boldsymbol{q}_{i j}^{l+1}=\boldsymbol{q}_{i j}^l+\left\{\frac{\boldsymbol{Q}_i^{l, h}\left(\boldsymbol{K}_j^{l, h}\right)^T}{\sqrt{d}} \mid h \in[1, H]\right\}
$$


其中 $q_{ij}^l$ 表示第l层原子i、j的原子对表示，H是attention头的个数。因此这个公式其实就是原子i和原子j的特征相乘去更新它们的原子对表示。

也想用空间信息去更新原子表示，所以还有一个pair到atom的信息传递过程。

$$
\operatorname{Attention}\left(\boldsymbol{Q}_i^{l, h}, \boldsymbol{K}_j^{l, h}, \boldsymbol{V}_j^{l, h}\right)=\operatorname{softmax}\left(\frac{\boldsymbol{Q}_i^{l, h}\left(\boldsymbol{K}_j^{l, h}\right)^T}{\sqrt{d}}+\boldsymbol{q}_{i j}^{l-1, h}\right) \boldsymbol{V}_j^{l, h}
$$

和正常的attention公式不一样的地方就在于，qk相乘之后还加上了这个原子对的特征表示，再和v相乘。也就是transformer的相对位置编码的处理过程。


#### 预测坐标

以上过程可以学到一个比较好的分子的3D表示，但是建模3D分子的平移旋转不变性的难点在如何输出一个3D的分子。首先介绍分子的平移旋转的不变性。

$$
\begin{aligned}
\vec{x} & =\operatorname{flatten}(\mathbf{X}) \\
\hat{E} & =\vec{x} \cdot \vec{w}+b
\end{aligned}
$$


---

直观上来看，cij表示某条边的信息。

# 利用多种表示的工作

## DMP

### Motivation

这篇论文感觉idea是来自一个比较有名的对比学习模型——BYOL。不过作者说观察到：GNN 擅长处理环多的分子，不擅长较长的分子。Transformer 擅长处理较长的分子，不擅长环多的分子。因此想将二者结合。

![](/images/blog/DMP-motivation.png)

### BYOL

先在这里详细讲讲BYOL吧。BYOL最大的创新是没有负样本或者聚类中心这样比较明确的对比的东西来做对比，自己跟自己对比就取得了很好的效果。BYOL的模型图如下（这篇DMP的配图几乎是一模一样了）：

![](/images/blog/BYOL.png)

上下两种颜色分别为一个编码器，它们的架构相同，但参数不同。上面的参数是正常更新的，下面的参数是跟MoCo一样进行动量更新的，也就是一个动量编码器。编码器最后跟SimCLR一样，接了一个projection head。如果是传统的方法，在编码出了特征之后，会让这两个来自同一个样本的特征尽量接近。但BYOL的做法是，继续加了一个predictor（mlp），通过这个模块又产生了新的特征，并且希望这个新的特征和下面的动量编码器得到的特征尽可能一样。也就是从一个匹配的任务转换成了自己预测自己的任务（一个视角的特征预测另一个视角的特征）。直接计算mse loss。

原样本x已经产生了两种view v和v'，我们可以直接将二者的输入反转，图中是拿v预测v'，也可以用v'预测v，因此最终的loss是两个对称的相加。


我在想模型为什么不会输出一样的特征作为捷径？看见网上的解释是因为只有一支使用了preidtcor，这样不对称的结构使得模型输出不同视角的特征。
感觉也有点像autoencoder。区别大概在于BYOL想让学到的特征能预测另一种特征，这样在不同视角中共享的特征才是更本质和鲁棒的？autoencoder只是学习到和原来样本尽量相似的特征。

### Method


回到DMP。DMP并不是从一个样本去投影出两个视角的特征，而是将一个分子的两种表示（SMILES与分子图）作为两个视角。除此之外和BYOL的区别大概就在于还加上了对分子图和SMILES的MLM任务。这张图其实是把BYOL中v预测v'，和v'预测v的两个流画在了一起。经过projector投影后的特征是p，再经过predictor投影后的特征是q。DMP的loss由以下部分组成：
1. MLM on Transformer
2. MLM on GNN
3. Dual-view consistency：q_smiles和p_graph尽量相似+q_graph和p_smiles尽量相似

![](/images/blog/DMP.png)

### Limitations


这篇论文也讨论了不足之处。想起之前看的一篇多模态的工作的出发点也是两种不同模型作为分支，训练开销太大，他发现Transformer的强大使得它能够处理各种模态的数据，self- attention的参数在不同模态之间可以共享，只要区分ffn的参数就行。那么就不用像这篇论文一样用一个GNN和一个Transformer了。但是作者的出发点是GNN和Transformer的模型结构决定了它们擅长处理的分子不同，所以这二者确实很难舍弃。。。或许由回到了GROVER，将GNN和Transformer结合？但是不能像它一样直接上两个一样结构的模型，得设计一下哪些参数共享，哪些不能共享。

作者提到可以动态地选择用哪个分支，感觉也是挺有必要的。因为一些性质可能只需要一种信息，多种信息反而成为了噪声。不过如果是分子和文本这样的多模态，文本端大概只是起到了一种辅助作用？直接用分子端就行。

### Experiment

作者做了两种实验。

#### 分子性质预测

数据规模也是10M。不过也试了下100M的，表明数据更多还会继续提升。

![](/images/blog/DMP-exp.png)


这里面效果最好的DMP_TF表示是用Transformer得到的特征进行finetune。

#### 分子生成

这个任务是retrosynthesis，定义为：给定一个无法直接获得的分子，我们想要获得能够生成它的几个分子。看见作者的评价标准是通过beam search获得多个分子，然后计算topk的准确率。这个任务也太像kp了吧，然后作者用的是one2one。（能上one2set做这个任务吗）

Transformer端的：

![](/images/blog/DMP-exp4.png)



GNN端的：

![](/images/blog/DMP-exp2.png)





## GraphMVP


### Motivation


分子的3D信息对性质预测很重要，但缺少大量数据。那么可以利用好现有数据，用2D+3D的数据预训练，finetune时可以只用2D的数据。预训练时学到的特征应使两种视角的互信息最大化，那么在finetune时即使没有3D数据，特征也包含了隐式的3D信息。





### Method


本文的方法利用两种视角的分子表示，学习信息更加丰富的特征。2D数据聚焦于分子的拓扑结构，3D数据则聚焦于分子的能量和空间结构。并采用了两个互补的自监督任务：
1. 对比学习任务 - 不同分子之间（inter-molecule level）
2. 生成式任务 -  同一分子的各部分之间（intra-molecule level）
为2D和3D分子提出新的重建损失VRR。

模型框架如下：

![](/images/blog/graphmvp.png)

其他的都没啥好说的，就是它新提出的VRR大概要解释一下。VRR的公式如下：

$$
\begin{aligned}
\mathcal{L}_{\mathrm{G}}=\mathcal{L}_{\mathrm{VRR}}= & \frac{1}{2}\left[\mathbb{E}_{q\left(\boldsymbol{z}_{\boldsymbol{x}} \mid \boldsymbol{x}\right)}\left[\left\|q_{\boldsymbol{x}}\left(\boldsymbol{z}_{\boldsymbol{x}}\right)-\mathrm{SG}\left(h_{\boldsymbol{y}}\right)\right\|^2\right]+\mathbb{E}_{q\left(\boldsymbol{z}_{\boldsymbol{y}} \mid \boldsymbol{y}\right)}\left[\left\|q_{\boldsymbol{y}}\left(\boldsymbol{z}_{\boldsymbol{y}}\right)-\mathrm{SG}\left(h_{\boldsymbol{x}}\right)\right\|_2^2\right]\right] \\
& +\frac{\beta}{2} \cdot\left[K L\left(q\left(\boldsymbol{z}_{\boldsymbol{x}} \mid \boldsymbol{x}\right) \| p\left(\boldsymbol{z}_{\boldsymbol{x}}\right)\right)+K L\left(q\left(\boldsymbol{z}_{\boldsymbol{y}} \mid \boldsymbol{y}\right) \| p\left(\boldsymbol{z}_{\boldsymbol{y}}\right)\right)\right]
\end{aligned}
$$

看公式的第一部分。  
相比于普通的重建损失，会将输入的东西的特征解码，这里并不会对2D的特征解码，而是直接将它映射到3D空间。其实就是在用2D信息预测3D信息，这不就是BYOL的loss吗？而上面那篇论文只计算一个余弦相似度。  
所以它也引了BYOL，也引了SimSiam。终于懂了SG的作用，就是防止模型坍塌。BYOL使用动量更新的编码器来防止模型坍塌，SimSiam发现不需要动量编码器，直接共享参数，有SG和predictor来保持不对称的结构就可以。  
第二部分是计算两个东西的KL散度。  
所以这里的重建损失其实就是结合了mse和KL散度。


