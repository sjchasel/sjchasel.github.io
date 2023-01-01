---
layout: post
title: 【论文阅读】Multimodal Molecule-Language Models
categories: 论文笔记
keywords: NLP, Bio
mathjax: true
---

[TOC]

# Paper list

[1] EMNLP21 Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries

code: https://github.com/cnedwards/text2mol

[2]NATURE COMMUNICATIONS22清华 A deep-learning system bridging molecule structure and biomedical text with comprehension comparable to human professionals

model(包含[code](https://github.com/thunlp/KV-PLM)): https://drive.google.com/drive/folders/1xig3-3JG63kR-Xqj1b9wkPEdxtfD_4IX

[3]EMNLP22 Translation between Molecules and Natural Language

code: https://github.com/blender-nlp/MolT5


[4]Arxiv22.9.12人大 A Molecular Multimodal Foundation Model Associating Molecule Graphs with Natural Language

code: https://github.com/BingSu12/MoMu 和 https://github.com/yangzhao1230/GraphTextRetrieval

[5]Arxiv22.12.21唐建 Multi-modal Molecule Structure-text Model for Text-based Retrieval and Editing

code: https://github.com/chao1224/MoleculeSTM  
还是空的


# Preliminaries

分子的表示方式可以从维度分成以下三类：

- 1D：以SMILES为代表，将分子化成一个字符串。比如
![](https://bkimg.cdn.bcebos.com/pic/fd039245d688d43f87944d97b052c51b0ef41bd58d0b?x-bce-process=image/watermark,image_d2F0ZXIvYmFpa2UxMTY=,g_7,xp_5,yp_5)
这种表示也能立即将NLP中的语言模型迁移过来使用，非常方便。存在的问题大概有，在分子中相邻、相连的原子，在SMILES中可能隔得很远，因此不利于学习分子中重要的官能团这种信息。
- 2D：将分子的原子作为节点，分子的化学键作为连接节点的边。采用图神经网络来学习分子的表示。但是在现实世界中其实没有化学键这种东西，化学键是人为建模出来的帮助我们理解分子的结构。所以这种表示存在的问题大概就是不太自然，无法直接学习分子的本质信息，并且没有三维坐标信息。
- 3D：分子是一个有三维信息的立体的东西，并且因为蛋白质和小分子的结合是三维形态的嵌入，所以考虑分子的三维信息在这类任务上很重要。可以用点云、密度图等来表示3D的分子，大家也会参考论文 E (n) equivariant graph neural  networks 使用E(3) Transformer来建模分子。对分子进行3D的建模时要注意的问题就是分子具有旋转和平移上的等变性。整个分子进行任意角度的平移、旋转，并不会对它的性质有什么改变。

# Motivation

将自然语言模态和分子模态结合的好处在于：
1. 药物的设计更加自由和直观。药学家只需要说出自己想要的性质，就能检索/生成/优化出想要的分子。
2. 数据库中存在大量人类对分子的标注文本，这些人类对分子的理解（以文本描述存在）可以作为外部知识，在对分子建模的时候可以结合这部信息，来学习更好的表示。
3. 将文本和分子结构联系起来，可以提高分子的可解释性（感觉就是在找词语和分子的官能团的共现关系）。比如研究者发现“pollutant（污染物）”这个自然语言总是会检索出拥有“F-C”子结构的分子[1]。

# Task and Evaluation

我将涉及到的任务分成检索、生成、分类三类进行介绍，并同时介绍这些任务的评价指标。

## retrieval task

论文[1]中，对检索任务的定义是给定一个文本query和一堆分子，去检索和文本描述最相关的分子（目标只有一个）。用Hits@1和MRR作为评价指标。作者在实验中也汇报了Mean Rank（MR）和Hits@10。

![论文1-task](/images/blog/text2mol.png)

---

论文[2]中，检索任务叫Versatile reading tasks，所用的数据集是PCdes。包含了PubChem数据库中15K个分子的SMILES和对它们的性质描述。训练集是10500条，验证集是1500条，测试集3000条。其中子任务2 CHEMIchoice如下图所示：

![multiplechoice task CHEMIchoice](/images/blog/KVPLM-reading.png)


它细分成两种子任务：
1. cross-information retrieval：双向的检索。从3k个分子/文本描述中检索出最相似的一个文本描述/分子。汇报了模型的 recall@20 得分。
2. match judging：也就是上图，multiplechoice task CHEMIchoice。从数据中采样四个描述句子，选择和分子最相似的一句。三个负例的选择会排除掉和正例特别相似的。

---

论文[4]中，这个任务包括：
1. Graph-text retrieval：给定分子图，检索出最相似的文本描述。
2. Text-graph retrieval：给定文本描述，检索出最相似的分子图。

完全follow论文[2]的设置，也用PCdes数据集。
他提到检索的范围也是follow[2]的，从随机sample的一个batch（64条数据）和整个测试集（3k）中进行检索。
在论文[2]中，文本描述是从文档中sample的句子，这篇论文将其称为sentence-level retrieval，并且也尝试了整个文本描述去检索的设置——paragraph-level retrieval。

---

论文[5]中检索任务是zero-shot场景下的structure-text retrieval task。

从DrugBank数据库构造了三个数据集，提取的字段有描述文本、pharmacodynamics field？和 anatomical therapeutic chemical (ATC) field？。

同样有双向的两个任务，从化学结构检索文本和从文本检索化学结构。检索的个数有4、10、20三个设置，比之前的论文少很多。

## generation task

### molecule caption[3,4]

如下图，类似image caption。输入分子（SMILES），输出对这个分子的描述。

![](/images/blog/molecule_caption.png)

因此评价指标可以采用BLEU、ROUGE、METEOR，比较目标文本的相似度。

除此之外，这个任务是论文[3]提出的，还能使用论文[2]的模型Text2Mol。这是一个检索排序模型，通过将分子和文本描述映射到同一个空间中，来计算它们的余弦相似度。因此可以用这个相似度来作为评价指标。我们希望生成描述和给定的分子之间的相似度尽量大。


### text-guided molecule generation[3]


论文[3,4]都是输入文本，生成符合描述的分子。

在论文[3]中，

![](/images/blog/text_molecule.png)

同样可以使用Text2Mol的相似度指标，希望生成的分子和给定的描述之间的相似度尽量大。

也可以计算生成分子和目标分子之间的相似度。比如：
- fingerprint metrics：对于分子生成，比较生成的分子和ground-truth分子指纹的古本相似度。
- SMILES metrics：SMILES之间的编辑距离和BLEU score。


### zero-shot text-to-graph molecule generation[4]


在论文[4]中，是在zero-shot场景下进行分子生成。
他认为他和论文[3]中text-guided molecule generation任务的不同在于：他提供的描述是一些指定的属性或条件，需要去生成满足条件的新分子。但是[3]的任务是在描述一个已经存在的分子。

举个例子，他的描述会更加抽象一些，比如“The molecule has high water solubility and barrier permeability with low toxicity”。但是论文[3]中的描述会对结构描述得很清楚，比如“The molecule is a member of the class of monohydroxy-1,4-benzoquinones that is 2-hydroxy-1,4-benzoquinone carrying an additional methyl substituent at position 5. It is a conjugate acid of a 2-oxido-5-methylquinone”。这种描述已经可以确定地定位一个具体的分子了，即“CC1=CC(=O)C(=O)C=C1O”。

（是不是因为没有这种抽象的文本与分子对数据，才只能做zero-shot？）

### Text-based Molecule Editing[5]

这个任务也做在zero-shot场景下。
随机从ZINC数据库中采200个分子，构造文本的prompt也作为输入。有四种prompt：
1. Single-objective editing：单目标优化，比如 “molecule with high solubility”、“molecule more like a drug”。
2. Multi-objective (compositionality) editing：多目标优化，比如 “molecule with high solubility and high permeability”。
3. Binding-affinity-based editing：应该是让分子针对某个蛋白质去优化，希望生成的分子能有更高的亲和性。比如 “This molecule is tested positive in an assay that are inhibitors and substrates of an enzyme protein. It uses molecular oxygen inserting one oxygen atom into a substrate, and reducing the second into a water molecule.”
4. Drug relevance editing：希望生成的分子和某些药物相似，比如“this molecule looks like Penicillin”。


## classification task

### 分子结构性质预测[2,4]

分子结构性质预测，有一个benchmark叫MoleculeNet，有四个分类任务：
1. BBBP：blood-brain barrier penetration dataset（血脑屏障穿透数据集），2053个小分子的二分类，判断目标为penetration/non-penetration。
2. SIDER：Side Effect Resource database of marketed drugs and adverse drug reactions（已上市药品及药物不良反应资源库）。1427个药物在27个器官上进行二分类。
3. Tox21：8014个分子，在12个目标上进行有毒或无毒的分类。
4. HIV：判断41127个分子抑制HIV病毒复制的能力是活跃还是不活跃。

论文[2]在这个benchmark上测试，论文[4,5]除了这四个数据集的任务外，还多测了四个：
1. ToxCast
2. ClinTox
3. MUX
4. BACE

### 命名实体识别[2]

所用数据集为BC5CDR，化学-疾病关系检测语料库，1500 abstracts 均分成train/dev/test。每个数据集中有超过5k个化学的mention，使用SciBERT用的版本做NER。这个数据集也可以做关系抽取。

### 关系抽取[2]

所用数据集为Chemprot，是判断小分子和蛋白质之间的反应关系，总共有13种关系，比如inhibitor, product-of。
- train：1020 abstracts (230k tokens) 
- dev：612 abstracts (110k tokens) 
- test：800 abstracts (180k tokens)

### chemical reaction classification task（few-shot）[2]

所用数据集为USPTO 1k TPL。


# Datasets

这里介绍论文中提出的一些新数据集

## ChEBI-20[1]

来源于PubChem、Chemical Entities of Biological Interest (ChEBI)。前者提供小分子，后者提供这些小分子的描述文本，总共有102,980个数据对。筛选出描述大于20个词的，能被RDKit识别的小分子，剩下33，010对。
按8/1/1划分数据集。在做排序时是对数据集中的所有分子排序。

论文[3]也使用这个数据集进行finetune/训练baseline。但是这个数据集中的描述总是在开头说这个分子的名字，比如“Rostratin D is an organic disulfide isolated from ...”，所以作者将开头的名字替换成了"The
molecule is [...]" (e.g., “The molecule is an organic disulfide isolated from ...”)。

## KV-PLM的预训练数据[2]

预训练数据来自S2orc数据库，这是一个包含英文学术论文PDF的数据库。爬取了30万论文，包含10亿token。75%的论文是药学、生物学、化学领域，其他的是计算机领域。只用摘要、引言和结论。用SciSpacy对其进行命名实体识别，找到其中的化合物名词。和PubChem数据库中分子的名字/同义词进行匹配，从而获得文献中小分子的SMILES。将SMILES插入到文本的名词后面。

怎么不公开！

## MoMu的模态对齐finetune数据[4]

从PubChem中搜集了前5w个分子的名字、同义词、SMILES。用OGB的smiles2graph将其转成图。用分子的名字作为query，从S2orc数据库中检索相关的描述文本。获得了15,613 graph-document数据对。
弱监督方式收集的数据，会比较粗糙。

## PubChemSTM[5]


之前数据集的构造方式是用同义词字段去检索文本，但PubChem里还有一个叫“string”的字段，提供了更全面的对分子的注释。作者利用这个字段构造了包含250K个分子和281K个structure-text数据对的PubChemSTM，大小是之前数据集的28倍多，是目前分子文本多模态领域目前最大的数据集。

# Papers

## Text2Mol

### Method

分子和文本分别有一个encoder，文本的用SciBERT，分子的尝试了MLP和GCN两种结构，似乎都是获得分子的摩根指纹，用Mol2vec获得embedding，但我不太清楚指纹的构建方式，之后仔细看看。

![](/images/blog/text2mol_model.png)

除了这两个encoder之外，还有一个transformer decoder，作用是将文本encoder的输出解码成为分子encoder的输出。
loss采用了论文 CLIP 中的 symmetric contrastive loss。

但是作者发现仅用这个loss没什么用，因为cross-modal attention结构的存在，使得模型无视文本信息。信息从一个encoder泄漏到另一个encoder。
所以作者修改了损失函数，加上一个二分类任务，判断一个分子和一个文本描述是否匹配，来强迫模型去学习文本信息。

（CLIP的loss还有意义吗？）


为了增强模型的可解释性，利用cross-modal attention模块的最后一层的attention值作为支持度，对分子的token和文本的token进行关联规则挖掘。对于每对（文本token，分子token）这样的规则，我们都得到了置信度值。

对于每个（文本，分子）对，它的得分一般采用topk个规则的置信度值的平均值表示，但是作者修改了一下，将平均值和余弦相似度做了个线性插值，即：

$$S(a,b)=\alpha cos(a,b) + (1-\alpha)AR(1,b)$$

其中 $\alpha\in [0,1]$，是从验证集上计算出来的超参数。

还做了个集成学习，取多个模型的平均值。

### Experiment


![](/images/blog/text2mol_exp.png)

主要是分析集成学习的效果。
MLP和GCN结构在对不同官能团排序的能力很不一样，同一个结构的不同初始参数也有不同。集成起来能达到最好的效果。


对于模型的缺陷以及下一步改进：
作者认为描述文本决定了这个模型的上限，虽然通过关联规则挖掘学习到了一些文本和分子结构的关系，但是这是不够的，模型仍然对一些基础知识无法识别。比如描述中出现“oxide” 意味着这个分子会有氧原子，而模型没有生成。因此融合更多外部知识可能可以继续提高模型的表现。


## KV-PLM

### Method

用SciBERT对模型进行初始化，加上了个分类层对下游任务进行适配。作者尝试了用SciBERT的词表对SMILES进行分词 和 用BPE算法对SMILES分词 两种设置。

![](/images/blog/KV-PLM.png)

然后就做mask language model的任务。

### Experiment


![](/images/blog/KVPLM-exp1.png)

对比了六种基于BERT的baseline。发现如下：
1. SciBERT没有在SMILES数据上预训练，在分子结构性质预测任务上的表现也不错。作者认为SMILES的语言模式和自然语言有一定联系。但是我感觉只是因为SciBERT的预训练数据里是包含SMILES的？SMILES不就是把分子构成树然后遍历吗，这似乎是确定的规则？
2. KV-PLM*（BPE对SMILES分词）版本没有直接拿SciBERT词表对SMILES分词的表现好。可能是因为在BPE的词表下，一些决定了分子性质的官能团被忽略了。
3. 单语言预训练很有效。


## molT5

### Method

![](/images/blog/molT5.png)


用T5.1.1作为初始化参数，采用replace corrupted spans任务做预训练。单语言的预训练数据分别为C4和Chemformer用的一亿条SMILES（竟然有这么大的数据集？？？！）。预训练阶段并没有做语言的对齐。

分子文本对数据用在两个下游任务的finetune中。

### Experiment

![](/images/blog/molT5-exp1.png)

![](/images/blog/molT5-exp2.png)

这个对比公平吗？所有的baseline要么是没有单语预训练，要么是参数更小。


1. RNN经常产生SMILES语法无效的序列；
2. 数据集太小，无法finetune transformer；
3. 作者在两个任务都列举了一堆case来说明自己模型的理解能力。


## MoMu

### Method

这篇文章的分子是2D graph形式。

和之前论文一样，为了弥补molecule-text pair的数据稀少的问题，作者采用了在单模态上预训练的模型初始化两个encoder。分子和文本的encoder分别是GIN（How powerful are graph neural networks?）和Sci-BERT（MoMu-S）或KV-PLM（MoMu-K）。然后再用自己收集的数据通过对比学习对模型进行finetune，实现模态的对齐。

对于graph-text对，采用graph数据增强的方式采样两个graph，从文档中随机采样两个不同的句子。用DeClip中对比学习的做法，并将 inter-modal 和 intra-modal contrastive learning 作为预训练的任务，loss是InfoNCE。

![](/images/blog/MoMu.png)

### Experiment

#### Cross-modality retrieval

![](/images/blog/MoMu-exp1.png)

a是正常的检索任务，旁边的d是zero-shot设置。不用PCdes数据集对MoMu finetune，直接使用预训练后的模型。并且因为PCdes中可能有数据被包含在预训练数据中，所以作者又收集了一些新的数据进行测试。

MoMu-K和MoMu-S的区别在于文本的encoder初始化参数是SciBERT还是KV-PLM*。但后者并没有比前者更好，说明从SMILES这样的文本序列中学习到的分子的结构信息很难迁移到对分子图的编码上。

#### molecule caption

作者对这个任务的实验设置是，将MoMu模型对分子图编码的特征作为一个额外的特征向量给MolT5模型。  
效果好不是很正常吗？要是模型训得好，这额外的特征没有用的话线性层的权重为0不就行了。  


![](/images/blog/MoMu-exp2.png)


#### Zero-shot text-to-graph molecule generation


采用MolFlow作为分子的生成器，预训练好的MoMu和MolFlow（在ZINC250K上预训练）的参数都不会再调整。

从高斯分布中采样q，用MolFlow生成分子图。将这个分子图输入给MoMu的molecule encoder，文本描述输给MoMu的text encoder，获得二者的特征向量，最大化两个特征向量之间的余弦相似度。优化后的q给MolFlow生成最后的分子。

MoMu可以和任何能够进行反向梯度传播的分子生成模型兼容，能够搜索的分子空间取决于这个分子生成模型。MoMu去寻找和文本语义相似的新分子。

这个实验也只是展示了一些case，输入的文本还是挺神奇（抽象的）。

![](/images/blog/MoMu-exp3.png)


作者对实验结果的分析如下：
1. MolT5只能生成多个分子，但MoMu可以采样出很多不同的分子；
2. MoMu处理抽象描述的能力强，MolT5经过了数据的finetune，只能处理那种非常具体的描述的文本。MoMu认为“美丽的”分子局部对称和拉伸，认为“万能的”分子有很多基团。认为“奇怪的”分子有乱七八糟的支链。
我觉得这是作者收集的弱监督数据里，包含人类对分子的一些主观评价吧。而且这些评价对新药发现有很大的意义吗。。。
3. MoMu能根据特定的性质描述文本生成想要的分子。他展示出来的case确实是这样，但是MolT5也有好的case。不是很明白这么多case study的意义。


#### Molecule property prediction

![](/images/blog/MoMu-exp4.png)

在不同数据集上进行finetune，用molecule encoder编码出的特征进行性质预测。  
用t-SNE对finetune前后的分子表示进行可视化，finetune后MoMu的分子表示是更加分散的。



## MoleculeSTM

### Method

![](/images/blog/MoleculeSTM.png)

这个模型的有效性来源于两点：
1. Open vocabulary：模型的文本不受限于对一些分子的描述，而是认识更多的专有名词。所以有能力面对一些需要挖掘新关系的场景。
2. Compositionality：在多目标先导化合物优化上，以往的方法需要大型的数据库用于检索，或是设计多个分类器去完成多个目标。但通过自然语言，可以很方便地组合多个目标，语言模型也能够理解其中复杂的语义。比如只需要设计这样一个文本的prompt“molecule is soluble in water and has high permeability”，就能表达两种目标的需求。

模型的结构：
1. Chemical structure branch：即考虑了SMILES又考虑了分子图的表示。对于SMILES，用预训练encoder MegaMolBART；对于分子图，用 GraphMVP预训练的graph isomorphism network (GIN)，这个模型里也包含了一些三维空间信息。（然后怎么结合，似乎没看见哪里有写）
2. Textual description branch：SciBERT。


对比学习：  
结合了EBM-NCE和InfoNCE


# 感想

## 任务

分子和文本的结合感觉从数据的本质上来看是多模态任务，这两种模态的信息实在是太不对齐了。文本描述可以有很多种，就像对图像的描述，每个人写的都不同。人类描写的文本一定无法将分子结构蕴含的信息全面表达，但又可以直白地说出结构的一些性质。  

但当分子用SMILES表示时，感觉可以参考多语言的模型。论文[3]也说了他们的模型像mBART。  

当分子用2D图或3D的形式来表示，才更像多模态任务。只能用不同的encoder去编码分子和文本这两种模态，再思考如何对齐和融合。

类似文本图像多模态的任务需要对齐文本的短语描述和图像中的一些实体，这个任务也在对齐描述性质的词/短语以及分子的官能团。各篇论文的case study也都在展示模型这方面的能力：


![](/images/blog/KVPLM-case.png)

![](/images/blog/text2mol-case.png)


## 模型

能不能直接将一些对齐的做法迁移过来。比如mask一个模态，用其他模态的信息去预测。不过论文[2]算是也在利用一个模态预测另一个模态吧，真的有很大用处吗？想分析一下每个token的attention大小都是多少。自从transformer出现后也是有一堆文章去分析attention都在关注序列的哪些地方，然后发现一些不合理之处。

## 数据

1. 【数据增强】对齐的数据还是太少了，无法离开大型的预训练模型。IJCAI22一篇做分子性质预测的任务里第一次提出分子数据增强的方法，简单看了下增强的方法看起来不是很靠谱，准备把整篇论文一起看看。

2. 【引入更多外部知识】这些文章中只做了粗粒度数据对的对齐，能不能类似visual grounding的做法做多模态中细粒度的对齐。比如用Text2Mol中通过关联规则挖掘出来的，分子官能团与文本的对应关系作为先验知识，或者不知道有没有数据库也提供这样的知识。


3. 【数据形式】这些文章做粗粒度的对齐的同时，都展示了细粒度的对齐结果。引入文本，其实还是想将文本的一些词语/短语和决定分子这些性质的官能团联系起来，然后组合起来去理解整个的语义。所以用SMILES这样的1D形式是感觉天然地不合适这种任务。因为序列会把官能团拆分开来。最起码也得在分子图上划分子图吧。

