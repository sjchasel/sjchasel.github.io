---
layout: post
title: 【论文合集】关键短语抽取论文整理
categories: ["论文笔记", "NLP"]
# key: kp_papers
---

[TOC]

# 基于统计的方法

## KP-Miner 2009

论文：El-Beltagy S R, Rafea A. KP-Miner: A Keyphrase Extraction System for English and Arabic Documents[J]. Information Systems, 2009, 34(1): 132-144.


一些理论依据

1. 因为复合短语的出现频率会比单词少，需要一个因子来提升对复合短语的提取
2. 需要加上语言学特征，不然谁出现的多谁就是关键短语了
3. TF-IDF倾向于提取特定领域的关键词，但通用的短语提取器也是需要的
4. 越早出现的短语可能越重要。但出现次数多到一定程度时这个词就不是关键短语了。



方法

1. Step 1: candidate keyphrase selection

   根据一些规则和停用词

   + least allowable seen frequency (lasf) factor：出现次数超过这个数才能作是算候选短语。根据文档长度和语种有差别。

   + cutoff constant (CutOff )：如果出现的太晚，不会是关键短语。只对某些领域的文档有效，

2. Step 2: candidate keyphrases weight calculation
+ boosting factor: $ B_{\mathrm{d}}=\left| N_{\mathrm{d}}\right| /\left(\left|P_{\mathrm{d}}\right| \alpha\right) $
有最大值；$N_d$是文档$d$中所有候选关键短语的个数；$|P_d|$是文档中长度超过1的候选短语的个数。$\alpha$是一个使其最终的值别太大的常数。

+ term position associated factor：可选 是否考虑位置

+ final score：$ w_{ij}=tf_{ij}*idf*B_i*P_f $
其中 $ tf_{ij} $ 是术语 $ t_j $ 在文档$D_i$中的出现频率，$ idf $ 是 $ log_2\frac{N}{n} $ , $ N $ 是语料库中文档的数量, $ n $ 是术语 $ t_j $ 至少出现了一次的文档数量。

3. 排序

## RAKE 2010 简单快速

[RAKE]Rose S, Engel D, Cramer N, et al. Automatic Keyword Extraction from Individual Documents[A]// Text Mining: Applications and Theory[M]. Wiley, 2010, 1: 1-20. 

https://medium.datadriveninvestor.com/rake-rapid-automatic-keyword-extraction-algorithm-f4ec17b2886c

https://blog.csdn.net/qq_29003925/article/details/80943689

无监督、领域独立、语言独立的方法，从单个文档中提取关键短语。据观察，关键短语经常包含多个单词，但很少包含标准的标点符号或停止词，如“and”，“the”，“and of”，或其他词汇意义最小的单词。

 RAKE接收**一个停止单词列表、一组短语分隔符和一组单词分隔符**。然后，它使用停止词和短语分隔符将文档文本划分为候选关键字，这些关键字是出现在文本中的内容词序列。在这些候选关键字中出现的单词是有意义的，它允许我们在不使用任意大小的滑动窗口的情况下识别单词的出现。例如，如果我们将下面的段落作为RAKE模型的输入

![RAKE-example](https://miro.medium.com/max/1400/1*shZJ_e3_HUvJsZFgRIta5g.png)

系统会将上面的段落分解成如下所示的关键字列表：

![img](https://miro.medium.com/max/1400/1*dpSB0NYJsyaEZvHUhy8Rtw.png)

下一步，考虑窗口长度为关键字中的单词数，我们在该上下文中构建单词级共现矩阵，并使用degree(单词)/frequency(单词)计算单个单词得分，然后将单词得分相加得到短语得分。

## YAKE 2018

[YAKE] Campos R, Mangaravite V, Pasquali A, et al. A Text Feature Based Automatic Keyword Extraction Method for Single Documents[A]//Proceedings of the 40th European Conference on IR Research. 2018: 684-691. 

---

只需要一个停止词列表，语言无关的。整个算法有4个步骤：

1. 预处理出候选关键短语。用segtok分句；我们可以再根据大小为2、3、4分成这个长度的短语。

2. 抽取特征。总共有五个特征

   1. Casing：首字母是否大写
   2. Position of Word in the text：出现在文档前面的词更重要
   3. Word frequency：词频
   4. Term Relatedness to Context：如果一个词出现的环境越多样化，那么这个词就越可能是普通的词。比如停用词出现在各种位置。
   5. Term Different Sentence：经常出现在不同句子中的单词得分较高

3. 打分

   $$S(t)=(Trel*Tposition)/Tcase+((Tnorm/Trel)+(Sentence/Trel))$$

4. 去重。基于Levenshtein距离的重复数据删除方案：如果一个词与已经选择的词之间的Levenshtein距离较小，则不选择该词。

5. 最终排序。分数越小越好。top-k个

## Won's method

[]Won M, Martins B, Raimundo F. Automatic Extraction of Relevant Keyphrases for the Study of Issue Competition[C]// Proceedings of the 20th International Conference on Computational Linguistics and Intelligent Text Processing. 2019. 

---

1. Candidate identifification

2. Candidate scoring

   1. Term Frequency (tf)

   2. Inverse Document Frequency (idf)

   3. Relative First Occurrence (rfo)

      公式中$a\in [0,1]$，表示第一次出现的位置，可能是在文档的百分之几处？第一次出现得越前，这个值就越小，1-a越大，rfo越大。词频越大，rfo也越大。

   4. Length (len)

$$\begin{aligned}
t f\left(w_{1} \ldots w_{n}\right) &=\sum_{i=1}^{n} f r\left(w_{i}, d\right) \\
t f-i d f\left(w_{1} \ldots w_{n}\right) &=\sum_{i=1}^{n} f r\left(w_{i}, d\right) \times \log \left(\frac{N}{1+\left|d \in D: w_{i} \in d\right|}\right.\\
r f o\left(w_{1} \ldots w_{n}\right) &=(1-a)^{f r\left(w_{1} \ldots w_{n}, d\right)} \\
\operatorname{len}\left(w_{1} \ldots w_{n}\right) &=\left\{\begin{array}{l}
1: n=1 \\
2: n \geq 2
\end{array}\right.\\
H_{1} &=t f \times r f o \times l e n \\
H_{2} &=t f-i d f \times r f o \times l e n
\end{aligned}$$

H1没有逆文档频率，因此是只在整个文档上计算。H2考虑了整个语料库的信息。

3. Top n-rank candidates

   根据文档长度决定要多少个短语

   $n_{keys}=2.5\times\log_{10}(doc size)$

# 基于图的方法

## 未用图传播算法

特点：构图，聚类

### KeyGraph 1998

KeyGraph首次将文档构图来提取关键短语。预先定义一组候选词或短语作为图上的节点，出现在同一个句子中的节点相连，边的权值为共现次数。识别图上的联通子图，表示文档的一个主要的concept。运用统计方法提取和文档cencept相关的短语。

---

KeyGraph选择出现频率较高的短语或词作为图的节点，按照共现关系连边。

1. 图的构建

   1. 出现频率在前30%的词或短语，称为term，作为图的节点
   2. 当两个节点在同一个句子中出现时才会连边，权重为共现次数
   3. 可以将图上的最大联通子图作为一类

2. 计算文档中每个term的key value

   $$key(w)=1-\sum_{g\in G}(1-\frac{based(w,g)}{neighbors(g)})$$

   其中，$based(w,g)$表示term w和类别g中的所有term在同一个句子中的共现次数；$neighbors(g)$表示包含了类别g中的term的句子中的term数量。二者的除数表示term w和类别g的相关程度。

   $key(w)$表示term w和所有类别的相关程度。

3. 第二步的前K个term如果没出现在之前构的图里，就加入。这一步是考虑那些出现频率不高，但却对文档很重要的词或短语。

   每个term的重要性是与之相连的边的权重的和。提取前K个term作为keywords（keyphrases）

### 2009 WWW

此算法将短语作为图上节点，边的权值代表基于维基百科的统计信息计算的两个短语的语义相似度（由两个短语所链接的维基百科页面共享的链接数决定）。在图上运用Girvan–Newman algorithm，根据**图的稠密程度**划分出多个子图。每个子图的重要性由子图中所包含的节点和边的统计信息计算得到，排名最高的几个子图可以认为是和文档最相关的几个主题。这些主题中包含的短语作为文档的关键短语。

---

1. candidate terms extraction

   n-gram

   用维基百科的信息修改词干，规范形式。

2. word sense disambiguation

   维基百科里对每个词会列出所有可能的含义，作者通过前人的方法为每个候选关键短语找到最相似的那个意思，每个短语就有了一个维基百科的页面。

3. building semantic graph

   构图，节点是短语，边有权值，是两个短语的语义相似度。

4. discovering community structure of the semantic graph

   用来自M. E. J. Newman and M. Girvan. Finding and evaluating community structure in networks. Physical Review E, 69:026113, 2004.（Girvan Newman algorithm）的方法将图划分成多个子图。

5. selecting valuable communities

   基于community的：

   1. density：边的平均权值

   2. informativeness：短语的平均keyphraseness measure。这个东西的计算方法是来自O. Medelyan, I. H. Witten, and D. Milne. Topic indexing with Wikipedia. In Wikipedia and AI workshop at the AAAI-08 Conference (WikiAI08), Chicago, US, 2008.

      $$ \text { Keyphraseness }(a) \approx \frac{\operatorname{count}\left(D_{\text {Link }}\right)}{\operatorname{count}\left(D_{a}\right)}$$

      分子是这个词作为一个链接出现的维基百科的文章数，分母是正文里出现了这个词的维基百科的文章数。

   将这两个指标相乘对community排序。最终的结果都会有一个明显的下降的地方，我们可以选出下降前的几个community。

   ![截屏2022-04-03 下午3.39.17.png](https://s2.loli.net/2022/04/03/UkEsg6XrfNGuvaA.png)

   

### KeyCluster 2009

[KeyCluster] Liu Z, Li P, Zheng Y, et al. Clustering to Find Exemplar Terms for Keyphrase Extraction[C]//Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing. 2009: 257-266.

---

KeyCluster获得一系列候选词，基于共现频率或维基百科的信息计算两个词之间的语义相关性，作为两个词之间的距离。运用聚类算法将词聚类，获得每个类别的中心词，将这些中心词扩充成多个名词短语作为文章的关键短语。

---

1. Candidate term selection

   去停用词等规则

   单个词作为候选term，用于后续聚类

2. Calculating term relatedness

   1. Cooccurrence-based Term Relatedness

   2. Wikipedia-based Term Relatedness

      来自2007年的IJCAI《Computing Semantic Relatedness using Wikipedia-based Explicit Semantic Analysis》

3. Term clustering

4. From exemplar terms to keyphrases

   聚类中心作为seed term（单个词）

   (JJ) ∗ (NN|NNS|NNP)+

   选择一个或多个包含seed term的短语

## TextRank及其改进

特点：节点是词，短语的重要性是单词重要性之和，统计特征决定权值/加入外部文档

### TextRank 2004

[TextRank]Mihalcea R, Tarau P. TextRank: Bringing Order into Text[C]//Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing. 2004: 404-411.

单词为图的节点，定义一个窗口大小，在同一个窗口内的词就有边。因此文档化成了无向无权图。作者实验了2-10的窗口大小，发现2效果最好。为了防止图中有很多噪声，只将名词和形容词构图。构图后，使用PageRank算法：

$$S(V_i)=(1-d)+d*\sum_{j=In(V_i)}\frac{1}{|Out(V_j)|}*S(V_j)$$

d是阻尼系数，防止陷入死循环，设为0.85。$In(V)$是出度，$Out(V)$是入度，因为是无向图，因此出度等于入度等于和这个节点连接的边数。$S(V)$是节点的当前分数。

![Page Rank calculation for undirected graph](https://miro.medium.com/max/1218/1*vF3BSj0_0RFmXtbX6lSEiQ.png)

当每个词的分数收敛后，选出得分靠前的单词。在文档中标注后，将邻居单词合并成短语，计算每个短语的得分。

### SingleRank 2008

[SingleRank]Wan X, Xiao J. Single Document Keyphrase Extraction Using Neighborhood Knowledge[C]//Proceedings of the 23rd AAAI Conference on Artificial Intelligence. 2008: 855-860. 

TextRank使用的图是无权的，SingleRank将权重引入。权重由这两个词在一个大小为w的窗口中出现的次数决定，最终的计算公式就变成了：

$$S(V_k)=(1-\alpha)+\alpha\sum_{m\in NB(V_k)}\frac{C(V_j,V_m)}{\sum_{Vk\in NB(V_m)}C(V_m,V-j)}S(V_m)$$

### ExpandRank 2008

给定的文档搜集一些相似的文档来提供更多信息帮助关键短语抽取。使用基于余弦相似度的TF-IDF来得到K个最近的文档。

在文档集上进行构图。也用语法过滤器过滤了一些不会是关键短语的词。这篇论文的图是无向带权图。 两个单词之间的权是两个单词在整个文档集上的共现次数乘以原始文档与附近相关文档的相似度。即：

$$aff(v_i,v_j)=\sum_{d_p\in D}sim(d_0,d_p)*count_{d_p}(v_i,v_j)$$

排序算法收敛后，单词合并成短语，只选择名词结尾短语。一个短语的总得分是由单个单词的得分的总和计算出来的。

### CiteTextRank 2014 

[CiteTextRank]Gollapalli S D, Caragea C. Extracting keyphrases from research papers using citation networks[C]//Proceedings of the AAAI conference on artificial intelligence. 2014, 28(1).

![截屏2022-04-03 下午8.41.33.png](https://s2.loli.net/2022/04/03/iZWhHkUIRXGKylp.png)

+ d的cited上下文是dj中引用了d的那段上下文

+ d的citing上下文是d中引用了dj的那段上下文

  

图的构造是由：

1. d的内容，即global context $N_d^{C_{td}}$
2. d的被引上下文cited contexts $N_d^{C_{tg}}$
3. d的引用上下文citing context and textually-similar global contexts $N_d^{Sim}$【？？？】

组成。这是三种context。

节点是候选word，用共现窗口连边。边的权重是：

$$w_{ij}=w_{ji}=\sum_{t\in T}\sum_{c\in C_t}\lambda_t\cdot cossim(c,d)\cdot\#_c(v_i,v_j)$$

其中，$cossim(c,d)$是d中的上下文c和文档d的tf-idf向量的余弦相似度。$\#_c(v_i,v_j)$是上下文c中两个词的共现频率。$\lambda_t$是第t种context的权重。

最后，应用PageRank算法：

$$s(v_i)=(1-\alpha)+\alpha\sum_{v_j\in Adj(v_i)}\frac{w_{ji}}{\sum_{v_k\in Adj(v_j)}w_{jk}}s(v_j)$$

短语的重要性是单词重要性的和。

### PositionRank 2017

[PositionRank]Florescu C, Caragea C. PositionRank: An Unsupervised Approach to Keyphrase Extraction from Scholarly Documents[C]//Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics. 2017: 1105-1115.

https://www.sohu.com/a/227306727_100118081

对PageRank算法进行调整，将单词在一篇长文档中出现的所有位置的信息纳入其中。PositionRank的主要思想是为文档中较早出现的单词分配更大的权重(或概率)，文档中出现较晚的词权重就小。算法主要包括三个基本步骤：

1. 构建**词图**。使用名词和形容词作为候选词，在无向词图中形成节点。其中，节点之间的边基于共现滑动窗口。

2. 基于位置的PageRank算法。他们权衡每个候选词在文档中的相反位置。例如，如果一个单词出现在下面的位置:2nd, 5th和10th，那么与这个单词相关的权重是1/2 + 1/5 + 1/10 = 4/5 = 0.8。然后创建一个向量，并设置为每个候选词的归一化权重。

   对每个单词进行归一化的位置感知评分：

   $$\widetilde{p}=[\frac{p_1}{p_1+p_2+...+p_{|V|}},\frac{p_2}{p_1+p_2+...+p_{|V|}},...,\frac{p_{|V|}}{p_1+p_2+...+p_{|V|}}]$$

   Position bias Page Rank：

   $$S(v_i)=(1-\alpha)\cdot \widetilde{p}_i+\alpha\cdot\sum_{v_j\in Adj(v_i)}\frac{w_{ji}}{O(v_j)}S(v_j)$$

3. 形成候选短语。在文档中有连续位置的候选词被连接起来形成候选短语。在这些候选短语之上应用另一个正则表达式过滤器[(形容词)*(名词)+]，最长为3(即，unigrams、bigrams和trigrams)，以生成最后一组关键短语。最后，通过对组成该短语的单个单词的得分进行总结，对该短语进行评分。

## TopicRank及其改进

### TopicRank 2013

[Topic rank]TopicRank: Graph-Based Topic Ranking for Keyphrase Extraction

![截屏2022-04-04 上午1.02.17.png](https://s2.loli.net/2022/04/04/c6RLA9frFjoK4vH.png)

四个步骤：

1. 预处理。在整个输入文档上执行句子切分、单词标记和词性标记。并选择最长的名词和形容词序列作为候选关键短语。

2. 主题识别。如果两个短语有25%的部分一样，那么它们是相似的。使用Hierarchical Agglomerative Clustering (HAC)算法进行聚类分主题。

3. 基于图的排序。

   $$w_{i,j}=\sum_{c_i\in t_i}\sum_{c_j\in t_j}dist(c_i,c_j)$$

   $$dist(c_i,c_j)=\sum_{p_i\in pos(c_i)}\sum_{p_j\in pos(c_j)}\frac{1}{|p_i-p_j|}$$

   这个图上的节点是主题，是全联接的无向图。

   权值和两个主题中的两个短语的相对位置有关。共现次数多，权值越大；距离越近，权值越大。也就是两个主题中包含的短语挨得越近，这两个主题之间的权值就越大吧。

   创建完了图，使用TextRank算法对主题进行排序。

   对第i个主题$t_i$：$S(t_i)=(1-\lambda)+\lambda\times \sum_{t_j\in V_i}\frac{w_{j,i}\times S(t_j)}{\sum_{t_k\in V_j}w_{j,k}}$

4. 选择关键短语。选择k个主题，使得能够选择不同的关键短语。为了能够选出最能代表主题的短语，有三种策略：
   1. 选择出现在文档最前面的那个短语
   2. 选择出现的最频繁的
   3. 选择主题的质心，也就是在一个主题内与其他短语最相似的那个短语

### MultipartiteRank 2018

Unsupervised Keyphrase Extraction with Multipartite Graphs

本篇论文将主题内的边移除了，避免在HAC的效果不准确的情况下，同一个主题中的短语相互提升权重一起入选，从而涵盖更多主题。

本文延续TopicRank的思想，先用HAC算法将候选词分成不同的topic，希望最后的结果能够覆盖不同的主题。词图的构建方法：**节点是候选短语而不是主题、除了同主题下的节点，其他节点互相连接，构成多分图。**边的权重是两个候选短语位置的倒差数综合。

![https://ithelp.ithome.com.tw/upload/images/20200917/20128558fJPf42EgDx.png](https://ithelp.ithome.com.tw/upload/images/20200917/20128558fJPf42EgDx.png)

权重的计算公式为：

$$w_{i,j}=\sum_{p_i\in P(c_i)}\sum_{p_j\in P(c_j)}\frac{1}{|p_i-p_j|}$$

其中$P(c_i)$是$c_i$的位置集合。

本文的关键操作：改变权重，提升显著的候选词。提升每个主题中第一个出现的候选词。方法是把同主题的出边的权重乘上一个被提升词的位置函数，加到被提升词的入边的权重上：

$$w_{i,j}=w_{i,j}+\alpha\cdot e^{\frac{1}{p_i}}\cdot\sum_{c_k\in\tau(c_j)\\/\{c_j\}}w_{ki}$$

![https://ithelp.ithome.com.tw/upload/images/20200917/20128558gl0MTJrib0.png](https://ithelp.ithome.com.tw/upload/images/20200917/20128558gl0MTJrib0.png)

然后跑没有bias的PageRank

## TPR及其改进

特点：改变pagerank的重置概率

### TopicalRank/TPR 2010

[TopicalRank/TPR]Zhiyuan Liu, Wenyi Huang, Yabin Zheng, and Maosong Sun. 2010. Automatic keyphrase extraction via topic decomposition. In Proceedings of the 2010 Conference on Empirical Methods in Natural Language Processing, pages 366–376.

其主要思想是在文档语料库中进行LDA算法来融入主题信息。文档构图后，用LDA寻找文档的潜在主题，根据主题修改词图的权重，为每个主题都运行一次PageRank。

![截屏2022-04-04 上午3.39.12.png](https://s2.loli.net/2022/04/04/c3pCQvwDoAyr5ha.png)

在语料库上用LDA得到每个词的主题分布，即$pr(z|w)$，其中z是主题。也可以得到一个文档的主题分布$pr(z|d)$。

1. Constructing Word Graph

2. Topical PageRank

   原始的PageRank公式如下，第二项表示有$1-\lambda$的概率会跳到其他节点上，每个节点的概率都一模一样。

   $$R\left(w_{i}\right)=\lambda \sum_{j: w_{j} \rightarrow w_{i}} \frac{e\left(w_{j}, w_{i}\right)}{O\left(w_{j}\right)} R\left(w_{j}\right)+(1-\lambda) \frac{1}{|V|}$$

   作者认为可以对一些节点施加更大的权重，就是改变第二项，最终公式为：

   $$R_{z}\left(w_{i}\right)=\lambda \sum_{j: w_{j} \rightarrow w_{i}} \frac{e\left(w_{j}, w_{i}\right)}{O\left(w_{j}\right)} R_{z}\left(w_{j}\right)+(1-\lambda) p_{z}\left(w_{i}\right)$$

   这时每个节点的概率不等分，而是给定这个主题下这个词的概率。也就是主题z有多大程度关注这个词，和当前主题密切相关的词会有更大的权重。

   因为有很多个主题，所以我们分别运行很多次这个算法，得到了某个主题下的单词的权重。

3. 排序

   短语的权重是组成这个短语的单词的权重之和。最终要乘每个文档给定时的主题分布。所有主题上的重要性加起来做最后的排序。

   $$R(p)=\sum_{z=1}^{K} R_{z}(p) \times p r(z \mid d)$$

### SingleTopicalRank(STPR) 2015

[SingleTopicalRank(STPR)]Sterckx et al. (2015) 改进就是对每个文档只需要运行一次图传播算法，通过计算词在整个（所有）主题下的重要性。

词-主题概率分布是给定一个主题，这个词在这个主题下的概率：$\vec{P}(w_i|Z)=(P(w_i|z_1),...,P(w_i|z_K))$

文档-主题概率分布是给定一个文档，这个文档属于一个主题的概率：$\vec{P}(Z|d)=(P(z_1|d),...,P(z_k|d))$

![截屏2022-04-04 上午3.38.32.png](https://s2.loli.net/2022/04/04/uVi9qnUTJ4SDjpf.png)

Single weight-value的计算公式为这两个分布的余弦相似度：

$$W\left(w_{i}\right)=\frac{\vec{P}\left(w_{i} \mid Z\right) \cdot \vec{P}(Z \mid d)}{\left\|\vec{P}\left(w_{i} \mid Z\right)\right\| \cdot\|\vec{P}(Z \mid d)\|}$$

可以理解为文档d上第i个词的topical word importance。

所以最后的PageRank算法为：

$$\begin{array}{r}
R\left(w_{i}\right)=\lambda \cdot \sum_{j: w_{j} \rightarrow w_{i}}\left(\frac{e\left(w_{j}, w_{i}\right)}{O\left(w_{j}\right)} \cdot R\left(w_{j}\right)\right) \\
+(1-\lambda) \cdot \frac{W\left(w_{i}\right)}{\sum_{w \in \nu} W(w)}
\end{array}$$

### SalienceRank 2017

[SalienceRank]Salience Rank: Efficient Keyphrase Extraction with Topic Modeling

https://www.weiweicheng.com/research/slidesposters/cheng-dsl17slides.pdf

https://www.sohu.com/a/227490899_100118081

是针对TopicalRank(TPR)的改进。TopicalRank(TPR)需要运行K次PageRank（K是主题数）。本文提出的改进方法只需要运行一次，并且能灵活抽取出平衡主题特异性和语料特异性的短语。它说STPR是SalienceRank的一个特例，只考虑了主题特异性，没考虑语料库特异性。

1. 单词w的主题特异性（前人的定义）

   $$TS(w)=\sum_{t\in T}p(t|w)\log\frac{p(t|w)}{p(t)}=KL(p(t|w)||p(t))$$

   其中$p(t)$是由主题t随机选择词的似然，$p(t|w)$是词w由主题t生成的概率。词的主题特异性就是由这两个东西的分歧程度得到。

   可以理解为，这个词在不同主题间共享地越少，主题特异性越高。

   需要做归一化。

2. 单词w的语料库特异性

   $$CS(w)=p(w|corpus)$$

   也就是语料库中的单词频率

3. 单词w的salience

   $$S(w)=(1-\alpha)CS(w)+\alpha TS(w)$$

   其中$\alpha$是平衡二者的参数。当它很大时，表示更重视主题特异性，单词在主题间不太共享。当它很小时，更重视语料库特异性，表示提取的单词经常出现在特定的语料库中。这像一个开关，控制我们想要获得什么样的关键短语。

最终算法是：

$$R(w_i)=\lambda\sum_{j:w_j\rightarrow w_i}\frac{e(w_j,w_i)}{Out(w_j)}R(w_j)+(1-\lambda)S(w_i)$$



这两种特异性的计算方法都可以不一样，SingleTPR就是一种主题特异性的实现方法。

## 其他

### SGRank 2015 图和统计特征混合

[SGRank]Danesh S, Sumner T, Martin J H. SGRank: Combining Statistical and Graphical Methods to Improve the State of the Art in Unsupervised Keyphrase Extraction[C]//Proceedings of the 14th Joint Conference on Lexical and Computational Semantics. 2015: 117-126. 

在SGRank算法中，节点为短语，边按共现关系连接。改进点是统计特征被更充分地利用，体现在以下方面：

1. 根据统计特征的两次排序筛选掉了不重要的短语
2. 决定了两个短语的之间边的权重
   1. 两个短语之间的相关性
   2. 单个短语的重要性

---

1. 候选短语识别

   n-gram、词性筛选、最小出现次数

2. 评分，根据修改的tf-idf

   KP-Miner的公式

3. 根据一些统计特征rerank上一步的前几名

   $$w_s(t,d)=(tf(t,d)-subSumCount(t,d))*idf(t)*PFO(t,d)*TL(t)$$

   其中，PFO是第一次出现的位置——Position of First Occurrence factor (PFO)：$PFO(t,d)=\log(\frac{cutoffPosition}{p(t,d)})$。

   TL(t)是短语长度。

   subSumCount(t,d)是在最后的候选短语列表中，包含t比t还长的短语在文档d上的频率。

4. 构图继续排序，结果输出

   图的节点是上一步获得的前几名候选短语，边根据共现关系相连。因为经过上面两步筛选后词已经很稀疏了，窗口大小设为1500。

   两个短语之间的相关程度是：$w_d(t_i,t_j)=\frac{\sum_{i=1}^{tf(t_i)}\sum_{j=1}^{tf(t_j)}\log(\frac{winSize}{|pos_i-pos_j|})}{numCo-occurrences(t_i,t_j)}$

   对于单个候选短语，希望统计特征权值大的在PageRank里的权重也更大，能被经过更多次。所以边的权重在两个短语之间的相关程度上还要和两个短语第二步计算的统计特征相乘，为：$w_e(t_i,t_j)=w_d(t_i,t_j)*w_s(t_i)*w_s(t_j)$

   最后：$S\left(V_{i}\right)=(1-d)+d * \sum_{j \in \operatorname{In}\left(V_{i}\right)} \frac{w_{e}(j, i) * S\left(V_{i}\right)}{\sum_{k \in O u t\left(V_{j}\right)} w_{e}(j, k)}$

# DL

## WordAttractionRank 2015

[Word Attraction Rank]Corpus-independent Generic Keyphrase Extraction Using Word Embedding Vectors

基于图的排序模型，考虑了来自分布式词表示的语义信息。

构建词图，单词为节点，边通过大小为K的滑动窗口决定。每个边的权重是Attraction Score=Attraction Force * Phraseness Score。

Attraction Force：$f(w_i,w_j)=\frac{freq(w_i)*freq(w_j)}{d^2}$

其中d是两个word embedding的欧几里得距离，freq(w)是单词w在文档中的出现次数。这个公式想要表达的是两个单词之间的吸引力（受到牛顿万有引力的启发）。

Pbraseness score：$dice(w_i,w_j)=\frac{2*freq(w_i,w_j)}{freq(w_i)+freq(w_j)}$

这个分数是想识别出短语搭配。

最后，两个单词之间的权重就是它们相乘。

## Key2Vec 2018 

[Key2Vec] Key2Vec: Automatic Ranked Keyphrase Extraction from Scientific Articles using Phrase Embeddings

1. 预处理

   文本分句、用Spacy中的命名实体识别模型和名词短语chunking将名词短语抽取出来。

2. 训练短语嵌入模型

   使用Fasttext， 在1,147,000 scientific abstracts上训练Fasttext-skipgram模型。词典里包含单字和多词短语。

3. 关键短语抽取步骤

   1. 候选短语选择：和预处理一样。

   2. 打分。

      第i个文档$d_i$有一个主题向量$\hat{\tau_{d_i}}$，可以调整？

      对于一个给定的文档，抽取一个theme excerpt和一个主题短语的集合。前者一般用前十句话加上标题。短语从里面提取。这些短语的平均就是主题向量了。

      计算每个候选短语和主题向量的相似度，作为thematic weight $w_{c_j}^{d_i}$

   3. 排序

      构建词图，使用weighted personalized PageRank algorithm。

      两个词之间的权值的计算过程是：

      $$\begin{gathered}
      \operatorname{semantic}\left(c_{j}^{d_{i}}, c_{k}^{d_{i}}\right)=\frac{1}{1-\operatorname{cosine}\left(c_{j}^{d_{i}}, c_{k}^{d_{i}}\right)} \\
      \operatorname{cooccur}\left(c_{j}^{d_{i}}, c_{k}^{d_{i}}\right)=\operatorname{PMI}\left(c_{j}^{d_{i}}, c_{k}^{d_{i}}\right) \\
      \operatorname{sr}\left(c_{j}^{d_{i}}, c_{k}^{d_{i}}\right)=\operatorname{semantic}\left(c_{j}^{d_{i}}, c_{k}^{d_{i}}\right) \times \operatorname{cooccur}\left(c_{j}^{d_{i}}, c_{k}^{d_{i}}\right)
      \end{gathered}$$

      PageRank算法为：

      $$R\left(c_{j}^{d_{i}}\right)=(1-d) w_{c_{j}}^{d_{i}}+d \times \sum_{c_{k}^{d_{i}} \in \varepsilon\left(c_{j}^{d_{i}}\right)}\left(\frac{\operatorname{sr}\left(c_{j}^{d_{i}}, c_{k}^{d_{i}}\right)}{\left|\operatorname{out}\left(c_{k}^{d_{i}}\right)\right|}\right) R\left(c_{k}^{d_{i}}\right)$$



