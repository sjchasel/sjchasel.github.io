---
layout: post
title: 【学习笔记】CS224N 01 Introduction and Word Vectors
categories: 学习笔记
keywords: NLP, CS224N
mathjax: true
# key: cs224n_1
---



CS224N_Introduction and Word Vectors



# Introduction——word embedding

如何用计算机语言表示一个词？目前nlp中遵循的一个原理是：

+ distributional semantics：一个单词的语义是由经常和它一起出现的上下文决定的

通过建模单词的上下文信息，得到每个词的word embedding。是一个dense的向量，通常为300维。这就是一个词的分布式表示，因为这个词的意思分布在了300个维度上。叫词嵌入的原因是这个词被嵌入在了一个300维的空间中。



# Word2vec（Mikolov et al.2013）

## Idea

+ 我们有一个较大的语料库
+ 有一个词汇表，这个词汇表里的每个词我们都用一个向量表示
+ 对于文本中的每个位置t，都会有处于这个位置的中心词c和在c周围的词o
+ 计算c和o的相似度，来计算**给定中心词c的条件下周围词o出现的概率**
+ 不断调整每个词向量使概率最大化

![image.png](https://s2.loli.net/2022/02/19/S5cVa9pkxd1XtYr.png)

## training objective

使用最大似然估计

+ 最大似然估计，通俗理解来说，**就是已经有样本信息（数据），得到最具有可能导致这些样本结果出现的模型参数值！**

![截屏2022-02-19 下午10.42.35.png](https://s2.loli.net/2022/02/19/Ntouz5I6qG31Pea.png)

如何计算概率？

对于每个词，会有两个词向量：

1. $v_w$：当w是中心词c时使用
2. $u_w$：当w是上下文词o时使用

那么给定中心词，周围词出现的概率可按如下公式计算

$$P(o|c)=\frac{\exp(u_o^Tv_c)}{\sum_{w\in V}\exp(u_w^Tv_c)}$$

+ 在公式的分子上，我们使用点积来衡量两个词的相似性。比如，如果它们都是正的或都是负的，相乘起来就都会为正，加起来值就很大。如果它们的含义相反，就会出现负值使值变小。
+ 取幂使得最后的值都是正的。
+ 因为是概率，我们希望它的和是正的。所以分母是做个归一化。V表示上下文词的集合。也就是给定一个中心词，每个上下文词出现的概率之和为1。

# supplement from note

Word2vec实际上包含多种算法

两种算法：

1. continuous bag-of-words（CBOW）：已知周围词来预测中心词
2. skip-gram：已知中心词预测周围词（上面提到的）

两种训练目标：

1. negative sampling
2. hierarchical softmax

## N-gram language model

语言模型要做的事，是给每个句子分配一个概率。真实的句子概率大，混乱无效的句子概率小。



最简单的模型就是**Unigram language model**：

$$P(w_1, w_2,...,w_n)=\prod^n_{i=1}P(w_i)$$

它表示这句话出现的概率是这句话中每个词出现的概率的乘积。（若一个词在语料库中出现了n词，语料库中有N个词，那么这个词的概率是n/N）但事实上每个词出现的概率都是依赖于上下文的，这样独立的计算方式并不好，没有考虑词序。



我们再多考虑旁边的一个词，这个模型就叫**Bigram language model**：

$$P(w_1,w_2,...,w_n)=\prod^n_{i=2}P(w_i|w_{i-1})$$

当然这也有缺点，还能考虑更长的窗口。但是窗口越长，计算难度也越大。



N-gram language model：

$$P(w_1,w_2,...,w_m)=\prod_{i=1}^mP(w_i|w_{i-n+1},...,w_{i-1})$$

## Continuous Bag of Words Model (CBOW)

**!!!!周围词预测中心词**

对于每个词，我们需要学习两个向量：

1. v（input vector）：当这个词是上下文词时
2. u（output vector）：当这个词是中心词时



对于每个中心词$w_i$，当窗口大小为m时，我们可以知道2m个它的上下文词的向量：$v_{i-m},v_{i-m+1},...,v_{i+m}$。将这些向量加起来除以2m，得到当前中心词的上下文表示$\hat{v}$。

每个词的中心词表示和当前的上下文表示和做点积，就是它们的相似度。有$|V|$个词，因此我们得到了维度为$1\times |V|$的点积向量，这个向量进行softmax，转换成概率。位置j上的概率表示已知中心词$w_i$的周围词，第j个词出现的概率。因为这是第i个词的上下文向量，那么我们的标签就是第i个位置为1，其他位置为0的one-hot向量。

优化我们的目标，可以使得上下文表示和中心词的相似度变大，也就是使距离相近的词的表示更加接近。





![截屏2022-02-21 下午11.49.05.png](https://s2.loli.net/2022/02/21/RKvQtn2HUxo5cjl.png)





采用交叉熵作为我们的损失函数。

$$H(\hat{y},y)=-\sum_{j=1}^{|V|}y_j\log(\hat{y}_j)$$

若只考虑第i个词的情况，真实的$y$是一个one-hot向量，因此实际上只计算了：$H(\hat{y},y)=-y_i\log{(\hat{y}_i)}=-\log{(\hat{y}_i)}$

预测值$\hat{y}_i$越大，这个式子的值就越小，损失就越小。



最终的预测目标可写成

$$minimize J=-logP(w_c|w_{c-m},...,w_{c-1},w_{c+1},...,w_{c+m})\\=-logP(u_c|\hat{v})\\=-log \frac{\exp{(u_c^T\hat{v})}}{\sum_{j=1}^{|V|}\exp{(u_j^T\hat{v})}}\\=-u_c^T\hat{v}+log\sum_{j=1}^{|V|}\exp{(u_j^T\hat{v})}$$

## Skip-Gram Model



**!!!!中心词预测周围词**

跟cbow差不多，如下图

![截屏2022-02-22 上午12.08.46.png](https://s2.loli.net/2022/02/22/cd8ITuCz1GyPRXo.png)

虽然用多个one-hot和many-hot（？？有这个说法嘛）按照CBOW的公式来说应该是一样的，但同样的输入有不同的输出结果我还是觉得好怪啊！



用朴素贝叶斯的假设来分解概率。我们认为，给了中心词后，周围每个词的出现都是相互独立的。

所以训练目标为

![截屏2022-02-22 上午12.16.33.png](https://s2.loli.net/2022/02/22/vXlKZUY4f8RNGbw.png)

此处u和v的含义和CBOW相反。

同样使用交叉熵损失来优化

$$J=\sum_{j=0,j\ne m}^{2m}H(\hat{y},y_{c-m+j})$$

$\hat{y}$是第c-m+j个上下文词的one-hot向量。

## Negative sampling

### 思想  


回顾之前写的损失函数，会发现它的复杂度是O(\|V\|)。我们的语料库可以很大，这样计算起来就太困难了。我们可以采用更简单的形式来逼近它。  


> 修改优化目标函数，这个策略称之为“**Negative Sampling（负采样）**“，使得每个训练样本只去更新模型中一小部分的weights。

比如，我们的词典大小为10000，dim=300时，意味着每个样本的计算在词向量这部分都需要调整10000*300个参数。但每个样本的目标总是one-hot向量，也就是只有一个数和我们的输入相关，为1，称为正样本，其他9999个词对应的标签都为0，称为负样本。全部更新不仅消耗计算资源，而且效率很低。因此我们在负样本中进行采样，只更新采样到的样本的向量。



采样的概率和这个词出现的频率有关，出现的越频繁的词越容易被采样到。

### 计算方式

以上是negative sampling的思想。虽然这个方法是基于skip-gram提出的，但它只是给模型换了个优化目标，也可以用在CBOW中。

现在我们有中心词c和它的上下文词汇w，这个pair是出现在我们的语料库中的。我们需要进行负采样的样本就是没有出现在语料库中的pair。我们使用sigmoid函数来计算一个pair存在于语料库中的概率：

$$P(D=1|w,c,\theta)=\sigma(v_c^Tv_w)=\frac{1}{1+e^{-v_c^Tv_w}}$$

我们想要最大化的目标就是每个存在于语料库中的pair的概率以及每个不存在于语料库中的pair不属于这个语料库的概率。

![截屏2022-02-27 下午5.51.23.png](https://s2.loli.net/2022/02/27/f9MR3b8pQuCXyqE.png)

以上是最大化似然，损失函数就是负对数似然：

$$J=\sum_{(w,c)\in D}\log \frac{1}{1+\exp{-u^T_wv_c}}-\sum_{(w,c)\in \widetilde{D}}\log{\frac{1}{1+\exp{u_w^Tv_c}}}$$

其中，$\widetilde{D}$是采样出的负样本集。



+ 对于skip-gram，中心词c预测周围词c-m+j，我们的目标可以写成：

$$-\log{\sigma(u_{c-m+j}^T\cdot v_c)}-\sum^K_{k=1}\log{\sigma{(-\widetilde{u}_k^T\cdot v_c)}}$$



+ 对于CBOW，周围词预测中心词c。先将周围词做个平均得到$\hat{v}=\frac{v_{c-m}+v_{c-m+1}+...+v_{c+m}}{2m}$，目标为：

$$-\log{\sigma{(u_c^T\cdot \widetilde{v})}}-\sum^K_{k=1}\log{\sigma{(-\widetilde{u}^T_k\cdot \hat{v})}}$$



在以上公式中，$\{\widetilde{u}_k|k=1,...,K\}$是按照概率采样出来的集合。

### 负采样方法

每个词被采样到的概率和它在语料库中出现的频率的四分之三次幂有关，做个归一化就是每个词的概率。



## Hierarchical Softmax

- [ ] ...





# 补充资料

+ [课程材料](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1204/)
+ [Word embedding](https://medium.com/data-science-group-iitr/word-embedding-2d05d270b285)
+ [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
