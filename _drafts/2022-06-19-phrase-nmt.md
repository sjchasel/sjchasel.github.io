---
layout: post
title: 【论文阅读】Towards Neural Phrase-based Machine Translation
categories: 论文笔记
keywords: NLP, NMT, NAT
mathjax: true
---





# Background——SWAN

ICML 2017 Sequence Modeling via Segmentations  
code: https://github.com/posenhuang/NPMT  


在这个任务中，我们的输出是一个序列，但输入可能是序列也可能不是，我们分情况考虑。  

## non-sequence to sequence

输入是一个向量，输出是序列 $y_{1:T}$ , 我们想要建模的概率是 $p(y_{1:T}|x)$。  

在输出序列中的短语我们称为segment，而几个短语组成输出序列的方案叫segmentation，所有可能的短语组成序列的方案集合是 $S_y$ , 其中任意一个方案为 $a_{1:{\tau}_a} \in S_y$ 。  

它可以表示成 $\pi (a_{1:{\tau}_a})=y_{1:T}$ , $\pi()$ 这个操作就是把从 $a_1$ 到 $a_{\tau _a}$ 的序列拼接起来。

![](/images/blog/swan1.png)

对于一个长度为T的输出序列来说，可能的segment（短语）的个数是从1加到T，那么个数就是与 $T^2$ 相关。可能的序列组合segmentation的个数是指数级的。  

因为我们实现不知道应该采用哪个segmentation，所以序列生成的概率定义为每个segmentation的概率的和：

$$p(y_{1:T}|x) \triangleq \sum_{a_{1:\tau_a}\in S_y}p(a_{1:\tau_a}|x)=\sum_{a_{1:\tau_a}\in S_y}\prod_{t=1}^{\tau_a}p(a_t|x,\pi(a_{1:t-1}))$$

在不同segmentation的计算中，概率的计算似乎是一个以segment为基本单位的自回归式。但因为segmentation的个数是指数级的，这个公式不能按这样直接计算。

## sequence to sequence

当输入的x不是一个单纯的vector了，而是一个序列 $x_{1:T'}$ 时，这里有一个很强且不太现实的假设：每个输入元素 $x_t$ 都可以生成一个短语 $a_t$ ，且所有的短语拼接起来就是输出y序列了。那么这种情况中，我们允许有空segment，即 $a_t=\{\$\}$ 。  


![](/images/blog/swan2.png)

因为一个输入元素可以是对应一个空的短语，所以这个模型的名字就是 Sleep-WAke Networks（SWAN） 。公式和上面的差不多，只不过条件中的x要改成 $x_{1:T'}$ ，segment的个数 $\tau_a$ 改成输入序列的长度 $T'$。  

可能的segmentation的个数是 $O(T'T^2)$，同样的公式中 $|S_y|$ 是指数级别的难以计算。

## Carrying over information across segments

大概意思就是说在一个segmentation中各个segment不是相互独立的，而是依赖于前面的segment拼接起来的结果，以这个和输入作为条件。

![](/images/blog/swan3.png)


# Motivation


翻译过程中粒度的改变  
以往是将单词视为基本单位，一个词一个词地翻译  
现在是认为短语更有意义  
按照短语来翻译更自然  

短语翻译的工作中  
Sleep-WAke Networks (SWAN)需要src和tgt单调对齐：作者提出解决方法a new layer to perform (soft) local reordering on input sequences  

# Methid——NPMT(neural phrase-based machine translation)

## Overview

NPMT的两个关键之处在于使用了SWAN这个模型，以及为了减轻SWAN对语言对对齐的假设限制，而提出的soft reordering layer。

![](/images/blog/npmt_model.png)

## SWAN in NPMT

SWAN可以看成是这样的生成模型：
1. 对于t=1...T'，给一个初始表示 $x_t$ ，从RNN中采样连续的字符直到遇到 $\$$。这个短语序列就是 $a_t$。
2. 将 $\{a_1,...,a_{T'}\}$ 拼接起来组成输出序列 $y_{1:T}$。


在训练的时候，输出序列y的概率是计算所有segmentation的和。但是这个东西的数量是指数级别的。原文用了动态规划的方法解决这个问题。

$$p(y_{1:T}|x_{1:T'}) \triangleq \sum_{a_{1:T'}\in S_y}\prod^{T'}_{t=1}p(a_t|x_t)$$

怎么在这篇论文里有可以并行了，不需要依赖前面的短语拼接成的序列了...

## Local Reordering of Input Sequence


![](/images/blog/npmt3.png)

这一层输入的就是一堆word embedding，为 $e_{1:T'}$，而这一层的输出是 $h_{1:T'}$，用于输入到后面的RNN中。h的计算方式是：

$$h_t=tanh(\sum_{i=0}^{2\tau}\sigma(w_i^T[e_{t-\tau;\dots;e_t;\dots;e_{t+\tau}}])e_{t-\tau+i})$$  

注意到，这个式子里集成了 $2\tau +1$ 个embedding，这个数字是the local reordering window size，可以表示为 $\sigma_t$ 。激活函数后的结果和权重w决定了ht中有多少信息来自不同的e。比如在b图中，h2的大部分信息来自e3，h3的大部分信息来自e2，这样就实现了reorder。

# Experiment

做的都是机器翻译的。

![](/images/blog/npmt_exp1.png)

![](/images/blog/npmt_exp2.png)

![](/images/blog/npmt_exp3.png)