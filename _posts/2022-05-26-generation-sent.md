---
layout: post
title: 【论文阅读】A Unified Generative Framework for Aspect-Based Sentiment Analysis
categories: 论文笔记
keywords: NLP, sentiment
mathjax: true
---

ACL 2021 A Unified Generative Framework for Aspect-Based Sentiment Analysis

code：https://github.com/yhcc/BARTABSA

# Motivation and Background

ABSA竟然能分出七个子任务...这篇文章也是将信息抽取任务的多种表现形式用seq2seq模型统一起来生成。

![sent](/images/blog/sent_subtask.png)

其中，a表示aspect term，是需要判断情感极性的实体，s表示情感极性，o表示opinion term。通过o判断a的s。这七种子任务为：


1. Aspect Term Extraction(AE)：输入一句话，【抽取】出所有需要判断情感极性的aspect term
2. Opinion Term Extraction (OE)：从一句话中【抽取】出所有的opinion term
3. Aspect-level Sentiment Classification (ALSC)：输入这句话和其中的aspect term，【分类】判断它的情感极性
4. Aspect-oriented Opinion Extraction (AOE)：【抽取】输入这句话和其中的aspect term，找到它的opinion term
5. Aspect Term Extraction and Sentiment Classification (AESC)：输入一句话，从中【抽取】出所有的aspect term并且【分类】判断其情感极性
6. Pair Extraction (Pair)：输入一句话，【抽取】出所有的aspect term和opinion term
7. Triplet Extraction (Triplet)：输入一句话，【抽取】出所有的a和o，并【分类】判断s

看来这边还是抽取和分类的结合，这个任务看起来并不需要生成，但是生成可以代替抽取和分类，从而将它们统一起来。


# Method

观察上面的任务，有抽取和分类两个需求。作者将抽取任务转成生成pointer indexes，将分类任务转成生成 class indexes generation（类别是有限的因此可以这样操作）。

## Task Definition

同样的，我们先定义任务，将它们的形式统一。
对于除了 ALSC 和 AOE 以外的任务，输入都是一句话。我们可以将输出定义成以下形式。上标s和e分别表示start index和end index。
+ AE: $Y=[a_1^s, a_1^e, ..., a_i^s, a_i^e, ...]$
+ OE: $Y=[o_1^s, o_1^e, ..., o_i^s, o_i^e, ...]$
+ AESC: $Y=\left[a_{1}^{s}, a_{1}^{e}, s_{1}^{p}, \ldots, a_{i}^{s}, a_{i}^{e}, s_{i}^{p}, \ldots\right]$
+ Pair: $Y=\left[a_{1}^{s}, a_{1}^{e}, o_{1}^{s}, o_{1}^{e}, \ldots, a_{i}^{s}, a_{i}^{e}, o_{i}^{s}, o_{i}^{e}, \ldots\right]$
+ Triplet: $Y=\left[a_{1}^{s}, a_{1}^{e}, o_{1}^{s}, o_{1}^{e}, s_{1}^{p}, \ldots, a_{i}^{s}, a_{i}^{e}, o_{i}^{s}, o_i^e, s_i^p, \ldots\right]$

而对于ALSC和AOE两个任务，我们把指定的aspect也加入到target中去生成：
+ ALSC: $Y=\left[\underline{a^{s}}, \underline{a^{e}}, s^{p}\right]$
+ AOE: $Y=\left[\underline{a^{s}}, \underline{a^{e}}, o_{1}^{s}, o_{1}^{e}, \ldots, o_{i}^{s}, o_{i}^{e}, \ldots\right]$

一个例子如下：

![task](/images/blog/sent_example.png)

但是这两个任务的输入，给的aspect term放哪呢？直接拼接在句子上吗？

## Model

![model](/images/blog/sent_model.png)

晕，这两篇论文的模型图就是换了个色系吧。这两篇论文真的是...

同样的，BART encoder和BART decoder。因为decoder输出的是index，所以需要转换回token输到下一步。encoder得到了输入的memory $\bf H^e$ ，如何在decoder中得到第t步的概率分布呢？输入编码器的 memory 和之前输出的token，得到了第t步的hidden state $h_t^d$。

$$
\begin{aligned}
\mathbf{E}^{e} &=\text { BARTTokenEmbed }(X), \\
\hat{\mathbf{H}}^{e} &=\operatorname{MLP}\left(\mathbf{H}^{e}\right) \\
\overline{\mathbf{H}}^{e} &=\alpha \hat{\mathbf{H}}^{e}+(1-\alpha) \mathbf{E}^{e} \\
\mathbf{C}^{d} &=\operatorname{BARTTokenEmbed}(C) \\
P_{t} &=\operatorname{Softmax}\left(\left[\overline{\mathbf{H}}^{\mathbf{e}} ; \mathbf{C}^{d}\right] \mathbf{h}_{t}^{d}\right)
\end{aligned}
$$

在第一个式子中，编码的是输入句X。第二个式子中将encoder的输出过了一个MLP。第三个式子是把这两部分加权求和，权重当然是个超参数。这个东西好像叫残差连接？
第四个式子是将类别进行embed，第五个式子得到当前步的概率分布。H和C拼起来算表示这个概率分布是在输入文的index和类别的index上的。

不是很懂为什么非要强调一个index2token的转化过程，词表里的词就是原文的词和类别的词不就是正常的解码过程，输出结果的时候再把token转成index。代码里真的有index2token的过程吗...

这个模型好像除了模型结构在各个子任务上是统一的之外就没啥统一的了。不同的子任务还需要用不同的target模版去训，不像其他一些预训练模型是给提示来转换不同的任务。

# Experiment


![exp](/images/blog/sent_exp.png)

其实感觉这种抽取的形式挺抽象的，但实验还是有效果。