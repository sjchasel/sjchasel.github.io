---
layout: post
title: 【论文阅读】A Unified Generative Framework for Various NER Subtasks
categories: ["论文笔记", "NLP"]
mathjax: true
---

ACL 2021 A Unified Generative Framework for Various NER Subtasks

code：https://github.com/yhcc/BARTNER

# Motivation

命名实体识别可以继续细分成以下三个子任务：

1. flat NER
2. nested NER：实体中有实体
3. discontinuous NER：实体词在句子中不连续

![subtasks](/images/blog/generation_ner.png)

以往解决 NER 任务用的方法是字符级别的序列标注或者对 text span 进行分类，但这些方法无法同时解决这三种子任务。

# Method

首先，我们需要将三种NER任务的输出统一，就能用seq2seq模型统一生成结果了，也能够运用预训练模型BART了。可以用entity pointer index sequence来表示文本中的实体和其位置。

## NER Task

先对 NER 任务做个定义。
输入一句话 $X=[x_1, x_2, ..., x_n]$ ，我们想要得到的输出是 $Y=[s_{11}, e_{11}, ..., s_{1j}, e_{1j}, t_1, ..., s_{i1}, e_{i1}, ..., s_{ik}, e_{ik}, t_i]$ 。其中s表示指示这个实体的text span的start index， e则是end index。因为一个实体可能有多个span（在discontinuous NER中），所以在一个或多个span后再加一个t来作为实体的tag index。t的范围是n到n+l，l是实体标签的个数。往后移动n是为了不和原文中的index弄混。

## Model

![model](/images/blog/generation_ner_model.png)

如上图所示，其实模型就是一个带pointer mechanism的seq2seq model，用了BART。

在解码的每一步中，输入的是上一步生成的token，吐出的是index。所以在送入下一步的输入时，需要将index映射回token得到其embedding。
...好的写到这里又跑去看sentiment那篇了，模型都一模一样。

## BPE

这边还提到了BART使用BPE算法会引起的一些问题，sentiment那篇并没有提。因为BPE算法可能会把一个word表示成好几个token，那么如何识别原文中的span呢。作者定义了三种方式如下：


![bpe](/images/blog/ner_bpe.png)

这个例子有三个实体（有重叠），PER的word是x1和x3，LOC的word是x1，x2，x3和x4，ORG的word是x4。

1. $\bf Span$：实体首词的第一个BPE token和尾词的最后一个BPE token；
2. $\bf BPE$：所有的BPE token；
3. $\bf Word$：每个word的第一个BPE token。

# Experiment

![res1](/images/blog/ner_res1.png)
![res2](/images/blog/ner_res2.png)

