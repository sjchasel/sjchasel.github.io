---
layout: post
title: 【论文阅读】Prefix-Tuning： Optimizing Continuous Prompts for Generation
categories: 论文笔记
keywords: NLP, Prompt
mathjax: true
# key: prefix_tuning
---

# Background

普通的微调就8说了

Lightweight fine- tuning：思想是冻结大部分参数，只调整一部分。这就需要找到哪些参数微调起来是对任务有帮助的，识别模型中的重要组件。一个方法是移除一些参数。（？）一个方法是插入一些参数，比如训练“side” network（Side-Tuning: A Baseline for Network Adaptation via Additive Side Networks）、adapter-tuning。这些方法大约需要微调3.6%的参数（本文只需要训练0.1%的参数就能达到差不多的效果）。

Prompt：transformer的模型对文本长度有要求（然后呢，作者就不说了？是说prompt会占用文本的位置吧？让能够输入的文本更短了）。2020年被提出的AutoPrompt采用一段离散的trigger tokens引导模型得到正确的输出。

controllable generation：在训练阶段进行控制的代表模型是CTRL。在解码阶段进行控制的模型代表是GeDi、PPLM。但它们不能进行细粒度的控制（？）。

# Motivation

fine-tuning：每将预训练模型应用在一个任务上时，就需要存储这整个模型，更新所有参数。随着预训练模型大小的增长，微调技术的开销也越大。

也有研究针对这个问题提出lightweight fine-tuning技术：冻结大部分预训练参数，在模型上加上一些可以训练的小部件。比如adapter-tuning，它在预训练模型的层之间插入task- specific层。还有现在大火的prompt，甚至不需要加入额外的层，只要给点提示。

Prefix-Tuning就是只需要优化一个很小的task- specific vector。对于transformer来说，相当于在文本前面加上了一些虚拟的token，与prompt的形式很像，但不同的是这些虚拟token完全可以训练，不需要表达任何真实的含义。

还有一个好处在于，在一些个性化的场景中，我们也可以对每个user训练自己的prefix，计算开销不大又能充分发挥预训练模型的作用，避免cross- contamination（交叉污染？）。更进一步，我们可以在一个batch中处理不同user或者task的数据，毕竟只要替换prefix参数就行（相当于多加了几个词在数据前面），这在其他方法中是不可能的。

---

作者的intuition部分：

一个正确的上下文可以引导LM输出我们想要的东西，而不必改变它的参数。比如上文给Barack，它很可能会生成Obama。因此想要找到一个可以引导LM生成想要的结果的上下文，这个上下文可以影响模型是如何编码x的，因为它知道要从x中生成什么。它也通过改变next token distribution来引导模型生成y。

如何找到这个上下文？人类很难去手动定义。同data- driven的方式去找到一些离散的token也很难计算。那么就去优化一组连续的embedding吧。这样它的影响可以传递到每一层中，并且比离散的token能包含更多的含义。

# Method

在LM中：z=[prefix;x;y]

在encoder-decoder中：z=[prefix;x;predix';y]

重参数化训练

# Experiment

任务和模型：table-to-text（GPT-2）和summarization（BART）

table-to-text task datasets：E2E（1 domain——restaurant reviews）、WebNLG（14 domains）、DART（open-domain）

metrics：BLEU、NIST、METEOR、ROUGE-L、CIDEr

结论：

1. prefix- tuning的参数比微调小1000倍，但在table-to-text全数据的情况下可以和微调相比，但在摘要上稍微差一些。
2. 在小数据的设置下，prefix- tuning在两个任务上都比微调好。
3. Prefix-tuning also extrapolates better to tables (for table-to-text) and
   articles (for summarization) with unseen topics.
4. 只训练embedding层的参数没有给每层都加prefix参数好