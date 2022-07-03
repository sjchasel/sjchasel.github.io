---
layout: post
title: 【学习笔记】迈向大规模高效自然语言处理
categories: 学习笔记
keywords: NLP, 讲座
mathjax: true
---


# Introduction

![](/images/blog/towards.png)

![](/images/blog/towards2.png)

自然语言的表示从规则式转成了分布式的，维度变低。
这很符合神经学的一些发现，大脑里也有这样的现象：相近的词存在相同的脑区里。

![](/images/blog/towards4.png)

如何设计模型，需要结合语言的结构。语言的结构有几个特点：
1. 它首先是一个序列
2. 序列中又蕴含着层次结构
3. 具有递归的性质

仅从这几个特点理解自然语言，都有着自己的缺点。
1. Long-term Dependency：有些词在序列上相隔很远，但是意思上的关联很近。
2. Polysemy：多义。理解需要结合上下文，甚至是比较长的上下文。

---

Transformer刚出现时，只在机器翻译里取得很好的应用，而其他任务表现一般。因为其他任务数据不多，训练不动这么大的参数量。但BERT这样的预训练模型出现后，迁移到各种下游任务上都有效。

![](/images/blog/towards8.png)

是不是模型更大就更有效？
我们发现，在GPT-3上，随着参数量的提升，few-shot场景提升得很快。
但是大模型训练的开销非常大。

---

![](/images/blog/towards11.png)

想要加速一下大模型，首先可以看大模型分为几个生命阶段：
1. Architecture：设计更有效的架构。
2. Pre-Training：硬件、系统的设计、分布式计算。
3. Fine-Tuning：如何把一个预训练好的模型迁移到下游任务。
4. Deploy：部署模型时，我们希望一个模型可以解决多个问题。
5. Inference：仍然可以加速。



---

# Model-Effciency


## Star-Transformer

![](/images/blog/towards13.png)

+ Tranformer
    + 全连接
    + 复杂度：$O(L^2d)$,无法处理长文档
    + 容易过拟合

+ Star-Transformer
    + 复杂度：$O(2Ld)$
    + 引入局部性先验

**引入中继节点代替全连接，降低了模型的复杂度**


以下是一些受Star-Transformer启发的工作/和其思想相似的

![](/images/blog/towards14.png)

## Attention

![](/images/blog/towards15.png)


如何用理论指导来设计attention的连接——归一化的信息负载（NIP）


## Hypercube Transformer mapping

![](/images/blog/towards16.png)

没太明白，把一个序列用三维结构进行折叠？折叠之后进行编号，如果序列很长可以引入更高维度。
有连接的顶点就把它连上，没有的就不连。展开后就变成：

![](/images/blog/towards17.png)

信息传输效率更优。

## Unified Model

![](/images/blog/towards25.png)

想要用一个模型兼具生成能力（BART）和理解能力（BERT），在中文上。


![](/images/blog/towards26.png)


共享编码器，上面橙色的是面向理解的解码器，右边绿色的是面向生成的解码器。解码器比较浅，生成效率提高两倍以上。

# Knowledge Efficiency


如何给预训练模型引入知识：

![](/images/blog/towards30.png)

![](/images/blog/towards31.png)

mask文本、mask知识实体、mask关系

![](/images/blog/towards32.png)

# Parameter Efficiency

迁移到下游任务时。

如果对于每个任务，都需要fine-tune一个bert，是非常（参数）低效的。

## Task Token

![](/images/blog/towards39.png)

让所有任务共享一个模型，但是输入不同 task token 的时候，输出也会不一样。

![](/images/blog/towards40.png)

（左）现在 prompt-tuning/prefix-tuning 就类似这个思想。
（右）还可以加入adaptor结构。

都是加入一些少量可学习的参数。
但是调参的效率都非常低：**low tuning-efficiency: memory- and time-expensive**


# Tuning Efficiency

想要找到高效调参的方法

## 标签调节

![](/images/blog/towards42.png)

![](/images/blog/towards43.png)

Feature Space是一个很大的空间，而标签的空间较小。我们可以调节从标签空间到Feature Space的映射关系。
Feature Inducer: 融合x的信息  
Label Pointer: 指针网络来确定是哪个标签  

特征只需要算一遍，然后就可以保存下来，在第二轮继续使用。

## 无梯度优化

![](/images/blog/towards48.png)


![](/images/blog/towards49.png)


![](/images/blog/towards50.png)


# One Model Fits All

提升部署效率


NLP的任务可以总结成七种范式：

![](/images/blog/towards54.png)


比如把文本分类问题转成文本匹配问题

![](/images/blog/towards55.png)


构造一些有意义的标签，然后让bert进行匹配：

![](/images/blog/towards56.png)

最有名的还是T5，将各种任务变成seq2seq的生成任务。
但是问题是我们很难想到如何将一些非生成任务转成生成任务，大概是需要人工思考如何构造标签。这里也提到了之前写过的两篇关于用生成范式统一ABSA和NER各种子任务的的unified framework。

# Inference Efficiency


![](/images/blog/towards67.png)

![](/images/blog/towards68.png)

这方面比较关注动态路由。动态路由的核心是：PTM是overthinking的。比如在一些任务我们不需要用到那么多层。

![](/images/blog/towards69.png)

那么我们可以在每一层都判断一下模型的confidence，如果已经很自信了，就不再往后走了。training里有early stop，这里inference有early exiting。  
这里的思想是逐层退出，但我们还可以逐个token退出。就是某些token学好后我们就不再更新它了：

![](/images/blog/towards70.png)

这种方法可能更加适合nlp。我们会在每个token上判断它有没有学好。  
这里有个问题是如何预测confidence，参考HashEE等工作。  
预测样本的难度是比较难的，预测的confidence也不是很准，但这个方法已经很有效了。神奇！那么这个问题解决了会不会更有效呢。

# Evaluation of Efficient NLP

需要兼顾精度和效率，提出ELUE score。

![](/images/blog/towards79.png)
![](/images/blog/towards84.png)
![](/images/blog/towards87.png)


提出baseline——ElasticBERT。


# What's Next?

![](/images/blog/towards102.png)

+ 关注非注意力的建模方法，比如图网络。
+ 动态路由是模型设计的一个重要方向
+ 如何将知识嵌入预训练模型
+ 关注上下文学习（In-Context Learning），比prompt更好的学习场景
+ 可靠性：模型的预测是一致的、安全的