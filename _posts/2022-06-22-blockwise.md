---
layout: post
title: 【论文阅读】Blockwise Parallel Decoding for Deep Autoregressive Models
categories: 论文笔记
keywords: NLP
mathjax: true
---

NIPS 2018  
code: Tensor2Tensor  
一种加速自回归模型贪婪解码的技巧

# Motivation

循环神经网络、卷积神经网络、自注意力机制，所需要的计算都会随着序列长度的增加上升，并且在生成序列的时候是需要一个词一个词地生成。  
本文提出了加速自回归模型的技巧，精度不变时可增速2x，稍微牺牲一些精度可以增速7、8x。

# Background——Greedy Decoding

我们想要实现的是从 $x=(x_1, x_2, ..., x_n)$ 到 $y=(y_1, y_2, ..., y_m)$ 。按照序列的顺序生成时，我们训练的模型可以得到给定x和已知一部分y时，下一个y在词表上的概率分布。

$$\log p(y|x)=\sum_{j=1}^{m-1}\log p(y_{j+1}|y_{\leq j},x)$$

在得到了 $y_{j+1}$ 的概率分布后，可以简单地使用greedy decoding，即取概率最大的那个词作为结果： $y^{*}=argmax_y p(y|x)$ 。这样我们就知道了前j+1个token，将它们输入模型继续计算j+2个token。  

这样一直取max，再输入模型的过程是比较耗时的。

# Method

## Blockwise Parallel Decoding


首先，我们会训练一堆辅助模型。相对的，我们正常的自回归模型叫做base model，p，也为 $p_1$ 。辅助模型为 $p_2, p_3, ..., p_k$ 。下标可以看作为，这个模型在给定x和 $y_{\leq j}$ 的情况下，预测的是后面第几个token。

有了这些模型，我们的预测结果可以分为三步：

1. Predict：使用base model和一堆辅助model去预测一个序列。但它们是不同的模型，因此是并行预测的。我们预测出了从 $y_{j+1}$ 到 $y_{j+k}$ k个token。  
2. Verify：因为在训练的时候，第一步得到的“序列”之间是没有建模它们的序列关系的。预测出来的词不一定准确。我们需要在这一步判断接不接受以上各个辅助模型的预测结果。这里我们使用的是base model来看从 $p_2$ 预测出来的每个token是不是给了前面token的greedy decoding结果。
如果verfiy的结果是✅✅❌✅，我觉得是截断到前面连续正确的token就行。因为第三个是错误的话，输入了错误的前序序列去判断第四个，也不一定正确。
假设这一步我们接受了 $\hat{k}$ 个token。
3. Accept：$j\leftarrow j+\hat{k}$ ，即又从第j+ $\hat{k}$ 个token开始预测。


原文中关于accept到哪的说明：

> By stopping early if the base model and the proposal models start to diverge in their predictions


整个结果可以展示为下图：

![process](/images/blog/blockwise1.png)

分析一下这个过程所用的时间和自回归模型的时间。
1. 在predict中。每个token的预测是由不同模型来完成的。因此预测k个token只需要并行一步，而在自回归模型中要预测k个token，只能每个都等待前面的预测出来才能进行预测。因此需要k步。
2. 在verify中。因为我们已经有整个序列了（前序序列和后面k个token），因此对每个token的verify是可以并行的，但都是由p/p1 model完成。在普通的自回归过程中没有这一步。
3. 在accept中。没什么特别的，只是更新前序序列。


因此，我们是需要交替两次模型的调用。第一次是在predict里，p1到pk同时使用，第二次是在verfiy里，只用p1。就算预测出来的k个token全都可以被接受，要预测m个token的序列我们需要2m/k步。

## Combined Scoring and Proposal Model

在以上的三个基本步骤的基础上，我们还可以继续加速，从2m/k到m/k+1，那就是将第n次verify和第n+1次predict结合：

![process2](/images/blog/blockwise2.png)


> This can be implemented for instance by increasing the dimensionality of the final projection layer by a factor of k and computing k separate softmaxes per position. 

原文是这样描述的，不过看了它给的模型图就很好懂了！

![model](/images/blog/blockwise_model.png)

先将输出放大三倍，再分成三等份，加上残差连接，去预测3个连续的token。如果第j+3个token是不可接受的，那么我们在验证j+3个token的时候就已经获得了以j+2个token为前序序列时的预测结果，也就是下一步的prediction。第一次迭代还是需要做的，因此最好的情况是m/k+1。

## Approximate Inference

还可以继续加速，但是需要损失一些精度了。
到目前为止，因为我们用p1去verify每个预测的token，其实最后的结果是和简单地使用自回归模型进行贪婪解码是一样的。但我们可以稍微放宽verify的标准，这样我们预测出来的token会有较多的被接受，就可以速度更快地预测完所有token。

### Top-k Selection

在verify的时候，我们可以不要求它和argmax的结果一样，而是在前k个就行，即：

$$\hat{y}_{j+1}\in top-k_{y_{j+1}}p1(y_{j+1}|\hat{y}_{\leq j+i-1},x)$$

### Distance-Based Selection

我们也可以定义一个距离的超参，如果输出和argmax的结果相差不大，那么仍然可以接受。

### Minimum Block Size

预测出来的第一个token还是算是greedy decoding的，因此肯定正确。从第二个token开始才可能是错的。但是为了保证速度，我们可以规定必须最少接受多少token（最小为2）。

# Experiment

## Implementation 

1. 预训练模型的参数是冻结还是全部微调？
2. 蒸馏：用beam search的结果教模型

## Machine Translation


![nmt](/images/blog/blockwise_exp1.png)

不考虑使用蒸馏或者fine tune，k=6的时候精度稍微好一些并且几乎加速了两倍。如果愿意牺牲一些精度，还可以加速更多倍。使用蒸馏和fine tune效果都更好了，因为更多的token被接受，速度就更快了。


## Image Super-Resoluti


![image](/images/blog/blockwise_exp2.png)


## Wall-Clock Speedup

![time](/images/blog/blockwise_exp3.png)

真正被节约的时间。