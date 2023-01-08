---
layout: post
title: 【学习】对比学习论文串烧
categories: 学习笔记
keywords: CV, 对比学习
mathjax: true
---

# 百花齐放阶段

## InstDist

Unsupervised Feature Learning via Non- Parametric Instance Discrimination  
是MoCo中提到的memory bank的方法  

贡献：
+ 提出个体判别任务，和NECloss做对比学习取得了不错的无监督表征学习的结果。
+ 提出用其他数据结构存储负样本，并对这些特征进行更新的方法。

Motivation：对豹子图片的分类，概率大的都是和豹相关的类别，比如猎豹、雪豹、美洲豹，概率小的都是完全不相关的类别。因为豹子的图片确实都长的非常像。因此作者将传统的类别推广到每张图片都是一个类别。  

Method：
+ 通过对比学习的方式去区分每张图片
+ 将ImageNet的每张图片编码成128维的特征向量，存储起来，作为memory bank。做对比学习的时候从这里面抽取4096个样本作为负样本进行对比学习。每次学习后，编码器的参数更新了，就再计算一次当前batch里的样本的特征，以动量的方式更新memory bank。

## InvaSpread

Unsupervised Embedding Learning via Invariant and Spreading Instance Feature  
可以说是SimCLR的前身

贡献（新方法）：
+ 没有用额外的数据结构存储大量的负样本，正负样本来源于一个batch
+ 只用了一个编码器来编码正负样本，正负样本之间的特征具有一致性。

Motivation：
+ Invariant：同一个图片的特征在特征空间应该接近
+ Spreading：不同图片的特征应该分开

Method：
选取个体判别任务作为代理任务。假设一个batch中有256张图片，经过数据增强又得到了256张图片。对于一张图片，正样本是增强后的那张图片，负样本是除了锚样本和正样本以外的所有样本，也就是（256-1）*2个。  
这样做的好处是可以只使用一个编码器做端到端的训练，保证了每次正负样本特征的一致性。

## CPC

Representation Learning with Contrastive Predictive Coding

贡献：
提出了新的代理任务

Method：有一系列具有顺序的输入x，每个都通过编码器得到一系列输出。前t个特征再输入给类似RNN这样的时序模型，获得一个上下文向量。用上下文向量去预测t步后面的特征，作为锚样本。正样本就是t步后面的输入x经过编码器获得的特征向量，负样本是任意其他的特征。

## CMC

Contrastive Multiview Coding

一个物体通过不同视角被观测到的特征，虽然有不同，但是重要的特征是共享的。因此想要模型学习这种视角不变的重要特征。

因此一个物体很多的视角都可以被当成正样本，负样本就是任何不配对的视角。

让大家发现对比学习可以灵活地用在不同领域。但是在多模态上进行应用时，对于每个模态都需要用一个编码器去获得特征，计算资源消耗大。比如CLIP就用了BERT去编码文本，ViT去编码图像。ICLR22有篇MA-CLIP提出用一个Transformer去编码不同模态的数据，也取得了很好的效果。让我想起了KV-PLM，一个BERT去编码SMILES和文本，但它可能是把SMILES和文本当成不同语言。



# CV双雄

## MoCo

主要贡献：
+ 把之前对比学习的方法归纳成字典查询的问题
+ 提出了队列和动量编码器两个改进，形成一个又大又一致的字典，能够帮助更好的对比学习

## SimCLR

A Simple Framework for Contrastive Learning of Visual Representation

正负样本的构造和InvaSpread一致，也是只有一个编码器。它和之前工作不一样的地方在于，编码器编码出特征后，又过了一个projector（mlp+relu），这个全连接层在ImageNet上直接提了10个点。  

只有训练的时候才用这个projector，在下游任务应用时不用，所以仍然可以和之前的工作进行公平对比。

除此之外，用了更多的数据增强技术、更大的batch、更长的训练时间。

<!-- ## SwAV

Motivation：给定一张图片，从不同视角生成不同的特征，希望从一个视角得到的特征可以生成另一个视角得到的特征。

Method：
+ 对比学习+聚类
+ 之前的对比学习需要和大量的负样本的特征拉远，太耗费资源。本文的改进是想和聚类的中心进行对比 -->

# 没有负样本

## BYOL

Bootstrap Your Own Latent A New Approach to Self-Supervised Learning 

一个视角的特征去预测另一个视角的特征。两个视角的特征用到的编码器结构相同，参数不同。被预测的那个分支的编码器参数是预测分支编码器参数的动量更新。

## SimSiam

在之前对比学习的工作中，性能看起来总是被一个一个的小点堆起来的。因此这篇论文进行分析，化繁为简，不需要负样本（也就不需要大batch size），也不需要动量编码器。


# Transformer

## MoCo v3

提出稳定ViT训练的trick：原始模型对图片tokenize成patch是一个可训练的全连接层，把这个模块冻住不训练，就可以加大batch size也能稳定训练整个模型。

## DINO

MoCo v3 + （BYOL + batch norm）

