---
layout: post
title: 【论文阅读】Text is no more Enough!A Benchmark for Profile-based Spoken Language Understanding
categories: ["NLP", "论文笔记"]
mathjax: true
# key: text_not_enough
---



Text is no more Enough!

A Benchmark for Profile-based Spoken Language Understanding



# Background—— spoken language understanding（SLU）

![img](http://www.xiaojiejing.com/wp-content/uploads/2018/12/t7zxx7q9ni.png)

## intent detection

对用户的输入进行分类。

## slot filing

槽填充是一个序列标注任务。将用户输入的文本中的一些有意义的实体识别出来，从而将用户隐式的意图转换成显式的指令让计算机理解。

比如对于一句话：今天深圳的天气怎么样。我们可以将其标注成$B_{date}I_{date}B_{location}I_{location}OOOOOO$

对于处理“询问天气这个意图”，就具备了日期和地点两个关键信息。

# Motivation

> 传统的slu模型接收用户的纯文本输入，输出intent和slots。但这样简单的结构不足以满足现实世界的复杂要求。有时候用户的输入是很模棱两可的，模型无法理解。



传统的SLU模型在槽填充和意图识别两个任务上达到了96%、99%的精度，在1990年提出的ATIS数据集上。但现实情况中的对话系统效果还是不能让人完全满意。

因为学术界的这些研究的假设是仅依靠用户的纯文本输入就能进行意图识别和槽填充。而现实中，用户的输入很可能是非常模棱两可的，仅从用户的输入是无法进行这两种任务的。

比如：当用户说play monkey king时。仅靠这句话是无法知道用户想做什么的。因为monkey king可能是一首歌，也可能是卡通片。

但作者认为加上一些profile信息可以解决这个问题。因为如果我们知道这个用户在跑步，那么他的意图很可能是放歌，而不是放卡通片。

# Method

## Overview

作者基于以上情况提出了新的任务—— **Pro**fifile-based **S**poken **L**anguage **U**nderstanding（PROSLU）。模型不仅依赖于纯文本，也依赖于个人的信息来预测正确的intent和slots。

作者同样提出了一个数据集。人工标注、中文、大于5k个utterance和相应的profile信息。profile信息包括：

1. Knowledge Graph（KG）：相互关联的实体和丰富的属性。使用公开的大型KG如CNDBpedia、OwnThink构建。
2. User Profile（UP）：用户的设置和信息
3. Context Awareness（CA）：用户的状态和环境的信息

![截屏2022-01-11 上午12.02.23.png](https://s2.loli.net/2022/01/11/MNE8mbs396UDTBh.png)



作者提出了一个多层级的knowledge adapter（是2021 acl findings https://aclanthology.org/2021.findings-acl.121.pdf 吗？）去利用个人信息。当模型的输入——用户的utterance是模棱两可的，sota纯文本模型就失效了。但作者的模型在意图识别（sentence-level）和槽填充（token-level）两个任务上都表现很好。

## Dataset

由于作者针对的是「用户输入的文本经常模棱两可」这个问题，要搜集这样的数据，先**对模棱两可/语义模糊下个定义**。这又分为两种情况：

1. Ambiguous Mentions：大概是模糊的指代？就是说提到的实体可以指很多东西。比如Monkey King可以是歌、小说、电视剧。
2. Ambiguous Descriptions：模糊的描述。人们说话会省去上下文/场景的，导致对话系统无法仅仅从一句话就知道人们的指令含义。比如“我要买张去上海的票”，不知是机票还是火车票。



对于**标签**的设置

1. Intent detection：Intent Group就是那个含义丰富的实体所有可能的意图。比如{PlayMusic, PlayVideo,PlayAudioBook}
2. Slot Filing：Slot label就是和intent相关的。作者没细说。



两个模糊场景下的数据构建：

1. 描述模糊：随意选择intent，slot label根据这个intent指定。个人信息采用启发式方法？看起来很人工指定...
2. 指代模糊： hard-coded heuristics?

​	给了intent和slot之后，人工去写一些模糊的utterance。

## Model

### General SLU Model

一个Shared Encoder对utterance进行编码。

Intent Detection Decoder：句子的表示做softmax分类

Slot Filling Decoder：单向LSTM，相当于对每个词分类来进行标注。

### Supporting Information Representations

同一个实体和它的不同指代被连成了一个序列，通过BiLSTM获得一系列hidden state。最后一个h为KG的表示。

UP：对同一类型不同实体的偏好，总和为1，不同类型的拼在一起。比如一个人对music、video、audiobook、subway、bus、driving的偏好拼起来是[0.5,0.3,0.2,0.4,0.1,0.5]，通过线性变换转成特征向量。

CA：包括一个人的movement state、posture、geographic location等信息。对于movement state，一个one-hot向量指示这个人的状态。比如walking、running、stationary，[0,1,0]表示这个人在running。应该也是几个拼起来，再同上变换。

### Multi-level Knowledge Adapter

Knowledge Adapter： attention fusion mechanism。对三种表示（KG、UP、CA）做个注意力求和。但作者没说query是啥...

Sentence-level Knowledge Adapter for Intent Detection：句子表示作为query，对三种表示求和后再和句子表示拼起来做分类。

Word-level Knowledge Adapter for Slot Filling：每步的embedding $e_t$作为query对三种表示做注意力求和得到$s_t^{info}$，$[e_t;intent\ embedding;y_{t-1}(上一步的slot embedding)]+s_t^{info}$作为输入过lstm，每个hidden state做分类来进行序列标注。

# Experiment

![截屏2022-01-15 下午2.25.39.png](https://s2.loli.net/2022/01/15/AnN9DCcU1vmtHM5.png)



