---
layout: post
title: 【学习笔记】分子中的不变性与等变性
categories: 学习笔记
keywords: Bio
mathjax: true
---



1. 介绍相关的两个概念：不变性与等变性。
2. Uni-Mol捕捉分子的旋转、平移不变性的思路。
3. Uni-Mol的公式细节。



# 不变性与等变性

![](/images/blog/in-eq.png)

这是个很直观的例子。左边是不变性（invariance），图像经过S这个变换（平移）前后，经过函数f处理得到的结果都一样。f很好地建模了图像的平移不变性。

右边展示的是等变性（equivariance）。图像先经过S变换，再经过f处理，得到的结果与先经过f处理，再经过S变换得到的结果相同。也就是f和S这两种变换的顺序是可以互换的。

我们可以把左边的图中的f看成是CNN最后的池化操作。无论是最大池化还是平均池化，无论内部的特征的顺序如何变化，池化后的结果都一样。因此池化后这种在全局级别上的特征是具有不变性的。  
而右边的f可以看成是CNN中的卷积操作。（虽然现在的CNN的卷积只具有近似的等变性）图像如何平移、旋转，输出的特征也会进行相应的平移、旋转。因此这种细粒度的特征是等变的。

回到分子领域，对于原子/键级别的特征，我们希望它们具有的是平移、旋转等变性。但对于整个分子来说，它应该具有平移、旋转不变性。我们的模型通常都是以原子这样的局部特征作为基本单位，因此它应该具有等变性。当一个东西具有等变性时，我们也能很容易将其转化成具有不变性，比如做最大/平均池化，来获得整个分子的特征。

# Uni-Mol的实现思路

参考Transformer的相对位置编码。以1D的文本来看，相对位置编码由于位置特征是和两个字符联系在一起的，因此整个文本进行平移的时候，位置特征并不会产生变化，因此具有平移的不变性。  

在建模3D分子时，Uni-Mol将输入的每个原子离散的坐标，转换成了原子两两之间的相对位置。这样，如果分子进行了平移、旋转等变换，虽然输入的原始坐标数据不同了，但是原子之间的相对特征是不会发生改变的，相当于模型处理的仍然是同一个分子。  
（不知道最早是哪篇论文先提出用相对表示来实现模型的等变性/不变性能力，只知道Uni-Mol继续简化了等变性图神经网络的公式）


在之前的一些研究中，为了让模型学习到分子的平移、旋转不变性，会预先对分子数据进行数据增强，也就是进行很多次平移/旋转操作，但输出都一样，来让模型学习到这种能力。这种方式需要大量的计算资源。


# Uni-Mol的实现细节

## 原子对表示

输入的信息是每个原子的三维坐标，由此可以计算原子对之间的相对距离。但跟序列用整数表示位置和距离不同，原子坐标是3维空间中的连续变量。所以用欧式距离来表示原子对之间的相对距离，然后通过仿射变换和高斯核函数将它们映射到特征空间中。

放射变换的作用就是，在对原子对距离进行变换的时候，根据原子对类型的不同有不同的变换。

$$
\boldsymbol{p}_{i j}=\left\{\mathcal{G}\left(\mathcal{A}\left(d_{i j}, t_{i j} ; \boldsymbol{a}, \boldsymbol{b}\right), \mu^k, \sigma^k\right) \mid k \in[1, D]\right\}\\ \quad \mathcal{A}(d, r ; \boldsymbol{a}, \boldsymbol{b})=a_r d+b_r
$$



## 编码

参考Alpha-Fold，在每层中，都有两个信息传递的过程：从atom到pair和从pair到atom。

为了更新坐标的表示，也就是需要更新原子对的特征，我们将信息从atom传到pair。

$$
\boldsymbol{q}_{i j}^{l+1}=\boldsymbol{q}_{i j}^l+\left\{\frac{\boldsymbol{Q}_i^{l, h}\left(\boldsymbol{K}_j^{l, h}\right)^T}{\sqrt{d}} \mid h \in[1, H]\right\}
$$


其中 $q_{ij}^l$ 表示第l层原子i、j的原子对表示，H是attention头的个数。因此这个公式其实就是原子i和原子j的特征相乘去更新它们的原子对表示。

也想用空间信息去更新原子表示，所以还有一个pair到atom的信息传递过程。

$$
\operatorname{Attention}\left(\boldsymbol{Q}_i^{l, h}, \boldsymbol{K}_j^{l, h}, \boldsymbol{V}_j^{l, h}\right)=\operatorname{softmax}\left(\frac{\boldsymbol{Q}_i^{l, h}\left(\boldsymbol{K}_j^{l, h}\right)^T}{\sqrt{d}}+\boldsymbol{q}_{i j}^{l-1, h}\right) \boldsymbol{V}_j^{l, h}
$$

和正常的attention公式不一样的地方就在于，qk相乘之后还加上了这个原子对的特征表示，再和v相乘。这也和transformer的相对位置编码的处理方式一致。


## 坐标预测

坐标预测的部分也参考等变性图神经网络（EGNN），但和EGNN不同的是：EGNN在每层都会用pair表示来预测一下坐标，但是Uni-Mol只在最后一层，并且使用的是第一层的pair表示和最后一层pair表示的差距（称为delta pair representation）。因此Uni-Mol的计算效率优于EGNN。


EGNN的每层都会经过以下计算：

$$
\begin{aligned}
\mathbf{m}_{i j} & =\phi_e\left(\mathbf{h}_i^l, \mathbf{h}_j^l,\left\|\mathbf{x}_i^l-\mathbf{x}_j^l\right\|^2, a_{i j}\right) \\
\mathbf{x}_i^{l+1} & =\mathbf{x}_i^l+C \sum_{j \neq i}\left(\mathbf{x}_i^l-\mathbf{x}_j^l\right) \phi_x\left(\mathbf{m}_{i j}\right) \\
\mathbf{m}_i & =\sum_{j \neq i} \mathbf{m}_{i j} \\
\mathbf{h}_i^{l+1} & =\phi_h\left(\mathbf{h}_i^l, \mathbf{m}_i\right)
\end{aligned}
$$

第一个公式是计算节点i和节点j之间边的特征，如果这个系统进行了旋转/平移，因为xi和xj的相对距离不变，因此mij也不会发生变化。mij是具有不变性的。  
第二个公式是节点i的坐标更新方式，其中$φ_x$是将边的特征mij转换成标量，作为i与j相对变化的权重，C就是1/j的个数。也就是说xi的变动是与节点i相关的所有相对变化的加权平均值。这个公式使得坐标具有了等变性，EGNN中的公式证明如下：

$$
\begin{aligned}
Q \mathbf{x}_i^l+g+C \sum_{j \neq i}\left(Q \mathbf{x}_i^l+g-Q \mathbf{x}_j^l-g\right) \phi_x\left(\mathbf{m}_{i, j}\right) & =Q \mathbf{x}_i^l+g+Q C \sum_{j \neq i}\left(\mathbf{x}_i^l-\mathbf{x}_j^l\right) \phi_x\left(\mathbf{m}_{i, j}\right) \\
& =Q\left(\mathbf{x}_i^l+C \sum_{j \neq i}\left(\mathbf{x}_i^l-\mathbf{x}_j^l\right) \phi_x\left(\mathbf{m}_{i, j}\right)\right)+g \\
& =Q \mathbf{x}_i^{l+1}+g
\end{aligned}
$$

其中Qx+g代表对坐标xi进行平移、旋转的变换。左边的式子是先进行坐标变换，再进行第l层的编码，可以证明与右边——对第l层编码的结果$x^{l+1}$进行坐标变化的结果相等。因此模型在处理坐标上就具有旋转、平移的等变性。  
最后两个公式和普通的GNN一样。

Uni-Mol在最后一层进行坐标更新：

$$
\hat{\boldsymbol{x}}_i=\boldsymbol{x}_i+\sum_{j=1}^n \frac{\left(\boldsymbol{x}_i-\boldsymbol{x}_j\right) c_{i j}}{n}\\ \quad c_{i j}=\operatorname{ReLU}\left(\left(\boldsymbol{q}_{i j}^L-\boldsymbol{q}_{i j}^0\right) \boldsymbol{U}\right) \boldsymbol{W}
$$

思路与EGNN相同，但是计算了从最开始到最后一层的相对位置总的变化程度，不需要每层都更新坐标，减小计算量。

# 参考

+ E(n) Equivariant Graph Neural Networks
+ Uni-Mol: A Universal 3D Molecular Representation Learning Framework
+ [Deep Learning – Equivariance and Invariance](https://www.doc.ic.ac.uk/~bkainz/teaching/DL/notes/equivariance.pdf)
+ [Deep Learning for Molecules and Materials Book - 9. Input Data & Equivariances](https://dmol.pub/dl/data.html)