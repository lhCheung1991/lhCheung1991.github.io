---
layout: post
title: "Caffe 源码阅读之 innerproduct 层解析"
categories: blogs
tags: deep-learning hpc
---
{:refdef: style="text-align: center;"}
Linghan Cheung
{:refdef}
{:refdef: style="text-align: center;"}
lhcheung1991@gmail.com
{:refdef}
{:refdef: style="text-align: center;"}
(请在文章底部留下宝贵的评论和建议, [Gitment](https://github.com/imsun/gitment)会将留言以issue形式完整保存方便后续查阅)
{:refdef}
## 摘要
---
<br>
&emsp;&emsp;深度学习在计算机视觉领域大放异彩, 许多在传统方法下无法解决的问题正在被一一攻克. 在整个学术社区共同探索深度学习这块新大陆的潜力时, 高昂的计算成本, 新算法的可复现性, 新方法的易用性等都会制约这个探索过程的顺利进行, 此时, 一个高效, 可扩展, 易用的深度学习框架就显得至关重要. $$Caffe^{[1]}$$作为2014开源的深度学习框架, 其使用了 NVIDIA GPU 作为主要计算平台, 获得了非常高效的运行效率, 而且代码组织结构清晰明了, 易于扩展, 因而一经发布就受到了学术社区的欢迎. 虽然现在已经有 PyTorch[2], TensorFlow[3]等更为现代的深度学习框架, 但 Caffe 还是以各项综合优势在深度学习系统领域有不少的簇拥, 许多 state of the art 的算法如物体检测的 RCNN 系列, SSD 等都使用 Caffe 进行实现, 而其以层为单位进行神经网络的搭建的组织方式, 深远地影响了后续的众多深度学习框架, 对于从事深度学习系统开发的工程师而言, Caffe 也在源源不断地提供着养料. 本文的主要工作如下:

1. 分析 innerproduct 在使用链式求导法则求解过程中的前向计算与后向计算的行为;
2. 分析 Caffe 中所使用的 GEMM 函数, 以及使用 GEMM 实现的 innerproduct 的计算行为.

<br>
## innerproduct 的运算原理
---
<br>
&emsp;&emsp;innerproduct 对应神经网络中的全连接层, 因为全连结层的运算实质上是若干的输入向量与权值矩阵中的权值向量做內积的过程, 故 Caffe 将其命名为innerproduct. 对于只有一个输入的全连接层, 其运算过程如下所示:

$$
\vec{y} = \vec{x}W + \vec{b}\tag{1}
$$

&emsp;&emsp;其中 $$\vec{x}$$ 是 $$1 \times K$$ 的行向量, 表示输入; $$W$$ 是 $$K \times N$$ 的权值矩阵, 其每一列为一个权值向量, 对应全连接层中一个神经元的参数; $$\vec{b}$$ 是 $$1 \times N$$ 的偏置项行向量; $$\vec{y}$$ 是 $$1 \times N$$ 的行向量, 表示对应的输出. 由于我们通常使用 $$Stochastic \ Gradient \ Descent^{[4]}$$ 进行神经网络的训练, 所以网络的输入是一个 batch 的数据, 此时上述只有一个输入的运算过程应该修改为:

$$
Y = XW + B\tag{2}
$$

&emsp;&emsp;其中 $$X$$ 是 $$M \times K$$ 的矩阵, 每一个行向量为一个输入; $$W$$ 是 $$K \times N$$ 的权值矩阵, 其每一列为一个权值向量, 对应全连接层中一个神经元的参数; $$Y$$ 是 $$M \times N$$ 的矩阵, 每一个行向量为对应的一个输出. 由于每一个神经元的输出都会加上对应的偏置项, 所以此时 B 为:

$$
B_{M,N} = [1, ..., 1]_{1,M}^{T} \times \vec{b}\tag{3}
$$

&emsp;&emsp;假设我们最后的损失函数为: 

$$
L(X) = Loss(Y, label) = Loss(XW + B, label)\tag{4}
$$

&emsp;&emsp;为了使用 gradient-based 的算法对模型进行优化, 我们必须求得损失函数对参数 $$W, b$$ 的梯度, 即 $$\frac{\partial{Loss}}{\partial{W}}$$ 和  $$\frac{\partial{Loss}}{\partial{b}}$$, 同时我们还需要知道 $$\frac{\partial{Loss}}{\partial{X}}$$, 因为我们使用链式求导法则进行梯度的求解, 此时 $$X$$ 有可能是浅层网络结构的输出, 在对浅层网络结构的参数进行求导时会需要使用到梯度 $$\frac{\partial{Loss}}{\partial{X}}$$ . 所以, 在 innerproduct 的运算中, 重点是下述三个偏导数的求解:

$$
\begin{eqnarray}
\frac{\partial{Loss}}{\partial{W}} & = & \frac{\partial{Loss}}{\partial{Y}} \cdot \frac{\partial{Y}}{\partial{W}}\tag{5} \\
\frac{\partial{Loss}}{\partial{b}} & = & \frac{\partial{Loss}}{\partial{Y}} \cdot \frac{\partial{Y}}{\partial{b}}\tag{6} \\
\frac{\partial{Loss}}{\partial{X}} & = & \frac{\partial{Loss}}{\partial{Y}} \cdot \frac{\partial{Y}}{\partial{X}}\tag{7}
\end{eqnarray}
$$

&emsp;&emsp;对于 $$Equation \ (5)$$, 我们将其进行展开, 则对 $$\frac{\partial{Y}}{\partial{W}}$$ 有:

$$
\begin{eqnarray}
\because Y_{i,j} & = & \sum_{k=1}^{K}{X_{i,k} \cdot W_{k,j}}\tag{8} \\
\therefore \frac{\partial{Y_{i,j}}}{\partial{W_{k,j}}} & = & X_{i,k}, k=1, 2, ..., K\tag{9}
\end{eqnarray}
$$

&emsp;&emsp;而对于 $$W_{k,j}$$, 我们有:

$$
\frac{\partial{Y_{i,j}}}{\partial{W_{k,j}}} = X_{i,k}, i=1, 2, ..., M\tag{10}
$$

&emsp;&emsp;由 $$Equation \ (8)(9)(10)$$, 我们可以看出, $$W_{k,j}$$ 在 innerproduct 中的运算只与 $$X_{i,k}$$ 相关, 故在计算 $$\frac{\partial{Y}}{\partial{W_{k,j}}}$$ 时亦是如此, 因而有:

$$
\frac{\partial{Y}}{\partial{W_{k,j}}} = \frac{\partial{Y_{i,j}}}{\partial{W_{k,j}}} = X_{i,k}, i=1, 2, ..., M\tag{11}
$$

&emsp;&emsp;由 $$Equation \ (5)(11)$$, 我们可以得到:

$$
\begin{eqnarray}
\frac{\partial{Loss}}{\partial{W_{k,j}}} & = & \frac{\partial{Loss}}{\partial{Y}} \cdot \frac{\partial{Y}}{\partial{W_{k,j}}} \\
& = & \sum_{i=m}^{M}{\frac{\partial{Loss}}{\partial{Y_{i,j}}} \cdot \frac{\partial{Y_{i,j}}}{\partial{W_{k,j}}}} \\
& = & \sum_{i=m}^{M}{\frac{\partial{Loss}}{\partial{Y_{i,j}}} \cdot X_{i,k}}\tag{12}
\end{eqnarray}
$$

&emsp;&emsp;由 $$Equation \ (5)(12)$$, 我们可以得到:

$$
\frac{\partial{Loss}}{\partial{W}} = [[\frac{\partial{Loss}}{\partial{Y}}]^{T} \cdot X]^{T}\tag{13}
$$

&emsp;&emsp;对于 $$Equation \ (6)$$, 我们有类似的运算过程, 最后可得:

$$
\frac{\partial{Loss}}{\partial{b}} = [[\frac{\partial{Loss}}{\partial{Y}}]^{T} \cdot [1, ..., 1]_{1,M}^{T}]^{T}\tag{14}
$$

&emsp;&emsp;对于 $$Equation \ (7)$$, 我们由$$(8)$$对其进行展开, 则对 $$\frac{\partial{Y}}{\partial{X}}$$ 有:

$$
\begin{eqnarray}
\frac{\partial{Y_{i,j}}}{\partial{X_{i,k}}} & = & W_{k,j}, k=1, 2, ..., K\tag{15}
\end{eqnarray}
$$

&emsp;&emsp;而对于 $$X_{i,k}$$, 我们有:

$$
\frac{\partial{Y_{i,j}}}{\partial{X_{i,k}}} = W_{k,j}, j=1, 2, ..., N\tag{16}
$$

&emsp;&emsp;由 $$Equation \ (15)(16)$$, 我们可以看出, $$X_{i,k}$$ 在 innerproduct 中的运算只与 $$W_{k,j}$$ 相关, 故在计算 $$\frac{\partial{Y}}{\partial{X_{i,k}}}$$ 时亦是如此, 因而有:

$$
\frac{\partial{Y}}{\partial{X_{i,k}}} = \frac{\partial{Y_{i,j}}}{\partial{X_{i,k}}} = W_{k,j}, j=1, 2, ..., N\tag{17}
$$

&emsp;&emsp;由 $$Equation \ (7)(17)$$, 我们可以得到:

$$
\begin{eqnarray}
\frac{\partial{Loss}}{\partial{X_{i,k}}} & = & \frac{\partial{Loss}}{\partial{Y}} \cdot \frac{\partial{Y}}{\partial{X_{i,k}}} \\
& = & \sum_{j=1}^{N}{\frac{\partial{Loss}}{\partial{Y_{i,j}}} \cdot \frac{\partial{Y_{i,j}}}{\partial{X_{i,k}}}} \\
& = & \sum_{j=1}^{N}{\frac{\partial{Loss}}{\partial{Y_{i,j}}} \cdot W_{k,j}}\tag{18}
\end{eqnarray}
$$

&emsp;&emsp;由 $$Equation \ (7)(18)$$, 我们可以得到:

$$
\frac{\partial{Loss}}{\partial{X}} = \frac{\partial{Loss}}{\partial{Y}} \cdot W^{T} \tag{19}
$$

&emsp;&emsp;综上, 我们得到 $$Equation \ (2)$$ 和 $$Equation \ (13)(14)(19)$$, 分别代表了 Caffe 中 innerproduct 层在前向计算和后向计算时所遵循的运算原理, 我们进行一下简单的归总.

$$
\begin{eqnarray}
Y & = & XW + B\tag{2} \\
\frac{\partial{Loss}}{\partial{W}} & = & [[\frac{\partial{Loss}}{\partial{Y}}]^{T} \cdot X]^{T}\tag{13} \\
\frac{\partial{Loss}}{\partial{b}} & = & [[\frac{\partial{Loss}}{\partial{Y}}]^{T} \cdot [1, ..., 1]_{1,M}^{T}]^{T}\tag{14} \\
\frac{\partial{Loss}}{\partial{X}} & = & \frac{\partial{Loss}}{\partial{Y}} \cdot W^{T} \tag{19}
\end{eqnarray}
$$

<br>
## Caffe 中 innerproduct 的实现
---
<br>

<br>
## 引用
---
<br>
[1] Jia, Y., Shelhamer, E., Donahue, J., Karayev, S., Long, J., & Girshick, R., et al. (2014). Caffe: Convolutional Architecture for Fast Feature Embedding. Acm International Conference on Multimedia (pp.675-678). ACM. [http://caffe.berkeleyvision.org](http://caffe.berkeleyvision.org)

[2] pytorch@github(2018), Tensors and Dynamic neural networks in Python with strong GPU acceleration. [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)

[3] tensorflow@github(2018), Computation using data flow graphs for scalable machine learning. [https://github.com/tensorflow/tensorflow](https://github.com/tensorflow/tensorflowg)

[4] wikipedia(2018), Stochastic gradient descent. [https://en.wikipedia.org/wiki/Stochastic_gradient_descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)