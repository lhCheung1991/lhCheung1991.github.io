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
&emsp;&emsp;深度学习在计算机视觉领域大放异彩, 许多在传统方法下无法解决的问题正在被一一攻克. 在整个学术社区共同探索深度学习这块新大陆的潜力时, 高昂的计算成本, 新算法的可复现性, 新方法的易用性等都会制约这个探索过程的顺利进行, 此时, 一个高效, 可扩展, 易用的深度学习框架就显得至关重要. $$Caffe^{[1]}$$作为2014开源的深度学习框架, 其使用了 NVIDIA GPU 作为主要计算平台, 获得了非常高效的运行效率, 而且代码组织结构清晰明了, 易于扩展, 因而一经发布就受到了学术社区的欢迎. 虽然现在已经有 $$PyTorch^{[2]}$$, $$TensorFlow^{[3]}$$等更为现代的深度学习框架, 但 Caffe 还是以各项综合优势在深度学习系统领域有不少的簇拥, 许多 state-of-the-art 的算法如物体检测的 RCNN 系列, SSD 等都使用 Caffe 进行实现, 而其以层为单位进行神经网络的搭建的组织方式, 深远地影响了后续的众多深度学习框架, 对于从事深度学习系统开发的工程师而言, Caffe 也在源源不断地提供着养料. 本文的主要工作如下:

1. 分析 innerproduct 在使用链式求导法则求解过程中的前向计算与后向计算的演算原理;
2. 分析 Caffe 中所使用的 $$GEMM(GEneral \ Matrix \ Multiplication)^{[6]}$$ 函数, 以及使用 GEMM 实现的 innerproduct 的计算行为.

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
&emsp;&emsp;由第一节的讨论我们可以发现, innerproduct 的核心操作就是 GEMM. 事实上, 在现代的神经网络理论和深度学习框架中, 其核心的计算操作本质上都是 GEMM 操作$$^{[5][6][7]}$$, 所以, Caffe 中 innerproduct 层的实现是围绕着使用 GEMM 来展开的.

&emsp;&emsp;在 [caffe/src/caffe/util/math_functions.cpp](https://github.com/BVLC/caffe/blob/master/src/caffe/util/math_functions.cpp#L13) 中, Caffe 对 GEMM 函数进行了封装, 如下所示:
```c++
template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}
```
此处我们仅对单精度浮点数的运算函数进行解析. 在上述函数 `void caffe_cpu_gemm<float>(...)` 中, 调用了 `cblas.h` 中的函数 `void cblas_sgemm(...)`, 其实现的计算为 $$C = \alpha \cdot A \cdot B + \beta \cdot C$$, 即将矩阵 A, B 做矩阵乘法并乘以系数 $$\alpha$$, 再加上 $$\beta$$ 乘以矩阵 C 后得到的矩阵, 其中, 矩阵 A 为 $$M \times K$$, 矩阵 B 为 $$K \times N$$, 矩阵 C 为 $$M \times N$$ $$^{[8]}$$. 在 [caffe/src/caffe/util/math_functions.cu](https://github.com/BVLC/caffe/blob/master/src/caffe/util/math_functions.cu#L13), Caffe 提供了使用 cuBLAS 的 GPU 版本实现, 其原理与 CPU 版本相仿, 如下所示, 有一点需要注意的是, cuBLAS 采用与 Fortran 相同的列优先存储方式, 所以使用上需要注意行列数的变换:
```c++
template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}
```

&emsp;&emsp;在 [caffe/include/caffe/layers/inner_product_layer.hpp](https://github.com/BVLC/caffe/blob/master/include/caffe/layers/inner_product_layer.hpp#L42) 中, 为了对 innerproduct 的矩阵运算进行相应的配置, 类 `InnerProductLayer` 在继承类 `Layer` 后, 增加了如下的数据成员, 其中, `M_, K_, N_` 与 $$Equation \ (2)$$ 中矩阵参数的意义是一致的:
```c++
  int M_;
  int K_;
  int N_;
  bool bias_term_;
  Blob<Dtype> bias_multiplier_;
  bool transpose_;  ///< if true, assume transposed weights
```

Caffe 中的层都要继承类 `Layer`, 并实现类中函数 `void SetUp(...)` 中所调用的用于初始化本层配置的虚函数, 如下所示, 所以, 在类 `InnerProductLayer` 的实现中, 初始化工作主要是在函数 `void LayerSetUp(...)`, `void Reshape(...)` 中完成:
```c++
  void SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CheckBlobCounts(bottom, top);
    LayerSetUp(bottom, top);
    Reshape(bottom, top);
    SetLossWeights(top);
  }
```

在 [caffe/src/caffe/layers/inner_product_layer.cpp](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/inner_product_layer.cpp#L9) 中, 类 `InnerProductLayer` 实现了虚函数 `void LayerSetUp(...)`:
```c++
template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // 使用从 prototxt 中读取到的 LayerParameter 进行初始化, 此时输出矩阵 Y 的列数就是
  // N_ = num_output 的值
  const int num_output = this->layer_param_.inner_product_param().num_output();
  ......;
  N_ = num_output;
  // 将输入的 Tensor 从指定的 axis 开始全部拍平, 这样就能得到输入矩阵 X 的行数为 
  // M_ = batchsize, 输入矩阵 X 的列数为 K_
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  K_ = bottom[0]->count(axis);
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    ......;
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      // 此时需要注意的是默认情况下, 参数矩阵 W 为 N_ x K_, 而不是 Equation (2)
      // 中的 K x N, 也就是说, 参数矩阵 W 默认采用的是转置形式, 所以, 在前向计算时需要
      // 将参数矩阵进行转置, 而后向计算时则不需要转置
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    ......;
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
    }
  }  // parameter initialization
}
```
在 `void LayerSetUp(...)` 中, 我们可以得到以下几个要点: 一使用从 prototxt 中读取到的 LayerParameter 进行初始化, 此时输出矩阵 Y 的列数就是 `N_ = num_output` 的值; 二将输入的 Tensor 从指定的 axis 开始全部拍平, 这样就能得到输入矩阵 X 的行数为 `M_ = batchsize`, 输入矩阵 X 的列数为 `K_`, 与 $$Equation \ (2)$$ 一致; 三此时需要注意的是默认情况下, 参数矩阵 W 为 `N_ x K_`, 而不是 $$Equation \ (2)$$ 中的 `K x N`, 也就是说, 参数矩阵 W 默认采用的是转置形式, 所以, 在前向计算时需要将参数矩阵进行转置, 而后向计算时则不需要转置.

有了输入矩阵 X 的数据和参数矩阵 W, b 的数据, 输出矩阵 Y 所需要的内存空间等参数就可以确定了. Caffe 要求每个具体的层都需要提供 `void Reshape(...)` 函数, 用来计算本层对应的输出所需要的资源, 在 [caffe/src/caffe/layers/inner_product_layer.cpp](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/inner_product_layer.cpp#L57) 中, 类 `InnerProductLayer` 实现了虚函数 `void Reshape(...)`:
```c++
template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ......;
  // M_ = batchsize
  M_ = bottom[0]->count(0, axis);
  // 输出 reshape 为 M_ x N_
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape); // !!! 在此处申请空间 !!!
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}
```
在上面的操作中, 输出矩阵 Y 首先被确定为 `M_ x N_`, 然后调用对应 Blob `top[0]` 的 `Reshape` 函数进行内存空间的申请, 关于类 `Blob` 的 `Reshape` 操作定义在 [caffe/src/caffe/blob.cpp](https://github.com/BVLC/caffe/blob/master/src/caffe/blob.cpp#L22), 此处我们不展开讨论. 

&emsp;&emsp;到此, Caffe 已经完成进行 GEMM 操作的所有准备, 接下来可以进行 forward 和 backward 的计算.

&emsp;&emsp;对于

$$
Y = XW + B\tag{2}
$$

```c++
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
```
如上代码所示, 先分别获得 `X, W, b, Y` 所对应的数据 `bottom_data`, `weight`, `this->blobs_[1]->cpu_data()`, `top_data`, 然后执行 `caffe_cpu_gemm()`, 需要注意的是, 此时矩阵 W 会进行转置. 而向量 `b` 会通过 $$B_{M,N} = [1, ..., 1]_{1,M}^{T} \times \vec{b}$$ 计算得到矩阵 `B`, 再加到结果矩阵 `Y` 上.

&emsp;&emsp;对于

$$
\frac{\partial{Loss}}{\partial{W}} = [[\frac{\partial{Loss}}{\partial{Y}}]^{T} \cdot X]^{T}\tag{13}
$$

```c++
if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }
}
```

如上代码所示, `top_diff` 是由上层网络所计算的 $$\frac{\partial{Loss}}{\partial{Y}}$$, `bottom_data` 是本层的输入矩阵 X, 默认情况下, 程序会执行 `else` 分支, 需要注意的是, `top_diff` 进行了转置, 而 `bottom_data` 不需要转置, 这与我们的演算结果是一致的, 而二者的乘积就不需要跟演算一样再进行转置了, 因为参数矩阵 W 的梯度矩阵 `this->blobs_[0]->mutable_cpu_diff())` 和 W 一样为 $$N\_ \times K\_$$.

&emsp;&emsp;对于

$$
\frac{\partial{Loss}}{\partial{b}} = [[\frac{\partial{Loss}}{\partial{Y}}]^{T} \cdot [1, ..., 1]_{1,M}^{T}]^{T}\tag{14}
$$

```c++
if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
}
```

&emsp;&emsp;对于

$$
\frac{\partial{Loss}}{\partial{X}} = \frac{\partial{Loss}}{\partial{Y}} \cdot W^{T} \tag{19}
$$

```c++
if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
}
```
如上代码所示, `top_diff` 是由上层网络所计算的 $$\frac{\partial{Loss}}{\partial{Y}}$$, 默认情况下, 程序会执行 `else` 分支, 需要注意的是, 此时不需要像演算过程那样对参数矩阵 W 进行转置, 因为其本身已经是 $$N\_ \times K\_$$.

&emsp;&emsp;对应的使用 cuBLAS 的 GPU 版本实现, 其原理与 CPU 版本相仿, 此处不再赘述, 请参考 [caffe/src/caffe/layers/inner_product_layer.cu](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/inner_product_layer.cu).

<br>
## 总结
---
<br>
&emsp;&emsp;本文给出了 Caffe 中 innerproduct 层的运算原理解析, 并对具体的代码实现进行了对应的讲解, 希望能对使用 Caffe 的工程师有参考价值.


<br>
## 引用
---
<br>
[1] Jia, Y., Shelhamer, E., Donahue, J., Karayev, S., Long, J., & Girshick, R., et al. (2014). Caffe: Convolutional Architecture for Fast Feature Embedding. Acm International Conference on Multimedia (pp.675-678). ACM. [http://caffe.berkeleyvision.org](http://caffe.berkeleyvision.org)

[2] pytorch@github(2018), Tensors and Dynamic neural networks in Python with strong GPU acceleration. [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)

[3] tensorflow@github(2018), Computation using data flow graphs for scalable machine learning. [https://github.com/tensorflow/tensorflow](https://github.com/tensorflow/tensorflowg)

[4] wikipedia(2018), Stochastic gradient descent. [https://en.wikipedia.org/wiki/Stochastic_gradient_descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)

[5] petewarden.com(2015), Why GEMM is at the heart of deep learning. [https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/](https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/)

[6] wikipedia(2018), Basic Linear Algebra Subprograms. [https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms#Level_3)

[7] lhcheung1991@github(2017), 卷积神经网络在 ARM-CPU 上的推断计算综述. [https://lhcheung1991.github.io/blogs/2017/08/29/deeplearning-inference-benchmark-survey.html](https://lhcheung1991.github.io/blogs/2017/08/29/deeplearning-inference-benchmark-survey.html)

[8] Intel Developer Zone(2018), cblas_?gemm Computes a matrix-matrix product with general matrices. [https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm](https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm)