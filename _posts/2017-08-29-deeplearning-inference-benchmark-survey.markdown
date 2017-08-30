---
layout: post
title: "深度学习的推断计算框架综述"
categories: blogs
---

Linghan Cheung 

lhcheung1991@gmail.com

<br>
## 摘要
---
<br>
  深度学习在计算机视觉、语音识别等领域大放异彩，许多在传统方法下无法解决的问题正在被一一攻克。然而，高昂的计算成本也极大地限制了深度学习的使用，在移动端设备、嵌入式设备等计算资源比较拮据的平台上其计算密集的特性尤为突出。本文对现阶段，工业界所使用的深度学习推断阶段的软件操作库、计算框架等，做了充分调研，由底向上勾勒出深度学习推断阶段的技术轮廓。本文主要工作如下：

1. 总结深度学习推断阶段的主要操作，指出其性能瓶颈所在；
2. 从软件库层面总结现阶段可用的开源库；
3. 从深度学习框架层面总结现阶段可用的开源框架；
4. 总结。

<br>

## 深度学习推断的主要操作

---

<br>
  对于大部分的卷积神经网络而言，卷积层是最消耗时间的部分，而全连接层则是参数量最多的部分[2]。



## 可用的开源库

---

<br>
  ACL(ARM-Compute Library)——The Computer Vision and Machine Learning library is a set of functions optimised for both ARM CPUs and GPUs using SIMD technologies. 

  NEON——在现代的软件系统中，当需要在32位微处理器上处理16位数据（如语音）或8位数据（如图片）时，有部分计算单位无法被用到。基于 SIMD（单指令多数据）计算模型可在这种情况下提高计算性能，如本来的一个32位数加法指令，可同时完成4个8位数的加法指令。如下图所示，为 ARMv6 `UADD8 R0, R1, R2` 指令 ，其利用32位通用处理器同时进行4个8位数的加法，这样的操作保证了4倍的执行效率而不需要增加额外的加法计算单元。从 ARMv7 架构开始，SIMD 计算模型便通过一组在特定的64位、128位向量寄存器（不同于通用寄存器）上进行操作的指令得到扩展，这组指令便成为 NEON，NEON 技术已经在 ARM Cortex-A 系列处理器上得到支持。NEON 指令由 ARM/Thumb 指令流进行执行，相比需要额外加速设备的加速方法，NEON 简化了软件的开发、调试和集成。如下图为 `VADD.I16 Q0, Q1, Q2` 指令对存储在 Q1, Q2 中的128数据以16位为单位进行并行加法。

{:refdef: style="text-align: center;"}
![]({{site.url}}/assets/2017-08-29-deeplearning-inference-benchmark-survey/arm-neon-simd-example.png)
{:refdef}

{:refdef: style="text-align: center;"}
![]({{site.url}}/assets/2017-08-29-deeplearning-inference-benchmark-survey/arm-neon-simd-example2.png)
{:refdef}

## 引用

---

[1] csarron@github(2017), Embedded and mobile deep learning research resources. https://github.com/csarron/emdl

[2] Wu, Jiaxiang, et al. "Quantized convolutional neural networks for mobile devices." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2016.

[3] Arm@youtube(May 24, 2017), Compute Library: optimizing computer vision and machine learning on ARM. https://www.youtube.com/watch?v=xwESWqdJt_Y

[4] Arm(2017), The ARM Computer Vision and Machine Learning library is a set of functions optimised for both ARM CPUs and GPUs using SIMD technologies. https://arm-software.github.io/ComputeLibrary/v17.03.1/index.xhtml

[5] Arm(2009), Introducing NEON Development Article. http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dht0002a/BABIIFHA.html
