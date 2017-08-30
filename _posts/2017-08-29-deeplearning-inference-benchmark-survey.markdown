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
  NEON——在现代的软件系统中，当需要在32位微处理器上处理16位数据（如语音）或8位数据（如图片）时，有部分计算单位无法被用到。基于 SIMD（单指令多数据）计算模型可在这种情况下提高计算性能，如本来的一个32位数加法指令，可同时完成4个8位数的加法指令。如下图所示，为 ARMv6 `UADD8 R0, R1, R2` 指令 ，其利用32位通用处理器同时进行4个8位数的加法，这样的操作保证了4倍的执行效率而不需要增加额外的加法计算单元。从 ARMv7 架构开始，SIMD 计算模型便通过一组在特定的64位、128位向量寄存器（不同于通用寄存器）上进行操作的指令得到扩展，这组指令便成为 NEON，NEON 技术已经在 ARM Cortex-A 系列处理器上得到支持。NEON 指令由 ARM/Thumb 指令流进行执行，相比需要额外加速设备的加速方法，NEON 简化了软件的开发、调试和集成。

{:refdef: style="text-align: center;"}
![]({{site.url}}/assets/2017-08-29-deeplearning-inference-benchmark-survey/arm-neon-simd-example.png)
{:refdef}

&emsp;&emsp;如下图为 `VADD.I16 Q0, Q1, Q2` 指令对存储在向量寄存器 Q1, Q2 中的128数据以16位为单位进行并行加法。
{:refdef: style="text-align: center;"}
![]({{site.url}}/assets/2017-08-29-deeplearning-inference-benchmark-survey/arm-neon-simd-example2.png)
{:refdef}

&emsp;&emsp;NEON 指令支出8、16、32、64位有符号/无符号整型数，支持32位单精度浮点数，使用 `VCVT` 指令可进行数据类型的转换。NEON 指令的寄存器组由32位/64位寄存器组成，在 NEON 指令的眼中，寄存器组可被看成16个128位的4字寄存器，既Q0 - Q15，也可被看成32个64位的双字寄存器，既D0 - D31，如下图所示，这种不同的寄存器组视图不需要通过特定的指令进行切换，而是由 NEON 指令来决定执行时需要的寄存器组视图。
{:refdef: style="text-align: center;"}
![]({{site.url}}/assets/2017-08-29-deeplearning-inference-benchmark-survey/arm-neon-register.jpg)
{:refdef}

&emsp;&emsp;要使用最新的 NEON 指令，需要使用最新的 GNU、RealView 编译工具，这两支编译器均支持 NEON 指令集。要使用 NEON 指令，最直接的方式是使用汇编语言，如下代码分别为使用 GNU assembler(Gas) 和 RVCT（RealView Compilation Tools）汇编调用 NEON 指令的函数，其参数的传递和返回均通过 NEON 寄存器。

```c++
// using Gas, add -mfpu=neon to the assembler command line
// compilation: arm-none-linux-gnueabi-as -mfpu=neon asm.s
.text
.arm
.global double_elements
double_elements:
vadd.i32 q0,q0,q0
bx       lr
.end

// specify a target processor that supports NEON instructions
// armasm --cpu=Cortex-A8 asm.s
AREA RO, CODE, READONLY
ARM
EXPORT double_elements
double_elements
VADD.I32 Q0, Q0, Q0
BX       LR
END
```
&emsp;&emsp;在代码中使用 intrinsic 函数和数据类型是使用 NEON 指令的另一种方式，这种方式还会提供如类型检查、自动的寄存器分配等特性。intrinsic 函数的使用类似于 C/C++ 函数接口的使用，但在编译时，intrinsic 函数会被更低层次的指令序列代替，这也意味着工程师能使用高级语言描述更低层次的架构行为，编译器能在编译阶段帮助工程师进行性能上的优化，如下代码实现了上述汇编代码相同的功能。
```c++
// NEON intrinsics
// same syntax for GNU & RVCT

// NEON intrinsics with GCC
// arm-none-linux-gnueabi-gcc -mfpu=neon intrinsic.c
// Depending on your toolchain, you might also have to add 
// -mfloat-abi=softfp to indicate to the compiler that 
// NEON variables must be passed in general purpose registers.

// NEON intrinsics with RVCT
// armcc --cpu=Cortex-A9 intrinsic.c

#include <arm_neon.h>

uint32x4_t double_elements(uint32x4_t input)
{
    return(vaddq_u32(input, input));
}
```
&emsp;&emsp;还有一种使用 NEON 指令的方式，那便是由编译器进行自动向量化。由于不使用 NEON 的汇编指令和 intrinsics，所以这种方法能够保持代码的可移植性。由于 C/C++ 语言并不能显示地描述代码的并行行为，所以工程师需要给编译器一些提示，以便编译器能够使用 NEON 指令，如下所示代码。
```c++
// GCC
// arm-none-linux-gnueabi-gcc -mfpu=neon -ftree-vectorize -ftree-vectorizer-verbose=1 -c vectorized.c

// RVCT
// armcc --cpu=Cortex-A9 -O3 -Otime --vectorize --fpmode=fast --remarks -c vectorized.c

// 使用 __restrict 保证指针 pa, pb 没有在内存中没有交叠的地方
void add_ints(int * __restrict pa, int * __restrict pb, unsigned int n, int x)
{
    unsigned int i;
    // n & ~3 使 n 的低2位为0，既 n 为4的倍数。假设此处 int* 是32位整型指针，n 为4，则
    // 编译器会将下面的循环使用 VADD.I32 Q, Q, Q 进行优化，Q 为128位寄存器
    for(i = 0; i < (n & ~3); i++) pa[i] = pb[i] + x;
}
```

ACL(ARM-Compute Library)——The Computer Vision and Machine Learning library is a set of functions optimised for both ARM CPUs and GPUs using SIMD technologies. 

## 引用

---

[1] csarron@github(2017), Embedded and mobile deep learning research resources. https://github.com/csarron/emdl

[2] Wu, Jiaxiang, et al. "Quantized convolutional neural networks for mobile devices." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2016.

[3] Arm@youtube(May 24, 2017), Compute Library: optimizing computer vision and machine learning on ARM. https://www.youtube.com/watch?v=xwESWqdJt_Y

[4] Arm(2017), The ARM Computer Vision and Machine Learning library is a set of functions optimised for both ARM CPUs and GPUs using SIMD technologies. https://arm-software.github.io/ComputeLibrary/v17.03.1/index.xhtml

[5] Arm(2009), Introducing NEON Development Article. http://infocenter.arm.com/help/index.jsp?topic=/com.arm.doc.dht0002a/BABIIFHA.html
