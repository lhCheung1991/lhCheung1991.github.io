---
layout: post
title: "ncnn源码解析"
categories: blogs
tags: deep-learning, hpc, arm
---

&nbsp;&nbsp;&nbsp;&nbsp;在 ncnn 给出的使用 demo 中我们可以看出, 类 `ncnn::Net` 是网络进行推断的主体类, 用户新建一个 `ncnn:Net` 的对象, 并并通过其 `ncnn:Net:load_param()` 和 `ncnn:Net:load_model()` 函数分别对网络的结构和网络的训练参数进行加载以便在内存中构建推断所需的网络信息:

```c++
ncnn::Net squeezenet;
squeezenet.load_param("squeezenet_v1.1.param");
squeezenet.load_model("squeezenet_v1.1.bin");
```

&nbsp;&nbsp;&nbsp;&nbsp;ncnn 在类 `ncnn:Net` 的 `ncnn:Net::load_param()` 函数中通过一个 `while(!feof(fp)){...}` 语句循环处理 ncnn 网络结构文件中的每一层, 其网络结构文件可以通过 `tools/caffe2ncnn` 对 `Caffe` 的导出模型进行转换得到, 使用此种非加密的转换方式转换后的网络结构文件为可读的, 其格式如下所示(上述代码中的squeezenet_v1.1.param):

```c++
7767517
75 83
Input            data             0 1 data 0=3 1=227 2=227
Convolution      conv1            1 1 data conv1 0=64 1=3 2=1 3=2 4=0 5=1 6=1728
ReLU             relu_conv1       1 1 conv1 conv1_relu_conv1 0=0.000000
Pooling          pool1            1 1 conv1_relu_conv1 pool1 0=0 1=3 2=2 3=0 4=0
Convolution      fire2/squeeze1x1 1 1 pool1 fire2/squeeze1x1 0=16 1=1 2=1 3=1 4=0 5=1 6=1024
......
```

```c++
/**
int layer_to_index(const char* type); 是定义在 layer.cpp 中的全局函数, 其
定义如下:

int layer_to_index(const char* type)
{
    for (int i=0; i<layer_registry_entry_count; i++)
    {
        if (strcmp(type, layer_registry[i].name) == 0)
            return i;
    }
    return -1;
}

其功能是根据 layer 的 type 与 layer_registry_entry 数组 layer_registry 中的
元素进行 name 的对比, 从而确定该层类型在数组 layer_registry 中的索引, 而 
layer_registry_entry 是定义在 layer.h 中的结构体, 其定义如下:

struct layer_registry_entry
{
#if NCNN_STRING
    // layer type name
    const char* name;
#endif // NCNN_STRING
    // layer factory entry
    layer_creator_func creator;
    // typedef Layer* (*layer_creator_func)();
};

其中的 layer_creator_func 是返回 Layer* 指针的函数指针类型, creator 将存放具体类型
layer 的构建函数指针, 这些函数指针在 layer.cpp 编译链接时会得到初始化, 具体地, 
在 layer.cpp 中, ncnn 声明了如下的数组:

#include "layer_declaration.h"
static const layer_registry_entry layer_registry[] =
{
#include "layer_registry.h"
// layer_registry.h 内容示例
// #if NCNN_STRING
// {"AbsVal",AbsVal_arm_layer_creator},
// #else
// {AbsVal_arm_layer_creator},
// ......
};

layer.cpp 在预处理阶段就会完成数组 layer_registry 内容的填充, 在链接阶段会将 layer 
构建函数指针进行填充, 链接到具体层的创建方法. 每个层类在定义时都会包含 layer.h 中的宏定义:

#define DEFINE_LAYER_CREATOR(name) \
    ::ncnn::Layer* name##_layer_creator() { return new name; }

如类 AbsVal_arm:
namespace ncnn {
DEFINE_LAYER_CREATOR(AbsVal_arm)
...
}

通过这样的定义, DEFINE_LAYER_CREATOR(AbsVal_arm) 会产生函数 
ncnn::Layer* AbsVal_arm_layer_creator() { return new AbsVal_arm; }, 从而实现
数组 layer_registry 中 layer 构建函数指针注册. 最后, 通过 layer.h 中的 create_layer 函数

Layer* create_layer(int index)
{
    if (index < 0 || index >= layer_registry_entry_count)
        return 0;

    layer_creator_func layer_creator = layer_registry[index].creator;
    if (!layer_creator)
        return 0;

    return layer_creator();
}

通过 layer_to_index() 函数返回的 index 找到相应的层的创建函数指针并调用 layer_creator() 即可
初始化对应的层对象
**/

int typeindex = layer_to_index(layer_type);
Layer* layer = create_layer(typeindex);
```


