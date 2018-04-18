---
layout: post
title: "使用 YOLOv2 在移动设备上进行人脸检测"
categories: blogs
tags: deep-learning, hpc, arm, object-detection
---
{:refdef: style="text-align: center;"}
Linghan Cheung
{:refdef}
## 摘要
---
<br>
&emsp;&emsp;目标检测是一个基本的计算机视觉研究领域, 而自从 RCNN 将深度学习技术引入到目标检测之后, 

<br>
## 引用
---
<br>
[1] DragonFive@Github(2017), 目标检测算法总结. [https://dragonfive.github.io/2017-07-30/object_detection_with_dl/](https://dragonfive.github.io/2017-07-30/object_detection_with_dl/)

[2] 浅小思@Zhihu(2017), 目标检测(1)-Selective Search. [https://zhuanlan.zhihu.com/p/27467369](https://zhuanlan.zhihu.com/p/27467369)

[3] 浅小思@Zhihu(2017), 目标检测(2)-RCNN. [https://zhuanlan.zhihu.com/p/27473413](https://zhuanlan.zhihu.com/p/27473413)

[4] 浅小思@Zhihu(2017), 目标检测(3)-SPPNet. [https://zhuanlan.zhihu.com/p/27485018](https://zhuanlan.zhihu.com/p/27485018)

[5] 浅小思@Zhihu(2017), 目标检测(4)-FastRCNN. [https://zhuanlan.zhihu.com/p/27582096](https://zhuanlan.zhihu.com/p/27582096)

[6] Girshick, R. (2015). Fast R-CNN. IEEE International Conference on Computer Vision (pp.1440-1448). IEEE Computer Society.

[7] Girshick, R. (2015). Fast R-CNN [PowerPoint slides]. Retrieved from [http://www.robots.ox.ac.uk/~tvg/publications/talks/fast-rcnn-slides.pdf](http://www.robots.ox.ac.uk/~tvg/publications/talks/fast-rcnn-slides.pdf).

[8] shenxiaolu1984@CSDN(2016), Fast RCNN算法详解. [http://blog.csdn.net/shenxiaolu1984/article/details/51036677](http://blog.csdn.net/shenxiaolu1984/article/details/51036677)

[9] 晓雷@Zhihu(2016), 原始图片中的ROI如何映射到到feature map. [https://zhuanlan.zhihu.com/p/24780433](https://zhuanlan.zhihu.com/p/24780433)

[10] 雪柳花明@360doc(2017), 目标检测(一): Faster R-CNN详解. [http://www.360doc.com/content/17/0809/10/10408243_677742029.shtml](http://www.360doc.com/content/17/0809/10/10408243_677742029.shtml)

[11] Dang Ha The Hien@Medium.com(2017), A guide to receptive field arithmetic for Convolutional Neural Networks. [https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)