# Opencv homework README

> 自己动手使用C++和opencv实现一些简单的计算机视觉算法

<!-- TOC -->

- [Opencv homework README](#opencv-homework-readme)
    - [Codes](#codes)
    - [Tools](#tools)
    - [test_picture](#test_picture)

<!-- /TOC -->
## Codes

- gaus_filter.cpp
手写了个生成 **N*N** 大小的高斯矩阵，同时还能指定高斯函数的sigma大小。最终对图片进行高斯高通滤波和高斯低通滤波。

- hybrid_image.cpp
采用了比较简单的实现方法，直接应用gaus_filter.cpp中的方法得到的高低通滤波后的图像，直接进行叠加得到hybrid_image.cpp

>需要注意的是高斯矩阵大小和sigma大小接近**3:1**的时候能得到比较好的效果

- edge_detection.cpp
实现了canny算法，达到边缘检测的效果。调用了createTrackbar函数，可以在界面中调整双阈值检测中的低阈值大小。

- feature_matching.cpp
采用了orb算法，但是因为太复杂没做方向的判断和质心的计算，其实就是很简单的用FAST特征点检测 + brief特征描述子。

>**匹配结果比较失败**。没有调用和算法层面有关的库函数，全是自己根据论文和网上的博客来实现的，因此效果不是很好，目前尚不知道为什么。

- SAD.cpp
采用了SAD(Sum of absolute differences,绝对差值求和)来实现双目深度估计

>发现网上的博客都说是参考opencv库函数SAD.h来写的= =于是选择了加入

## Tools

- opencv 4.1.0
- VS2015

## test_picture

- hybrid_image使用cat.bmp dog.bmp能达到比较好的效果，其他的满足hybrid_image某些条件的图片也可以
- SAD测试过网上和论文中的图片，只有left.png right.png能达到好一点的效果(毕竟SAD是个简陋的算法)