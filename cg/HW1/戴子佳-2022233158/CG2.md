# CG2

1、快速凸包算法

[凸包问题——快速凸包算法 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/166105080)

快速凸包算法也可以看成是增量算法的一种变种，与随机增量算法相比，它们的不同就在于每次迭代从面的外部点集中选择的点不同。随机增量算法从外部点集中随机的选择一个点，但是快速凸包算法是选择距离最远的一个点，著名的开源代码Qhull[1]、 CGAL[2]都有快速凸包算法的C++实现。本篇文章介绍三维的快速凸包算法的原理和实现。

2、gift wrapping

[3D凸包算法 gift wrapping - 哔哩哔哩 (bilibili.com)](https://www.bilibili.com/read/cv13096782/)

3、增量法

[增量法（三维凸包） - 塔塔开 - 博客园 (cnblogs.com)](https://www.cnblogs.com/tatakai/p/15039928.html)

![image-20220924163825495](C:\Users\80624\AppData\Roaming\Typora\typora-user-images\image-20220924163825495.png)