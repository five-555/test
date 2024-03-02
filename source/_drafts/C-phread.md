---
title: C++并行编程
categories: 技术研究 算法实践 学习笔记
date: 2024-03-02 15:05:28
tags: [C++, Thread, 线程]
cover:
top_img:
---
### 

### 

> 导入头文件thread

* 创建线程	`thread th(function, arg)`

  创建线程需要绑定一个函数，function表示已经定义的函数名，通过arg可以传入函数的参数，线程在std标准库当中

* join和detach

  join：调用此接口，当前线程会一直阻塞，直到目标线程完成，如果目标线程十分耗时，主线程会一直阻塞

  detach：让目标线程称为守护线程（daemon threads）。一旦detach之后，目标线程将独立执行，即便其对应的thread对象销毁也不影响线程的执行，并且，无法再与其通信

  可以通过`joinable()`接口查询是否可以对接口进行join和detach

```C++
// 创建线程

```



参考连接:https://zhuanlan.zhihu.com/p/340278634