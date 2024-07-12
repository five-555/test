---
title: C++中常见容器的使用方法
categories: 技术研究
date: 2024-07-12 09:52:08
tags:
cover:
top_img:
---

# 序列容器

> 不知道如何解释，凭感觉吧，就是我们想象中的那样。

## vector

`vector`是我们常用的动态数组，它通过模板泛型编程的方式，让我们能够使用`vector`来存储任意对象。

`vector`的底层是内存当中一段连续的地址空间，地址空间连续的最大好处就是能够支持用户的随机访问，我们可以和操作数组一样，使用下标的方式来获取到容器中的对象。在`vector`中有两个关键的成员变量`size`和`capacity`，其中`size`表示的是当前容器中存储的对象实际个数，而`capacity`则是容器的最大容量。


* 常用的接口

## list

## deque

## array

# 容器适配器

## queue

## stack

## priority_queue

# 关联容器

## set

## map

## mutilset

## mutilmap

# 无序关联容器

## unordered_set

## unordered_map

## unordered_mutilset

## unordered_mutilmap
