---
title: the_go_programming_language
date: 2024-02-12 13:29:11
tags: [golang Go语言圣经]
categories: 学习笔记
cover: top_img.png
top_img: top_img.png
---

### GO语言基础要点

四种变量声明方式

* 第一种是短变量声明，只能在函数内部使用，不能用于包变量
* 第二种依赖于字符串的默认初始化零值机制
* 第三种用得较少，除非用于声明多个变量
* 第四种当变量类型与初值类型相同时，类型冗余，但是类型不同时，变量类型就必须了

```go
s := ""
var s string
var s = ""
var s string = ""
```

