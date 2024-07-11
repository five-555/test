---
title: 如何使用GDB进行调试
date: 2024-07-11 15:12:23
tags: [C++, GDB, Debug]
categories: 技术研究
cover:
top_img:
---

# 如何使用GDB调试

GDB (全称为GNU Debuger) 是一款由 GNU 开源组织发布、基于 UNIX/LINUX 操作系统的命令行程序调试工具。对于一名 Linux 下工作的 C++ 程序员，GDB 是必不可少的工具。
Linux下工作的程序员都习惯用命令行开发，所以这种命令行调试工具并不会觉得难用；其次，基于 Linux 服务器等的无图形界面开发，使用 Vim+GDB 可以在任意一台电脑上直接调试，不用花时间安装复杂的 IDE 环境。

## GDB常见调试命令

1、启动GDB

> 对于 C/C++ 程序调试，需要在编译前加入 -g 参数选项，表示加入一些调试信息。这样在编译后生成的可执行文件才能使用 GDB 进行调试。

```bash
g++ -g main -o demo // 生成可执行文件

gdb demo // 启动GDB调试
```

2、相关命令

```bash
// 查看接下来的10行
list

// 打断点
break n
break file:n
break func // 在函数入口

// 查看断点
info break

// 删除断点
delete    // 所有
delete n    // 删除第n个断点
disable n    // 关闭
enable    // 开启


// 运行控制指令
run  // 全速运行，遇到断点会停下，run 可以用 r 缩写替代
next // 单步调试，next 可以用 n 缩写替代
step // 单步调试，step 可以用 s 缩写替代
continue // 继续执行直到遇到下一个断点停下，没有断点直接结束，可以用 c 缩写替代
until n  // 直接跳转到指定行号程序，n 表示行号
finish   // 当程序运行在函数中时，直接执行完当前函数并打印函数返回信息
kill  // 直接结束当前进程，程序可以从头运行
quit  // 退出 GDB，quit 可以用 q 缩写替代

// 打印变量x信息
print x

// 显示当前堆栈帧的局部变量的名称和值
info locals
// 显示当前函数的参数
info args
// 显示所有全局和静态变量的名称和值，可能会比较多
info variables
// 查看某个特定类型的所有实例
info variables int/string
// 查看复杂表达式，数组结构体等
print myStruct.fileName
print myArray[3]
```

## 打印vector

打印整个 vector：

```bash
print myVector
```

打印特定的元素：

```
print myVector[0]
```

手动打印`std::vector`的元素`std::vector`的数据通常是连续存储的，可以根据其内部表示来查看元素。以下是一个例子，使用指向开始和结束的指针（取决于`std::vector`在你的`STL`实现中的具体布局）：

```bash
// n小于size
print *myVector._M_impl._M_start@n
```

这里，`_M_impl._M_start`是 GCC 的 `libstdc++` 实现细节，它可能因不同版本的 `STL` 或不同编译器而异。也就是说，如果使用的是别的库比如 LLVM's libc++，那么内部成员名可能会有不同。

在上面的命令中：
- myVector 是 vector 变量的名称。
- _M_impl._M_start 是 vector 内部的起始元素指针（对于 libstdc++）。
- @ 操作符是 GDB 的一个功能，它允许按照数组的语法打印从某个指针开始的一系列对象。
- myVector.size() 是 vector 的大小，告诉 GDB 应该打印多少个元素。

如果不确定 vector 内部结构的话，你可以先打印出 vector 对象自身来探索它的内部结构：

```bash
 print myVector
```

然后根据显示的结构信息来适配上面的命令。

**注**： 如果程序使用了优化编译选项（比如 `-O2`），编译器可能会省略掉一些调试信息或改变代码的布局。如果发现 `GDB` 无法获取 `vector` 的正确信息，需要在没有优化的情况下重新编译程序。
