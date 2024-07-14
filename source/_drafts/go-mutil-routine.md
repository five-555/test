---
title: Go语言学习-协程、管道、并发
categories: 技术研究
date: 2024-07-10 22:10:44
tags:
cover:
top_img:
---

# Goroutines和Channels32kol

`Go`语言中的并发程序一般会使用两种手段来实现。第一种是`goroutine`，第二种是`channel`，其支持`CSP`：communicating sequential processes，叫做顺序通信进程。

`CSP`是一种现代的并发编程模型，在这种编程模型中值会在不同的运行实例（groutine）中传递，尽管大多数情况下仍然是被限制在单一实例中。

## Goroutines-协程

> 每一个并发的执行单元叫做一个`goroutine`（协程），如果有多个函数，函数之间不存在调用关系，那么这两个函数就有并发执行的可能。我们可以把协程理解为一个轻量级线程。它们允许我们使用的时候并发的执行函数，和传统的操作系统线程相比，协程更加高效，消耗的资源更少。

当一个程序启动时，其主函数会在一个单独的`goroutine`中运行，称为主协程，新的协程会使用`go`语句来创建。在语法上，我们只要在一个方法调用前加上关键字`go`，就会使得这个函数在一个新创建的`goroutine`中运行。

当我们的协程已经启动以后，除了从主函数退出或者直接终止程序之外，没有其它的编程方法能够让一个`goroutine`来打断另一个的执行，当然我们也有协程间的通信来让一个协程请求其它的协程，但是这也是当前协程主动让出的结果。


```go
f()    // call f(); wait for it to return
go f() // create a new goroutine that calls f(); don't wait
```

## example-并发的Clock服务



