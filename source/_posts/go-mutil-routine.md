---
title: Go语言学习-协程、管道、并发
categories: 技术研究
date: 2024-07-10 22:10:44
tags:
cover:
top_img:
---


# Goroutines和Channels

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

> 一个连接中开启一个协程

在我们的例子中，实现了使用协程的方式去服务多个客户端的代码，它会在客户端与服务端建立连接以后，调用`go handleConn`，随后服务端会开启一个协程，让这个服务函数异步的在客户端中执行。随后主协程就会继续执行for循环语句，如果没有随后的连接，就会阻塞在`accept`，如果有连接，就会再次开启一个协程。

**服务端**

```go
// Clock1 is a TCP server that periodically writes the time.
package main

import (
    "io"
    "log"
    "net"
    "time"
)

func main() {
    listener, err := net.Listen("tcp", "localhost:8000")
    if err != nil {
        log.Fatal(err)
    }

    for {
        conn, err := listener.Accept()
        if err != nil {
            log.Print(err) // e.g., connection aborted
            continue
        }
        
        // 关键并发语句
        go handleConn(conn)
    }
}

func handleConn(c net.Conn) {
    defer c.Close()
    for {
        _, err := io.WriteString(c, time.Now().Format("15:04:05\n"))
        if err != nil {
            return // e.g., client disconnected
        }
        time.Sleep(1 * time.Second)
    }
}
```

**客户端**

```go
// Netcat1 is a read-only TCP client.
package main

import (
    "io"
    "log"
    "net"
    "os"
)

func main() {
    conn, err := net.Dial("tcp", "localhost:8000")
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()
    mustCopy(os.Stdout, conn)
}

// 连接中的数据在终端显示出来
func mustCopy(dst io.Writer, src io.Reader) {
    if _, err := io.Copy(dst, src); err != nil {
        log.Fatal(err)
    }
}
```

# Channels-通道

我们说`goroutine`是`Go`语言程序的并发体，而这些比线程还要更轻量级的并发体之间的通信方式是通过`channel`来完成的，一个`channel`是一个通信机制，它可以让一个`goroutine`通过它给另一个`goroutine`发送值信息。每个`channel`都有一个特殊的类型，这个类型是`channel`可以发送的数据类型。

例如：我们可以创建一个可发送`int`数据类型的`channel`，对应着一个`make`创建的底层数据结构的引用，我们可以指定`make`中的第`2`个参数，对应的是`channel`的容量。

```go
ch := make(chan int)

ch = make(chan int)    // unbuffered channel
ch = make(chan int, 0) // unbuffered channel
ch = make(chan int, 3) // buffered channel with capacity 3
```

当我们复制一个`channel`或用于函数参数传递时，我们只是拷贝了一个`channel`引用，因此调用者和被调用者将引用同一个`channel`对象。

一个`channel`有发送和接受两个主要操作，都是通信行为，一个发送语句将一个值从一个`gorountine`通过`channel`发送到另一个执行接收操作的`gorountine`。

```go
ch <- x  // a send statement
x = <-ch // a receive expression in an assignment statement
<-ch     // a receive statement; result is discarded
```

`channel`还支持`close`操作，用于关闭通道

```go
close(ch)
```

**channel**操作的工作原理

在`channel`中除了有一个由循环队列维护的`buffer`缓冲以外，还有两个队列，分别是`recvq`和`sendq`，维护的是当前等待接收和等待发送的`gorountine`

* 发送操作

    1、获取`channel`互斥锁

    2、检查：检查是否有等待的接收者

        如果有等待的接收者，直接将数据复制到接收者，然后将接收者从等待队列中唤醒

        如果没有接收者并且缓冲区未满，将数据放入缓冲区

        如果没有空闲缓存且没有等待接收者，将发送者放入等待队列，并挂起`gorountine`

    3、解锁：释放互斥锁

    4、如果发送者被挂起，运行时会调度其他`gorountine`继续执行

* 接收操作

> 通过发送操作很容易进行推导

## example-不带缓存的channel

`main`和匿名函数使用`channel`完成同步，在匿名函数中，会把输入放入到连接的`socket`缓冲区中，并且在完成以后往管道里发送一个`struct`，因为`channel`的`buf`为0，如果没有接收者的话，就会将`rountine`加入到`sendq`中，并将协程挂起，直到在`main`中将通道里的数据取出。或者`main`想从`channel`中取数据，发现`buf`没有数据，也会挂起，等待另一个协程。

```go
func main() {
    conn, err := net.Dial("tcp", "localhost:8000")
    if err != nil {
        log.Fatal(err)
    }
    done := make(chan struct{})
    go func() {
        io.Copy(os.Stdout, conn) // NOTE: ignoring errors
        log.Println("done")
        done <- struct{}{} // signal the main goroutine
    }()
    mustCopy(conn, os.Stdin)
    conn.Close()
    <-done // wait for background goroutine to finish
}
```

我们的`channel`还可以用于将多个`grountine`连在一起，如果我们的连接图是一个有向无环图的话并且我们的`buf`为0的话，，我们天然的就相当于控制了不同协程的执行顺序，因为下一个协程的调度依赖于通道是否有上一个协程传入的数据，否则就会一直处于挂起状态。**这不就是我们操作系统中的同步么**，我们发现使用`channel`能够达到类似于信号量来控制线程同步的效果。

同时，`channel`中的缓存数量就相当于我们信号量的个数，在操作系统中的`PV`操作也可以类比为`channel`中存取相应的元素。
