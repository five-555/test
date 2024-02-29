---
title: build_sample_redis
categories: 技术研究 算法实践 学习笔记
date: 2024-02-27 11:29:57
tags:
cover:
top_img:
---

## Socket编程相关语法

### 服务端

* 创建句柄

函数原型：`int socket(int domain, int type, int protocol);`

domin：指定通信域，ipv4或ipv6，AF_INET表示ipv4地址

type：指定套接字类型

protocol：0表示为TCP协议

```c++
// IPV4地址、提供面向连接的可靠数据传输服务、TCP协议
int fd = socket(AF_INET, SOCK_STREAM, 0);
if (fd < 0) {
    die("socket()");
}
```



* 设置socket可选项
* Bind，绑定ip和端口号
* Listen

```c++
// server
fd = 
```

### 客户端

* 创建句柄
* 设置IP地址和端口号
* Connect

**使用的协议**

len+msg的结合

len表示后面的字长，msg表示要获取的数据



## 事件循环和非阻塞型IO

> 在服务器端网络编程中，处理并发连接有三种方法：forking、多线程（multi-threading）和事件循环（event loops）。forking会为每个客户端连接创建新的进程以实现并发。多线程使用线程而不是进程。事件循环使用轮询和非阻塞 I/O，通常在单个线程上运行。由于进程和线程的开销，大多数现代生产级软件使用事件循环来进行网络编程。



## 哈希表数据结构

* 数据结构

```c++
// 哈希节点
// hcode表示哈希码
// next表示用于处理哈希冲突的链表
struct HNode {
    HNode *next = NULL;
    uint64_t hcode = 0;
};

// 哈希表
// tab包含一个指针数组，存储哈希表
// mask掩码，用于从哈希码计算索引
// size表示哈希表中存储的节点数
struct HTab {
    HNode **tab = NULL;
    size_t mask = 0;
    size_t size = 0;
};

// 实际使用的哈希表接口
// 存储两个Htab，便于渐进式的调整大小
// resizing_pos用于跟踪调整大小过程中的当前位置
// redis快的原因，渐进式哈希
struct HMap {
    HTab ht1;   // newer
    HTab ht2;   // older
    size_t resizing_pos = 0;
};

// 接口
// 在哈希表中查找一个给定键值的点
HNode *hm_lookup(HMap *hmap, HNode *key, bool (*eq)(HNode *, HNode *));
// 向哈希表中插入一个新节点
void hm_insert(HMap *hmap, HNode *node);
// 从哈希表中移除并返回给定键的节点
HNode *hm_pop(HMap *hmap, HNode *key, bool (*eq)(HNode *, HNode *));
// 返回哈希表中节点的总数
size_t hm_size(HMap *hmap);
// 销毁哈希表，释放所有相关资源
void hm_destroy(HMap *hmap);
```





