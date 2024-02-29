---
title: Redis数据库
categories: 学习笔记
date: 2024-02-26 22:08:29
tags: [数据库, Database, Redis]
cover:
top_img:
---

## Redis数据库

> Redis是在Web2.0的时代背景下产生的，在大规模数据存储和高新能数据访问方面提供了解决方案，可以用作数据库、缓存和消息代理

### NoSQL数据库简介

NoSQL（Not only SQL），泛指非关系型数据库，以简单的key-value模式存储，能够增加数据库的扩展能力

特点：不支持SQL标准、不支持ACID、远超于SQL性能

**适用场景**

* 对数据的高并发读写
* 海量数据的读写
* 对数据高可拓展性

|  数据库  |                             特点                             |
| :------: | :----------------------------------------------------------: |
| Memcache | 数据库都在内存中，一般不支持持久化<br />支持简单的键值对模式，支持类型单一<br />一般作为缓存数据库辅助持久化的数据库 |
|  Redis   | 数据库都在内存中，支持持久化，主要用作备份恢复<br />除了键值对以外，还支持多种数据结构的存储<br />一般用作缓存数据库辅助持久化的数据库 |
| MongoDB  | 高性能、开源、模式自由的文档数据库<br />数据库在内存中内存不足，会把不常用的数据存在硬盘<br />虽然是键值对模式，但是对value（尤其是json）提供了丰富的查询功能<br />支持二进制及大型对象 |

### Redis6

* Redis后台启动

```shell
# redis后台启动
# 进入redis目录，复制redis.conf文件
cp redis.conf /etc/redis.conf
# 修改daemonize yes
# 使用conf启动
redis-server /etc/redis.conf
```

* 单线程+IO多路复用（epoll）

### 常用五大数据类型

**key**操作

```shell
SET key value [EX seconds] [PX milliseconds] [NX|XX]：设置键的值。可以指定过期时间（seconds 或 milliseconds），并且可以选择仅在键不存在时才进行设置（NX 参数），或者仅在键已存在时才进行设置（XX 参数）
GET key：获取键的值
DEL key [key ...]：删除一个或多个键
EXISTS key：检查键是否存在
EXPIRE key seconds：为键设置过期时间（秒）
PEXPIRE key milliseconds：为键设置过期时间（毫秒）
TTL key：获取键的剩余生存时间（秒）
PTTL key：获取键的剩余生存时间（毫秒）
RENAME key newkey：重命名键
RENAMENX key newkey：仅当 newkey 不存在时，才重命名键
TYPE key：获取键的数据类型
KEYS pattern：查找符合给定模式的所有键
RANDOMKEY：从当前数据库中随机返回一个键
SCAN cursor [MATCH pattern] [COUNT count]：迭代当前数据库中的键
SORT key [BY pattern] [LIMIT offset count] [GET pattern [GET pattern ...]] [ASC|DESC] [ALPHA] [STORE destination]：对列表、集合或有序集合进行排序。
MGET key [key ...]：获取多个键的值
MSET key value [key value ...]：同时设置多个键的值
MSETNX key value [key value ...]：仅当所有给定键都不存在时，才设置多个键的值
SETEX key seconds value：设置键的值，并同时设置过期时间（秒）
PSETEX key milliseconds value：设置键的值，并同时设置过期时间（毫秒）
```

* String字符串

底层是一个简单的动态字符串，是可以修改的字符串，内部结构实现上，采用预分配冗余空间的方式减少内存的频繁分配。

会有一个capacity和一个len参数来控制，容量一般会高于长度，len要超过capacity时会对字符串进行扩容，小于1M时成倍扩容，大于1M时，一次扩容1M，最多为520M

* List列表

单键多值：Redis列表时简单的字符串列表，按照插入顺序排序

底层实现是一个**双向链表**，对两端的操作性能高，通过索引下标的操作中间的节点性能会较差

* Set集合
* Hash哈希
* Zset有序集合（跳跃表）

### Redis6配置文件详解

### Redis6的发布和订阅

