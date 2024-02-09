---
title: C++使用zlib库来压缩文件
date: 2024-01-26 10:34:44
tags: [C++, zlib]
categories: 技术研究
cover:
top_img:
---

## C++使用zlib库来压缩文件

zlib压缩库提供内存压缩和解压缩功能，包括对未压缩的完整性检查数据，提供支持的压缩方法为：deflation，默认使用压缩数据格式为zlib格式。

zlib库支持读取和写入gzip(.gz)格式的文件，zlib格式旨在紧凑且快速，可用于内存和通信渠道。gzip格式设计用于文件系统上的单文件压缩，比zlib具有更大的头部以维护目录信息，并且使用与zlib不同且更慢的检查方法。

该库不安装任何信号处理程序。解码器检查压缩数据的一致性，因此即使在输入损坏的情况下，库也不应崩溃。

### 数据流结构

```
typedef voidpf (*alloc_func)(voidpf opaque, uInt items, uInt size);
typedef void   (*free_func)(voidpf opaque, voidpf address);
```

* `typedef voidpf (*alloc_func)(voidpf opaque, uInt items, uInt size);`这个函数指针通常用于内存分配，允许用户自定义的内存分配函数
* `typedef void   (*free_func)(voidpf opaque, voidpf address);`这个函数指针通常用于内存释放，允许用户自定义的内存释放函数

### 基本功能

```
ZEXTERN int ZEXPORT deflateInit(z_streamp strm, int level);
```

* `level`表示压缩级别，要么为`Z_DEFAULT_COMPRESSION`，要么介于0-9之间，1表示最佳速度，9表示最佳压缩，0表示没有压缩，`Z_DEFAULT_COMPRESSION`默认在6级别。
* `deflateInit` 返回 `Z_OK` 如果成功，则返回 `Z_MEM_ERROR` 如果没有 足够的内存，`Z_STREAM_ERROR` `level` 不是有效的压缩级别，`Z_VERSION_ERROR` *zlib* 库版本 （`zlib_version`） 不兼容 替换为调用方 （`ZLIB_VERSION`） 假定的版本。如果没有错误消息，`则 msg` 设置为 null。`deflateInit` 不 执行任何压缩：这将由 `deflate（）` 完成。

```
ZEXTERN int ZEXPORT deflate(z_streamp strm, int flush);
```

