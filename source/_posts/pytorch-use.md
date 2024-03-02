---
title: PyTorch使用方法
categories: 技术研究
date: 2024-03-02 17:11:36
tags: [PyTorch, DeepLearning, 神经网络, AI]
cover:
top_img:
---
## PyTorch基础语法

Pytorch是Facebook主导开发的，基于Python的科学计算包，主要有一下两个特点：

比NumPy更灵活，可以使用GPU的强大计算能力

开源高效的深度学习研究平台

* ### 张量

  > PyTorch中的Tensors可以使用GPU计算

```python
import torch

# 可以返回填充了未初始化数据的张量
torch.empty(5, 3)

# 创建一个随机初始化矩阵
torch.rand(5, 3)

# 创建一个0填充的矩阵，指定数据类型为long：
torch.zeros(5, 3, dtype=torch.long)

# 创建 Tensor 并使用现有数据初始化：
x = torch.tensor([5.5, 3])
x

#根据现有张量创建新张量。这些方法将重用输入张量的属性，除非设置新的值进行覆盖
x = x.new_ones(5, 3, dtype=torch.double)  # new_* 方法来创建对象
x

# 覆盖 dtype，对象的 size 是相同的，只是值和类型发生了变化
x = torch.randn_like(x, dtype=torch.float)
x

# 获得张量的size，返回值为tuple
x.size()
```

* ### 操作

```python
#加法操作
y = torch.rand(5, 3)
x + y

torch.add(x, y)

# 提供输出Tensor作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
result

# 替换，将x加到y
y.add_(x)
y
```

任何以下划线结尾的操作都会用结果替换原变量

可以使用与Numpy索引方式相同的操作来对张量进行操作

```python
# torch.view可以改变张量的维度和大小
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # size -1 从其他维度推断

x.size(), y.size(), z.size()

# 如果张量只有一个元素，使用 .item() 来得到 Python 数据类型的数值
x = torch.randn(1)

x, x.item()
```

> 官方文档：[torch — PyTorch 1.12 documentation](https://pytorch.org/docs/stable/torch.html)

* ### NumPy转换

```python
# PyTorch张量转换为NumPy数组
a = torch.ones(5)
a
b = a.numpy()
b

# 了解 NumPy 数组的值如何变化，a和b的数值都会加1
a.add_(1)
a, b

# NumPy 数组转换成 PyTorch 张量时，可以使用 from_numpy 完成
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
a, b
```

所有的 Tensor 类型默认都是基于 CPU， CharTensor 类型不支持到 NumPy 的转换

* ### CUDA张量

CUDA张量是能够再GPU设备在运算的张量。使用.to方法可以将Tensor移动到GPU设备中去

```python
# is_available 函数判断是否有 GPU 可以使用
if torch.cuda.is_available():
    device = torch.device("cuda")          # torch.device 将张量移动到指定的设备中
    y = torch.ones_like(x, device=device)  # 直接从 GPU 创建张量
    x = x.to(device)                       # 或者直接使用 .to("cuda") 将张量移动到 cuda 中
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # .to 也会对变量的类型做更改
```

## Autograd自动求导

> PyTorch中所有神经网络的核心是autograd

`autograd`为张量上的所有操作提供了自动求导。它是一个在运行时定义的框架，这意味着反向传播是根据你的代码来确定如何运行。

`torch.Tensor` 是这个包的核心类。如果设置`.requires_grad`为`True`，那么将会追踪所有对于该张量的操作。当完成计算后通过调用`.backward()`会自动计算所有的梯度，这个张量的所有梯度将自动积累到`.grad`属性。这也就完成了自动求导的过程。

要组织张量追踪历史记录，可以调用`.detach()`方法将其与计算历史记录分离。为了防止跟踪历史记录（和使用内存），可以将代码块包装在`with tirch.no_grad():`语句中。这一点在评估模型时特别有用，因为模型可能具有`requires_grad=True`的可训练参数，但是并不需要计算梯度。

自动求导还有另一个重要的类`Function`。`Tensor`和`Function`互相连接并生成一个非循环图，存储了完整的计算历史。

如果需要计算导数，可以在`Tensor`上调用`.backward()`。如果`Tensor`是一个标量（即它只包含一个元素数据）则不需要为`backward()`指定任何参数。但是如果他又多个元素，则需要指定一个`gradient`参数来匹配张量的形状。

```python
# 创建一个张量并设置 requires_grad=True 用来追踪他的计算历史
x = torch.ones(2, 2, requires_grad=True)
x

# 对张量进行操作，也就是计算过程
y = x + 2
y

# 结果 y 已经被计算出来了，所以，grad_fn 已经被自动生成了
y.grad_fn

# 然后，再对 y 进行操作
z = y * y * 3
out = z.mean()

z, out

# .requires_grad_( ... ) 可以改变现有张量的 requires_grad 属性。 如果没有指定的话，默认输入的 flag 是 False
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
```

* ### 梯度

通过反向传播打印对应结点的梯度，因为`out`是一个纯量Scalar，`out.backward()`等于`out.backward(torch.tensor(1))`

```python
out.backward()

# 打印其梯度
x.grad
```

关于Autograd的更多操作

```python
x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

y

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)

x.grad

# 如果 .requires_grad=True 但是你又不希望进行 Autograd 的计算，那么可以将变量包裹在 with torch.no_grad() 中
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)
```

> `autograd`和`Function`的官方文档：[Automatic differentiation package - torch.autograd — PyTorch 1.12 documentation](https://pytorch.org/docs/stable/autograd.html)

## 神经网络

PyTorch中，我们可以使用`torch.nn`来构建神经网络

`torch.nn`依赖`autograd`来定义模型并求导。`nn.Module`中包含了构建神经网络所需的各个层和`forward(input)`方法，该方法返回神经网络的输出。

​	神经网络的典型训练过程如下

1、定义包含可学习参数（权重）的神经网络模型

2、在数据集上迭代

3、通过神经网络处理输入

4、计算损失

5、将梯度反向传播回网络结点

6、更新网络的参数，一般可以使用梯度下降等最优化方法

​	PyTorch的`nn.Conv2d()`详解，该类作为二维卷积的实现

```python
in_channels		# 输入张量的channels数
out_channels	# 期望的四维输出张量的channels数
kernel_size		# 卷积核的大小，一般用5×5、3×3
stride=1		# 卷积核在图像窗口上每次平移的间隔，步长
padding=0		
```

