---
title: 深度学习实践实验-卷积神经网络
categories: 算法实践
date: 2023-01-15 18:10:11
tags: [深度学习, 人工智能, 神经网络]
cover:
top_img:
---
# 卷积神经网络

### 1、手写数字识别

  通过MNIST数据集训练得到一个手写数字分类器。要求设计一个至少包含2个卷积层和池化层的卷积神经网络。卷积核的尺寸不小于5*5，要求训后的得到的网络在测试集准确率不低于96%（要求在网络中使用dropout）

 **完成程度：**获取MNIST数据集并保存，获取图片的训练集和测试集，构建卷积神经网络模型，包含2个卷积层，2个池化层，2个全连接层，在最后一个全连接前加入一个dropout层。定义模型和损失函数，并将模型和损失函数送入到GPU当中去，使用训练集训练模型，用测试集进行测试验证，最终准确率有99.35%。

### 2、CIFAR-10分类网络

  通过CIFAR-10数据集训练得到一个手写数字分类器。要求设计一个至少包含2个卷积层和池化层的卷积神经网络。卷积核的尺寸统一采用3*3，要求训后的得到的网络在测试集上的准确率不低于70%（要求在网络中使用BatchNorm）

**完成程度：**下载CIFAR-10实验数据集，并将其划分成训练集和测试集，查看图片的尺寸，图片尺寸为32*32，一共有3个通道，定义卷积神经网络，一共包含5个卷积层，5个BN层，3个池化层，2个全连接层，最后一个全连接层前加一个dropout层。在GPU上利用训练集训练网络模型，一共进行20测迭代，最终在测试集上进行测试验证，模型训练的准确性为78.84%。



## 手写数字识别


```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
from torchvision import datasets
```


```python
# 获取训练数据
train_data = datasets.MNIST(root='./',
                      train=True,
                      transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize([0.1307, ], [0.3081, ])
                      ]),
                      download=True
                     )
```


```python
from torch.utils.data import TensorDataset, DataLoader, Dataset
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

test_data = datasets.MNIST(root='./',
                      train=True,
                      transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize([0.1307, ], [0.3081, ])
                      ])
                     )


test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)
```


```python
# 查看图片示例
print(train_data[50][0].numpy().shape)
from matplotlib import pyplot as plt
img = train_data[50][0].numpy()
label = train_data[50][1]

plt.imshow(img[0, :])
plt.show()
```

    (1, 28, 28)




![png](deep-learning-test04/output_5_1.png)
    



```python
# 网络构建
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 5, padding=2)
        self.fc1 = nn.Linear((28*28)//(4*4)*8, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # 1*28*28, 4*28*28
        x = self.conv1(x)
        x = F.relu(x)
        # 4*14*14
        x = self.pool(x)
        
        # 8*14*14
        x = self.conv2(x)
        x = F.relu(x)
        # 8*7*7
        x = self.pool(x)
        
        x = x.view(-1, (28*28)//(4*4)*8)
        x = self.fc1(x)
        x = F.relu(x)
        
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
        
    def feature_maps(self, x):
        map1 = self.conv1(x)
        map1 = F.relu(map1)
        map2 = self.pool(map1)
        map2 = self.conv2(map2)
        map2 = F.relu(map2)
        return (map1, map2)
    
```


```python
net = model()
net = net.cuda()
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```


```python
# 训练
for epoch in range(20):
    for i,data in enumerate(train_loader):
        x, y = data
        net.train()
        pred = net(x.cuda())
        loss = loss_fn(pred, y.cuda())
        
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(epoch, "损失值:",loss)
```

    0 损失值: tensor(0.1942, device='cuda:0', grad_fn=<NllLossBackward0>)
    1 损失值: tensor(0.3168, device='cuda:0', grad_fn=<NllLossBackward0>)
    2 损失值: tensor(0.0090, device='cuda:0', grad_fn=<NllLossBackward0>)
    3 损失值: tensor(0.2042, device='cuda:0', grad_fn=<NllLossBackward0>)
    4 损失值: tensor(0.0557, device='cuda:0', grad_fn=<NllLossBackward0>)
    5 损失值: tensor(0.1490, device='cuda:0', grad_fn=<NllLossBackward0>)
    6 损失值: tensor(0.0090, device='cuda:0', grad_fn=<NllLossBackward0>)
    7 损失值: tensor(0.0358, device='cuda:0', grad_fn=<NllLossBackward0>)
    8 损失值: tensor(0.1654, device='cuda:0', grad_fn=<NllLossBackward0>)
    9 损失值: tensor(0.0103, device='cuda:0', grad_fn=<NllLossBackward0>)
    10 损失值: tensor(0.0287, device='cuda:0', grad_fn=<NllLossBackward0>)
    11 损失值: tensor(0.0289, device='cuda:0', grad_fn=<NllLossBackward0>)
    12 损失值: tensor(0.1634, device='cuda:0', grad_fn=<NllLossBackward0>)
    13 损失值: tensor(0.0111, device='cuda:0', grad_fn=<NllLossBackward0>)
    14 损失值: tensor(0.0598, device='cuda:0', grad_fn=<NllLossBackward0>)
    15 损失值: tensor(0.0598, device='cuda:0', grad_fn=<NllLossBackward0>)
    16 损失值: tensor(0.0384, device='cuda:0', grad_fn=<NllLossBackward0>)
    17 损失值: tensor(0.0041, device='cuda:0', grad_fn=<NllLossBackward0>)
    18 损失值: tensor(0.0288, device='cuda:0', grad_fn=<NllLossBackward0>)
    19 损失值: tensor(0.0030, device='cuda:0', grad_fn=<NllLossBackward0>)



```python
# 测试验证
def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)
```


```python
# 验证测试
rights = 0
length = 0
for i, data in enumerate(test_loader):
    x, y = data
    net.eval()
    pred = net(x.cuda())
    rights = rights + rightness(pred, y.cuda())[0]
    length = length + rightness(pred, y.cuda())[1]

print(rights, length, rights/length)
```

    tensor(59609, device='cuda:0') 60000 tensor(0.9935, device='cuda:0')



## CIFAR-10分类网络

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, TensorDataset
```


```python
# 获取训练数据
train_data = datasets.CIFAR10(root='./',
                      train=True,
                      transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize([0.1307, ], [0.3081, ])
                      ]),
                      download=True
                     )
```

    Files already downloaded and verified



```python
# 获取训练数据
test_data = datasets.CIFAR10(root='./',
                      train=False,
                      transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize([0.1307, ], [0.3081, ])
                      ])
                     )
```


```python
# 数据加载器
train_dataset = DataLoader(train_data, batch_size=4, shuffle=True)
test_dataset = DataLoader(test_data, batch_size=4, shuffle=True)
```


```python
# 随机查看一张图片
print(np.random.randint(3000))
print(train_data[0][0].shape)
from matplotlib import pyplot as plt
img = train_data[50][0].numpy()

plt.figure(figsize=(12, 12))
# 在plt.imshow()输入彩色图像时，需要对通道进行转化
# pytorch中时(3, height, width)，imshow中是（height, width, 3）
plt.imshow(img.transpose(1, 2, 0))
```

    638
    torch.Size([3, 32, 32])


    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).





    <matplotlib.image.AxesImage at 0x27e882a9c70>




​    
![png](deep-learning-test04/output_4_3.png)
​    



```python
# 定义网络
# 要求至少包含5个卷积层和池化层
# 卷积核的尺寸统一为3*3
# 要求网络中使用BatchNorm
class model(nn.Module):
    def __init__(self):
        # 图片尺寸为[3, 32, 32]
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.5)
        self.batch_norm1 = nn.BatchNorm2d(4)
        self.batch_norm2 = nn.BatchNorm2d(8)
        self.batch_norm3 = nn.BatchNorm2d(16)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.batch_norm5 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*4*4, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        # 第一个卷积层，输入3*32*32，输出4*32*32
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        
        # 第二个卷积层，输入4*32*32，输出8*32*32
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        
        # 第三个卷积层，输入8*32*32，输出16*32*32
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batch_norm3(x)
        # 输入16*32*32，输出16*16*16
        x = self.pool1(x)
        
        # 第四个卷积层，输入16*16*16，输出32*16*16
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        # 输入32*16*16，输出32*8*8
        x = self.pool2(x)
        
        # 第五个卷积层，输入32*8*8，输出64*8*8
        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = F.relu(x)
        # 输入64*8*8，输出64*4*4
        x = self.pool1(x)
        
        # 全连接层
        x = x.view(-1, 64*4*4)
        x = self.fc1(x)
        x = F.relu(x)
        
        # 随机失活20%参数
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
net = model()
print(net)
net.to(device)
# 使用交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
# 最优化方法
opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

    model(
      (conv1): Conv2d(3, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv2): Conv2d(4, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv3): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv4): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (conv5): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (pool2): AvgPool2d(kernel_size=2, stride=2, padding=0)
      (dropout): Dropout2d(p=0.5, inplace=False)
      (batch_norm1): BatchNorm2d(4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (batch_norm2): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (batch_norm3): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (batch_norm4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (batch_norm5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc1): Linear(in_features=1024, out_features=512, bias=True)
      (fc2): Linear(in_features=512, out_features=10, bias=True)
    )



```python
from torchinfo import summary
summary(net, input_size=(64, 3, 32, 32))
```

    D:\02_soft\anaconda3\envs\pytorch\lib\site-packages\torch\nn\functional.py:1331: UserWarning: dropout2d: Received a 2-D input to dropout2d, which is deprecated and will result in an error in a future release. To retain the behavior and silence this warning, please use dropout instead. Note that dropout2d exists to provide channel-wise dropout on inputs with 2 spatial dimensions, a channel dimension, and an optional batch dimension (i.e. 3D or 4D inputs).
      warnings.warn(warn_msg)





    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    model                                    [64, 10]                  --
    ├─Conv2d: 1-1                            [64, 4, 32, 32]           112
    ├─BatchNorm2d: 1-2                       [64, 4, 32, 32]           8
    ├─Conv2d: 1-3                            [64, 8, 32, 32]           296
    ├─BatchNorm2d: 1-4                       [64, 8, 32, 32]           16
    ├─Conv2d: 1-5                            [64, 16, 32, 32]          1,168
    ├─BatchNorm2d: 1-6                       [64, 16, 32, 32]          32
    ├─MaxPool2d: 1-7                         [64, 16, 16, 16]          --
    ├─Conv2d: 1-8                            [64, 32, 16, 16]          4,640
    ├─BatchNorm2d: 1-9                       [64, 32, 16, 16]          64
    ├─AvgPool2d: 1-10                        [64, 32, 8, 8]            --
    ├─Conv2d: 1-11                           [64, 64, 8, 8]            18,496
    ├─BatchNorm2d: 1-12                      [64, 64, 8, 8]            128
    ├─MaxPool2d: 1-13                        [64, 64, 4, 4]            --
    ├─Linear: 1-14                           [64, 512]                 524,800
    ├─Dropout2d: 1-15                        [64, 512]                 --
    ├─Linear: 1-16                           [64, 10]                  5,130
    ==========================================================================================
    Total params: 554,890
    Trainable params: 554,890
    Non-trainable params: 0
    Total mult-adds (M): 289.00
    ==========================================================================================
    Input size (MB): 0.79
    Forward/backward pass size (MB): 42.21
    Params size (MB): 2.22
    Estimated Total Size (MB): 45.22
    ==========================================================================================




```python
# 开始训练
for epoch in range(20):
    for i, data in enumerate(train_dataset):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        net.train()
        pred = net(x)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()
    
    print('第', epoch, '个epoch，loss为：', loss)
```

    D:\02_soft\anaconda3\envs\pytorch\lib\site-packages\torch\nn\functional.py:1331: UserWarning: dropout2d: Received a 2-D input to dropout2d, which is deprecated and will result in an error in a future release. To retain the behavior and silence this warning, please use dropout instead. Note that dropout2d exists to provide channel-wise dropout on inputs with 2 spatial dimensions, a channel dimension, and an optional batch dimension (i.e. 3D or 4D inputs).
      warnings.warn(warn_msg)


    第 0 个epoch，loss为： tensor(1.4621, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 1 个epoch，loss为： tensor(1.1412, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 2 个epoch，loss为： tensor(0.7093, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 3 个epoch，loss为： tensor(1.1387, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 4 个epoch，loss为： tensor(1.1270, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 5 个epoch，loss为： tensor(0.1355, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 6 个epoch，loss为： tensor(2.1455, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 7 个epoch，loss为： tensor(0.2261, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 8 个epoch，loss为： tensor(0.3833, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 9 个epoch，loss为： tensor(1.3244, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 10 个epoch，loss为： tensor(0.1666, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 11 个epoch，loss为： tensor(0.8256, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 12 个epoch，loss为： tensor(0.8673, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 13 个epoch，loss为： tensor(0.0255, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 14 个epoch，loss为： tensor(0.8387, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 15 个epoch，loss为： tensor(0.1123, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 16 个epoch，loss为： tensor(0.0009, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 17 个epoch，loss为： tensor(0.0298, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 18 个epoch，loss为： tensor(0.5178, device='cuda:0', grad_fn=<NllLossBackward0>)
    第 19 个epoch，loss为： tensor(0.3033, device='cuda:0', grad_fn=<NllLossBackward0>)



```python
# 测试验证
def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)
```


```python
# 验证测试
rights = 0
length = 0
for i, data in enumerate(test_dataset):
    x, y = data
    x = x.to(device)
    y = y.to(device)
    net.eval()
    pred = net(x)
    rights = rights + rightness(pred, y)[0]
    length = length + rightness(pred, y)[1]

print(rights, length, rights/length)
```

    tensor(7884, device='cuda:0') 10000 tensor(0.7884, device='cuda:0')


