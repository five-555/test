---
title: 深度学习实践实验-SoftMax回归
categories: 算法实践
date: 2023-01-08 18:03:21
tags: [深度学习, 人工智能]
cover:
top_img:
---
# SoftMax回归

### 1、聚类和分类

通过sklearn库提供的聚类算法生成K类数据，以这些数据做为数据集训练神经网络，利用softmax层和交叉熵损失函数对数据进行分类。聚类参数要求K>3，数据样本不少于1000，其他参数参考课件。对聚类后的数据按9：1的原则划分训练集和测试集，利用在训练集上训练得到的模型对测试集上的数据进行验证，要求模型的准确率不低于99%。

**完成程度**：使用sklearn.datasets中的make_blobs函数生成1200个样本数据，样本种类为4，样本中心分别为[-5, 5], [0, -2], [4, 8], [7, 3]，方差分别为[1.5,1.5,1.2,1]，每样样本数为300个，对样本数据进行划分，按照9训练集：1测试集的比例进行划分，构建网络，使用交叉熵损失函数，使用训练集对模型进行训练，在测试集上完成测试验证。



### 2、鸢尾花分类

  Iris数据集包含150个样本，对应数据集的每行数据。每行数据包含每个样本的四个特征和样本的类别信息，iris数据集是用来给鸢尾花做分类的数据集，每个样本包含了花萼长度、花萼宽度、花瓣长度、花瓣宽度四个特征，请用神经网络训练一个分类器，分类器可以通过样本的四个特征来判断样本属于山鸢尾花、变色鸢尾还是维吉尼亚鸢尾。数据集文件iris.csv。要求模型的准确率不低于99%。

**完成程度**：加载鸢尾花数据集iris.csv，查看样本数据，对数据进行标准化处理，将使用到的数据划分为训练集和测试集，搭建网络模型，使用交叉熵损失函数，使用训练集对模型进行训练，在测试集上完成测试验证。



## 聚类和分类


```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch.nn.functional as F
from sklearn.datasets import make_blobs

data, target = make_blobs(n_samples=1200, n_features=2, centers=[[-5, 5], [0, -2], [4, 8], [7, 3]], cluster_std=[1.5,1.5,1.2,1])
plt.scatter(data[:, 0], data[:, 1], c=target, marker='o')
plt.show()
```


​    
![png](deep-learning-test03/output_1_0.png)
​    



```python
# 数据准备
# 将训练集和测试集按照9：1的比例进行划分
# 一共1200个数据，1080个训练集，120个测试集
data = torch.from_numpy(data)
data = data.type(torch.FloatTensor)
target = torch.from_numpy(target)
target = target.type(torch.LongTensor)

train_x = data[:1080]
train_y = target[:1080]

test_x = data[1080:]
test_y = target[1080:]

# 训练数据集
train_dataset = TensorDataset(train_x, train_y)
# 测试数据集
test_dataset = TensorDataset(test_x, test_y)

# 加载器
train_loader = DataLoader(dataset=train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size=16,shuffle=True)
```


```python
# 构建网络
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(2, 5)
        self.out = nn.Linear(5, 4)
    
    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.out(x)
        return x

net = model()
# 交叉熵损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
opt = torch.optim.SGD(net.parameters(), lr=0.01)
```


```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
```


```python
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(2, 4)
        self.out = nn.liner(4, 1)
    
    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.out(x)
        return x
```


```python
net = model()

loss_fn = nn.MSELoss()
opt = opt
```


```python
# 训练
for epoch in range(1000):
    for i, data in enumerate(train_loader):
        x, y = data
        pred = net(x)
        loss = loss_fn(pred, y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    if(epoch%100==0):
        print(loss)
```

    tensor(1.2986, grad_fn=<NllLossBackward0>)
    tensor(0.0348, grad_fn=<NllLossBackward0>)
    tensor(0.0084, grad_fn=<NllLossBackward0>)
    tensor(0.0275, grad_fn=<NllLossBackward0>)
    tensor(0.0479, grad_fn=<NllLossBackward0>)
    tensor(0.0559, grad_fn=<NllLossBackward0>)
    tensor(0.0055, grad_fn=<NllLossBackward0>)
    tensor(0.0015, grad_fn=<NllLossBackward0>)
    tensor(0.0134, grad_fn=<NllLossBackward0>)
    tensor(0.0884, grad_fn=<NllLossBackward0>)



```python
def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)
```


```python
# 对测试集进行预测
# 将测试集绘制出来
print(test_x.shape, test_y.shape, test_x[2], target)
plt.scatter(test_x[:, 0], test_y, c=target[1080:], marker='o')

rights = 0
length = 0
for i, data in enumerate(test_loader):
    x, y = data
    pred = net(x)
    rights = rights + rightness(pred, y)[0]
    length = length + rightness(pred, y)[1]
    print(y)
    print(torch.max(pred.data, 1)[1], '\n')

print(rights, length, rights/length)
```

    torch.Size([120, 2]) torch.Size([120]) tensor([4.7586, 6.5412]) tensor([1, 0, 1,  ..., 0, 0, 1])
    tensor([0, 0, 2, 1, 1, 0, 1, 0, 3, 0, 1, 3, 3, 1, 1, 1])
    tensor([0, 0, 2, 1, 1, 0, 1, 0, 3, 0, 1, 3, 3, 1, 1, 1]) 
    
    tensor([0, 1, 2, 1, 2, 0, 0, 1, 2, 2, 3, 2, 1, 3, 2, 2])
    tensor([0, 1, 2, 1, 2, 0, 0, 1, 2, 2, 3, 2, 0, 3, 2, 2]) 
    
    tensor([0, 0, 3, 0, 3, 0, 2, 0, 2, 1, 3, 0, 3, 2, 0, 0])
    tensor([0, 0, 3, 0, 3, 0, 2, 0, 2, 1, 3, 0, 3, 2, 0, 0]) 
    
    tensor([3, 3, 0, 0, 2, 0, 0, 3, 0, 3, 3, 1, 2, 2, 3, 3])
    tensor([3, 3, 0, 0, 2, 0, 0, 3, 0, 3, 3, 1, 2, 2, 3, 3]) 
    
    tensor([0, 0, 1, 0, 1, 0, 3, 2, 1, 1, 1, 2, 2, 2, 0, 0])
    tensor([0, 0, 1, 0, 1, 0, 3, 2, 1, 1, 1, 2, 2, 2, 0, 0]) 
    
    tensor([2, 1, 1, 0, 3, 2, 2, 1, 1, 1, 1, 0, 2, 2, 0, 2])
    tensor([2, 1, 1, 0, 3, 2, 2, 1, 1, 1, 1, 0, 2, 2, 0, 2]) 
    
    tensor([0, 1, 3, 2, 2, 2, 3, 1, 1, 0, 3, 2, 0, 3, 1, 0])
    tensor([0, 1, 3, 2, 2, 2, 3, 1, 1, 0, 3, 2, 0, 3, 1, 0]) 
    
    tensor([2, 1, 2, 1, 1, 1, 0, 3])
    tensor([2, 1, 2, 1, 1, 1, 0, 3]) 
    
    tensor(119) 120 tensor(0.9917)




![png](deep-learning-test03/output_9_1.png)
    

## 鸢尾花分类

* 通过花萼长度，花萼宽度，花瓣长度，花瓣宽度4个特征
* 使用神经网络训练一个分类器对数据集iris.csv进行分类


```python
# 导入相关函数
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
```


```python
# 数据预处理
data = pd.read_csv('iris.csv')
for i in range(len(data)):
    if data.loc[i, 'Species'] == 'setosa':
        data.loc[i, 'Species'] = 0
    if data.loc[i, 'Species'] == 'versicolor':
        data.loc[i, 'Species'] = 1
    if data.loc[i, 'Species'] == 'virginica':
        data.loc[i, 'Species'] = 2

data.head()
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>




```python
data = data.drop('Unnamed: 0', axis=1)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>




```python
data = shuffle(data)
print(data.head())
data.index = range(len(data))
data.head()
```

         Sepal.Length  Sepal.Width  Petal.Length  Petal.Width Species
    62            6.0          2.2           4.0          1.0       1
    122           7.7          2.8           6.7          2.0       2
    130           7.4          2.8           6.1          1.9       2
    125           7.2          3.2           6.0          1.8       2
    112           6.8          3.0           5.5          2.1       2

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.0</td>
      <td>2.2</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.7</td>
      <td>2.8</td>
      <td>6.7</td>
      <td>2.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.4</td>
      <td>2.8</td>
      <td>6.1</td>
      <td>1.9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.2</td>
      <td>3.2</td>
      <td>6.0</td>
      <td>1.8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.8</td>
      <td>3.0</td>
      <td>5.5</td>
      <td>2.1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>

</div>




```python
# 将数据进行标准化处理
col_titles = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
for i in col_titles:
    mean, std = data[i].mean(), data[i].std()
    data[i] = (data[i]-mean)/std

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Sepal.Length</th>
      <th>Sepal.Width</th>
      <th>Petal.Length</th>
      <th>Petal.Width</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.189196</td>
      <td>-1.966964</td>
      <td>0.137087</td>
      <td>-0.261511</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.242172</td>
      <td>-0.590395</td>
      <td>1.666574</td>
      <td>1.050416</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.879882</td>
      <td>-0.590395</td>
      <td>1.326688</td>
      <td>0.919223</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.638355</td>
      <td>0.327318</td>
      <td>1.270040</td>
      <td>0.788031</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.155302</td>
      <td>-0.131539</td>
      <td>0.986802</td>
      <td>1.181609</td>
      <td>2</td>
    </tr>
  </tbody>
</table>

</div>




```python
# 数据集处理
# 划分训练集和测试集
train_data = data[:-32]
train_x = train_data.drop(['Species'], axis=1).values
train_y = train_data['Species'].values.astype(int)
train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
train_y = torch.from_numpy(train_y).type(torch.LongTensor)

test_data = data[-32:]
test_x = test_data.drop(['Species'], axis=1).values
test_y = test_data['Species'].values.astype(int)
test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
test_y = torch.from_numpy(test_y).type(torch.LongTensor)

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)
```


```python
# 构建神经网络
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(4, 5)
        self.out = nn.Linear(5, 3)
    
    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.out(x)
        return x
    
net = model()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=0.05)
```


```python
# 模型训练
for epoch in range(10000):
    for i, data in enumerate(train_loader):
        x, y = data
        pred = net(x)
        loss = loss_fn(pred, y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    if epoch%1000==0:
        print('第', epoch, '个epoch，损失值为：', loss)
```

    第 0 个epoch，损失值为： tensor(1.0210, grad_fn=<NllLossBackward0>)
    第 1000 个epoch，损失值为： tensor(0.0007, grad_fn=<NllLossBackward0>)
    第 2000 个epoch，损失值为： tensor(0.0010, grad_fn=<NllLossBackward0>)
    第 3000 个epoch，损失值为： tensor(0.0004, grad_fn=<NllLossBackward0>)
    第 4000 个epoch，损失值为： tensor(0.0008, grad_fn=<NllLossBackward0>)
    第 5000 个epoch，损失值为： tensor(0.0007, grad_fn=<NllLossBackward0>)
    第 6000 个epoch，损失值为： tensor(0.0010, grad_fn=<NllLossBackward0>)
    第 7000 个epoch，损失值为： tensor(0.0001, grad_fn=<NllLossBackward0>)
    第 8000 个epoch，损失值为： tensor(0.0006, grad_fn=<NllLossBackward0>)
    第 9000 个epoch，损失值为： tensor(0.0001, grad_fn=<NllLossBackward0>)



```python
def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)
```


```python
# 验证测试
net = net.cpu()
rights = 0
length = 0
for i, data in enumerate(test_loader):
    x, y = data
    pred = net(x)
    rights = rights + rightness(pred, y)[0]
    length = length + rightness(pred, y)[1]
    print(y)
    print(torch.max(pred.data, 1)[1], '\n')

print(rights, length, rights/length)
```

    tensor([2, 0, 1, 2, 1, 2, 1, 2, 2, 0, 1, 1, 0, 2, 0, 0])
    tensor([2, 0, 1, 2, 2, 2, 1, 2, 2, 0, 1, 1, 0, 2, 0, 0]) 
    
    tensor([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 2, 1, 1, 1, 2])
    tensor([2, 0, 1, 1, 0, 1, 0, 1, 0, 2, 1, 2, 1, 1, 1, 2]) 
    
    tensor(29) 32 tensor(0.9062)


