---
title: C++优先队列用法
categories: 技术研究
date: 2024-03-06 15:13:01
tags: [C++, 优先队列]
cover:
top_img:
---
## C++优先队列用法

优先队列是C++中STL的派生容器，它仅考虑最高优先级元素，队列遵循FIFO策列，而优先队列根据优先级弹出元素，优先级最高的元素首先弹出。

函数原型：`priority_queue<Type, Container, Functional> m_queue;`

Type：表示数据类型

Container：表示容器类型（必须是用数组实现的容器，比如vector，deque等）

Functional：表示比较的方式，当需要用自定义的数据类型时才需要传入这三个参数，使用基本数据类型时，只需要传入数据类型，默认是大顶堆

```c++
// 升序队列
priority_queue <int, vector<int>, greater<int>> q;
// 降序队列
priority_queue <int, vector<int>, less<int>> q;

//greater和less是std实现的两个仿函数（就是使一个类的使用看上去像一个函数。其实现就是类中实现一个operator()，这个类就有了类似函数的行为）
```

### 成员函数

`bool empty() const` ：返回值为true，说明队列为空

`int size() const` ：返回优先队列中元素的数量

`void pop()` ：删除队列顶部的元素，也即根节点

`int top()` ：返回队列中的顶部元素，但不删除该元素

`void push(int arg)`：将元素arg插入到队列之中

### 使用仿函数重载，自定义function参数

小于是升序，大于是降序

> set集合，sort排序方法，也可以自定义仿函数进行比较
>
> `set<int, compare> my_set`，`sort(vec.begin(), vec.end(), compare)`

```c++
struct Node {
    int size;
    int price;
};

// 在类中重载，（）只能重载在类里面
class Cmp {
public:
	// 大于表示降序，小于表示升序
    bool operator()(const Node &a, const Node &b) {
        return a.size == b.size ? a.price > b.price : a.size < b.size;
    }
};
int main() {
    priority_queue<Node, vector<Node>, Cmp> priorityQueue;
    for (int i = 0; i < 5; i++) {
        priorityQueue.push(Node{i, 5 - i});
    }
    while (!priorityQueue.empty()) {
        Node top = priorityQueue.top();
        cout << "size:" << top.size << " price:" << top.price << endl;
        priorityQueue.pop();
    }
    return 0;
}
```

