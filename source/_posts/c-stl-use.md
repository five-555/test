---
title: C++中常见容器的使用方法
categories: 技术研究
date: 2024-07-12 09:52:08
tags:
cover:
top_img:
---


# 序列容器

> 不知道如何解释，凭感觉吧，就是我们想象中的那样。

## vector

`vector`是我们常用的动态数组，它通过模板泛型编程的方式，让我们能够使用`vector`来存储任意对象。

`vector`的底层是内存当中一段连续的地址空间，地址空间连续的最大好处就是能够支持用户的随机访问，我们可以和操作数组一样，使用下标的方式来获取到容器中的对象。在`vector`中有两个关键的成员变量`size`和`capacity`，其中`size`表示的是当前容器中存储的对象实际个数，而`capacity`则是容器的最大容量。这两个成员是实现`vector`扩容的关键。


**常用的接口**

| 类别     | 成员函数                 | 描述                                                   |
|----------|--------------------------|--------------------------------------------------------|
| 容量相关 | `size()`                 | 返回容器中元素的个数。                                 |
|          | `max_size()`             | 返回容器所能容纳的最大元素数量。                       |
|          | `resize(n, val)`         | 调整容器的大小到 `n` 个元素，如果 `n` 大，则以 `val` 填充新位置。 |
|          | `capacity()`             | 返回不需要重新分配内存空间的情况下容器可容纳的元素数量。|
|          | `empty()`                | 检测容器是否为空。                                     |
|          | `reserve(n)`             | 请求容器容量至少为 `n` 个元素。                        |
|          | `shrink_to_fit()`        | 请求减少容器容量以节省空间。（C++11）                  |
| 修改内容 | `clear()`                | 移除所有元素，容器大小变为 0。                         |
|          | `insert(pos, elem)`      | 在迭代器 `pos` 指定的位置插入元素 `elem`。             |
|          | `emplace(position, args...)` | 在迭代器 `position` 指定的位置就地构造元素。（C++11）|
|          | `erase(pos)`             | 删除迭代器 `pos` 指定位置上的元素。                    |
|          | `push_back(elem)`        | 在容器尾部添加一个新元素 `elem`。                      |
|          | `emplace_back(args...)`  | 在容器尾部就地构造一个新元素。（C++11）                |
|          | `pop_back()`             | 移除容器尾部的最后一个元素。                           |
|          | `resize(n)`              | 改变容器中元素的数量为 `n`。                           |
|          | `swap(vec)`              | 与另一个同类型向量 `vec` 交换数据。                    |
| 元素访问 | `operator[]`             | 访问指定位置的元素。                                   |
|          | `at(index)`              | 访问指定位置的元素，包含边界检查。                     |
|          | `front()`                | 访问第一个元素。                                       |
|          | `back()`                 | 访问最后一个元素。                                     |
|          | `data()`                 | 返回指向容器头部元素的指针。                           |
| 迭代器   | `begin()` / `cbegin()`   | 返回指向容器开始的迭代器。                             |
|          | `end()` / `cend()`       | 返回指向容器结束（最后元素的下一个位置）的迭代器。     |
|          | `rbegin()` / `crbegin()` | 返回反向迭代器的起始位置。                             |
|          | `rend()` / `crend()`     | 返回反向迭代器的结束位置。                             |

* `vector`扩容机制
    
    > 通常能够触发扩容的行为有：往容器中添加元素、使用`resize`或`reserve`使大小超过容量、通过构造函数初始化超出默认容量

    `vector`扩容机制是通过重新分配内存来实现的，当向容器中添加元素，并且元素的数量超过了其内部数组容量时，会自动触发扩容。

    1、重新分配：首先，`vector` 会申请一个新的、更大的内存块来存储元素。新容量通常是当前容量的两倍，但这个增长因子并不是标准规定的，可能因不同的库实现而异。

    2、拷贝或移动元素：现有元素会被拷贝（或移动，如果它们支持 `move` 语义）到新的内存地址。

    3、释放旧内存：一旦旧元素被成功转移，原来的内存块将被释放。

    4、更新容量：`vector` 更新其容量值以反映新内存块的大小。

* `reserve`和`resize`
 
    如果我们能够预知到我们需要使用的`vector`容器的最大个数，我们可以使用`reserve`来为我们需要用到的容器，提前申请一片足够大的内存，这样就可以保证在程序运行过程中不需要去对容器进行扩容，因为`vector`的扩容是一个代价相对较大的一件事。

    `resize`是重新为容器设置`size`大小，如果容器当前的对象大于设置的`size`，则容器会被截断，如果小于，则会使用默认构造去填充新的后续空间，如果比`capacity`还要大，会涉及到触发容器的扩容。

    `reserve`是重新设置容量的大小，不会更改`size`的值，只有当需要申请的容量大于`capacity`，才会起作用，否则不会起任何作用。

* `push_back`和`emplace_back`

    使用`push_back()`向`vector`添加元素时，是在向函数传递一个已经存在的对象。这个对象是通过调用拷贝构造函数或移动构造函数来添加到容器末尾的。如果传递的是左值（例如一个变量），那么会调用拷贝构造函数；如果传递的是右值，则会调用移动构造函数（前提是该类型支持移动语义）。

    `emplace_back()`不需要传入一个已经构造好的对象，而是接受任意数量和类型的参数，并将它们直接传递给元素类型的构造函数。也就是说`emplace_back()`试图在容器管理的内存区域中直接构造对象，避免了额外的拷贝或移动步骤。

## list

`list`是我们常用的双向链表，通过节点来组织起来的数据结构，节点中会存有两个指针分别用来指向下一个节点的地址，和上一个节点的地址，通过链接的方式就能够形成一个双向链表的结构。通常链表在内存当中的地址并不是连续的，所以链表并不支持随机存取，无法通过下标的方式来进行访问。但是相比于`vector`来说，链表对于插入和删除工作来说更具有优势。

其实，我们能够很容易的自己构建自己的链表，只需要定义一个包含指向下一个节点的指针就可以了，所以在实际应用中，我个人用`list`这个数据结构并不是很多。自己的链表可以根据实际业务来给定相关特定的属性。

**常用的接口**

| 类别         | 成员函数                       | 描述                                                         |
|--------------|--------------------------------|--------------------------------------------------------------|
| 容量相关     | `size()`                       | 返回容器中元素的个数。                                       |
|              | `empty()`                      | 检查容器是否为空。                                           |
|              | `max_size()`                   | 返回容器所能容纳的最大元素数量。                             |
| 修改内容     | `clear()`                      | 移除所有元素，容器大小变为 0。                               |
|              | `insert(position, val)`        | 在迭代器 `position` 指定的位置插入值为 `val` 的元素。       |
|              | `emplace(position, args...)`   | 在迭代器 `position` 指定的位置就地构造元素。（C++11）      |
|              | `erase(position)` / `erase(first, last)` | 删除位于迭代器 `position` 或 `[first, last)` 范围内的元素。|
|              | `push_back(val)`               | 在容器尾部添加一个新元素。                                   |
|              | `emplace_back(args...)`        | 在容器尾部就地构造一个新元素。（C++11）                     |
|              | `pop_back()`                   | 移除容器尾部的最后一个元素。                                 |
|              | `push_front(val)`              | 在容器头部添加一个新元素。                                   |
|              | `emplace_front(args...)`       | 在容器头部就地构造一个新元素。（C++11）                     |
|              | `pop_front()`                  | 移除容器头部的第一个元素。                                   |
|              | `resize(n)`                    | 改变容器中元素的数量为 `n`。                                 |
|              | `swap(list)`                   | 与另一个同类型列表 `list` 交换数据。                         |
|              | `merge(x)`                     | 合并两个已排序的列表。                                       |
|              | `remove(val)`                  | 移除所有值为 `val` 的元素。                                   |
|              | `remove_if(predicate)`         | 移除满足特定条件的所有元素。                                 |
|              | `reverse()`                    | 反转列表中的元素顺序。                                       |
|              | `sort()`                       | 对列表中的元素进行排序。                                     |
|              | `unique()`                     | 移除连续重复的元素。                                         |
| 元素访问     | `front()`                      | 访问第一个元素。                                             |
|              | `back()`                       | 访问最后一个元素。                                           |
| 迭代器       | `begin()` / `cbegin()`         | 返回指向容器开始的迭代器。                                   |
|              | `end()` / `cend()`             | 返回指向容器结束（最后元素的下一个位置）的迭代器。           |
|              | `rbegin()` / `crbegin()`       | 返回反向迭代器的起始位置。                                   |
|              | `rend()` / `crend()`           | 返回反向迭代器的结束位置。                                   |

**链表Q&A**

* 检查链表是否有环

    快慢指针

* 反转链表

    头插法

* 合并有序链表

    归并

* 寻找倒数第k个节点

    先后节点

* 旋转链表

    多次反转链表

## array

`array`是一种固定大小的数组，相对于传统的数组来说，提供了更加安全和方便的接口，它的长度在编译的时候就已经确定，所以不支持动态的调整数组的大小。

**常用的接口**

| 类别         | 成员函数                   | 描述                                                         |
|--------------|----------------------------|--------------------------------------------------------------|
| 容量相关     | `size()`                   | 返回数组中元素的个数。                                       |
|              | `max_size()`               | 返回数组所能容纳的最大元素数量。                             |
|              | `empty()`                  | 检查数组是否为空。                                           |
| 元素访问     | `operator[]`               | 使用下标访问元素，不进行边界检查。                           |
|              | `at(size_type n)`          | 使用下标访问元素，进行边界检查。                             |
|              | `front()`                  | 访问第一个元素。                                             |
|              | `back()`                   | 访问最后一个元素。                                           |
|              | `data()`                   | 返回指向数组首个元素的指针。                                 |
| 修改内容     | `fill(const T& value)`     | 用指定的值填充整个数组。                                     |
|              | `swap(array& other)`       | 交换两个数组的内容。                                         |
| 迭代器       | `begin()` / `cbegin()`     | 返回指向数组首个元素的迭代器。                               |
|              | `end()` / `cend()`         | 返回指向数组末尾元素之后位置的迭代器。                       |
|              | `rbegin()` / `crbegin()`   | 返回指向数组最后一个元素的反向迭代器。                       |
|              | `rend()` / `crend()`       | 返回指向数组第一个元素之前位置的反向迭代器。                 |


## deque

`deque`是`STL`容器中的双端队列，是支持能够从头增加元素的一种容器，相比于`vector`在插入上来说，会更加灵活。`deque`的内部不是使用单一连续内存块来存储元素的，而是采用一个中央控制器来管理多个固定大小的数组（成为缓冲区或段）。每个缓冲区可能存储多个元素，中央控制器是一个动态数组，包含指向这些缓冲区的指针。这样的设计，能够使得在两端添加或者一处元素时无需移动其他元素。

`deque`和`vector`都支持随机访问，但是相比于`vector`之下，`deque`对某一个元素的访问，效率相比之下会低一些，`deque`所支持的随机访问并不是通过首地址+偏移量来直接映射到内存地址空间的，而是会通过一个中央控制器的结构来计算出所要查询的元素所在的块以及在块中的偏移量。

**常用的接口**

| 类别         | 成员函数                               | 描述                                                         |
|--------------|----------------------------------------|--------------------------------------------------------------|
| 容量相关     | `size()`                               | 返回容器中元素的个数。                                       |
|              | `empty()`                              | 检查容器是否为空。                                           |
|              | `max_size()`                           | 返回容器所能容纳的最大元素数量。                             |
| 修改内容     | `clear()`                              | 移除所有元素，容器大小变为 0。                               |
|              | `insert(position, val)`                | 在迭代器 `position` 指定的位置插入值为 `val` 的元素。       |
|              | `insert(position, n, val)`             | 在迭代器 `position` 指定的位置插入 `n` 个值为 `val` 的元素。|
|              | `insert(position, first, last)`        | 在迭代器 `position` 指定的位置插入来自范围 `[first, last)` 的元素。|
|              | `emplace(position, args...)`           | 在迭代器 `position` 指定的位置就地构造元素。（C++11）      |
|              | `erase(position)` / `erase(first, last)` | 删除位于迭代器 `position` 或 `[first, last)` 范围内的元素。|
|              | `push_back(val)`                       | 在容器尾部添加一个新元素。                                   |
|              | `emplace_back(args...)`                | 在容器尾部就地构造一个新元素。（C++11）                     |
|              | `pop_back()`                           | 移除容器尾部的最后一个元素。                                 |
|              | `push_front(val)`                      | 在容器头部添加一个新元素。                                   |
|              | `emplace_front(args...)`               | 在容器头部就地构造一个新元素。（C++11）                     |
|              | `pop_front()`                          | 移除容器头部的第一个元素。                                   |
|              | `resize(n)`                            | 改变容器中元素的数量为 `n`。                                 |
|              | `swap(deque)`                          | 与另一个同类型双端队列 `deque` 交换数据。                   |
| 元素访问     | `front()`                              | 访问第一个元素。                                             |
|              | `back()`                               | 访问最后一个元素。                                           |
|              | `operator[]`                           | 使用下标访问元素，不进行边界检查。                           |
|              | `at()`                                 | 使用下标访问元素，进行边界检查。                             |
| 迭代器       | `begin()` / `cbegin()`                 | 返回指向容器开始的迭代器。                                   |
|              | `end()` / `cend()`                     | 返回指向容器结束（最后元素的下一个位置）的迭代器。           |
|              | `rbegin()` / `crbegin()`               | 返回反向迭代器的起始位置。                                   |
|              | `rend()` / `crend()`                   | 返回反向迭代器的结束位置。                                   |

# 容器适配器

前面我们讲了`deque`的相关用法，我们可以发现，这是一种双端都不受限制的容器，因为在容器的头部和尾部，我们都可以进行容器元素的插入和删除操作，而如果我们把容器的某一端的插入或者删除操作给予一定的限制，那么就可以衍生出另外两种常用的容器队列`queue`和栈`stack`，而在我们的`C++`容器中，`queue`和`stack`的底层默认便是基于`deque`来实现的。

## queue

`queue`是一个先进先出`FIFO`的数据结构，底层是基于`deque`来实现的，通常我们会限制`deque`一端的入队和另一端的出队，这样我们就能够实现一个队列的数据结构，通常用于按照顺序处理元素的场景。比较经典的就是在我们的广度优先算法中使用比较广泛，还有树的层序遍历。

```C++
#include <iostream>
#include <vector>
#include <queue>

void bfs(const std::vector<std::vector<int>>& graph, int start) {
    std::queue<int> q;
    std::vector<bool> visited(graph.size(), false);

    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        std::cout << "Visited node: " << node << std::endl;

        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                q.push(neighbor);
                visited[neighbor] = true;
            }
        }
    }
}

int main() {
    // 这里图的描述方式是邻接表的方式
    std::vector<std::vector<int>> graph = {
        {1, 2},
        {0, 3, 4},
        {0, 4},
        {1, 5},
        {1, 2, 5},
        {3, 4}
    };

    bfs(graph, 0);
    return 0;
}

```

**常用的接口**

| 接口          | 描述                                           | 示例代码                           |
|---------------|------------------------------------------------|------------------------------------|
| `push(value)` | 将元素添加到队列的末尾                         | `q.push(1);`                       |
| `pop()`       | 移除队列的第一个元素                           | `q.pop();`                         |
| `front()`     | 返回队列的第一个元素，但不删除                 | `int front = q.front();`           |
| `back()`      | 返回队列的最后一个元素，但不删除                | `int back = q.back();`             |
| `empty()`     | 检查队列是否为空                               | `bool isEmpty = q.empty();`        |
| `size()`      | 返回队列中元素的个数                           | `std::size_t size = q.size();`     |
| `emplace(args...)` | 就地构造元素并添加到队列的末尾           | `q.emplace(2);`                    |
| `swap(queue)` | 交换两个队列的内容                             | `q1.swap(q2);`                     |

## stack

`stack`也是一种受限制的双端队列，我们会同时限制一端的入队和出队，这样就能够实现一个栈的性质。适用的场景有：表达式求值（后缀表达式）、括号匹配、深度优先遍历等。

```C++
#include <iostream>
#include <vector>
#include <stack>

void dfs(const std::vector<std::vector<int>>& graph, int start) {
    std::stack<int> s;
    std::vector<bool> visited(graph.size(), false);

    s.push(start);
    visited[start] = true;

    while (!s.empty()) {
        int node = s.top();
        s.pop();
        std::cout << "Visited node: " << node << std::endl;

        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                s.push(neighbor);
                visited[neighbor] = true;
            }
        }
    }
}

int main() {
    std::vector<std::vector<int>> graph = {
        {1, 2},
        {0, 3, 4},
        {0, 4},
        {1, 5},
        {1, 2, 5},
        {3, 4}
    };

    dfs(graph, 0);
    return 0;
}
```

**常用的接口**

| 接口            | 描述                                           | 示例代码                           |
|-----------------|------------------------------------------------|------------------------------------|
| `push(value)`   | 将元素添加到栈顶                               | `s.push(1);`                       |
| `pop()`         | 移除栈顶的元素                                 | `s.pop();`                         |
| `top()`         | 返回栈顶的元素，但不删除                       | `int top = s.top();`               |
| `empty()`       | 检查栈是否为空                                 | `bool isEmpty = s.empty();`        |
| `size()`        | 返回栈中元素的个数                             | `std::size_t size = s.size();`     |
| `emplace(args...)` | 就地构造元素并添加到栈顶                   | `s.emplace(2);`                    |
| `swap(stack)`   | 交换两个栈的内容                               | `s1.swap(s2);`                     |


## priority_queue

[参考：C++优先队列的用法](https://www.zdon.fun/2024/03/06/c-priority-queue/)

# 关联容器

关联容器和顺序容器底层的实现方式存在一定的差异，关联容器通常底层的实现采用的数的结构，我们这里用到的是红黑树，而红黑树是一种相对平衡的二叉搜索树。

**红黑树的特点**

自平衡：红黑树通过颜色属性（红或黑）和旋转操作来保持树的平衡，确保树的高度约为 O(log n)。

节点颜色规则：

    每个节点是红色或黑色。

    根节点是黑色。

    叶子节点（NIL节点）是黑色。

    红色节点的子节点必须是黑色（即红色节点不能连续）。

    从任一节点到其每个叶子的路径都包含相同数量的黑色节点。

旋转操作：红黑树使用左旋和右旋操作来调整树的结构以保持平衡。

## map

`map`是`C++`当中的关联容器，可以将元素按照键值对的形式组织起来，在默认情况下，我们对一个`map`使用迭代器进行遍历的话，我们可以发现`map`中的`key`是按照递增的方式来输出的，这是因为对`map`的遍历过程实际上就是一个二叉搜索树的中序遍历的一个过程。

通常情况下，我们的`key`是唯一的，如果我们在已经存在`key`的情况之下，再次插入同样的`key`和`value`的话，只会对原始`key`所对应的`value`进行更新操作，如果我们想要存储同一个`key`的多个`value`值，我们可以使用`STL`中的另一个容器`mutilmap`。

在我们构造`map`时，默认会按照`key`的升序构造，同时，我们可以通过加入`greater`参数来让我们的`map`按照`key`的降序构造。

```C++
std::map<int, std::string, std::greater<int>> m;

// 自定义比较函数，小于升序，大于降序
struct CustomCompare {
    bool operator()(int a, int b) const {
        return a > b;
    }
};

std::map<int, std::string, CustomCompare> m;

```

如果想要使用自定义的对象作为`map`的`key`，我们需要在对象中，对`<`进行运算符重载，表示自定义的比较函数。

```C++
class Point {
public:
    int x, y;

    Point(int x, int y) : x(x), y(y) {}

    bool operator<(const Point& other) const {
        if (x != other.x) return x < other.x;
        return y < other.y;
    }

    // 用于输出 Point 对象
    friend std::ostream& operator<<(std::ostream& os, const Point& point) {
        os << "(" << point.x << ", " << point.y << ")";
        return os;
    }
};

int main() {
    std::map<Point, std::string> pointMap;

    // 插入键值对
    pointMap[Point(1, 2)] = "Point 1";
    pointMap[Point(2, 3)] = "Point 2";
    pointMap[Point(1, 3)] = "Point 3";

    // 遍历并输出 map 中的键值对
    for (const auto& pair : pointMap) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    return 0;
}
```

**常用的接口**

| 接口                     | 描述                                                         | 示例代码                                      |
|--------------------------|--------------------------------------------------------------|-----------------------------------------------|
| `insert({key, value})`   | 插入键值对                                                   | `m.insert({1, "one"});`                       |
| `erase(iterator)`        | 移除指定位置的元素                                           | `m.erase(m.begin());`                         |
| `erase(key)`             | 移除指定键的元素                                             | `m.erase(1);`                                 |
| `find(key)`              | 查找键，返回指向键值对的迭代器，不存在则返回 `end()`         | `auto it = m.find(1);`                        |
| `operator[](key)`        | 访问或插入指定键的元素                                       | `std::string value = m[1];`                   |
| `begin()` / `end()`      | 返回指向第一个元素和最后一个元素之后位置的迭代器             | `for (auto it = m.begin(); it != m.end(); ++it)` |
| `size()`                 | 返回元素个数                                                 | `size_t size = m.size();`                     |
| `empty()`                | 检查容器是否为空                                             | `bool isEmpty = m.empty();`                   |
| `clear()`                | 清空所有元素                                                 | `m.clear();`                                  |
| `emplace(key, value)`    | 插入键值对，如果键已存在则不插入                             | `m.emplace(4, "four");`                       |
| `count(key)`             | 返回指定键的元素个数（0或1）                                 | `size_t count = m.count(2);`                  |

## set

`set`可以理解成`key`和`value`均相等的`map`。其余的性质和`map`都类似，只是在`map`中存储的是键值对，而在`set`中存储的是`key`。

**常用的接口**

| 接口                   | 描述                                                         | 示例代码                                      |
|------------------------|--------------------------------------------------------------|-----------------------------------------------|
| `insert(value)`        | 插入元素                                                     | `s.insert(10);`                               |
| `erase(iterator)`      | 移除指定位置的元素                                           | `s.erase(s.begin());`                         |
| `erase(value)`         | 移除指定值的元素                                             | `s.erase(10);`                                |
| `find(value)`          | 查找元素，返回指向元素的迭代器，不存在则返回 `end()`         | `auto it = s.find(10);`                       |
| `count(value)`         | 返回元素的数量（`set` 中只可能为 0 或 1）                    | `size_t count = s.count(10);`                 |
| `begin()` / `end()`    | 返回指向第一个元素和最后一个元素之后位置的迭代器             | `for (auto it = s.begin(); it != s.end(); ++it)` |
| `size()`               | 返回元素个数                                                 | `size_t size = s.size();`                     |
| `empty()`              | 检查容器是否为空                                             | `bool isEmpty = s.empty();`                   |
| `clear()`              | 清空所有元素  

## unordered_map

从命名我们就可以看出，`unordered_map`是无序的，无序所指的是它的`key`，当我们需要遍历`unordered_map`的时候，输出的键值对的顺序是随机的。

在`unordered_map`的底层是用哈希表来完成的，所以我们在插入，以及查询哈希表中的元素可以在`O(1)`的时间复杂度以内完成，前提是在哈希冲突较少的情况下。

**常用的接口**

| 接口                   | 描述                                                         | 示例代码                                      |
|------------------------|--------------------------------------------------------------|-----------------------------------------------|
| `insert({key, value})` | 插入键值对                                                   | `um.insert({1, "one"});`                       |
| `erase(iterator)`      | 移除指定位置的元素                                           | `um.erase(um.begin());`                         |
| `erase(key)`           | 移除指定键的元素                                             | `um.erase(1);`                                |
| `find(key)`            | 查找键，返回指向键值对的迭代器，不存在则返回 `end()`         | `auto it = um.find(1);`                       |
| `operator[](key)`      | 访问或插入指定键的元素                                       | `std::string value = um[1];`                   |
| `begin()` / `end()`    | 返回指向第一个元素和最后一个元素之后位置的迭代器             | `for (auto it = um.begin(); it != um.end(); ++it)` |
| `size()`               | 返回元素个数                                                 | `size_t size = um.size();`                     |
| `empty()`              | 检查容器是否为空                                             | `bool isEmpty = um.empty();`                   |
| `clear()`              | 清空所有元素                                                 | `um.clear();`                                  |
| `emplace(key, value)`  | 插入键值对，如果键已存在则不插入                             | `um.emplace(4, "four");`                       |
| `count(key)`           | 返回指定键的元素个数（0或1）                                 | `size_t count = um.count(2);`                  |

## unordered_set

`unordered_set`是`C++`中的无序容器，它存储唯一的元素，并且以无序的方式进行组织。底层实现是基于哈希表，因此插入和查找元素的时间复杂度是常数级别的。

**常用的接口**

| 接口                   | 描述                                                         | 示例代码                                      |
|------------------------|--------------------------------------------------------------|-----------------------------------------------|
| `insert(value)`        | 插入元素                                                     | `us.insert(10);`                              |
| `erase(iterator)`      | 移除指定位置的元素                                           | `us.erase(us.begin());`                       |
| `erase(value)`         | 移除指定值的元素                                             | `us.erase(10);`                               |
| `find(value)`          | 查找元素，返回指向元素的迭代器，不存在则返回 `end()`         | `auto it = us.find(10);`                      |
| `count(value)`         | 返回元素的数量（`unordered_set` 中只可能为 0 或 1）          | `size_t count = us.count(10);`                |
| `begin()` / `end()`    | 返回指向第一个元素和最后一个元素之后位置的迭代器             | `for (auto it = us.begin(); it != us.end(); ++it)` |
| `size()`               | 返回元素个数                                                 | `size_t size = us.size();`                    |
| `empty()`              | 检查容器是否为空                                             | `bool isEmpty = us.empty();`                  |
| `clear()`              | 清空所有元素                                                 | `us.clear();`                                 |


## unordered_set

`unordered_set`是`C++`中的无序容器，它存储唯一的元素，并且以无序的方式进行组织。底层实现是基于哈希表，因此插入和查找元素的时间复杂度是常数级别的。

**常用的接口**

| 接口                   | 描述                                                         | 示例代码                                      |
|------------------------|--------------------------------------------------------------|-----------------------------------------------|
| `insert(value)`        | 插入元素                                                     | `us.insert(10);`                              |
| `erase(iterator)`      | 移除指定位置的元素                                           | `us.erase(us.begin());`                       |
| `erase(value)`         | 移除指定值的元素                                             | `us.erase(10);`                               |
| `find(value)`          | 查找元素，返回指向元素的迭代器，不存在则返回 `end()`         | `auto it = us.find(10);`                      |
| `count(value)`         | 返回元素的数量（`unordered_set` 中只可能为 0 或 1）          | `size_t count = us.count(10);`                |
| `begin()` / `end()`    | 返回指向第一个元素和最后一个元素之后位置的迭代器             | `for (auto it = us.begin(); it != us.end(); ++it)` |
| `size()`               | 返回元素个数                                                 | `size_t size = us.size();`                    |
| `empty()`              | 检查容器是否为空                                             | `bool isEmpty = us.empty();`                  |
| `clear()`              | 清空所有元素                                                 | `us.clear();`                                 |
