---
title: 高性能键值存储系统实现
tags:
  - 数据库
  - Redis
  - 数据库缓存
  - C++
categories: 算法实践
date: 2024-02-27 11:29:57
cover:
top_img:
top: true
---
### 

# 简易Redis的实现

这个项目实现了一个简易的redis，使用hashmap来管理键值对数据，使用hashmap+AVL数来管理Zset数据，并实现了hashmap的渐进式扩容，减少因为扩容重新哈希化带来rehash的代价。使用poll技术来实现IO多路复用，完成一个服务端与多个客户端建立连接的过程，使用双向链表来管理连接，通过最小堆来控制val的生存时间，并通过将两者结合的方式，控制poll的最大超时时间，用来确保每一次poll的陷入内核和退出内核在主进程中都有处理的任务。

1、最小堆的维护，当为某一个key设置好ttl时，会将key当中需要维护的ttl放入到最小堆当中，每一次轮询结束以后，会统一进行处理，已经失效的key

2、双向链表的维护，poll当中，会把第一个fd设置成为用于处理连接事件的fd，当有连接事件发生的时候，会处理连接，接受一个新的连接，并将其放入到双向链表的头部，会有一个conn的结构体，里面实现了读写缓存，以及接受当前连接的空闲队列节点，以及建立连接的时间，然后会把这个连接放入到空闲队列当中的头部。

3、过期时间的处理，会在每一次poll以及对应的事件处理结束以后，对当前的key进行ttl的检查，包括conn的过期时间和key的过期时间，会对空闲队列中的一些长期占用时间的连接进行清除，以及最小堆当中过期的key进行清理（不断地pop掉最小堆当中的数据，直到没有那些超时数据）。

4、线程池的作用，当缓存数据过多时，实现异步清除。

## 一、实现的命令

```txt
// hashtable
set key value	向哈希表中插入键值对
get key			从哈希表中查找键对应的值
del key			从哈希表中删除键
keys			打印出哈希表中所有的keys

// zset
zadd key score name		向键为key中，插入name，score
zscore key name			按照key和name查找score
zrem key name			删除键为key中的name元素
zscore key name			查找键为key的name的score
zquery

```



## 二、Socket编程相关语法

### 服务端

* 创建句柄

函数原型：`int socket(int domain, int type, int protocol);`

domin：指定通信域，ipv4或ipv6，AF_INET表示ipv4地址

type：指定套接字类型

protocol：0表示为TCP协议

* 设置socket可选项
* Bind，绑定ip和端口号
* Listen

```c++
void Server::run_server() {
    
    // init coding...

    if (fd < 0) {
        die("socket()");
    }

    int val = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));

    // bind
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = ntohs(1234);
    addr.sin_addr.s_addr = ntohl(0);
    int rv = bind(fd, (const sockaddr *)&addr, sizeof(addr));
    if (rv) {
        die("bind()");
    }

    // listen
    rv = listen(fd, SOMAXCONN);
    if (rv) {
        die("listen()");
    }

    // coding...
}

int main() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    // bind
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = ntohs(1234);
    addr.sin_addr.s_addr = ntohl(0);    // wildcard address 0.0.0.0
    Server server(addr, fd);
    
    server.run_server();
    return 0;
}

```

### 客户端

* 创建句柄
* 设置IP地址和端口号
* Connect

```C++
void Client::run_client(int argc, char **argv) {
    if (fd < 0) {
        die("socket()");
    }

    int rv = connect(fd, (const struct sockaddr *)&addr, sizeof(addr));
    if (rv) {
        die("connect");
    }

    // coding...

    close(fd);
}

int main(int argc, char **argv) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_port = ntohs(1234);
    addr.sin_addr.s_addr = ntohl(INADDR_LOOPBACK);  // 127.0.0.1
    
    Client client(addr, fd);
    client.run_client(argc, argv);

    return 0;
}
```

**使用的协议**

len+msg的结合

len表示后面的字长，msg表示要获取的数据



## 三、事件循环和非阻塞型IO

> 在服务器端网络编程中，处理并发连接有三种方法：forking、多线程（multi-threading）和事件循环（event loops）。forking会为每个客户端连接创建新的进程以实现并发。多线程使用线程而不是进程。事件循环使用轮询和非阻塞 I/O，通常在单个线程上运行。由于进程和线程的开销，大多数现代生产级软件使用事件循环来进行网络编程。

在项目中用到了IO多路复用中的poll技术来实现，在服务端使用单个进程和多个客户端建立连接，在客户端看来，像是每一个客户端都和一个独立的服务端建立了连接并进行数据通信，而在服务端看来，它所完成的则是在不同的时间段服务不同的客户，但由于时间片比较小，就会让客户端感觉好像实现了多个进程通信的过程。


## 四、哈希表数据结构

### 数据结构

* 哈希节点

  哈希表节点是以链表的形式存储，包含一个next节点和一个hcode（哈希化以后的映射值）

```c++
// 哈希表节点
struct HNode {
    HNode *next = NULL;
    uint64_t hcode = 0;
};
```

* 哈希表

  哈希表主要载体是一个HNode的二维指针，第一个指针代表的是哈希值，第二个指针存储的是链表，链表是用来解决哈希冲突的方式。size表示的是当前哈希表中的总结点数目，mask表示掩码，为哈希表第一个维度大小减1。

  在哈希表中实现了，insert，lookup，detach，init接口，分别表示节点的插入，节点的查找，节点的删除。

  需要使用有参的构造函数初始化，传入的参数为整型，整型必须为2的幂，便于构建mask。

  * 节点的插入过程

    1、先计算当前节点中hcode应该属于的pos（即链表位置），计算方式是与`&`上mask

    2、使用头插法，将节点插入到对应的位置，修改size大小

  * 节点的查找过程

    1、查找哈希化后的链表位置。

    2、在链表中查找当前节点。

  * 节点的删除过程`HNode *detach(HNode **from)`

    1、解除引用，获取链表位置

    2、绕过当前节点，指向下一节点

    3、具体使用需要先找到节点位置，再调用detach

```C++
// 哈希表
// 接口: insert, lookup, detach
class HTab {
private:
    HNode **tab = NULL;
    size_t size = 0;
    size_t mask = 0;

public:
    HTab() = default;
    // 传入的n必须为2的幂
    HTab(size_t n): size(0), mask(n-1){
        assert((n & (n-1)) == 0);
        tab = new HNode*[n];
    };
    ~HTab() { delete[] tab; }

    void insert(HNode *node);
    HNode *lookup(HNode *key, bool (*eq)(HNode *, HNode *));
    HNode *detach(HNode **from);

    // setter
    void set_tab(HNode **t) { tab = t; }
    void set_mask(size_t m) { mask = m; }
    void set_size(size_t s) { size = s; }

    // getter
    HNode **get_tab() { return tab; }
    size_t get_mask() { return mask; }
    size_t get_size() { return size; }
};
```

* 哈希map

  哈希表中含有两个HTab，用于完成渐进式的rehash。第一张表存储新的键值，第二张表存储旧的键值，当在第一张表中查找不到元素时，会再在第二张表查找。

  固定值的设置，k_max_load_factor表示负载因子，设置为8，即表示当前的总的HMap的节点数大于Mask的8倍的时候，就会对哈希表进行扩容。k_resizing_work的大小表示将哈希扩容分配到各个语句当中的单次转移的节点数量。resizing_pos记录resizing的位置。

  哈希Map实现了，insert，lookup，pop，size，destroy的接口，分别表示**插入，查找，删除，返回哈希map节点数，以及销毁**。同时会在负载过大时，重新申请更大的内存，执行resize操作，并分发到其他语句当中。

  start_resizing：检查ht2节点数是否大于零，若大于零，说明正在进行rehash中，将指定数量的节点rehash

  start_resizing：分配一个更大的表给ht1，要提前将ht2指向ht1

  * 插入的实现

    1、将节点插入到ht1当中，如果ht1为空，则新建ht1

    2、检查ht2是否为空，如果为空，则检查表1的负载因子，大于特定值，则开始resizing，如果ht2不为空，则说明正在resize，执行help_resizing。

  * 查找的实现

    1、执行start_resizing

    2、分别在ht1和ht2中查找

  * 删除的实现

    1、执行start_resizing

    2、执行查找，用返回节点执行节点删除

```c++

// 哈希map
// 使用两张哈希表用于渐进式rehash
// 接口: insert, lookup, pop, size, destroy
// private: start_resizing, help_resizing
class HMap {
public:
    HTab ht1;   // newer
    HTab ht2;   // older
private:
    size_t resizing_pos = 0;    
    const size_t k_resizing_work = 128; // constant work
    const size_t k_max_load_factor = 8; // constant load factor
    void help_resizing();
    void start_resizing();

public:
    HMap() : ht1(1), ht2(1), resizing_pos(0) {};
    HMap(size_t n): ht1(n), ht2(1), resizing_pos(0) {};
    ~HMap() { ht1.~HTab(); ht2.~HTab(); }

    // 插入、查找、删除、大小、销毁
    void insert(HNode *node);
    HNode *lookup(HNode *key, bool (*eq)(HNode *, HNode *));
    HNode *pop(HNode *key, bool (*eq)(HNode *, HNode *));
    size_t size();
    void destroy();
};
```

实际应用过程中还会有一个结构体

将哈希表节点和键值对进行绑定

```c++
struct Entry {
    struct HNode node;
    std::string key;
    std::string val;
};
```

### 渐进式rehash过程

在哈希表的构建时，使用的是用二维指针来存储哈希表，解决哈希冲突的方式是常用的拉链法（使用链表存储映射值相等的哈希元素），第一个维度作为哈希映射的key值来定位到具体的映射链表，第二个维度则是用来解决哈希冲突的，考虑到哈希表定义本身的属性，我们是希望哈希表的哈希冲突尽可能少，也就是我们的链表的长度尽可能短，随着我们插入的数据不断增加，我们产生的哈希冲突也是会增加的。

在这里我们定义了一个负载因子`k_max_load_factor`，在实际插入的过程中，当当前用于查找的哈希表中的负载（元素个数/掩码）大于负载因子，会启动rehash的过程，我们会申请一张比原来哈希表大两倍的哈希表，将当前哈希表中的数据重新映射到这一张新的哈希表中。这样我们原来短，高的哈希表，就会变成长，矮的哈希表。

我们知道，当哈希表中的元素过多的时候，需要对所有元素都一起进行rehash是一个非常耗时的过程，如果我们只有一张哈希表，我们在进行rehash的过程中，还不能够允许其他插入、删除以及查询操作，为了解决这样一个问题，我们采用将rehash分散在各个其他语句的步骤，在我们负载因子达到一定程度的时候，我们会启动渐进式rehash，当我们有其他插入或者查询语句到来的时候，我们会先进行部分node的rehsah，再进行查询语句。我们可以看下面例子。

```C++
HNode *HMap::lookup(HNode *key, bool (*eq)(HNode *, HNode *)) {
    help_resizing();
    HNode **from = ht1.lookup(key, eq);
    from = from ? from : ht2.lookup(key, eq);
    return from ? *from : NULL;
}

// class HMap
void HMap::help_resizing() {
    size_t nwork = 0;
    while (nwork < k_resizing_work && ht2.get_size() > 0) {
        // scan for nodes from ht2 and move them to ht1
        HNode **from = &ht2.get_tab()[resizing_pos];
        if (!*from) {
            resizing_pos++;
            continue;
        }

        ht1.insert(ht2.detach(from));
        nwork++;
    }

    if (ht2.get_size() == 0 && ht2.get_tab()) {
        // done
        delete[] ht2.get_tab();
        ht2 = HTab{};
    }
}
```



## 五、平衡二叉树

平衡二叉树是一种特殊的二叉搜索树，对平衡二叉树来说，它的中序遍历是有序的，并且左右子树的高度差会处于一个平衡的状态，这样可以保证每次查找都能够在比较快的时间内完成。

### 数据结构

> 平衡二叉树本身就是由节点构成的

depth：以当前节点为根节点所在树的高度

cnt：以当前节点为根节点的总节点数，便于进行区间统计

left：左节点，right：右节点，parent：父节点

提供的辅助函数

* rot_left：对当前节点进行左旋操作

```
// 对b节点进行左旋，返回d节点
  b         d
 / \       /
a   d ==> b
   /     / \
  c     a   c
```

* rot_right：镜像操作

* avl_fix_left

  1、左子树的右子树太深需要先左旋后右旋

  2、左子树的左子树太深，直接右旋

```
   root
   /            左旋(root->left)
  A
 / \
B   C
   / \
  D   E
  
   root
   /              右旋(root)
  C
 / \
A   E
/ \
B   D

    C
   / \
  A   root
 / \    \
B   D    E
```

* avl_fix_right也类似

* avl_fix：对平衡二叉树的修复，当执行删除和插入节点时，调用这个函数保持节点的平衡

* avl_del：删除节点

  1、当当前要删除的节点不存在右节点时，如果有左子树，将左子树向上提，如果没有左子树，说明删除的是root，最终都是返回左子树，返回左子树

  2、如果存在右节点，递归的找到右节点当中的最小元素，修改树的结构

* avl_offset：从当前节点出发，按照偏移来进行查找节点

  1、初始化pos为0，表示当前节点相对于起始节点的位置

  2、如果pos等于偏移量，则返回

  3、如果小于偏移量，并且加上右节点的节点数大于offset，则说明在右子树

  4、如果大于偏移量，并且减去左节点的节点数小于offset，说明在左子树

  5、否则在父节点

```C++
// 平衡二叉树
class AVLNode
{
private:
    uint32_t depth = 0;
    uint32_t cnt = 0;
    AVLNode *left = NULL;
    AVLNode *right = NULL;
    AVLNode *parent = NULL;

public:
    AVLNode(AVLNode *node);
    AVLNode();
    ~AVLNode();
    AVLNode *avl_fix(AVLNode *node);
    AVLNode *avl_del(AVLNode *node);
    AVLNode *avl_offset(AVLNode *node, int64_t offset);

    // get 
    uint32_t avl_depth(AVLNode *node);
    uint32_t avl_cnt(AVLNode *node);
    uint32_t max(uint32_t lhs, uint32_t rhs);
    void avl_update(AVLNode *node);
    
    AVLNode *rot_left(AVLNode *node);
    AVLNode *rot_right(AVLNode *node);
    AVLNode *avl_fix_left(AVLNode *root);
    AVLNode *avl_fix_right(AVLNode *root);

private:
};
```



## 六、zset数据结构

zset在Redis中是使用跳表+哈希表来实现的

这里使用AVL树+哈希表来实现

* ZNode

  ZNode中需要存储AVL树节点以及哈希节点，同时存储score是用于排序的键，ZNode中携带的数据可以提供快速查找，也支持有序访问，node同时属于AVL树和哈希表。

```C++
class ZNode {
public:
    AVLNode tree; // AVL树节点
    HNode hmap;   // 哈希表节点
    double score; // 分数
    size_t len;   // 名称长度
    char name[0];

    ZNode(const char* name, size_t len, double score);
    ~ZNode() = default;
};
```

* ZSet

  在ZSet中包含一个tree指向平衡二叉树的根节点，平衡二叉树关联着ZNode中的tree，同时维护一张HMap哈希表，用于对数据的快速查找

|                             接口                             |                    功能                     |
| :----------------------------------------------------------: | :-----------------------------------------: |
|      ZNode *zset_lookup(const char *name, size_t len);       |  根据传入的name来进行查找，从哈希表中查找   |
|  bool zset_add(const char *name, size_t len, double score);  |          向ZSet中添加一个节点元素           |
|        ZNode *zset_pop(const char *name, size_t len);        |    从ZSet中弹出一个元素，按照name来弹出     |
| ZNode *zset_query(double score, const char *name, size_t len); |     查找ZSet中分数以及name都相等的ZNode     |
|      ZNode *znode_offset(ZNode *node, int64_t offset);       |     根据分数的偏移从AVLtree中查找ZNode      |
|                     void zset_dispose();                     | 清空当前的ZSet，包括AVL树的清空和hmap的清空 |

```C++
class ZSet
{
private:
    AVLNode *tree = nullptr;
    HMap hmap;

public:
    ZSet() = default;
    ~ZSet() { zset_dispose(); }

    // helper
    ZNode *znode_new(const char *name, size_t len, double score);
    uint32_t min(size_t lhs, size_t rhs);
    bool zless(AVLNode *lhs, double score, const char *name, size_t len);
    bool zless(AVLNode *lhs, AVLNode *rhs);
    void zset_update(ZNode *node, double score);
    void tree_add(ZNode *node);
    
    // interface
    bool zset_add(const char *name, size_t len, double score);
    // 根据键值来查找哈希表
    ZNode *zset_lookup(const char *name, size_t len);
    ZNode *zset_pop(const char *name, size_t len);
    ZNode *zset_query(double score, const char *name, size_t len);
    ZNode *znode_offset(ZNode *node, int64_t offset);
    void znode_del(ZNode *node);
    void tree_dispose(AVLNode *node);
    void zset_dispose();
};
```

zset_lookup

按照name在ZSet中查找节点

* 直接通过hmap查找

```C++
ZNode *ZSet::zset_lookup(const char *name, size_t len) {
    if(!tree) {
        return nullptr;
    }
    HKey key;
    key.node.hcode = str_hash((uint8_t*)name, len);
    key.name = name;
    key.len = len;
    HNode *found = hmap.lookup(&key.node, &hcmp);
    return found ? container_of(found, ZNode, hmap) : nullptr;
}
```

zset_add

* 先查找set中是否存在有对应的节点，如果有则更新value的值
* 如果没有，则生成一个新的znode并插入到zset中

```C++
bool ZSet::zset_add(const char *name, size_t len, double score) {
    ZNode *node = zset_lookup(name, len);
    if (node) {
        zset_update(node, score);
        return false;
    } else {
        node = znode_new(name, len, score);
        hmap.insert(&node->hmap);
        tree_add(node);
        return true;
    }
}
```

zset_pop

> 通过name在zset数据结构中弹出对应的节点

* 先从哈希表中查找对应的哈希节点
* 通过找到的节点找到节点的Znode
* 在AVL书中删除对应的Znode，并返回当前节点

```C++
ZNode *ZSet::zset_pop(const char *name, size_t len) {
    if(!tree) {
        return nullptr;
    }

    HKey key;
    key.node.hcode = str_hash((uint8_t*)name, len);
    key.name = name;
    key.len = len;
    HNode *found = hmap.lookup(&key.node, &hcmp);
    if(!found) {
        return nullptr;
    }

    ZNode *node = found->owner;
    tree = tree->avl_del(&node->tree);
    return node;
}
```

zset_query

> 通过分数以及name查找的znode

* 先通过分数在AVL树上进行查找，是一个二分查找的过程
* 再通过比较查找得到的节点的name是否相等来对比返回结果

```C++
ZNode *ZSet::zset_query(double score, const char *name, size_t len) {
    AVLNode *found = nullptr;
    AVLNode *cur = tree;
    while(cur) {
        if(zless(cur, score, name, len)) {
            cur = cur->right;
        } else {
            found = cur;
            cur = cur->left;
        }
    }
    return found ? found->owner : nullptr;
}

bool ZSet::zless(AVLNode *lhs, double score, const char *name, size_t len) {
    // 根据指针偏移找到ZNode
    auto zl = lhs->owner;
    if (zl->score != score) {
        return zl->score < score;
    }
    int rv = memcmp(zl->name, name, min(zl->len, len));
    if (rv != 0) {
        return rv < 0;
    }
    return zl->len < len;
}
```

znode_offset

> 根据score的偏移在AVL树中查找Znode。在我们的代码中AVL节点的定义，参考上面的avl_offset文字算法伪代码

```C++
ZNode *ZSet::znode_offset(ZNode *node, int64_t offset) {
    AVLNode *tnode = node ? tree->avl_offset(&node->tree, offset) : nullptr;
    return tnode ? tnode->owner : nullptr;
}
```


## 七、测试效果

**测试哈希表**

单个客户端短连接

![test_hashtable](build-sample-redis/test_01.png)

单个客户端持续连接交互式

> TODO:另一台电脑上代码，对客户端有改进

**测试zset**

单个客户端短连接