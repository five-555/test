---
title: 常见的排序算法
tags: [排序, 算法, 选择排序, 冒泡排序, 插入排序, 归并排序, 快速排序, 堆排序, 计数排序]
categories: 算法研究
date: 2024-02-21 16:53:53
cover:
top_img:
---
## 排序算法

> 输入：整数数组nums
>
> 输出： 按照升序排序

```C++
// 函数接口
vector<int> sortArray(vector<int>& nums) {
    
    return nums;
}
```

### 选择排序

思想：每次选择数组当前数组中最小的元素，放置到数组当前未选定的最前的位置

时间复杂度：n^2

```C++
vector<int> sortArray(vector<int>& nums) {
    int n = nums.size();
    
    for(int i = 0; i < n-1; ++i){
        for(int j = i+1; j < n; ++j){
            if(nums[i] > nums[j]){
                std::swap(nums[i], nums[j]);
            }
        }
    }
    return nums;
}
```



### 冒泡排序

思想：

每次比较相邻的元素，如果前面的元素大于后面的元素，则进行交换

当某一轮没有进行交换时，说明数组已经有序

时间复杂度：n^2

```c++
vector<int> sortArray(vector<int>& nums) {
    int n = nums.size();
    // 冒泡排序
    for(int i = 0; i < n; ++i){
        bool flag = 0;
        // 每一次最大的元素都能够沉到最下面
        for(int j = 0; j < n-i-1; ++j){
            if(nums[j] > nums[j+1]){
                std::swap(nums[j], nums[j+1]);
                flag = 1;
            }
        }
        if(flag == 0)return nums;
    }
    return nums;
}
```



### 插入排序

思想：

从前往后选择元素，插入到前面已经排序好的数组元素当中

始终保证当前元素前半部分都是有序的，直到所有元素遍历完

时间复杂度：n^2

```c++
vector<int> sortArray(vector<int>& nums) {
    int n = nums.size();
    // 插入排序
    for(int i = 1; i < n; ++i){
        for(int j = i-1; j >= 0; --j){
            // 注意比较的是相邻的元素，而不是num[i]
            if(nums[j+1] >= nums[j]){
                break;
            }
            else{
                std::swap(nums[j+1], nums[j]);
            }
        }
    }
    return nums;
}
```



### 归并排序

思想：分治法

将长度为n的数组分为两个长度为n/2的数组

继续分为长度为n/4的数组，最后分为长度为1的数组

分别对长度为1的两两数组进行合并

对长度为2的两两数组进行合并

长度为4的两两数组进行合并

最后对长度为n/2的数组进行合并得到的就是长度为n的有序数组

**因为合并过程中依赖的小序列都是有序的，通过选择最小元素很容易合并**

时间复杂度：nlog(n)

```c++
void merge(vector<int>& nums, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> L(n1), R(n2);
    
    for (int i = 0; i < n1; i++)
        L[i] = nums[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = nums[mid + 1 + j];
    
    int i = 0, j = 0, k = left;
    
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            nums[k] = L[i];
            i++;
        } else {
            nums[k] = R[j];
            j++;
        }
        k++;
    }
    
    while (i < n1) {
        nums[k] = L[i];
        i++;
        k++;
    }
    
    while (j < n2) {
        nums[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(vector<int>& nums, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(nums, left, mid);
        mergeSort(nums, mid + 1, right);
        merge(nums, left, mid, right);
    }
}

vector<int> sortArray(vector<int>& nums) {
    mergeSort(nums, 0, nums.size() - 1);
    return nums;
}
```



### 快速排序

思想：分治法，快排要注意要从right开始

选择一个主元

将小于主元的元素放在左边

将大于主元的元素放在右边

主元的位置则可以确定

分别对主元左边的数组和右边的数组再次进行快速排序

时间复杂度：nlog(n)

```c++
int partion(vector<int>& nums, int left, int right){
    int value = nums[left];
    int idx = left;
    // left += 1; left不用+1，相等的情况已经考虑了
    
    while(left < right){
        while(left < right && nums[right] >= value)right--;
        nums[idx] = nums[right];
        idx = right;
        while(left < right && nums[left] <= value)left++;
        nums[idx] = nums[left];
        idx = left;
    }
    nums[idx] = value;
    return left;
}
    
void quicksort(vector<int>& nums, int left, int right){
    if(left >= right)return;
    
    int mid = partion(nums, left, right);
    quicksort(nums, left, mid - 1);
    quicksort(nums, mid+1, right);
}

vector<int> sortArray(vector<int>& nums) {
    quicksort(nums, 0, nums.size() - 1);
    return nums;
}
```



### 堆排序

思想：

将数组看成一棵完全二叉树，按照数组中的元素建立大顶堆

交换堆顶元素和当前最末端元素，此时最大的元素到了数组尾部，锁定位置

对当前堆进行更新

时间复杂度：nlg(n)

> 建堆时间为lg(n)
>
> 取出元素为1，更新堆为lg(n)

```c++
void heapify(vector<int>& nums, int n, int i) {
    int largest = i;  // 初始化最大值为当前节点
    int left = 2 * i + 1;  // 左孩子节点的索引为 2*i + 1
    int right = 2 * i + 2;  // 右孩子节点的索引为 2*i + 2

    // 如果左孩子节点比当前节点大
    if (left < n && nums[left] > nums[largest])
        largest = left;

    // 如果右孩子节点比当前最大值大
    if (right < n && nums[right] > nums[largest])
        largest = right;

    // 如果最大值不是当前节点
    if (largest != i) {
        // 交换当前节点和最大值节点的值
        swap(nums[i], nums[largest]);

        // 递归地对受影响的子树进行堆化
        heapify(nums, n, largest);
    }
}

void make_heap(vector<int>& nums) {
    int n = nums.size();

    // 构建堆（重新排列数组）
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(nums, n, i);
}

vector<int> sortArray(vector<int>& nums) {
    make_heap(nums);

    // 逐个从堆中提取元素
    for (int i = nums.size() - 1; i > 0; i--) {
        // 将当前根节点移动到末尾
        swap(nums[0], nums[i]);

        // 对减小后的堆进行堆化
        heapify(nums, i, 0);
    }
    return nums;
}

```



### 希尔排序

思想：

使用一定的间隔（数组长度的一半）对数组进行分组，然后对每个分组进行插入排序

随着排序的进行，间隔逐步减小，直到间隔为1，最终完成排序

时间复杂度：n^1.3

设置

```c++
void shellSort(vector<int>& arr) {
    int n = arr.size();

    // 初始化间隔gap为数组长度的一半，然后逐步缩小间隔直至为1
    for (int gap = n / 2; gap > 0; gap /= 2) {
        // 对每个间隔进行插入排序
        for (int i = gap; i < n; i++) {
            int temp = arr[i];
            int j;

            // 将arr[i]插入到正确的位置
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
}
vector<int> sortArray(vector<int>& nums) {
    shellSort(nums);
    return nums;
}
```



### 计数排序

时间复杂度：n+k，k为当前数组中的最大值

> 如果有负数还需要进行另外处理

```c++
std::vector<int> countingSort(std::vector<int>& nums) {
    // 找到数组中的最大值
    int max_num = *std::max_element(nums.begin(), nums.end());

    // 创建计数数组，并初始化为0
    std::vector<int> count(max_num + 1, 0);

    // 统计每个元素出现的次数
    for (int num : nums) {
        count[num]++;
    }

    // 根据计数数组重建排序后的数组
    std::vector<int> sortedArray(nums.size());
    int index = 0;
    for (int i = 0; i <= max_num; ++i) {
        while (count[i] > 0) {
            sortedArray[index++] = i;
            count[i]--;
        }
    }

    return sortedArray;
}
vector<int> sortArray(vector<int>& nums) {
    return countingSort(nums);
}
```

