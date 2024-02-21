---
title: 常见的排序算法
tags:
  - 排序 算法 选择排序 冒泡排序 插入排序 归并排序 快速排序 堆排序 计数排序
categories: 算法研究
date: 2024-02-21 16:53:53
cover:
top_img:
---
### 

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

### 计数排序

