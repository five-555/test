---
title: Git常见用法
date: 2024年1月4日
tags: git
categories: 技术研究
top_img: image-20231204110955462.png
---

#### 怎样撤销一个已经push到远端的版本

```
每次push之前线pull一下

1、查看当前提交的信息，找到需要撤回到的版本号复制，一串十六进制的数
git log

2、使用git reset
git reset --soft 复制的版本号

3、强制回退当前版本号
// 确认一下当前版本
git log
// 谨慎使用，强制使用本地仓库代码修改远程仓库
git push orgin master --force
```

#### 新建分支并同步到远端的分支

```
# 在本地新建一个名字为branch_name的分支，并与远端的origin/branch_name同步
git checkout -b branch_name origin/branch_name
```

#### 解决git clone超时的问题

从github上clone代码仓库报错`Failed to connect to github.com port 443 after 21038 ms: Couldn't connect to server`且尝试去ping一下github官网会丢包

![image-20231204110955462](git/image-20231204110955462.png)

解决方案

修改系统的hosts，跳过域名解析的过程，直接用ip地址访问

```
192.30.255.112 github.com git
185.31.16.184 github.global.ssl.fastly.net
```

![image-20231204111125551](git/image-20231204111125551.png)

修改hosts需要给文件更高的权限

![image-20231204111350240](git/image-20231204111350240.png)

#### github中git push出现超时的问题

![image-20231204112945547](git/image-20231204112945547.png)

解决方案

1、打开本机的代理服务器

![image-20231204113100512](git/image-20231204113100512.png)

2、取消git config里面的http和https代理

![image-20231204113208566](git/image-20231204113208566.png)

3、设置http代理服务器

![image-20231204113304822](git/image-20231204113304822.png)

#### linux中输出一个文件夹下面的所有文件名

- **`/path/to/directory`**: 替换为目标目录的路径。
- **`-maxdepth 1`**: 限制`find`的搜索深度为1，即仅在指定的目录中搜索，而不会搜索其子目录。
- **`-type f`**: 限制搜索结果为普通文件（不包括目录和其他类型的文件）。
- **`-exec basename {} \;`**: 对每一个找到的文件执行`basename`命令，即输出文件的基本名称。`{}`是`find`命令的占位符，表示每个找到的文件的路径。`\;`表示命令结束。

```
find /path/to/directory -maxdepth 1 -type f -exec basename {} \;
```

#### git查看远端仓库地址

```
git remote -v

# 更改远程仓库
git remote set-url origin 仓库地址
```

#### 查看代码贡献量

> 按照各个作者的修改代码总数排序

```
git log --pretty="%aN" | sort | uniq -c | while read count author; do echo -n "$author "; git log --author="$author" --pretty=tformat: --numstat | awk '{ add += $1; subs += $2 } END { total = add + subs; printf "%d\n", total }'; done | sort -rnk2
```

