---
title: "「UCB CS61A 课程学习阶段性小结四」链表"
subtitle: ""
description: ""
slug: f5c5a0
date: 2024-11-27T15:55:10+08:00
lastmod: 2024-11-27T15:55:10+08:00
draft: false

resources:
# 文章特色图片 
- name: featured-image
  src: featured-img.webp
# 首页预览特色图片
- name: featured-image-preview
  src: featured-img.webp

# 标签
tags: [
  'Python',
  '面向对象编程',
  '链表'
]
# 分类
categories: ["学习"]
# 合集(如果下面这一行注释掉，就不会显示系列为空了)
collections: ["CS61A 学习记录"]
# 从主页面中去除
hiddenFromHomePage: false
# 从搜索中去除
hiddenFromSearch: false

lightgallery: false

# 否开启表格排序
table:
  sort: false
toc:
  enable: true
  auto: true
expirationReminder:
  enable: false
  # ...
code:
  copy: true
  # ...
edit:
  enable: false
  # ...
math:
  enable: true
  # ...
mapbox:
  accessToken: ""
  # ...
share:
  enable: true
  # ...
comment:
  enable: true
  # ...
library:
  css:
    # someCSS = "some.css"
    # 位于 "assets/"
    # 或者
    # someCSS = "https://cdn.example.com/some.css"
  js:
    # someJS = "some.js"
    # 位于 "assets/"
    # 或者
    # someJS = "https://cdn.example.com/some.js"
seo:
  images: []
  # ...
---

由于后面关于 Scheme 的相关内容比较多，并且最后的项目 Scheme 是使用 Python 写一个 Scheme 语言的解释器。

受限于篇幅原因，还是决定把有关 Scheme 的内容单独写成一片[文章](https://blog.mulbry.top/posts/2024/11/f38a77/)，本篇也会比较短。

本篇文章的范围是从 `lab06` 到 `lab08` 和 `hw06` 的内容。

## 实验 6: 面向对象编程

本实验是面向对象部分的实验课，这个部分需要结合一下之前[数据抽象](https://blog.mulbry.top/posts/2024/11/5a1140/#%e5%ae%9e%e9%aa%8c-4-%e6%a0%91%e9%80%92%e5%bd%92%e6%95%b0%e6%8d%ae%e6%8a%bd%e8%b1%a1)的内容。

三个问题还是比较简单，像第二个问题是模拟了一个邮件系统，邮件系统创建了客户端模块，邮件类和邮件服务器模块。

第三个问题中需要对**类内的属性**和**类中的实例**进行操作，比如说在以下代码中

```python
class A:
    x = 0
    def __init__(self, y):
        self.y = y
        A.x += y
```

`x` 是类 `A` 的**类属性**，`y` 是类 `A` 的**实例属性**。

- `x` 是属于类 `A` 的，而不是某个特定的实例。所有 `A` 类的实例共享这个变量。
- `y` 是属于类 `A` 的实例属性，每一个实例可以有不同的 `y` 值。

搞清楚以上的内容后，结合三道题目给的具体场景以及文档中的说明，就可以很容易地完成这个实验了。

## 作业 6: OOP 和链表

本次作业重点应该是链表相关的操作，因为如果链表不熟悉的话，后面关于 Scheme 的内容会非常吃力。

如问题 3 中的存储数位，如果采用链表和递归的方式完成这道题，就需要注意一下链表插入的顺序和递归获取到数字的顺序。

```python
def store_digits(n):
    "*** YOUR CODE HERE ***"
    def foo(n, result=None):
        if result is not None:
            result = Link(n % 10, result)
        else:
            result = Link(n % 10)
        
        if n < 10:
            return result
        else:
            return foo(n // 10, result)
    return foo(n)
```

如问题 3、问题 4 和问题 5，和之前对列表的操作也是类似的。

```python
def two_list(vals, counts):
    if len(counts) == 0:
        return Link.empty
    counts[0] -= 1
    if counts[0] == 0:
        return Link(vals[0], two_list(vals[1:], counts[1:]))
    else:
        return Link(vals[0], two_list(vals, counts))
```

## 实验 7: 继承

继承，将类之间的关系看成一棵树，与其他的高级编程语言类似，Python 继承可以通过 `class A(B):` 的方式实现。

Python 中将属性分为类属性和实例属性，在继承的过程中，子类可以继承父类的类属性，但是不能继承父类的实例属性。还有一点需要注意的是，子类可以重写父类的方法。

在 `proj ants` 项目中，`Ant` 类是所有蚂蚁的父类，`HarvesterAnt` 和 `ThrowerAnt` 是 `Ant` 的子类。

这就是在很大程度上减少了代码的重复性，提高了代码的可维护性。

该实验中依旧链表的内容，像问题 4 对链表的操作就非常接近之后 Scheme 实验中链表的操作。

像问题 6 也需要注意一下添加一个链表中的节点后，下一个需要访问的节点是 `current.rest` 还是 `current.rest.rest`。

```python
def duplicate_link(s, val):
    if s is Link.empty:
        return
    if s.first == val:
        s.rest = Link(val, s.rest)
        duplicate_link(s.rest.rest, val)
    else:
        duplicate_link(s.rest, val)
```

## 实验 8: Mutable Trees

实验 8 的内容比较简单，目的就是熟悉一下树的结构。

### 实验 8. 问题 5: Maximum Path Sum

这个问题是求树中从根节点到叶子节点的最大路径和。

函数返回子树中的最大路径和。

假设要求解以 $r$ 为根的最大路径，该路径就是 $r$ 子树中最大的路径再加上 $r$ 上标签上的值。

```python
def max_path_sum(t):
    """Return the maximum path sum of the tree.

    >>> t = Tree(1, [Tree(5, [Tree(1), Tree(3)]), Tree(10)])
    >>> max_path_sum(t)
    11
    """
    "*** YOUR CODE HERE ***"
    if t.is_leaf():
        return t.label
    else:
        max_path = 0
        for b in t.branches:
            max_path = max(max_path, max_path_sum(b))
        return t.label + max_path
```
