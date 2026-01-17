---
title: "「UCB CS61A 课程学习阶段性小结一」Python 基础语法"
subtitle: ""
description: ""
slug: 7e6684
date: 2024-08-05T12:24:36+08:00
lastmod: 2024-08-05T12:24:36+08:00
draft: false

resources:
# 文章特色图片
- name: featured-image
  src: featured-img.webp
# 首页预览特色图片
- name: featured-image-preview
  src: featured-img.webp

# 标签
tags: ['Python', '编程基础']
# 分类
categories: ["学习"]
# 合集(如果下面这一行注释掉，就不会显示系列为空了)
collections: ['CS61A 学习记录']
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
## 前言

本篇文章为我在学习 CS61A 课程的第一次阶段性小结，本篇文章的范围就是从 `lab0` 到 `proj1` 这部分内容。

不过在写这篇文章的时候，实际的进度已经到了 `proj2`，再加上作者有一点点的编程基础，所以在小结中主要记录一些我思考的内容、遇到的一些问题和完成作业的的体验。

~~其实就是个人写的作业时候的流水帐罢了~~

<!--more-->
个人做的是 `2024 Fall` 的版本，课程相关资料：

> [!NOTE] 课程资料
> 课程主页: [CS61A](https://cs61a.org/) \
> 课程教材: [composing programs](https://www.composingprograms.com/) \
> 课程教材中文翻译: [https://composingprograms.netlify.app/](https://composingprograms.netlify.app/) \
> `2024 Fall` 课程资料及个人代码: [mulberror/CS61A](https://github.com/mulberror/CS61A)

## 实验 0: Getting Started

该 `lab` 就是给各位学习课程的学生熟悉一下 `hw` 和 `lab` 的编写方式，以及**本地测试**的过程。

在 2023 年左右的时候，因为一些原因让这些公开课的资料需要通过 UCB 本校验证才可以访问，所以非 UCB 的学生并**不能使用全部的测试内容**，以及课程主页只会显示当前这个学期的内容，也就是如果看课程主页的话，只能**跟着当期进度**进行学习。

不过以上这一点不用太过在意，因为课程最重要的 `proj` 部分的测试样例是**十分足够**的，一般 `proj` 的每一个部分都会有至少 100 组左右的数据。

有关于一些代码编写或者代码调试的相关内容可以去详细去看一下[课程主页](https://cs61a.org/lab/lab00/)。

简单介绍一下就是一般的 `hw` 或者 `lab` 的测试数据就是 `***YOU CODE HERE***` 上方的那个部分。

对于 `lab` 来说，部分题目会有未解锁的测试数据，输入以下的命令：

```bash
python3 ok -q <problem_name> -u
```

然后需要通过回答问题，进行完整测试样例的解锁。这些问题主要包括：

- 基础概念（选择题）
- 看程序写结果（大部分题目）

所有的题目完成之后再查看一下本次作业的分数就可以了，输入一下命令查看一下是不是所有的分数都拿满就可以了。

```bash
python3 ok --score
```

## 作业 2: 高阶函数

`hw01`、`lab01` 前面两个部分都比较简单，所以这几个部分写在一起。

不过虽然简单，前面几张的内容都在强调一个概念，就是**面向对象**，感觉也可以叫做**函数式编程**。

比较明显的一点就是在 `hw01` 中，对于 `x + y` 操作，用 `add(x, y)` 函数进行替代。

其实就是在 python 里面任何东西都是对象，不管是一个变量、还是一个函数，这一点就有点像 C 语言中的指针（函数指针、对象指针），但是指针比 python 这个直接操作对象要底层，更重要的是 python 中的内容更加的好用。

### 作业 2. make_repeater

举一个简单的例子，`hw02` 中的 `make_repeater` 这个部分：

传入的是一个可以使用的函数 `f`，以及一个循环的次数 `n`；从 `make_repeater(square, 3)(5)` 这个调用方式可以看出传出的也是一个函数。

正如教材中所说的，“函数其实是一种抽象方法，它描述了与特定参数值无关的的复合操作。”

本道题目比较简单，在后面的**匿名函数**中的一道拓展题，需要深刻理解这么一点。

```python
def make_repeater(f, n):
    def foo(x):
        for i in range(n):
            x = f(x)
        return x
    return foo
```

## 项目 1: 骰子游戏

### 杂谈

不得不说，国外课程的作业和项目这种安排和设置都非常的上心，完整的环境加上文档，和国内的 CS 教育形成了鲜明的对比。以“骰子游戏”这个项目来说，学生需要完成的就是仔细阅读文档，然后补充完成代码，就可以得到一个可以完整运行的一个项目 demo。这种明显可见的国内外差距是需要时间来弥补的。

有一说一，如果我站在一个**真正的编程初学者**，我会觉得这个 proj 还是有一定难度的。在编写的时候一定要先阅读好文档，回答好设定的问题，然后再进行代码的编写，看清每一个接口的输入和输出分别是什么。

本项目中还需要学习者要养成代码复用的习惯，不管是在**减少代码量**还是在**提高代码可读性方面**上。

### 项目 1. 第 1 阶段: 游戏规则

这一部分主要是一些模拟问题，看清楚给定的问题是什么，比如输出量是**分数的该变量**还是**分数的最终值**。

稍微提一下这个阶段的最后一个部分 `play`，该部分就是给两个玩家的策略 `strategy`、分数更新方式 `update`、骰子 `dice`，然后让你模拟一下整个游戏。

本身肯定是一个判断两个人都没有达到的目标分数的 while 循环，内层就是两个玩家通过自己的决策，轮流进行分数的更新。

轮流有一个简单的小技巧，用一个 `who` 变量表示当前是谁来操作，`who` 就是在进行 `0/1` 变换，这就可以使用异或操作。

```python
def play(strategy0, strategy1, update,
         score0=0, score1=0, dice=six_sided, goal=GOAL):
    who = 0  # Who is about to take a turn, 0 (first) or 1 (second)
    # BEGIN PROBLEM 5
    "*** YOUR CODE HERE ***"
    while score0 < goal and score1 < goal:
        if who == 0:
            score0 = update(strategy0(score0, score1), score0, score1, dice)
        else:
            score1 = update(strategy1(score1, score0), score1, score0, dice)
        who = 1 - who
        # who ^= 1
    # END PROBLEM 5
    return score0, score1
```

### 项目 1. 第 2 阶段: 游戏策略

Problem 7 是一个编写检查策略在所有情况下是否会选择投掷相同个数的骰子。

其实后面的几个任务我觉得存在一定的问题，后面的问题基本都是进行很多次的投掷，然后计算其平均值，然后取最大的投掷数量。但是选择使用真正的骰子进行模拟，会不会影响实际的投掷结果？

如果除去问题，后面几个问题还是有一定难度的，主要函数作为一个实体也在程序中进行传递，比较突出锻炼个人的抽象能力。

如果有问题的话，具体代码的实现可以在我的 github 仓库中查看，如果有更加优秀的写法，欢迎和我进行交流。
