---
title: "「UCB CS61A 课程学习阶段性小结二」递归与链表"
subtitle: ""
description: ""
slug: fae47f
date: 2024-08-23T08:25:42+08:00
lastmod: 2024-08-23T08:25:42+08:00
draft: false

resources:
# 文章特色图片
- name: featured-image
  src: featured-img.webp
# 首页预览特色图片
- name: featured-image-preview
  src: featured-img.webp

# 标签
tags: ['Python', '编程基础', 'Lambda 表达式', '搜索']
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
本篇文章主要记录一下从 `lab2` 到 `proj cats` 中的内容。

感觉该部分的课程内容主要还是在进行 Python 语言的教学，不过添加了更多的进阶的技巧，像高阶函数、`lambda` 函数等。

<!--more-->

## 实验 2: 高阶函数与 Lambda 函数

个人感觉高阶函数其实就是进行函数的嵌套，和前面的课程所强调的相同，将函数看成一个对象进行传递并修改。

这其实也就是 Python 作为一个函数式编程语言的特性之一，对函数的处理非常灵活。

虽然处理可以非常的灵活，但是这也需要保证代码的可读性和严格性，比如在调用函数时，带不带括号就代表了完全不同的意义。

如实验开篇示例的代码：

```python
>>> def cake():
...    print('beets')
...    def pie():
...        print('sweets')
...        return 'cake'
...    return pie
>>> chocolate = cake()
______

>>> chocolate
______

>>> chocolate()
```

这里的 `chocolate` 是一个函数名，也就是该函数这个对象，而 `chocolate()` 则调用了该函数，返回了内部处理的 `pie` 函数。

### 实验 2. 问题 8: 函数循环嵌套

以该问题进一步理解高阶函数的概念。

定义一个函数 `cycle`，该函数将三个函数 `f1`、`f2` 和 `f3` 作为参数。`cycle` 将返回另一个函数 `g`，该函数应接受整数参数 `n` 并返回另一个函数 `h`。最后一个函数 `h` 应该接受一个参数 `x`，并根据 `n` 是什么，将 `f1`、`f2` 和 `f3` 应用于 `x`。下面是最后一个函数 `h` 应该对 `x` 对 `n` 的几个值执行的操作。

本问题并不困难，甚至可以拓展到给定一个函数的列表，然后根据列表的长度来进行循环嵌套，这就是一个典型的将函数看作对象的例子。

```python
def cycle(f1, f2, f3):
    f = [f1, f2, f3]
    def g(n):
        def cycle(x):
            for i in range(n):
                x = f[i % 3](x)
            return x
        return cycle
    return g
```

### Lambda 函数

Lambda 函数（匿名函数），在很多的编程语言当中都有这个特性，Python 也不例外。

这种函数没有函数名，只有函数体，如果用一个对象来承接了这个函数体，那么在使用该对象的时候也就相当于调用了该函数体，当然也可以将 Lambda 函数作为一个结果传出来。针对一些需要复用的简单代码可以简化代码量，但是也会降低代码的可读性。

## 作业 3. 递归、树递归

本节作业主要是进行递归的练习，递归简单的讲就是函数调用自己的过程。部分的作业是将之前练习循环的题目转换成要求使用递归实现。

递归和循环相同，有边界条件、如何寻找下一个状态诸类问题需要考虑。不过递归比较方便考虑更加复杂的情况，因为循环寻找下一个状态的方式一般比较单一，而递归可以进行搜索的操作。

### 作业 3. 问题 4: 数美元

以该问题为例，本道题目的意思为给定以下 4 种美元的面值，问有多少种选择面值的方式可以得到 `total` 元的方式。

因为需要找到多少种选择的方式，所以并不需要考虑顺序，所以可以考虑按照递增或者递减的顺序选择面值，刚好对应了给出的 `next_larger_coin` 和 `next_smaller_coin`。

```python
def next_larger_coin(coin):
    if coin == 1:
        return 5
    elif coin == 5:
        return 10
    elif coin == 10:
        return 25

def next_smaller_coin(coin):
    if coin == 25:
        return 10
    elif coin == 10:
        return 5
    elif coin == 5:
        return 1
```

可以考虑选择构造一个递减的序列，这样在递归的过程中记录一下上一次选择的面值 `last`，然后当次从 `last` 开始往小的进行选择即可。

题目还要求程序实现中不能出现任何的 for 循环，所以在递归的内部还需要用递归模拟一个循环。

```python
def count_coins(total):
    ans = 0
    def dfs(n, last):
        nonlocal ans
        if n == 0:
            ans = ans + 1
            return

        def g(x):
            if n - x >= 0:
                dfs(n - x, x)
            if x == 1:
                return
            g(next_smaller_coin(x))
        
        g(last)
    
    dfs(total, 25)
    return ans
```

### 作业 3. 问题 7: 匿名阶乘

这道题目我觉得是非常有难度的，本身 Lambda 表达式受限于其不存在函数名，因此非常不好处理递归这个问题。

没有函数名，那么程序就没有办法在内部对自身进行调用。

如果学过 C++ 11 以上并且了解过 C++ 中用 Lambda 表达式进行递归，这一部分应该会比较好理解。

在上文中写过，Lambda 函数可以用一个对象进行承接，在对该对象进行访问的时候相当于对该函数体进行了调用。

所以比较正确的方式就是将函数体传入到编写的递归中，然后在递归中调用该函数体。

```python
from operator import sub, mul

def make_anonymous_factorial():
    return lambda x : (lambda f, x : 1 if x == 1 else mul(x, f(f, sub(x, 1))))(lambda f, x : 1 if x == 1 else mul(x, f(f, sub(x, 1))), x)
```

首先 `lambda x` 返回的是参数为数字的函数，该函数内调用了一个 `lambda f, x` 这样的 Lambda 函数，其中的 `x` 的含义是递归的参数，`f` 则是递归的函数体，每次递归将该 `f` 调用并传给写一个需要用的函数，这样就实现了递归的效果。

## 实验 3: 递归、列表

实验 3 的内容相对比较简单，该部分内容对应 [2.3 序列](https://composingprograms.netlify.app/2/3) 中的内容。

可以重点关注一下列表的**一些操作**、**切片**、**复制**这种概念（这一点在循环中删除列表中的内容有很大的帮助）。

还有关于列表推导式的内容，在列表初始化的时候可以简化代码。

另外一个部分（两道题目）还是关于递归的内容，比较简单。

## 项目 2: 小猫打字

[小猫打字](https://cs61a.org/proj/cats/)是本课程的第二个大型项目，也是对之前所学的内容进行综合性的练习，主要是对 Python 语言的综合性练习，写起来感觉非常好。

### 项目 2. 第 1 阶段

第一阶段是补充完整项目的一些基础功能，大部分的实验内容都在文档中有详细的解释，按照文档中的提示一步一步进行编写即可。

### 项目 2. 第 2 阶段

第二阶段是需要实现一个自动更正的功能，这部分重点考察一下**递归**等内容。

在实验的过程中还加入一个优化的要素，也就是锁定了最大的差异值 `limit`，在递归搜索差异值时锁定了递归树的深度，由于此处的搜索的复杂度为指数级的，这样可以大大减少搜索的时间。

### 项目 2. 问题 7: 最小编辑距离

例如[问题 7](https://cs61a.org/proj/cats/#problem-7-3-pts)，需要找到 `typed` 和 `source` 两个字符串最少需要多少次操作才能变成相同的字符串，这里的操作有三种，分别是**插入、删除、替换**。

搜索的思路是每次考虑两个字符串的第一个字符应该如何操作，一次操作肯定要使**至少两个字符串中一个首字母匹配**。

因为插入的字符是任意的，所以给 `typed` 字符串的开头加一个字母其实就相当于将 `source` 字符串的开头删除一个字母，另外两个操作也类似。

所以可以看成只对 `typed` 字符串进行操作：

- 插入: `source` 字符串的开头删除一个字符。
- 删除: `typed` 字符串的开头删除一个字符。  
- 替换: 如果两个字符串的首字符相同，那么考虑后面的字符串，否则就需要一次操作将第一个字符修改相同，然后考虑后面的字符串。

因为题目还给定了一个 `limit` 锁定了答案，所以在递归的过程中可以加入一些优化，比如两个字符串中较长的超过了 `limit`，那么就可以直接返回 `limit + 1`。

```python
@memo_diff
def minimum_mewtations(typed, source, limit):
    # assert False, 'Remove this line'
    if typed == source: # Base cases should go here, you may add more base cases as needed.
        # BEGIN
        "*** YOUR CODE HERE ***"
        return 0
        # END
    # Recursive cases should go below here
    if min(len(typed), len(source)) > limit:
        return limit + 1
    if limit < 0: # Feel free to remove or add additional cases
        # BEGIN
        "*** YOUR CODE HERE ***"
        return 0
        # END
    if len(typed) == 0 or len(source) == 0:
        return len(source) + len(typed)
    if typed[0] == source[0]:
        return minimum_mewtations(typed[1:], source[1:], limit)
    
    ans = limit + 1
    if min(len(typed), len(source) - 1) < limit:
        ans = min(ans, minimum_mewtations(typed, source[1:], limit - 1) + 1)
    if min(len(typed) - 1, len(source)) < limit:
        ans = min(ans, minimum_mewtations(typed[1:], source, limit - 1) + 1)
    if min(len(typed), len(source)) - 1 < limit:
        ans = min(ans, minimum_mewtations(typed[1:], source[1:], limit - 1) + (0 if typed[0] == source[0] else 1))
    return ans
```

在函数前有一个 `@memo_diff` 的装饰器，该部分是[第 4 阶段: 额外挑战](https://cs61a.org/proj/cats/#phase-4-efficiency-extra-challenge)的内容，这个装饰器是用来进行递归的优化，将递归的结果进行记忆化，这样可以避免重复的递归计算。

装饰器可以简单的理解为在函数调用前后进行一些操作，此处因为有 `limit` 这个参数限定递归的深度，就可以针对该 `limit` 进行优化。

在上述的递归过程中，如果答案超过了 `limit`，那么就可以直接返回 `limit + 1`，递归只搜索完了较浅的答案，并没有搜索深度大于等于 `limit + 1` 的答案。

所以可以将 `limit', source, typed` 记录下来，如果遇到了相同的 `source, typed` 并且 $limit' \geq limit$，那么就可以直接返回 `limit + 1`，否则再进行搜索，然后更新 `limit' = limit`。

```python
def memo_diff(diff_function):
    """A memoization function."""
    cache = {}

    def memoized(typed, source, limit):
        # BEGIN PROBLEM EC
        "*** YOUR CODE HERE ***"
        if (typed, source) not in cache:
            value = diff_function(typed, source, limit)
            cache[(typed, source)] = (value, limit)
            return value
        else:
            val, lim = cache[(typed, source)]
            if limit <= lim:
                return val
            else:
                new_val = diff_function(typed, source, limit)
                cache[(typed, source)] = (new_val, limit)
                return new_val
        # END PROBLEM EC

    return memoized
```
