---
title: "「UCB CS61A 课程学习阶段性小结三」面向对象编程"
subtitle: ""
description: ""
slug: 5a1140
date: 2024-11-24T14:56:29+08:00
lastmod: 2024-11-24T14:56:29+08:00
draft: false

resources:
# 文章特色图片
- name: featured-image
  src: featured-img.webp
# 首页预览特色图片
- name: featured-image-preview
  src: featured-img.webp

# 标签
tags: ['Python', '编程基础', '数据抽象', '搜索', '游戏开发']
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

本篇文章的范围是从 `lab04` 到 `proj 3 ant` 的内容，该部分的内容需要更加对面向对象编程的概念有一定了解。

还有就是关于指针的概念，比如说 `lab05` 和 `hw05` 的内容就是关于指针的。

该部分的项目 `proj 3 ant` 的拓展问题稍微有点问题，我的代码可以通过测试，但是在实际使用的时候会出现卡死的情况。

<!--more-->

## 实验 4: 树递归，数据抽象

本次实验内容首先是 Python 中的字典，也就是 C++ 中的 `map`。

Python 中的字典内部是一种数据的映射，将键映射到值，如果是第一次接触到这种数据结构的话，可以看成一个下标类型比较自由的列表。

该实验的第二个问题就是把字典看成列表，使用列表推导的方式生成一个字典。

### 实验 4. 问题 3: 购买水果

本道题目是将字典和递归搜索进行结果，题意需要找到所有购买水果的方案，通过题目需要输出所有的方案就可以看出需要使用搜索。

每一个水果都至少需要买一个，所以假如说当前搜索到了第 i 个水果，就枚举该水果购买多少，然后递归搜索下一个水果。

```python
def buy(fruits_to_buy, prices, total_amount):
    def add(fruits, amount, cart):
        if fruits == [] and amount == 0:
            print(cart)
        elif fruits and amount > 0:
            fruit = fruits[0]
            price = prices[fruit]
            # print('DEBUG', fruit, price)
            for k in range(1, amount // price + 1):
                # Hint: The display function will help you add fruit to the cart.
                add(fruits[1:], amount - k * price, cart + display(fruit, k))
    add(fruits_to_buy, total_amount, '')
```

### 实验 4. 数据抽象

该部分对应教科书中的 [数据抽象](https://composingprograms.netlify.app/2/2) 部分的内容。

教科书上写的有点复杂，根据个人的看法，其实就是把复合数据更加的模块化，每一个部分都被一个独立的程序所控制。

感觉有点像类和调用类中元素的接口的感觉。

比如说 `lab04` 中用一个列表作为一个复合数据表示一个城市的信息，然后每一个城市的信息都需要进行合法性的判断，调用的时候也需要调用一个函数来进行取出。

在其他的编程语言中，例如 C++ 中，这种数据抽象的方式就像类的概念。（Python 非常强调数据抽象，也就是模块化的概念）

## 作业 4: 序列、数据抽象、树

### 作业 4. 问题 2: Deep Map

本道题目需要实现一个函数，该函数可以对一个嵌套的列表进行操作，对于列表中的每一个元素都进行操作。

这里就牵扯到一个关于对于列表的修改，其实在上文也提到过，函数的参数中的列表其实就是原本列表，也是一个指针，在函数内部对列表进行的修改会影响到原本的列表。

如果需要在迭代的过程中**删除**一个元素，`for i in list(s):` 或者 `for i in s[:]` 这样的写法复制了一个列表，在删除元素时不会导致指针错误，这样是比较好的写法。

回到正题，本道题目首先需要对当前列表中的元素进行判断，如果元素依旧为列表，那么就递归处理，否则就对元素进行修改。

```python
def deep_map(f, s):
    for i in range(len(s)):
        if type(s[i]) == list:
            deep_map(f, s[i])
        else:
            s[i] = f(s[i])
```

### 作业 4. 问题 4: 平衡

本题目是练习数据抽象相关的内容，平衡就是要求每一个中点的两端的**力矩**相等。

所以对于每一个中间的支点首先需要判断是否为叶子，如果不是就要判断两端返回的**重量和距离的乘积**是否相同。

```python
def balanced(m):
    if is_planet(m):
        return True
    else:
        return balanced(end(left(m))) and 
               balanced(end(right(m))) and 
               total_mass(end(left(m))) * length(left(m)) == total_mass(end(right(m))) * length(right(m))
```

## 实验 5: 可变性、迭代器

Python 属于弱类型语言，Python 中的一些对象是可变的，内容和类型都是可以变化的。

可变性的内容对应教材中的 [2.4 可变数据](https://composingprograms.netlify.app/2/4) 部分的内容，该部分的教材并没有细看，不清楚讲的怎么样。

### 实验 5. 问题 3: 分组

编写一个函数，该函数采用列表 `s` 和函数 `fn`，并返回一个字典，该字典根据应用 `fn` 的结果对 `s` 的元素进行分组。

对于列表中的每一个元素通过 `fn` 函数计算出对应的键值，然后需要针对该键值进行分组。如果没有这个组，就创建一个新的组。

```python
def group_by(s, fn):
    grouped = {}
    for ele in s:
        key = fn(ele)
        if key in grouped:
            grouped[key].append(ele)
        else:
            grouped[key] = [ele]
    return grouped
```

### 实验 5. 迭代器

在 Python 中迭代器也是一个对象，可以通过 `iter()` 函数来获取一个迭代器对象。

像数组的遍历既可以使用循环遍历，也可以使用迭代器进行遍历，假如说遍历一个数组 `a`，可以使用 `i = next(t)` 或者 `for i in iter(a):`。

不过迭代器的使用是有限制的，一旦迭代器遍历完了，再次调用 `next()` 函数就会抛出 `StopIteration` 异常。就算是在函数中使用迭代器，到主程序中也是一样的。

在本实验中的可选问题中可以使用切片来简化代码。

> [!NOTE]+ **切片简单介绍**
> 比如说给一个 `list s`，可以使用 `s[::2]` 来获取所有的偶数索引的元素，切片遵循 `[start:stop:step]` 的规则。\
> 其中 `start` 是切片的起始位置（包含），`stop` 是切片的结束位置（不包含），`step` 是切片的步长。\
> `start` 默认值是 0，即从头开始；`stop` 默认值是列表的长度，即到列表的末尾；`step` 默认值是 1，即每次取一个元素。\
> 比如说 `s[::-1]` 就是将列表倒序。\

## 作业 5: 生成器

作业 5 的内容需要保证迭代器有一定了解，一般的对于列表的迭代器是获得对应迭代器上列表的值，那么如果需要获得是函数的值或者说是需要一个**无限长的列表**中的值，那么就需要用到生成器。

生成器主要是通过 `yield` 关键字来实现，`yield` 关键字可以将函数变成一个生成器，每次调用生成器的时候，就会执行到 `yield` 关键字的地方，然后返回对应的值。

像第三题楼梯的问题，每次可以走 1 格或者 2 格，要求每次调用 `next(it)` 返回下一个走楼梯的方案。

将 `yield` 看成是一个 `return`，这个值通过递归的方式一层一层的返回。

```python
def stair_ways(n):
    if n == 0:
        yield []

    for i in [1, 2]:
        if n - i >= 0:
            for way in stair_ways(n - i):
                yield [i] + way
```

`yield` 后面还可以跟 `from`，这样就可以将一个**生成器**的值传递给另一个**生成器**，例如第一道题目。

这道题目在循环的时候其实是做过的，现在修改了两个部分：首先，使用生成器的方式；其次，原本的列表是有限长度的，现在到 1 的时候一直返回 1。

```python
def hailstone(n):
    if n == 1:
        while True:
            yield 1
    else:
        yield n
        if n % 2 == 0:
            yield from hailstone(n // 2)
        else:
            yield from hailstone(3 * n + 1)
```

上面这个 `hailstone` 函数就是一个生成器，`yield from` 接受该生成器的值，然后返回给调用者。如果 `n = 1`，那么就会一直返回 1。

## 项目 3. 蚂蚁大战蜜蜂

{{< figure src="/img/splash.webp" title="项目的插图（还是非常好看的）" >}}

这个项目非常像以前玩的植物大战僵尸，项目质量非常高，本身对于初学者来说还是比较有难度的，感觉也培养了一定的游戏开发的能力。

首先是在进行项目架构的设计，项目中有很多的类，每一个类都有自己的属性和方法，这样就可以很好的对项目进行管理，此处用到继承的概念。

场上的每一个对象都是对应了一个类的实例，蜜蜂和蚂蚁发动行动都需要在 `gamestate` 的统一管理下，实现了简单游戏场景的设计。

场上的每一个地板（也就是放置虫子或者虫子经过的地方）都是一个 `Place` 类的实例，和虫子之间是一个双向的关系。

**非常推荐在写这个项目之前，仔细阅读一下[项目的架构图](https://cs61a.org/proj/ants/diagram/ants_diagram.pdf)**，如果只想最低限度完成项目也要把基本的类之间的关系搞清楚。

- 项目的阶段 1 和之前的两个项目一样，实现一些关于游戏基础玩法的内容，就比如蚂蚁进行攻击的方式，消灭一只蜜蜂需要在地板上也对蜜蜂的对象进行删除。

- 项目的阶段 2 是在原本只有产阳光的蚂蚁和普通攻击的蚂蚁的基础上，增加一些新的蚂蚁，比如说能够反伤的蚂蚁、城墙蚂蚁等等，这些都是在类的继承上进行修改的。

- 项目的阶段 3 是进行了地板类型的扩充以及一种新的蚂蚁——蚁后。地板类型扩充了水面地形，普通的蚂蚁是无法在上面生存的，所以在虫子属性上要加一个**防水性**的判断，当虫子放置到地板上时，如果该虫子并不具备防水性，那么就直接失去全部生命。蚁后能够为我方的蚂蚁进行增幅，这个也需要对蚂蚁的属性进行增添，然后在**蚁后的攻击方式**上进行修改，额外的击败蚁后也是一个**额外的失败条件**。

还有一点，在对于 `list ls` 的删除的时候，如果是使用遍历方式进行删除，需要创建一个切片或者复制，也就是使用 `ls[:]` 切片或者创建复制 `list(ls)`，因为直接在原先的列表中进行删除会让指针错误。（`ants` 中的 `reduce_health` 这一个函数会引起昆虫的删除，所以在需要注意这一点）这一点在之前的实验和作业中其实也出现过，这个项目中也需要注意。


{{< admonition todo >}}
  <!-- TODO -->
将该项目的额外问题的解答补充上\
第一个额外问题的条件个人感觉有些奇怪，需要在函数内部把另外一个类内的函数给修改掉，我的代码虽然能够通过测试，但是游戏运行起来后这个蚂蚁会出现奇怪的问题。
以下是我的第一版解答代码

```python
# exist bug
class SlowThrower(ThrowerAnt):
    """ThrowerAnt that causes Slow on Bees."""

    name = 'Slow'
    food_cost = 6
    # BEGIN Problem EC 1
    implemented = False   # Change to True to view in the GUI
    # END Problem EC 1

    def throw_at(self, target):
        # BEGIN Problem EC 1
        "*** YOUR CODE HERE ***"
        if not target.action_flag:
            target.past_action = target.action
            target.flag = True
            
        def new_action(gamestate):
            if target.slow_time == 0 or gamestate.time % 2 == 0:
                target.past_action(gamestate)
            if target.slow_time > 0:
                target.slow_time -= 1
        
        target.action = new_action
        target.slow_time = 5   
```

{{< /admonition >}}