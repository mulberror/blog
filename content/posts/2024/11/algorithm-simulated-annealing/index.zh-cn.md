---
title: "模拟退火算法"
subtitle: ""
description: ""
slug: a1a745
date: 2024-11-21T17:01:23+08:00
lastmod: 2024-11-21T17:01:23+08:00
draft: false

resources:
# 文章特色图片
- name: featured-image
  src: featured-img.webp
# 首页预览特色图片
- name: featured-image-preview
  src: featured-img.webp

# 标签
tags: ['随机化算法','模拟退火']
# 分类
categories: ["算法"]
# 合集(如果下面这一行注释掉，就不会显示系列为空了)
collections: ["算法竞赛"]
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

模拟退火算法是一种求解多峰函数全局最值的算法，本质上是使用随机化对爬山算法的改进。

## 爬山算法

爬山算法是一种求解极值的算法，顾名思义，从一个起始点 $x$ 出发，进行迭代计算出一个新的点 $f(x)$，如果 $f(x)$ 比 $x$ 更优，那么就移动到 $f(x)$，否则停止迭代。

显然，爬山算法因为其**只要找不到更优解直接停止迭代**的思路，十分容易陷入局部最优解当中。

那么模拟退火就是在爬山算法的基础上，使用随机化的思路，以**一定的概率**接受一个比当前解更差的解，从而避免陷入局部最优解。

## 模拟退火

主要思路：假设当前的状态为 $S$，一个新状态为 $S'$，如果 $S'$ 比 $S$ 更优，那么接受 $S'$，否则以概率 $P$ 接受 $S'$。

该接受概率 $P$ 也有一定限制：

- 迭代前期的接受概率较高，以便于在搜索空间中更广泛的搜索；迭代后期的接受概率较低，防止算法舍弃了全局最优解。模拟退火算法中，这个迭代次数就是用“温度$T$”这样的概念进行控制的。随着迭代的次数增加，温度 $T$ 逐渐降低。
- 令 $\Delta E(\Delta E\geq 0)$ 表示两个状态的能量差，量化了 $S'$ 比 $S$ 更差的程度。$\Delta E$ 越大，接受概率越低。也就是如果差太多了，也不选择接受。

基于以上两条限制，模拟退火算法中常用的接受概率公式来源于 [Metropolis-Hastings](https://www.wikiwand.com/en/articles/Metropolis%E2%80%93Hastings_algorithm) 算法：

$$
P(S, S', T) = exp(-\frac{\Delta E}T)
$$

{{<figure src="/img/simulated-annealing.gif" title="Wikipedia 模拟退火可视化过程" width="90%">}}

上图中可以看到，随着时间的推移，接受更劣解的频率越来越低，逐渐收敛在全局最优解附近，就像温度高的金属逐渐降温的过程，这就是“退火”这个名字的由来。

## 算法流程

设定初始状态 $S$，初始温度 $T$，终止温度 $T_{end}$，降温速率 $\alpha$。

进入迭代过程：

- 产生一个新状态 $S'$；
- 根据初始状态 $S$ 和新状态 $S'$ 计算能量差 $\Delta E$；
- 如果新状态 $S'$ 比当前状态 $S$ 更优，则直接接受 $S'$；
- 否则，以概率 $P(S, S', T)$ 接受 $S'$；
- 降低温度 $T$，如果达到了终止温度 $T_{end}$，则停止迭代

## 代码实现

```cpp
void SA(State S, double T, double T_end, double alpha) {
    while (T > T_end) {
        State S_new = generateNewState(S);
        double deltaE = S_new.energy() - S.energy();
        if (deltaE > 0 || exp(-deltaE / T) > rand() / RAND_MAX) {
            S = S_new;
        }
        T *= alpha;
    }
}
```

不过由于模拟退火算法本身还是一个随机算法，所以在实际应用中需要针对各参数进行多次调整，以及在算法前进行预处理、算法执行完后对结果进行处理。

## 参考资料

- Wikipedia: Simulated annealing: <https://www.wikiwand.com/en/Simulated_annealing>
