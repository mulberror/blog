---
title: "AGC010 做题记录"
subtitle: ""
description: ""
slug: 82582f
date: 2022-09-01T20:41:31+08:00
lastmod: 2022-09-01T20:41:31+08:00
draft: false

resources:
# 文章特色图片
- name: featured-image
  src: featured-img.webp
# 首页预览特色图片
- name: featured-image-preview
  src: featured-img.webp

# 标签
tags: []
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

## 赛次总结

整场是偏向博弈和构造的思维场，总体思维都较为巧妙。

## A. Addition

### 题意

给定一个 n 个整数的数组 A，每次可以删去一对奇偶性相同的 A_i,A_j，再添加一项 A_i+A_j。

判断是否能够通过若干次操作后使得数组只剩下一项。

### 参考程序

```cpp
#include <bits/stdc++.h>

using i64 = long long;

int main() {
  int n;
  std::cin >> n;
  int ans = 0; 
  for (int i = 0; i < n; i++) {
    int x; 
    std::cin >> x; 
    if (x % 2 == 1) {
      ans++; 
    }
  }

  if (ans % 2 == 0) {
    std::cout << "YES\n";
  } else {
    std::cout << "NO\n";
  }
  return 0; 
}
```

## B. Boxes

### 题意

有 n 个箱子围成一圈，第 i 个箱子有 A_i 个石头。

每次可以选择箱子，箱子的编号为 i，然后对于每个 j∈[1,N]，将第 (i+j) 个箱子移除 j j 个石头。

其中编号为 n+k 的箱子，视为编号为 k 的箱子。

如果箱子中石头的个数不足移除的个数，那么就不能进行这个操作。

### 题解

题目要求判断这个数组是否能是若干个循环的 [1,n] 的数组相加而成的。

考虑对数组差分，会发现问题被转化为：差分数组是否是若干个循环 1,1,−(n−1) 的数组的和的形式。

对于每个位置，解出 −(n−1) 的个数。

−x(n−1)+k−x=d i f

x=k−d i f n x=k−d i f n

需要保证的是每一次解出来的 x x 次数不能为负数，并且 x x 的总和必须要是用总数计算出来的次数。

```cpp
#include <bits/stdc++.h>

using i64 = long long;

const int N = 1e5 + 5;

int n;
i64 a[N];

int main() {
  std::cin >> n;
  i64 sum = 0; 
  for (int i = 0; i < n; i++) {
    std::cin >> a[i]; 
    sum += a[i]; 
  }
  if (sum % (1ll * (n + 1) * n / 2) != 0) {
    std::cout << "NO\n";
    return 0; 
  }
  i64 k = sum / (1ll * (n + 1) * n / 2);
  i64 ans = 0; 
  bool flag = 1; 
  for (int i = 0; i < n; i++) {
    i64 dif = a[(i + 1) % n] - a[i]; 
    if ((k - dif) % n != 0 || k < dif) {
      flag = 0; 
      break; 
    }
    ans += (k - dif) / n; 
  }
  if (ans != k || !flag) {
    std::cout << "NO\n"; 
  } else {
    std::cout << "YES\n"; 
  }
  return 0; 
}
```

## C. Clean

### 题意

给定一个 n 个节点的树，一开始第 i 号节点上有 $A_i$ 个石头。

每次可以选择一对树叶，然后移除这两个节点路径所有节点上一个石头。

如果有路径上有节点没有石头，那么就不能进行这个操作。

判断是否可以通过若干次操作后，移除所有节点的石头。

### 题解

对于每一个节点 u，路径是由两部分组成：经过节点 u 通往子树的路径 $x_u$，经过节点 $u$ 通往父节点的路径数 $y_u$。

记 $sum$ 为 u 节点的子节点的通往节点 u 的路径和 $\sum y_v$，就是子节点中还未匹配的路径。

因为这些路径如何匹配都需要经过节点 u，所以可以列出以下方程。
$$
2 x_u+y_u=sum \\
x_u+y_u=a[u]
$$
那么对于每一个节点都是需要满足这个条件，并且有以下若干种情况需要特判：

* 根节点不能有上传的路径，即 $y_root=0$
* 每一个子节点上传的路径不能超过 $a_u$
* 如果只存在两个解，那么无法找到适合的根节点，直接判断 $a[1]=a[2]$

### 参考程序

```cpp
#include <bits/stdc++.h>

using i64 = long long;

const int N = 1e5 + 5;

int n;
std::vector<int> g[N];
int a[N];
i64 up[N];
bool flag;
int root;

void dfs(int u, int fa) {
  if ((int)g[u].size() == 1) {
    up[u] = a[u];
    return; 
  }
  i64 sum = 0, mx = 0;
  for (auto v : g[u]) {
    if (v != fa) {
      dfs(v, u); 
      sum += up[v];
      mx = std::max(mx, up[v]);
      std::cerr << "debug: " << u << ' ' << v << ' ' << up[v] << '\n';
    }
  }
  i64 x = sum - a[u];
  i64 y = 2 * a[u] - sum;
  if (u == root) {
    std::cerr << "dbg:" << x << ' ' << y << ' ' << mx << ' ' << a[u] << '\n';
  }
  if (x < 0 || y < 0 || mx > a[u]) {
    flag = 0;
  }
  up[u] = y; 
  std::cerr << "node : " << u << ' ' << y << '\n';

  if (u == root && y != 0) {
    flag = 0; 
  }
}

int main() {
  memset(up, 0, sizeof up);
  std::cin >> n;
  for (int i = 1; i <= n; i++) {
    std::cin >> a[i];
  }
  for (int i = 1; i < n; i++) {
    int u, v;
    std::cin >> u >> v;
    g[u].push_back(v);
    g[v].push_back(u);
  }
  if (n == 2) {
    std::cout << (a[1] == a[2] ? "YES" : "NO") << '\n';
    return 0; 
  }
  flag = 1;
  for (int i = 1; i <= n; i++) {
    if ((int)g[i].size() > 1) {
      root = i; 
      break;
    }
  } 
  dfs(root, 0);
  std::cout << (flag ? "YES" : "NO") << '\n';
  return 0; 
}
```

## D. Decrementing

### 题意

A A 和 B B 可以轮流对一个最大公约数为 1 1 的数组进行操作：

每次可以选择一个不小于 2 2 的数，然后使其−1−1，然后再让所有的数除去他们的最大公因子。

如果有一个人无法进行操作，那么就输掉比赛，每个人都绝顶聪明，问哪一个人可以赢得比赛。

### 题解

考虑问题的简化版。

如果没有除去 gcd gcd 的操作，那么当前的状态只和所有数到 1 1 的操作次数的奇偶性有关，并且不会存在干扰状态的操作。

首先因为每一次操作完后的数组的最大公因子都为 1 1，所以可以判定是由至少一个奇数和若干个偶数组成的。

对于原题，原先的必胜态，是至少一个奇数和奇数个偶数组成情况，那么只需要对偶数进行操作，然后使对手处于至少两个奇数和偶数个偶数的必败态，并且没有方法改变当前的必败态。

必败态，是至少一个奇数和偶数个偶数的组成情况，如果当前只有一个奇数，那么就可以对奇数进行操作，这样可能可以改变当前的状态，但是如果不止一个奇数，那么对手一定在当前操作后继续制造出一个奇数，那么就一直存在于必败态中。

```cpp
#include <bits/stdc++.h>

using i64 = long long;

const int N = 1e5 + 5;

int n;
int a[N];

int gcd(int x, int y) {
  return y == 0 ? x : gcd(y, x % y); 
}

bool check() {
  i64 sum = 0;
  int evenCnt = 0, pos = 0;
  for (int i = 1; i <= n; i++) {
    sum += a[i] - 1;
    if (a[i] > 1 && a[i] % 2 == 1) {
      evenCnt++;
      pos = i;
    }
  }
  if (sum % 2 == 1) {
    return 1;
  } else if (evenCnt != 1) {
    return 0;
  }
  a[pos]--;

  int _gcd = a[1];
  for (int i = 2; i <= n; i++) {
    _gcd = gcd(_gcd, a[i]);
  }
  for (int i = 1; i <= n; i++) {
    a[i] /= _gcd;
  }
  return check() ^ 1;
}

int main() {
  std::cin >> n;
  for (int i = 1; i <= n; i++) {
    std::cin >> a[i];
  }
  if (check()) {
    printf("First\n");
  } else {
    printf("Second\n");
  }
  return 0; 
}
```

* 参考题解：[Atcoder AGC010 题解](https://blog.csdn.net/litble/article/details/83183748)