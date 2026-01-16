---
title: "APIO 2019 做题总结"
subtitle: ""
description: ""
slug: a2d8f4
date: 2022-06-27T09:26:16+08:00
lastmod: 2022-06-27T09:26:16+08:00
draft: false

resources:
# 文章特色图片
- name: featured-image
  src: featured-img.webp
# 首页预览特色图片
- name: featured-image-preview
  src: featured-img.webp

# 标签
tags: ["找规律","APIO","线段树","并查集","图论","数据结构","分治"]
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

## 奇怪装置

### 解题思路

考虑循环节大小，设循环节长度为 \(S\)，二元组分别为两个函数表示

\[x(t)=(t+\lfloor \frac tB\rfloor) \; mod \; A \]

\[y(t)=t \; mod \; B \]

\[x(t+S)=(t+S+\lfloor \frac {t+S}B\rfloor) \; mod \; A \]

\[y(t+S)=(t+S) \; mod \; B \]

\[t\equiv t+S \mod B \]

显然

\[S\mod B=0 \]

那么令\(S=kB\)

\[x(t+kB)\equiv (t+\lfloor \frac tB\rfloor+kB+k)\mod A \]

\[k(B+1)\mod A=0 \]

显然

\[k=\frac A{\text{gcd}(B+1,A)} \]

循环节大小为

\[B\cdot \frac A{\text{gcd}(B+1,A)} \]

然后就是离散化后线段覆盖即可

时间复杂度 \(\Theta(n \log n)\)

### 参考程序

```cpp
#include <bits/stdc++.h>
#define inf 0x3f3f3f3f
#define INF 0x3f3f3f3f3f3f3f3f

using namespace std;

typedef long long ll;
typedef unsigned long long ull;

const int N = 2e6 + 5;

ll A, B, cir, ans; 
int n, tot;
pair<ll, ll> sec[N];

template <class T> 
T gcd(T x, T y) {
  return !y ? x : gcd(y, x % y); 
}

ll calc(ll x) {
  return x % cir ? x % cir : cir; 
}

int main() {
  scanf("%d %lld %lld", &n, &A, &B);
  cir = A / gcd(A, B + 1);
  if (3e18 / cir < B) 
    cir = 3e18;
  else 
    cir *= B;
  tot = n;  
  for (int i = 1; i <= n; i++) {
    scanf("%lld %lld", &sec[i].first, &sec[i].second);
    if (sec[i].second - sec[i].first + 1 >= cir) {
      printf("%lld\n", cir);
      return 0;
    }
    sec[i] = make_pair(calc(sec[i].first), calc(sec[i].second));
    if (sec[i].first > sec[i].second) {
      sec[++tot] = make_pair(sec[i].first, cir);
      sec[i] = make_pair(1, sec[i].second);
    }
  }
  sec[++tot] = make_pair(cir + 1, cir);
  sort(sec + 1, sec + 1 + tot);
  ll left = sec[1].first, right = sec[1].second;
  for (int i = 2; i <= tot; i++) {
    if (sec[i].first <= right) {
      right = max(right, sec[i].second);
    } else {
      ans += right - left + 1; 
      left = sec[i].first, right = sec[i].second;
    }
  }
  printf("%lld\n", ans);
  return 0;
}
```

## 路灯

### 解题思路

我们称能够相互到达的子段为关键段。

维护关键段可以用线段树维护。

先考虑断开某一个关键段 \([l,r]\) 的某一条边 \(k\)，那么这个操作只对 \([l,k]\) 到 \([k+1,r]\) 内的答案有贡献。同理加入这条边也只对这两个关键段有贡献。

考虑将这个贡献差分，就得到了：

* 若插入，在第 \(i\) 个时刻，\([l,k]\) 到 \([k+1,r]\) 内的答案 \(-i\)
* 若删除，在第 \(i\) 个时刻，\([l,k]\) 到 \([k+1,r]\) 内的答案 \(+i\)

一个特判，如果每个询问的最后时刻 \(t\)，若这两个点是联通的，那么我们需要 \(+t\)。

接下来，我们将两个关键段抽象成二维坐标系内的 \(x\) 轴和 \(y\) 轴，也就是每次在 \((l,k+1,k,r)\) 内的矩形进行带权修改。

因此考虑离线分治计算答案，是一个二维数点问题。

时间复杂度 \(\Theta (n \log^2n)\)

### 参考程序

```cpp
#include <bits/stdc++.h>

using namespace std;

typedef long long ll;

const int N = 3e5 + 5;

char s[N];
int n, q, etot;
int a[N], ans[N];
bool vis[N];

struct Node {
  pair<int, int> bd;
  bool tag;
} tr[N << 2];

struct Op {
  int t, x1, y1, x2, y2, id, v;
} op[N];

struct Event {
  int x, ys, ye, v;

  bool operator<(const Event& rhs) const {
    return x < rhs.x;
  }
} evt[N << 1];

int fw[N];

void pushdown(int x) {
  if (tr[x].tag) {
    tr[x << 1].bd = tr[x << 1 | 1].bd = tr[x].bd;
    tr[x << 1].tag = tr[x << 1 | 1].tag = 1;
    tr[x].tag = 0;
  }
}

void build(int x, int l, int r) {
  if (l == r) {
    tr[x].bd = make_pair(l, l);
    return;
  }
  int mid = (l + r) >> 1;
  build(x << 1, l, mid);
  build(x << 1 | 1, mid + 1, r);
}

void modify(int x, int l, int r, int ql, int qr, pair<int, int> val) {
  if (ql <= l && r <= qr) {
    tr[x].bd = val;
    tr[x].tag = 1;
    return;
  }
  pushdown(x);
  int mid = (l + r) >> 1;
  if (ql <= mid)
    modify(x << 1, l, mid, ql, qr, val);
  if (qr > mid)
    modify(x << 1 | 1, mid + 1, r, ql, qr, val);
}

pair<int, int> query(int x, int l, int r, int p) {
  if (l == r) 
    return tr[x].bd;
  pushdown(x);
  int mid = (l + r) >> 1;
  if (p <= mid) 
    return query(x << 1, l, mid, p);
  else 
    return query(x << 1 | 1, mid + 1, r, p);
}

void add(int p, int v) {
  for (int i = p; i <= n + 1; i += i & -i)
    fw[i] += v; 
}

int query(int p) {
  int ans = 0; 
  for (int i = p; i >= 1; i -= i & -i)
    ans += fw[i];
  return ans;
}

void addEvent(int x, int ys, int ye, int v) {
  evt[++etot] = (Event){x, ys, ye, v};
}

void solve(int l, int r) {
  if (l == r)
    return;
  int mid = (l + r) >> 1;
  solve(l, mid), solve(mid + 1, r);
  etot = 0;
  for (int i = l; i <= mid; i++)
    if (op[i].t == 1) {
      addEvent(op[i].x1, op[i].y1, op[i].y2, op[i].v);
      addEvent(op[i].x2 + 1, op[i].y1, op[i].y2, -op[i].v);
    }
  sort(evt + 1, evt + 1 + etot);
  sort(op + mid + 1, op + 1 + r, [](Op a, Op b) {
    return a.x1 < b.x1;
  });
  int pt = 1; 
  for (int i = mid + 1; i <= r; i++)
    if (op[i].t == 2) {
      while (pt <= etot && evt[pt].x <= op[i].x1) {
        add(evt[pt].ys, evt[pt].v);
        add(evt[pt].ye + 1, -evt[pt].v);
        pt++;
      }
      ans[op[i].id] += query(op[i].y1);
    }
  for (int i = 1; i < pt; i++) {
    add(evt[i].ys, -evt[i].v);
    add(evt[i].ye + 1, evt[i].v);
  }
}

int main() {
#ifndef ONLINE_JUDGE
  freopen("input.in", "r", stdin);
  freopen("output.out", "w", stdout);
#endif
  scanf("%d %d", &n, &q);
  scanf("%s", s);
  for (int i = 1; i <= n; i++)
    a[i] = s[i - 1] - '0';
  build(1, 1, n + 1);
  for (int i = 1; i <= n; i++) 
    if (a[i] == 1 && !vis[i]) {
      int it = i;
      vis[i] = 1; 
      while (a[it]) 
        it++, vis[it] = 1; 
      modify(1, 1, n + 1, i, it, make_pair(i, it));
    }
  memset(ans, 0, sizeof ans);
  memset(vis, 0, sizeof vis);
  char opt[10];
  for (int i = 1; i <= q; i++) {
    scanf("%s", opt);
    op[i].t = (opt[0] == 'q' ? 2 : 1);
    op[i].id = i; 
    if (op[i].t == 1) {
      int k; 
      scanf("%d", &k);
      if (a[k] == 1) {
        pair<int, int> p = query(1, 1, n + 1, k);
        op[i].x1 = p.first, op[i].x2 = k;
        op[i].y1 = k + 1, op[i].y2 = p.second;
        op[i].v = i; 
        modify(1, 1, n + 1, p.first, k, make_pair(p.first, k));
        modify(1, 1, n + 1, k + 1, p.second, make_pair(k + 1, p.second));
      } else {
        pair<int, int> p1 = query(1, 1, n + 1, k), p2 = query(1, 1, n + 1, k + 1);
        op[i].x1 = p1.first, op[i].x2 = k; 
        op[i].y1 = k + 1, op[i].y2 = p2.second;
        op[i].v = -i;
        modify(1, 1, n + 1, p1.first, p2.second, make_pair(p1.first, p2.second));
      }
      a[k] ^= 1;
    } else {
      scanf("%d %d", &op[i].x1, &op[i].y1);
      pair<int, int> p = query(1, 1, n + 1, op[i].x1);
      if (p.second >= op[i].y1 && p.first <= op[i].y1) 
        ans[i] += i; 
      vis[i] = 1; 
    }
  }
  memset(fw, 0, sizeof fw);
  solve(1, q);
  for (int i = 1; i <= q; i++)
    if (vis[i])
      printf("%d\n", ans[i]);
  return 0;
}
```

## 桥梁

### 解题思路

#### 子任务 链

本质是一个序列问题。

你可以发现每一次的操作是进行一次向左边扩展和向右边扩展，找到第一条不满足的边。也就是说你需要找到一个包含起点的最大区间，满足在上面的所有边权都满足 \(\geq y\)。

因为支持修改操作，所以考虑用二分答案，线段树查询的方法在 \(O(nlog^2n)\) 的时间解决。

#### 子任务 只有查询

我们假设联通块中最小的边权为 \(min\)，那么某一个起点开始能够遍历整个联通块的充要条件是 \(min \ge y\)。

也就是最小边的边权 \(\ge y\)，这就容易想到用 \(Kruscal\) 重构树来实现这个过程。

建立出 \(Kruscal\) 重构树之后，以每一个节点为根节点的子树，都可以被以根节点为起点的路径包含，所以只需要倍增找到**最后一个满足条件的祖先节点**，然后其子树大小即为答案。

也可以采用离线做法，将操作离线后，按照 \(y\) 值从大到小查询，将所有的边按照顺序加入到并查集中，**答案就是联通块大小**。

### 正解

采用「子任务 只有查询」中的做法，考虑暴力。

我们按照询问分块，在当前块中的所有操作都相当于和前面所有块的到的新图**重新暴力**求一遍答案，然后**更新为新图**。

将当前块中不需要修改的边加入到联通块中，考虑修改的边，将在当前操作之前的每一条边最后一次操作得到的权值和标准值 \(y\) 进行比较，否则视为未被修改。

每一次操作，同一条边可能被不同的修改操作，所以考虑要**可撤销并查集**。

### 参考程序
 
```cpp
#include <bits/stdc++.h>
#define inf 0x3f3f3f3f
#define INF 0x3f3f3f3f3f3f3f3f

using namespace std;

typedef long long ll;
typedef unsigned long long ull;

namespace FastIO {

template <class T> 
void rd(T& x) {
  x = 0;
  char ch = 0;
  bool f = 0;
  while (!isdigit(ch)) f |= ch == '-', ch = getchar(); 
  while (isdigit(ch)) x = x * 10 + (ch ^ 48), ch = getchar();
  f ? x = -x : 1;
}

template <class T> 
void ptf(T x) {
  if (x < 0) putchar('-'), x = -x;
  if (x > 9) ptf(x / 10);
  putchar(x % 10 + 48);
}

void read(int& x) { rd(x); }
void read(long long& x) { rd(x); }
void read(unsigned int& x) { rd(x); }
void read(unsigned long long& x) { rd(x); }
void read(char& x) { x = getchar(); }
void read(string& x) { cin >> x; }
template <class T, class R> 
void read(pair<T, R>& x) {
  read(x.first), read(x.second);
}
template <class T> 
void read(vector<T>& x) { 
  for (auto& ele : x) read(x); 
}

void write(int x) { ptf(x); }
void write(long long x) { ptf(x); }
void write(unsigned long long x) { ptf(x); }
void write(unsigned int x) { ptf(x); }
void write(char x) { putchar(x); }
void write(char* x) { printf("%s", x); }
void write(string x) { cout << x; }
template <class T> 
void write(vector<T> x) {
  for (auto ele : x)  (x); 
}
template <class T, class R> 
void write(pair<T, R> x) {
  write(x.first), putchar(','), write(x.second);
}
template <class T> 
void writeln(T x) {
  write(x), puts("");
}

}

using FastIO::read;
using FastIO::write;
using FastIO::writeln;

namespace RevocableDisjoinSet {

const int N = 1e5 + 5; 

struct DisjoinSet {
  int top, n; 
  int fa[N], sz[N];
  pair<int, int> stk[N];

  void init(int x) {
    n = x, top = 0;
    for (int i = 1; i <= n; i++) {
      fa[i] = i; 
      sz[i] = 1;
    }
  }

  int get(int x) {
    return fa[x] == x ? x : get(fa[x]); 
  }

  void merge(int x, int y) {
    int p1 = get(x), p2 = get(y);
    if (p1 != p2) {
      if (sz[p1] > sz[p2]) { swap(p1, p2); }
      fa[p1] = p2; 
      sz[p2] += sz[p1];
      stk[++top] = make_pair(p1, p2);
    }
  }

  void revoke(int goal) {
    while (top > goal) {
      fa[stk[top].first] = stk[top].first;
      fa[stk[top].second] = stk[top].second; 
      sz[stk[top].second] -= sz[stk[top].first];
      top--; 
    }
  }

  int check(int x, int y) {
    return get(x) == get(y);
  }
};

}

// ===============

RevocableDisjoinSet::DisjoinSet dsu;

const int N = 5e4 + 5, M = 1e5 + 5; 

struct Ed {
  int u, v, w, id; // 现在第 i 条边 原来的编号
  
  bool operator<(const Ed& rhs) const {
    return w > rhs.w || (w == rhs.w && id < rhs.id);
  }
} ed[M], ed2[M], tmp[M];

struct Op {
  int t, x, y, id;

  bool operator<(const Op& rhs) const {
    return y > rhs.y;
  }
} op[M], op1[M], op2[M];

int n, m, q, tot;
int ismdy[M], did[M], ans[M];
int rk[M]; // 第 i 条边的 rank (按照 w 排序)

/**
 * 对于当前的边 i， 原来的编号为 ed[i].i 
 * 其中原来的编号是没有排序过的，现在的编号为排序完的下标
 */
void solve() {
  dsu.init(n);
  memset(ismdy, 0, sizeof ismdy);
  int t1 = 0, t2 = 0, pt = 1; // pt 遍历现在的编号
  for (int i = 1; i <= tot; i++) 
    if (op[i].t == 1) {
      op1[++t1] = op[i];
      ismdy[op[i].x] = 1; // 将这条边的原来的编号打上修改标记
    } else 
      op2[++t2] = op[i];
  // op2 是询问 op1 是修改
  sort(op2 + 1, op2 + 1 + t2);
  for (int i = 1; i <= t2; i++) {
    while (pt <= m && ed[pt].w >= op2[i].y) {
      if (!ismdy[ed[pt].id]) 
        dsu.merge(ed[pt].u, ed[pt].v);
      pt++; 
    }
    int temp = dsu.top;
    for (int j = 1; j <= t1; j++) 
      did[op1[j].x] = 0;
    for (int j = 1; j <= t1; j++) 
      if (op1[j].id < op2[i].id) 
        did[op1[j].x] = j; 
    for (int j = 1; j <= t1; j++) 
      if (did[op1[j].x] == j) {
        if (op1[j].y >= op2[i].y) 
          dsu.merge(ed[rk[op1[j].x]].u, ed[rk[op1[j].x]].v);
      } else if (did[op1[j].x] == 0 && ed[rk[op1[j].x]].w >= op2[i].y) 
        dsu.merge(ed[rk[op1[j].x]].u, ed[rk[op1[j].x]].v);
    ans[op2[i].id] = dsu.sz[dsu.get(op2[i].x)];
    dsu.revoke(temp);
  }
  memset(did, 0, sizeof did);
  int l = 1, r = 1, num = 0, t = 0;
  for (int i = t1; i >= 1; i--) 
    if (!did[op1[i].x]) {
      did[op1[i].x] = 1; 
      ed2[++num] = ed[rk[op1[i].x]];
      ed2[num].w = op1[i].y; 
    }
  sort(ed2 + 1, ed2 + 1 + num);
  while (l <= m && r <= num) {
    while (l <= m && did[ed[l].id]) 
      l++;
    if (ed[l].w >= ed2[r].w) 
      tmp[++t] = ed[l], l++;
    else 
      tmp[++t] = ed2[r], r++;
  }
  while (l <= m) {
    if (!did[ed[l].id])
      tmp[++t] = ed[l];
    l++;
  }
  while (r <= num) 
    tmp[++t] = ed2[r], r++;
  for (int i = 1; i <= m; i++) {
    ed[i] = tmp[i];
    rk[ed[i].id] = i; 
  }
}

int main() {
  read(n), read(m);
  for (int i = 1; i <= m; i++) {
    read(ed[i].u), read(ed[i].v), read(ed[i].w);
    ed[i].id = i; 
  }
  sort(ed + 1, ed + 1 + m);
  for (int i = 1; i <= m; i++) 
    rk[ed[i].id] = i; 
  read(q);
  tot = 0;
  for (int i = 1; i <= q; i++) {
    read(op[++tot].t), read(op[tot].x), read(op[tot].y);
    op[tot].id = i; 
    if (tot == 500) {
      solve();
      tot = 0;
    }
  }
  if (tot) 
    solve();
  for (int i = 1; i <= q; i++) 
    if (ans[i]) 
      writeln(ans[i]);
  return 0;
}
```