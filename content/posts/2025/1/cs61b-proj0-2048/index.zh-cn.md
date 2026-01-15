---
title: "「CS61B Proj0」2048"
subtitle: ""
description: ""
slug: 5793a4
date: 2025-01-24T16:17:53+08:00
lastmod: 2025-01-24T16:17:53+08:00
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
  'CS61B',
  'Java',
  '2048',
]
# 分类
categories: ["学习"]
# 合集(如果下面这一行注释掉，就不会显示系列为空了)
collections: ["CS61B 学习记录"]
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

cs61b 课程中简单练手的 proj，熟悉一下 Java 的语法。

最近在写 2024 的年终总结，而且接下来的 2025 年需要应付一些考试，cs61b 的课程学习就简单记录一下。

做的版本是 21sp 的课程 skeleton。

我的课程代码仓库：[CS61B](https://github.com/mulberror/CS61B)

## emptySpaceExists

完成 `Model` 类中的 `emptySpaceExists` 方法，判断棋盘 `board b` 中是否存在空位。

使用双重循环遍历棋盘，如果存在空位则返回 `true`。

需要简单注意一下，访问棋盘中的元素，需要调用 `board.tile` 方法，并且需要注意该 `tile` 返回是否为 `null`。

```java
public static boolean emptySpaceExists(Board b) {
    int size = b.size();
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (b.tile(i, j) == null) {
                return true;
            }
        }
    }
    return false;
}
```

## maxTileExists

完成 `Model` 类中的 `maxTileExists` 方法，判断棋盘 `board b` 中是否存在最大的方块。

也是使用双重循环对棋盘进行遍历，如果存在最大的方块则返回 `true`，需要简单注意一下，需要使用 `Model` 类中 `MAX_PIECE` 的判断是否为最大方块。

```java
public static boolean maxTileExists(Board b) {
    int size = b.size();
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (b.tile(i, j) != null && b.tile(i, j).value() == MAX_PIECE) {
                return true;
            }
        }
    }
    return false;
}
```

## atLeastOneMoveExists

完成 `Model` 类中的 `atLeastOneMoveExists` 方法，判断棋盘 `board b` 中是否存在至少一个可以移动的方式。

首先判断是否存在一个空格，如果存在则返回 `true`。

然后对棋盘的行列分别进行遍历，判断两个连续的**非空**格中的值是否相同，如果相同则返回 `true`。

```java
public static boolean atLeastOneMoveExists(Board b) {
    if (emptySpaceExists(b)) {
        return true;
    }
    int size = b.size();
    for (int i = 0; i < size; i++) {
        int lastx = -1, lasty = -1;
        for (int j = 0; j < size; j++) {
            if (b.tile(i, j) != null) {
                if (b.tile(i, j).value() == lastx) {
                    return true;
                }
                lastx = b.tile(i, j).value();
            }
            if (b.tile(j, i) != null) {
                if (b.tile(j, i).value() == lasty) {
                    return true;
                }
                lasty = b.tile(j, i).value();
            }
        }
    }
    return false;
}
```

## tilt

该部分是本次 proj 的相对难点，需要实现 `Model` 类中的 `tilt` 方法，实现棋盘的移动。

首先只考虑向上移动，可以考虑使用一个 `postion` 变量表示格子会移动到的位置，初始化为 `0`。

按照列进行遍历，如果遇到一个非空格子 `t`，我们尝试将其移动到 `position` 位置：

- 如果 `position` 位置上的格子 `target` 也是非空格子且值相同，就将这两个格子**合并**，并且更新 `position` 位置和 `score` 分数。
- 如果无法合并，先更新 `position` 位置，然后将 `t` 移动到 `position` 位置。

最后检查游戏是否结束，如果有改变则设置 `changed` 为 `true`。

接下来需要考虑向其他方向进行移动，在实验文档中给出了可以使用旋转的方法，这里使用 `board.setViewingPerspective` 方法旋转视角，统一成向上移动，然后再旋转回来。

```java
public boolean tilt(Side side) {
    boolean changed;
    changed = false;

    board.setViewingPerspective(side);
    int size = board.size();
    for (int i = 0; i < size; i++) {
        int place_position_y = size - 1;
        for (int j = size - 1; j >= 0; j--) {
            if (board.tile(i, j) != null) {
                if (j == place_position_y) {
                    continue;
                }
                changed = true;
                Tile t = board.tile(i, j);
                Tile target = board.tile(i, place_position_y);
                if (target != null) {
                    if (target.value() == t.value()) {
                        score += target.value() + t.value();
                        board.move(i, place_position_y, t);
                        place_position_y--;
                    } else {
                        place_position_y--;
                        board.move(i, place_position_y, t);
                    }
                } else {
                    board.move(i, place_position_y, t);
                }
            }
        }
    }
    board.setViewingPerspective(Side.NORTH);
    checkGameOver();
    if (changed) {
        setChanged();
    }
    return changed;
}
```