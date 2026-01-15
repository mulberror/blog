---
title: "用 Hugo + Vercel 重建博客"
subtitle: ""
description: ""
slug: e64b61
date: 2024-10-22T00:51:43+08:00
lastmod: 2024-10-24T00:51:43+08:00
draft: false

resources:
# 文章特色图片
- name: featured-image
  src: featured-img.webp
# 首页预览特色图片
- name: featured-image-preview
  src: featured-img.webp

# 标签
tags: ["hugo", "vercel", "博客"]
# 分类
categories: ["建站"]
# 系列
collections: ["小站重生记"]
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
---

## 前言

我之前也是用 Hugo 框架搭建的博客，使用的主题是 PaperMod。

不过因为大部分存货都丢失了，再加上碰巧见看到了 [ryan4yin](https://thiscute.world/) 大佬的博客，从而了解到了 LeaveIt 系的主题以及 [Lruihao](https://lruihao.cn/) 大佬制作的 [FixIt](https://fixit.lruihao.cn/zh-cn/) 主题。

FixIt 主题非常符合我一开始预想的博客样式，所以就直接花了一段时间进行了个人博客的重建，并且在这个期间还花了一点时间从零学了点前端知识，简单地写了一个简陋的个人主页 [My Little World](https://www.mulbry.top/)。

<!--more-->

## 相关版本信息

- Hugo: v0.136.4
- FixIt: 0.3.X

## Hugo + Gitpage

### 本地配置

这个部分对于每个 Hugo 用户应该都比较熟练，在这里简单记录一下。

首先就是安装 Go 和 Hugo，根据官方文档配置即可 [Hugo Document](https://gohugo.io/documentation/)。

安装好依赖后，就可以进行本地博客相关的配置了。

我选择的是 [FixIt](https://github.com/HEIGE-PCloud/FixIt) 主题，直接就参考了作者编写的文档 [主题文档 - 基本概念](https://fixit.lruihao.cn/zh-cn/documentation/installation/)。

不过因为作者的文档适用 FixIt 0.2.X 的版本，所以在写这篇文章的时候，也就是我配置博客时还是遇到少许问题。

### 部署 Gitpage

这部分主要依据 Hugo 官方文档中的配置教程 [Host on GitHub Pages](https://gohugo.io/hosting-and-deployment/hosting-on-github/)。

我之前尝试了三个主题的配置，如果单纯的将 `hugo server` 生成的 `\public` 静态资源直接推到 github 仓库中，最后渲染出来的静态页面多少渲染都会出问题，因此还是采取官方文档的部署方法。

根据官方配置的方法，一般只需要注意将 `.github/workflows/hugo.yaml` 中的 branch 信息以及 `HUGO_VERSION` 中的 Hugo 版本信息替换成自己对应的即可。

## Vercel

Vercel 从 github 中同步并部署项目的功能十分方便。

在本地资源 push 到 github 仓库后，Vercel 会直接同步资源并且部署的速度非常快。

使用 github 账号登录 Vercel 或者注册 Vercel 账号再关联自己的 github 账号，然后点击 `Add New...` 从 github 仓库中导入部署到 Gitpage 的博客。

{{<figure src="/img/port-from-vercel.webp" title="Vercel 中导入对应 github 项目" width="90%">}}

{{<figure src="/img/choose-hugo.webp" title="选择 Hugo 框架" width="90%">}}

导入后，Vercel 中的项目和选择的 github 仓库就连接在一起了，只要本地 commit 一次，Vercel 中的项目也会重新部署一次。

并且每一个托管的项目都会给一个独立的 url 连接，这样方便自己的域名解析。

## 小结
本来是打算写一段配置 Cloudflare 的过程的。但是这段过程其实挺魔幻的，还多亏了我的室友。

这一段经历以及在配置博客的过程中遇到的一些问题打算还是另起文章来记录 OvO。
