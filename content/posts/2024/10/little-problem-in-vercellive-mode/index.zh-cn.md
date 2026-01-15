---
title: "LeaveIt 系主题在 Vercel Live 模式下遇到的加载问题"
subtitle: ""
description: ""
slug: 07a220
date: 2024-10-27T09:36:11+08:00
lastmod: 2024-10-27T09:36:11+08:00
draft: false

resources:
- name: featured-image
  src: featured-img.webp
- name: featured-image-preview
  src: featured-img.webp

# 标签
tags: ["Vercel", "博客", "Cookie"]
# 分类
categories: ["网站"]
# 系列(如果下面这一行注释掉，就不会显示系列为空了)
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

> [!NOTE]-
> 本篇文章原是在 DoIt 主题上测试的。不过由于更换了主题，并且经过测试，发现 LeaveIt 系的主题都存在这样的问题，在这里提一嘴。

## 前言

书接上文，在大致完成了主题 FixIt 博客的搭建后，我将博客通过 Vercel 进行了部署。

Vercel 是支持通过 Vercel Live 的模式帮助网站持有者来实时进行一些修改的，并且通过 Vercel 进入到的自己的网站，都默认会进入到该模式中。

<!--more-->

## 遇见的问题

{{<figure src="/img/missing-of-js.webp" title="部分 js 文件未能加载成功" width="100%">}}

如上图所示，左侧加载博客标题的动画和右侧搜索栏的打开都需要加载 js 文件，但是这两个部分都没有进行加载。

而且更加奇怪的是，我通过 Vercel 部署的连接，也就是那个 `xxx.vercel.app` 格式的连接进入，就发现是可以正常加载的，这让我感到非常的疑惑。

这里需要特别感谢我的好室友 [MLAcookie](https://mlacookie.top/)，帮助我一起找到解决方法。

一开始我们觉得这个是 cdn 缓存的问题，但是在 [MLAcookie](https://mlacookie.top/) 和另外一个室友的电脑上用相同的环境进行访问后，发现他们都加载成功了。后来查看了请求到的文件，发现浏览器已经请求到了对应的 js 文件。

{{<figure src="/img/js-loaded-in-local.webp" title="本地已请求到打印动画的 js 文件" width="100%">}}

所以就基本排除是 cdn 缓存的问题，我们将下一个目标锁定在本地加载的过程中出现异常。

## 本地调试
我和室友那首先对比了一下请求到的文件，发现两边请求到的文件有较大的差异，经过多次比对和拒绝某些特定文件的请求后发现，这些多出来的文件主要是来自 `rocket-loader.js` 调用的 `feedback.js` 这个文件中的请求。

在控制台给请求到的网页文件打了端点，本地的加载在 `liveload.js` 这个地方抛出了异常。经过逐步运行确定该错误最终来自 `vercel-live` 模块的调用，为一个神秘的反复调用本地栈空间导致的爆栈错误，不过由于对该方面的知识欠缺，所以更加原理的问题并没有理解。

{{<figure src="/img/liveload-error.webp" title="本地渲染时抛出的异常" width="100%">}}

所以大致可以确定问题就是 Vercel Live 模式本地出现了一些神秘的问题，导致最后页面没有正确地渲染出来。

## 最终解决
我换了一台全新的电脑，并没有从 Vercel 进入到部署的博客中，而是进行直连，上述问题就不存在了。

回到我一开始的电脑，我尝试退出 Vercel 能不能一样解决问题。不过不幸的是 js 加载依旧未成功，页面中悬停着 Vercel Live 的评论相关按键，我还是通过该模式进行页面的访问。

查看了本地 Cookie 情况，可以发现在退出登录的情况下，部署的项目还是通过本地的 Cookie 识别并启动了 Live 模式。

{{<figure src="/img/cookie.webp" title="部署项目通过本地 Cookie 启动 Vercel Live 模式" width="100%">}}

不过经过尝试，无法直接通过阻止使用该 Cookie 来解决问题，主题依旧会因为抛出异常而停止加载。

{{<figure src="/img/blocking-cookie-error.webp" title="直接阻止 Cookie 导致错误" width="100%">}}

以上主题亮暗色调节的部分需要读取 Cookie 的数据，如果直接阻止 Cookie 会导致上图代码中的第 16 行判断 `localStorage` 存在，但是因为无法读取而导致了错误。

## 解决方案

如果你也遇到了该问题，可以通过删除**本网站的**的 Cookie 解决，这个方法在每一次通过 Vercel 访问博客的时候都需要实施一次，虽然不会造成什么问题但显得非常的麻烦。

{{<figure src="/img/delete-cookie.webp" title="删除本网站的 Cookie" width="70%">}}

Vercel 本身的功能已经足够强大并且稳定了，所以第一次从 github 导入项目后就可以直接不用管 Vercel 了。

不想用这么麻烦的方法的话，我还是建议不要通过 Vercel 进入，直接用域名进入罢 OvO。
