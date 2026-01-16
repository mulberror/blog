---
title: "FixIt theme encountered loading problems in Vercel Live mode"
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
tags: ["Vercel", "Blog", "Cookie"]
# 分类
categories: ["Build Site"]
# 系列(如果下面这一行注释掉，就不会显示系列为空了)
collections: ["My Blog Reborn"]
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

## Preface
Continuing from the previous section, after roughly completing the setup of the FixIt blog, I deployed the blog using Vercel.

Vercel supports a Live mode that allows website owners to make real-time modifications, and any website accessed through Vercel will automatically enter this mode.

<!--more-->

## Problems Encountered

{{<figure src="/img/missing-of-js.webp" title="Some JS files failed to load" width="100%">}}

As shown in the image, the animation for loading the blog title on the left and the opening of the search bar on the right require loading JS files, but neither of these components loaded successfully.

Moreover, it was even stranger that when accessing the blog through the Vercel deployment link (in the format of `xxx.vercel.app`), everything loaded correctly, which left me puzzled.

I would like to especially thank my good roommate [MLAcookie](https://mlacookie.top/) for helping me find a solution.

Initially, we thought this might be a CDN caching issue. However, when [MLAcookie](https://mlacookie.top/) and another roommate accessed the blog using the same environment, they successfully loaded everything. After checking the requested files, we found that the browser had indeed requested the corresponding JS files.

{{<figure src="/img/js-loaded-in-local.webp" title="JS files requested locally" width="100%">}}

Thus, we could rule out CDN caching as the problem, and our next target was to identify where the loading process locally was failing.

## Local Debugging
My roommate and I first compared the requested files and noticed significant differences. After several rounds of comparison and rejecting specific file requests, we found that the extra files were mainly being called by `rocket-loader.js`, specifically from a request in `feedback.js`.

By inspecting the webpage files in the console, we discovered that the local loading threw an exception in `liveload.js`. After gradually running through the code, we identified that the error ultimately stemmed from the `vercel-live` module, leading to a mysterious stack overflow error due to repeated calls in the local stack space. However, due to my lack of knowledge in this area, I couldn't fully understand the underlying principles.

{{<figure src="/img/liveload-error.webp" title="Exception thrown during local rendering" width="100%">}}

Thus, we could roughly determine that the issue was a mysterious problem in Vercel Live mode locally, causing the page to fail to render correctly.

## Final Resolution
I switched to a completely new computer and accessed the blog directly instead of through Vercel, and the above issue was resolved.

Returning to my original computer, I attempted to log out of Vercel to see if that would solve the problem. Unfortunately, the JS still failed to load, and the Vercel Live comment buttons remained hovering on the page, as I was still accessing the page through that mode.

Checking the local cookie situation, I found that even when logged out, the deployed project still recognized and activated Live mode through local cookies.

{{<figure src="/img/cookie.webp" title="Deployed project activates Vercel Live mode through local cookies" width="100%">}}

However, I found it impossible to directly block the use of this cookie to resolve the issue; the theme still failed to load due to the thrown exception.

{{<figure src="/img/blocking-cookie-error.webp" title="Directly blocking the cookie leads to errors" width="100%">}}

The dark-light toggle part of the theme needs to read data from the cookie. Directly blocking the cookie causes the check in line 16 of the code to fail, leading to errors because it cannot read `localStorage`.

## Solution

If you encounter this problem, you can resolve it by deleting **the cookies for this site**. This method needs to be implemented each time you access the blog through Vercel, which, while not problematic, is quite cumbersome.

{{<figure src="/img/delete-cookie.webp" title="Delete the cookies for this site" width="70%">}}

Vercel's functionality is already robust and stable, so once you import your project from GitHub, you can typically proceed without needing to interact with Vercel further.

If you want to avoid this cumbersome method, I recommend accessing the blog directly through its domain instead of going through Vercel. OvO
