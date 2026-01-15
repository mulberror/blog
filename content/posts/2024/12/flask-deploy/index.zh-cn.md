---
title: "记录一次大作业的 Flask 项目服务器部署"
subtitle: ""
description: ""
slug: ec18fe
date: 2024-12-25T17:38:05+08:00
lastmod: 2024-12-25T17:38:05+08:00
draft: false

resources:
# 文章特色图片
- name: featured-image
  src: featured-img.webp
# 首页预览特色图片
- name: featured-image-preview
  src: featured-img.webp

# 标签
tags: ['Flask', 'Python', '服务器', '部署', '代理']
# 分类
categories: ["网站"]
# 合集(如果下面这一行注释掉，就不会显示系列为空了)
# collections: [""]
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

由于本学期的创新实践课程要求，我和同学合作开发了一个简易的基于 gemini 的电影 GPT 系统，并且该项目需要部署到服务器上，所以就简单记录一下服务器部署的过程。

项目的后端是基于 Python Flask 框架开发的，前端是基于 Vue.js 开发的。

<!--more-->

因为 12 月和 1 月初遇上了期末周，所以很长时间并没有更新博客内容，这篇文章也是 12 月就开始写的，不过因为事情有点多，所以一直到 1 月初才发布。

## 配置信息

- 服务器：华为云 ECS（CentOS）。
- 项目：Flask 项目
- Python 版本：3.10
- 项目地址：[movie-gpt](https://github.com/mulberror/movie-gpt)

## Flask 项目部署

部署并启动后端项目的方式有 `gunicorn` 以及 `uwsgi`，两种方式选取其一即可，这边将两种方式都简单记录一下。

### 依赖安装

本地生成 `requirements.txt` 文件：

```bash
pip freeze > requirements.txt # 生成依赖文件
```

将自己的后端项目上传到服务器后，先创建一个 `.venv` 虚拟环境：

```bash
python3 -m venv .venv
```

激活虚拟环境：

```bash
source .venv/bin/activate
```

进入到项目对应的虚拟环境后，安装依赖：

```bash
pip install -r requirements.txt
```

### 使用 gunicon 启动项目

首先安装 `gunicorn`：

```bash
pip3 install gunicorn
pip3 install greenlet # 使用异步必须安装
pip3 install eventlet # 使用eventlet workers
pip3 install gevent   # 使用gevent workers
```

比如我的项目的入口文件是 `main.py`，开放端口为 `5010`：

```bash
cd /home/wwwroot/movie-gpt
source .venv/bin/activate
gunicorn -w 4 -b 127.0.0.1:5010 main:app
gunicorn -w 4 -b 127.0.0.1:5010 main:app -D # 后台运行

# 查看 gunicon 进程
ps -ef | grep gunicorn # 查看gunicorn的进程状态

# 查看端口占用
lsof -i :5010
kill -9 $(lsof -i tcp:5010 -t) # 杀掉占用5010端口的进程
```

### 使用 uwsgi 启动项目

我采用的是 `uwsgi` 的方式启动项目。

进入到项目的目录下，启动虚拟环境，创建一个 `uwsgi.ini` 配置文件：

```ini
[uwsgi]
http-socket = 127.0.0.1:5010 # 启动使用的端口
chdir = /home/wwwroot/movie-gpt # 项目路径
wsgi-file = main.py # 项目入口文件
callable = app # Flask 项目的实例名
processes = 4 # 进程数
threads = 1 # 线程数
stats = 127.0.0.1:5011 # 状态监控端口
```

启动 `uwsgi` 项目，并设置后台运行：

```bash
uwsgi uwsgi.ini
uwsgi -d --ini config.ini # 后台运行
```

查看 `uwsgi` 进程，查看端口占用情况、杀掉相应进程的方式和 `gunicon` 是一样的。

```bash
ps -ef | grep uwsgi
```

在启动项目后端后，就可以使用 `apifox` 工具进行测试相应端口。

## 服务器网络代理配置

由于我后端的部分需要调用外网的 API，所以需要配置服务器的网络代理。

这边采用的是 `Clash` 代理方式，首先就是在服务器上安装 `Clash`。

### 运行 Clash

先进入到对应文件目录下，给予 `Clash` 可执行权限并运行：

```bash
chmod +x clash-linux-amd64 # 给予可执行权限
./clash-linux-amd64 # 运行 Clash
```

直接运行后，会在当前目录下生成一个空的 `config.yaml` 配置文件，并且提示“找不到MMDB文件”。

配置文件部分可以在本地下载好代理的配置文件，然后上传到服务器上。

MMDB 文件可以在 [GeoLite2](https://dev.maxmind.com/geoip/geoip2/geolite2/) 下载或者在 github 上有大佬仓库提供了[下载连接](https://github.com/P3TERX/GeoLite.mmdb)。

将以上的部分配置好后，就可以正常运行代理了。

### 后台运行与全局代理

然后配置一个后台运行和自动启动，进入到服务器 `/etc/systemd/system/` 目录下，创建一个 `clash.service` 文件：

```bash
vim /etc/systemd/system/clash.service
```

然后进行配置文件的设置：

```bash
[Unit]
Description=Clash service
After=network.target

[Service]
Type=simple
User=root
ExecStart= #这里写你的clash运行的绝对路径
Restart=on-failure
RestartPreventExitStatus=23

[Install]
WantedBy=multi-user.target
```

然后启动，并查看启动是否正常：

```bash
systemctl start clash
systemctl status clash
```

如果需要配置全局代理，那就进入到服务器的 `/etc/profile` 文件中，添加以下内容：

```bash
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890
```

最后访问一下 `google.com` 测试一下代理是否生效。

```bash
curl google.com # 如果返回正常，说明代理生效
```

## 参考文章

- [Flask项目部署到服务器](https://www.cnblogs.com/Mystogan/p/16144753.html)
- [Linux中安装Clash并且实现全局代理](https://www.fuxi.info/archives/273)