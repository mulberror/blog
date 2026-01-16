# Mulberry's Hugo Blog

## 提交 commit 规范

- `newpost [post name]`：创建新文章
- `modify [post name]`：修改文章内容
- `config`：更改博客基本设置
- `feat`：博客添加新功能

## 本地预览

```bash
hugo server --disableFastRender -e production --buildDrafts # 无视 draft
```

## 工作流

```bash
hugo server --disableFastRender -e production
# 上传 index.json 文件到 algolia，本地编译完上传即可
$env:http_proxy="127.0.0.1:7890"
$env:https_proxy="127.0.0.1:7890"
python3 algolia.py
# 上传网站文件
git add .
git commit -m "fix: complete post cs61a-stage2"
git push -u origin main

```

## 创建文章

```bash
python3 newpost.py xxx [-e] # e if both English
```
