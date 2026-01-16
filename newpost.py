import os
import argparse
import shutil
import subprocess

# python3 blog createPost <name> [-e]
# 创建一篇文章
def newpost(en_name, year, month):
    post_name = en_name
    path = f"posts/{year}/{month}/{post_name}"
    content_path = f"./content/{path}"
    
    # 创建必要的目录
    os.makedirs(content_path, exist_ok=True)
    
    # 通过完整 zsh 命令找到 hugo（兼容 alias）
    result = subprocess.run(
        f"/bin/zsh -i -c 'hugo new {path}/{post_name}.md'",
        shell=True,
        capture_output=True,
        text=True
    ).returncode
    
    # 如果 hugo 命令失败，直接创建文件
    if result != 0:
        with open(f"{content_path}/{post_name}.md", "w") as f:
            f.write("---\n")
            f.write("title: \"\"\n")
            f.write("date: \n")
            f.write("draft: true\n")
            f.write("---\n\n")
    
    os.rename(f"{content_path}/{post_name}.md", f"{content_path}/index.zh-cn.md")
    shutil.copy(f"{content_path}/index.zh-cn.md", f"{content_path}/index.en.md")
    os.makedirs(f"{content_path}/image/", exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("en_name")
    parser.add_argument("year")
    parser.add_argument("month")

    args = parser.parse_args()
    newpost(args.en_name, args.year, args.month)
