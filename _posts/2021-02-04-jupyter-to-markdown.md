---
layout: post
title: "[Python] 주피터 노트북을 지킬 마크다운으로 옮겨보자"
categories: tip
tags: [python]
comments: true
---


논문을 읽으면서 주피터로 내용을 정리하고 블로그에 포스팅하는 일이 종종 있는데, 그때마다 주피터 노트북의 내용을 마크다운으로 옮기는 수작업을 해왔습니다. 앞으로는 논문 공부와 블로그 포스팅을 좀 더 열심히 하기 위해 😅 주피터 노트북을 지킬 마크다운으로 퍼블리시해주는 스크립트를 짜봤습니다. 제가 사용하는 환경에 맞춰서 예외는 거의 고려하지 않고 작성한 코드이니 혹시나 필요하시면 살펴보시고 수정해서 쓰시면 될 것 같습니다! 개인적으로 주피터 노트북 파일은 `_drafts` 에 놓고 작업하기 때문에 각자 사용하시는 디렉토리 구조와 다를 수 있을 것 같습니다!


```python
import argparse
import os
import re
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('fname', type=str, help='notebook filename')  # 변환할 노트북 파일
parser.add_argument('title', type=str, help='post title')  # 포스트 제목
parser.add_argument('category', type=str, help='post category')  # 포스트 카테고리
parser.add_argument('--tags', nargs="+", help='post tags')  # 포스트 태그

args = parser.parse_args()
fname = args.fname.split(".")[0]

# yaml 프론트매터 생성
contents = f"""---
layout: post
title: "{args.title}"
categories: {args.category}
tags: {args.tags}
comments: true
use_math: true
---\n
"""

os.system(f"jupyter nbconvert --to markdown {args.fname}")

with open(f"{fname}.md", "r") as mdfile:
    
    contents += mdfile.read()
    p = re.compile(r"(!\[.*\])\((.*[.]png)\)")  # 이미지 경로 수정
    contents = p.sub(r'\1(/assets/img/docs/\2)', contents)  # 이미지 경로 수정

    
with open(f"{fname}.md", "w") as mdfile:

    mdfile.write(contents)

shutil.move(f"{fname}.md", "../_posts")  # 포스트 디렉토리로 옮기자

if os.path.exists(f"{fname}_files"):  # 만약 이미지가 포함되어있으면
    shutil.move(f"{fname}_files", "../assets/img/docs/")  # 이미지는 지정한 이미지 파일 경로로 옮기자

```



```bash
$ python publish.py 2021-02-04-LSI.ipynb \
"[ML/stat] Indexing by Latent Semantic Analysis" \
docs \
--tags ml stat
```