---
layout: post
title: "[Python] 장고 블로그 댓글 기능"
categories: [doc]
tags: [python, django]
comments: true
---

학회 심화 스터디 프로젝트로 웹사이트 개발을 진행하는 중입니다. 다른 학회원들 두 명과 함께 파트를 나누어 애플리케이션 개발을 하기로 했는데, 제가 맡은 부분은 블로그 앱입니다. 운이 좋게도 가장 간단한 애플리케이션에 당첨이 되었네요. 아무튼 기본적인 포스팅 기능들은 거의 개발이 되었고, 댓글 기능을 구현해보는 중입니다. 지금 참고하는 책에는 DISQUS를 활용하는 방법만 설명이 되어있는데, DISQUS는 이미 이 블로그에서 활용을 하고 있어서 이번에는 자체적으로 댓글 기능을 개발해서 써보려고 합니다. 인터넷에 돌아다니는 소스들은 거의 함수형 뷰로 작성되어있는데, 지금 참고하는 책의 저자는 클래스형 뷰의 장점을 굉장히 강조하셔서 저는 클래스형 뷰로 짜보는 중입니다. 아직 완벽하지는 않지만 대충 기능은 구현이 되어서, 코드를 정리해봅니다. 해보고 나니 간단한 기능인데 나름대로 애를 많이 먹어서 정리해둡니다...ㅎㅎ 인증 기능 구현이 끝나면 인증 관련 내용까지 추가해서 업데이트 하겠습니다.

## 1. models.py

먼저 댓글을 저장할 모델을 만들어주어야 합니다. 댓글과 포스팅이 N:1의 관계이므로, `Comment` 모델에서 `Post` 모델로 ForeignKey를 걸어줍니다. 아직 회원가입 등 인증 기능을 개발해놓지 않아서 댓글 내용을 저장할 필드만 생성하였습니다.

```python
class Post(models.Model):
	...

class Comment(models.Model):
	post = models.ForeignKey(Post, on_delete = models.CASCADE)
	content = models.CharField('COMMENT', max_length=200)
	def __str__(self):
		return self.content
```

## 2. forms.py

다음으로 사용자의 입력을 받을 폼 클래스를 생성합니다. `django.forms.ModelForm`을 상속하여 폼 클래스를 만들었습니다. 

```python
from blog.models import Comment
from django import forms

class CommentForm(forms.ModelForm):
	content = forms.CharField(widget=forms.Textarea, label='')
	class Meta:
		model = Comment
		fields = ('content',)
```

## 3. views.py

댓글 처리는 포스트의 내용을 보여주는 디테일 뷰 안에서 한꺼번에 담당하도록 만들었습니다. 먼저 `get_context_data` 메소드를 오버라이딩하여 컨텍스트 데이터에 댓글 폼을 담아서 템플릿에 넘겨줍니다. 다음으로 `post` 메소드를 오버라이딩합니다. 즉 클라이언트의 요청이 포스트 방식일 경우, 제출된 데이터 양식의 유효성 검증을 하고, 통과하면 모델을 갱신하고 현재 페이지로 리다이렉트 해줍니다.

```python
from django.http import HttpResponseRedirect
from djnago.shortcuts import render 

class PostDV(DetailView):
	model = Post

	def get_context_data(self, **kwargs):
		context = super().get_context_data(**kwargs)
		context['form'] = CommentForm(auto_id=False)
		return context

	def post(self, request, *args, **kwargs):
		form = CommentForm(request.POST, auto_id=False)
		if form.is_valid():
			post = self.get_object()
			comment = form.save(commit=False)
			comment.post = post
			comment.save()
			return HttpResponseRedirect(reverse('blog:post_detail', args=[post.slug]))
		else:
			return render(request, self.template_name, {'form':form})
```

## 4. template file

마지막으로 템플릿 파일에 댓글 폼과 제출 버튼을 표시하고, 해당 게시물에 달린 댓글 목록을 출력해줍니다. 

```
{% raw %}
<h4>Comments</h4>
<div class="hr"></div>

<div>
	<form action="." method="post">{% csrf_token %}
		{{ form.as_p }}
		<input type="submit" value="Comment">
	</form>
	
	<br><br>

	<ul>
		{% for comment in object.comment_set.all %}
		<li>{{ comment.content }}</li>
		{% endfor %}
	</ul>
</div>
{% endraw %}
```
