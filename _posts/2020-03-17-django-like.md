---
layout: post
title: "[Python] 장고 블로그 좋아요 기능"
categories: [doc]
tags: [python, django]
comments: true
---

지난번에는 댓글 기능을 구현해보았는데, 하다보니 좋아요 기능도 개발해보고 싶어져서 만들어봤습니다. 사실 좋아요와 댓글 기능을 개발해보면서 느낀 점이 꽤 많습니다. 원래 저희 스터디의 메인 주제가 SQL이었던만큼, 웹사이트를 개발해나가는 과정에서 데이터베이스 쪽에 관심을 많이 가졌어요. 각 엔터티 간의 관계는 어떠한지, 어느 정도의 정규성을 만족해야 효율적으로 자원을 사용할 수 있을지 등의 문제들을 실제로 접해보고 고민해보는 시간이 되었던 것 같습니다. 책으로 볼 때는 그렇구나 하면서 넘어갔던 부분인데, 역시 직접 만들어보니 데이터베이스 설계가 생각보다 만만한 작업이 아니라는 것을 깨닫게 되었습니다. 물론 ORM이라는 것이 최대한 데이터베이스를 감추기 위한 개념인 것 같기도 하고, 실제 SQL 문장은 한 줄도 작성하지 않았지만 추상적으로나마 데이터베이스를 경험해본 것 같네요. 잡설이 길었는데, 본론으로 들어가보도록 하겠습니다.

## 1. models.py

댓글 기능을 구현할 때, 댓글 모델을 먼저 만들었듯이 이번에도 좋아요 모델을 먼저 만들어두고 시작했습니다. 좋아요는 게시물과 유저의 쌍이 존재하는가/아닌가의 진리값으로 나타낼 수 있다고 생각했고, 이를 그대로 테이블에 구현했습니다. 즉 `Post`, `User`에 ForeignKey를 걸어주고, 두 필드가 UniqueKey를 이루도록 하였습니다. 정렬은 댓글과 마찬가지로 시간 역순으로 설정해주었습니다. `Post` 모델에는 좋아요 갯수를 반환하는 메소드를 지정하고 속성으로 만들어줍니다.

```python
class Post(models.Model):
	...
	@property
	def like_count(self):
		return self.like_set.count()

class Like(models.Model):
	post = models.ForeignKey(Post, on_delete=models.CASCADE)
	user = models.ForeignKey(User, on_delete=models.CASCADE)
	create_dt = models.DateTimeField("CREATE DATE", auto_now_add=True, null=True)
	
	class Meta:
	    unique_together = (("post", "user"),)
	    ordering = ['-create_dt']
```

## 2. views.py

댓글과 좋아요의 처리는 하나의 뷰에서 모두 담당하도록 만들었습니다. 이를 위해서는 먼저 포스트 요청으로 들어온 데이터가 댓글에 의한 것인지 좋아요에 의한 것인지를 구분해주어야 합니다. 간단하게 조건문을 사용하여 처리해주었습니다. 템플릿에서 넘어온 포스트 요청에 `comment`라는 이름 값이 존재하는지를 판단하여 참이라면 댓글에 의한 요청으로 처리하고, 아니라면 좋아요에 의한 요청으로 처리합니다. 이후 포스트 객체를 얻어와서 `get_or_create` 메소드로 해당 유저의 아이디로 좋아요 레코드를 생성해봅니다. 만약 레코드가 새로 생성되지 않았다면 이미 해당 유저의 좋아요 레코드가 존재하는 것이므로 좋아요를 취소한 것으로 간주하여 해당 레코드를 삭제해줍니다. 작업이 끝아면 해당 포스트의 url로 리다이렉트 시켜줍니다.

```python
def post(self, request, *args, **kwargs):
	if 'comment' in self.request.POST:
		... # 생략

	else:
		post = self.get_object()
		post_like, post_like_created = post.like_set.get_or_create(user=request.user)
		if not post_like_created:
			post_like.delete()
		return redirect(post.get_absolute_url())
```

## 3. 템플릿 파일

폰트어썸 아이콘을 활용하여 현재 유저가 좋아요를 표시하지 않는 경우에는 빈 하트가 출력되고, 이미 좋아요를 표시한 경우에는 꽉 찬 하트가 표시되도록 만들어주었습니다. 이후 몇 명의 사용자가 좋아요를 표시했는지, 어떤 사용자가 좋아요를 표시했는지 출력해주도록 합니다. 

```
{% raw %}
<form action="." method="post">{% csrf_token %}
	<button class='btn'>
		{% if user_likes %}
		<i class="fas fa-heart" style="color:red; font-size: 17px;"></i>
		{% else %}
		<i class="far fa-heart" style="color:red; font-size: 17px;"></i>
		{% endif %}
	</button>
	<span class="small">
		{% if user_likes %}
		{{ user }} 님 외 {{ object.like_count }}명이 이 게시물을 좋아합니다
		{% else %}
		{{ object.like_set.first.user }} 님 외 {{ object.like_count }}명이 이 게시물을 좋아합니다
		{% endif %}
	</span>
</form>
{% endraw %}
```

## 결과물

완성된 결과물은 대충 다음과 같습니다. 

![](/assets/img/docs/django-like.png)