---
layout: post
title: "[Python] Django 개발의 기초: MTV"
categories: [doc]
tags: [python, django]
comments: true
---

장고의 MTV 방식은 웹 페이지를 세 가지 영역으로 나눠서 개발하는 방식입니다. 크게 테이블을 정의하는 모델 영역, 사용자가 보게 될 화면의 모습을 정의하는 템플릿 영역, 애플리케이션의 제어 흐름 및 처리 로직을 정의하는 영역으로 구분해서 개발을 진행합니다. 따라서 처음 장고 프로젝트/앱을 생성하면 모델은 `models.py` 파일에, 템플릿은 `templates/*.html` 파일에, 뷰는 `views.py` 파일에 작성할 수 있도록 뼈대가 만들어집니다. 물론 장고에는 MTV 이외에도 url 패턴을 정의하는 `urls.py`, 폼을 정의하는 `forms.py` 등 중요한 파일들이 많으며, 솔직히 웹개발을 처음 접하는 입장으로써 큰 그림을 보기가 상당히 어렵습니다. 하지만 이러한 부분들은 차차 공부해나가기로 하고, 우선은 MTV를 중심으로 살펴봅시다. 

- Model: 테이블을 정의(`models.py`)
- Template: 사용자가 보게 될 화면의 모습을 정의(`templates/*.html`)
- View: 애플리케이션의 제어 흐름 및 처리 로직을 정의(`views.py`)

## 1. 모델

위에서도 언급했듯 `models.py`는 테이블을 정의하는 파일입니다. 예를 들어 하나의 블로그 앱을 구현한다고 가정해봅시다. 그러면 블로그의 게시물을 저장할 데이터베이스와 테이블이 필요할 것이고, 각각의 게시물이 하나의 레코드에 해당할 것입니다. 하지만 장고에서는 직접 SQL을 작성하여 데이터베이스와 테이블을 관리하지 않아도 됩니다. 장고의 특징 중 하나는 데이터베이스 처리에 ORM<sup>Object Relation Mapping</sup> 기법을 사용한다는 점입니다. 즉 테이블을 하나의 클래스로 정의하고 테이블의 컬럼을 클래스의 속성으로 매핑합니다. 테이블 클래스는 `django.db.models.Model` 클래스를 상속받아 정의하며, 각 클래스의 변수 타입도 장고에서 미리 정의한 필드 클래스를 사용합니다. 이후 테이블에 대한 CRUD(Create, Read, Update, Delete) 기능을 클래스 객체에 대해 수행하면, 장고가 내부적으로 데이터베이스에 반영합니다. 이를 위해서 장고는 1.7 버전부터 마이그레이션 개념을 도입했습니다. 마이그레이션이란 테이블 및 필드의 생성, 삭제, 변경 등과 같이 데이터베이스에 대한 변경 사항을 알려주는 정보로, 장고는 이런 마이그레이션 정보를 이용해 변경 사항을 실제 데이터베이스에 반영하는 명령어들을 제공합니다. 아래는 블로그의 게시물을 클래스로 구현한 예시입니다.

```python
from django.db import models

class Post(models.Model):
	title = models.CharField('TITLE', max_length=50)
	slug = models.SlugField('SLUG', max_length=50, help_text = 'one word for title alias', allow_unicode=True)
	description = models.CharField('DESCRIPTION', max_length=100, help_text = 'simple description')
	content = models.TextField('CONTENT')
	create_dt = models.DateTimeField('CREATE DATE', auto_now_add = True)
	modify_dt = models.DateTimeField('MODIFY DATE', auto_now = True)

	class Meta:
		verbose_name = 'post'
		verbose_name_plural = 'posts'
		db_table = 'blog_posts'
		ordering = ('-modify_dt',)

	def __str__(self):
		return self.title
```

SQL의 테이블 정의와 굉장히 유사하기 때문에 이것을 파이썬 클래스로 구현하는 방법만 익혀두면 됩니다. 문자열, 슬러그, 날짜 등 장고에 미리 정의된 필드 클래스들과 제약조건들에 주의합시다. 컬럼을 제외한 기타 속성들은 `Meta` 클래스 안에 다시 정의합니다. 메타 클래스 안에서 테이터베이스에 저장될 테이블 이름, 레코드를 읽어올 때의 정렬 순서 등을 지정해줄 수 있습니다. 

## 2. 템플릿

모든 장고의 구성요소들이 그러하지만, 뷰와 템플릿은 특히 떼어놓고 설명할 수 없는 영역입니다. 우선 템플릿은 말 그대로 하나의 웹 페이지가 갖는 양식이며, 장고의 템플릿 태그(`{%%}`, `{{}}`)를 포함하는 `html` 파일들입니다. 장고는 사용자의 요청에 따라 템플릿 파일 안에 적절한 컨텍스트 변수들을 대입하여 완성된 문서를 사용자에게 보여줍니다. 아래는 블로그 게시물 아카이브를 보여주는 템플릿 예시입니다.

```
{% block content %}
<h3>Post Archives until {% now "N d, Y"%}</h3>
<div class="hr"></div>

{% for date in date_list %}
<a href="{% url 'blog:post_archive_year' date|date:'Y' %}{% endurl %}" class="btn btn-outline-primary btn-sm mx-1">{{ date|date:"Y" }}</a>
{% endfor %}

<br><br>

<div>
	<ul>
		{% for post in object_list %}
		<li><a href="{{ post.get_absolute_url }}">{{ post }}</a><small>&emsp;{{ post.modify_dt|date:"M d, Y" }}</small></li>		
		{% endfor %}
	</ul>
</div>
{% endblock %}
```

장고의 템플릿 태그 안에 들어있는 `date_list`, `object_list` 등이 컨텍스트 변수입니다. 이러한 컨텍스트 변수들이 어떤 값들을 가리키는지는, 해당 템플릿과 연결된 뷰에 의해서 결정됩니다. 즉 사용자의 요청이 들어오면 뷰가 적절한 처리를 해서 템플릿에 컨텍스트 변수들을 넘겨주고, 템플릿에 컨텍스트 변수들이 채워넣어져 최종적으로 사용자가 보는 하나의 문서가 완성되는 것입니다. 처음에는 이러한 연결관계를 파악하는 것이 정말 어려웠는데(물론 지금도 어렵습니다), 하다보니 점점 장고의 구조에 익숙해지는 것 같습니다.

## 3. 뷰

뷰는 웹 요청을 받아서 최종 응답 데이터를 웹 클라이언트로 반환하는 함수 혹은 클래스입니다(둘 중 무엇으로 작성하든 상관은 없습니다). 사용자로부터 요청이 들어오면, 뷰는 데이터베이스로부터 적절한 데이터를 검색, 가공하여 알맞은 형식으로 템플릿에 전달하고, 사용자는 템플릿을 통해 요청에 대한 결과를 확인하게 되는 것입니다. 장고의 놀라운 점은 이미 사용자의 요청에 대한 수많은 처리 로직을 구현해놓은 제네릭 뷰를 제공한다는 점입니다. 대표적으로 조건에 맞는 레코드 목록을 전달하는 `ListView` 레코드에 대한 구체적 정보를 전달하는 `DetailView` 등이 있습니다. 다음은 `Post` 테이블의 레코드 목록을 전달하는 `ListView`의 예시입니다.

```python
class PostLV(ListView):
	model = Post
	template_name = 'blog/post_all.html'
	context_object_name = 'posts'
	paginate_by = 2
```

이 뷰는 `Post` 모델, 그리고 `blog/post_all.html` 이라는 템플릿과 연결되어 있습니다. `Post` 모델(테이블이라고 생각해주시면 됩니다)로부터 적절한 데이터를 검색하여 이들의 목록을 `posts`라는 변수명으로 `blog/post_all.html` 템플릿에 전달합니다. 그러면 해당 템플릿에서는 장고 템플릿 태그를 사용하여 전달받는 객체 목록에 `posts` 라는 컨텍스트 변수명으로 접근할 수 있게 되는 것입니다. 추가적으로 URLconf라는 용어는 URL과 뷰를 매핑해주는 `urls.py` 파일을 말합니다. URL 패턴별로 이름을 지정할 수 있고, 패턴 그룹(예를 들면 앱)에 대해 이름공간을 지정할 수도 있습니다. 즉 URL에 대한 요청이 들어오면, `urls.py` 패턴에 등록된 url일 경우 장고는 적절한 뷰를 찾아 요청에 대한 작업을 수행하여 그 결과를 사용자에게 반환합니다.


### 참고문헌

- 김석훈, 파이썬 웹 프로그래밍(실전편)