---
layout: post
title: "[Python] 장고 RESTful API"
categories: [doc]
tags: [python, django]
comments: true
---

장고는 RESTful API 개발을 위한 프레임워크 역시 지원합니다. 이 사실을 알게 된 다음부터 언젠가 시도해보려고 미뤄두고 있었는데 이제서야 해보네요 ㅎㅎ 공식 문서와 튜토리얼을 참고해서 일단은 저에게 필요한 내용 위주로 정리하였습니다.

## 1. Intro

기본 장고 패키지와 별도로 `djangorestframework` 패키지를 설치해주어야 합니다. 이후 원하는 위치에 프로젝트와 앱 하나를 만들고 migrate를 실행해줍니다. 저는 exchange 프로젝트 안에 api라는 앱을 생성하였습니다. 최종적으로는 사용자의 요청에 따라 환율 예측값을 반환하는 RESTful API를 구상하고 있는데, 우선은 환율 정보만 담아서 반환해보도록 하려고 합니다. 다음으로 superuser를 생성하고, `settings.py` 파일에 우리가 만든 앱과 `rest_framework`를 추가해줍니다. 필요하다면 다른 설정도 같이 해줍니다.

```bash
$ pip install djangorestframework
$ django-admin startproject exchange
$ cd excange
$ django-admin startapp api
$ python manage.py migrate
$ python manage.py createsuperuser
```

```python
# exchange/settings.py
INSTALLED_APPS = [
    ...생략...
    'rest_framework',
    'api'
]
```

## 2. Model

이제 api 앱에 데이터베이스를 구성해보겠습니다. 환율 정보는 [한국수출입은행](https://www.koreaexim.go.kr/site/program/openapi/openApiView?menuid=001003002002001&apino=2&viewtype=C)에서 API로 제공하고 있습니다. 여기에 들어있는 데이터 중 필요한 것들만 골라 모델의 필드로 지정해줍니다. 아직 `week` 필드는 어떻게 처리할지 결정하지 못해서 null을 허용하였습니다. 모델을 생성한 후 마이그레이션까지 마쳐줍니다.

```python
from django.db import models

class Exchange(models.Model):
    class Meta:
        db_table = 'exchange'
        ordering = ['-date']

    date = models.DateTimeField('DATE')
    cur_nm = models.CharField("CUR_NM", max_length=20)
    cur_unit = models.CharField("CUR_UNIT", max_length=10)
    ttb = models.FloatField("TTB")
    tts = models.FloatField("TTS")
    week = models.CharField("WEEK", null=True, max_length=50)
```

```bash
$ python manage.py makemigrations
$ python manage.py migrate
```

모델이 완성되었으니 모델에 데이터를 채워넣겠습니다. 아래 스크립트를 통해 한국수출입은행 2019년 1월부터 현재까지의 데이터를 받아와 db에 채워넣었습니다. 판다스와 sqlite3의 조합으로 손쉽게 데이터프레임을 db로 옮길 수 있습니다.

```python
import requests
from datetime import datetime
from pandas import json_normalize, to_datetime, date_range, concat
import sqlite3
import numpy as np

key = # 발급받은 api 키를 넣어줍니다
curr_date = str(datetime.now().date())
url = 'https://www.koreaexim.go.kr/site/program/financial/exchangeJSON?'

dates = np.array([x.replace("-", "") for x in date_range("20190101", "20200417").astype(str).values])

result = []

for curr_date in dates:
    query = f'authkey={key}&searchdate={curr_date}&data=AP01'
    uri = url + query
    res = requests.get(uri)
    data = json_normalize(res.json()).assign(date = curr_date)
    result.append(data)
    print(curr_date)

result = concat(result)[['date', 'cur_nm', 'cur_unit', 'ttb', 'tts']]

result = result.assign(
    date = to_datetime(result.date, format='%Y%m%d'),
    cur_unit = result.cur_unit.str.slice(0,3),
    ttb = result.ttb.str.replace(",","").astype("float16"),
    tts = result.tts.str.replace(",","").astype("float16")
)

conn = sqlite3.connect('db.sqlite3');
result.to_sql('exchange', conn, index=False, if_exists='append')
conn.close()
```

## 2. Serializer

> Serializers allow complex data such as querysets and model instances to be converted to native Python datatypes that can then be easily rendered into JSON, XML or other content types. Serializers also provide deserialization, allowing parsed data to be converted back into complex types, after first validating the incoming data.

이제 직렬화기serializer를 만들어줍니다. 장고 REST framework 홈페이지에서는 serializer에 대해 위와 같이 설명하고 있습니다. 기본적으로 serializer는 복잡한 쿼리셋이나 장고 모델 등을 파이썬의 네이티브 데이터 타입으로 변환해주는 기능을 한다고 보면 되겠네요(폼과 유사한 개념이라고 합니다). 먼저 모델을 네이티브 파이썬으로 변환해준 후, 다시 json이나 xml 등으로 변환하는 흐름인 것 같습니다. 반대로 파싱된 데이터를 다시 장고 모델로 변환하는 것도 가능하다고 합니다. 공식 문서에서는 역직렬화, 저장, 유효성 검증 등까지 다루고 있지만 여기서는 `ModelSerializer`만 사용해보겠습니다. `ModelSerializer`는 장고 모델에 대한 직렬화를 편리하게 제공하는 클래스입니다. 즉 장고 모델의 필드에 대응하는 필드를 가진 직렬화기를 손쉽게 만들 수 있습니다.

```python
from rest_framework.serializers import ModelSerializer
from api.models import Exchange

class ExchangeSerializer(ModelSerializer):
    class Meta:
        model = Exchange
        fields = ['date', 'cur_nm', 'cur_unit','ttb', 'tts', 'week']
```

## 3. View

뷰와 url은 지금까지 봐왔던 장고 코드와 크게 다르지 않았습니다. `rest_framework`에도 마찬가지로 간편한 제네릭 뷰들이 존재하기 때문에 그냥 가져와서 사용하면 됩니다. 만약 쿼리셋을 필터링하고 싶은 경우에는 `get_queryset` 메소드를 오버라이딩해주면 됩니다. 공식 문서에서는 유저 기반 필터링과 쿼리 파라미터 기반 필터링이라는 두 가지 경우를 구분해서 설명하고 있습니다. 제가 하고싶은 일은 특정 통화만 필터링하여 반환하는 것이므로 여기서는 쿼리 파라미터 기반 필터링만 다뤄보겠습니다. 직렬화기와 권한 클래스를 지정한 후 `get_queryset`메소드를 오버라이딩합니다. 요청의 쿼리 파라미터에서 cur_unit 값을 가져와 모델을 필터링해주고, 이 결과를 반환합니다.

```python
from rest_framework.generics import ListAPIView
from rest_framework.permissions import IsAuthenticated
from api.models import Exchange
from api.serializers import ExchangeSerializer

class ExchangeListView(ListAPIView):
    serializer_class = ExchangeSerializer
    permission_classes = [IsAuthenticated] # 로그인한 사용자만 접근 가능

    def get_queryset(self):
        cur_unit = self.request.query_params.get('cur_unit')
        queryset = Exchange.objects.filter(cur_unit = cur_unit)
        return queryset
```

## 4. URL

url 역시 지금까지 했던 방식으로 똑같이 짜주면 됩니다. 먼저 `exchange/urls.py` 파일에 앱과 rest_framework가 자동으로 포함하는 url들을 지정해줍니다.

```python
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('api.urls')),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]
```

다음으로 `api/urls.py` 파일을 생성하고 위에서 만들어둔 뷰와 연결시켜줍니다. 결과적으로 특정 통화에 대한 환율 정보를 받아오려면, "BASE_URL/api/exchange/?cur_unit=USD"와 같이 요청하면 될 것입니다.

```python
from django.urls import path, include
from api.views import ExchangeListView

app_name = 'api'
urlpatterns = [
    path('exchange/', ExchangeListView.as_view())
]
```

일본 옌화의 환율을 조회해보면 다음과 같습니다. 

![](/assets/img/docs/restful.png)