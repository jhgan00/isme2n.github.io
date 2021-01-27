---
layout: post
title: "[Django] 장고를 활용한 ML 대시보드 (1)"
categories: doc
tags: [python, django, ml]
comments: true
---

요즘은 미래에셋의 2020 금융 빅데이터 페스티벌을 준비하는 중입니다. 보험금 청구건 분류 주제에서 예선을 나름 좋은 성적으로 통과하고, 지금은 본선 보고서 제출까지 마친 상태에요. 이 공모전을 준비하면서 장고를 활용한 대시보드를 제작했는데, 여유가 생긴 김에 관련 코드를 정리해보려고 합니다. 사실 글 자체는 길지 않은데 자잘한 코드가 많아져서 부득이하게 포스팅을 나눴습니다 😅 

공모전에 주어진 데이터를 그대로 사용할 수는 없으니 캐글의 [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) 데이터를 사용합니다. 목표는 훈련된 모델을 장고 서버에 올린 후 POST 요청이 들어오면 실시간으로 모델의 예측 및 SHAP 해석 결과를 보여주는 것입니다. 구체적으로 1) POST 요청예 대해 예측을 실행하는 API 만들기 2) 저장된 데이터의 목록 보여주기 3) 개별 예측결과의 디테일(SHAP) 보여주기 세 부분으로 구성됩니다. 각각 장고 `APIView`, `ListView`, `DetailView` 를 통해 구현할 예정입니다. 사실 다른 코드를 찾아보면 클래스 기반 뷰보다는 함수형 뷰를 많이 사용하시는 것 같은데, 저는 아직 함수형 뷰가 익숙하지 않은 것 같아요 ㅎㅎ



## 1. 템플릿 선택


<div class="github-card" data-github="app-generator/django-admin-dashboards" data-width="500" data-height="200" data-theme="default"></div>
<script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>

대시보드 디자인을 처음부터 전부 다 하기에는 능력이 부족해서 무료 템플릿들을 찾아봤는데, 개인적으로는 위 저장소에서 제공하는 템플릿들이 마음에 들었습니다. 이번에는 [블랙 테마](https://github.com/app-generator/django-dashboard-black)를 사용해보도록 할게요! 저장소를 클론한 후 초기 설정을 하고, 서버를 실행한 후 로그인 테스트를 해봅니다. 

```zsh
git clone https://github.com/app-generator/django-dashboard-black.git
cd django-dashboard-black
pip install -r requirements.txt
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```


![](/assets/img/docs/django-login.png)



## 2. 예측모델 준비

장고 서버에 머신러닝 모델을 올리는 것이 주된 목표이므로 모델은 간단하게만 만들어줍니다. 카드 거래는 시간 순서로 생성되는 데이터이기 때문에 시간에 따라서 임의로 훈련, 검증, 테스트 데이터셋을 구분했습니다. 훈련된 모델은 서버에서 활용하기 위해 장고 프로젝트의 `models` 디렉토리에 파일로 저장했습니다. 


```python
import os
import warnings
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, recall_score

warnings.filterwarnings("ignore", category=UserWarning)

# 판다스 데이터프레임 -> lightgbm dataset
def lgb_dataset(df: pd.DataFrame) -> lgb.Dataset:
    X = df.drop(["Time", "Class"], axis=1, errors="ignore")
    y = df.Class
    return lgb.Dataset(data=X, label=y)


# 1. 데이터셋
data = pd.read_csv("data/creditcard.csv")
trainset = data.query("Time <= 100000").drop("Time", axis=1)
validset = data.query("Time > 100000 & Time <= 120000").drop("Time", axis=1)
testset = data.query("Time > 120000")


# 2. lightgbm 파라미터
params = dict(
    learning_rate = 0.005,
    objective = "binary",
    num_iterations = 1000,
    early_stopping_rounds = 100,
    verbosity=-1
)

# 3. 모델 훈련
model = lgb.train(
    params = params,
    train_set = lgb_dataset(trainset),
    valid_sets = lgb_dataset(validset),
    verbose_eval=100
    
)

# 4. 성능 측정
cutoff = 0.5
true = testset.Class.values
pred = (model.predict(testset.drop(["Time", "Class"], axis=1)) > cutoff).astype(int)
f1 = round(f1_score(testset.Class.values, pred), 4)
recall = round(recall_score(testset.Class.values, pred), 4)

print("="*50)
print(f"F1 SCORE: {f1}")
print(f"RECALL: {recall}")
print("="*50)

# 5. 모델 저장
if not os.path.isdir("models"):
    os.mkdir("models")
model.save_model("models/classifier", num_iteration = model.best_iteration)
```	


```
\\(  python train.py
Training until validation scores don't improve for 100 rounds
[100]   valid_0's binary_logloss: 0.00505178
[200]   valid_0's binary_logloss: 0.00394079
[300]   valid_0's binary_logloss: 0.00350547
[400]   valid_0's binary_logloss: 0.00330293
[500]   valid_0's binary_logloss: 0.00314134
[600]   valid_0's binary_logloss: 0.00309302
[700]   valid_0's binary_logloss: 0.00313283
Early stopping, best iteration is:
[626]   valid_0's binary_logloss: 0.00308583

============================================================
F1 SCORE: 0.7797
RECALL: 0.697
============================================================
```




## 3. REST API

데이터 입력은 폼으로 일일이 받기보다 API를 통햬 받으려고 합니다. 즉 요청이 들어오면 API 뷰에서 예측을 실행하고, 데이터베이스에 예측 결과를 등록힌 후 디테일뷰를 통해서 분석 결과를 확인할 수 있게 하겠습니다. 사실 이 방법이 최선인지는 모르겠지만, 어차피 폼을 통해 일일이 데이터를 입력할 일은 없을 것 같아 API를 선택했습니다.

### 3.1. core/settings.py

```python
INSTALLED_APPS = [
	... 생략 ...
    'rest_framework'
]
```

먼저 `core/settings.py` 파일의 `INSTALLED_APPS`에 `rest_framework`를 추가합니다.

### 3.2. app/models.py

예측 결과를 저장할 장고 모델을 만듭니다. 기존의 데이터에서 정답 컬럼 `Class`를 제외하고 모델의 예측 확률 컬럼을 추가했습니다. 모델 생성 후 마이그레이션을 실행해줍니다.

```python
class CardTransaction(Model):
	Time = models.IntegerField()
	... 생략 ...
	pred = models.FloatField()
```

```zsh
python manage.py makemigrations
python manage.py migrate
```

### 3.3. app/admin.py

```python
from django.contrib import admin
from app.models import CardTransaction


@admin.register(CardTransaction)
class CardTransactionAdmin(admin.ModelAdmin):
    list_display = ["Time", "Amount", "pred"]
```

테이블을 관리 페이지에서 확인할 수 있도록 등록해줍니다.

### 3.4. app/serializers.py

유효성 검증과 레코드 생성을 담당할 시리얼라이저를 만듭니다.

```python
from rest_framework import serializers
from app.models import CardTransaction

class CardTransactionSerializer(serializers.ModelSerializer):
    class Meta:
        model = CardTransaction
        fields = '__all__'
```

### 3.5. app/views.py

실제로 요청을 받아서 처리할 뷰를 만듭니다. SHAP value는 굳이 저장하지 않고 디테일뷰를 확인하는 경우에만 실시간으로 분석해서 제공하겠습니다. `TreeExplainer` 는 SHAP 해석기 중에서도 빠른 편이라 실시간으로 분석을 실행하더라도 큰 문제가 없을 것으로 보입니다.

```python
class PredictView(APIView):

    def post(self, request):
        data = request.data.dict()

        # 예측: SHAP VALUE는 DB에 저장하지 않고 디테일 확인하는 경우에만 분석해서 제공
        predictors = pd.Series(data).astype(float).drop("Time").values
        pred = PredictionConfig.classifier.predict(predictors.reshape((1,-1)))
        data["pred"] = pred

        # 유효성 검사 & 저장
        serializer = CardTransactionSerializer(data=data)
        assert serializer.is_valid(), serializer.errors
        serializer.save()

        return Response(status=200, data={"pred": pred})
```

### 3.6. app/urls.py

뷰에 대응하는 url을 추가합니다.

```python
from app import views
urlpatterns = [
    ... 생략 ...
    path('api', views.PredictView.as_view(), name="api")
]
```


### 3.7. 테스트 및 데이터 입력

만들어진 API 뷰를 테스트해봅시다. 서버를 실행한 후 몇 개의 테스트 데이터를 골라서 API로 던져봅니다. 

```python
import pandas as pd
import requests

data = pd.read_csv("../data/creditcard.csv")
testset = data.query("Time > 120000").head(10).drop("Class", axis=1)
for i in range(10):
    data = testset.iloc[i].to_dict()
    response = requests.post("http://127.0.0.1:8000/api", data=data)
    print("Probability", response.json()["pred"][0])
```

```zsh
 \\) python apitest.py
Probability 0.01
Probability 0.01
Probability 0.01
Probability 0.01
Probability 0.01
Probability 0.01
Probability 0.01
Probability 0.01
Probability 0.01
Probability 0.01
```

정상적인 응답이 돌아오는 것을 확인한 후 DB 쉘을 열어서 쿼리를 날려봅니다. 역시 데이터가 정상적으로 등록된 것을 확인할 수 있습니다. 

```zsh
python manage.py dbshell
sqlite> SELECT TIME, AMOUNT, PRED FROM APP_CARDTRANSACTION;
12001|29.4|0.01
12001|29.4|0.01
12003|59.0|0.01
12003|7.76|0.01
12004|0.89|0.01
12005|9.77|0.01
12006|9.8|0.01
12008|2.5|0.01
12012|9.8|0.01
12012|24.5|0.01
12017|8.0|0.01
sqlite> DELETE FROM APP_CARDTRANSACTION;
```

API가 잘 동작하는 것을 확인했으므로 시간을 절약하기 위해 한꺼번에 DB에 입력하겠습니다 😅 `to_sql` 메소드에서 테이블 이름을 대문자로 입력하는 경우에 `append` 파라미터가 제대로 작동하지 않는 것 같습니다. 테이블 이름에는 소문자를 사용해주세요!

```python
import sqlite3
import pandas as pd
from app.config import PredictionConfig

testset = pd.read_csv("../data/creditcard.csv").query("Time > 120000").drop("Class",axis=1)
pred = PredictionConfig.classifier.predict(testset.drop("Time", axis=1)) * 100
pred = pred.round(2)
testset = testset.reset_index().assign(pred = pred).rename(columns={"index":"id"})

conn = sqlite3.connect("db.sqlite3")
testset.to_sql(name="app_cardtransaction", con=conn, if_exists="append", index=False)
conn.close()
```