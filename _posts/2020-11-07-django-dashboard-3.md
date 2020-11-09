---
layout: post
title: "[Django] 장고를 활용한 ML 대시보드 (3)"
categories: doc
tags: [python, django, ml]
comments: true
---

이번 포스팅에서는 각 거래건의 디테일을 보여주는 화면을 구성해보겠습니다! 이 부분만 구현하면 이제 핵심적인 기능들은 모두 완성이네요 😀 디자인 감각이 부족해서 저는 이 정도가 최선이었지만 ㅎㅎ 웹 디자인에 감각이 있으신 분들이라면 좀 더 예쁘게 만들어볼 수 있을 것 같습니다. 


![](/assets/img/docs/django-detailview.png)

## 1. core/templates/ui-detail.html

디테일 화면은 적절한 뷰를 찾지 못해 직접 구성했습니니다. 총 세 개의 `<row>` 로 구성된 간단한 화면입니다. 가장 위에는 제목을 달아주었고, 둘째 행에는 거래ID, 거래시각, 거래금액 등의 정보를 카드로 보여줍니다. 마지막 행에서는 SHAP의 force plot 을 표시하겠습니다. SHAP 플롯을 뽑아서 보여주는 것 외에는 특별한 것이 없습니다. SHAP 플롯은 뷰에서 그린 후 버퍼에 png로 저장하고, 해당 데이터의 URI를 톔플릿으로 넘겨주는 방식입니다. 사실 이 작동 방식을 완전히 이해한 것은 아니라서 나중에 더 공부를 해봐야 할 것 같기는 합니다 ..ㅎㅎ

```html
<div class="row">
    <div class="col-12">
        <div class="card card-chart">
            <div class="card-header ">
                <div class="row">
                    <div class="col-sm-6 text-left">
                        <h5 class="card-category">Transaction details</h5>
                        <h2 class="card-title">거래건 상세정보</h2>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>


<div class="row">
    <div class="col-3">
        <div class="card">
            <div class="card-body">
                <div class="row">
                    <div class="col">
                        <h4 class="card-title text-uppercase text-danger mb-0">ID</h4>
                        <h2 class="card-text mb-0 mt-3 text-white">{{ object.id }}</h2>
                    </div>
                    <div class="col-auto">
                        <h1 class="card-title"><i class="tim-icons icon-badge text-danger"></i></h1>
                    </div>
                </div>
            </div>
        </div>
    </div>
    ... 생략 ...
</div>

<div class="row">
    <div class="col">
        <div class="card card-chart">
            <div class="card-header">
                <h4 class="card-title text-danger">SHAP FORCE PLOT</h4>
            </div>
            <div class="card-body">
                <img src="data:image/png;base64,{{ force_plot_uri|safe }}" width=100% height=100%>
            </div>
        </div>
    </div>
</div>
```

## 2. app/views.py

뷰에서는 실제로 SHAP 플롯을 생성합니다. 시리얼라이저를 통해 장고 오브젝트를 파이썬 네이티브 자료형으로 변환합니다. 이렇게 만들어진 한 줄의 데이터를 통해 SHAP value를 구한 후 `matplotlib` 플롯을 생성합니다. 이 결과를 `png` 포맷으로 버퍼에 저장합니다. `bbox-inches=tight` 를 설정해주지 않으면 어노테이션이 누락됩니다! 만들어진 플롯을 바이트 인코딩하고 URI를 생성하여 컨텍스트에 전달합니다. 이 방법으로 SHAP 뿐만 아니라 `matplotlib` 플롯들 역시 웹페이지에 띄울 수 있으니 기억해두면 좋을 것 같습니다 다음으로 `segment` 역시 컨텍스트에 추가해서 반환해주면 끝입니다. 

```python
import pandas as pd
import numpy as np
import shap
import io
import base64
import urllib
import matplotlib.pyplot as plt

...생략...

class TransactionDetailView(DetailView):
    template_name = "ui-detail.html"
    model = CardTransaction

    def get_context_data(self, **kwargs):
        dropcols = ["id", "Time", "pred"]
        context = super().get_context_data(**kwargs)
        predictors = pd.Series(CardTransactionSerializer(self.object).data).drop(dropcols)
        X = predictors.values.reshape((1,-1))

        expected_value = PredictionConfig.explainer.expected_value
        shap_values = PredictionConfig.explainer.shap_values(X)

        fplot = shap.force_plot(
            np.around(expected_value[1], decimals=3),
            np.around(shap_values[1], decimals=3),
            np.around(X, decimals=3),
            feature_names=predictors.index,
            matplotlib=True, show=False
        )

        buffer = io.BytesIO()
        fplot.savefig(buffer, bbox_inches="tight", format="png")
        buffer.seek(0)
        string = base64.b64encode(buffer.read())
        uri = urllib.parse.quote(string)
        context["force_plot_uri"] = uri
        context["segment"] = self.request.path

        return context
```

`TransactionDetailRedirectionView` 는 사이드바에서 디테일 메뉴를 클릭했을 때 최신 거래건으로 연결되도록 리다이렉션하는 뷰입니다. 템플릿에 일일이 최신 데이터의 아이디를 전달하기보다는 리다이렉션을 하는 편이 나을 것 같습니다.

```python
class TransactionDetailRedirectionView(View):
    def get(self, request):
        latest = CardTransaction.objects.latest("Time")
        dst = f"tr-detail/{latest.id}"
        return redirect(dst)
```

## 3. app/urls.py

만들어진 페이지의 URL을 등록해줍니다.

```python
urlpatterns = [
	... 생략 ...
	path('tr-detail', views.TransactionDetailRedirectionView.as_view()),
	path('tr-detail/<int:pk>', views.TransactionDetailView.as_view(), name="tr-detail")
]
```

## 4. core/templates/includes/sidebar.html

만들어진 페이지를 사이드바에도 추가해줍니다. 위에서 `/tr-detail` 주소에 리다이렉션을 구현해놓았기 때문에 해당 메뉴를 클릭하면 최신 거래건에 대한 디테일 페이지로 넘어가게 됩니다.

```html
{% raw %}
<li class="{% if 'tr-detail' in segment %} active {% endif %}">
    <a href="/tr-detail">
        <i class="tim-icons icon-notes"></i>
        <p>Predictions</p>
    </a>
</li>
{% endraw %}
```