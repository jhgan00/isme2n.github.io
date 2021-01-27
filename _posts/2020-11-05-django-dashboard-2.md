---
layout: post
title: "[Django] 장고를 활용한 ML 대시보드 (2)"
categories: doc
tags: [python, django, ml]
comments: true
---

지난 포스팅에서는 API 기능을 완성했습니다. 이번에는 저장된 데이터의 목록을 보여주는 화면을 구성해보겠습니다! 사실 이 작업을 하면서 처음 사용해보는 기술들이 많았습니다. 그래서 많이 헤매고 애도 먹었지만 그만큼 완성한 후에 성취감도 컸던 것 같아요 😀

![](/assets/img/docs/django-listview.png)

## 1. core/templates/ui-tables.html

우선 입력된 거래건들을 보여줄 템플릿을 만들어보려고 합니다. `core/templates/ui-tables.html` 을 기반으로 약간의 수정을 거치면 예쁘게 리스트를 보여줄 수 있을 것 같네요 😀 서버에서 전달받은 `object_list` 를 돌면서 거래건의 정보들을 `<td>` 태그 안에 채워줬습니다. 거래건의 모든 내용을 표시할 필요는 없으니 ID, 거래시간, 거래액 그리고 모델의 예측 정보만을 표시해주도록 합시다. 모델의 예측 확률을 프로그레스 바로 표현해주고, 행을 클릭하면 해당 거래건의 디테일뷰로 넘어갈 수 있도록 `<tr>` 태그에 링크를 걸어줍니다. 행 위에 커서가 올라오면 하이라이트가 되도록 css를 추가해주었습니다.

```html
{% raw %}
<div class="card-body">
    <div class="table-responsive" style="overflow: hidden">
        <table class="table table-hover" id="table">
            <thead class="text-primary">
            <tr>
                <th style="width:20%;">ID</th>
                <th style="width:20%;">TIME</th>
                <th style="width:20%;">Amount</th>
                <th class="text-center" style="width:20%;">Prediction</th>
            </tr>
            </thead>
            <tbody id="features">
            {% for object in object_list %}
                <tr style = "cursor:pointer;" onClick = " location.href='/tr-detail/{{ object.id }}' ">
                    <td>{{ object.id }}</td>
                    <td>{{ object.Time }}</td>
                    <td>{{ object.Amount }}</td>
                    <td class="text-center">
                            <span class="mr-2">{{ object.pred }}%</span>
                            <div class="progress">
                                <div class="progress-bar bg-danger" role="progressbar"
                                     aria-valuenow="100" aria-valuemin="0" aria-valuemax="100"
                                     style="width: {{ object.pred }}%;">
                                </div>
                            </div>
                    </td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
        <style type="text/css">
            #table:hover tbody tr:hover td {background: #8965e0;}
        </style>
    </div>
</div>
{% endraw %}
```

테이블 하단에 페이지 이동을 위한 버튼을 생성합니다. 페이지 이동에는 jQuery/ajax를 사용했습니다. ajax는 응답으로 전체 HTML 문서를 받아오는 것이 아니라 필요한 데이터만을 받아서 화면을 갱신할 수 있는 기술이라고 합니다. 그래서 요청이 발생해도 화면 전체를 새로 출력할 필요 없이 필요한 데이터만 업데이트를 하게 되는 원리인 것 같네요! 사실 자바스크립트, jQuery, ajax 등에 대해 정확한 이해를 하고 쓴 것은 아니라 잘 이해가 되지 않는 부분이 많고 코드도 난잡해졌네요 😅 이번에는 구현이 되었다는 사실에 의의를 두고 조금 더 공부를 해봐야 할 것 같습니다. 요점은 페이지 이동 버튼을 누를 때 `tr-list` URL로 포스트 요청을 보내 새로운 데이터를 받아오는 동작입니다. 정상적으로 응답을 받았은 후에는 자바스크립트 코드를 통해 html 문서의 내용을 업데이트해줍니다. 이 작업을 하면서 난생 처음으로 자바스크립트 코드를 만져봤는데 신기하고 흥미로운 경험이었습니다 ㅎㅎ

```html
{% raw %}
<div class="card-footer py-4">
    <ul class="pagination justify-content-end mb-0">
        <li id="btn_first" class="page-item">
            <a class="page-link" href="1" id="btn_first_link"><i class="fas fa-angle-double-left"></i></a>
        </li>

        <li id="btn_prev" class="page-item">
            <a class="page-link" href="" id="btn_prev_link"><i class="fas fa-angle-left"></i></a>
        </li>

        {% for i in paginator.page_range %}
        <li id="btn_{{ i }}" class="page-item"><a href="{{ i }}" class="page-link">{{ i }}</a></li>
        {% endfor %}

        <li id="btn_next" class="page-item">
            <a class="page-link" href="2" id="btn_next_link"><i class="fas fa-angle-right"></i></a>
        </li>

        <li id="btn_end" class="page-item">
            <a class="page-link" href="{{ paginator.num_pages }}" id="btn_end_link"><i class="fas fa-angle-double-right"></i></a>
        </li>
    <script>
        \\( (".page-link").click(function (event) {
            event.preventDefault();
            var page_n =  \\)(this).attr('href');
            var page_end = \\( ('#btn_end_link').attr('href');

            if(page_n == '1'){
                var prev_page = 1;
                var next_page = 2;
            }
            else if(page_n == page_end){
                var prev_page = parseInt(page_n) - 1;
                var next_page = parseInt(page_n);
            }
            else{
                var prev_page = parseInt(page_n) - 1;
                var next_page = parseInt(page_n) + 1;
            }

            // ajax
             \\).ajax({
                type: "POST",
                url: "{% url 'tr-list' %}", // name of url
                data: {
                    page_n: page_n, //page_number
                    csrfmiddlewaretoken: '{{ csrf_token }}',
                },
                success: function (resp) {
                    //loop
                    \\( ("[id^=btn]").removeClass("active")
                     \\)('#btn_' + page_n).addClass("active")
                    \\( ('#btn_prev_link').attr("href", prev_page)
                     \\)('#btn_next_link').attr("href", next_page)
                    \\( ('#features').html('')
                     \\).each(resp.results, function (i, val) {
                        $('#features').append(
                            '<tr style = "cursor:pointer;" onClick = " location.href=\'/tr-detail/' + val.id + '\' ">' +
                                '<td>' + val.id + '</td>' +
                                '<td>' + val.Time + '</td>' +
                                '<td>' + val.Amount + '</td>' +
                                '<td class="text-center">' +
                                    '<span class="mr-2">' + val.pred + '%</span>' +
                                    '<div class="progress">' +
                                        '<div class="progress-bar bg-danger" role="progressbar" ' +
                                             'aria-valuenow="100" aria-valuemin="0" aria-valuemax="100" ' +
                                             'style="width:'  + val.pred + '%;">' +
                                        '</div>' +
                                    '</div>' +
                                '</td>' +
                            '</tr>'
                        )

                    });
                },
                error: function () {
                }
            }); //
        });
       </script>
    </ul>
</div>
{% endraw %}
```


## 2. core/templates/layouts/base.html

```html
<script
  src="https://code.jquery.com/jquery-3.4.1.min.js"
  integrity="sha256-CSXorXvZcTkaix6Yvo6HppcZGetbYMGWSFlBw8HfCJo="
  crossorigin="anonymous">
</script>
```

베이스 템플릿의 `<head>` 태그 안쪽에 jQuery 라이브러리를 포함시켜줍니다.

## 3. app/views.py

장고 제네릭 뷰의 리스트 뷰를 상속합니다. 템플릿 파일과 모델을 지정해주고, 페이지네이션도 원하는 만큼 걸어줍니다. 모든 거래건을 보여줄 필요는 없으니 의심스러운 거래의 정보만 보여주도록 합시다. `get_queryset` 메소드를 오버라이딩하여 `pred` 값이 50을 넘기는 경우만 반환합니다. 다음으로 ajax 요청을 처리하기 위해 `post` 메소드를 오버라이딩합니다. 요청에서 `page_n` 변수를 꺼내온 후 해당 페이지의 데이터를 JSON 응답으로 돌려줍니다. 마지막으로 `get_context_data` 메소드를 오버라이딩하여 사이드바 템플릿에서 사용할 `segment` 변수를 컨텍스트에 추가해줍니다.

```python
class SuspiciousTransactionListView(ListView):
    template_name = "ui-tables.html"
    model = CardTransaction
    paginate_by = 10

    def get_queryset(self):
        return CardTransaction.objects.filter(pred__gt=50)

    def post(self, request, **kwargs):
        paginator = self.get_paginator(self.get_queryset(), 10)
        page_n = self.request.POST.get("page_n")
        results = list(paginator.page(page_n).object_list.values('id', 'Time', 'Amount', 'pred'))
        return JsonResponse({"results":results})

    def get_context_data(self, *, object_list=None, **kwargs):
        context = super().get_context_data(**kwargs)
        context["segment"] = self.request.path
        return context
```

## 4. app/urls.py

```python
urlpatterns = [
    ... 생략 ...
    path('tr-list', views.SuspiciousTransactionListView.as_view(), name="tr-list"),
]
```

리스트뷰의 URL을 추가합니다.

### 5. core/templates/includes/sidebar.html

```html
{% raw %}
<li class="{% if 'tr-list' in segment %} active {% endif %}">
    <a href="/tr-list">
        <i class="tim-icons icon-coins"></i>
        <p>Transactions</p>
    </a>
</li>
{% endraw %}
```

작업한 페이지를 사이드바에 추가해주면 끝입니다! 이제 각 거래건의 디테일을 보여주는 페이지만 구현하면 핵심적인 기능은 완성입니다 😀