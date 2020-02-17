---
layout: post
title: "[HeatIsland] Landsat 데이터 다운로드"
categories: project
tags: [heatisland, python]
css: ['typo.css']
---

구글 클라우드 스토리지가 이미 랜샛 데이터를 제공하고 있어서, 이를 통해 필요한 데이터를 쿼리하고 다운로드하는 과정을 프로그램화해보자.

>
Landsat은 인공 위성을 사용하여 지구를 지속적으로 관찰하기 위한 목적에 따라 USGS와 NASA가 공동으로 개발한 프로그램입니다. Landsat 프로그램은 지표면을 우주에서 가장 오랫동안 지속적으로 기록한 대장정으로서, 1972년 Landsat 1 위성부터 시작되었습니다. Landsat 4부터 시작하여 각 위성은 다중 스펙트럼 및 열 계측기를 사용하여 지표면을 2주에 한 번씩 30미터 해상도로 촬영했습니다. 이 컬렉션에는 Landsat 4, 5, 7, 8을 통해 얻은 전체 USGS 기록 자료가 포함되어 있습니다. 여기에는 전체 운영 기간 동안 35년에 걸쳐 400만장 이상의 고유 장면을 촬영한 데이터가 포함되어 있습니다. USGS 및 NASA의 개방 데이터 정책 덕분에 이 데이터세트는 Google Public Cloud Data 프로그램의 일부로 무료 제공됩니다. 즉, Google Cloud의 일부로 누구나 사용할 수 있습니다.

- Landsat 4: 1982 - 1993
- Landsat 5: 1984 - 2013
- Landsat 7: 1999 - 현재
- Landsat 8: 2013 - 현재

|Tier|Description|
|----|-----------|
|Tier 1 (T1) | Data that meets geometric and radiometric quality requirements|
|Tier 2 (T2) | Data that doesn't meet the Tier 1 requirements|
|Real Time (RT) | Data that hasn't yet been evaluated (it takes as much as a month)|

## 0. 패키지 및 설정

필요한 패키지들을 불러오고 구글 어플리케이션을 사용할 수 있는 계정정보를 등록한다.

```python
from google.cloud import storage, bigquery
import os
import pandas as pd
import seaborn as sns
```

```python
# 구글 어플리케이션 계정 등록
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"{os.getcwd()}/dsyonsei.json"
```

## 1. 데이터 쿼리

``google.cloud.bigquery``를 통해 필요한 데이터를 쿼리한다. 조건은 다음과 같다.

- WRS: 116, 34
- cloud_cover: 10% 미만
- date_acquired: 6,7,8 월


```python
# 클라이언트 객체 생성
bigquery_client = bigquery.Client()

# 쿼리문 작성
# 모든 위성에서 wrs좌표와 운량, 계절 조건을 만족하는 영상 선택
QUERY = (
    'SELECT * '
    'FROM `bigquery-public-data.cloud_storage_geo_index.landsat_index` '
    'WHERE wrs_path=116 '
    'AND wrs_row=34 '
    'AND cloud_cover<10 '
    'AND REGEXP_CONTAINS(date_acquired, r"[0-9]{4}-0[678]-[0-9]{2}")'
)

# 쿼리
query_job = bigquery_client.query(QUERY)

# 쿼리 결과를 데이터프레임으로 변환
landsat_df = bigquery.table.RowIterator.to_dataframe(query_job.result())

# date_acquired를 쪼개서 year, month 컬럼 생성
landsat_df = landsat_df.assign(year = [date.split("-")[0] for date in landsat_df['date_acquired']],
                              month = [date.split("-")[1] for date in landsat_df['date_acquired']])

# 연도, 월, 컬렉션, 위성별로 데이터 개수 출력
```

```R
landsat = py$landsat_df

landsat = landsat %>%
  mutate(group = case_when(year < 1975 ~ "~1975",
                           year < 1980 ~ "~1980",
                           year < 1985 ~ "~1985",
                           year < 1990 ~ "~1990",
                           year < 1995 ~ "~1995",
                           year < 2000 ~ "~2000",
                           year < 2005 ~ "~2005",
                           year < 2010 ~ "~2010",
                           year < 2015 ~ "~2015",
                           year < 2020 ~ "~2020"))

landsat %>%
  group_by(group) %>%
  ggplot() +
  geom_bar(aes(x=factor(group, ordered=T), fill=factor(group, ordered=T))) +
  labs(x="연도(그룹)별 데이터 수")
```

![CountByYear](/assets/article_images/CountByYear.png)

파이썬으로 받아온 데이터를 R로 넘겨서 연도별 데이터 수를 시각화했다. 보다시피 데이터가 아주 풍부한 시기가 있는가 하면 빈곤한 시기도 있다. 데이터가 많다면 적절한 기준을 통해서 가장 퀄리티가 좋은 데이터를 뽑아내야 하고, 데이터가 없다면 대체할만한 다른 데이터를 찾거나 어쩔 수 없이 누락된 채로 두어야 한다.

## 1. 데이터 다운로드

다음으로는 걸러진 인덱스 데이터를 이용하여 실제로 위성사진을 다운로드하는 함수를 작성한다. 다운로드 함수의 개요는 다음과 같다.

1. 데이터에 해당하는 BaseURL의 리스트를 인자로 전달받는다.
2. 각 행에 해당하는 데이터의 prefix를 설정한다.
3. prefix를 통해 필요한 데이터를 다운로드한다.

먼저 google storage에 접근할 수 있는 클라이언트 객체를 생성하고, 랜샛 데이터가 담긴 버킷을 가져온다.

```python
storage_client = storage.Client()
bucket = storage_client.get_bucket('gcp-public-data-landsat')
```

다음으로 필요한 데이터의 prefix를 설정해준다. prefix에 대한 정보는 `base_url` 컬럼으로부터 가져올 수 있다. `base_url`에서 'gcp-public-data-landsat'은 버킷 이름을 가리키고, 'gcp-public-data-landsat' 이하의 문자열이 해당 데이터의 prefix가 된다. 따라서 모든 베이스 URL에 대해서 prefix는 다음과 같은 형태로 표현할 수 있다.

```python
landsat_df.base_url[0]
```

    'gs://gcp-public-data-landsat/LM05/PRE/116/034/LM51160341984180HAJ00'

```python
landsat_df.base_url[0].split("gcp-public-data-landsat/")[1]
```

    'LM05/PRE/116/034/LM51160341984180HAJ00'


지금까지 완성된 prefix에는 지표온도를 계산하는데 필요하지 않은 밴드들 역시 포함되어 있다. 따라서 지표온도를 계산하는데 필요하지 않은 밴드들을 걸러내는 작업이 필요하다. 위성별로 지표온도 계산에 필요한 밴드는 다음과 같다. 각 위성별로 필요한 밴드를 알 수 있으므로 이에 맞게 prefix를 조정한다.

|Spacecraft|Bands required|
|---|---|
|Landsat4|Band 6|
|Landsat5|Band 6|
|Landsat7|Band 6|
|Landsat8|Band 4, 5, 10|

```python
import pandas as pd
import os
from google.cloud import storage, bigquery

def download_landsat(index):

    '''
    - index: pandas DataFrame(from google bigquery)
    - download destinatinion: /Downloads/(date_acquired)/(product_id or scene_id(PRE))
    '''
    # 데이터별 prefix를 추출: 행별로 적용
    def get_prefix(row):
        # date-acquired로 디렉토리 이름 설정
        directory = row.date_acquired.replace("-","")
        url = row.base_url
        url = url.split("gcp-public-data-landsat/")[1]
        ID = url.split("/")[-1]
        # 위성별로 필요한 밴드 선택
        if url.split("/")[0] == "LC08":
            prefix = [f"{url}/{ID}_B4.TIF", f"{url}/{ID}_B5.TIF", f"{url}/{ID}_B10.TIF"]
        elif url.split("/")[0] == "LE07":
            prefix = [f"{url}/{ID}_B6_VCID_1.TIF"]
        else :
            prefix = [f"{url}/{ID}_B6.TIF"]

        results = [directory, prefix]
        return results

    # get_prefix에서 반환되는 results를 통해 데이터를 다운로드하는 함수
    def download_prefix(results):
        directory = results[0]
        prefix = results[1]
        if not os.path.exists(f"Downloads\\{directory}"):
            os.mkdir(os.path.join(f"Downloads\\{directory}"))
        for product in prefix:
            filename = product.split('/')[-1]
            blob = bucket.blob(product)
            blob.download_to_filename(f"Downloads\\{directory}\\{filename}")

    # storage에 접근할 클라이언트 객체 생성, 버킷 객체 생성
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f"{os.getcwd()}/dsyonsei.json"
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('gcp-public-data-landsat')
    results = index.apply(get_prefix, axis=1)
    for result in results:
        print(f'Downloading:{result[1]}')
        download_prefix(result)
        print(f'colmpleted')

if not os.path.exists(f"Downloads"):
    mkdir("Downloads")
index = pd.read_csv("DownloadIndex.csv")
download_landsat(index)

```
