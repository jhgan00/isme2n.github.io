---
layout: post
title: "[R] 코로나 관련 논문"
categories: [blog]
tags: [r]
comments: true
---

이번 학기에 수강하는 범주형자료분석 수업에서 교수님깨서 코로나 관련 논문을 하나 소개해주셨어요(사실은 과제지만..ㅎㅎ). [Relationship between the ABO Blood Group and the COVID-19 Susceptibility](https://www.medrxiv.org/content/10.1101/2020.03.11.20031096v2) 라는 논문인데, 중국의 데이터를 사용하여 코로나19와 혈액형, 나이, 성별에 대한 통계분석을 진행한 연구입니다. 우한 지역의 병원 두 곳(Jinyintan, Renmin)과 선전 지역의 병원 한 곳의 표본이 사용되었는데, Jinyintan 지역에서의 검정 결과 대조군에 비해 감염자 중에서 혈액형 A형의 비율이 유의미하게 높았고, O형의 비율이 유의미하게 낮았다고 합니다. 물론 이걸 그대로 인과관계로 해석할 수는 없겠지만요! 연구에서는 SPSS와 STATA를 사용하여 오즈비를 검정하였는데, R을 사용해서 검정을 돌려봐도 같은 결과가 나오는 것을 볼 수 있습니다.

```r
> data = matrix(c(1188, 2506, 715, 1173), byrow=T, ncol=2)
> dimnames(data) = list(group=c("control", "patient"), type=c("a","others"))
> prop = prop.test(data)
> odds = prop\\( estimate/(1 - prop \\)estimate)
> theta= odds[2]/odds[1]
> ASE = sum(1/data) %>% sqrt
> logtheta.CI = log(theta) + c(-1, 1) * 1.96 * ASE
> exp(logtheta.CI)
[1] 1.145176 1.443687
```