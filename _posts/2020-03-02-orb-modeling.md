---
layout: post
title: "[Orbit] 월간 데이콘2 천체 유형 분류: 예측 모델링"
categories: [project]
tags: orbit
comments: true
---

최종 모델은 `lightgbm` 을 사용하였습니다. 다른 참가자분들도 역시 부스팅 계열의 모델을 주로 사용하신 것으로 보입니다. 아직까지는 딥러닝보다 더 잘 먹히는 고전적 방법들이 존재하는 것 같네요. 저희 팀이 사용한 패키지 목록은 아래와 같습니다.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
from bayes_opt import BayesianOptimization

np.random.seed(0)
train, test = pd.read_csv("./data/train.csv"), pd.read_csv("./data/test.csv")
```

## 1.  Data Preparation

### 1.1. 아웃라이어 처리

EDA 결과에서 발견된 이상치들을 제거해주기로 합니다. 각 변수마다 0.1% 안에 들지 못하는 값이 있는 경우 모두 잘라내줍니다. 사실 트리 모델이 아웃라이어에 강건하기는 하지만, 처리를 해주었을 때와 해주지 않았을 때 리더보드에서의 점수차이가 있었기 때문에 처리를 해주는 것으로 결론을 내렸습니다. 

```python
def clip(colname):
    start, end = train[colname].quantile(0.001), train[colname].quantile(1- 0.001)
    bools = (train[colname] >= start) & (train[colname] <= end).rename(colname)
    return bools

mask = pd.concat(list(map(clip, train.columns[3:].values)), axis=1).apply(all, axis=1)
train = train[mask]
```


###  1.2. 파생변수 계산

새로운 변수들을 만들어냅니다. 선형 종속인 변수들을 추가해주는게 큰 의미가 있을까 우려를 했는데, cross validation 결과 유의미한 향상을 보였습니다. 순위권 분들의 코드를 보니 생각보다 정말 많은 피쳐들을 사용하셨다는 사실을 알 수 있었습니다. 사실 저희 팀에서 사용한 변수도 총 56개로 적은 편은 아니라고 생각했는데, 조금 더 열심히 피쳐를 만들고 검증해보는 실험을 반복했더라면 조금 더 나은 결과를 얻을 수 있었을 것 같다는 아쉬움이 남습니다. 

1. 각 측정방법(psf, fiber, model, petro)별로 u-g-r-i-z 밴드의 간격 계산
2. 각 측정방법(psf, fiber, model, petro)별로 u-g-r-i-z 밴드의 표준편차 계산
3. 각 측정방법(psf, fiber, model, petro) 간 같은 밴드의 차이 계산: 성능을 유의미하게 올려주지 못하는 경우에는 제외하였습니다
4. `fiberID`별로 `fiberMag_*` 변수의 평균값 계산


```python
def interval(metric, df):
    result = pd.DataFrame({
        metric + "Step_ug" : df[metric+"_u"] - df[metric+"_g"],
        metric + "Step_gr" : df[metric+"_g"] - df[metric+"_r"],
        metric + "Step_ri" : df[metric+"_r"] - df[metric+"_i"],
        metric + "Step_iz" : df[metric+"_i"] - df[metric+"_z"]
    })
    return result

fiber = train.groupby("fiberID")[['fiberMag_u', 'fiberMag_g', 'fiberMag_r', 'fiberMag_i', 'fiberMag_z']].mean().reset_index()
fiber.columns = ['fiberID', 'fiberMean_u', 'fiberMean_g', 'fiberMean_r', 'fiberMean_i', 'fiberMean_z']
train, test = pd.merge(train, fiber, how='left'), pd.merge(test, fiber, how='left')

xtrain, xtest = train.iloc[:,3:], test.iloc[:,2:]

trainInterval = pd.concat(list(pd.Series(['psfMag','fiberMag', 'modelMag', 'petroMag']).apply(interval, df=xtrain)), axis=1)
testInterval = pd.concat(list(pd.Series(['psfMag','fiberMag', 'modelMag', 'petroMag']).apply(interval, df=xtest)), axis=1)
xtrain, xtest = pd.concat([xtrain, trainInterval], axis=1), pd.concat([xtest, testInterval], axis=1)

xtrain = xtrain.assign(
    psf_fiber_u = xtrain.psfMag_u - xtrain.fiberMag_u,
    psf_fiber_g = xtrain.psfMag_g - xtrain.fiberMag_g,
    psf_fiber_r = xtrain.psfMag_r - xtrain.fiberMag_r,
    psf_fiber_i = xtrain.psfMag_i - xtrain.fiberMag_i,
    psf_fiber_z = xtrain.psfMag_z - xtrain.fiberMag_z,
    psf_model_u = xtrain.psfMag_u - xtrain.modelMag_u,
    psf_model_g = xtrain.psfMag_g - xtrain.modelMag_g,
    psf_model_r = xtrain.psfMag_r - xtrain.modelMag_r,
    psf_model_i = xtrain.psfMag_i - xtrain.modelMag_i,
    psf_model_z = xtrain.psfMag_z - xtrain.modelMag_z,
    fiber_petro_u = xtrain.fiberMag_u - xtrain.petroMag_u,
    fiber_petro_g = xtrain.fiberMag_g - xtrain.petroMag_g,
    fiber_petro_r = xtrain.fiberMag_r - xtrain.petroMag_r,
    fiber_petro_i = xtrain.fiberMag_i - xtrain.petroMag_i,
    fiber_petro_z = xtrain.fiberMag_z - xtrain.petroMag_z,
    psfMag_std = xtrain[['psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z']].std(axis=1),
    fiberMag_std = xtrain[['fiberMag_u', 'fiberMag_g', 'fiberMag_r', 'fiberMag_i', 'fiberMag_z']].std(axis=1),
    modelMag_std = xtrain[['modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z']].std(axis=1),
    petroMag_std = xtrain[['petroMag_u', 'petroMag_g', 'petroMag_r', 'petroMag_i', 'petroMag_z']].std(axis=1)
).assign(fiberID = train.fiberID)

xtest = xtest.assign(
    psf_fiber_u = xtest.psfMag_u - xtest.fiberMag_u,
    psf_fiber_g = xtest.psfMag_g - xtest.fiberMag_g,
    psf_fiber_r = xtest.psfMag_r - xtest.fiberMag_r,
    psf_fiber_i = xtest.psfMag_i - xtest.fiberMag_i,
    psf_fiber_z = xtest.psfMag_z - xtest.fiberMag_z,
    psf_model_u = xtest.psfMag_u - xtest.modelMag_u,
    psf_model_g = xtest.psfMag_g - xtest.modelMag_g,
    psf_model_r = xtest.psfMag_r - xtest.modelMag_r,
    psf_model_i = xtest.psfMag_i - xtest.modelMag_i,
    psf_model_z = xtest.psfMag_z - xtest.modelMag_z,
    fiber_petro_u = xtest.fiberMag_u - xtest.petroMag_u,
    fiber_petro_g = xtest.fiberMag_g - xtest.petroMag_g,
    fiber_petro_r = xtest.fiberMag_r - xtest.petroMag_r,
    fiber_petro_i = xtest.fiberMag_i - xtest.petroMag_i,
    fiber_petro_z = xtest.fiberMag_z - xtest.petroMag_z,
    psfMag_std = xtest[['psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i', 'psfMag_z']].std(axis=1),
    fiberMag_std = xtest[['fiberMag_u', 'fiberMag_g', 'fiberMag_r', 'fiberMag_i', 'fiberMag_z']].std(axis=1),
    modelMag_std = xtest[['modelMag_u', 'modelMag_g', 'modelMag_r', 'modelMag_i', 'modelMag_z']].std(axis=1),
    petroMag_std = xtest[['petroMag_u', 'petroMag_g', 'petroMag_r', 'petroMag_i', 'petroMag_z']].std(axis=1)
).assign(fiberID = test.fiberID)
```
### 1.3. ytrain 인코딩

문자열로 된 타겟 데이터를 숫자로 인코딩해주고, `lightgbm` 데이터셋으로 만들어줍니다. 

```python
ytrain = train.type
encoder = LabelEncoder().fit(ytrain)
labels = encoder.transform(ytrain)
dataset = lgb.Dataset(xtrain, labels)
```

## 2. 하이퍼파라미터 튜닝

하이퍼파라미터 튜닝에는 베이지안 옵티마이제이션을 사용하였습니다. 건드린 파라미터는 `num_leaves`, `feature_fraction`, `bagging_fraction`, `max_depth`, `lambda_l1`, `lambda_l2`, `min_split_gain`, `min_child_weight`, `scale_pos_weight` 입니다. 베이지안 옵티마이제이션에 대한 자세한 내용은 SUALAB Research Blog에 포스팅된 [Bayesian Optimization 개요: 딥러닝 모델의 효과적인 hyperparameter 탐색 방법론 (1)](http://research.sualab.com/introduction/practice/2019/02/19/bayesian-optimization-overview-1.html)(김길호) 를 참고하였습니다. 다른 분들의 코드에서는 hyperopt 패키지가 심심치 않게 등장했음을 알 수 있었습니다. 아직 자세히 들여다보지는 않았지만, 추후 사용해보면서 다른 방식의 옵티마이제이션과 차이를 정리해보려고 합니다. 

```python
def lgb_eval(
    num_leaves,
    feature_fraction,
    bagging_fraction,
    max_depth,
    lambda_l1,
    lambda_l2,
    min_split_gain,
    min_child_weight,
    scale_pos_weight
):
    params = {
        'learning_rate': 0.023,
        'booster': 'goss',
        'objective': 'multiclass',
        'num_class': 19,
        'n_jobs':24
    }
    
    params["num_leaves"] = int(round(num_leaves))
    params['feature_fraction'] = max(min(feature_fraction, 1), 0)
    params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
    params['max_depth'] = int(round(max_depth))
    params['lambda_l1'] = max(lambda_l1, 0)
    params['lambda_l2'] = max(lambda_l2, 0)
    params['min_split_gain'] = min_split_gain
    params['min_child_weight'] = min_child_weight
    params['scale_pos_weight'] = scale_pos_weight
    
    cv_result = lgb.cv(
        params,
        dataset,
        num_boost_round= 2000, 
        early_stopping_rounds=30,
        nfold=5,
        seed=0, 
        shuffle=True,
        stratified=True
    )
    
    return min(cv_result['multi_logloss-mean']) * -1

pbounds = {
    'num_leaves': (24, 256),
    'feature_fraction': (0.1, 0.9),
    'bagging_fraction': (0.8, 1),
    'max_depth': (5, 32),
    'lambda_l1': (0, 5),
    'lambda_l2': (0, 3),
    'min_split_gain': (0.001, 0.1),
    'min_child_weight': (5, 50),
    'scale_pos_weight': (1.0, 1.3)
}

optimizer = BayesianOptimization(
    lgb_eval,
    pbounds,
    random_state=0
)

optimizer.maximize(init_points=5, n_iter=200)
```

## 3. 모델 검증 및 훈련

### 3.1. 파라미터 세팅

베이지안 옵티마이제이션으로 잡아놓은 파라미터를 세팅하고, `learning_rate`를 더 낮게 잡아줍니다. 점수 차이가 굉장히 작은 대회였기 때문에 learning rate를 충분히 낮게 잡아줄 필요가 있었습니다.

```python
params = {
    'learning_rate': 0.01,
    'booster': 'goss',
    'objective': 'multiclass',
    'num_class': 19,
    'bagging_fraction': 0.952043278059383,
    'feature_fraction': 0.21987400683878874,
    'lambda_l1': 0.4663681524440205,
    'lambda_l2': 2.538089586029138,
    'max_depth': 9,
    'min_child_weight': 6.338458091153288,
    'min_split_gain': 0.08935815884666935,
    'num_leaves': 90,
    'scale_pos_weight': 1.0662804565907613
}
```

### 3.2. 교차검증

마지막 제출에 사용한 교차검증 옵션은 아래와 같습니다. 사실 저희 팀이 많이 고민했던 지점이 교차검증 결과가 퍼블릭 리더보드에서 재현이 되지 않는다는 점이었습니다. 결과적으로 프라이빗 스코어에서는 그 차이가 조금 줄어들기는 하였지만, 무시할 수 있는 수준은 아니었습니다. 뒤늦게 토론 게시판을 들여다보니 비슷한 문제를 겪고 계시는 분들이 꽤나 있었던 것 같습니다. 이후 순위권 분들의 코드를 카피해보면서 어떤 점이 달랐는지 배워보고자 합니다. 

```python
result = lgb.cv(
    params,
    dataset,
    num_boost_round= 3000,
    early_stopping_rounds=100,
    nfold=5,
    stratified=True,
    shuffle=True,
    verbose_eval=10,
    seed=0
)
```

### 3.3. 모델 훈련

교차검증을 기준으로 부스팅 이터레이션을 잡고 모델을 훈련합니다.

```python
model = lgb.train(
        params,
        dataset,
        num_boost_round = 1750
    )
```

## 4. 제출 파일 생성

```python
submission = pd.read_csv("../data/sample_submission.csv")
result = lgbmodel.predict(xtest)
result = pd.DataFrame(result, columns = encoder.inverse_transform(np.arange(0,19))).assign(id = test['id'])[submission.columns]
result.to_csv("../submission/lgbm_fin3.csv", index=False)
```
