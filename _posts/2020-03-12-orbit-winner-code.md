---
layout: post
title: "[Orbit] 우승자 코드 리뷰 "
categories: [project]
tags: orbit
comments: true
---

정신 차리고 보니 이제 곧 개강이라는 사실을 깨달아서 미뤄둔 공부 및 작업들을 급하게 마무리하는 중입니다. 한동안 장고와 자바, 선형대수의 늪에 빠져 있다가 이제서야 우승자 코드 리뷰를 진행하게 되었습니다..! 사실 우승자분이 제출 마감 후 몇 분 되지 않아서 곧바로 코드를 올려주셔서 마음만 먹었다면 훨씬 더 일찍 끝낼 수 있었던 작업이지만 미루다 보니 여기까지 왔네요 ㅎㅎ 아무튼 시작해보도록 하겠습니다!

## 1. 라이브러리 및 데이터 

우선 라이브러리에서 눈에 띄는 점은 `sklearn.decomposition` 모듈인데, 다양한 factorization방법들을 시도해보신 것 같습니다. 처음 보는 분해들도 많네요..! 어떻게 사용하셨을지 상당히 궁금해집니다.


```python
from sklearn.decomposition import TruncatedSVD, PCA, FastICA, FactorAnalysis, KernelPCA, DictionaryLearning
from sklearn.decomposition import IncrementalPCA, LatentDirichletAllocation,MiniBatchSparsePCA, SparsePCA
```

## 2. 데이터 전처리

가장 먼저 타겟 데이터를 숫자로 인코딩하고, psfMag, fiberMag, petroMag, modelMag에 해당하는 컬럼 이름들을 리스트로 저장해주셨습니다.

```python
column_number = {}
for i, column in enumerate(submission.columns[1:]):
    column_number[column] = i
def to_number(x, dic):
    return dic[x]
train['type_num'] = train['type'].apply(lambda x: to_number(x, column_number))


psfMag_col = [c for c in train.columns if c.find('psfMag')!=-1]
fiberMag_col = [c for c in train.columns if c.find('fiberMag')!=-1]
petroMag_col = [c for c in train.columns if c.find('petroMag')!=-1]
modelMag_col = [c for c in train.columns if c.find('modelMag')!=-1]
```

## 3. EDA

저희 팀이 했던 것과 유사한 분석은 생략하고, 다른 점 위주로 짚어보겠습니다. 우선 저희 팀은 각 변수의 분포를 통계량을 통해서만 파악했는데, 아래처럼 직접 밀도 그래프를 그려보니 변수들의 분포가 눈에 훨씬 잘 들어오는 것 같습니다. 또한 파생변수들 역시 적절하게 시각화를 하신 부분도 눈에 띕니다. 저희 팀은 파생변수들로 모델링만 해보고 따로 EDA를 해보지는 않았는데, 이러한 부분에서도 부족한 점이 많이 드러나는 것 같습니다.


```python
def plot_category_hist(data, col_list, category):
    for c in col_list:
        u, d = np.percentile(data[c],99.5), np.percentile(data[c], 0.05)
        plt.figure(figsize=(12,5))
        for t in data[category].unique():
            ser = data[(data[c].between(d, u)) & (data[category] == t)][c]
            sns.distplot(ser)
        plt.title(c)
        plt.legend(data[category].unique())
        plt.show()
```

```python
plot_category_hist(train, psfMag_col, 'type')
```

![](/assets/img/docs/output_6_1.png)

![](/assets/img/docs/output_6_2.png)

![](/assets/img/docs/output_6_3.png)

![](/assets/img/docs/output_6_4.png)



```python
train_eda = train.copy()

for c in psfMag_col:
    u = np.percentile(train_eda[c],99)
    d = np.percentile(train_eda[c],0.1)
    train_eda = train_eda[train_eda[c].between(d, u)]

diff_feature = []
for c1, c2 in itertools.combinations(psfMag_col[::-1],2):
    new_c = f'{c1}_{c2}_diff'
    train_eda[new_c] = train_eda[c1]-train_eda[c2]
    diff_feature.append(new_c)

"""
for i, (c1,c2) in enumerate(itertools.combinations(diff_feature,2)):
    plt.figure(figsize=(8,8))
    sns.scatterplot(c1,c2, data=train_eda, hue='type')
    plt.title(f'{c1} vs {c2}')
    plt.grid()
    plt.show()
    draw_count+=1 
"""

# 2 Mb 용량 제한 때문에 대표 하나만 그림
c1 = 'psfMag_z_psfMag_r_diff'
c2 = 'psfMag_r_psfMag_u_diff'
plt.figure(figsize=(8,8))
sns.scatterplot(c1,c2, data=train_eda, hue='type')
plt.title(f'{c1} vs {c2}')
plt.grid()
plt.show()
```


![png](/assets/img/docs/output_7_0.png)


## 4. 변수 선택

개인적으로 변수 선택 및 모델 구축 파트가 이 커널의 하이라이트가 아닌가 싶은데요, 정말 많은 피쳐들을 생성하고 검증해주셨습니다. 일단 피쳐를 많이 만들고, permutation importance를 활용하여 적절하게 가지치기를 하는 방법을 사용하셨습니다. 개인적으로 궁금했던 부분이 permutation importance를 계산하는 부분이었는데, 이와 관련해서는 `eli5`라는 라이브러리를 사용하셨다고 답변해주셨습니다. Permutation importance를 구한 이후에는 실험을 통해 적절한 cutoff를 설정하여 추후 학습 시 이 피쳐들을 제거하는 방식이라고 하네요! 직접 만드신 변수 몇 가지 정도만 추려서 리뷰를 해보겠습니다. 가장 먼저 각 행/열별로 통계량을 계산해서 컬럼으로 추가해주고, CV 스코어에 따라서 잘라내주신 것 같습니다. 다음으로 저희 팀도 시도했던 diff 관련 변수들인데, `itertools`를 적절하게 활용하셔서 정말 다양한 조합을 시도하셨습니다. 마지막으로 신기하게 보았던 부분이 decomposition 부분인데, `TruncatedSVD`와 `FastICA`라는 방법으로 분해해주고 다시 permutation importance를 활용해서 적절한 변수들을 추려주었다고 합니다.


```python
# zip 함수를 이용하여 각 Row별, Magnitude별 max, min, max-min, std, sum을 구한다.
# mean, skew, 등 다른 것들 시도 시 cv 점수가 안 좋아져서 사용하지 않음
for prefix, g in zip(['psfMag','fiberMag','petroMag','modelMag'], [psfMag_col, fiberMag_col, petroMag_col, modelMag_col]):
    train[f'{prefix}_max'] = train[g].max(axis=1)
    test[f'{prefix}_max'] = test[g].max(axis=1)
    
    train[f'{prefix}_min'] = train[g].min(axis=1)
    test[f'{prefix}_min'] = test[g].min(axis=1)
    
    train[f'{prefix}_diff'] = train[f'{prefix}_max'] - train[f'{prefix}_min']
    test[f'{prefix}_diff'] = test[f'{prefix}_max'] - test[f'{prefix}_min']
    
    train[f'{prefix}_std'] = train[g].std(axis=1)
    test[f'{prefix}_std'] = test[g].std(axis=1)
    
    train[f'{prefix}_sum'] = train[g].sum(axis=1)
    test[f'{prefix}_sum'] = test[g].sum(axis=1)
```

```python
# diff feature 추가 예: psfMag_z - psfMag_i 
# sdss lagacy solution 등을 보면 대 부분 mag간 차이를 사용하기 때문에 이런 diff feature가 의미가 있을 것이라고 판단
# 그리고 각 magnitude에서만 diff를 구하는 것이 아닌 itertools combinations를 활용하여 전체 magnitude에서 diff를 구함
# 총 190가지 조합이 나오고 여기서 안 좋은 것은 permutation importance를 활용하여 feature 제거 수행
diff_feature = []
for c1, c2 in itertools.combinations(psfMag_col[::-1]+fiberMag_col[::-1]+petroMag_col[::-1]+modelMag_col[::-1],2):
    new_c = f'{c1}_{c2}_diff'
    train[new_c] = train[c1]-train[c2]
    test[new_c] = test[c1]-test[c2]
    diff_feature.append(new_c)
```

```python
def get_decomposition_feature(train, test, feature, param, decompose_func, prefix):
    n_components = param['n_components']
    de = decompose_func(**param)
    de_train = de.fit_transform(train[feature])
    de_test = de.transform(test[feature])
    train = pd.concat([train, pd.DataFrame(de_train,columns=[f'{prefix}_{c}' for c in range(n_components)])],axis=1)
    test = pd.concat([test, pd.DataFrame(de_test,columns=[f'{prefix}_{c}' for c in range(n_components)])],axis=1)
    return train, test

org_feature = psfMag_col+fiberMag_col+petroMag_col+modelMag_col
# decompostion해서 다시 feature로 추가, 원래 original feature만 사용하고 5개로 축소
decom_common_param = {'n_components':5,'random_state':42}
train, test = get_decomposition_feature(train, test, org_feature, decom_common_param, TruncatedSVD, 'tsvd5')
train, test = get_decomposition_feature(train, test, org_feature, decom_common_param, FastICA, 'ica5')
```

## 5. 모델 학습 및 검증

모델링 부분에서 가장 큰 차이점은 dart 부스터를 사용하셨다는 점 정도인 것 같습니다. 처음에는 `hyperopt` 패키지가 베이지안 옵티마이제이션과 다른 방식인 줄 알았는데, 페이퍼의 초록을 보니 다음과 같이 적혀 있네요. 결국에는 베이지안 옵티마이제이션인데, 더 널리 쓰이는 패키지인 모양입니다. 나중에 한번 사용해봐야겠네요 ㅎㅎ

> Abstract—Sequential model-based optimization (also known as Bayesian optimization) is one of the most efficient methods (per function evaluation) of
function minimization. This efficiency makes it appropriate for optimizing the
hyperparameters of machine learning algorithms that are slow to train. The
Hyperopt library provides algorithms and parallelization infrastructure for performing hyperparameter optimization (model selection) in Python. This paper
presents an introductory tutorial on the usage of the Hyperopt library, including
the description of search spaces, minimization (in serial and parallel), and the
analysis of the results collected in the course of minimization. The paper closes
with some discussion of ongoing and future work.

```python
# hyper optimization으로 찾아낸 parameter
# lightgbm dart 사용, 보다 lb 0.03 정도 좋음
# gbdt가 0.3285라면 dart는 0.3255, goss는 0.3300
lgb_param_dart = {'objective': 'multiclass', 
 'num_class': 19, 
 'boosting_type': 'dart', 
 'subsample_freq': 5, 
 'num_leaves': 92, 
 'min_data_in_leaf': 64, 
 'subsample_for_bin': 23000, 
 'max_depth': 10, 
 'feature_fraction': 0.302, 
 'bagging_fraction': 0.904, 
 'lambda_l1': 0.099, 
 'lambda_l2': 1.497, 
 'min_child_weight': 38.011, 
 'nthread': 32, 
 'metric': 'multi_logloss', 
 'learning_rate': 0.021, 
 'min_sum_hessian_in_leaf': 3, 
 'drop_rate': 0.846244, 
 'skip_drop': 0.792465, 
 'max_drop': 65,
 'seed': 42,
 'n_estimators': 1000}
```

## 6. 결론

결론적으로, 우승자분의 모델링 과정과 저희 팀의 모델링 과정에 엄청나게 큰 차이는 없었던 것 같습니다. 그보다는 사소하지만 정말 많은 시간과 노력, 열정을 요구하는 섬세한 작업들에서 차이가 많이 벌어진 것 같습니다. 특히 변수 하나하나를 일일히 분석해보고 실험해본다는 것은 저희 팀으로서는 상상도 하지 못한 방법이었습니다. 데이터 분석이 어쩌면 3D 업종일지도 모른다는 생각을 다시 한번 하게 되지만, 그래서 더 재미있는 분야인 것 같기도 합니다. 아무튼 이번 대회를 통해서 정말 좋은 경험을 했고, 발전의 계기가 생겼던 것 같습니다!