---
layout: post
title: "[ML/Stat] EM Algorithm for latent variable models (2)"
categories: doc
tags: [ml, stat]
comments: true
use_math: true
---

이번에는 EM 알고리즘을 통해서 잠재변수 모델을 실제로 추정하고, 클러스터링을 시행봅니다 😀 K-means 클러스터링의 soft assignment 버전이라고 생각하시면 편할 것 같습니다! 아래 자료들을 참고하여 작성한 코드입니다. 아이리스 데이터셋에서 사용한 `plus_plus` 함수는 직접 작성한 것이 아니며, 아래 [`centroid_initialization.py`](https://gist.github.com/mmmayo13/3d5c2b12218dfd79acc27c64b3b7dd86#file-centroid_initialization-py) 의 코드를 사용하였습을 미리 밝힙니다!

- [Lecture 14 - Expectation-Maximization Algorithms \| Stanford CS229: Machine Learning (Autumn 2018)](https://www.youtube.com/watch?v=rVfZHWTwXSA&t=2192s)
- [27. EM Algorithm for Latent Variable Models](https://www.youtube.com/watch?v=lMShR1vjbUo)
    - [Lecture Slides](https://davidrosenberg.github.io/mlcourse/Archive/2017Fall/Lectures/13c.EM-algorithm.pdf)
- [CSC 411: Lecture 13: Mixtures of Gaussians and EM](http://nlp.chonbuk.ac.kr/BML/slides_uoft/13_mog.pdf)
- [centroid_initialization.py](https://gist.github.com/mmmayo13/3d5c2b12218dfd79acc27c64b3b7dd86#file-centroid_initialization-py)

## 1. Mixture of Gaussians

지난번 포스팅에서 예를 들었던 가우시안 혼합 모형의 모수를 실제로 추론해보자. [앤드류 응 교수의 강의](https://www.youtube.com/watch?v=rVfZHWTwXSA&t=2192s) 31분 쯤부터를 주로 참고해서 만들었다. `pdf` 와 `log_likelihood` 는 각각 일변량 정규분포의 밀도와 로그우도 $\log p(\mathbf{x})$를 계산한다.


```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")

# 정규분포의 pdf
def pdf(x, mean, std):
    return np.exp(-((x-mean)**2)/(2 * (std**2))) / (std * (2 * np.pi)**0.5)

# Evidence 의 로그우도
def log_likelihood(x, mu, sigma, p):
    return np.log((pdf(x, mu, sigma) * p).sum(axis=1)).sum().round(4)

# 관측괎과 잠재변수의 결합확률밀도
def p_xz(x, pis, means, sigmas):
    density = pdf(x, means, sigmas)
    return density * pis
#     return np.array([pdf(x, mean, cov) * pi for pi, mean, cov in zip(pis, means, covs)])

np.random.seed(2021)

x1 = np.random.normal(1, 1, 100)  # N(1, 1)
x2 = np.random.normal(10, 3, 50)  # N(10, 3^2)
x = np.append(x1, x2).reshape((-1, 1))
z = np.hstack([np.zeros(shape=(100,), dtype=np.int8), np.ones(shape=(50,), dtype=np.int8)])
df = pd.DataFrame(dict(x=x.flatten(), z=z))

plt.figure(figsize=(16, 4))
plt.subplot(1, 2, 1)
sns.kdeplot(
   data = df, x="x",
   fill=True, common_norm=False, palette="crest",
   alpha=.5, linewidth=0,
)

plt.subplot(1, 2, 2)
sns.kdeplot(
   data = df, x="x", hue="z",
   fill=True, common_norm=False, palette="crest",
   alpha=.5, linewidth=0,
)

plt.suptitle("Univariate Gaussian Mixture", fontsize=15)
plt.show()
```


    
![png](/assets/img/docs/output_4_0.png)
    


### 1.1. Choose initial $\theta^{old}$

EM 알고리즘은 수렴이 보장되지만 그것이 전역 최대라는 보장이 없다. 즉 초기값을 어떻게 설정하는지에 따라 알고리즘의 수렴 결과가 달라질 수 있다. ELBO의 전역 최대가 결국은 로그우도의 전역 최대와 같기는 하지만, 알고리즘이 항상 전역 최대를 찾아갈 수 있는가는 다른 문제이다. 그래서 K-means++ 와 같이 서로 충분히 멀리 떨어진 점들을 선택하기 위한 기법들이 연구되어왔다. 이번 예제는 간단한 인조 데이터이므로 대충 초기화해보자. 찾아야 할 파라미터는 두 정규분포의 평균과 분산, 그리고 클러스터 할당의 혼합계수(mixing coefficient) 이다. 평균은 각각 최대값과 최소값, 분산은 데이터 전체를 통해 구한 분산으로 대충 초기화했다. 혼합계수는 0.5씩으로 동일하게 초기화했다.


```python
pis = np.array([[0.5, 0.5]], dtype=np.float64)  # 혼합계수 pi
means = np.array([x.min(), x.max()])  # 각 가우시안의 평균 mu
stds = np.array([[np.std(x), np.std(x)]], dtype=np.float64)  # 각 가우시안의 표준편차 sigma
```


```python
logL = log_likelihood(x, means, stds, pis).round(4)  # 초기화된 파라미터의 로그우도
eps = 1e-9  # 로그우도의 개선이 입실론보다 작으면 수렴으로 간주하고 최적화를 멈춘다
```

### 1.2. Expectation - Maximization

$$
q^{\ast}(z) = p(Z=z \vert x, \theta) \\
\begin{align}
\gamma^{(c)} &:=
\frac {p(Z=c \vert \theta ^{old}) p(x \vert Z=c, \theta ^{old})} {\sum_{j=1}^{k} p(Z=j \vert \theta ^{old}) p(x \vert Z=j, \theta ^{old})}
\end{align}
$$

베이즈 룰을 사용하여 $q(z)$ 를 위의 $\gamma^{(c)}$ 처럼 나타낼 수 있다. $\gamma^{(z)}$ 는 하나의 데이터가 클러스터 $z$ 에 속할 조건부확률을 나타낸다. 이를 이용해서 $\log p(\mathbf{x})$ 를 최대화하는 파라미터를 구하면 다음과 같다. 아래첨자가 데이터, 위첨자가 클러스터를 나타낸다. 미분으로 MLE를 구하는 자세한 과정은 [여기](http://nlp.chonbuk.ac.kr/BML/slides_uoft/13_mog.pdf)를 참고하면 된다. 

$$
\begin{align}
& \mu_c^{new}    = \frac{1}{n^{(c)}} \sum_{i=1}^n \gamma_i^{(c)} x_i \\
& \sigma_c^{new} = \frac{1}{n^{(c)}} \sum_{i=1}^n \gamma_i^{(c)} (x_i - \mu_{mle})^2 \\ 
& \pi_c^{new}    = \frac{n^{(c)}}{n} \\
& where \space n^{(c)} = \sum_{i=1}^n \gamma_i^{(c)}  \space and \space n = \sum_{c=1}^k n^{(c)} 
\end{align}
$$


```python
fmt = "EPOCH: {:>5} log-likelihood: {:>.5f} gain: {:>.5f}"
for epoch in range(1, 101):
    
    joint = p_xz(x, pis, means, stds)
    gammas = joint / joint.sum(axis=1, keepdims=True)
    n_c = gammas.sum(axis=0, keepdims=True)

    means = (gammas * x).sum(axis=0, keepdims=True) / n_c
    stds = np.sqrt((gammas * (x - means)**2).sum(axis=0, keepdims=True) / n_c)
    pis = n_c / n_c.sum()

    logL_new = log_likelihood(x, means, stds, pis)
    gain = logL_new - logL
    
    assert gain >= 0

    print(fmt.format(epoch, logL_new, gain))

    if gain < eps:
        
        print("=" * 50, end="\n\n")
        print("Algorithm Converged!!")
        print(fmt.format(epoch, logL_new, gain), end="\n\n")
        print("=" * 50)

        break
    
    else:
        
        logL = logL_new
```

    EPOCH:     1 log-likelihood: -390.07090 gain: 137.82580
    EPOCH:     2 log-likelihood: -362.58360 gain: 27.48730
    EPOCH:     3 log-likelihood: -354.97490 gain: 7.60870
    EPOCH:     4 log-likelihood: -354.27950 gain: 0.69540
    EPOCH:     5 log-likelihood: -354.24200 gain: 0.03750
    EPOCH:     6 log-likelihood: -354.23990 gain: 0.00210
    EPOCH:     7 log-likelihood: -354.23980 gain: 0.00010
    EPOCH:     8 log-likelihood: -354.23980 gain: 0.00000
    ==================================================
    
    Algorithm Converged!!
    EPOCH:     8 log-likelihood: -354.23980 gain: 0.00000
    
    ==================================================


    
![png](/assets/img/docs/output_12_1.png)
    


# 2. Iris Dataset 

이번에는 아이리스 데이터로 실험해보자. 다변량 데이터이기 다변량 정규분포에 대한 이해가 조금 필요하다. 물론 수식만 읽을 수 있으면 왜 되는지는 몰라도 구현할 수는 있다. 이번에는 초기화를 대충 하지 않고 Kmeans++ 의 방법을 사용했다. 초기화 함수인 `plus_plus` 는 By Matthew Mayo의 [GIST](https://gist.github.com/mmmayo13/3d5c2b12218dfd79acc27c64b3b7dd86#file-centroid_initialization-py) 에서 가져왔다. 그 외에는 전부 직접 작성했다.


```python
import pandas as pd
from sklearn.datasets import load_iris
```


```python
def pdf(x, mu, sigma):
    
    """pdf of the multivariate gaussian distribution
    
    param x: np.ndarray (n, d)
    param mu: np.ndarray (1, d)
    param sigma: np.ndarray (d, d)
    
    returns: probability densities of x. np.ndarray (n, d)
    """
    
    d = sigma.shape[-1]
    x_m = x - mu
    denominator = np.sqrt((2 * np.pi) ** d * np.linalg.det(sigma))
    numerator = np.exp(
        -np.matmul(
            np.matmul(x_m[:, np.newaxis, :], np.linalg.inv(sigma)),
            x_m[..., np.newaxis]
        ).squeeze() * 0.5
    )
    
    return numerator / denominator
```


```python
def p_xz(x, pis, means, covs):
    
    """compute joint densities  p(x,z). k is the number of clusters.
    
    param x: np.ndarray (n, d)
    param pis: np.ndarray (k,)
    param means: np.ndarray (k, 1, d)
    param covs: (k, d, d)
    
    returns: joint densities of x and z. np.ndarray (k, n)
    """
    
    return np.array([pdf(x, mean, cov) * pi for pi, mean, cov in zip(pis, means, covs)])
```


```python
def log_likelihood(x, pis, means, covs):
    
    """compute log likelihood  log[p(x)].
    
    param x: np.ndarray (n, d)
    param pis: np.ndarray (k,)
    param means: np.ndarray (k, 1, d)
    param covs: (k, d, d)
    
    returns: log likelihood. np.float64
    """
    
    return np.log(p_xz(x, pis, means, covs).sum(axis=0)).sum()
```


```python
def plus_plus(ds, k, random_state=2021):
    
    
    """
    Create cluster centroids using the k-means++ algorithm.
    Parameters
    ----------
    ds : numpy array
        The dataset to be used for centroid initialization.
    k : int
        The desired number of clusters for which centroids are required.
    Returns
    -------
    centroids : numpy array
        Collection of k centroids as a numpy array.
    Inspiration from here: https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
    """

    np.random.seed(random_state)
    centroids = [ds[0]]

    for _ in range(1, k):
        dist_sq = np.array([min([np.inner(c-x,c-x) for c in centroids]) for x in ds])
        probs = dist_sq/dist_sq.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        
        for j, p in enumerate(cumulative_probs):
            if r < p:
                i = j
                break
        
        centroids.append(ds[i])

    return np.array(centroids)
```


```python
iris = load_iris()
x = iris.get("data")  # 4 X 150 행렬
xcols = iris.get("feature_names")  # 4 X 150 행렬
y = iris.get("target")  # 4 X 150 행렬
```

### 1.1. Choose initial $\theta^{old}$

역시 혼합계수, 클러스터별 평균, 공분산행렬을 초기화해야 한다. 혼합계수와 공분산행렬은 이전 예제에서와 같은 방식으로 초기화하였다. 평균은 위에서 언급한대로 Kmeans++ 의 방식으로 초기화하였다.


```python
pis = np.array([1/3, 1/3, 1/3])
means = plus_plus(x, 3)
covs = np.repeat(np.cov(x.T)[np.newaxis, ...], 3, axis=0)
```


```python
logL = log_likelihood(x, pis, means, covs)
eps = 1e-19
```

### 1.2. Expectation - Maximization

$$
\begin{align}
\gamma^{(c)} &:=
\frac {p(Z=c \vert \theta ^{old}) p(\mathbf{x} \vert Z=c, \theta ^{old})} {\sum_{j=1}^{k} p(Z=j \vert \theta ^{old}) p(\mathbf{x} \vert Z=j, \theta ^{old})}
\end{align}
$$

$$
\begin{align}
& \mu_c^{new}    = \frac{1}{n^{(c)}} \sum_{i=1}^n \gamma_i^{(c)} \mathbf{x}_i \\
& \Sigma_c^{new} = \frac{1}{n^{(c)}} \sum_{i=1}^n \gamma_i^{(c)} (\mathbf{x}_i - \mu_{mle}) (\mathbf{x}_i - \mu_{mle})^T \\ 
& \pi_c^{new}    = \frac{n^{(c)}}{n} \\
& where \space n^{(c)} = \sum_{i=1}^n \gamma_i^{(c)}  \space and \space n = \sum_{c=1}^k n^{(c)} 
\end{align}
$$

수식상으로는 크게 달라진 것이 없다. 행렬/벡터 연산을 직접 구현해본 경험이 많다면 쉽게 할 수 있을 것 같다. 개인적으로는 직접 넘파이로 구현해본 경험은 많이 없어서 고생을 좀 하다가 결국 반복문으로 타협했다 😅 


```python
fmt = "EPOCH: {:>5} log-likelihood: {:>.5f} gain: {:>.5f}"
for epoch in range(1, 1001):
    
    joint = p_xz(x, pis, means, covs)[..., np.newaxis]
    gammas = joint / joint.sum(axis=0, keepdims=True)
    N_c = gammas.sum(axis=1)

    means = []
    covs = []
    pis = []
    
    # 각 클러스터별로 파라미터를 업데이트한다
    for gamma, N in zip(gammas, N_c):

        mean = (gamma * x).sum(0, keepdims=True) / N

        x_m = (x - mean)[:, :, np.newaxis]
        cov = np.sum(np.matmul(x_m,  x_m.transpose((0, 2, 1))) * gamma[:, np.newaxis], axis=0) / N

        means.append(mean)
        covs.append(cov)

    pis = N_c / N_c.sum()
    means = np.array(means)
    covs = np.array(covs)

    logL_new = log_likelihood(x, pis, means, covs)
    
    gain = logL_new - logL
    assert gain >= 0

    if epoch % 50 == 0:

        print(fmt.format(epoch, logL_new, gain))

    if logL_new - logL < eps:

        print("=" * 50, end="\n\n")
        print("Algorithm Converged!!")
        print(fmt.format(epoch, logL_new, gain), end="\n\n")
        print("=" * 50)
        break

    else:
        
        logL = logL_new
```

    EPOCH:    50 log-likelihood: -189.42864 gain: 0.00215
    EPOCH:   100 log-likelihood: -189.35420 gain: 0.00039
    EPOCH:   150 log-likelihood: -189.34158 gain: 0.00032
    EPOCH:   200 log-likelihood: -186.80848 gain: 0.18393
    ==================================================
    
    Algorithm Converged!!
    EPOCH:   249 log-likelihood: -186.56946 gain: 0.00000
    
    ==================================================


클러스터링 결과가 실제 품종과 얼마나 일치하는지 체크해보자. 대충 17개 정도의 데이터가 실제 품종과 일치하지 않는 클러스터로 분류되었다. 꽃잎 관련 피쳐를 통해서 시각화해보면 2번 품종의 대각선 아래 데이터들이 1번 클러스터로 분류되었음을 알 수 있다. KDE 플롯을 살펴보면 원 데이터에도 대각선 위쪽에 몰려있는 경향이 존재하는데, 이를 반영한 결과로 보인다.



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: center;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
    <!-- <tr>
      <th>species</th>
      <th></th>
      <th></th>
      <th></th>
    </tr> -->
  </thead>
  <tbody style="text-align: center;">
    <tr>
      <th>0</th>
      <td>50</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>49</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>16</td>
      <td>34</td>
    </tr>
  </tbody>
</table>
</div>

    
![png](/assets/img/docs/output_29_1.png)
    

## 참고자료


- [Lecture 14 - Expectation-Maximization Algorithms \| Stanford CS229: Machine Learning (Autumn 2018)](https://www.youtube.com/watch?v=rVfZHWTwXSA&t=2192s)
- [27. EM Algorithm for Latent Variable Models](https://www.youtube.com/watch?v=lMShR1vjbUo)
    - [Lecture Slides](https://davidrosenberg.github.io/mlcourse/Archive/2017Fall/Lectures/13c.EM-algorithm.pdf)
- [CSC 411: Lecture 13: Mixtures of Gaussians and EM](http://nlp.chonbuk.ac.kr/BML/slides_uoft/13_mog.pdf)
- [centroid_initialization.py](https://gist.github.com/mmmayo13/3d5c2b12218dfd79acc27c64b3b7dd86#file-centroid_initialization-py)