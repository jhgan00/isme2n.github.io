---
layout: post
title: "[ML/Stat] EM Algorithm for latent variable models"
categories: doc
tags: [ml, stat]
comments: true
use_math: true
---


다음 강의들을 참고하여 개인적으로 EM 알고리즘에 대해 공부한 내용입니다! 쉽지 않네요 😅


- [Lecture 14 - Expectation-Maximization Algorithms \| Stanford CS229: Machine Learning (Autumn 2018)](https://www.youtube.com/watch?v=rVfZHWTwXSA&t=2192s)
- [27. EM Algorithm for Latent Variable Models](https://www.youtube.com/watch?v=lMShR1vjbUo)
    - [Lecture Slides](https://davidrosenberg.github.io/mlcourse/Archive/2017Fall/Lectures/13c.EM-algorithm.pdf)

# 1. EM Algorithm for latent variable models

EM은 관측되지 않은 잠재변수가 존재할 때 확률분포의 최대가능도 파라미터를 찾는 알고리즘이다. 솔직히 이렇게 들어서는 잘 감이 오지 않는다. 예시를 통해서 알아보자. 다음과 같은 1차원 데이터의 분포를 가정한다. 하나의 분포를 통해서 설명할 수도 있겠지만, 두 개의 가우시안이 혼합된 분포로 설명하는 편이 더 나아 보인다. $n$ 개의 데이터가 존재하고 확률변수 $X, Z$가 각각 알려진 값과 숨겨진 클러스터(가우시안 분포)에 대응한다고 해보자. $X$는 임의의 실수 값을 가지며 $Z$ 는 0 또는 1의 값을 가진다. 


```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```


```python
np.random.seed(2021)

x1 = np.random.normal(1, 1, 100)
x2 = np.random.normal(10, 3, 50)
x = np.append(x1, x2)
np.random.shuffle(x)

sns.displot(x, kind="kde", rug=True)
```



![png](/assets/img/docs/output_4_1.png)


완전한 데이터(complete data) $\{(x_1, z_1), (x_1, z_1), \space ... \space, (x_n, z_n)\}$ 가 존재한다면 두 개의 분포에 대해서 각각 MLE를 추정할 수 있다. 즉 어떤 데이터가 어떤 가우시안 분포로부터 나왔는지를 아는 상태이다. 하지만 실제로 가진 데이터는 $\{x_1, x_2, \space ... \space , x_n\}$ 뿐이다. 각각이 어떤 분포로부터 나왔는지를 알 수 없는 것이다. 이를 불완전한 데이터(incomplete data)라고 부른다. $X$ 와 $Z$의 결합확률분포 모델이 파라미터 $\theta$를 가진다고 하면 모델을 $p(x, z \vert \theta)$와 같이 표현할 수 있다.

- 실제로 관측된 확률변수 $X: x_1, x_2, ... x_n$
- $X$에 대응하는 잠재변수 $Z: z_1, z_2, ... z_n$
- Complete data: $\{(x_1, z_1), (x_1, z_1), \space ... \space, (x_n, z_n)\}$ 
- Incomplete data: $\{x_1, x_2, \space ... \space , x_n\}$
- $X$ 와 $Z$의 결합확률분포 모델: $p(x, z \vert \theta)$

# 2. ELBO

우리의 목적은 데이터의 로그 가능도를 최대화하는 파라미터 $\theta$ 를 찾는 것이다. 이 때 하나의 분포만으로는 데이터를 설명하기 어렵기 때문에 잠재변수 $Z$를 도입하여 가우시안 혼합 모형을 전제하고자 한다. $X$ 와 $Z$의 결합확률분포 모델을 $p(x, z \vert \theta)$라고 표현하자. 그렇다면 $p(x)$는 결합분포에서 주변합을 구한 형태로 나타낼 수 있다. 이러한 폼을 베이즈 정리에서 **Evidence** 라고 부른다. 하지만 대부부의 경우 **Evidence** 를 직접 최대화하는 것은 어렵다. 그래서 **Evidence의 하한인 ELBO(Evidence Lower Bound)를 최대화하는 방식을 택한다.** 이후 **ELBO를 최대화하는 파라미터는 곧 Evidence 를 최대화하는 파라미터와 같다**는 것을 보일 수 있다. 데이터 전체에 대한 합은 생략하고 적어보도록 하자.

$$
arg \space max_{\theta} \log p(x \vert \theta) \\
p(x \vert \theta) = \sum_{z}p(x, z \vert \theta)\\
$$

잠재변수 $Z$의 PMF를 $q$라 하면 ELBO는 다음과 같이 유도할 수 있다. 로그는 오목(concave)함수이므로 젠슨 부등식에 의해 로그의 기대값은 기대값의 로그보다 작거나 같다. 데이터 $x$가 주어진 상황에서 ELBO는 $q$와 $\theta$ 의 함수이므로 $\mathcal{L}(q, \theta)$ 라 하자. 어떤 PMF $q(z)$에 대해서도 $\mathcal{L}(q, \theta)$ 는 Evidence 보다 작거나 같다. 목적함수를 정의하였으므로 이제 이를 최대화하는 알고리즘을 기술해보도록하자.

$$
\begin{align} 
\log p(x \vert \theta) &= \log \left[ \sum_{z}p(x, z \vert \theta) \right] \\ &=
\log \left[ \sum_{z} q(z) \frac{p(x, z \vert \theta)}{q(z)}) \right] \quad \text{log of an expectation} \\  & \geq
\underbrace{ \sum_{z} q(z) \left[ \log \frac{p(x, z \vert \theta)}{q(z)}) \right] }_{ ELBO \space \mathcal{L}(q, \theta) } \quad \text{expectation of log, by Jensen's inequality}
\end{align}
$$

$$
\therefore \text{For any q(z), } \log p(x \vert \theta) \geq \mathcal{L}(q, \theta) 
$$

# 3. Expectation-Maximization

EM 알고리즘은 다음과 같이 진행된다.

1. 파라미터 $\theta^{old}$ 를 랜덤하게 초기화한다.
2. $q^{\ast} = arg \space max_{q} \mathcal{L}(q, \theta^{old})$ 를 찾는다.
3. $\theta^{new} = arg \space max_{\theta} \mathcal{L}(q*, \theta)$ 를 찾는다.
4. 로그우도가 수렴하지 않았다면 $\theta^{old} \leftarrow \theta^{new}$ 로 놓고 2번으로 돌아간다.

여기에서 언제나 $\log p(x \vert \theta^{new}) \geq \log p(x \vert \theta^{old})$ 임을 보일 수 있다. **즉 알고리즘의 진행에 따라 로그우도는 단조증가한다.** 그 이유를 알기 위해서 ELBO의 수식을 조금 다른 형태로 뜯어볼 필요가 있다.

## 3.1. Maximizing over $q$ for fixed $\theta = \theta^{old}$

알고리즘의 2번 스텝에서는 고정된 $\theta^{old}$ 를 통해 ELBO를 최대화하는 PMF $q(z)$를 찾는다. ELBO 수식을 약간 변형하면 Negative KL Divergence 와 Evidence의 합으로 분해된다. Evidence 항은 $z$를 주변화시킨 확률이므로 $q$가 관여하지 않는다. 즉 우리는 KL Divergence를 최소화하는 $q$를 찾는 것이다. KL Divergence는 두 분포가 같을 때 최소값을 가지므로 $q(z) = p(z \vert x, \theta^{old})$ 이고 이때 KL Divergence의 값은 0이므로 결국 $\mathcal{ L }(q^{\ast}, \theta^{old}) = \log p(x \vert \theta)$ 가 된다. 앞에서 ELBO는 **Evidence의 하한**이라고 했다. 따라서 $q^{\ast}$ 에서 ELBO와 Evidence의 값이 같아진다는 것은 $q^{\ast}$ 에서 두 함수가 접한다는 뜻이 된다.

$$
\begin{align}
\mathcal{L}(q, \theta) &=
 \sum_{z} q(z) \left[ \log \frac{p(x, z \vert \theta)}{q(z)} \right] \\ &=
 \sum_{z} q(z) \left[ \log \frac{p(z \vert x, \theta) p(x \vert \theta)}{q(z)}) \right] (\text{conditional probability})\\ &=
 \sum_{z} q(z) \log \frac{p(z \vert x, \theta)}{q(z)}) + \sum_{z} q(z)\log p(x \vert \theta) \\ &=
  -KL \left[ q(z) \vert \vert p(z \vert x, \theta) \right] + \underbrace { \log p(x \vert \theta) }_{\text{no q here}} \space
\end{align}
$$


$$
\therefore q^{\ast} = arg \space max_q \mathcal{L}(q, \theta^{old}) \implies \mathcal{L(q^{\ast}, \theta^{old})} = \log p(x \vert \theta^{old})
$$


## 3.2. Maximizing over $\theta$ for fixed $q^{\ast}$


다음으로는 $arg \space max_{theta} \mathcal{L}(q^{\ast}, \theta)$ 를 찾아야 한다. ELBO 수식을 변형하면 $\theta$가 관여하지 않는 둘째 항은 무시할 수 있다. 첫째 항은 $X$와 $Y$의 결합분포, 즉 완전한 데이터(complete data)의 로그우도 기대값이다. 즉 ELBO를 최대화하는 것은 완전한 데이터의 로그우도 기대값을 최대화하는 것과 같다. $q^{\ast}$를 찾아 완전한 데이터의 로그우도 기대값 수식을 세우는 과정을 E스텝, 이 기대값을 최대화하는 $\theta^{new}$ 를 찾는 3번 스텝을 M스텝이라고 부른다. 잠재변수 $Z$의 값을 정확히 모르기 때문에 주어진 PMF $q^{\ast}(z)$ 에서의 기대값을 사용한다고 보면 될 것 같다.


$$
\begin{align}
\mathcal{L}(q, \theta) &=
 \sum_{z} q(z) \left[ \log \frac{p(x, z \vert \theta)}{q(z)} \right] \\ &=
 \underbrace { \sum_{z} q(z) \log p(x, z \vert \theta) }_{\text{expectation of complete data log-lokelihood}} - \underbrace { \sum_{z} q(z) \log q(z) }_{\text{no theta here}} \\ &
\therefore \theta^{new} = arg \space max_\theta \sum_{z} q^{\ast}(z) \log p(x, z \vert \theta)
\end{align}
$$


## 3.3. EM Gives Monotonically Increasing Likelihood

지금까지 논의한 결과를 종합하면 다음 부등식이 성립한다. 즉 EM 알고리즘의 이터레이션에 따라서 로그우도가 단조증가함을 보일 수 있다.


$$
\begin{align}
\log p(x \vert \theta^{new}) &\geq   \mathcal{L}(q^{\ast}, \theta^{new}) \quad \mathcal{L}\text{ is a lower bound of evidence} \\ &\geq
\mathcal{L}(q^{\ast}, \theta^{old})\quad \text{By definition of } \theta^{new} \\ &=
\log p(x \vert \theta^{old}) \quad \text{Lower bound of evidence is tight at } q^{\ast}
\end{align}
$$


## 3.4. Global Maximum

또한 ELBO의 글로벌 맥시멈 $\theta^\ast$이 로그우도의 글로벌 맥시멈과 같다는 사실을 보일 수 있다. ELBO의 글로벌 맥시멈을 $\mathcal{L}(q^{\ast}, \theta^\ast)$ 라 하자. 전역 최대이므로 다음이 성립한다. 위에서는  $q$를 업데이트할 때 $q^{\ast}(z) = p(z \vert x, \theta^{old})$ 로 썼지만 글로벌 맥시멈에서는 결국 $\theta$가 고정되므로 $q^{\ast}(z) = p(z \vert x, \theta^\ast)$ 가 된다.


$$\mathcal{L}(q^{\ast}, \theta^\ast) \geq \mathcal{L}(q, \theta), \space \forall (q, \theta) \quad where \quad q^{\ast}(z) = p(z \vert x, \theta^\ast) $$


이 때 $\theta^\ast$은  ELBO 뿐만 아니라 Evidence의 전역 최대점이기도 하다. 임의의 파라미터 $\theta′$와 이에 대해 $q′(z) = p(z \vert x, \theta′)$ 을 만족하는 $q′$ 를 가정하자. 이 경우 Lower bound가 주어진 $\theta′$ 에 대해 최대화된 상태로 더 이상 증가할 수 없으며 Evidence와 접해 있다. 즉 둘은 같은 값을 가진다. 이후 전역 최대점에 대해서 $\mathcal{L}(q^{\ast}, \theta^\ast) \geq \mathcal{L}(q^∗, \theta^∗)$ 가 성립한다. 마지막으로 $\mathcal{L}(q^∗, \theta^∗)$ 는 타이트한 ELBO 이므로 Evidence와 같다.


$$
\begin{align}
\log p(x|\theta′) &= \mathcal{L}(q′, \theta′) + \underbrace { KL[q′,p(z|x, \theta′)] }_{\text{equals to 0}} \\ &=
\mathcal{L}(q′,\theta′) \\ & \leq
\mathcal{L}(q^∗, \theta^∗) \quad \text{Global maximum} \\ &=
\log p(x \vert \theta^∗) \quad \text{Lower bound is tight}
\end{align} \\
$$



# 4. Mixture of Gaussians

마지막으로, 위에서 가우시안 혼합 모형의 모수를 실제로 추론해보자. 각 클러스터의 비율과 모수를 나름대로 가깝게 추론해내는 것을 볼 수 있다. ELBO의 전역 최대가 결국은 로그우도의 전역 최대와 같기는 하지만, 알고리즘이 항상 전역 최대를 찾아갈 수 있는가는 다른 문제이다. EM 알고리즘은 수렴이 보장되지만 그것이 전역 최대라는 보장이 없다. 즉 초기값을 어떻게 설정하는지에 따라 알고리즘의 수렴 결과가 달라질 수 있다. 참고한 앤드류 응 교수의 강의에서는 K-Means 알고리즘을 예시도 랜덤 센트로이드에서 시작하기보다 데이터에서 초기값을 뽑아내는 것을 추천한다. 이 코드에서도 유사한 방법을 따랐다.


```python
np.random.seed(2021)

x1 = np.random.normal(1, 1, 100)
x2 = np.random.normal(10, 3, 50)
x = np.append(x1, x2)
np.random.shuffle(x)
x = x.reshape((-1, 1))
```


```python
# 정규분포의 pdf
def pdf(x, mean, std):
    return np.exp(-((x-mean)**2)/(2 * (std**2))) / (std * (2 * np.pi)**0.5)
```


```python
# Evidence 의 로그우도
def log_likelihood(x, mu, sigma, p):
    return np.log((pdf(x, mu, sigma) * p).sum(axis=1)).sum().round(4)
```

## 4.1. Choose initial $\theta^{old}$


```python
p = np.array([[0.5, 0.5]], dtype=np.float64)
mu = np.random.choice(x.flatten(), 2, replace=True)[np.newaxis, :]
sigma = np.array([[np.std(x), np.std(x)]], dtype=np.float64)
logL = log_likelihood(x, mu, sigma, p).round(4)
eps = 1e-6
```

## 4.2. Expectation - Maximization


```python
density = pdf(x, mu, sigma)
dp = density * p
pz_x = (dp / dp.sum(axis=1, keepdims=True)).sum(axis=0, keepdims=True)
```


```python
row_format = "{:^15}|{:^15}|{:^15}|{:^15}|{:^15}|{:^15}|{:^15}|{:^15}"
print(row_format.format("iteration", "log-likelihood", "P(Z=0)", "P(Z=1)", "mu_1", "mu_2", "sigma_1", "sigma_2"))
print(row_format.format("=" * 15, "=" * 15, "=" * 15, "=" * 15, "=" * 15, "=" * 15, "=" * 15, "=" * 15))


for i in range(100):
    
    density = pdf(x, mu, sigma)
    dp = density * p
    pz_x = (dp / dp.sum(axis=1, keepdims=True))
    
    p = pz_x.mean(axis=0)[np.newaxis, :].round(4)
    sigma = np.sqrt((pz_x * (x - mu)**2).sum(axis=0) / pz_x.sum(axis=0))[np.newaxis, :].round(4)
    mu = ((x * pz_x).sum(axis=0) / pz_x.sum(axis=0))[np.newaxis, :].round(4)
    
    logL_new = log_likelihood(x, mu, sigma, p).round(4)
    gain = logL_new - logL
    
    assert gain >= 0
    
    print(row_format.format(i+1, logL, p[0][0], p[0][1], mu[0][0], mu[0][1], sigma[0][0], sigma[0][1]))
    
    if gain < eps:
        
        
        print(row_format.format("=" * 15, "=" * 15, "=" * 15, "=" * 15, "=" * 15, "=" * 15, "=" * 15, "=" * 15))
        print(row_format.format("-", logL, p[0][0], p[0][1], mu[0][0], mu[0][1], sigma[0][0], sigma[0][1]))
        
        print("\nAlgorithm Converged!")
        
        break
    
    logL = logL_new
```

```
       iteration   |log-likelihood |    P(Z=0)     |    P(Z=1)     |     mu_1      |     mu_2      |    sigma_1    |    sigma_2    
    ===============|===============|===============|===============|===============|===============|===============|===============
           1       |   -459.966    |    0.5786     |    0.4214     |    1.7004     |    8.0079     |    2.6411     |    5.1511     
           2       |   -418.9214   |    0.5904     |    0.4096     |    1.3054     |     8.759     |    1.5495     |    4.7224     
           3       |   -382.1583   |    0.6168     |    0.3832     |     1.102     |    9.5997     |    1.0135     |    4.0415     
           4       |   -360.4603   |    0.6368     |    0.3632     |    1.0722     |    10.1216    |    0.9333     |     3.429     
           5       |   -356.5254   |    0.6489     |    0.3511     |    1.0748     |    10.4287    |    0.9394     |    3.0157     
           6       |   -354.8216   |    0.6553     |    0.3447     |    1.0845     |    10.5818    |    0.9495     |    2.8023     
           7       |   -354.3142   |    0.6576     |    0.3424     |    1.0901     |    10.6362    |    0.9549     |    2.7281     
           8       |   -354.2459   |    0.6583     |    0.3417     |    1.0921     |    10.6517    |     0.957     |     2.708     
           9       |   -354.2402   |    0.6585     |    0.3415     |    1.0926     |    10.6558    |    0.9576     |    2.7028     
          10       |   -354.2398   |    0.6585     |    0.3415     |    1.0928     |    10.6569    |    0.9578     |    2.7015     
    ===============|===============|===============|===============|===============|===============|===============|===============
           -       |   -354.2398   |    0.6585     |    0.3415     |    1.0928     |    10.6569    |    0.9578     |    2.7015     
    
    Algorithm Converged!
```

## 참고자료

- [Lecture 14 - Expectation-Maximization Algorithms \| Stanford CS229: Machine Learning (Autumn 2018)](https://www.youtube.com/watch?v=rVfZHWTwXSA&t=2192s)
- [27. EM Algorithm for Latent Variable Models](https://www.youtube.com/watch?v=lMShR1vjbUo)
