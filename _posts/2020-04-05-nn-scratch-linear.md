---
layout: post
title: "[Python] NeuralNet from Scratch(1)"
categories: [doc]
tags: [python, ml]
comments: true
---

넘파이를 사용해서 선형 모형부터 역전파 알고리즘까지 차례차례 구현해봅니다! 간만에 시간이 생겨서 기초적인 신경망 이론을 복습하다가 [바람의 머신러닝](https://www.youtube.com/watch?v=xgT9xp977EI)이라는 강의를 우연히 보게 되었는데, 기존의 수식적인 접근에 더해 계산 그래프를 활용하여 상당히 쉽게 설명을 잘 해주셨습니다. 무엇보다도 강사님 목소리가 굉장히 좋으세요 ㅎㅎ 개인적으로는 역전파 부분만 시청했는데 아예 신경망을 처음 접하시는 분들이라면 모든 강의 다 들어보셔도 좋을 것 같습니다! 

## 1. 회귀모형: 선형 모델과 경사하강

### 1.1. 선형 출력 구현하기

우선 경사하강이라는 알고리즘의 이해를 위해서 간단한 선형 모형부터 시작해봅시다(선형 모형에 대한 설명은 생략합니다). 입력을 받아서 선형 출력을 내보내는 모델을 만들어보는 것이 목표입니다. 사용할 데이터셋은 scikit learn의 보스턴 데이터이고, 스케일링만 해주었습니다. 학습 데이터와 평가 데이터를 따로 분리하지는 않았습니다. 학습 과정에서 훈련 오차를 출력하기 위해서 `mse` 함수를 정의합니다.


```python
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

boston = load_boston()
x, y = StandardScaler().fit_transform(boston['data']), boston['target'].reshape((-1, 1))

def mse(y, yhat):
    return np.sqrt(np.power(y - yhat, 2).mean())
```

그러면 선형모형 클래스를 만들어 입력을 받아서 출력을 내보내는 기능까지만 구현해보겠습니다. `__init__` 메소드에서는 인스턴스를 생성하면 입력 데이터의 사이즈를 받아서 가중치 행렬과 편향을 초기화합니다. 가중치는 랜덤으로, 편향은 0으로 초기화하였습니다. 다음으로 예측을 내보내는 `predict` 메소드를 구현해봅니다. 예측은 단순히 입력과 가중치의 곱에 편향을 더해서 내보내부는 선형 결합입니다. 아직 가중치와 편향의 업데이트를 실행하는 경사하강 알고리즘은 구현하지 않았습니다.


$$y = x^T w + b$$


```python
class LinearRegression:
    def __init__(self, input_size):
        self.input_size = input_size # 입력 데이터의 사이즈(피쳐 갯수)
        self.weight = np.random.randn(input_size, 1) # 가중치 초기화: 랜덤
        self.bias = np.zeros((1,)) # 편향 초기화: 0
        
    def predict(self, x):
        return np.matmul(x, self.weight) + self.bias # 선형 결합으로 출력
```

### 1.2. 경사하강 구현하기

출력을 내보내는 다중회귀모형이 간단하게 구현되었습니다. 이제 가중치 업데이트를 위한 경사하강을 구현해봅시다. 선형회귀 클래스 안에 경사하강을 위한 gradientDescent 메소드를 별도로 구현하였습니다. 학습을 위해 필요한 데이터와 학습률을 인자로 받습니다. 먼저 출력 `yhat`을 계산해줍니다. 다음으로 가중치 업데이트를 위한 미분을 계산합니다. 체인 룰을 사용해서 간단하게 계산해줄 수 있습니다. 이후 역전파를 다룰 때에도 체인 룰은 계속 등장합니다. 개별 데이터마다 미분을 계산하는 것이 아니라 행렬을 사용해서 한번에 계산한 후 업데이트 해줍니다.
    
$$ L = \frac{1}{2} (y-\hat{y})^2, \space \frac{L}{\partial \hat{y}} = -(y-\hat{y})$$

$$\frac{\partial \hat{y}}{\partial w_i} = x_i, \space \frac{\partial \hat{y}}{\partial b} = 1$$

```python
class LinearRegression::
    ...생략...
    def gradientDescent(self, x, y, learning_rate):
        yhat = self.predict(x) # 출력 계산
        
        grad = (yhat - y)
        grad_weight = np.matmul(x.T, grad) / y.shape[0]
        grad_bias = grad.mean()
        
        self.weight -= learning_rate * grad_weight
        self.bias -= learning_rate * grad_bias
        
        return yhat
```


다음으로 경사하강을 반복적으로 수행해줄 메소드를 구현합니다. 반복수와 학습률을 인자로 받아 정해진 에포크만큼 경사하강을 반복합니다. 100바퀴마다 학습 오차를 출력하도록 하였습니다.


```python
class LinearRegression:
    ...생략...
    def train(self, x, y, epochs=10, learning_rate = 0.001):
        print(f"Training on {x.shape[0]} samples ...")
        
        for epoch in range(epochs):
            yhat = self.gradientDescent(x, y, learning_rate = learning_rate)
            if epoch % 200 == 0:
                train_loss = mse(y, yhat)
                print(f"[{epoch}] train_loss: {train_loss}")
```

### 1.3. 모델 훈련

이제 실제로 모델을 만들고 학습을 진행해봅시다. 학습률 0.01로 1,000바퀴 학습을 진행하였습니다. 오차가 안정적으로 줄어드는 것을 확인할 수 있습니다.

```python
linear = LinearRegression(input_size = x.shape[1])
linear.train.(x, y, epochs=1000, learning_rate=0.01)
print("MSE:", mse(y, linear.predict(x)))
```

```bash
(base) jhgan@jhgan-ThinkPad-E595:~$ python LinearRegression.py
Training on 506 samples ...
[0] train_loss: 26.01388262584
[200] train_loss: 5.870023866355405
[400] train_loss: 4.853665990364972
[600] train_loss: 4.771235781577091
[800] train_loss: 4.7365009309260255
MSE: 4.716836782322638
```



## 2. 분류 모형: 다중 분류기

### 2.1. 소프트맥스 출력 구현하기

위에서 만든 선형회귀 모델을 변형하여 다중 분류기를 구현해봅시다. 아이리스 데이터를 사용해서 진행합니다. 크로스 엔트로피 함수를 정의하고, 계산을 쉽게 하기 위해서 정답인 y를 원 핫 인코딩해줍니다.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

iris = load_iris()
x, y = StandardScaler().fit_transform(iris['data']), pd.get_dummies(iris['target']).values

def cross_entropy(y, yhat):
    return -(y * np.log(yhat)).mean()
```

이제 입력을 받아서 소프트맥스 출력을 내보내는 클래스를 구현해봅시다. 소프트맥스는 각 클래스별 확률을 출력해야 하므로 우선 가중치 행렬을 $$(input, \space output)$$로 확장해줍니다. 이렇게 해서 데이터마다 클래스 각각의 확률을 출력할 수 있습니다. 역시 가중치는 랜덤으로, 편향은 0으로 초기화해주었습니다. 

$$z_i = x^T c_i(W), \space p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

```python
class SoftmaxClassifier:
    
    def __init__(self, input_size, n_labels):
        self.input_size = input_size # 입력 데이터의 사이즈(피쳐 갯수)
        self.weight = np.random.randn(input_size, n_labels) # 가중치 행렬 확장: 클래스 갯수만큼
        self.bias = np.zeros((n_labels,)) # 편향 초기화: 0
        
    def predict(self, x, return_x=False):
        logit = np.matmul(x, self.weight) + self.bias # 선형결합 계산
        softmax = np.exp(logit)/ np.exp(logit).sum(axis=1).reshape((x.shape[0],1)) # 소프트맥스 계산
        return softmax
```

### 1.2. 경사하강 구현하기
    
출력값과 오차 함수가 변경되었으므로 미분도 그에 맞게 변경해줍니다. 이후 파라미터를 업데이트해주는 과정은 모두 같습니다. 크로스 엔트로피 에러와 소프트맥스 함수의 조합은 상당히 간단한 미분 결과를 만들어줍니다. 역시 개별 데이터마다 연산을 반복하는 것이 아니라 행렬로 처리한다는 점을 주의해주시면 됩니다.

$$L = -\sum y_i ln p_i = -y ln p$$

$$\frac{\partial p}{\partial z} = p(1-p), \space \frac{\partial L}{\partial p} = -\frac{y}{p}, \space \frac{\partial L}{\partial z} = -y(1-p)$$

```python
class SoftmaxClassifier:
    ...생략...
    def gradientDescent(self, x, y, learning_rate):
            yhat = self.predict(x)
            grad = -1 *(y - yhat) * y # d error/d softmax
            grad_weight = np.matmul(x.T, grad) / y.shape[0] # d error / d weight
            grad_bias = grad.mean(axis=0) # d error / d bias
            
            self.weight -= learning_rate * grad_weight
            self.bias -= learning_rate * grad_bias
            
            return yhat
                
    def train(self, x, y, epochs=100, learning_rate = 0.001):
        print(f"Training on {x.shape[0]} samples ...")
        for epoch in range(epochs):
            yhat = self.gradientDescent(x, y, learning_rate = learning_rate)
            if epoch % 100 == 0:
                train_loss = cross_entropy(y, yhat)
                print(f"[{epoch}] train_loss: {train_loss}")
```

### 1.3. 모델 훈련

마지막으로 모델을 만들고 학습을 진행해봅시다. 학습률 0.01로 1,500바퀴 학습을 진행하였습니다. 역시 오차가 안정적으로 줄어드는 것을 확인할 수 있습니다.

```python
model = SoftmaxClassifier(input_size = x.shape[1], n_labels = 3)
model.train(x, y, epochs=1500, learning_rate=0.01)
yhat = model.predict(x)

print("Cross Entropy:", cross_entropy(y, yhat))

cm = confusion_matrix(
    y.argmax(axis=1),
    yhat.argmax(axis=1)
)

print(cm)
```

```bash
(base) jhgan@jhgan-ThinkPad-E595:~$ python SoftmaxClassifier.py
Training on 150 samples ...
[0] train_loss: 0.7867740033156846
[100] train_loss: 0.5146303389616008
[200] train_loss: 0.36702948706265615
[300] train_loss: 0.28394572557080633
[400] train_loss: 0.2302256535632078
[500] train_loss: 0.19235148637889693
[600] train_loss: 0.16474832722275634
[700] train_loss: 0.14444293000818012
[800] train_loss: 0.1294332290038627
[900] train_loss: 0.11823812915590903
[1000] train_loss: 0.10976198836207084
[1100] train_loss: 0.10321606755036845
[1200] train_loss: 0.09804614615055966
[1300] train_loss: 0.09386801611162929
[1400] train_loss: 0.09041613762520748
Cross Entropy: 0.08750602639805916
[[49  1  0]
 [ 0 41  9]
 [ 0  1 49]]

```