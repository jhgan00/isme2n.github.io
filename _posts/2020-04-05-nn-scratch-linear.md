---
layout: post
title: "[Python] NeuralNet from Scratch(1)"
categories: [doc]
tags: [python]
comments: true
---

넘파이를 사용해서 선형 모형부터 역전파 알고리즘까지 차례차례 구현해봅니다!

## 1. 회귀모형: 선형 모델과 경사하강

우선 입력을 받아서 선형 출력을 내보내는 모델을 만들어봅시다. 사용할 데이터셋은 scikit learn의 보스턴 데이터이고, 스케일링만 해주었습니다.


```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

np.random.seed(0)

boston = load_boston()
x, y = StandardScaler().fit_transform(boston['data']), boston['target'].reshape((-1, 1))
```

선형회귀모형 클래스를 만들고 입력을 받아서 출력을 내보내는 기능까지만 구현해보겠습니다. 인스턴스를 생성하면 입력 데이터의 사이즈를 받아서 가중치 행렬과 편향을 초기화합니다. 가중치는 랜덤으로, 편향은 0으로 초기화하였습니다. 다음으로 예측을 내보내는 `predict` 메소드를 구현해봅니다. 예측은 단순히 입력과 가중치의 곱에 편향을 더해서 내보내부는 선형 결합입니다.


$$Y = XW + b \space \space$$




```python
def mse(y, yhat):
    return np.sqrt(np.power(y - yhat, 2).mean())
```


```python
class LinearRegression:
    def __init__(self, input_size):
        self.input_size = input_size # 입력 데이터의 사이즈(피쳐 갯수)
        self.weight = np.random.randn(input_size, 1) # 가중치 초기화: 랜덤
        self.bias = np.zeros((1,)) # 편향 초기화: 0
        
    def predict(self, x):
        return np.matmul(x, self.weight) + self.bias # 선형 결합으로 출력
```

출력을 내보내는 다중회귀모형이 간단하게 구현되었습니다. 이제 가중치 업데이트를 위한 경사하강을 구현해봅시다. 

1. gradientDescent

    경사하강을 위한 gradientDescent 메소드를 별도로 구현하였습니다. 학습을 위해 필요한 데이터와 학습률을 인자로 받습니다. 먼저 출력 `yhat`을 계산해줍니다. 다음으로 가중치 업데이트를 위한 미분을 계산합니다. 에러를 $$\frac{1}{2}MSE$$ 로 놓고 미분을 계산한 다음 모든 케이스의 평균을 구해줍니다. 이렇게 구한 그래디언트를 학습률에 곱해 원래 파라미터들에 더해주면 끝입니다! 중간중간 학습 진행 상황을 출력하기 위해 출력값 `yhat`을 반환합니다.


2. train

    경사하강을 반복적으로 수행하는 메소드입니다. 반복수와 학습률을 인자로 받아 정해진 에포크만큼 경사하강을 반복합니다. 100바퀴마다 학습 오차를 출력하도록 하였습니다.


```python
class LinearRegression:
    def __init__(self, input_size):
        self.input_size = input_size # 입력 데이터의 사이즈(피쳐 갯수)
        self.weight = np.random.randn(input_size, 1) # 가중치 초기화: 랜덤
        self.bias = np.zeros((1,)) # 편향 초기화: 0
        
    def predict(self, x):
        return np.matmul(x, self.weight) + self.bias # 선형 결합으로 출력
    
    def gradientDescent(self, x, y, learning_rate):
        yhat = self.predict(x) # 출력 계산
        
        grad = y - yhat
        grad_weight = np.matmul(x.T, grad) / y.shape[0]
        grad_bias = grad.mean()
        
        self.weight += learning_rate * grad_weight
        self.bias += learning_rate * grad_bias
        
        return yhat
        
    def train(self, x, y, epochs=10, learning_rate = 0.001):
        print(f"Training on {x.shape[0]} samples ...")
        
        for epoch in range(epochs):
            yhat = self.gradientDescent(x, y, learning_rate = learning_rate)
            if epoch % 200 == 0:
                train_loss = mse(y, yhat)
                print(f"[{epoch}] train_loss: {train_loss}")
```

이제 실제로 모델을 만들고 학습을 진행해봅시다. 학습률 0.01로 1,000바퀴 학습을 진행하였습니다.


```python
linear = LinearRegression(input_size = x.shape[1])
linear.train(x, y, epochs = 1000, learning_rate = 0.01)
```

    Training on 506 samples ...
    [0] train_loss: 26.01388262584
    [200] train_loss: 5.870023866355405
    [400] train_loss: 4.853665990364972
    [600] train_loss: 4.771235781577091
    [800] train_loss: 4.7365009309260255



```python
mse(y, linear.predict(x) )
```




    4.716836782322638



## 2. 분류 모형: 다중 분류기

위에서 만든 선형회귀 모델을 변형하여 다중 분류기를 구현해봅시다. 아이리스 데이터를 사용해서 진행합니다. 크로스 엔트로피 계산을 쉽게 하기 위해서 정답인 y를 원 핫 인코딩해줍니다.


```python
from sklearn.datasets import load_iris
iris = load_iris()
x, y = StandardScaler().fit_transform(iris['data']), pd.get_dummies(iris['target']).values
```

1. \_\_init\_\_

    각 클래스별 확률을 출력해야 하므로 우선 가중치 행렬을 확장해줍니다. 이렇게 해서 데이터마다 3개의 클래스 각각 확률을 출력할 수 있습니다.

$$Y = softmax(WX + b)$$

2. predict

    다음으로 신경써주어야 할 부분은 출력단에서 소프트맥스 함수를 씌워주는 것입니다. `predict` 메소드를 수정하여 먼저 선형결합을 계산한 후, 소프트맥스를 계산하여 출력합니다.
    
    
3. gradientDescent

    함수의 구조가 변경되었으므로 미분값도 그에 맞게 변경해줍니다. 이후 파라미터를 업데이트해주는 과정은 모두 같습니다.


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
        
    def gradientDescent(self, x, y, learning_rate):
            yhat = self.predict(x)
            grad = (y - yhat) * y # d error/d softmax
            grad_weight = np.matmul(x.T, grad) / y.shape[0] # d error / d weight
            grad_bias = grad.mean(axis=0) # d error / d bias
            
            self.weight += learning_rate * grad_weight
            self.bias += learning_rate * grad_bias
            
            return yhat
                
    def train(self, x, y, epochs=100, learning_rate = 0.001):
        print(f"Training on {x.shape[0]} samples ...")
        for epoch in range(epochs):
            yhat = self.gradientDescent(x, y, learning_rate = learning_rate)
            if epoch % 100 == 0:
                train_loss = cross_entropy(y, yhat)
                print(f"[{epoch}] train_loss: {train_loss}")
```


```python
model = SoftmaxClassifier(input_size = x.shape[1], n_labels = 3)
```


```python
model.train(x, y, epochs=500, learning_rate=0.01)
```

    Training on 150 samples ...
    [0] train_loss: 0.8168613260040681
    [100] train_loss: 0.5547144619357202
    [200] train_loss: 0.42357290723074265
    [300] train_loss: 0.34728003215281744
    [400] train_loss: 0.29499750970728744



```python
confusion_matrix(
    y.argmax(axis=1),
    model.predict(x).argmax(axis=1)
)
```




    array([[49,  1,  0],
           [ 0, 24, 26],
           [ 0,  0, 50]])


