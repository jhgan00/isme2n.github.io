---
layout: post
title: "[Python] NeuralNet from Scratch(2)"
categories: [doc]
tags: [python, ml]
comments: true
---

이제 본격적으로 신경망과 역전파 알고리즘을 구현해봅시다. 사용할 데이터는 MNIST 손글씨 데이터입니다. [Building an Artificial Neural Network using pure Numpy](https://towardsdatascience.com/building-an-artificial-neural-network-using-pure-numpy-3fe21acc5815)을 참고하여 만들었습니다.


```python
import numpy as np
import pandas as pd
from keras.datasets import mnist

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

xtrain, xtest = xtrain.reshape([xtrain.shape[0], -1])/255, xtest.reshape([xtest.shape[0], -1])/255
ytrain, ytest = pd.get_dummies(ytrain).values, pd.get_dummies(ytest).values

np.random.seed(0)
```

    Using TensorFlow backend.



```python
def cross_entropy(y, yhat, eps=1e-10):
    yhat = np.clip(yhat, eps, 1-eps)
    return -(y * np.log(yhat)).sum(axis=1).mean()
```

## 1. 순전파 과정

### 1.1. 선형 출력 구현

먼저 완전 연결 레이어를 표현할 `Dense` 클래스를 정의해주겠습니다. 일단은 활성화 함수 없이 선형 출력만으로 `Dense` 레이어를 구성하고, 활성화 함수 클래스는 별도로 만들어주도록 하겠습니다. 이렇게 생각하면 앞에서 만들었던 선형 모델과 다를 것이 없습니다. 즉 가중치와 편향을 초기화한 후에 입력을 받아서 출력을 내보내는 기능을 구현하면 됩니다. 우선 여기까지만 진행해보겠습니다. 참고한 글에서 가중치 초기화를 할 때 0.01을 곱해주는데, 웨이트를 조정해주는 편이 잘 수렴하는 것 같네요 ㅎㅎ


```python
class Dense:
    # input_size: n_feature
    def __init__(self, input_size, output_size):
        self.weight = np.random.randn(input_size, output_size) * 0.1 # 가중치 초기화
        self.bias = np.zeros(output_size) # 편향 초기화
    
    def forward(self, input):
        self.input = input
        output = np.matmul(input, self.weight) + self.bias # 출력: xw + b
        return output
```

앞에서 만들었던 선형 모델과 다른 점은, 뒤에 이어질 레이어가 갖는 노드 개수만큼 가중치 행렬을 확장해준다는 점입니다. 즉 레이어가 갖는 가중치 행렬은 $(input, output)$ 크기가 됩니다. 예를 들어서 우리가 MNIST 데이터를 분류하려고 한다면, 마지막 레이어에서는 총 10개의 출력을 내보내주면 될 것입니다. `forward` 메소드는 입력을 받아서 선형 출력을 내보내는 메소드입니다. 이 결과를 활성화 함수 클래스에 전달하여 다음 레이어로 나가는 출력을 만들어줄 것입니다.

### 1.2. 활성화 함수 구현

이번에는 활성화 클래스들을 만들어보겠습니다. 우선 `Activation` 클래스를 정의합니다. 이 클래스의 `forward` 메소드를 통해 순전파 출력을 만들어냅니다. `forward` 메소드는 `activate` 메소드를 호출하고, `activate` 메소드는 활성도를 계산합니다(아직 어떤 활성화 함수를 사용할지 모르므로 pass로 남겨둡니다). 이 결과를 `self.output` 속성에 저장한 후 반환합니다. 따라서 이 클래스를 상속받는 다른 클래스들은 `activate` 메소드를 오버라이딩하여 다른 활성화 함수들을 사용할 수 있습니다. 최종적으로 다중 분류기를 만드는 것이 목적이므로 우선 `Softmax` 클래스를 만들고, 최종 출력을 제외한 단계에서 사용할 `Sigmoid`와 `ReLU`도 만들어줍니다. 사실 구현하고보니 시그모이드 클래스는 사용하지 않았네요..ㅎㅎ 나중에 쓰고싶은 마음이 생길지도 모르니 만들어봅니다.


```python
# 활성층 클래스
class Activation:
    def __init__(self):
        self.output = None
    
    # 활성화 함수
    def activate(self, input):
        pass
    
    # 순전파
    def forward(self, input):
        output = self.activate(input)
        self.output = output
        return output

## Activation 클래스를 상속

# 소프트맥스
class Softmax(Activation):    
    def activate(self, input):
        self.input = input
        return np.exp(input) / np.exp(input).sum(axis=1, keepdims=True) # 소프트맥스 함수

# 시그모이드
class Sigmoid(Activation):
    def activate(self, input):
        self.input = input
        return 1 / (1 + np.exp(-input)) # 시그모이드 함수
    
# 렐루
class ReLU(Activation):
    
    def activate(self, input):
        self.input = input
        return np.maximum(0, input) # 렐루
```
### 1.3. 모델 구현

이번에는 여러 레이어를 포함하는 최종 모델 클래스를 만들어보겠습니다. 모델을 생성할 때는 여러 레이어 객체를 포함하는 반복 가능한 객체를 받습니다. `predict` 메소드에서는 반복문을 통해 순전파를 일으키고, 마지막 레이어의 `output` 속성을 반환합니다.


```python
class Model:
    def __init__(self, layers):
        self.layers = layers
                
    def predict(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return self.layers[-1].output # 마지막 레이어의 출력
```

만들어진 모델에서 순전파가 잘 작동하는지 점검해봅시다. 만들어둔 레이어를 순서대로 쌓아주고, `predict` 메소드를 실행해보겠습니다. 결과가 소프트맥스 확률로 잘 출력되는 것 같습니다.


```python
model = Model([
    Dense(xtrain.shape[1], 128),ReLU(),
    Dense(128, 128),Sigmoid(),
    Dense(128, 10), Softmax()
])
```


```python
model.predict(xtest)[0]
```




    array([1.33764420e-02, 8.30580729e-04, 6.65167507e-15, 4.72093915e-10,
           1.31613654e-03, 9.49280536e-03, 9.74984032e-01, 1.71687247e-09,
           9.93118365e-10, 4.32546551e-10])



## 2. 역전파 과정

### 2.1. Dense 역전파

활성화 함수를 제외하면, 순전파까지는 이전에 다뤘던 선형 모형과 큰 차이가 없었습니다. 역전파 역시 크게 다르지는 않습니다. 뒤쪽 레이어에서 전달되는 미분값을 받아서 파라미터를 업데이트하고 앞쪽 레이어로 넘겨주는 과정을 만들어주면 됩니다. `Dense` 레이어부터 다시 시작해보겠습니다. `Dense` 레이어는 단순한 선형 출력을 내보내므로, 미분 역시 간단합니다. $y = xw + b$ 이므로 가중치에 대한 미분은 $x$, 편향에 대한 미분은 1, $x$에 대한 미분은 $w$입니다. 여기에 역전파로 전해진 값인 `grad_output`을 곱해주면 끝입니다.


```python
class Dense:
    ...생략...
    
    def backward(self, grad_output, learning_rate):
        grad_input = np.matmul(grad_output, self.weight.T)# 이전 레이어로 넘겨줄 미분
        
        grad_weight = np.matmul(self.input.T, grad_output) # weight로 미분
        grad_bias = grad_output.sum(axis=0) # bias로 미분
        
        self.weight -= learning_rate * grad_weight # weight 업데이트
        self.bias -= learning_rate * grad_bias # bias 업데이트
        
        return grad_input # 역전파
```

### 2.2. Activation 역전파

다음으로 `Activation` 클래스의 역전파를 구현해봅시다. 순전파와 마찬가지로 두 개의 메소드를 만들어주었습니다. `backward` 메소드는 이전 레이어로부터 입력을 받은 후 `gradient` 메소드를 호출합니다. 자식 클래스에서는 `gradient` 메소드를 오버라이딩하여 활성화 함수에 맞는 미분을 계산할 것입니다. 이 결과를 이전 레이어로 넘겨줍니다. 소프트맥스, 시그모이드, 렐루 모두 미분 폼이 매우 간단합니다. 활성층에는 별다른 파라미터가 존재하지 않으므로 업데이트 없이 즉시 넘겨주면 됩니다.


```python
# 활성층 클래스
class Activation:
    ...생략...
    
    # 미분 계산
    def gradient(self, grad_output):
        pass
    
    def backward(self, grad_output, learning_rate):
        grad_input = grad_output * self.gradient(grad_output)
        return grad_input

## Activation 클래스 상속

# 소프트맥스
class Softmax(Activation):
    ...생략...
    def gradient(self, grad_output):
        return self.output * (1 - self.output)

# 시그모이드
class Sigmoid(Activation):
    ...생략...
    def gradient(self, grad_output):
        return self.output * (1 - self.output)

# 렐루
class ReLU(Activation):
    ...생략...
    def gradient(self, grad_output):
        grad_input = (self.input > 0)
        return grad_input 
```

### 2.3. 오차 미분

이제 역전파 과정에 최초로 전달될 오차의 미분을 구해줍니다. 손실함수로 크로스 엔트로피를 사용하므로 이에 대한 미분을 구하면 아래와 같습니다.


```python
def grad_cross_entropy(ytrue, yhat, eps=1e-10):
    yhat = np.clip(yhat, eps, 1-eps)
    return (-(ytrue / yhat) + (1 - ytrue)/(1 - yhat)) / ytrue.shape[0]
```

### 2.4. 모델 수정

마지막으로 모델 클래스에 `train` 메소드를 추가해주겠습니다. 훈련은 미니배치 방식으로 진행하였습니다(귀찮아서 풀 배치로 끝내려고 했는데 수렴이 너무 느리네요...ㅎㅎ). 우선 데이터 갯수만큼 인덱스를 만들고 랜덤하게 섞어줍니다. 이후 이 결과를 배치 사이즈만큼 잘라내서 배치 데이터를 추출합니다. 이렇게 뽑힌 미니배치 데이터를 사용해서 순전파/역전파를 진행하면서 각 배치마다 파라미터를 업데이트합니다. 이를 주어진 에포크만큼 반복합니다.


```python
class Model:
    ...생략...
    def train(self, x, y, learning_rate=0.01, epochs = 200, batch_size = 512):
        idx = np.arange(len(x)) # 데이터 사이즈만큼 인덱스 생성
        n_batches = int(len(x) / batch_size) # 배치 개수 계산
        # 주어진 에포크만큼 반복
        for epoch in range(epochs):    
            np.random.shuffle(idx) # 인덱스 셔플
            for i in range(n_batches):
                start, end = batch_size * i, batch_size * (i+1) # 배치 사이즈만큼 간격 생성 : e.x. 0~512, 512~1024, ...
                batch_idx = idx[start:end] # 간격으로 배치 인덱스 추출
                input, ytrue = x[batch_idx], y[batch_idx] # 배치 인덱스로 배치 데이터 추출

                # 순전파
                for layer in self.layers:
                    input = layer.forward(input) 
                yhat = self.layers[-1].output
                
                # 역전파: 파라미터 업데이트
                grad_output = grad_cross_entropy(ytrue, yhat)
                for layer in np.flip(self.layers):
                    grad_output = layer.backward(grad_output, learning_rate)
            
            # 훈련 진행을 파악하기 위해 에포크마다 오차 계산
            yhat = self.predict(x)
            train_loss = cross_entropy(y, yhat)
            print(f"[{epoch}] train_loss: {train_loss}")
```

### 2.5. 모델 학습 및 테스트

이제 지금까지 만든 모델을 점검해보겠습니다. 필요한 함수와 클래스들을 `ANN.py`에 정리하고 검증을 실행합니다. 학습률 0.1, 20바퀴 훈련으로 테스트 데이터에서 약 0.967의 정확도를 얻었습니다.

```python
from ANN import *
import numpy as np
import pandas as pd
from keras.datasets import mnist

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

xtrain, xtest = xtrain.reshape([xtrain.shape[0], -1])/255, xtest.reshape([xtest.shape[0], -1])/255
ytrain, ytest = pd.get_dummies(ytrain).values, pd.get_dummies(ytest).values

np.random.seed(0)

model = Model([
    Dense(xtrain.shape[1], 100), ReLU(),
    Dense(100, 200), ReLU(),
    Dense(200, 10), Softmax()
])

model.train(xtrain, ytrain, epochs = 20, learning_rate = 0.1)

ypred = model.predict(xtest)
ce = cross_entropy(ytest, ypred)
accuracy = (ytest.argmax(axis=1) == ypred.argmax(axis=1)).sum() / len(ytest)
print(f"Cross Entropy: {ce}")
print(f"Accuracy: {accuracy}")

```

```bash
(tensor) jhgan@jhgan-ThinkPad-E595:~$ python backpropagation.py
Using TensorFlow backend.
[0] train_loss: 0.41900695764324003
[1] train_loss: 0.32841351730431445
[2] train_loss: 0.28469834647736436
[3] train_loss: 0.2472436575904893
[4] train_loss: 0.2238040783638741
[5] train_loss: 0.2064766890869934
[6] train_loss: 0.18788879900726665
[7] train_loss: 0.18239761839564864
[8] train_loss: 0.1632899771929538
[9] train_loss: 0.1534925609936448
[10] train_loss: 0.14647234566527104
[11] train_loss: 0.14002885367460152
[12] train_loss: 0.1305398540221366
[13] train_loss: 0.12422088380877823
[14] train_loss: 0.11884933890719969
[15] train_loss: 0.1151885636595044
[16] train_loss: 0.10738619489117324
[17] train_loss: 0.1037703651472939
[18] train_loss: 0.09902387780023811
[19] train_loss: 0.0971087026489792
Cross Entropy: 0.11808091754439508
Accuracy: 0.9645
```

### 2.6. 코드 정리

```python
# ANN.py

import numpy as np

def cross_entropy(y, yhat, eps=1e-10):
    yhat = np.clip(yhat, eps, 1-eps)
    return -(y * np.log(yhat)).sum(axis=1).mean()

def grad_cross_entropy(ytrue, yhat, eps=1e-10):
    yhat = np.clip(yhat, eps, 1-eps)
    return (-(ytrue / yhat) + (1 - ytrue)/(1 - yhat)) / ytrue.shape[0]

class Dense:
    # input_size: n_feature
    def __init__(self, input_size, output_size):
        self.weight = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros(output_size)
    
    def forward(self, input):
        self.input = input
        output = np.matmul(input, self.weight) + self.bias
        return output
    
    def backward(self, grad_output, learning_rate):
        grad_input = np.matmul(grad_output, self.weight.T)# 이전 레이어로 넘겨줄 미분
        
        grad_weight = np.matmul(self.input.T, grad_output) # weight로 미분
        grad_bias = grad_output.sum(axis=0) # bias로 미분
        
        self.weight -= learning_rate * grad_weight # weight 업데이트
        self.bias -= learning_rate * grad_bias # bias 업데이트
        
        return grad_input # 역전파

class Activation:
    def __init__(self):
        self.output = None
        
    def activate(self, input):
        pass
        
    def forward(self, input):
        output = self.activate(input)
        self.output = output
        return output
    
    # 미분 계산
    def gradient(self, grad_output):
        pass
    
    def backward(self, grad_output, learning_rate):
        grad_input = grad_output * self.gradient(grad_output)
        return grad_input

## Activation 클래스 상속

# 소프트맥스
class Softmax(Activation):
    
    def activate(self, input):
        self.input = input
        return np.exp(input) / np.exp(input).sum(axis=1, keepdims=True) # 소프트맥스 함수
    
    def gradient(self, grad_output):
        return self.output * (1 - self.output)

# 시그모이드
class Sigmoid(Activation):
    
    def activate(self, input):
        self.input = input
        return 1 / (1 + np.exp(-input)) # 시그모이드 함수
    
    def gradient(self, grad_output):
        return self.output * (1 - self.output)

# 렐루
class ReLU(Activation):
    
    def activate(self, input):
        self.input = input
        return np.maximum(0, input)
    
    def gradient(self, grad_output):
        grad_input = (self.input > 0)
        return grad_input 

class Model:
    def __init__(self, layers):
        self.layers = layers
                
    def predict(self, x):
        input = x
        for layer in self.layers:
            input = layer.forward(input)
        return self.layers[-1].output
    
    def train(self, x, y, learning_rate=0.01, epochs = 200, batch_size = 512):
        idx = np.arange(len(x)) # 데이터 사이즈만큼 인덱스 생성
        n_batches = int(len(x) / batch_size) # 배치 개수 계산
        # 주어진 에포크만큼 반복
        for epoch in range(epochs):    
            np.random.shuffle(idx) # 인덱스 셔플
            for i in range(n_batches):
                start, end = batch_size * i, batch_size * (i+1) # 배치 사이즈만큼 간격 생성 : e.x. 0~512, 512~1024, ...
                batch_idx = idx[start:end] # 간격으로 배치 인덱스 추출
                input, ytrue = x[batch_idx], y[batch_idx] # 배치 인덱스로 배치 데이터 추출

                # 순전파
                for layer in self.layers:
                    input = layer.forward(input) 
                yhat = self.layers[-1].output
                
                # 역전파: 파라미터 업데이트
                grad_output = grad_cross_entropy(ytrue, yhat)
                for layer in np.flip(self.layers):
                    grad_output = layer.backward(grad_output, learning_rate)
            
            # 훈련 진행을 파악하기 위해 에포크마다 오차 계산
            yhat = self.predict(x)
            train_loss = cross_entropy(y, yhat)
            print(f"[{epoch}] train_loss: {train_loss}")
```

```python
# backpropagation.py

from ANN import *
import numpy as np
import pandas as pd
from keras.datasets import mnist

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()

xtrain, xtest = xtrain.reshape([xtrain.shape[0], -1])/255, xtest.reshape([xtest.shape[0], -1])/255
ytrain, ytest = pd.get_dummies(ytrain).values, pd.get_dummies(ytest).values

np.random.seed(0)

model = Model([
    Dense(xtrain.shape[1], 100), ReLU(),
    Dense(100, 200), ReLU(),
    Dense(200, 10), Softmax()
])

model.train(xtrain, ytrain, epochs = 20, learning_rate = 0.1)

ypred = model.predict(xtest)
ce = cross_entropy(ytest, ypred)
accuracy = (ytest.argmax(axis=1) == ypred.argmax(axis=1)).sum() / len(ytest)
print(f"Cross Entropy: {ce}")
print(f"Accuracy: {accuracy}")

```

## 참고자료

- Rohit Agrawal
, [Building an Artificial Neural Network using pure Numpy](https://towardsdatascience.com/building-an-artificial-neural-network-using-pure-numpy-3fe21acc5815)
