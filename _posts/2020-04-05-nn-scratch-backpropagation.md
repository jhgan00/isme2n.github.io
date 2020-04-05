---
layout: post
title: "[Python] NeuralNet from Scratch(2)"
categories: [doc]
tags: [python]
comments: true
---

오차 역전파 구현


```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

np.random.seed(0)
```


```python
bc = load_breast_cancer()
x, y = StandardScaler().fit_transform(bc['data']), bc['target'].reshape((-1,1))
```


```python
def cross_entropy(ytrue, yhat, eps=1e-10):
    yhat = np.clip(yhat, eps, 1-eps)
    return -(ytrue * np.log(yhat) + (1-ytrue) * np.log(1- yhat)).mean()
```


```python
class Dense:
    # input_size: n_feature
    def __init__(self, input_size, output_size):
        self.weight = np.random.randn(input_size, output_size)
        self.bias = np.zeros(shape = (output_size,))
    
    def forward(self, input):
        self.input = input
        logit = np.matmul(input, self.weight)
        return logit
    
    def backward(self, grad_output):
        grad_input = np.matmul(grad_output, self.weight.T)
        
        grad_weight = np.matmul(self.input.T, grad_output)
        grad_bias = grad_output.sum(axis=0)
        
        self.weight -= learning_rate * grad_weight
        self.bias -= learning_rate * grad_bias
        
        return grad_input
```


```python
class Sigmoid:
    def __init__(self):
        self.sigmoid = None
    
    def forward(self, input):
        sigmoid = 1 / (1 + np.exp(-input))
        self.sigmoid = sigmoid
        return sigmoid
    
    def backward(self, grad_output):
        grad_input = grad_output * self.sigmoid * (1 - self.sigmoid)
        return grad_input
```


```python
def grad_cross_entropy(ytrue, yhat):
    ytrue, yhat = ytrue.reshape((-1,1)), yhat.reshape((-1, 1))
    return -((ytrue / yhat) + (ytrue - 1) / (1 - yhat)) / ytrue.shape[0]
```


```python
class Model:
    def __init__(self, layers):
        self.layers = layers
    
    def train(self, x, y, learning_rate, epochs = 200, verbose_round = 50):
        for epoch in range(epochs):
            input = x
            for layer in self.layers:
                input = layer.forward(input)

            yhat = self.layers[-1].sigmoid
            train_loss = cross_entropy(y, yhat)
            
            grad_output = grad_cross_entropy(y, yhat)

            for layer in np.flip(self.layers):
                grad_output = layer.backward(grad_output)
                
            if epoch % verbose_round == 0:
                print(f"[{epoch}] train_loss: {train_loss}")
                
    def predict(self, x):
        input = x
        for layer in self.layers:
            input = layer.forward(input)
        return self.layers[-1].sigmoid
```


```python
D1, A1 = Dense(x.shape[1], 128), Sigmoid()
D2, A2 = Dense(128, 1), Sigmoid()
```


```python
model = Model([D1, A1, D2, A2])
learning_rate = 0.01
```


```python
model.train(x, y, 0.2, epochs = 1000)
```

    [0] train_loss: 1.5326826056975693
    [50] train_loss: 0.4672773442711708
    [100] train_loss: 0.34186423439396735
    [150] train_loss: 0.2892485045666735
    [200] train_loss: 0.25604465728873804
    [250] train_loss: 0.23225616592826867
    [300] train_loss: 0.21419043133226778
    [350] train_loss: 0.1999273140376879
    [400] train_loss: 0.18831968426238488
    [450] train_loss: 0.17863595874414703
    [500] train_loss: 0.17039055035660292
    [550] train_loss: 0.16325095279371749
    [600] train_loss: 0.15698296693908065
    [650] train_loss: 0.151417140485504
    [700] train_loss: 0.1464276911865904
    [750] train_loss: 0.1419189980308388
    [800] train_loss: 0.1378167714454446
    [850] train_loss: 0.13406216552252903
    [900] train_loss: 0.1306077706364633
    [950] train_loss: 0.1274148279873056



```python
ypred =  model.predict(x)
```


```python
cross_entropy(y, ypred)
```




    0.12445125178791992




```python
confusion_matrix(y, np.ceil(ypred - 0.5))
```




    array([[194,  18],
           [ 12, 345]])


