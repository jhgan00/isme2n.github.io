---
layout: post
title: "[Python] 텐서플로 2.0 입문자 튜토리얼"
categories: [doc]
tags: [python, ml]
comments: true
---

개강하고 과제에 치여서 살다가, 간만에 시간이 생겼네요 ㅎㅎ 텐서플로 2.0 공식 사이트의 초보자용 튜토리얼을 가볍게 따라해봅니다. 공식 홈페이지에서도 복잡한 텐서플로 코드가 아닌 케라스 API를 사용하는 튜토리얼을 제공하고 있네요! 

# 1. 기본 이미지 분류

```python
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
print(tf.__version__)
```

    2.1.0


## 1.1. 데이터 로드

유명한 패션 MNIST 데이터를 사용해서 튜토리얼을 진행합니다. 손글씨 대신에 옷 이미지를 담고있는 데이터네요. 텐서플로에 내장된 패션 MNIST 데이터를 로드하면 넘파이 어레이를 반환해줍니다. 훈련 데이터의 사이즈는 (60000, 28, 28), 테스트 데이터의 사이즈는 (10000, 28, 28)이고, 각각의 픽셀값은 0 ~ 255의 정수입니다. 예측하려는 값은 0부터 9 사이의 정수이고, 각각의 숫자는 옷의 종류를 의미합니다. 첫 번째 트레인 데이터를 그려본 결과 신발처럼 보이네요! 정확히는 ankle boot라고 합니다. 이미지의 픽셀 크기를 조절하는 데에 어떤 기준이 있는지는 잘 모르겠습니다. 간단히 검색해본 결과로는 mean을 사용하는 것이 좋을 것 같다는 의견들이 있는 대세인 것 같네요.


```python
fashin_mnist = keras.datasets.fashion_mnist
(train_image, train_labels), (test_image, test_labels) = fashin_mnist.load_data()

print(
    "xtrain_shape: ", train_image.shape, "\n",
    "xtest_shape: ", test_image.shape, "\n",
    "ytrain_size: ", train_labels.shape, "\n",
    "ytrain_size: ", test_labels.shape, "\n",
)
```

    xtrain_shape:  (60000, 28, 28) 
     xtest_shape:  (10000, 28, 28) 
     ytrain_size:  (60000,) 
     ytrain_size:  (10000,) 
    



```python
plt.imshow(train_image[0])
plt.colorbar()
plt.grid(False)
```


![](/assets/img/docs/output_6_0.png)


```python
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

## 1.2. 데이터 전처리

모델에 데이터를 주입하기 전에 간단한 전처리를 해줍니다. 각 픽셀값을 255로 나눠서 데이터를 0~1 사이로 스케일링해주었습니다.


```python
train_image, test_image = train_image / 255.0, test_image / 255.0
```


```python
plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([]); plt.yticks([]); plt.grid(False);
    plt.imshow(train_image[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
```


![](/assets/img/docs/output_10_0.png)


## 1.3. 모델 구성

케라스는 모델을 만들고, 컴파일을 하는 방식으로 돌아갑니다.

### 층 설정

신경망의 기본적인 구성 요소는 층이고, 층은 주입된 데이터에서 특징를 추출합니다. 피쳐를 1차원 배열로 변환해주는 `Flatten`레이어를 가장 앞에 넣어줍니다. 이 레이어에서는 가중치의 업데이트(학습)은 일어나지 않고 단지 주입된 데이터를 변형해주기만 합니다. 다음에는 두 개의 `Dense` 레이어를 쌓아주었는데, 128개의 노드를 가진 레이어 이후 결과 출력을 위해서 소프트맥스 레이어를 쌓아줍니다.


```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])
```

### 모델 컴파일

모델의 구조를 완성하였다면 이제 컴파일을 할 차례입니다. 컴파일 단게에서는 모델을 훈련하기 위해 필요한 몇 가지 설정을 추가해주어야 합니다. 즉 손실 함수, 옵티마이저, 메트릭을 설정해줍니다. 튜토리얼인만큼 메트릭은 간단하게 accuracy 스코어로 설정해준 것 같네요. 


```python
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
```

### 모델 훈련

케라스에서는 `fit` 메소드를 통해서 모델을 훈련합니다.


```python
model.fit(train_image, train_labels, epochs = 5)
```

    Train on 60000 samples
    Epoch 1/5
    60000/60000 [==============================] - 17s 275us/sample - loss: 0.5020 - accuracy: 0.8224
    Epoch 2/5
    60000/60000 [==============================] - 14s 234us/sample - loss: 0.3746 - accuracy: 0.8643
    Epoch 3/5
    60000/60000 [==============================] - 14s 232us/sample - loss: 0.3355 - accuracy: 0.8787
    Epoch 4/5
    60000/60000 [==============================] - 14s 236us/sample - loss: 0.3123 - accuracy: 0.8836
    Epoch 5/5
    60000/60000 [==============================] - 13s 217us/sample - loss: 0.2956 - accuracy: 0.8916





    <tensorflow.python.keras.callbacks.History at 0x7fcec3b1b410>



### 정확도 평가

테스트 셋에서 모델의 성능을 측정해봅니다. 케라스 모델이 `evaluate`라는 메소드를 가지고 있다는 사실을 처음 알았네요. 상당히 간편한 기능인 것 같습니다.


```python
test_loss, test_acc = model.evaluate(test_image, test_labels, verbose = 2)
```

    10000/10000 - 1s - loss: 0.3415 - accuracy: 0.8769


### 예측값 만들기

예측도 `predict` 메소드로 간단하게 수행할 수 있습니다. 결과는 소프트맥스 확률로 반횐되는 것을 알 수 있습니다.


```python
predictions = model.predict(test_image)
```


```python
predictions[0]
```




    array([1.3411199e-06, 1.3930752e-08, 5.5810409e-07, 1.6061969e-07,
           1.7584125e-06, 2.4212729e-03, 2.2270599e-06, 1.0316484e-01,
           1.3998107e-05, 8.9439392e-01], dtype=float32)



# 2. 케라스와 텐서플로 허브를 사용한 영화 리뷰 텍스트 분류

잊을만 하면 만지게 되는 텍스트 데이터입니다 ㅎㅎ IMDB 리뷰 텍스트를 긍정/부정으로 분류하는 간단한 문제입니다. 해당 데이터를 balanced, 즉 긍정과 부정 텍스트의 비중이 같도록 조정되었다고 하네요. 

 텐서플로 허브에 대해서는 이번에 처음 들어봤는데, 전이 학습을 위한 라이브러리/플랫폼이라고 합니다. 전이학습에 대해서 예전에 재미있는 설명을 들어본 적이 있는데, '사과를 깎는 법을 배운 기계에게 배를 깎는 법을 가르치는 기술'정도로 이해할 수 있다고 했던 것 같네요 ㅎㅎ 출력층 이전까지의 가중치를 freeze시키고 출력 레이어에서의 weight를 조정하는 방식이 대표적이라고 들어본 것 같습니다. 허브와 데이터셋 모두 텐서플로와 함께 설치되는 것 같지는 않고, 따로 설치해주어야 하는 것 같습니다.

## 2.1. 데이터 로드

먼저 imdb 데이터셋을 가져오고 적절히 훈련 데이터와 검증 데이터를 분리해줍니다. 튜토리얼의 코드는 제 환경에서 에러가 나서 스플릿 부분은 일부 수정하였습니다. 훈련 데이터는 텐서로 제공되는 것 같네요.


```python
import tensorflow_hub as hub
import tensorflow_datasets as tfds
```


```python
splits = ("train[:60%]", "train[60%:80%]", "train[80%:]")

(train_data, validation_data, test_data) = tfds.load(
    name="imdb_reviews", 
    split=splits,
    as_supervised=True
)
```


```python
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch[0]
```




    <tf.Tensor: shape=(), dtype=string, numpy=b"This was an absolutely terrible movie. Don't be lured in by Christopher Walken or Michael Ironside. Both are great actors, but this must simply be their worst role in history. Even their great acting could not redeem this movie's ridiculous storyline. This movie is an early nineties US propaganda piece. The most pathetic scenes were those when the Columbian rebels were making their cases for revolutions. Maria Conchita Alonso appeared phony, and her pseudo-love affair with Walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning. I am disappointed that there are movies like this, ruining actor's like Christopher Walken's good name. I could barely sit through it.">



## 2.2. 모델 구성

신경망 구성을 위해서는 세 개의 중요한 구조적 결정이 필요하다고 하네요.

- 어떻게 텍스트를 표현할 것인가?
- 모델에서 얼마나 많은 층을 사용할 것인가?
- 각 층에서 얼마나 많은 은닉 유닛을 사용할 것인가?

사실 모든 신경망 모델에서 위와 같은 고민을 해야하죠. 일단 텍스트 데이터를 다루는 중이니 첫번째 문제에 집중해봅니다. 튜토리얼에서 제시하는 전략은 사전 훈련된 텍스트 임베딩 모델을 통해서 문장을 벡터화해주는 것입니다. 

### 레이어 구성

먼저 문장을 임베딩시키기 위해서 텐서플로 허브 모델을 사용하는 케라스 층을 만들어봅니다. 해당 url에 존재하는 pretrained 모델을 가져와서 레이어를 만들어주고, 레이어에 데이터를 넣어주면 간단하게 실수 벡터로 변환되는 것을 볼 수 있습니다. 우리 모델에 적합한 파라미터로 업데이트를 해주기 위해 `trainable=True`로 설정해줍니다.


```python
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable = True)
hub_layer(train_examples_batch[:3])
```




    <tf.Tensor: shape=(3, 20), dtype=float32, numpy=
    array([[ 1.765786  , -3.882232  ,  3.9134233 , -1.5557289 , -3.3362343 ,
            -1.7357955 , -1.9954445 ,  1.2989551 ,  5.081598  , -1.1041286 ,
            -2.0503852 , -0.72675157, -0.65675956,  0.24436149, -3.7208383 ,
             2.0954835 ,  2.2969332 , -2.0689783 , -2.9489717 , -1.1315987 ],
           [ 1.8804485 , -2.5852382 ,  3.4066997 ,  1.0982676 , -4.056685  ,
            -4.891284  , -2.785554  ,  1.3874227 ,  3.8476458 , -0.9256538 ,
            -1.896706  ,  1.2113281 ,  0.11474707,  0.76209456, -4.8791065 ,
             2.906149  ,  4.7087674 , -2.3652055 , -3.5015898 , -1.6390051 ],
           [ 0.71152234, -0.6353217 ,  1.7385626 , -1.1168286 , -0.5451594 ,
            -1.1808156 ,  0.09504455,  1.4653089 ,  0.66059524,  0.79308075,
            -2.2268345 ,  0.07446612, -1.4075904 , -0.70645386, -1.907037  ,
             1.4419787 ,  1.9551861 , -0.42660055, -2.8022065 ,  0.43727064]],
          dtype=float32)>




```python
model = tf.keras.Sequential([
    hub_layer,
    tf.keras.layers.Dense(16, activation = 'relu'),
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 /assets/img/docs/Output Shape              Param #   
    =================================================================
    keras_layer_2 (KerasLayer)   (None, 20)                400020    
    _________________________________________________________________
    dense_4 (Dense)              (None, 16)                336       
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 17        
    =================================================================
    Total params: 400,373
    Trainable params: 400,373
    Non-trainable params: 0
    _________________________________________________________________


### 모델 컴파일

특별할 것은 없습니다.


```python
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
```

### 모델 훈련

이번에는 미니 배치 방식으로 모델을 훈련합니다. 배치 사이즈는 512, 에포크는 20번으로 잡아주었네요.


```python
history = model.fit(
    train_data.shuffle(10000).batch(512),
    epochs = 20,
    validation_data = validation_data.batch(512),
    verbose=1
)
```

    Epoch 1/20
    30/30 [==============================] - 7s 218ms/step - loss: 1.2485 - accuracy: 0.4491 - val_loss: 0.8638 - val_accuracy: 0.4146
    Epoch 2/20
    30/30 [==============================] - 5s 182ms/step - loss: 0.7707 - accuracy: 0.4837 - val_loss: 0.7176 - val_accuracy: 0.5580
    Epoch 3/20
    30/30 [==============================] - 5s 175ms/step - loss: 0.6892 - accuracy: 0.5823 - val_loss: 0.6733 - val_accuracy: 0.6092
    Epoch 4/20
    30/30 [==============================] - 4s 138ms/step - loss: 0.6464 - accuracy: 0.6307 - val_loss: 0.6381 - val_accuracy: 0.6430
    Epoch 5/20
    30/30 [==============================] - 4s 137ms/step - loss: 0.6092 - accuracy: 0.6709 - val_loss: 0.6092 - val_accuracy: 0.6758
    Epoch 6/20
    30/30 [==============================] - 5s 160ms/step - loss: 0.5781 - accuracy: 0.7067 - val_loss: 0.5798 - val_accuracy: 0.7016
    Epoch 7/20
    30/30 [==============================] - 5s 166ms/step - loss: 0.5433 - accuracy: 0.7385 - val_loss: 0.5503 - val_accuracy: 0.7312
    Epoch 8/20
    30/30 [==============================] - 5s 173ms/step - loss: 0.5078 - accuracy: 0.7674 - val_loss: 0.5178 - val_accuracy: 0.7568
    Epoch 9/20
    30/30 [==============================] - 4s 148ms/step - loss: 0.4718 - accuracy: 0.7987 - val_loss: 0.4859 - val_accuracy: 0.7842
    Epoch 10/20
    30/30 [==============================] - 4s 134ms/step - loss: 0.4349 - accuracy: 0.8211 - val_loss: 0.4552 - val_accuracy: 0.8070
    Epoch 11/20
    30/30 [==============================] - 4s 134ms/step - loss: 0.4019 - accuracy: 0.8371 - val_loss: 0.4279 - val_accuracy: 0.8230
    Epoch 12/20
    30/30 [==============================] - 5s 171ms/step - loss: 0.3688 - accuracy: 0.8570 - val_loss: 0.4037 - val_accuracy: 0.8328
    Epoch 13/20
    30/30 [==============================] - 5s 157ms/step - loss: 0.3414 - accuracy: 0.8697 - val_loss: 0.3815 - val_accuracy: 0.8388
    Epoch 14/20
    30/30 [==============================] - 4s 150ms/step - loss: 0.3146 - accuracy: 0.8810 - val_loss: 0.3629 - val_accuracy: 0.8490
    Epoch 15/20
    30/30 [==============================] - 4s 142ms/step - loss: 0.2879 - accuracy: 0.8919 - val_loss: 0.3463 - val_accuracy: 0.8554
    Epoch 16/20
    30/30 [==============================] - 5s 171ms/step - loss: 0.2663 - accuracy: 0.9023 - val_loss: 0.3325 - val_accuracy: 0.8582
    Epoch 17/20
    30/30 [==============================] - 4s 143ms/step - loss: 0.2454 - accuracy: 0.9103 - val_loss: 0.3226 - val_accuracy: 0.8616
    Epoch 18/20
    30/30 [==============================] - 4s 145ms/step - loss: 0.2285 - accuracy: 0.9181 - val_loss: 0.3146 - val_accuracy: 0.8656
    Epoch 19/20
    30/30 [==============================] - 5s 170ms/step - loss: 0.2136 - accuracy: 0.9239 - val_loss: 0.3089 - val_accuracy: 0.8686
    Epoch 20/20
    30/30 [==============================] - 5s 175ms/step - loss: 0.2003 - accuracy: 0.9297 - val_loss: 0.3045 - val_accuracy: 0.8708



```python
result = model.evaluate(test_data.batch(512), verbose = 2)
for name, value in zip(model.metrics_names, result):
    print(f"{name}: {value}")
```

    loss: 0.29697889983654024
    accuracy: 0.8773999810218811


약간의 오버피팅이 있기는 하지만 나쁘지 않은 결과를 얻었습니다. 튜토리얼 설명에 따르면 고급 방법을 사용한 모델은 95%에 가까운 정확도를 보여준다고 하네요!

# 3. 사전 처리된 텍스트로 텍스트 분류

이전 절에서 다룬 imdb 데이터인데 다른 방식으로 전처리를 해주는 것 같네요. 

## 3.1. 데이터 로드

리뷰를 미리 정수 시퀀스로 변해준 데이터라고 합니다. `num_words` 파라미터는 가장 많이 등장하는 순으로 단어를 선택해주는 파라미터라고 합니다. 사실 이렇게 단어만 정수로 인코딩해서 데이터를 전처리하는 경우 각 문서마다 shape이 달라지는 문제가 발생하는데, 어떻게 해결을 해주었을지 궁금하네요!


```python
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)
```


```python
print(f"train_size: {len(train_data)}, label: {len(train_labels)}")
```

    train_size: 25000, label: 25000



```python
len(train_data[0]), len(train_data[1])
```




    (218, 189)



## 3.2. 데이터 추가 전처리

여기에서 정수 배열로 변환된 리뷰 데이터를 텐서로 다시 변환해줍니다. 튜토리얼에서는 원 핫 인코딩 방식과 패딩 방식을 제시하고 패딩 방식을 선택하였습니다(1만 개의 단어를 원 핫 인코딩하는 방식은 희소성 문제, 메모리 문제 등으로 사실은 거의 사용이 불가능한 방식이죠).정수 배열의 길이가 모두 같도록 패딩을 추가해준다는 아이디어는 사실 처음 들어봤습니다. 이런 방식을 사용하면 성능이 어떨지 기대가 되네요! 아무래도 이전 절보다 성능 향상을 기대할 수 있기 때문에 순서를 뒤로 배치하지 않았을까 하는데 차차 살펴보도록 하겠습니다 ㅎㅎ 구체적으로는 패딩 값은 0, 배열의 뒤쪽으로 채워지도록 하였고 최대 길이는 256으로 설정하였습니다.


```python
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value = 0, padding='post', maxlen=256
)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value = 0, padding='post', maxlen=256
)
```


```python
train_data[0]
```




    array([   1,   14,   22,   16,   43,  530,  973, 1622, 1385,   65,  458,
           4468,   66, 3941,    4,  173,   36,  256,    5,   25,  100,   43,
            838,  112,   50,  670,    2,    9,   35,  480,  284,    5,  150,
              4,  172,  112,  167,    2,  336,  385,   39,    4,  172, 4536,
           1111,   17,  546,   38,   13,  447,    4,  192,   50,   16,    6,
            147, 2025,   19,   14,   22,    4, 1920, 4613,  469,    4,   22,
             71,   87,   12,   16,   43,  530,   38,   76,   15,   13, 1247,
              4,   22,   17,  515,   17,   12,   16,  626,   18,    2,    5,
             62,  386,   12,    8,  316,    8,  106,    5,    4, 2223, 5244,
             16,  480,   66, 3785,   33,    4,  130,   12,   16,   38,  619,
              5,   25,  124,   51,   36,  135,   48,   25, 1415,   33,    6,
             22,   12,  215,   28,   77,   52,    5,   14,  407,   16,   82,
              2,    8,    4,  107,  117, 5952,   15,  256,    4,    2,    7,
           3766,    5,  723,   36,   71,   43,  530,  476,   26,  400,  317,
             46,    7,    4,    2, 1029,   13,  104,   88,    4,  381,   15,
            297,   98,   32, 2071,   56,   26,  141,    6,  194, 7486,   18,
              4,  226,   22,   21,  134,  476,   26,  480,    5,  144,   30,
           5535,   18,   51,   36,   28,  224,   92,   25,  104,    4,  226,
             65,   16,   38, 1334,   88,   12,   16,  283,    5,   16, 4472,
            113,  103,   32,   15,   16, 5345,   19,  178,   32,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
              0,    0,    0], dtype=int32)




```python
x_val = train_data[:10000]
partial_x_train = train_data[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
```

## 3.3. 모델 구성


정수 배열을 그대로 사용하는 것이 아니라 역시 임베딩 레이어를 거치도록 모델을 구성하였습니다. 출력되는 텐서의 최종 차원은 (batch, sequence, embedding)이 됩니다. 다음 레이어는 입력의 차원을 통일해주기 위한 average pooling 레이어입니다. 풀링 레이어로 들어가는 텐서는 (batch, sequence, embedding)의 차원을 갖는데, 이 중에서 sequence 차원에 대해서 평균을 계산하여 차원을 통일시켜주는 개념입니다(개인적으로 이러한 방법이 얼마나 효과적일지는 의문이네요). 이어지는 레이어에서는 특별한 것이 없습니다.


```python
vocab_size = 10000
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 16, input_shape=(None,)),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 /assets/img/docs/Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, None, 16)          160000    
    _________________________________________________________________
    global_average_pooling1d_1 ( (None, 16)                0         
    _________________________________________________________________
    dense_7 (Dense)              (None, 16)                272       
    _________________________________________________________________
    dense_8 (Dense)              (None, 1)                 17        
    =================================================================
    Total params: 160,289
    Trainable params: 160,289
    Non-trainable params: 0
    _________________________________________________________________



```python
model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)
```

### 3.4. 모델 훈련


```python
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs = 40,
    batch_size = 512,
    validation_data = (x_val, y_val),
    verbose = 1
)
```

    Train on 15000 samples, validate on 10000 samples
    Epoch 1/40
    15000/15000 [==============================] - 2s 146us/sample - loss: 0.6923 - accuracy: 0.5182 - val_loss: 0.6909 - val_accuracy: 0.5212
    Epoch 2/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.6877 - accuracy: 0.6027 - val_loss: 0.6841 - val_accuracy: 0.7214
    Epoch 3/40
    15000/15000 [==============================] - 1s 84us/sample - loss: 0.6766 - accuracy: 0.7398 - val_loss: 0.6692 - val_accuracy: 0.7448
    Epoch 4/40
    15000/15000 [==============================] - 1s 73us/sample - loss: 0.6546 - accuracy: 0.7692 - val_loss: 0.6430 - val_accuracy: 0.7662
    Epoch 5/40
    15000/15000 [==============================] - 1s 80us/sample - loss: 0.6200 - accuracy: 0.7891 - val_loss: 0.6058 - val_accuracy: 0.7775
    Epoch 6/40
    15000/15000 [==============================] - 1s 81us/sample - loss: 0.5746 - accuracy: 0.8073 - val_loss: 0.5618 - val_accuracy: 0.8020
    Epoch 7/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.5230 - accuracy: 0.8260 - val_loss: 0.5129 - val_accuracy: 0.8189
    Epoch 8/40
    15000/15000 [==============================] - 1s 87us/sample - loss: 0.4709 - accuracy: 0.8467 - val_loss: 0.4679 - val_accuracy: 0.8338
    Epoch 9/40
    15000/15000 [==============================] - 1s 85us/sample - loss: 0.4239 - accuracy: 0.8613 - val_loss: 0.4300 - val_accuracy: 0.8447
    Epoch 10/40
    15000/15000 [==============================] - 1s 66us/sample - loss: 0.3840 - accuracy: 0.8727 - val_loss: 0.3985 - val_accuracy: 0.8528
    Epoch 11/40
    15000/15000 [==============================] - 1s 74us/sample - loss: 0.3502 - accuracy: 0.8823 - val_loss: 0.3744 - val_accuracy: 0.8592
    Epoch 12/40
    15000/15000 [==============================] - 1s 66us/sample - loss: 0.3230 - accuracy: 0.8897 - val_loss: 0.3547 - val_accuracy: 0.8658
    Epoch 13/40
    15000/15000 [==============================] - 1s 68us/sample - loss: 0.3001 - accuracy: 0.8978 - val_loss: 0.3400 - val_accuracy: 0.8705
    Epoch 14/40
    15000/15000 [==============================] - 1s 67us/sample - loss: 0.2806 - accuracy: 0.9035 - val_loss: 0.3288 - val_accuracy: 0.8716
    Epoch 15/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.2639 - accuracy: 0.9086 - val_loss: 0.3184 - val_accuracy: 0.8755
    Epoch 16/40
    15000/15000 [==============================] - 1s 68us/sample - loss: 0.2485 - accuracy: 0.9145 - val_loss: 0.3106 - val_accuracy: 0.8789
    Epoch 17/40
    15000/15000 [==============================] - 1s 67us/sample - loss: 0.2352 - accuracy: 0.9191 - val_loss: 0.3042 - val_accuracy: 0.8808
    Epoch 18/40
    15000/15000 [==============================] - 1s 75us/sample - loss: 0.2229 - accuracy: 0.9232 - val_loss: 0.2992 - val_accuracy: 0.8809
    Epoch 19/40
    15000/15000 [==============================] - 1s 66us/sample - loss: 0.2120 - accuracy: 0.9261 - val_loss: 0.2958 - val_accuracy: 0.8812
    Epoch 20/40
    15000/15000 [==============================] - 1s 68us/sample - loss: 0.2022 - accuracy: 0.9314 - val_loss: 0.2925 - val_accuracy: 0.8823
    Epoch 21/40
    15000/15000 [==============================] - 1s 69us/sample - loss: 0.1923 - accuracy: 0.9344 - val_loss: 0.2895 - val_accuracy: 0.8832
    Epoch 22/40
    15000/15000 [==============================] - 1s 76us/sample - loss: 0.1839 - accuracy: 0.9387 - val_loss: 0.2883 - val_accuracy: 0.8837
    Epoch 23/40
    15000/15000 [==============================] - 1s 68us/sample - loss: 0.1754 - accuracy: 0.9429 - val_loss: 0.2872 - val_accuracy: 0.8848
    Epoch 24/40
    15000/15000 [==============================] - 1s 67us/sample - loss: 0.1676 - accuracy: 0.9462 - val_loss: 0.2862 - val_accuracy: 0.8842
    Epoch 25/40
    15000/15000 [==============================] - 1s 70us/sample - loss: 0.1605 - accuracy: 0.9489 - val_loss: 0.2859 - val_accuracy: 0.8836
    Epoch 26/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.1535 - accuracy: 0.9525 - val_loss: 0.2859 - val_accuracy: 0.8847
    Epoch 27/40
    15000/15000 [==============================] - 1s 74us/sample - loss: 0.1468 - accuracy: 0.9552 - val_loss: 0.2871 - val_accuracy: 0.8849
    Epoch 28/40
    15000/15000 [==============================] - 1s 69us/sample - loss: 0.1411 - accuracy: 0.9570 - val_loss: 0.2883 - val_accuracy: 0.8839
    Epoch 29/40
    15000/15000 [==============================] - 1s 70us/sample - loss: 0.1351 - accuracy: 0.9592 - val_loss: 0.2885 - val_accuracy: 0.8860
    Epoch 30/40
    15000/15000 [==============================] - 1s 73us/sample - loss: 0.1293 - accuracy: 0.9625 - val_loss: 0.2901 - val_accuracy: 0.8852
    Epoch 31/40
    15000/15000 [==============================] - 1s 73us/sample - loss: 0.1241 - accuracy: 0.9642 - val_loss: 0.2935 - val_accuracy: 0.8843
    Epoch 32/40
    15000/15000 [==============================] - 1s 79us/sample - loss: 0.1197 - accuracy: 0.9658 - val_loss: 0.2953 - val_accuracy: 0.8846
    Epoch 33/40
    15000/15000 [==============================] - 1s 77us/sample - loss: 0.1145 - accuracy: 0.9679 - val_loss: 0.2970 - val_accuracy: 0.8844
    Epoch 34/40
    15000/15000 [==============================] - 1s 69us/sample - loss: 0.1098 - accuracy: 0.9692 - val_loss: 0.2991 - val_accuracy: 0.8845
    Epoch 35/40
    15000/15000 [==============================] - 1s 77us/sample - loss: 0.1053 - accuracy: 0.9706 - val_loss: 0.3020 - val_accuracy: 0.8850
    Epoch 36/40
    15000/15000 [==============================] - 1s 71us/sample - loss: 0.1014 - accuracy: 0.9718 - val_loss: 0.3050 - val_accuracy: 0.8834
    Epoch 37/40
    15000/15000 [==============================] - 1s 64us/sample - loss: 0.0971 - accuracy: 0.9737 - val_loss: 0.3078 - val_accuracy: 0.8837
    Epoch 38/40
    15000/15000 [==============================] - 1s 80us/sample - loss: 0.0933 - accuracy: 0.9757 - val_loss: 0.3120 - val_accuracy: 0.8821
    Epoch 39/40
    15000/15000 [==============================] - 1s 72us/sample - loss: 0.0895 - accuracy: 0.9769 - val_loss: 0.3147 - val_accuracy: 0.8830
    Epoch 40/40
    15000/15000 [==============================] - 1s 67us/sample - loss: 0.0860 - accuracy: 0.9783 - val_loss: 0.3190 - val_accuracy: 0.8816



```python
model.evaluate(test_data, test_labels, verbose=2)
```

    25000/25000 - 2s - loss: 0.3393 - accuracy: 0.8714





    [0.33929339721679685, 0.8714]



역시 세밀한 조정이 없으니 엄청난 오버피팅을 보이네요 ㅎㅎ 정확도 자체는 앞 절의 모델과 큰 차이 없는 듯 합니다.

# 4. 자동차 연비 예측하기: 회귀

이번에는 회귀 문제네요! 개인적으로 회귀 문제와 분류 문제에서 신경망 모델의 성능 차이가 있는지 항상 궁금했어요. 여기에서는 Auto MPG라는 데이터를 통해서 연비 예측 모델을 만드는 것 같네요. 처음으로 직사각형 테이블의 데이터가 나오니 반갑기도 하네요 ㅎㅎ 사실 딥러닝이 음성이나 이미지 등 거대한 비정형 데이터에서의 성능으로 유명해진 것 같은데, 작은 정형 데이터에서도 잘 작동할까요?


```python
import pathlib
import pandas as pd
from sklearn.preprocessing import StandardScaler
```

## 4.1. 데이터 로드


```python
dataset_path = keras.utils.get_file(
    "auto-mpg.data",
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)
```


```python
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MPG</th>
      <th>Cylinders</th>
      <th>Displacement</th>
      <th>Horsepower</th>
      <th>Weight</th>
      <th>Acceleration</th>
      <th>Model Year</th>
      <th>Origin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>393</th>
      <td>27.0</td>
      <td>4</td>
      <td>140.0</td>
      <td>86.0</td>
      <td>2790.0</td>
      <td>15.6</td>
      <td>82</td>
      <td>1</td>
    </tr>
    <tr>
      <th>394</th>
      <td>44.0</td>
      <td>4</td>
      <td>97.0</td>
      <td>52.0</td>
      <td>2130.0</td>
      <td>24.6</td>
      <td>82</td>
      <td>2</td>
    </tr>
    <tr>
      <th>395</th>
      <td>32.0</td>
      <td>4</td>
      <td>135.0</td>
      <td>84.0</td>
      <td>2295.0</td>
      <td>11.6</td>
      <td>82</td>
      <td>1</td>
    </tr>
    <tr>
      <th>396</th>
      <td>28.0</td>
      <td>4</td>
      <td>120.0</td>
      <td>79.0</td>
      <td>2625.0</td>
      <td>18.6</td>
      <td>82</td>
      <td>1</td>
    </tr>
    <tr>
      <th>397</th>
      <td>31.0</td>
      <td>4</td>
      <td>119.0</td>
      <td>82.0</td>
      <td>2720.0</td>
      <td>19.4</td>
      <td>82</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset.shape
```




    (398, 8)



## 4.2. 데이터 전처리

무조건 NA를 떨궈버리는건 좋지 않은 습관이지만, 일단은 튜토리얼이니 떨구고 진행하는 것 같네요 ㅎㅎ `Origin` 컬럼도 간단하게 원 핫 인코딩으로 처리해줍니다.


```python
dataset.isna().sum()
```




    MPG             0
    Cylinders       0
    Displacement    0
    Horsepower      6
    Weight          0
    Acceleration    0
    Model Year      0
    Origin          0
    dtype: int64




```python
dataset = dataset.dropna()
```


```python
dataset = pd.concat([
    dataset.drop(columns="Origin"),
    pd.get_dummies(dataset.Origin)
], axis=1)
```


```python
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_labels = train_dataset.MPG
test_labels = test_dataset.MPG
```


```python
normed_train_data = StandardScaler().fit_transform(train_dataset)
normed_test_data = StandardScaler().fit_transform(test_dataset)
```

## 4.3. 모델 구성

특별한 것은 없습니다


```python
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    
    return model
```


```python
model = build_model()
model.summary()
```

    Model: "sequential_6"
    _________________________________________________________________
    Layer (type)                 /assets/img/docs/Output Shape              Param #   
    =================================================================
    dense_15 (Dense)             (None, 64)                704       
    _________________________________________________________________
    dense_16 (Dense)             (None, 64)                4160      
    _________________________________________________________________
    dense_17 (Dense)             (None, 1)                 65        
    =================================================================
    Total params: 4,929
    Trainable params: 4,929
    Non-trainable params: 0
    _________________________________________________________________


## 4.4. 모델 훈련

케라스 콜백 클래스를 상속해서 훈련 과정을 점으로 표현해주는 `PrintDot` 클래스를 정의해서 사용하였네요. 신기합니다 ㅎㅎ


```python
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

EPOCHS = 1000

history = model.fit(
    normed_train_data,
    train_labels.values,
    epochs = EPOCHS,
    validation_split=0.2,
    verbose=0,
    callbacks=[PrintDot()]
)
```

    
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................
    ....................................................................................................


```python
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(8,12))

    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

plot_history(history)
```


![](/assets/img/docs/output_68_0.png)


훈련 과정에서 약 100 에포크 이후에는 모델 성능의 향상이 거의 없는 것 같으므로 early stopping을 사용해봅니다.


```python
model = build_model()
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
    normed_train_data, train_labels.values, epochs = EPOCHS, validation_split = 0.2, verbose = 0,
    callbacks = [PrintDot(), early_stop]
)

plot_history(history)
```

    
    ....................................................................................................
    ................................................


![](/assets/img/docs/output_70_1.png)



```python
loss, mae, mse = model.evaluate(normed_test_data, test_labels.values, verbose=2)
print("테스트 세트의 평균 절대 오차: {:5.2f} MPG".format(mae))
```

    78/78 - 0s - loss: 1.7830 - mae: 1.1521 - mse: 1.7830
    테스트 세트의 평균 절대 오차:  1.15 MPG



```python
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
```


![](/assets/img/docs/output_72_0.png)



```python
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
```


![](/assets/img/docs/output_73_0.png)

