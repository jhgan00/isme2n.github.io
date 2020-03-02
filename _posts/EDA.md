# 데이콘 천체 유형 분류

## Description

feature|description
---|---
psfMag|먼 천체를 한 점으로 가정하여 측정한 빛의 밝기(Point Spread Function Magnitudes)
fiberMag|광섬유를 통과하는 빛의 밝기
petroMag|천체의 위치와 거리에 상관없이 빛의 밝기를 비교하기 위한 수치(Petrosian Magnitudes)
modelMag|천체 중심으로부터 특정 거리의 밝기(Model Magnitudes)

## [SDSS Filters: UGRIZ Filter System](http://skyserver.sdss.org/dr1/en/proj/advanced/color/sdssfilters.asp)

filter|Wavelength
---|---
Ultraviolet|3543
Green|4770
Red|6231
Near Infrared|7625
Infrared|9134


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import RobustScaler
plt.style.use('ggplot')
```


```python
train, test, submission = pd.read_csv("../data/train.csv"), pd.read_csv("../data/test.csv"), pd.read_csv("../data/sample_submission.csv")
```


```python
print(f"data size: {train.shape} \n nulls: {train.isnull().sum().sum()}")
train.head()
```

    data size: (199991, 23) 
     nulls: 0





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
      <th>id</th>
      <th>type</th>
      <th>fiberID</th>
      <th>psfMag_u</th>
      <th>psfMag_g</th>
      <th>psfMag_r</th>
      <th>psfMag_i</th>
      <th>psfMag_z</th>
      <th>fiberMag_u</th>
      <th>fiberMag_g</th>
      <th>...</th>
      <th>petroMag_u</th>
      <th>petroMag_g</th>
      <th>petroMag_r</th>
      <th>petroMag_i</th>
      <th>petroMag_z</th>
      <th>modelMag_u</th>
      <th>modelMag_g</th>
      <th>modelMag_r</th>
      <th>modelMag_i</th>
      <th>modelMag_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0</td>
      <td>QSO</td>
      <td>601</td>
      <td>23.198224</td>
      <td>21.431953</td>
      <td>21.314148</td>
      <td>21.176553</td>
      <td>21.171444</td>
      <td>22.581309</td>
      <td>21.644453</td>
      <td>...</td>
      <td>22.504317</td>
      <td>21.431636</td>
      <td>21.478312</td>
      <td>21.145409</td>
      <td>20.422446</td>
      <td>22.749241</td>
      <td>21.465534</td>
      <td>21.364187</td>
      <td>21.020605</td>
      <td>21.147340</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>QSO</td>
      <td>788</td>
      <td>21.431355</td>
      <td>20.708104</td>
      <td>20.678850</td>
      <td>20.703420</td>
      <td>20.473229</td>
      <td>21.868797</td>
      <td>21.029773</td>
      <td>...</td>
      <td>21.360701</td>
      <td>20.778968</td>
      <td>20.889705</td>
      <td>20.639812</td>
      <td>20.646660</td>
      <td>21.492955</td>
      <td>20.758527</td>
      <td>20.753925</td>
      <td>20.693389</td>
      <td>20.512314</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>QSO</td>
      <td>427</td>
      <td>17.851451</td>
      <td>16.727898</td>
      <td>16.679677</td>
      <td>16.694640</td>
      <td>16.641788</td>
      <td>18.171890</td>
      <td>17.033098</td>
      <td>...</td>
      <td>17.867253</td>
      <td>16.738784</td>
      <td>16.688874</td>
      <td>16.744210</td>
      <td>16.808006</td>
      <td>17.818063</td>
      <td>16.697434</td>
      <td>16.641249</td>
      <td>16.660177</td>
      <td>16.688928</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3</td>
      <td>QSO</td>
      <td>864</td>
      <td>20.789900</td>
      <td>20.040371</td>
      <td>19.926909</td>
      <td>19.843840</td>
      <td>19.463270</td>
      <td>21.039030</td>
      <td>20.317165</td>
      <td>...</td>
      <td>20.433907</td>
      <td>19.993727</td>
      <td>19.985531</td>
      <td>19.750917</td>
      <td>19.455117</td>
      <td>20.770711</td>
      <td>20.001699</td>
      <td>19.889798</td>
      <td>19.758113</td>
      <td>19.552855</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4</td>
      <td>STAR_RED_DWARF</td>
      <td>612</td>
      <td>26.454969</td>
      <td>23.058767</td>
      <td>21.471406</td>
      <td>19.504961</td>
      <td>18.389096</td>
      <td>25.700632</td>
      <td>23.629122</td>
      <td>...</td>
      <td>25.859229</td>
      <td>22.426929</td>
      <td>21.673551</td>
      <td>19.610012</td>
      <td>18.376141</td>
      <td>24.877052</td>
      <td>23.147993</td>
      <td>21.475342</td>
      <td>19.487330</td>
      <td>18.375655</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



트레인 데이터 약 20만 건, 결측 없음

## 1. 타겟 데이터 분포

타겟 유형이 19개나 존재하고, 각 유형의 비율 역시 차이가 큼. 오버샘플링을 고려해야할까?


```python
plt.figure(figsize=(15,5)); plt.xticks(rotation = 45)
print(sns.barplot(data=train.type.value_counts().reset_index(), x="index", y="type"))
```

    AxesSubplot(0.125,0.125;0.775x0.755)



![](/assets/img/docs/output_8_1.png)



```python
pd.Series(train.type.unique()).sort_values()
```




    6                  GALAXY
    0                     QSO
    8              REDDEN_STD
    9                 ROSAT_D
    2        SERENDIPITY_BLUE
    5     SERENDIPITY_DISTANT
    13      SERENDIPITY_FIRST
    17     SERENDIPITY_MANUAL
    11        SERENDIPITY_RED
    16                    SKY
    7        SPECTROPHOTO_STD
    3                STAR_BHB
    14       STAR_BROWN_DWARF
    12            STAR_CARBON
    4           STAR_CATY_VAR
    18                STAR_PN
    1          STAR_RED_DWARF
    15         STAR_SUB_DWARF
    10       STAR_WHITE_DWARF
    dtype: object



뭔진 모르겠지만 대분류들이 존재하는 듯. `STAR` - `SPECTROPHOTO` - `SKY` - `SERENDIPITY` - `ROSAT` - `REDDEN` - `QSO` - `GALAXY`

결국 Y에 해당하는 정보라서 모델링에 활용할 수 있을지는 모르겠다.

## 2. X 변수들의 기초통계량

- 트레인 데이터랑 테스트 데이터랑 통계량 차이가 꽤 많이 남
- 트레인 데이터에 좀 극단적인 케이스들 혹은 오류가 있는 것 같음
- 분포에서 봉우리가 갈라지는 변수들도 보임.


```python
train.drop(columns=["id","fiberID"]).describe().drop("count").round(3)
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
      <th>psfMag_u</th>
      <th>psfMag_g</th>
      <th>psfMag_r</th>
      <th>psfMag_i</th>
      <th>psfMag_z</th>
      <th>fiberMag_u</th>
      <th>fiberMag_g</th>
      <th>fiberMag_r</th>
      <th>fiberMag_i</th>
      <th>fiberMag_z</th>
      <th>petroMag_u</th>
      <th>petroMag_g</th>
      <th>petroMag_r</th>
      <th>petroMag_i</th>
      <th>petroMag_z</th>
      <th>modelMag_u</th>
      <th>modelMag_g</th>
      <th>modelMag_r</th>
      <th>modelMag_i</th>
      <th>modelMag_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>mean</td>
      <td>-6.750</td>
      <td>18.675</td>
      <td>18.401</td>
      <td>18.043</td>
      <td>17.664</td>
      <td>10.850</td>
      <td>19.073</td>
      <td>19.134</td>
      <td>18.183</td>
      <td>18.001</td>
      <td>21.838</td>
      <td>18.454</td>
      <td>18.482</td>
      <td>17.687</td>
      <td>17.699</td>
      <td>20.111</td>
      <td>18.544</td>
      <td>18.182</td>
      <td>17.692</td>
      <td>17.189</td>
    </tr>
    <tr>
      <td>std</td>
      <td>11876.785</td>
      <td>155.423</td>
      <td>127.128</td>
      <td>116.622</td>
      <td>123.735</td>
      <td>4172.116</td>
      <td>749.256</td>
      <td>90.049</td>
      <td>122.379</td>
      <td>145.862</td>
      <td>789.472</td>
      <td>154.376</td>
      <td>97.240</td>
      <td>145.731</td>
      <td>142.692</td>
      <td>122.299</td>
      <td>161.728</td>
      <td>133.984</td>
      <td>131.183</td>
      <td>133.685</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-5310801.771</td>
      <td>-40022.466</td>
      <td>-27184.796</td>
      <td>-26566.311</td>
      <td>-24878.828</td>
      <td>-1864766.371</td>
      <td>-215882.917</td>
      <td>-21802.656</td>
      <td>-20208.516</td>
      <td>-26505.602</td>
      <td>-24463.432</td>
      <td>-25958.752</td>
      <td>-23948.589</td>
      <td>-40438.184</td>
      <td>-30070.729</td>
      <td>-26236.579</td>
      <td>-36902.402</td>
      <td>-36439.638</td>
      <td>-38969.417</td>
      <td>-26050.710</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>19.653</td>
      <td>18.701</td>
      <td>18.049</td>
      <td>17.748</td>
      <td>17.426</td>
      <td>19.940</td>
      <td>18.903</td>
      <td>18.259</td>
      <td>17.904</td>
      <td>17.606</td>
      <td>19.248</td>
      <td>18.114</td>
      <td>17.480</td>
      <td>17.050</td>
      <td>16.805</td>
      <td>19.266</td>
      <td>18.076</td>
      <td>17.423</td>
      <td>16.978</td>
      <td>16.706</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>20.871</td>
      <td>19.904</td>
      <td>19.454</td>
      <td>19.044</td>
      <td>18.612</td>
      <td>21.049</td>
      <td>20.069</td>
      <td>19.631</td>
      <td>19.189</td>
      <td>18.711</td>
      <td>20.367</td>
      <td>19.587</td>
      <td>19.183</td>
      <td>18.693</td>
      <td>18.175</td>
      <td>20.407</td>
      <td>19.548</td>
      <td>19.143</td>
      <td>18.642</td>
      <td>18.101</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>22.160</td>
      <td>21.150</td>
      <td>20.516</td>
      <td>20.074</td>
      <td>19.884</td>
      <td>22.338</td>
      <td>21.386</td>
      <td>20.774</td>
      <td>20.331</td>
      <td>20.133</td>
      <td>21.797</td>
      <td>21.004</td>
      <td>20.457</td>
      <td>20.019</td>
      <td>19.808</td>
      <td>21.993</td>
      <td>20.962</td>
      <td>20.408</td>
      <td>19.969</td>
      <td>19.820</td>
    </tr>
    <tr>
      <td>max</td>
      <td>18773.919</td>
      <td>3538.985</td>
      <td>3048.111</td>
      <td>4835.219</td>
      <td>9823.740</td>
      <td>4870.154</td>
      <td>248077.513</td>
      <td>12084.735</td>
      <td>8059.639</td>
      <td>18358.922</td>
      <td>298771.019</td>
      <td>12139.816</td>
      <td>7003.137</td>
      <td>9772.191</td>
      <td>17403.789</td>
      <td>14488.252</td>
      <td>10582.059</td>
      <td>12237.952</td>
      <td>4062.499</td>
      <td>7420.534</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.drop(columns=["id","fiberID"]).describe().drop("count").round(3)
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
      <th>psfMag_u</th>
      <th>psfMag_g</th>
      <th>psfMag_r</th>
      <th>psfMag_i</th>
      <th>psfMag_z</th>
      <th>fiberMag_u</th>
      <th>fiberMag_g</th>
      <th>fiberMag_r</th>
      <th>fiberMag_i</th>
      <th>fiberMag_z</th>
      <th>petroMag_u</th>
      <th>petroMag_g</th>
      <th>petroMag_r</th>
      <th>petroMag_i</th>
      <th>petroMag_z</th>
      <th>modelMag_u</th>
      <th>modelMag_g</th>
      <th>modelMag_r</th>
      <th>modelMag_i</th>
      <th>modelMag_z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>mean</td>
      <td>20.987</td>
      <td>19.878</td>
      <td>19.280</td>
      <td>18.873</td>
      <td>18.618</td>
      <td>21.185</td>
      <td>20.091</td>
      <td>19.498</td>
      <td>19.083</td>
      <td>18.827</td>
      <td>20.715</td>
      <td>19.462</td>
      <td>18.995</td>
      <td>18.617</td>
      <td>18.412</td>
      <td>20.739</td>
      <td>19.535</td>
      <td>18.935</td>
      <td>18.522</td>
      <td>18.281</td>
    </tr>
    <tr>
      <td>std</td>
      <td>2.112</td>
      <td>2.574</td>
      <td>1.709</td>
      <td>1.721</td>
      <td>1.702</td>
      <td>1.991</td>
      <td>1.865</td>
      <td>1.710</td>
      <td>1.634</td>
      <td>1.712</td>
      <td>2.807</td>
      <td>13.971</td>
      <td>1.979</td>
      <td>1.970</td>
      <td>2.373</td>
      <td>2.187</td>
      <td>1.958</td>
      <td>1.857</td>
      <td>1.797</td>
      <td>1.868</td>
    </tr>
    <tr>
      <td>min</td>
      <td>-7.248</td>
      <td>-42.664</td>
      <td>9.135</td>
      <td>-22.522</td>
      <td>13.350</td>
      <td>9.390</td>
      <td>8.189</td>
      <td>12.288</td>
      <td>12.689</td>
      <td>-8.456</td>
      <td>-98.182</td>
      <td>-1348.069</td>
      <td>-23.909</td>
      <td>-8.357</td>
      <td>-64.917</td>
      <td>12.420</td>
      <td>13.618</td>
      <td>13.383</td>
      <td>12.955</td>
      <td>12.396</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>19.656</td>
      <td>18.671</td>
      <td>18.038</td>
      <td>17.742</td>
      <td>17.425</td>
      <td>19.940</td>
      <td>18.892</td>
      <td>18.254</td>
      <td>17.905</td>
      <td>17.611</td>
      <td>19.249</td>
      <td>18.104</td>
      <td>17.475</td>
      <td>17.044</td>
      <td>16.806</td>
      <td>19.268</td>
      <td>18.065</td>
      <td>17.424</td>
      <td>16.972</td>
      <td>16.716</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>20.854</td>
      <td>19.910</td>
      <td>19.445</td>
      <td>19.033</td>
      <td>18.595</td>
      <td>21.041</td>
      <td>20.072</td>
      <td>19.628</td>
      <td>19.181</td>
      <td>18.700</td>
      <td>20.371</td>
      <td>19.583</td>
      <td>19.197</td>
      <td>18.684</td>
      <td>18.172</td>
      <td>20.413</td>
      <td>19.541</td>
      <td>19.156</td>
      <td>18.635</td>
      <td>18.096</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>22.161</td>
      <td>21.150</td>
      <td>20.490</td>
      <td>20.084</td>
      <td>19.879</td>
      <td>22.339</td>
      <td>21.403</td>
      <td>20.756</td>
      <td>20.332</td>
      <td>20.120</td>
      <td>21.804</td>
      <td>21.026</td>
      <td>20.428</td>
      <td>20.016</td>
      <td>19.816</td>
      <td>21.993</td>
      <td>20.981</td>
      <td>20.389</td>
      <td>19.971</td>
      <td>19.824</td>
    </tr>
    <tr>
      <td>max</td>
      <td>37.681</td>
      <td>182.654</td>
      <td>31.884</td>
      <td>47.227</td>
      <td>34.946</td>
      <td>41.170</td>
      <td>47.161</td>
      <td>29.267</td>
      <td>31.147</td>
      <td>26.479</td>
      <td>65.392</td>
      <td>106.963</td>
      <td>41.851</td>
      <td>52.222</td>
      <td>74.747</td>
      <td>32.641</td>
      <td>28.815</td>
      <td>27.580</td>
      <td>26.472</td>
      <td>24.462</td>
    </tr>
  </tbody>
</table>
</div>



### 트레인/테스트 데이터 변수들의 분포 비교


```python
for colname in train.columns[3:]:
    plt.figure(figsize = (15, 5))
    plt.subplot(1,2,1); plt.title(f"TRAIN: {colname}"); sns.distplot(train[colname].values)
    plt.subplot(1,2,2); plt.title(f"TEST:{colname}"); sns.distplot(test[colname].values)
```


![](/assets/img/docs/output_16_0.png)



![](/assets/img/docs/output_16_1.png)



![](/assets/img/docs/output_16_2.png)



![](/assets/img/docs/output_16_3.png)



![](/assets/img/docs/output_16_4.png)



![](/assets/img/docs/output_16_5.png)



![](/assets/img/docs/output_16_6.png)



![](/assets/img/docs/output_16_7.png)



![](/assets/img/docs/output_16_8.png)



![](/assets/img/docs/output_16_9.png)



![](/assets/img/docs/output_16_10.png)



![](/assets/img/docs/output_16_11.png)



![](/assets/img/docs/output_16_12.png)



![](/assets/img/docs/output_16_13.png)



![](/assets/img/docs/output_16_14.png)



![](/assets/img/docs/output_16_15.png)



![](/assets/img/docs/output_16_16.png)



![](/assets/img/docs/output_16_17.png)



![](/assets/img/docs/output_16_18.png)



![](/assets/img/docs/output_16_19.png)


### 임의로 트레인 데이터 잘라내보기


```python
def clip(colname):
    start, end = train[colname].quantile(0.00005), train[colname].quantile(1- 0.00005)
    bools = (train[colname] >= start) & (train[colname] <= end).rename(colname)
    return bools
```


```python
result = pd.concat(list(map(clip, train.columns[3:].values)), axis=1).apply(all, axis=1)
```


```python
for colname in train.columns[3:]:
    plt.figure(figsize = (15, 5))
    plt.subplot(1,2,1); plt.title(f"TRAIN: {colname}"); sns.distplot(train[result][colname].values)
    plt.subplot(1,2,2); plt.title(f"TEST:{colname}"); sns.distplot(test[colname].values)
```


![](/assets/img/docs/output_20_0.png)



![](/assets/img/docs/output_20_1.png)



![](/assets/img/docs/output_20_2.png)



![](/assets/img/docs/output_20_3.png)



![](/assets/img/docs/output_20_4.png)



![](/assets/img/docs/output_20_5.png)



![](/assets/img/docs/output_20_6.png)



![](/assets/img/docs/output_20_7.png)



![](/assets/img/docs/output_20_8.png)



![](/assets/img/docs/output_20_9.png)



![](/assets/img/docs/output_20_10.png)



![](/assets/img/docs/output_20_11.png)



![](/assets/img/docs/output_20_12.png)



![](/assets/img/docs/output_20_13.png)



![](/assets/img/docs/output_20_14.png)



![](/assets/img/docs/output_20_15.png)



![](/assets/img/docs/output_20_16.png)



![](/assets/img/docs/output_20_17.png)



![](/assets/img/docs/output_20_18.png)



![](/assets/img/docs/output_20_19.png)


## 3. 타겟별 평균/중위수

아웃라이어 걸러내고 그려본 히트맵


```python
plt.figure(figsize=(20,6))
plt.subplot(1,2,1); sns.heatmap(train[result].drop(columns=["id","fiberID"]).groupby("type").mean())
plt.subplot(1,2,2); sns.heatmap(train[result].drop(columns=["id","fiberID"]).groupby("type").median())
```

![](/assets/img/docs/output_23_1.png)


## 4. 필터별 크기 순서


```python
median = train.drop(columns=["id","fiberID"]).groupby("type").median().unstack().reset_index().rename(
    columns = {"level_0":"feature", 0:"median"}
)
```


```python
plt.figure(figsize = (12,18))
ax = sns.barplot(data=median, x="median", y="type", hue="feature")
```


![](/assets/img/docs/output_26_0.png)



```python
cnt = data.groupby("type").size().sort_values().rename("전체_빈도수").reset_index()
```


```python
def invOrder(metric):
    filter_u, filter_z = metric + "_u", metric + "_z"
    inv = train[train[filter_u] - train[filter_z] < 0].groupby("type").size().sort_values().rename("역순_빈도수").reset_index()
    result = pd.merge(cnt, inv, how='left').fillna(0).astype({"역순_빈도수":"int"})
    return result
```


```python
invOrder("petroMag")
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
      <th>type</th>
      <th>전체_빈도수</th>
      <th>역순_빈도수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>STAR_PN</td>
      <td>13</td>
      <td>3</td>
    </tr>
    <tr>
      <td>1</td>
      <td>SERENDIPITY_MANUAL</td>
      <td>61</td>
      <td>10</td>
    </tr>
    <tr>
      <td>2</td>
      <td>SKY</td>
      <td>97</td>
      <td>36</td>
    </tr>
    <tr>
      <td>3</td>
      <td>STAR_BROWN_DWARF</td>
      <td>495</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>STAR_SUB_DWARF</td>
      <td>1154</td>
      <td>3</td>
    </tr>
    <tr>
      <td>5</td>
      <td>STAR_WHITE_DWARF</td>
      <td>2155</td>
      <td>1400</td>
    </tr>
    <tr>
      <td>6</td>
      <td>SERENDIPITY_RED</td>
      <td>2540</td>
      <td>20</td>
    </tr>
    <tr>
      <td>7</td>
      <td>STAR_CARBON</td>
      <td>3252</td>
      <td>6</td>
    </tr>
    <tr>
      <td>8</td>
      <td>SERENDIPITY_DISTANT</td>
      <td>4646</td>
      <td>2855</td>
    </tr>
    <tr>
      <td>9</td>
      <td>ROSAT_D</td>
      <td>6452</td>
      <td>275</td>
    </tr>
    <tr>
      <td>10</td>
      <td>STAR_CATY_VAR</td>
      <td>6504</td>
      <td>40</td>
    </tr>
    <tr>
      <td>11</td>
      <td>SERENDIPITY_FIRST</td>
      <td>7119</td>
      <td>191</td>
    </tr>
    <tr>
      <td>12</td>
      <td>STAR_BHB</td>
      <td>13482</td>
      <td>346</td>
    </tr>
    <tr>
      <td>13</td>
      <td>STAR_RED_DWARF</td>
      <td>13738</td>
      <td>26</td>
    </tr>
    <tr>
      <td>14</td>
      <td>REDDEN_STD</td>
      <td>14613</td>
      <td>2</td>
    </tr>
    <tr>
      <td>15</td>
      <td>SPECTROPHOTO_STD</td>
      <td>14622</td>
      <td>1</td>
    </tr>
    <tr>
      <td>16</td>
      <td>SERENDIPITY_BLUE</td>
      <td>21742</td>
      <td>5048</td>
    </tr>
    <tr>
      <td>17</td>
      <td>GALAXY</td>
      <td>37118</td>
      <td>374</td>
    </tr>
    <tr>
      <td>18</td>
      <td>QSO</td>
      <td>49523</td>
      <td>6528</td>
    </tr>
  </tbody>
</table>
</div>


