---
title: 주택 가격 예측(1) - Data Read/ EDA  
date : 2020-05-29 19:28:30 -0400
categories : Kaggle update Project
---

### Project : 주택 가격 예측 / 고급 회귀 기법 [데이터 다운링크](https://github.com/parksengsik/parksengsik.github.io/blob/master/data/house_project_data.zip)

* 데이터 처리를 위한 패키지와 모듈 불러오기
```python
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats
from scipy.stats import norm, skew
from subprocess import check_output
color = sns.color_palette()
sns.set_style('darkgrid')
def igonore_warn(*args, **kwargs):
    pass
warnings.warn = igonore_warn
pd.set_option('display.float_format', lambda x : '{:.3f}'.format(x))
print(check_output(['ls','/Users/mac/Desktop/house_project/data']).decode("utf8"))
```

<br>

* 데이터 불러오기
```python
train = pd.read_csv('/Users/mac/Desktop/house_project/data/train.csv')
test = pd.read_csv('/Users/mac/Desktop/house_project/data/test.csv')
```
<br>
* train 파일 데이터에서 상위 5개를 불러오기
```python
train.head(5)
```
<img src="https://user-images.githubusercontent.com/60723495/83249507-a4a1c480-a1e1-11ea-8249-1c6ea5465b49.png" width="1000" height="200">

<br>

* test 파일 데이터에서 상위 5개 불러오기
```python
test.head(5)
```
<img src="https://user-images.githubusercontent.com/60723495/83250803-a8cee180-a1e3-11ea-94dc-7801312516a5.png" width="1000" height="200">

<br>

* 각각의 DataFrame의 size를 출력 /<br>  각각 DataFrame에서 맨앞의 id를 삭제한 후의 size를 출력<br>
```python
print('The train data size before dropping Id feature is : {}' .format(train.shape))
print('The test data size before dropping Id feature is : {}' .format(test.shape))
train_ID = train['Id']
test_ID = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
print('The train data size before dropping Id feature is : {}' .format(train.shape))
print('The test data size before dropping Id feature is : {}' .format(test.shape))
>>>The train data size before dropping Id feature is : (1460, 81)
The test data size before dropping Id feature is : (1459, 80)
The train data size before dropping Id feature is : (1460, 80)
The test data size before dropping Id feature is : (1459, 79)
```

<br>

* GrLivArea 와 SalePrice 간의 그래프를 출력
```python
fig, ax = plt.subplots()
ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
```
<img src="https://user-images.githubusercontent.com/60723495/83344400-3a079a80-a341-11ea-8ce1-1f3456b2f6d4.png" width="300" height="200">

  + SalePrice(주택 가격) 와 GrLivArea(생활면적크기)는 상식적으로 서로에게 영향을 미칠 것으로 예상이 된다.
  + GrLivArea가 커져도 가격이 오르지 않는 특이치가 발생하는 것을 볼 수 있다.
  
<br>

* GrLivArea 와 SalePrice 간의 이상치를 제거하고 그래프를 출력
```python
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
plt.ylabel('SalePrice', fontsize = 13)
plt.xlabel('GrLivArea', fontsize =13)
plt.show()
```
<img src="https://user-images.githubusercontent.com/60723495/83344412-9bc80480-a341-11ea-9943-dc2601582441.png" width="300" height="200">

  + 이러한 특이치는 SalePrice를 예측하는 데 안좋은 영향을 끼칠 것으로 사료되어 제거하 였다.
  
<br>

* 대상 변수인 SalePrice를 관한 빈도수와 오차 분포를 나타낸 그래프를 출력
```python 
sns.distplot(train['SalePrice'], fit = norm)
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n' .format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)], loc = 'best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
```
<img src="https://user-images.githubusercontent.com/60723495/83344432-d631a180-a341-11ea-92e0-ec294bc44e59.png" width="300" height="200">
<img src="https://user-images.githubusercontent.com/60723495/83344459-27da2c00-a342-11ea-8d3a-bc5695a89d77.png" width="300" height="200">

  + 대상 변수인 SalePrice에 대한 빈도수와 오차 분포를 그래프로 이루고 있다.
  + 기준이 되는 정규성을 가지는 그래프와 비교하여 분석한다.
  + 빈도수는 한쪽으로 치우치는 증상을 보이고 오차의 분포는 많은 차이를 보이고 있다.
  + 정규성과 비교하기 쉽게 대상 변수에 대한 정규화를 필요로 해 보인다.
  
<br>

* 대상 변수인 SalePrice를 관한 빈도수와 오차 분포를 로그변환하여 나타낸 그래프를 출력
```python
train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'], fit = norm)
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n' .format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma$ {:.2f})'.format(mu, sigma)], loc = 'best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
```
<img src="https://user-images.githubusercontent.com/60723495/83344479-72f43f00-a342-11ea-8534-ceeedbe0262f.png" width="300" height="200">
<img src="https://user-images.githubusercontent.com/60723495/83344490-ab941880-a342-11ea-85d8-3e003f11ebdd.png" width="300" height="200">

  + 로그변환 : 정규성을 높이고 분석(회귀 분석 등)에서 정확한 값을 얻기 위해서 사용한다, 즉 정규 분포가 아닌 것을 정규 분포에 가깝게 만드는 변환이다. 
