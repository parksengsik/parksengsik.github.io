---
title: 주택 가격 예측(2) - Data Processing  
date : 2020-05-29 19:28:30 -0400
categories : Kaggle update Project
---

* 대상 변수인 SalePrice를 제외한 Dataframe을 구성
```python
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1 , inplace=True)
print('all_data size is : {}'.format(all_data.shape))
```
