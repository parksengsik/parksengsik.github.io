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
* 각각의 변수에 값이 존재하지 않는 개수를 세어서 변수 안의 전체 개수를 나누어서 100을 곱해서 missing ratio 출력
```python
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' : all_data_na})
missing_data.head(20)
```
<img src="https://user-images.githubusercontent.com/60723495/83344886-d0d75580-a347-11ea-9bfb-dd4d2f71ddb8.png" width="300" height="1000">

* missing ratio를 토대로 막대 그래프화 시켜 출력
```python
f, ax = plt.subplots(figsize = (10, 10))
plt.xticks(rotation = '90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize = 15)
plt.ylabel('Percent of missing values', fontsize = 15)
plt.title('Percent missing data by feature', fontsize = 15)
```
<img src="https://user-images.githubusercontent.com/60723495/83344918-34fa1980-a348-11ea-8e19-6ca22749e3e0.png" width="500" height="500">
