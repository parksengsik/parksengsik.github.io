---
title: 영화 추천 시스템 - 프로그래밍(협업 필터링)  
date : 2020-06-02 11:11:30 -0400
categories : Kaggle update Project
---
### 협업 필터링(Collaborative filtering)



#### 1. 이번 Post에서는 RMSE(Root Mean Square Error)를 최소화하고 훌륭한 추천을 하기 위해 SVD(Single Value Discovery)와 같은 매우 강력한 알고리즘을 사용했던 서프라이즈 라이브러리를 사용한다.



#### 2. 'reader'에 Reader함수를 저장하고 Dataframe 'ratings'에 rationgs_small.csv를 불러와 저장하고 출력한다.
```python
reader = Reader()
ratings = pd.read_csv('ratings_small.csv')
ratings.head()
```
<img src="https://user-images.githubusercontent.com/60723495/83468966-4d3d7600-a4b9-11ea-8dbe-feddf32690bb.png" width="300" height="150">




#### 3. 'data'에 Dataframe 'ratings'의 column인 'userId'와 'movieId', 'rating'을 Reader함수를 적용하여 Dataset으로 저장한다.
```python
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
```



#### 4. 'svd'에 SVD함수를 저장한다.(SVD : 특이값 분해 알고리즘)
```python
svd = SVD()
```



#### 5. cross_validate함수를 활용하여 fold를 5개로 나눈 각각의 RMSE와 MAE를 구하고 평균값과 오차를 보여준다.
```python
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
```
<img src="https://user-images.githubusercontent.com/60723495/83470485-3dc02c00-a4bd-11ea-966d-4f3aeba9c400.png" width="800" height="150">



#### 6. 'trainset'에 fold를 나누지 않은 Dataset 'data'를 저장하고 svd에서 활용할 Dataset을 'trainset'으로 설정한다.
```python
trainset = data.build_full_trainset()
svd.fit(trainset)
```



#### 7. Dataframe 'ratings'의 column 'userId' value가 1인 값을 출력한다.
```python
ratings[ratings['userId'] == 1]
``` 
<img src="https://user-images.githubusercontent.com/60723495/83471552-eec7c600-a4bf-11ea-83e9-0f14f40ea787.png" width="300" height="500">




#### 8. userId가 1를 movieId가 302일때 실제등급이 3등급이라고 설정하고 svd를 작동시켜본다.
```python
svd.predict(1, 302, 3)
>>> Prediction(uid=1, iid=302, r_ui=3, est=2.594896522839313, details={'was_impossible': False})
```



#### 9. 이번 Post에서는 협업필러링에 대해 해보았는데 8번의 결과처럼 ID가 302인 영화의 경우 2.594으로 예상되고, 한 가지 놀라운 특징은 영화가 무엇인지(또는 그 안에 무엇이 들어있는지) 상관하지 않는다는 것이다. 그것은 순전히 할당된 영화 ID에 근거하여 작동하며, 다른 사용자들이 어떻게 영화를 예측했는지에 따라 등급을 예측한다.
