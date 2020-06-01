---
title: 영화 추천 시스템 - 프로그래밍(Data processing)  
date : 2020-06-01 14:02:30 -0400
categories : Kaggle update Project
---

### 영화 추천 프로그래밍(Data Processing)

#### 1. 데이터 정보
* movies_metadata.csv : 주요 영화 메타 데이터 파일. Full MovieLens 데이터 세트에 소개 된 45,000 개의 영화에 대한 정보가 포함되어 있습니다. 포스터, 배경, 예산, 수익, 출시일, 언어, 생산 국가 및 회사 등의 기능이 있습니다.
* keyword.csv : MovieLens 영화에 대한 영화 플롯 키워드가 들어 있습니다. 문자열 화 된 JSON 객체의 형태로 제공됩니다.
* credits.csv : 모든 영화에 대한 출연진 및 승무원 정보로 구성됩니다. 문자열 화 된 JSON 객체의 형태로 제공됩니다.
* links.csv : Full MovieLens 데이터 세트에 포함 된 모든 영화의 TMDB 및 TMDB ID가 포함 된 파일입니다.
* links_small.csv : 전체 데이터 세트의 9,000 개 영화로 구성된 작은 하위 집합의 TMDB 및 TMDB ID를 포함합니다.
* ratings_small.csv : 9,000 개의 영화에서 700 명의 사용자가 평가 한 100,000 개의 하위 집합.



#### 2. 데이터 불러오기
```python
md = pd.read_csv('movies_metadata.csv')
md.head()
```
<img src="https://user-images.githubusercontent.com/60723495/83372988-f6289a00-a401-11ea-9eb5-b6a9e7df6b32.png" width="1000" height="450">



#### 3. genres 데이터 변환
```python
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
md.head()
```
<img src="https://user-images.githubusercontent.com/60723495/83373200-7bac4a00-a402-11ea-8b1d-c254ec9c2197.png" width="1000" height="450">



#### 4. vote_counts와 vote_averages에 missing Data 처리 후에 cloumn인 vote_count와 vote_average 각각의 열을 저장(C는 vote_averages의 평균)
```python
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
print(C)
>>> 5.244896612406511
```



#### 5. vote_counts의 0.95분위에 있는 수 구하기
```python
m = vote_counts.quantile(0.95)
print(m)
>>> 434.0
```



#### 6. release_date의 값에서 년도만을 추출해서 year이라는 column을 추가하여 Dataframe에 나타낸다.
```python
md['year'] = pd.to_datetime(md['release_date'], errors = 'coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
print(md['year'])
```  
<img src="https://user-images.githubusercontent.com/60723495/83373742-5e787b00-a404-11ea-8e87-6df29162e944.png" width="300" height="350">



#### 7. vote_count에서 0.95 분위의 수보다 크고 missing Data가 아니면서 데이터 형변환한 Dataframe 출력
```python
qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title','year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified['vote_count'] = qualified['vote_count'].astype('int')
qualified['vote_average'] = qualified['vote_average'].astype('int')
print(qualified.shape)
qualified.head()
>>> (2274, 6)
```
<img src="https://user-images.githubusercontent.com/60723495/83374033-566d0b00-a405-11ea-8457-dc5f0ac0d6f9.png" width="1000" height="200">



#### 8. 가중 등급(WR) = (v/(v+m)*R)+(m/(v+m)*C)을 구하는 함수를 만들고 Dataframe의 column으로 추가하고 출력
```python
def weight_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m + v) * C)
qualified['wr'] = qualified.apply(weight_rating, axis = 1)
qualified.head()
```
<img src="https://user-images.githubusercontent.com/60723495/83374270-15292b00-a406-11ea-8778-d6495f8b2c3f.png" width="1000" height="200">



#### 9. 가중등급을 기준으로 내림차순 정렬을 하고 상위 250개를 저장하고 상위의 15개를 출력(Top Movie Dataframe 생성)
```python
qualified = qualified.sort_values('wr', ascending = False).head(250)
qualified.head(15)
```
<img src="https://user-images.githubusercontent.com/60723495/83374423-94b6fa00-a406-11ea-9911-b8c8b351b2e3.png" width="1000" height="500">
