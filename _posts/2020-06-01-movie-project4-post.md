---
title: 영화 추천 시스템 - 프로그래밍(콘텐츠 기반 필터링)  
date : 2020-06-01 14:02:30 -0400
categories : Kaggle update Project
---

### 콘텐츠 기반 필터링(content based filtering)

#### 1. 다음을 기반으로 두개의 컨텐츠 기반으로 구축
* 동영상 개요 및 태그 라인
* 영화 출연진, 제작진, 키워드 및 장르
* 모든 영화의 정보보다 일부를 가져와서 사용



#### 2. 각 영화별의 장르가 여러개로 구성되어져있다, 이러한 것을 세분화 하여 새로운 Dataframe에 저장하여 출력
```python
s = md.apply(lambda x: pd.Series(x['genres']), axis = 1).stack().reset_index(level = 1, drop = True)
s.name = 'genre'
gen_md = md.drop('genres',axis = 1).join(s)
gen_md.head()
```
<img src="https://user-images.githubusercontent.com/60723495/83378294-478d5500-a413-11ea-9230-2ae5cc135d3c.png" width="1000" height="550">



#### 3. 장르를 키워드로 받아서 가중치등급으로 내림차순 정렬한 Top250을 Dataframe을 생성하여 리턴 해주는 함수
```python
def build_chart(genre, percenttile = 0.85) : 
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages= df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percenttile)
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count'] + m) * x['vote_average']) + (m / (m + x['vote_count']) * C), axis = 1)
    qualified = qualified.sort_values('wr', ascending = False).head(250)
    return qualified
```

#### 4. 'Romance'라는 장르에서 Top 15를 출력
```python
build_chart('Romance').head(15)
```
<img src="https://user-images.githubusercontent.com/60723495/83378576-12353700-a414-11ea-834f-35cd4d980895.png" width="1000" height="450">



#### 5. 콘텐츠 기반 추천 시작 부분 - link_small.csv 파일을 links_small이란 Dataframed에 저장하고 TMDB ID의 missing Data 처리와 형변환한 값을 Dataframe에 다시 저장
```python
links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
```



#### 6. Movie_data의 데이터가 너무 커서 축소하여 새로운 Dataframe 생성(Id 형변환하여 links_samll Id와 비교하여 일치하는 곳에 'tmdbId' column을 생성하고 값을 추가)
```pyrhon
md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')
smd = md[md['id'].isin(links_small)]
smd.shape
smd.head()
>>> (9099, 25)
``` 
<img src="https://user-images.githubusercontent.com/60723495/83381216-0731d500-a41b-11ea-8798-392c7dd644cb.png" width="1000" height="450">



#### 7. Dataframe 'smd'안의 column인 'tagline'의 missing Data에 ''의 값을 넣고 'overview'의 column과 합쳐서 column 'description'에 저장하고 'description'의 missing data에 ''의 값으로 채우기
```python
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')
```



#### 8. Dataframe 'smd'을 Tfid벡터화시켜서 matrix(행렬화)하고 tfidf_matrix에 저장하고 사이즈 출력
```python
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description'])
tfidf_matrix.shape
>>> (9099, 268124)
```



#### 9. 변수 cosine_sim에 tfidf_matrix와 tfidf_matrix사이의 선형 커널을 계산하여 저장(첫번째 행의 값 출력)
```python
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[0]
>>> array([1.        , 0.00680476, 0.        , ..., 0.        , 0.00344913,
       0.        ])
```



#### 10. Dataframe 'smd'에 인덱스를 부여하고 'titles'에 'smd'의 column인 'title'을 저장하고 'indices'에 Series를 생성('smd'의 인덱스를 기본으로 한 'smd'의 column인 'title'을 값으로 하여)
```python
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])
```



#### 11. title를 입력 받아서 title의 index를 찾고 index를 기반하여 cosine유사성을 list화하여 sim_scores에 저장하고 오름차순으로 정렬하고 인덱스 1번부터 30번까지 슬라이싱하여 저장하고 'movie_indices'에 index를 차례대로 리스트화하여 titles에서 index를 'movie_indices'값을 기초로 하여 찾아서 리턴시켜주는 함수
```python
def get_recommendations(title) :
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]
```



#### 12. 영화 제목이 'The Godfather'와 'The Dark Knight'와 유사한 영화 10개씩 출력
```python
get_recommendations('The Godfather').head(10)
```
<img src="https://user-images.githubusercontent.com/60723495/83383288-c7212100-a41f-11ea-89db-69b939776809.png" width="400" height="300">

```python
get_recommendations('The Dark Knight').head(10)
```
<img src="https://user-images.githubusercontent.com/60723495/83383426-10717080-a420-11ea-9fb7-ffe75a45ae23.png" width="600" height="300">
