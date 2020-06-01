---
title: 영화 추천 시스템 - 프로그래밍(콘텐츠 기반 필터링 MetaData)  
date : 2020-06-01 17:30:30 -0400
categories : Kaggle update Project
---


### 콘텐츠 기반 필터링 - Metadata based Recommender

#### 1. 이번 추천은 Metadata인 credits.csv와 keywords.csv 파일에서 감독과 주연배우, 키워드로 유사성을 판별하여 추천하는 기법을 사용한다.



#### 2. 먼저 credits.csv와 keywords.csv 파일을 불러오기
```python
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')
```



#### 3. 각각의 Dataframe의 column인 'Id'를 형변환 시켜준다.
```python
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')
```



#### 4. 'md'인 Movie_data에 'Id'에 맞게 credits와 keywords의 column을 추가한다(column 수가 늘어난 것을 확인할 수 있을 것이다), 또한 데이터를 줄인 'smd' Dataframe을 사용한다.
```python
md.shape
>>> (45463, 25)
```

```python
md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')
smd = md[md['id'].isin(links_small)]
smd.shape
>>> (9219, 28)
```



#### 5. string안에 있는 dict를 다시 구조화하고 'cast_size'와 'crew_size'를 column에 추가한다.
```python
smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))
smd.shape
>>> (9219, 30)
```



#### 6. 감독이름을 찾아서 리턴 시켜주는 함수(값이 있을 경우에 이름을, 아닌 경우에 nan값을 넣어주게 된다)
```python
def get_director(x) :
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
```



#### 7. Dataframe 'smd'에 감독 데이터를 column 'director'에 추가한다.
```python
smd['director'] = smd['crew'].apply(get_director)
```



#### 8. 배우들의 이름을 넣고 그중에서 영향력있는 배우 3명을 column 'cast'에 다시금 넣어준다.
```python
smd['cast'] = smd['cast'].apply(lambda x : [i['name'] for i in x] if isinstance(x,list) else [])
smd['cast'] = smd['cast'].apply(lambda x : x[:3] if len(x) >= 3 else x)
``` 



#### 9. keyword 또한 정리하여 넣어준다.
```python
smd['keywords'] = smd['keywords'].apply(lambda x : [i['name'] for i in x] if isinstance(x, list) else [])
```



#### 10. 'cast'와 'director' 데이터를 소문자화 시켜주고 띄어쓰기를 없애준다, 감독은 강조를 위해 세번 반복해서 적어준다.
```python
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ","")))
smd['director'] = smd['director'].apply(lambda x : [x,x, x])
```



#### 11. column 'keywords'의 index를 기준으로 각 영화마다의 장르를 동일한 index로 표현한다.
```python
s = smd.apply(lambda x : pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
```



#### 12. 하나밖에 없거나 적은 키워드는 영향력이 적으므로 없앤다.
```python
s = s.value_counts()
s[:5]
```
<img src="https://user-images.githubusercontent.com/60723495/83388227-0c961c00-a429-11ea-84e3-9716b22da572.png" width="300" height="150">



#### 13. x로 받은 값이 keyword에 있는지 확인하고 필러링하는 함수
```python
def filter_keywords(x) :
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words
```



#### 14. keyword 필터링을 하고 SnowballStemmer을 이용해서 단어를 원형으로 바꾸고 띄어쓰기를 없앤다.
```python
stemmer = SnowballStemmer("english")
smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x : [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x : [str.lower(i.replace(" ","")) for i in x])
```



#### 15. 전체적인 데이터를 합쳐서 column 'soup'에 값을 넣어주고 각 'soup'의 값마다 띄어쓰기를 추가한다.
```python
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x : ' '.join(x))
```



#### 16. CountVectorizer를 사용하여 문서 집합에서 단어 토큰을 생성하고 각 단어의 수를 세어 벡터화시키고 'count_matrix'에 변환시켜서 저장한다.
```python
count = CountVectorizer(analyzer='word', ngram_range=(1,2), min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])
```



#### 17. cosine_similarity를 사용하여 count_matrix와 count_matrix사이의 cosine유사성을 측정하여 저장한다.
```python
cosine_sim = cosine_similarity(count_matrix, count_matrix)
```



#### 18. Dataframe ‘smd’에 인덱스를 부여하고 ‘titles’에 ‘smd’의 column인 ‘title’을 저장하고 ‘indices’에 Series를 생성(‘smd’의 인덱스를 기본으로 한 ‘smd’의 column인 ‘title’을 값으로 하여)
```python
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])
```



#### 19. MetaData를 사용한 cosine유사성을 get_recommendation함수를 동작시킨다, 각각의 'The Dark Knight'와 'Mean Girls'의 유사한 영화 10개를 출력한다.
```python
get_recommendations('The Dark Knight').head(10)
```
<img src="https://user-images.githubusercontent.com/60723495/83390089-37ce3a80-a42c-11ea-94a1-05ca8504b81e.png" width="350" height="200">

```python
get_recommendations('Mean Girls').head(10)
```
<img src="https://user-images.githubusercontent.com/60723495/83390160-55030900-a42c-11ea-9fbf-ffeeb060a50d.png" width="400" height="200">




#### 20. 위의 모든 작업을 함수화한다.
```python
def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weight_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified
```



#### 21. 함수를 사용하여 추천시스템을 동작시킨다.
```python
improved_recommendations('The Dark Knight')
```
<img src="https://user-images.githubusercontent.com/60723495/83390613-20dc1800-a42d-11ea-9cb5-5f6d56284a94.png" width="1000" height="300">

```python
improved_recommendations('Mean Girls')
```
<img src="https://user-images.githubusercontent.com/60723495/83390685-49641200-a42d-11ea-8c9c-c0f28463d776.png" width="1000" height="300">
