---
title: 영화 추천 시스템 - 프로그래밍(Hybrid Recommend System)  
date : 2020-06-02 11:55:30 -0400
categories : Kaggle update Project
---

### Hybrid Recommend System



#### 1. 이번 Post에서는 콘텐츠와 협업 필터링을 통해 아이디어를 모아 특정 사용자에게 영화 제안을 하는 System을 구축한다.



#### 2. 함수 'convert_int'는 x을 받아서 오류가 발생하지 않으면 int로 형변환을 시켜고 리턴 하고, 오류가 발생하면 nan이라는 값을 리턴하는 함수이다.
```python
def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan
```



#### 3. Datafreme 'id_map'에 links_small.csv 파일의 'movieId'와 'tmdbId'를 불러와서 저장하고 'tmdbID'를 형변환 시키고 이름을 'id' 바꾸어준다. Dataframe 'smd'의 column 'title'와 'id'을 'id'를 기준으로 'title'을 추가해준다.
```python
id_map = pd.read_csv('links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
```



#### 4. 'indices_map'에 Dataframe 'id_map'의 column 'id'를 저장한다.
```python
indices_map = id_map.set_index('id')
```



#### 5. 함수 'hybrid'는 userId와 title을 받아서 title의 index를 'idx'에 저장하고 Dataframe 'id_map'에서 title을 기반으로 id를 'tmdbId'에 저장하고 title을 기반으로 movieId를 'movie_id'에 저장하고 cosine유사성을 측정하고 리스트화 시켜서 오름차순으로 정렬하고 index 1번부터 25번까지 슬라이싱하여 'sim_scores'에 저장한다, sim_scores의 id를 토대로 'movies'에 'title' , 'vote_count', 'vote_average', 'year', 'id'를 저장하고 협업필터링에서 예상 등급 구하는 값을 추가하여 내림차순으로 재배열하여 Top10를 리턴하여 준다. 
```python
def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]  
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)
```



#### 6. userId가 1번이고 영화 'Avatar'와 비슷한 영화 추천
```python
hybrid(1, 'Avatar')
```
<img src="https://user-images.githubusercontent.com/60723495/83473939-b9be7200-a4c5-11ea-925b-e5d7a34bf6ae.png" width="800" height="350">



#### 7. userId가 500번이고 영화 'Avatar'와 비슷한 영화 추천
```python
hybrid(500, 'Avatar')
```
<img src="https://user-images.githubusercontent.com/60723495/83474071-0efa8380-a4c6-11ea-914f-033ef5f7503b.png" width="800" height="350">
