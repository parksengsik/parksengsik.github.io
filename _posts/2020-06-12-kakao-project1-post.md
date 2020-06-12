---
title: Kakao project(1) - 노래 추천 시스템  
date : 2020-06-02 11:55:30 -0400
categories : Kakao update Project
---

### Content Based Recommender



#### 1. 이번 Post에서는 Kakao Melon 공모전 준비를 하기위해서 이전의 영화 추천시스템를 모티브로 하여 작성하였다.



#### 2. 위의 라이브러리 중에 Mecab을 Google Colab에서 사용하려면 아래 순서대로 실행하기
```python
! git clone https://github.com/SOMJANG/Mecab-ko-for-Google-Colab.git 
```

```python
ls
>>> drive/  Mecab-ko-for-Google-Colab/  sample_data/
```

```python
%mkdir Mecab-ko-for-Google-Colab
cd Mecab-ko-for-Google-Colab
>>> /content/Mecab-ko-for-Google-Colab
```

```python
ls
>>> images/  install_mecab-ko_on_colab190912.sh  LICENSE  README.md
```
  + 경로를 Mecab-ko-for-Google-Colab으로 설정되어 있는 지 확인하고 아래를 실행

```python
! bash install_mecab-ko_on_colab190912.sh
```
  + 2분에서 3분가량이 소요된다는 것을 참고



#### 3. 위의 라이브러리 중에 konlpy를 설치하기
 ```python
pip install konlpy
```



#### 4. 필요로 한 라이브러리를 불러오기
```python
%matplotlib inline
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from konlpy.tag import Twitter
from konlpy.tag import Mecab
```



#### 5. 데이터 불러오기
 ```python
song_meta = pd.read_json('/content/drive/My Drive/Colab Notebooks/kakao/song_meta.json',typ = 'frame')
train = pd.read_json('/content/drive/My Drive/Colab Notebooks/kakao/train.json',typ = 'frame')
```



#### 6. 대용량 데이터에서 샘플을 만들기
 ```python
train = train.sort_values('like_cnt',ascending=False).reset_index()
train_top200 = train.loc[0:200] 
train_test = train_top200
train_test
```

<img src="https://user-images.githubusercontent.com/60723495/84466558-9610de00-acb4-11ea-919f-41bd453f9ecf.png" width="950" height="350">



#### 7.  list로 된 곡들을 곡 하나씩 나누어주기
```python
plylst_song_map = train_test[['id','songs']]
plylst_song_map_unset = np.dstack(
    (
        np.repeat(plylst_song_map.id.values, list(map(len, plylst_song_map.songs))), np.concatenate(plylst_song_map.songs.values)
    )
)
plylst_song_map = pd.DataFrame(data=plylst_song_map_unset[0],columns=plylst_song_map.columns)
plylst_song_map['id']=plylst_song_map['id'].astype(str)
plylst_song_map['songs']=plylst_song_map['songs'].astype(str)
del plylst_song_map_unset
```



#### 8. 곡마다 몇개의 플레이리스트에 수록되어있는지 확인하기
```python
plylst_song_cnt = pd.DataFrame(plylst_song_map.groupby('songs').id.nunique()).reset_index()
plylst_song_cnt.rename(columns={'songs' : 'id', 'id' : 'count'}, inplace=True)
plylst_song_cnt
```

<img src="https://user-images.githubusercontent.com/60723495/84466859-5dbdcf80-acb5-11ea-92ec-6e875b9e8ac9.png" width="200" height="400">



#### 9. song_meta 데이터에서 필요로 하는 컬럼과 곡마다의 플레이리스트 수록 횟수를 합친 Dataframe 만들기
```python
song_meta_content = song_meta[['id','song_name','artist_name_basket']]
song_meta_content['id'] = song_meta_content['id'].astype(int)
plylst_song_cnt['id'] = plylst_song_cnt['id'].astype(int)
song_meta_content_cnt = song_meta_content.merge(plylst_song_cnt,on='id')
```



#### 10. 샘플 데이터에서 수록되어 있는 곡 하나마다 컬럼을 늘리고 컬럼에서 'tags'와 'plylst_title'를 'description'로 합치고 정규표현식으로 한글과 영어를 뺀 나머지를 없애기 
```python
plylst_name = train_test[['plylst_title','songs','like_cnt','tags','id']]
s = plylst_name.apply(lambda x: pd.Series(x['songs']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'song'
plylst_name_meta = plylst_name.drop('songs',axis=1).join(s)
plylst_name_meta['song'] = plylst_name_meta['song'].astype(int)
s = plylst_name_meta.set_index('song')
s['tags'] = s['tags'].astype(str)
s['description'] = s['plylst_title'] + s['tags']
s = s.drop('tags',axis=1)
key = re.compile('[ㄱ-ㅎ|가-힣|a-z|A-Z|0-9|]+')
s['description'] = s['description'].apply(lambda x : key.findall(x))
s.sort_values('like_cnt',ascending=False)
```

<img src="https://user-images.githubusercontent.com/60723495/84467391-f9037480-acb6-11ea-847b-2af22c67cbb0.png" width="950" height="350">



#### 11. 데이터 전처리를 끝낸 Dataframe 만들기
```python
s = s.reset_index()
s['song'] = s['song'].astype(int)
song_num = s[['song','description','id','plylst_title']]
song_num=song_num.rename(columns={'song' : 'song_id', 'id' : 'plylst_id'})
song_meta_content_cnt['id'] = song_meta_content_cnt['id'].astype(int)
song_num['song_id'] = song_num['song_id'].astype(int)
song_meta_content_all = pd.DataFrame.merge(song_meta_content_cnt,song_num, how='left', left_on='id', right_on='song_id')
song_meta_content_all.head(50)
```

<img src="https://user-images.githubusercontent.com/60723495/84467634-b3937700-acb7-11ea-96c8-1b83ee125abc.png" width="950" height="250">



#### 12. Mecab을 사용하여 컬럼인 'description'을 명사별로 나누기
```python
mecab = Mecab()
test = song_meta_content_all
test['description'] = test['description'].astype(str)
test['description'] = test['description'].apply(lambda x : mecab.nouns(x))
```



#### 13. tfidfVectorizer을 사용하여 매트릭스화를 진행하기
```python
test['description'] = test['description'].astype(str)
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.8)
tfidf_matrix = tf.fit_transform(test['description'])
tfidf_matrix.shape
>>> (24949, 1493)
```



#### 14. cosine유사도 구하기
```python
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[0]
>>> array([1.        , 0.11054928, 0.        , ..., 0.        , 0.        ,
       0.        ])
```



#### 15. 노래 제목으로 이루어진 Dataframe 만들기(인덱스로 하여금 매치되는 곡을 찾기위함)
```python
titles = test['song_name']
indices = pd.Series(test.index, index=test['song_name'])
```



#### 16. 노래를 추천해주는 함수 만들기
```python
def get_recommendations(song_name) :
    idx = list(indices[song_name])
    print('매치된 곡 수 : {}\n\n'.format(len(idx)))
    # print(idx)
    for i in range(0,len(idx)):
        plylst = test.loc[idx[i]]['plylst_id']
        plylst_title = test.loc[idx[i]]['plylst_title']
        print('{}의 인덱스 {}번\n'.format(song_name,idx[i]))
        print('{}의 plylst ID는 {}({})\n'.format(song_name,plylst,plylst_title))
        sim_scores = list(enumerate(cosine_sim[idx[i]]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:31]
        song_indices = [i[0] for i in sim_scores]
        df = train_test[train_test['id']==plylst]
        songs = str(df['songs'].values)
        key = re.compile('[0-9|]+')
        songs = list(key.findall(songs))
        print(test.iloc[song_indices][['song_name','song_id']].head(10))
        print('\n')
        print(songs)
        print('\n\n') 
```



#### 17. '사랑했지만' 곡과 유사한 노래 추천 받기
```python
get_recommendations('사랑했지만')
```

<img src="https://user-images.githubusercontent.com/60723495/84468238-484aa480-acb9-11ea-9af9-57587b44deb5.png" width="950" height="500">

Colab 공유 : <https://colab.research.google.com/drive/1OfeVzPXOMcPGrwh3m5Iw8fKmuDmmkLuX?usp=sharing>
