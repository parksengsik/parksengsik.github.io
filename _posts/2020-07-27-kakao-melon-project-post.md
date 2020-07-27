---
title: Kakao arena - Melon Playlist Continuation
date : 2020-07-27 19:42:30 -0400
categories : Kakao Melon Project
---
### 1. 개요
* 상세설명 
    + 플레이리스트에 가장 어울리는 곡들을 예측할 수 있을까?
        - 플레이리스트에 있는 곡들과 비슷한 느낌의 곡들을 계속해서 듣고 싶은 적이 있으셨나요?
        - 이번 대회에서는 플레이리스트에 수록된 곡과 태그의 절반 또는 전부가 숨겨져 있을 때, 주어지지 않은 곡들과 태그를 예측하는 것을 목표로 합니다.
        - 만약 플레이리스트에 들어있는 곡들의 절반을 보여주고, 나머지 숨겨진 절반을 예측할 수 있는 모델을 만든다면, 플레이리스트에 들어있는 곡이 전부 주어졌을 때 이 모델이 해당 플레이리스트와 어울리는 곡들을 추천해 줄 것이라고 기대할 수 있을 것입니다.
* 대회목표 
    + 주어진 플레이리스트의 정보를 참조하여 해당 플레이리스트에 원래 수록되어 있었을 곡 리스트와 태그 리스트를 예측하는 것이 이 대회의 목표입니다. 각 플레이리스트별로 원래 플레이리스트에 들어있었을 것이라 예상되는 곡 100개, 태그 10개를 제출해 주세요
* 데이터셋 구성 
    + 플레이리스트 메타데이터
    + 곡 메타데이터
    + 곡에 대한 Mel-spectrogram
* 상금 (총 1408만원)
    + 1등 : 512만원 (1팀)
    + 2등 : 256만원 (2팀)
    + 3등 : 128만원 (3팀)
* 규칙 
    + 기본 규칙
        - 본 대회에 참가하는 모든 참가자는 카카오 아레나의 이용약관을 따라야 합니다.
        - 특히 공정경쟁에 위배되는 행위를 하였을 시에 참가 자격 박탈 등과 같은 제재가 이뤄질 수 있습니다.
    + 상세 규칙
        - 팀은 개인 또는 최대 4명까지 구성 가능하며, 카카오 크루와 일반 참가자가 혼합하여 팀을 이루는 것은 금지됩니다.
        - 부정행위 방지를 위해 최종 제출을 앞둔 마지막 7일 동안은 팀 탈퇴 및 해산이 금지됩니다.
        - 카카오톡 인증을 통한 1인 1계정으로 가입 가능하며 여러 계정으로 결과물을 제출할 수 없습니다.
        - 카카오 아레나 회원가입 시 연락 가능한 연락처를 기재 바랍니다. 대회 참가자와 연락이 닿지 않아 발생한 불이익은 전적으로 참가자 책임입니다.
        - 팀 단위로 하루 30회 제출로 제한됩니다. 만일 비정상적인 경로나 방식으로 30회 초과 제출할 경우 사후 점검을 통해 심사대상에서 불이익을 받을 수 있습니다.



### 2.프로그램 소스코드 설명
#### (1) 라이브러리 불러오기
``` python
import pandas as pd
import numpy as np
import re 
import time 
import datetime
import json

from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```
* pandas 와 numpy이는 데이터를 불러오고 전처리를 하는 과정에서 사용하였습니다.
* re는 데이터 전처리과정에서 정규표현식을 적용하기위해 사용하였습니다.
* datatime 과 time은 프로그램이 동작하는 과정을 시간으로 확인하기 위함으로 사용하였습니다.
* json은 최종 결과물을 json파일형식으로 저장하기 위해 사용하였습니다.
* CountVectorizer는 Cosine_similarity를 돌리기위해 데이터를 count를 이용하여 백터화시키기 위해 사용하였습니다.
* cosine_similarity는 백터화된 데이터를 cosine_similarity(코사인 유사도)를 측정하여 데이터간의 유사도 측정하기 위해 사용하였습니다.



#### (2) 데이터 불러오기
```python
train = pd.read_json('train.json', typ = 'frame')
song_meta = pd.read_json('song_meta.json', typ = 'frame')
val = pd.read_json('val.json', typ='frame')
test = pd.read_json('test.json', typ='frame')
```



#### (3) 데이터 전처리
```python
data = train[['id','songs','tags']]     
```
* train(플레이리스트 데이터)에서 컬럼 'id'와 'songs', 'tags'를 추출하여 Dataframe 'data' 저장합니다.
```python
data['tags'] = data['tags'].apply(lambda x: ' '.join(x))    
```
* Dataframe 'data'의 컬럼 'tags'를 리스트에서 하나의 문장으로 하여금 변환시킵니다.
```python
data_song = data[['tags','songs']]

data_song_unnsest = np.dstack(
    (
        np.repeat(data_song.tags.values, list(map(len, data_song.songs))),
        np.concatenate(data_song.songs.values)
    )
)

data_song = pd.DataFrame(data = data_song_unnsest[0], columns= data_song.columns)
data_song['songs'] = data_song['songs'].astype(str)
data_song['tags'] = data_song['tags'].astype(str)

del data_song_unnsest
```
* 플레이리스트에 수록된 곡들을 하나씩 분리하여 Dataframe 'data_song'에 저장합니다.
```python
data_song_tags = data_song.sort_values(by = ['songs','tags']).groupby('songs').tags.apply(list).reset_index(name = 'tags')
```
* Dataframe 'data_song'에서 곡별로 그룹화를 시켜 곡마다 'tags'를 합쳐주고 Dataframe 'data_song_tags'에 저장합니다.
```python
data_tags = data_song_tags[['songs','tags']]    
```
* Dataframe 'data_song_tags'에서 컬럼 'songs' 와 'tags'를 추출하여 Dataframe 'data_tags'에 저장한다. 
```python
data_tags['tags'] = data_tags['tags'].apply(lambda x: ' '.join(x))  
data_tags['tags'] = data_tags['tags'].apply(lambda x : x.split())   
```
* Dataframe 'data_tags'의 컬럼 'tags'를 리스트에서 하나의 문장으로 하여금 변환시킵니다.
* Dataframe 'data_tags'의 컬럼 'tags'를 문장에서 띄워쓰기 기준으로 리스트화 시킨다. 
```python
song_tag = data_tags[['songs','tags']]

song_tag_unnsest = np.dstack(
    (
        np.repeat(song_tag.songs.values, list(map(len, song_tag.tags))),
        np.concatenate(song_tag.tags.values)
    )
)

song_tag = pd.DataFrame(data = song_tag_unnsest[0], columns = song_tag.columns)
song_tag['songs'] = song_tag['songs'].astype(int)
song_tag['tags'] = song_tag['tags'].astype(str)

del song_tag_unnsest
```
* Dataframe 'song_tags'의 컬럼 'tags'를 tag 하나씩 분리하여 Dataframe 'song_tag'에 저장하였습니다.
```python
song_tag = song_tag.rename(columns = {'songs' : 'song_id','tags':'tag'})    
```
* Dataframe 'song_tag'의 컬럼명을 변환시켜줍니다.
```python
data_tags_song = song_tag.sort_values(by = ['song_id','tag']).groupby('tag').song_id.apply(list).reset_index(name = 'songs')  
```
* Dataframe 'song_tag'에서 tag별로 그룹화를 시켜 곡마다 'song_id'를 합쳐주고 Dataframe 'data_tags_song'에 저장합니다.
```python
data_song_tags['tags'] = data_song_tags['tags'].astype(str)
```
* Dataframe 'data_song_tags'의 컬럼인 'tags' 자료형을 string으로 바꿔줍니다.
```python
data_song_tags['tags'] = data_song_tags['tags'].apply(lambda x : x.lower())
```
* Dataframe 'data_song_tags'의 컬럼인 'tags'의 영어를 소문자로 바꿔줍니다.
```python
key = re.compile('[ㄱ-ㅎ|가-힣|a-z|A-Z|0-9|&|]+')
data_song_tags['tags'] = data_song_tags['tags'].apply(lambda x : key.findall(x))
```
* 한글과 영어, 숫자, 기호 &을 key로 설정합니다.
* Dataframe 'data_song_tags'의 컬럼인 'tags'의 값을 key를 기준으로 하여 정규표현식을 사용하여 바꿔줍니다.
```python
data_song_tags = data_song_tags.rename(columns={'songs':'song_id'})
```
* Dataframe 'data_song_tags'의 컬럼명을 변환합니다.
```python
song_id_list = data_song_tags['song_id'].astype(str).tolist()
```
* Dataframe 'data_song_tags'의 컬럼 'song_id'를 자료형 string으로 바꿔주고 list형태로 'song_id_list'에 저장합니다.
```python
tag_list = data_tags_song['tag'].astype(str).tolist()
```
* Dataframe 'data_song_tags'의 컬럼 'tag'를 자료형 string으로 바꿔주고 list형태로 'tag_list'에 저장합니다.
```python
val_train = test
val_train['songs'] = val_train['songs'].apply(lambda x: list(map(str, x)))
val_train['id'] = val_train['id'].astype(int)
```
* 최종 결과물 제출은 TEST.json 파일로 돌려야 함으로 'val_train'에 Dataframe 'test'를 저장하고 'val_train' 컬럼 'songs'에 list안에 value를 자료형 string으로 바꿔줍니다.
* Dataframe 'val_train'의 컬럼 'id'를 자료형 int로 바꿔줍니다.
```python
data_song_tags['tags'] = data_song_tags['tags'].astype(str)
count = CountVectorizer(ngram_range=(1,2), min_df=0)
count_matrix = count.fit_transform(data_song_tags['tags'])
```
* Dataframe 'data_song_tags'의 컬럼 'tags'를 자료형 string으로 바꿔줍니다.
* Dataframe 'data_song_tags'의 컬럼 'tags'를 counvector화 시켜줍니다.
```python
data_tags_song['songs'] = data_tags_song['songs'].apply(lambda x : list(set(x)))
data_tags_song['songs'] = data_tags_song['songs'].astype(str)
count = CountVectorizer(ngram_range=(1,2), min_df=0)
count_matrix_tag = count.fit_transform(data_tags_song['songs'])
```
* Dataframe 'data_song_tags'의 컬럼 'songs'을 집합화 시키고 다시 list화 시켜서 중복된 song_id를 없애줍니다.
* 그 이유는 위의 노래에서는 song_id별로 tag를 받아서 데이터의 수가 상대적으로 적었지만 tag에서는 상대적으로 데이터가 많아서 단일화 시켜주었습니다. 
* Dataframe 'data_song_tags'의 컬럼 'songs'을 자료형 string으로 바꿔줍니다.
* Dataframe 'data_song_tags'의 컬럼 'songs'을 counvector화 시켜줍니다.



#### (4) 모델링
```python
def sec(times):
    return str(datetime.timedelta(seconds=times)).split(".")[0]
```
* 함수 sec는 구간을 설정하여 소요되는 시간을 알아보고자 만들었습니다.
```python
 def cos_song(song_id) : 
    indices = pd.Series(data_song_tags.index, index=data_song_tags['song_id'])
   #  start = time.time()
    idx = indices[song_id]
    sim_scores = list(cosine_similarity(count_matrix[idx], count_matrix)[0])
    sim_scores = pd.Series(sim_scores)
    sim_scores = sim_scores.sort_values(ascending=False)
    sim_scores = sim_scores[1:101]
    song_indices = sim_scores.index.to_list()
   #  print('cos_song complete / time : {}'.format(sec(time.time()-start)))  
    return data_song_tags.iloc[song_indices][['song_id']]
```
* 함수 cos_song은 변수 'song_id'를 받아와 코사인 유사도를 측정하여 노래 100곡을 반환해주는 함수입니다. 
* Dataframe 'data_song_tags'의 컬럼 'song_id'를 index로 value를 'data_song_tags'의 index로 하여 'indices'에 저장합니다.
* Series 'indices'에서 변수 'song_id'를 넣어 index를 찾아 변수 'idx'에 저장합니다.
* count_matrix에서 index가 idx인 부분과 count_matrix를 cosine_similarity를 사용하여 유사도를 구하고 그 값을 list화 하여 'sim_scores'에 저장합니다.
* list 'sim_scores'을 Series화 하여 내림차순으로 재배열합니다.
* Series 'sim_scores'의 두번째부터 100번째까지 슬라이싱합니다.
* Series 'sim_scores'의 index를 list화 하여 'song_indices'에 저장합니다.
* Dataframe 'data_song_tags'의 index에서 list 'song_indices'의 값과 매치되는 컬럼 'song_id'을 조회하여 반환해줍니다.
```python
def recommend_result_cos(plylst_id):
    # start = time.time()
    songs = val_train[val_train['id'] == plylst_id]['songs'].tolist()[0]
    songs = [i for i in songs if i in song_id_list]
    recommend = pd.DataFrame(columns=['song_id'])
    remite = lambda x : 10 if x > 10 else x
    for x in range(0,remite(len(songs))) : 
        recommend = recommend.append(cos_song(songs[x]))
    recommend_cnt = recommend.reset_index().groupby('song_id').count()
    result_song  = recommend_cnt.sort_values('index',ascending=False).head(200).reset_index()
    result_songs = result_song['song_id'].tolist()
    result = [i for i in result_songs if i not in songs]
    # print('recommend_result_cos complete / time : {}'.format(sec(time.time()-start)))
    return result[0:100]
```
* 함수 recommend_result_cos는 변수 'plylst_id'를 받아와 플레이리스트 있는 곡들중 10곡을 한곡당 100곡씩 추천받아 중복횟수로 순위를 매겨 100곡을 반환해주는 함수입니다.
* data 'val_train'(test 데이터)의 컬럼 'id'에서 변수 'plylst_id'를 조회하여 컬럼 'songs'를 list화 하여 'songs'에 저장합니다.
* list 'son_id_list'에 포함되는 list 'songs'의 value만 남겨줍니다.
* 컬럼 'song_id'를 갖는 Dataframe 'recommend'를 만들어줍니다.
* remite를 걸어서 11곡이상 되면 10을 넣어서 10곡으로 함수 'cos_song'을 동작시키고 값을 Dataframe 'recommend'에 추가해줍니다.
* Dataframe 'recommed'를 index를 reset해주고 컬럼 'song_id' 기준으로 그룹화하여 count하여 'recommend_cnt'에 저장합니다.
* Dataframe 'recommend_cnt'을 컬럼 'index'을 기준으로 내림차순으로 재배열하고 200개를 슬라이싱하여 index를 초기화하여 'result_song'에 저장합니다.
* Dataframe 'result_song'의 컬럼 'song_id'를 list화 하여 'result_songs'에 저장합니다.
* list 'result_songs'에서 songs에 포함된 노래를 제외하고 'result'에 저장 해줍니다.
* list 'result'에서 100개를 슬라이싱하여 반환해줍니다.
```python
def tag(result):
    # start = time.time()
    result = result[0:50]
    recommend = pd.DataFrame(columns=['tag','song_id'])
    for i in range(0,len(result)):
        recommend = recommend.append(song_tag[song_tag['song_id']==int(result[i])][['song_id','tag']])
    recommend = recommend.groupby('tag').count().sort_values('song_id',ascending=False).head(15)
    # print('tag complete / time : {}'.format(sec(time.time()-start)))
    return recommend.index.tolist()
```
* 함수 tag는 변수 'result'에 앞의 recommend_result_cos의 return 값을 넣어서 100곡중의 50곡을 사용하여 tag를 추출하여 count하여 순위를 매겨 15개를 반환해주는 함수이다.
* 변수 'result'를 50개로 슬라이싱하여 줄여줍니다.
* 컬럼이 'tag', 'song_id'로 구성된 Dataframe 'recommend'를 만들어줍니다.
* Dataframe 'song_tag'에서 컬럼 'song_id'와 list 'result'의 value를 받아서 조회한 컬럼 'song_id'와 'tag'를 Dataframe 'recommend'에 추가 해줍니다.
* Dataframe 'recommend'에서 컬럼 'tag'를 기준으로 하여 그룹화하여 카운트하고 'song_id'를 기준으로 내림차순 재배열하고 15개로 슬라이싱합니다.
* Dataframe 'recommend의 index를 list화하여 반환해줍니다.(index가 tag로 설정되어 있습니다.)
```python
 def cos_tag(tag) : 
    indices = pd.Series(data_tags_song.index, index=data_tags_song['tag'])
    # start = time.time()
    idx = indices[tag]
    sim_scores = list(cosine_similarity(count_matrix_tag[idx], count_matrix_tag)[0])
    sim_scores = pd.Series(sim_scores)
    sim_scores = sim_scores.sort_values(ascending=False)
    sim_scores = sim_scores[1:11]
    song_indices = sim_scores.index.to_list()
    # print('cos_tag complete / time : {}'.format(sec(time.time()-start)))  
    return data_tags_song.iloc[song_indices][['tag']]
```
* 함수 cos_tag은 변수 'tag'를 받아와 코사인 유사도를 측정하여 tag 10개를 반환해주는 함수입니다. 
* Dataframe 'data_tags_song'의 컬럼 'tag'를 index로 value를 'data_tags_song'의 index로 하여 'indices'에 저장합니다.
* Series 'indices'에서 변수 'tag'를 넣어 index를 찾아 변수 'idx'에 저장합니다.
* count_matrix에서 index가 idx인 부분과 count_matrix를 cosine_similarity를 사용하여 유사도를 구하고 그 값을 list화 하여 'sim_scores'에 저장합니다.
* list 'sim_scores'을 Series화 하여 내림차순으로 재배열합니다.
* Series 'sim_scores'의 두번째부터 10번째까지 슬라이싱합니다.
* Series 'sim_scores'의 index를 list화 하여 'song_indices'에 저장합니다.
* Dataframe 'data_tags_song'의 index에서 list 'song_indices'의 값과 매치되는 컬럼 'tag'을 조회하여 반환해줍니다.
```python
def recommend_result_cos_tags(plylst_id):
    # start = time.time()
    tags = val_train[val_train['id'] == plylst_id]['tags'].tolist()[0]
    tags = [i for i in tags if i in tag_list]
    recommend = pd.DataFrame(columns=['tag'])
    for x in range(0,len(tags)) : 
        recommend = recommend.append(cos_tag(tags[x]))
    recommend_cnt = recommend.reset_index().groupby('tag').count()
    result_tag  = recommend_cnt.sort_values('index',ascending=False).head(15).reset_index()
    result_tags = result_tag['tag'].tolist()
    result = [i for i in result_tags if i not in tags]
    # print('recommend_result_cos_tags complete / time : {}'.format(sec(time.time()-start))) 
    return result[0:10]
```
* 함수 recommend_result_cos_tags는 변수 'plylst_id'를 받아와 플레이리스트 있는 tag들을 한곡당 10개씩 추천받아 중복횟수로 순위를 매겨 100곡을 반환해주는 함수입니다.
* data 'val_train'(test 데이터)의 컬럼 'id'에서 변수 'plylst_id'를 조회하여 컬럼 'tags'를 list화 하여 'tags'에 저장합니다.
* list 'tag_list'에 포함되는 list 'tags'의 value만 남겨줍니다.
* 컬럼 'tag'를 갖는 Dataframe 'recommend'를 만들어줍니다.
* list 'tags'의 개수만큼 동작시키고 Dataframe 'recommend'에 추가 해줍니다.
* Dataframe 'recommed'를 index를 reset해주고 컬럼 'tag' 기준으로 그룹화하여 count하여 'recommend_cnt'에 저장합니다.
* Dataframe 'recommend_cnt'을 컬럼 'index'을 기준으로 내림차순으로 재배열하고 15개를 슬라이싱하여 index를 초기화하여 'result_tag'에 저장합니다.
* Dataframe 'result_tag'의 컬럼 'tag'를 list화 하여 'result_tags'에 저장합니다.
* list 'result_tags'에서 tags에 포함된 노래를 제외하고 'result'에 저장 해줍니다.
* list 'result'에서 10개를 슬라이싱하여 반환해줍니다.
```python
def song(result):
    # start = time.time()
    recommend = pd.DataFrame(columns=['tag','song_id'])
    for i in range(0,len(result)):
        recommend = recommend.append(song_tag[song_tag['tag']==result[i]][['song_id','tag']])
    recommend = recommend.groupby('song_id').count().sort_values('tag',ascending=False).head(100)
    # print('song complete / time : {}'.format(sec(time.time()-start)))
    return recommend.index.tolist()
```
* 함수 song는 변수 'result'에 앞의 recommend_result_cos_tags의 return 값을 넣어서 10개의 tag를 사용하여 song을 추출하여 count하여 순위를 매겨 100개를 반환해주는 함수이다.
* 컬럼이 'tag', 'song_id'로 구성된 Dataframe 'recommend'를 만들어줍니다.
* Dataframe 'song_tag'에서 컬럼 'song_id'와 list 'result'의 value를 받아서 조회한 컬럼 'song_id'와 'tag'를 Dataframe 'recommend'에 추가 해줍니다.
* Dataframe 'recommend'에서 컬럼 'song_id'를 기준으로 하여 그룹화하여 카운트하고 'tag'를 기준으로 내림차순 재배열하고 100개로 슬라이싱합니다.
* Dataframe 'recommend의 index를 list화하여 반환해줍니다.(index가 song로 설정되어 있습니다.)
```python
def result_dict_song(plylst_id):
    start = time.time()
    songs = recommend_result_cos(plylst_id)
    tags_plylst = val_train[val_train['id'] == plylst_id]['tags'].tolist()[0]
    tags = tag(songs)
    tags = [i for i in tags if i not in tags_plylst][0:10]
    result_dict = {'id' : plylst_id, 'songs' : songs, 'tags': tags}
    print('result_dict_song complete / time : {}'.format(sec(time.time()-start)))
    return result_dict
```
* 함수 result_dict_song는 변수 'plylst_id'를 받아 노래를 중심으로 유사한 노래와 tag를 받아와 제출에 형식에 맞는 dict형식으로 만드는 함수입니다.
* songs 에는 함수 recommend_result_cos 결과값을 넣고 tags에는 함수 tag 결과값을 넣어줍니다.
* Dataframe 'val_train'에서 'id'가 'plylst_id'인 컬럼 'tags'를 list화 시켜 저장한 변수가 'tags_plylst'입니다.
* 'tags' value에서 'tags_plylst'를 제외시켜줍니다.
* 제출형식에 맞게 dict형식으로 만들어서 반환해 줍니다.
```python
def result_dict_tag(plylst_id):
    start = time.time()
    tags = recommend_result_cos_tags(plylst_id)
    songs_plylst = val_train[val_train['id'] == plylst_id]['songs'].tolist()[0]
    songs = song(tags)
    songs = [i for i in songs if i not in songs_plylst][0:10]
    result_dict = {'id' : plylst_id, 'songs' : songs, 'tags': tags}
    print('result_dict_song complete / time : {}'.format(sec(time.time()-start)))
    return result_dict
```
* 함수 result_dict_tag는 변수 'plylst_id'를 받아 tag를 중심으로 유사한 노래와 tag를 받아와 제출에 형식에 맞는 사전형식으로 만드는 함수입니다.
* tags 에는 함수 recommend_result_cos_tags 결과값을 넣고 songs에는 함수 song 결과값을 넣어줍니다.
* Dataframe 'val_train'에서 'id'가 'plylst_id'인 컬럼 'songs'를 list화 시켜 저장한 변수가 'songs_plylst'입니다.
* 'songs' value에서 'songs_plylst'를 제외시켜줍니다.
* 제출형식에 맞게 dict형식으로 만들어서 반환해 줍니다.
```python
val_test = test[['tags','id','songs','plylst_title']]
val_test['tags'] = val_test['tags'].apply(lambda x : len(x))
val_test['songs'] = val_test['songs'].apply(lambda x : len(x))
```
* Dataframe 'test'의 컬럼 'id','tags','song','plylst_title'를 Dataframe 'val_test'에 저장합니다.
* Dataframe 'val_test'의 컬럼 'tags'와 'songs'를 list안의 value 개수로 변환시켜줍니다.
```python
val_id = val_test['id'].tolist()
val_tags_cnt = val_test['tags'].tolist()
val_songs_cnt = val_test['songs'].tolist()
```
* Dataframe 'val_test'의 컬럼 'id'와 'tags', 'songs'을 각각 list화 시켜줍니다.
```python
dic_last_all = []
```
* list 'dic_last_all'를 만들어줍니다.
```python
for i in range(0,len(val_id)) : 
    if val_tags_cnt[i] == 0 :
        dic = result_dict_song(val_id[i])
        dic_last_all.append(dic)
    elif val_songs_cnt[i] == 0:
        dic = result_dict_tag(val_id[i])
        dic_last_all.append(dic)
    elif val_tags_cnt[i] == 0 and val_songs_cnt[i] == 0 :
        dic = {'id' : val_id[i], 'songs' : [], 'tags' : []}
        dic_last_all.append(dic)
    else :
        dic = result_dict_song(val_id[i])
        dic_last_all.append(dic) 
```
* val_id의 value 수만큼 반복하여 돌려줍니다.
* val_tags_cnt의 value가 0일때 함수 result_dict_song를 동작시켜서 list 'dic_last_all' 추가합니다.
* val_songs_cnt의 value가 0일때 함수 result_dict_tag를 동작시켜서 list 'dic_last_all' 추가합니다.
* val_tags_cnt의 value가 0이고 val_songs_cnt의 value가 0일때 {'id' : val_id[i], 'songs' : [], 'tags' : []}을 list 'dic_last_all' 추가합니다.
* 나머지 경우에는 함수 result_dict_song를 동작시켜서 list 'dic_last_all' 추가합니다.(나머지 경우란 val_tags_cnt와 val_songs_cnt가 둘다 있을 경우 입니다.)



#### (5) 추천을 받지 못한 경우와 조건에 만족하지못한 플레이리스트 채우기
```python
result_data = pd.DataFrame(dic_last_all)
```
* list 'dic_last_all'를 Dataframe화 시켜주고 'result_data'에 저장합니다.
```python
result_data['songs'] = result_data['songs'].apply(lambda x : len(x))
result_data['tags'] = result_data['tags'].apply(lambda x : len(x))
```
* Dataframe 'result_data'의 컬럼 'song'와 'tags'를 각각의 value 개수화하여 저장합니다.
```python
song_non = [144663,
 116573,
 357367,
 366786,
 654757,
 133143,
 349492,
 675115,
 463173,
 42155,
 396828,
 610933,
 461341,
 174749,
 520093,
 701557,
 549178,
 485155,
 650494,
 523521,
 13281,
 648628,
 449244,
 680366,
 169984,
 422915,
 11657,
 418935,
 187047,
 547967,
 422077,
 350309,
 627363,
 625875,
 300087,
 132994,
 215411,
 427724,
 442014,
 668128,
 582252,
 663256,
 253755,
 643628,
 448116,
 339802,
 348200,
 581799,
 26083,
 37748,
 341513,
 505036,
 199262,
 407828,
 105140,
 68348,
 140867,
 235773,
 209993,
 209135,
 339124,
 487911,
 493762,
 672550,
 509998,
 531820,
 27469,
 157055,
 519391,
 473514,
 232874,
 75842,
 117595,
 446812,
 295250,
 152422,
 224921,
 678762,
 351342,
 15318,
 377243,
 146989,
 246531,
 205179,
 108004,
 645489,
 464051,
 13198,
 302646,
 152475,
 343974,
 236393,
 95323,
 640657,
 459256,
 88503,
 362966,
 674160,
 424813,
 154858]
```
* 'song_non'는 mapping_cnt를 기준으로 한 상위 100개의 song list입니다. 
```python
tag_non = ['기분전환', '감성', '휴식', '발라드', '잔잔한', '드라이브', '힐링', '사랑', '새벽', '밤']
```
* 'tag_non'는 mapping_cnt를 기준으로 한 상위 10개의 tag list입니다. 
```python
song_fill = result_data[result_data['songs']<100].reset_index()
```
* Dataframe 'result_data'의 컬럼 'songs'가 100개미만인 data를 조회하고 index를 초기화하여 'song_fill'에 저장합니다.
```python
song_fill_list = song_fill['index'].tolist()
```
* Dataframe 'song_fill'의 컬럼 'index'를 list화 하여 'song_fill_list'에 저장합니다.
```python
 for i in song_fill_list:
    dic_last_all[i]['songs'].extend(song_non)
    dic_last_all[i]['songs'] = dic_last_all[i]['songs'][0:100]
```
* 추천 받아야할 노래 개수가 100곡이므로 개수가 채워지지 않은 부분에서 mapping_cnt가 높은 상위 노래 100개으로 채워넣어서 개수를 맞춰 주었습니다.
* song_fill_list의 value를 차례대로 넣어서 lsit 'dic_last_all' index의 'songs'에 채워넣고 개수에 맞게 100개씩 슬라이싱합니다.  
```python
tag_fill = result_data[result_data['tags']<10].sort_values('tags',ascending = False).reset_index()
```
* Dataframe 'result_data'의 컬럼 'tags'가 10개미만인 data를 조회하고 index를 초기화하여 'tag_fill'에 저장합니다.
```python
tag_fill_list = tag_fill['index'].tolist()
```
* Dataframe 'tag_fill'의 컬럼 'index'를 list화 하여 'tag_fill_list'에 저장합니다.
```python
for i in tag_fill_list:
    tag = dic_last_all[i]['tags']
    dic_last_all[i]['tags'].extend([i for i in tag_non if i not in tag])
    dic_last_all[i]['tags'] = dic_last_all[i]['tags'][0:10]
```
* 추천 받아야할 tag 개수가 10개이므로 개수가 채워지지 않은 부분에서 mapping_cnt가 높은 상위 tag 10개으로 채워넣어서 개수를 맞춰 주었습니다.
* tag_fill_list의 value를 차례대로 넣어서 lsit 'dic_last_all' index의 'tags'에 채워넣고 개수에 맞게 100개씩 슬라이싱합니다. 
```python
with open('results.json', 'w', encoding='utf-8') as make_file:
    json.dump(dic_last_all, make_file, ensure_ascii=False, indent="\t")
```
* list 'dic_last_all'를 json 파일로 저장해줍니다. 
