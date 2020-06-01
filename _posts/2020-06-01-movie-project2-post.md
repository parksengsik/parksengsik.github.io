---
title: 영화 추천 시스템(2) - 프로그래밍(라이브러리)  
date : 2020-06-01 10:24:30 -0400
categories : Kaggle update Project
---

#### 영화추천 프로그래밍

##### 활용할 라이브러리와 패키지 불러오기
```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats 
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
import warnings; warnings.simplefilter('ignore')
```
* %matplotlib inline : Jupyter 와 같은 IPython에서 matplotlib으로 plot을 만드는 데 사용한다.
* pandas : 데이터 조작과 분석에 쓰이는 라이브러리
* numpy : 행렬이나 다차원배열을 처리하는 라이브러리
* matplotlib : plot을 생성을 지원하는 라이브러리
* seaborn : matplotlib을 기반으로 다양한 색상 테마와 통계용 차트 등의 기능을 추가한 시각화 패키지
* scipy.stats : 과학기술계산용 함수안에 통계부분
* ast.literal_eval : 문법을 구조화 시켜주는 모듈로 string안에 dict를 다시 구조화한다.
* sklearn.feature_extraction.text : 문서 전처리용 클래스 제공
  + CountVectorizer : 문서 집합에서 단어 토큰을 생성하고 각 단어의 수를 세어 Bow 인코딩한 벡터를 만든다.
  + TfidfVectorizer : CountVectorizer와 비슷하지만 TF-IDF 방식으로 단어의 가중치를 조정한 Bow 벡터를 만든다
* sklearn.metrics.pairwise : 벡터 배열 X와 선택적 Y에서 거리 행렬을 처리하는 라이브러리
  + linear_kernel : X와 Y 사이의 선형 커널을 계산
  + cosine_silmilarity : X와 Y의 샘플 간 코사인 유사성을 계산
* nltk.stem.snowball.SnowballStemmer : 눈덩이 형태소 분석기(영어 외의 13개 국가의 언어에 대한 stemming을 지원)
* nltk.stem.wordnet.WordNetLemmatizer : stemmer처럼 형태소를 분석하지만 설정에 따라 동사, 복수명사등을 출력할 수 있다.
* nltk.corpus.wordnet : 영어 용 어휘 데이터베이스이며 nltk corpuss의 일부
* surprise : 명시적인 등급 데이터를 처리하는 추천시스템을 구축하고 분석하는 scikit
  + SVD : 특이값 분해(행렬을 특정한 구조로 분해하는 방식) 알고리즘
  + accurancy : rmse, mae등을 계산할 때 사용
* surprise.model_selection.cross_validate : 교차 유효성 검사
* surprise.model_selection.Kfold : fold를 나눌 때 사용되는 라이브러리
