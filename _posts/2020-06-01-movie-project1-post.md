---
title: 영화 추천 시스템(1) - 추천 시스템 이란?  
date : 2020-05-31 15:42:30 -0400
categories : Kaggle update Project
---

### 영화 추천 시스템

#### <추천 시스템의 기본>
##### 1. 콘텐츠 기반 필터링(content based filtering) 
* 사용자가 특정 아이템을 선호하는 경우 그 아이템과 비슷한 콘텐츠를 가진 아이템을 추천 해주는 방식.
* ex) 사용자 A가 item A에 굉장히 높은 평점을 주었을 때 그 item이 액션영화이며 '박성식'이라는 감독이었으면 '박성식' 감독의 다른 영화를 추천해주는 것 


##### 2. 협업필터링(collaborative filtering)
* 메모리기반이며, 종류로는 최근접 이웃 기반 과 잠재요인 협업필터링으로 나누어진다.
* 사용자가 아이템에 매긴 평점, 상품 구매 이력등의 사용자 행동양식을 기반으로 추천해주는 것을 의미 한다.



##### 2-1. 최근접 이웃 기반(nearest neighbor collaborative filtering)
* 사용자 - 아이템 행렬에서 사용자가 아직 평가하지 않은 아이템을 예측하는 것이 목표로 하는 협업 필터링을 말한다.
* 사용자 - 아이템 평점행렬과 같은 모습을 가지고 있어야 한다, column에는 item을 row에는 user을 가져야한다.
<img src="https://user-images.githubusercontent.com/60723495/83366679-958e6280-a3eb-11ea-97b0-80fc29b7e183.png" width="700" height="300">
* 그렇기 때문에 위 그림의 왼쪽과 같이 데이터가 주어지면 오른쪽 데이터로 변경하여야 한다.(이러한 작업을 Python pandas에서 pivot table로 지원)
* 단점은 공간 낭비가 많이 된다, 그 이유는 데이터가 드물기 때문이다.
* 최근접 이웃 기반은 또 2가지로 나뉘어진다.
  + 사용자 기반 : 비슷한 고객들이 ~한 item을 소비했다.
  + 아이템 기반 : ~한 item을 소비한 고객들은 다음과 같은 상품도구도 구매 했다.(사용자 기반보다 더 정확도가 높다)



##### 2-2. 잠재요인 기반
* 행렬분해(matrix factorization)을 기반으로 사용한다, 이는 대규모 다차원 행렬을 SVD와 같은 차원 감소기법으로 분해하는 과정에서 잠재요인을 찾아내어 뽑아내는 방법이다.
* 앞에서 말한 아이템 기반 최근접 이웃 기반보다 행렬분해방법이 더 많이 사용한다, 그 이유는 matrix factorization의 장점으로 공간을 더 잘 활용할 수 있다.
* 행렬분해으로 진행하는 collaborative filtering은 사용자 - 아이템 행렬 데이터를 이용해 '잠재요인'을 찾아낸다, 즉, 사용자 - 아이템 행렬을 '사용자 - 잠재요인', '아이템 - 잠재요인' 행렬로 분해한다.
<img src="https://user-images.githubusercontent.com/60723495/83367382-37637e80-a3ef-11ea-8eed-7ca230aa9c44.png" width="700" height="300">
* 행렬분해로 잠재요인을 찾아내는 방법의 장점은 '저장공간 절약'이다.
<img src="https://user-images.githubusercontent.com/60723495/83367750-169c2880-a3f1-11ea-88c6-4d75fd0fd943.png" width="700" height="300">

참고 사이트 2 : <https://lsjsj92.tistory.com/563?category=853217>
참고 사이트 2 : <https://lsjsj92.tistory.com/564?category=853217>
