---
title: 주택 가격 예측(3) - Modeling(1)  
date : 2020-05-31 14:50:30 -0400
categories : Kaggle update Project
---

* 모델링을 위한 각각의 패키지와 모듈 불러오기
```python
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
```
* Lasso Regularization
  + 선형 회귀의 Regularization(규제)을 적용하는 대안이다.
  + 계수를 0에 가깝게 만들려고 하여 이르 L1규제라고 하며, 어떤 계수는 0이 되기도 하는
데 이는 완전히 제외하는 Feature가 생긴다는 의미한다.
  + Feature 선택이 자동으로 이루어진다고 볼 수 있다.
  + Alpha 값의 기본값은 1.0이며, 과소 적합을 줄이기 위해서는 이 값을 줄여야 한다.
  + Grid Search 또는 Random Search를 alpha에 넣어서 사용한다.
  + Max_iter는 반복 실행하는 최대 횟수를 의미한다.
```python
lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
```

* Elastic Net Regularization
  + 변수도 줄이고, 분산 또한 줄이고 싶은 경우에 사용한다.
  + Lasso Regularization 와 Ridge Regularization의 혼합형이다.
  + 예) 실제 영향을 주는 변수는 A인데, 같이 붙어 다니는 B가 있는 경우에 통계적으로 B도
영향을 주는 것처럼 판단이 될 수 있다.
    - Lasso만 하면 A가 사라지고 B만 남거나, Ridge만 하면 beta를 전체적으로 줄여줘 서 변수 선택이 안되는 문제가 생긴다.
    - 이런 경우를 해결하기 위해 사용한다.
```python
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio= .9, random_state=3))
```

* Ridge Regularization
  + 회귀를 위한 선형 모델이다.
  + 가중치(w)의 모든 원소가 0에 가깝게 만들어 모든 Feature가 주는 영향을 최소화(기울
기를 작게 만든다) 한다.
  + Regularization(규제)은 과대 적합이 되지 않도록 모델을 강제로 제한한다는 의미한다.
  + Grid Search 또는 Random Search를 alpha에 넣어서 사용한다.
  + Max_iter는 반복 실행하는 최대 횟수를 의미한다.
```python
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
```

* Gradient Boost Regularization
  + 여러개의결정트리를묶어강력한모델을만드는또다른앙상블기법이다.
  + 회귀와 분류에 모두 사용할 수 있다.
  + Random Forest와 달리 이진 트리의 오차를 보완하는 방식으로 순차적으로 트리를 만들
었다.
  + 무작위성이 없고 강력한 사전 가지치기가 사용된다.
  + 1~5개의 깊지 않은 트리를 사용하기 때문에 메모리를 적게 사용하고 예측이 빠르다.
  + Learning_rate : 오차를 얼마나 강하게 보정할 것인지를 제어한다.
  + N_estimater : 값을 키우면 앙상블에 트리가 더 추가되어 모델의 복잡도가 커지고 훈련 세트에서의 실수를 바로잡을 기회가 많아지지만, 너무 크면 모델이 복잡해지고 과대 적 합이 될 수 있다.
```python
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features=
'sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state=5)
```

* Xgboost Regularization
  + Gradient Boost 알고리즘의 단점을 보완 해주기 위해 나왔다.
  + 과대 적합 방지가 가능한 규제가 포함되어 있다.
  + 분류와 회귀가 둘 다 가능하다.
  + 조기 종료(early stopping)을 제공한다.
  + Gradient Boost를 기반으로 한다.(즉, 앙상블 Boosting의 특징인 가중치 부여를 경사하 강법으로 한다)
```python
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3,min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571, subsample=0.5213, silent=1, random_state=7, nthread=-1)
```

* Xgboost의 hyperparmeter
  + N_estimators(혹은 num_boost_round) : 결정 트리의 개수
  + Max_depth : 트리의 깊이
  + Colsample_bytree : 칼럼의 샘플링 비율
  + Subsample : weak learner가 학습에 사용하는 데이터 샘플링 비율
  + Learning_rate : 학습률
  + Min_split_loss : 리프 노드를 추가적으로 나눌지 결정하는 값
  + Reg_lambda : L2 규제
  + Reg_alpha : L1 규제

* LightGBM Regularization
  + Xgboost가 학습시간이 느린 단점을 보완 해주기 위해 나왔다.
  + 대용량 데이터 처리가 가능하고, 다른 모델들보다 더 적은 자원을 사용한다.
  + 속도가 빠르며, GPU까지 지원해주기도 한다.
  + 너무 적은 수의 데이터를 사용하면 과대 적합의 문제 발생한다.
  + Leaf Wise(리프 중심) 트리 분할을 사용한다.
  + Tree의 균형은 맞추지 않고 leaf node를 지속적으로 분할하면서 진행한다.
  + Leaf node를 max delta loss 값을 가지는 Leaf node를 계속 분할해 간다.
  + 비 대칭적이고 깊은 Tree가 생성되지만 동일한 leaf를 생성할 때 leaf-wise는 Level-wise 보다 손실을 줄일 수 있다.
```python
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,learning_rate=0.05, n_estimators=720, max_bin=55, bagging_fraction=0.8 , bagging_freq=5, feature_fraction=0.2319, feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf=6, min_sum_hessian_in_leaf=11)
```

* LightGBM의 hyperparmeter
  + N_estimators : 반복하려는 트리의 개수
  + Learning_rate : 학습률
  + Max_depth : 트리의 최대 깊이
  + Min_child_samples : 리프 노드가 되기 위한 최소한의 샘플 데이터 수
  + Num_leaves:하나의트리가가질수있는최대리프개수
  + Fearure_fraction : 트리를 학습할 때마다 선택하는 feature의 비율
  + Reg_lambda : L2 규제
  + Reg_alpha : L1 규제

* LightGBM의 Leaf-wise 트리 분석
  + 기존의 Tree들은 Tree의 depth(깊이)를 줄이기위해서 Level-wise(균형 트리)분할을 사용하는데, LightGBM은 위와 같이 다르게 모델이 동작한다.
  + 균형을 잡아주어야 하기 때문에 Tree의 depth가 줄어든다, 그 대신 그 균형을 잡아주기 위한 연산이 추가 되는 것이 단점이다.
  <img src="https://user-images.githubusercontent.com/60723495/83345630-6f67b480-a350-11ea-8256-fda78e1e3d1a.png" width="600" height="300">

* RMSLE(Root Mean Square Logarithmic Error)
  + 과대평가 된 항목보다는 과소평가 된 항목에 페널티를 준다.
  + 오차를 제곱해서 평균한 값의 제곱근으로 값이 적을수록 정밀도가 높다.
  + 0에 가까운 값이 나올수록 정밀도가 높은 값이다.
  + RMSE와 RMSLE 차이
   - 아웃라이어에 강건 해진다.
   - 상대적 Error를 측정해준다.
   - Under Estimation에 큰 페널티를 부여한다.
```python
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring='neg_mean_squared_error',cv=kf))
    return(rmse)
```    
