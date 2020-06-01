---
title: 주택 가격 예측 - Modeling(2)  
date : 2020-05-31 15:42:30 -0400
categories : Kaggle update Project
---

* 각각의 모델로 구해지는 RMSLE의 중간값과 평균을 출력해준다.
```python
score = rmsle_cv(lasso)
print('\nLasso score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
>>> Lasso score: 0.1115 (0.0074)
```
```python
score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
>>> ElasticNet score: 0.1116 (0.0074)
```
```python
score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
>>> Kernel Ridge score: 0.1153 (0.0075)
```
```python
score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
>>> Gradient Boosting score: 0.1167 (0.0083)
```
```python
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
>>> LGBM score: 0.1155 (0.0067)
```
```python
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
>>> Xgboost score: 0.1158 (0.0064)
```

<br>

* Stacking(Meta Modeling)
  + 서로 다른 모델들을 조합해서 최고의 성능을 내는 모델을 생성하는 기법이다.
  + 서로의 장점은 취하고 약점은 보완 할 수 있게 된다.
  + 필요한 데이터 연산량이 많아야 한다.
  + 문제에 따라 정확도를 요구하기도 하지만, 안정성을 요구하기도 한다. 
  + 문제에 적절한 모델을 선택하는 것이 중요하다.

<br>

* 여러개의 기본 모델의 평균화를 시키는 클래스 
```python
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            model.fit(X,y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
```

<br>

* 클래스에 넣어서 위의 4개의 모델(ElasticNet 과 GradientBoost, kernel Ridge, lasso)의 RMSLE 값을 평균화하여 출력해준다.
```python
averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
score = rmsle_cv(averaged_models)
print('Averaged base models score : {:.4f} ({:.4f})\n'.format(score.mean(),score.std()))
>>> Averaged base models score : 0.1087 (0.0077)
```

<br>

* Stacking 기법을 사용하여 평균화 시키는 클래스
```python
class StackingAveragedModels(BaseEstimator,RegressorMixin,TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds =5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X,y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)
```

<br>

* 최종 Stacking 기법 설명
<img src="https://user-images.githubusercontent.com/60723495/83346389-d4bea400-a356-11ea-8ca9-e10ccd3989f8.png" width="700" height="300">

  + K-Fold를 Fold의 값에 5를 주어 각각의 모델을 5등분을 한다.
  + Stacking기법을 사용할 때 base Model 과 meta Model 주어지는데 Fold값이 5 이면 base Model은 3개의 Model을 주어야하고 나머지 2개의 Model에는 meta Model이 들어가게 된다. (Out-Of-Fold의 정의가 들어간다)
  + 5개의 모델에 대한 값을 모아서 새로운 Feature을 만들고 학습 시킨다.
  + Test file의 값을 예측하게 된다.

<br>

* Stacking 기법을 사용하기위해 base_models를 ElasticNet 과 GradientBoost, Kernel Ridge로 meta_model에 lasso를 넣어 평준화 하였다.
```python
stacked_averaged_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model= lasso)
score = rmsle_cv(stacked_averaged_models)
print('Stacking Averaged models score : {:.4f} ({:.4f})'.format(score.mean(),score.std()))
>>> Stacking Averaged models score : 0.1081 (0.0073)
```

<br>

* RMSLE를 구해주는 함수 
```python
def rmsle(y, y_pred) : 
    return np.sqrt(mean_squared_error(y, y_pred))
```

<br>

* Stacking 기법으로 RSMLE를 출력
```python
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))
>>> 0.07839506096666429
```

<br>

* Xgboost 모델로 RSMLE를 출력
```python
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))
>>> 0.0786103062413744
```

<br>

* LightGBM 모델로 RSMLE를 출력
```python
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))
>>> 0.07226288640876002
```

<br>

* 위에 3개 모델의 RSMLE를 퍼센트로 나누어 더하여 출력한다
```python
print('RMSLE score on train data : ')
print(rmsle(y_train, stacked_train_pred*0.80 + xgb_train_pred*0.10 + lgb_train_pred*0.10))
>>> RMSLE score on train data : 0.07559500505171629
```
  + 이렇게 가중치를 나눈 이유는 LightGBM과 Xgboost보다 stacking기법의 값이 RSMLE 제일 낮은 평균화된 값을 가지기 때문이다.

<br>

* 변수에 저장한다
```python
ensemble = stacked_pred*0.80 + xgb_pred*0.10 + lgb_pred*0.10
```

<br>

* test.csv의 id를 토대로 값을 SalePrice의 값에 넣어서 submission.csv에 저장한다.
```python
sub = pd.DataFrame()
sub['id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('data/submission.csv',index=False)
```

[프로그램 코드 다운로드](https://github.com/parksengsik/parksengsik.github.io/blob/master/Programfile/house_project_programming.zip)


