import pandas as pd
import pandasql as ps
import math
import re
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# separates the dataframe you pass in into a test dataframe of half claims, half not, to train on
def halfClaimsDataset(data):
    filterOnes = data[data['ClaimAmount'] > 0]
    filterZero = data[data['ClaimAmount'] == 0]
    combined = filterOnes.append(filterZero.head(2* filterOnes.shape[0]), ignore_index=True)
    combined['hasClaim'] = (combined['ClaimAmount'] > 0).astype(int)
    return combined


data = pd.read_csv("datasets/trainingset.csv")
trainingData = pd.read_csv("datasets/trainingset.csv")
testSet = pd.read_csv("datasets/testset.csv")

data['hasClaim'] = (data['ClaimAmount']).astype(int)
data = halfClaimsDataset(data)

trainingData['hasClaim'] = (trainingData['ClaimAmount'] > 0).astype(int)
print(trainingData.head(500).to_string())
train_ratio = 0.75
num_rows = data.shape[0]
train_set_size = int(train_ratio * num_rows)

data_out = data.loc[:, 'hasClaim']
data_out_FORMAE = trainingData.loc[:, 'hasClaim']

training_data_in = data[:train_set_size]
training_data_out = data_out[:train_set_size]
training_data_in_FORMAE = trainingData[:train_set_size]
training_data_out_FORMAE = trainingData[:train_set_size]

test_data_in = data[train_set_size:]
test_data_out = data_out[train_set_size:]
test_data_in_FORMAE = trainingData[train_set_size:]
test_data_out_FORMAE = trainingData[train_set_size:]

# making a test set of only claims, to test on
only_claims_test_in = test_data_in[test_data_in['ClaimAmount'] > 0]
only_claims_test_out = only_claims_test_in
only_claims_test_in = only_claims_test_in.drop(['ClaimAmount', 'hasClaim'], axis=1, inplace=False)
only_claims_test_out = only_claims_test_out.loc[:, 'hasClaim']

# dropping columns from training sets
training_data_in = training_data_in.drop(['ClaimAmount', 'hasClaim'], axis=1, inplace=False)
test_data_in = test_data_in.drop(['ClaimAmount', 'hasClaim'], axis=1, inplace=False)

training_data_in_FORMAE = training_data_in_FORMAE.drop(['ClaimAmount', 'hasClaim'], axis=1, inplace=False)
test_data_in_FORMAE = test_data_in_FORMAE.drop(['ClaimAmount', 'hasClaim'], axis=1, inplace=False)

# features to drop
drop_features = ['feature3', 'feature4', 'feature5', 'feature7',
                                      'feature9', 'feature13', 'feature14',
                                      'feature18']

training_data_in = training_data_in.drop('feature1', axis=1)
# testSet = testSet.drop(drop_features, axis=1)

test_data_in = test_data_in.drop('feature1', axis=1)

#initalizing and using logistic regression without parameter tuning, fitting on our half claims / half not claims dataset

param_grid = {
    'num_leaves': [31, 127],
    'reg_alpha': [0.1, 0.5],
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
    }
lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective='binary', num_boost_round=2000, learning_rate=0.01, metric='binary_error')
gsearch = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=2)
lgb_model = gsearch.fit(X=training_data_in, y=training_data_out)

print(lgb_model.best_params_, lgb_model.best_score_)

# Use score method to get accuracy of model on the dataset filled with only claims
score = lgb_model.score(only_claims_test_in, only_claims_test_out)
print("predicting on only claims:", score)

# --------------------------- Test Set

#adding hasClaim column

y_pred = lgb_model.predict(testSet)
testSet['hasClaim'] = y_pred

testSet.to_csv('logistic_train_result.csv', index=False)

# export = pd.DataFrame(columns=['rowIndex', 'ClaimAmount'])
# export['rowIndex'] = range(0, 30000)
# export['ClaimAmount'] = price_pred[0:30000]
# export.to_csv('random_forest.csv', index=False)

# print('Mean Absolute Error = ', mae)