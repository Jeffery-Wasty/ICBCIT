import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Loading csvs
data = pd.read_csv("trainingset.csv")
true_data = pd.read_csv("testset.csv")
testSet = pd.read_csv("category_true_test_xgb.csv")
originalTest = testSet.copy()
subset = data.dropna(axis=0, how='any', inplace=False)
train_ratio = 0.75
num_rows = subset.shape[0]
train_set_size = int(train_ratio * num_rows)

# in/out
data_in = subset.drop('ClaimAmount', axis=1, inplace=False)
data_out = subset.loc[:, 'ClaimAmount']

training_data_in = data_in[:train_set_size]
training_data_out = data_out[:train_set_size]

test_data_in = data_in[train_set_size:]
test_data_out = data_out[train_set_size:]

drop_features = ['feature4', 'feature9', 'feature13', 'feature14', 'feature15',
                                      'feature18']

training_data_in = training_data_in.drop(drop_features, axis=1)

testSet = testSet.drop(drop_features, axis=1)
testSet = testSet.drop(['PredictedCategory'], axis=1)

test_data_in = test_data_in.drop(drop_features, axis=1)

true_data = true_data.drop(drop_features, axis=1)

clf = RandomForestRegressor(min_samples_split=3, n_estimators=30)
clf.fit(training_data_in, training_data_out)
y_pred_val = clf.predict(true_data)

export = pd.DataFrame(columns=['ClaimAmount'])
export['ClaimAmount'] = y_pred_val
for i in range(len(export)):
    if originalTest.iloc[i]['PredictedCategory'] == 0:
        export.iloc[i, export.columns.get_loc('ClaimAmount')] = 0
export.to_csv('xgbv1-randomforest-true.csv')
