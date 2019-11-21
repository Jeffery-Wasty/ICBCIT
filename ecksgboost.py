from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import xgboost as xgb

data = pd.read_csv('datasets/trainingset.csv')

original_data = data.copy()

for i, row in data.iterrows():
    claim_val = 0
    if data.at[i, 'ClaimAmount'] > 0:
        claim_val = 1
    data.at[i, 'ClaimAmount'] = claim_val

train_ratio = 0.50
num_rows = data.shape[0]
train_set_size = int(train_ratio * num_rows)

shuffled_indices = list(range(num_rows))

train_indices = shuffled_indices[:train_set_size]
test_indices = shuffled_indices[train_set_size:]

original_data = original_data.iloc[train_indices, :]
train_data = data.iloc[train_indices, :]
test_data = data.iloc[test_indices, :]

#indexNames = test_data[ test_data['ClaimAmount'] != 1 ].index
#test_data.drop(indexNames, inplace=True)

training_data_in = train_data.loc[:, ['feature4', 'feature9', 'feature13', 'feature14', 'feature15', 'feature16',
                                      'feature18', 'ClaimAmount']]

test_data_in = test_data.loc[:, ['feature4', 'feature9', 'feature13', 'feature14', 'feature15', 'feature16',
                                      'feature18', 'ClaimAmount']]


training_data_in = training_data_in.drop('ClaimAmount', axis=1, inplace=False)
training_data_out = train_data.loc[:, 'ClaimAmount']

test_data_in = test_data_in.drop('ClaimAmount', axis=1, inplace=False)
test_data_out = test_data.loc[:, 'ClaimAmount']

data_dmatrix = xgb.DMatrix(data=training_data_in, label=training_data_out)
xg_reg = xgb.XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

xg_reg.fit(training_data_in,training_data_out)

y_pred = xg_reg.predict(test_data_in)

rmse = np.sqrt(mean_squared_error(test_data_out, y_pred))
print("RMSE: %f" % (rmse))
print(test_data_out)
print(y_pred)
print('Acc:', metrics.accuracy_score(test_data_out, y_pred))

#export = original_data.copy()
#export['PredictedCategory'] = y_train
#indexNames = export[ export['PredictedCategory'] != 1 ].index
#export.drop(indexNames, inplace=True)
#export.to_csv('category_trained.csv')
#print(export)