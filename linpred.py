import pandas as pd
import pandasql as ps
import math
import re
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


data = pd.read_csv("datasets/trainingset.csv")
testSet = pd.read_csv("datasets/testset.csv")

subset = data.dropna(axis=0, how='any', inplace=False)
data = data[data['ClaimAmount'] != 0]
print(data.head(100).to_string())
train_ratio = 0.75
num_rows = subset.shape[0]
train_set_size = int(train_ratio * num_rows)

data_in = subset.drop('ClaimAmount', axis=1, inplace=False)
data_out = subset.loc[:, 'ClaimAmount']

training_data_in = data_in[:train_set_size]
training_data_out = data_out[:train_set_size]

test_data_in = data_in[train_set_size:]
test_data_out = data_out[train_set_size:]

temp_test_in = test_data_in

drop_features = ['feature3', 'feature4', 'feature5', 'feature7',
                                      'feature9', 'feature13', 'feature14',
                                      'feature18']

training_data_in = training_data_in.drop(drop_features, axis=1)
testSet = testSet.drop(drop_features, axis=1)


test_data_in = test_data_in.drop(drop_features, axis=1)


linreg = LinearRegression()
linreg.fit(training_data_in, training_data_out)
price_pred =linreg.predict(test_data_in)

lst = []
for i in range(len(price_pred)):
    tmp = abs(test_data_out.values[i] - price_pred[i])
    lst.append(tmp)
mae = np.mean(lst)

# export = pd.DataFrame(columns=['rowIndex', 'ClaimAmount'])
# export['rowIndex'] = range(0, 30000)
# export['ClaimAmount'] = price_pred[0:30000]
# export.to_csv('random_forest.csv', index=False)

print('Mean Absolute Error = ', mae)