import pandas as pd
import pandasql as ps
import math
import re
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# separates the dataframe you pass in into a test dataframe of half claims, half not, to train on
def hasClaim(data):
    filterOnes = data[data['ClaimAmount'] > 0]
    filterZero = data[data['ClaimAmount'] == 0]
    combined = filterOnes.append(filterZero.head(filterOnes.shape[0]), ignore_index=True)
    combined = combined.sample(frac=1).reset_index(drop=True)
    combined['hasClaim'] = (combined['ClaimAmount'] > 0).astype(int)
    return combined


data = pd.read_csv("datasets/trainingset.csv")
testSet = pd.read_csv("datasets/testset.csv")

data['hasClaim'] = (data['ClaimAmount']).astype(int)

data = hasClaim(data)

subset = data.dropna(axis=0, how='any', inplace=False)

train_ratio = 0.75
num_rows = subset.shape[0]
train_set_size = int(train_ratio * num_rows)

data_out = subset.loc[:, 'hasClaim']

training_data_in = subset[:train_set_size]
training_data_out = data_out[:train_set_size]

test_data_in = subset[train_set_size:]
test_data_out = data_out[train_set_size:]

# making a test set of only claims, to test on
only_claims_test_in = test_data_in[test_data_in['ClaimAmount'] > 0]
only_claims_test_out = only_claims_test_in
only_claims_test_in = only_claims_test_in.drop(['ClaimAmount', 'hasClaim'], axis=1, inplace=False)
only_claims_test_out = only_claims_test_out.loc[:, 'hasClaim']

# dropping columns from training sets
training_data_in = training_data_in.drop(['ClaimAmount', 'hasClaim'], axis=1, inplace=False)
test_data_in = test_data_in.drop(['ClaimAmount', 'hasClaim'], axis=1, inplace=False)

# features to drop
drop_features = ['feature3', 'feature4', 'feature5', 'feature7',
                                      'feature9', 'feature13', 'feature14',
                                      'feature18']

# training_data_in = training_data_in.drop(drop_features, axis=1)
# testSet = testSet.drop(drop_features, axis=1)

# test_data_in = test_data_in.drop(drop_features, axis=1)

#initalizing and using logistic regression without parameter tuning
logistic = LogisticRegression()
logistic.fit(training_data_in, training_data_out)
has_claim_pred =logistic.predict(training_data_in)

print(has_claim_pred[:50])
print(test_data_out.head(50))

# Use score method to get accuracy of model on the dataset filled with only claims
score = logistic.score(only_claims_test_in, only_claims_test_out)
print("predicting on only claims:", score)


# export = pd.DataFrame(columns=['rowIndex', 'ClaimAmount'])
# export['rowIndex'] = range(0, 30000)
# export['ClaimAmount'] = price_pred[0:30000]
# export.to_csv('random_forest.csv', index=False)

# print('Mean Absolute Error = ', mae)