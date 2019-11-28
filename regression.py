import pandas as pd
import pandasql as ps
import math
import re
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# --- This file is for all regression models, the goal is to train it on predicted claims from the test set,
# then predict on the 'only claims' on the Submission Set (the dataset we will be handing in)
# All you need to do is:
#   1) make sure that the 'testSet' that you pass in has a 'hasClaim' column set to 0 if there is not a claim
#       or 1 if there is a claim.
#   2) Change the regression model to whatever regression you want to run
#   3) change the name of the file you want to export to as a csv at the bottom
# This will do everything else for you, it will break a

data = pd.read_csv("datasets/trainingset.csv")
testSet = pd.read_csv("logistic_train_result.csv")

# This part is just setting up the training dataset
subset = data.dropna(axis=0, how='any', inplace=False)

train_ratio = 0.75
num_rows = subset.shape[0]
train_set_size = int(train_ratio * num_rows)

data_in = subset.drop('ClaimAmount', axis=1, inplace=False)
data_out = subset.loc[:, 'ClaimAmount']

training_data_in = data_in[:train_set_size]
training_data_out = data_out[:train_set_size]

test_data_in = data_in[train_set_size:]
test_data_out = data_out[train_set_size:]

# This is where you can drop any features that you may want to
drop_features = ['feature3', 'feature4', 'feature5', 'feature7',
                                      'feature9', 'feature13', 'feature14',
                                      'feature18']

# training_data_in = training_data_in.drop(drop_features, axis=1)
# test_data_in = test_data_in.drop(drop_features, axis=1)

# ---- This part is where you can swap out one regression model for another
tuned_parameters = {'n_estimators': [10, 20, 30], 'max_depth': [None, 1, 2, 3], 'min_samples_split': [2, 3, 4]}
clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=2,
                   n_jobs=-1, verbose=1)
clf.fit(training_data_in, training_data_out)
# ypred = clf.predict(testSet)
# mae = np.mean([abs(test_data_out.values[i] - ypred[i]) for i in range(len(ypred))])
# print(mae)

# This is the part where it breaks apart the test set, predicts on only claims, and re-merges the dataset. No need to change anything here
fullTest = testSet.copy()
testSet = testSet[testSet['hasClaim'] == 1]
fullTest = fullTest[fullTest['hasClaim'] == 0]

testSet.drop("hasClaim", axis=1, inplace=True)
fullTest.drop("hasClaim", axis=1, inplace=True)

y_pred = clf.predict(testSet)

testSet['ClaimAmount'] = y_pred
fullTest['ClaimAmount'] = 0
combinedResult = testSet.combine_first(fullTest)
combinedResult = combinedResult.loc[:, ['rowIndex', 'ClaimAmount']]
combinedResult['rowIndex'] = (combinedResult['rowIndex']).astype(int)

# Change the filename to whatever you want to export your csv as, the format will be how it should be for submission
combinedResult.to_csv('logistic_random_forest_2.csv', index=False)





