from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

# loading csv's
# test_csv = pd.read_csv('testset.csv')
data = pd.read_csv('trainingset.csv')

# Change ClaimAmount to 0's and 1's for whether they claimed or not.
for i, row in data.iterrows():
    claim_val = 0
    if data.at[i, 'ClaimAmount'] > 0:
        claim_val = 1
    data.at[i, 'ClaimAmount'] = claim_val

# 75 to 25 ratio
train_ratio = 0.75
num_rows = data.shape[0]
train_set_size = int(train_ratio * num_rows)

shuffled_indices = list(range(num_rows))
random.seed(42)
random.shuffle(shuffled_indices)

train_indices = shuffled_indices[:train_set_size]
test_indices = shuffled_indices[train_set_size:]

train_data = data.iloc[train_indices, :]
test_data = data.iloc[test_indices, :]

# Filter out 0's in the test data, comment out if want to use entire training set.
indexNames = test_data[ test_data['ClaimAmount'] != 1 ].index
test_data = test_data.drop(indexNames, inplace=False)

# Obtain training rows with claims
indexNames = train_data[ train_data['ClaimAmount'] != 1 ].index
train_claims = train_data.drop(indexNames, inplace=False)

# Obtain training rows without claims
indexNames = train_data[ train_data['ClaimAmount'] != 0 ].index
train_no_claims = train_data.drop(indexNames, inplace=False)

no_claims_rows = train_no_claims.shape[0]
no_claims_size = int(0.05 * no_claims_rows)

print("Using", train_claims.shape[0], "claims")
print("Using", no_claims_size, "non-claims")

# Create training set with specific amount of 1's and 0's
train_data = train_claims.append(train_no_claims[:no_claims_size])

num_rows = train_data.shape[0]

shuffled_indices = list(range(num_rows))
# Use for random shuffling, can comment out if using k-fold cv
random.seed(42)
random.shuffle(shuffled_indices)

train_indices = shuffled_indices[:num_rows]
train_data = train_data.iloc[train_indices, :]

# Filter out certain columns
training_data_in = train_data.loc[:, ['feature3', 'feature4', 'feature5', 'feature7',
                                      'feature9', 'feature11', 'feature13', 'feature14',
                                      'feature15', 'feature16', 'feature18', 'ClaimAmount']]

test_data_in = test_data.loc[:, ['feature3', 'feature4', 'feature5', 'feature7',
                                      'feature9', 'feature11', 'feature13', 'feature14',
                                      'feature15', 'feature16', 'feature18', 'ClaimAmount']]

training_data_in = training_data_in.drop('ClaimAmount', axis=1, inplace=False)
training_data_out = train_data.loc[:, 'ClaimAmount']

test_data_in = test_data_in.drop('ClaimAmount', axis=1, inplace=False)
test_data_out = test_data.loc[:, 'ClaimAmount']

clf = RandomForestClassifier(n_estimators=100)
clf.fit(training_data_in, training_data_out)

y_pred = clf.predict(test_data_in)
y_train = clf.predict(training_data_in)

# Print accuracy of 'test' and 'training'

print('Acc:', metrics.accuracy_score(test_data_out, y_pred))
print('2nd Acc:', metrics.accuracy_score(training_data_out, y_train))

# Code to export dataframe

#export = test_csv.copy()
#export['PredictedCategory'] = y_true_test
#indexNames = export[ export['PredictedCategory'] != 1 ].index
#export.drop(indexNames, inplace=True)
#export.to_csv('category_test_randforest.csv')
#print(export)