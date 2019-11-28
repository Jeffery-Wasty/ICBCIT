from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
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
test_csv = pd.read_csv('datasets/testset.csv')
data = pd.read_csv('datasets/trainingset.csv')

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

train_indices = shuffled_indices[:train_set_size]
test_indices = shuffled_indices[train_set_size:]

train_data = data.iloc[train_indices, :]
test_data = test_csv.copy()

# Filter out 0's in the test data, comment out if want to use entire training set.
# indexNames = test_data[ test_data['ClaimAmount'] != 1 ].index
# test_data = test_data.drop(indexNames, inplace=False)

# Obtain training rows with claims
indexNames = train_data[train_data['ClaimAmount'] != 1].index
train_claims = train_data.drop(indexNames, inplace=False)

# Obtain training rows without claims
indexNames = train_data[train_data['ClaimAmount'] != 0].index
train_no_claims = train_data.drop(indexNames, inplace=False)

no_claims_rows = train_no_claims.shape[0]
no_claims_size = int(0.1 * no_claims_rows)

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
training_data_in = train_data.loc[:, ['feature4', 'feature9', 'feature13', 'feature14', 'feature15',
                                      'feature18', 'ClaimAmount']]

# Add Claim Amount when training, remove when testing
test_data_in = test_data.loc[:, ['feature4', 'feature9', 'feature13', 'feature14', 'feature15',
                                 'feature18']]

training_data_in = training_data_in.drop('ClaimAmount', axis=1, inplace=False)
training_data_out = train_data.loc[:, 'ClaimAmount']

# Uncomment when training, comment when testing
#test_data_in = test_data_in.drop('ClaimAmount', axis=1, inplace=False)
#test_data_out = test_data.loc[:, 'ClaimAmount']

# Cross Validation

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

#model1 = GridSearchCV(clf, param_grid=random_grid, n_jobs=-1)
#model1.fit(training_data_in, training_data_out)

#print("Best Hyper Parameters:\n",model1.best_params_)
best_clf = RandomForestClassifier(bootstrap=False, max_depth=10, max_features='auto',
                                  min_samples_leaf=4, min_samples_split=2, n_estimators=300)
best_clf.fit(training_data_in, training_data_out)

prediction = best_clf.predict(test_data_in)

# y_pred = clf.predict(test_data_in)
# y_train = clf.predict(training_data_in)

# Print accuracy of 'test' and 'training'

#print('Acc:', metrics.accuracy_score(test_data_out, y_pred))
#print('2nd Acc:', metrics.accuracy_score(training_data_out, y_train))

# Code to export dataframe

print("test_data:", test_data.shape[0])
print("prediction:", prediction.shape[0])

export = test_data.copy()
export['PredictedCategory'] = prediction
export.to_csv('category_true_test_randforest.csv')
print(export)

print(export.shape[0])
