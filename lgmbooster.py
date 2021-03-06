from sklearn import metrics
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import random
from sklearn import svm


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
#test_csv = pd.read_csv('testset.csv')
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
training_data_in = train_data.loc[:, ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7',
                                      'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature14',
                                      'feature15', 'feature16', 'feature17', 'feature18', 'ClaimAmount']]

# Add Claim Amount when training, remove when testing
test_data_in = test_data.loc[:, ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7',
                                      'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature14',
                                      'feature15', 'feature16', 'feature17', 'feature18', 'ClaimAmount']]

training_data_in = training_data_in.drop('ClaimAmount', axis=1, inplace=False)
training_data_out = train_data.loc[:, 'ClaimAmount']

# Uncomment when training, comment when testing
test_data_in = test_data_in.drop('ClaimAmount', axis=1, inplace=False)
test_data_out = test_data.loc[:, 'ClaimAmount']

# Cross Validation

random_grid = {
    'max_depth': range (2, 15, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05],
#    'eta': [.3, .2, .1, .05, .01, .005],
    'min_child_weight':range(1,6,2),
    'gamma':[i/10.0 for i in range(0,5)],
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}

params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
}

data_dmatrix = xgb.DMatrix(data=training_data_in, label=training_data_out)
data_testmatrix = xgb.DMatrix(data=test_data_in, label=test_data_out)
num_boost_rounds = 999

#clf = xgb.XGBClassifier(objective='binary:logistic', nthread=4, seed=42)
# {'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 180}

#best_clf = xgb.train(random_grid, )
#best_clf.fit(training_data_in, training_data_out)

print("Starting...")
#best_clf = GridSearchCV(estimator=clf, param_grid=random_grid, scoring='f1', n_jobs=4, cv=5)

#best_clf.fit(training_data_in, training_data_out)

param_grid = {
    'num_leaves': [31, 127],
    'reg_alpha': [0.1, 0.5],
    'min_data_in_leaf': [30, 50, 100, 300, 400],
    'lambda_l1': [0, 1, 1.5],
    'lambda_l2': [0, 1]
    }
lgb_estimator = lgb.LGBMClassifier(boosting_type='gbdt',  objective='binary', num_boost_round=2000, learning_rate=0.01)
best_clf = GridSearchCV(estimator=lgb_estimator, param_grid=param_grid, cv=5, n_jobs=4, scoring='f1')
best_clf.fit(X=training_data_in, y=training_data_out)

print(best_clf.best_params_, best_clf.best_score_)

prediction = best_clf.predict(test_data_in)

# y_pred = best_clf.predict(test_data_in)
y_train = best_clf.predict(training_data_in)

# Print accuracy of 'test' and 'training'

print('Acc:', metrics.accuracy_score(test_data_out, prediction))
print('2nd Acc:', metrics.accuracy_score(training_data_out, y_train))
