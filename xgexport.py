import pandas as pd
import xgboost as xgb
import random
import pickle

# loading csv's
test_csv = pd.read_csv('testset.csv')
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
test_data = test_csv.copy()

# Filter out 0's in the test data, comment out if want to use entire training set.
#indexNames = test_data[ test_data['ClaimAmount'] != 1 ].index
#test_data = test_data.drop(indexNames, inplace=False)

# Obtain training rows with claims
indexNames = train_data[ train_data['ClaimAmount'] != 1 ].index
train_claims = train_data.drop(indexNames, inplace=False)

# Obtain training rows without claims
indexNames = train_data[ train_data['ClaimAmount'] != 0 ].index
train_no_claims = train_data.drop(indexNames, inplace=False)

no_claims_rows = train_no_claims.shape[0]
no_claims_size = int(0.4 * no_claims_rows)

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
training_data_in = train_data.loc[:, ['feature1', 'feature2', 'feature4', 'feature5', 'feature6', 'feature7',
                                      'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature14',
                                      'feature15', 'feature16', 'feature17', 'feature18', 'ClaimAmount']]

# Add Claim Amount when training, remove when testing
test_data_in = test_data.loc[:, ['feature1', 'feature2', 'feature4', 'feature5', 'feature6', 'feature7',
                                      'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature14',
                                      'feature15', 'feature16', 'feature17', 'feature18']]

training_data_in = training_data_in.drop('ClaimAmount', axis=1, inplace=False)
training_data_out = train_data.loc[:, 'ClaimAmount']

# Uncomment when training, comment when testing
#test_data_in = test_data_in.drop('ClaimAmount', axis=1, inplace=False)
#test_data_out = test_data.loc[:, 'ClaimAmount']

num_boost_rounds = 999

best_clf = xgb.XGBClassifier(objective='binary:logistic', seed=90, num_round=num_boost_rounds, subsample=0.9, scale_pos_weight=5,
                        reg_alpha=1e-05, n_estimators=90, min_child_weight=2, max_depth=20, learning_rate=0.1, gamma=0.5, colsample_bytree=0.9)

best_clf.fit(training_data_in, training_data_out)

print("Starting...")

loaded_clf = pickle.load(open('xgb.sav', 'rb'))
load_predict = loaded_clf.predict(test_data_in)

export = test_data.copy()
export['PredictedCategory'] = load_predict
export.to_csv('category_true_test_xgb.csv')