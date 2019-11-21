from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random

def randomforest_kfoldCV(x, y, K, n):
    subset_in = np.array_split(x, K)
    subset_out = np.array_split(y, K)
    incorrect_arr = []
    for i in range(K):
        incorrect = 0
        validation_setin = subset_in[i]
        validation_setout = subset_out[i]
        if (i == 0):
            training_setin = np.concatenate(subset_in[1:])
            training_setout = np.concatenate(subset_out[1:])
        elif (i == K - 1):
            training_setin = np.concatenate(subset_in[0:i])
            training_setout= np.concatenate(subset_out[0:i])
        else:
            training_setin = np.concatenate(subset_in[0:i] + subset_in[i + 1:])
            training_setout = np.concatenate(subset_out[0:i] + subset_out[i + 1:])

        clf = RandomForestClassifier(n_estimators=100, max_features=5, max_leaf_nodes=n)
        clf.fit(training_setin, training_setout)
        predicted = clf.predict(validation_setin)

        validation_setout = np.array(validation_setout)

        for j in range(len(validation_setout)):
            if predicted[j] != validation_setout[j]:
                incorrect = incorrect + 1
        incorrect_arr.append(incorrect / len(validation_setout))
#        print('Acc:', metrics.accuracy_score(validation_setout, predicted))

    cv_error = sum(incorrect_arr) / K
    print('Acc for :', n, "features", cv_error)
    return cv_error


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy

test_csv = pd.read_csv('testset.csv')
data = pd.read_csv('trainingset.csv')

original_data = data.copy()

for i, row in data.iterrows():
    claim_val = 0
    if data.at[i, 'ClaimAmount'] > 0:
        claim_val = 1
    data.at[i, 'ClaimAmount'] = claim_val

train_ratio = 0.75
num_rows = data.shape[0]
train_set_size = int(train_ratio * num_rows)

shuffled_indices = list(range(num_rows))
random.seed(42)
random.shuffle(shuffled_indices)

train_indices = shuffled_indices[:train_set_size]
test_indices = shuffled_indices[train_set_size:]

original_data = original_data.iloc[train_indices, :]
train_data = data.iloc[train_indices, :]
test_data = data.iloc[test_indices, :]

#indexNames = test_data[ test_data['ClaimAmount'] != 1 ].index
#test_data.drop(indexNames, inplace=True)

indexNames = train_data[ train_data['ClaimAmount'] != 1 ].index
train_claims = train_data.drop(indexNames, inplace=False)

indexNames = train_data[ train_data['ClaimAmount'] != 0 ].index
train_no_claims = train_data.drop(indexNames, inplace=False)
no_claims_rows = train_no_claims.shape[0]
no_claims_size = int(0.15 * no_claims_rows)

train_data = train_claims.append(train_no_claims[:no_claims_size])

num_rows = train_data.shape[0]

shuffled_indices = list(range(num_rows))
random.seed(42)
random.shuffle(shuffled_indices)

train_indices = shuffled_indices[:num_rows]
train_data = train_data.iloc[train_indices, :]

print(train_data.head(50).to_string())

training_data_in = train_data.loc[:, ['feature4', 'feature9', 'feature13', 'feature14', 'feature15', 'feature16',
                                      'feature18', 'ClaimAmount']]

test_data_in = test_data.loc[:, ['feature4', 'feature9', 'feature13', 'feature14', 'feature15', 'feature16',
                                      'feature18', 'ClaimAmount']]

true_test = test_csv.loc[:, ['feature4', 'feature9', 'feature13', 'feature14', 'feature15', 'feature16',
                                      'feature18']]

training_data_in = training_data_in.drop('ClaimAmount', axis=1, inplace=False)
training_data_out = train_data.loc[:, 'ClaimAmount']

test_data_in = test_data_in.drop('ClaimAmount', axis=1, inplace=False)
test_data_out = test_data.loc[:, 'ClaimAmount']

#print("Error cv:", sklearn_kNN_kfoldCV(training_data_in, training_data_out, 5, 11))

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
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

clf = RandomForestClassifier(n_estimators=100)
#rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
clf.fit(training_data_in, training_data_out)

#y_true_test = clf.predict(true_test)

y_pred = clf.predict(test_data_in)
y_train = clf.predict(training_data_in)

#print(rf_random.best_params_)

#best_random = rf_random.best_estimator_
#random_accuracy = evaluate(best_random, test_data_in, test_data_out)

#ks = np.arange(2, 15).tolist()
#prediction_array = []

#for i in range(len(ks)):
#    predictions = randomforest_kfoldCV(training_data_in, training_data_out, 10, ks[i])
#    prediction_array.append(predictions)

#plt.plot(ks, prediction_array)
#plt.suptitle("Average cross-prediction error against number of neighbours checked")
#plt.xlabel("k = ")
#plt.ylabel("Average cross-prediction error")
#plt.legend(['y = avg prediction error'], loc='upper left')
#plt.show()


print('Acc:', metrics.accuracy_score(test_data_out, y_pred))
print('2nd Acc:', metrics.accuracy_score(training_data_out, y_train))

#export = test_csv.copy()
#export['PredictedCategory'] = y_true_test
#indexNames = export[ export['PredictedCategory'] != 1 ].index
#export.drop(indexNames, inplace=True)
#export.to_csv('category_test_randforest.csv')
#print(export)