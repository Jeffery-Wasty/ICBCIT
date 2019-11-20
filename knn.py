from sklearn import tree
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def sklearn_kNN_kfoldCV(x, y, K, k):
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

        neighbours = KNeighborsClassifier(n_neighbors=k)
        neighbours.fit(training_setin, training_setout)
        predicted = neighbours.predict(validation_setin)

        validation_setout = np.array(validation_setout)

        for j in range(len(validation_setout)):
            if predicted[j] != validation_setout[j]:
                incorrect = incorrect + 1
        incorrect_arr.append(incorrect / len(validation_setout))
        print('Acc:', metrics.accuracy_score(validation_setout, predicted))

    cv_error = sum(incorrect_arr) / K
    return cv_error

data = pd.read_csv('trainingset.csv')

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

training_data_in = train_data.loc[:, ['feature4', 'feature9', 'feature13', 'feature14', 'feature15', 'feature16',
                                      'feature18', 'ClaimAmount']]

test_data_in = test_data.loc[:, ['feature4', 'feature9', 'feature13', 'feature14', 'feature15', 'feature16',
                                  'feature18', 'ClaimAmount']]


training_data_in = training_data_in.drop('ClaimAmount', axis=1, inplace=False)
training_data_out = train_data.loc[:, 'ClaimAmount']

test_data_in = test_data_in.drop('ClaimAmount', axis=1, inplace=False)
test_data_out = test_data.loc[:, 'ClaimAmount']

print("Error cv:", sklearn_kNN_kfoldCV(training_data_in, training_data_out, 10, 23))
print("Cutoff")
ks = np.arange(1, 23, 2).tolist()
prediction_array = []

for i in range(len(ks)):
    predictions = sklearn_kNN_kfoldCV(training_data_in, training_data_out, 10, ks[i])
    prediction_array.append(predictions)

plt.plot(ks, prediction_array)
plt.suptitle("Average cross-prediction error against number of neighbours checked")
plt.xlabel("k = ")
plt.ylabel("Average cross-prediction error")
plt.legend(['y = avg prediction error'], loc='upper left')
plt.show()


#y_pred = clf.predict(test_data_in)
#y_train = clf.predict(training_data_in)

#print('Acc:', metrics.accuracy_score(test_data_out, y_pred))
#print('2nd Acc:', metrics.accuracy_score(training_data_out, y_train))

#export = original_data.copy()
#export['PredictedCategory'] = y_train
#indexNames = export[ export['PredictedCategory'] != 1 ].index
#export.drop(indexNames, inplace=True)
#export.to_csv('category_trained.csv')
#print(export)