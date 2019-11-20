from sklearn import tree
from sklearn import metrics
import pandas as pd
import numpy as np

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

train_indices = shuffled_indices[:train_set_size]
test_indices = shuffled_indices[train_set_size:]

original_data = original_data.iloc[train_indices, :]
train_data = data.iloc[train_indices, :]
test_data = data.iloc[test_indices, :]

training_data_in = train_data.loc[:, ['feature4', 'feature11',
                                      'feature9', 'feature13', 'feature14', 'feature15', 'feature16',
                                      'feature18', 'ClaimAmount']]

test_data_in = test_data.loc[:, ['feature4', 'feature11',
                                  'feature9', 'feature13', 'feature14', 'feature15', 'feature16',
                                  'feature18', 'ClaimAmount']]


training_data_in = training_data_in.drop('ClaimAmount', axis=1, inplace=False)
training_data_out = train_data.loc[:, 'ClaimAmount']

test_data_in = test_data_in.drop('ClaimAmount', axis=1, inplace=False)
test_data_out = test_data.loc[:, 'ClaimAmount']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(training_data_in, training_data_out)

y_pred = clf.predict(test_data_in)
y_train = clf.predict(training_data_in)

print('Acc:', metrics.accuracy_score(test_data_out, y_pred))
print('2nd Acc:', metrics.accuracy_score(training_data_out, y_train))

#export = original_data.copy()
#export['PredictedCategory'] = y_train
#indexNames = export[ export['PredictedCategory'] != 1 ].index
#export.drop(indexNames, inplace=True)
#export.to_csv('category_trained.csv')
#print(export)