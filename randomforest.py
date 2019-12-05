import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Loading csvs
data = pd.read_csv("trainingset.csv")
testSet = pd.read_csv("category_test_xgb.csv")
originalTest = testSet.copy()
subset = data.dropna(axis=0, how='any', inplace=False)
train_ratio = 0.75
num_rows = subset.shape[0]
train_set_size = int(train_ratio * num_rows)

# in/out
data_in = subset.drop('ClaimAmount', axis=1, inplace=False)
data_out = subset.loc[:, 'ClaimAmount']

training_data_in = data_in[:train_set_size]
training_data_out = data_out[:train_set_size]

test_data_in = data_in[train_set_size:]
test_data_out = data_out[train_set_size:]

drop_features = ['feature4', 'feature9', 'feature13', 'feature14', 'feature15',
                                      'feature18']

training_data_in = training_data_in.drop(drop_features, axis=1)

testSet = testSet.drop(drop_features, axis=1)
testSet = testSet.drop(['PredictedCategory'], axis=1)

test_data_in = test_data_in.drop(drop_features, axis=1)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 500, num = 10)]
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
#clf = GridSearchCV(RandomForestRegressor(), random_grid, cv=5,
#                   n_jobs=-1, verbose=1)
clf = RandomForestRegressor(min_samples_split=3, n_estimators=30)
clf.fit(training_data_in, training_data_out)
#print("Best Hyper Parameters:\n", clf.best_params_)
y_pred_val = clf.predict(test_data_in)


export = test_data_in.copy()
export.reset_index(drop=True)
export['ClaimAmount'] = y_pred_val
for i in range(len(export)):
    if originalTest.iloc[i]['PredictedCategory'] == 0:
        export.iloc[i, export.columns.get_loc('ClaimAmount')] = 0
#export.to_csv('category_test_randforest_randforest.csv')

lst = []
for n in range(len(export)):
    tmp = abs(list(test_data_out)[n] - export.iloc[n]['ClaimAmount'])
    lst.append(tmp)
mae = np.mean(lst)
print(mae)