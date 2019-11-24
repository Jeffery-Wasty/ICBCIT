import pandas as pd
import pandasql as ps
import math
import re
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from sklearn.metrics import r2_score


def forest_kfoldCV(x, y, K, n):

    rf = RandomForestRegressor(n_estimators=30, random_state=42, max_depth=1, min_samples_split=10)
    subset_in = np.array_split(x, K)
    subset_out = np.array_split(y, K)
    cut_off_x = len(subset_in[K - 1])
    cut_off_y = len(subset_out[K - 1])
    t_mae = []
    c_mae = []
    for i in range(len(subset_in)):
        subset_in[i] = subset_in[i][0:cut_off_x]
    for i in range(len(subset_out)):
        subset_out[i] = subset_out[i][0:cut_off_y]
    for i in range(K):
        validation_setin = np.array(subset_in[i])
        validation_setout = np.array(subset_out[i])
        if (i == 0):
            training_setin = np.concatenate(subset_in[1:])
            training_setout = np.concatenate(subset_out[1:])
        elif (i == K - 1):
            training_setin = np.concatenate(subset_in[0:i])
            training_setout= np.concatenate(subset_out[0:i])
        else:
            training_setin = np.concatenate(subset_in[0:i] + subset_in[i + 1:])
            training_setout = np.concatenate(subset_out[0:i] + subset_out[i + 1:])

        rf.fit(training_setin, training_setout)
#       print(pd.Series(rf.feature_importances_, index=training_data_in.columns))

        y_pred_val = rf.predict(validation_setin)
        y_pred_train = rf.predict(training_setin)

        lst = []
        for n in range(len(y_pred_val)):
            tmp = abs(validation_setout[n] - y_pred_val[n])
            lst.append(tmp)
        c_mae.append(np.mean(lst))
        print("cv", np.mean(lst))
        lst = []
        for i in range(len(y_pred_train)):
            tmp = abs(training_setout[i] - y_pred_train[i])
            lst.append(tmp)
        t_mae.append(np.mean(lst))

        print("training", np.mean(lst))

    train_error = np.mean(t_mae)
    cv_error = np.mean(c_mae)
    return cv_error, train_error


data = pd.read_csv("datasets/trainingset.csv")
testSet = pd.read_csv("category_test_randforest.csv")
originalTest = testSet.copy()
subset = data.dropna(axis=0, how='any', inplace=False)
data = data[data['ClaimAmount'] != 0]
#print(data.head(100).to_string())
train_ratio = 0.5
num_rows = subset.shape[0]
train_set_size = int(train_ratio * num_rows)

data_in = subset.drop('ClaimAmount', axis=1, inplace=False)
data_out = subset.loc[:, 'ClaimAmount']

training_data_in = data_in[:train_set_size]
training_data_out = data_out[:train_set_size]

test_data_in = data_in[train_set_size:]
test_data_out = data_out[train_set_size:]

temp_test_in = test_data_in

drop_features = ['feature3', 'feature4', 'feature5', 'feature7',
                                      'feature9', 'feature13', 'feature14',
                                      'feature18']

training_data_in = training_data_in.drop(drop_features, axis=1)
testSet = testSet.drop(drop_features, axis=1)
testSet = testSet.drop(['PredictedCategory'], axis=1)

test_data_in = test_data_in.drop(drop_features, axis=1)


pt3_train_arr = []
pt3_valid_arr = []
# for i in range(1):
#    kfold_result = forest_kfoldCV(training_data_in, training_data_out, 5, 0)
#    pt3_train_arr.append(kfold_result[1])
#    pt3_valid_arr.append(kfold_result[0])
#    print(i, "training: ", kfold_result[1], "cv: ", kfold_result[0])

#print(training_data_in.head(50).to_string())
#print(forest_kfoldCV(training_data_in, training_data_out, 5, 0))

#plt.plot(range(2, 12), pt3_train_arr)
#plt.plot(range(2, 12), pt3_valid_arr)
#plt.suptitle("Min samples to split vs. mae")
#plt.xlabel("Min samples to split ")
#plt.ylabel("Error")
#plt.legend(['y = train_error', 'y = cv_error'], loc='upper left')
#plt.show()

# print(y_pred_val[0])
# print(list(test_data_out)[0])

tuned_parameters = {'n_estimators': [10, 20, 30], 'max_depth': [None, 1, 2, 3], 'min_samples_split': [2, 3, 4]}
clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=5,
                   n_jobs=-1, verbose=1)
clf.fit(training_data_in, training_data_out)
y_pred_val = clf.predict(test_data_in)

# lst = []
# for n in range(len(y_pred_val)):
#     tmp = abs(list(test_data_out)[n] - y_pred_val[n])
#     lst.append(tmp)
# mae = np.mean(lst)
# print("mae for 75/25 split, no kfold: ", mae)
# r2 = r2_score(list(test_data_out), y_pred_val)
# print("r2 for 75/25 split, no kfold: ", r2)
print(len(test_data_in))
print(len(y_pred_val))
print(len(originalTest))

export = test_data_in[0:30000]
print(len(export))
export.reset_index(drop=True)
export['ClaimAmount'] = y_pred_val[0:30000]
for i in range(len(export)):
    if originalTest.iloc[i]['PredictedCategory'] == 0:
        export.iloc[i, export.columns.get_loc('ClaimAmount')] = 0
#export.to_csv('random_forest_category.csv', index=False)

lst = []
print(len(test_data_out))
print(len(export))
for n in range(len(export)):
    tmp = abs(list(test_data_out)[n] - export.iloc[n]['ClaimAmount'])
    lst.append(tmp)
mae = np.mean(lst)
print(mae)
#print(export)