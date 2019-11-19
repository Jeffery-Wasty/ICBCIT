import process_data
import one_hot_encode

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_validate
import warnings
from sklearn.exceptions import DataConversionWarning
import math

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


def kfoldCV(x, y, K, model):
    data_in = np.array_split(x, K)
    data_out = np.array_split(y, K)

    cut_off_x = len(data_in[K - 1])
    cut_off_y = len(data_out[K - 1])

    t_mae = []
    c_mae = []

    for i in range(len(data_in)):
        data_in[i] = data_in[i][0:cut_off_x]

    for i in range(len(data_out)):
        data_out[i] = data_out[i][0:cut_off_y]

    for i in range(K):
        val_x = np.array(data_in[i])
        val_y = np.array(data_out[i])

        if (i == 0):
            train_x = np.concatenate(data_in[1:])
            train_y = np.concatenate(data_out[1:])
        elif (i == K - 1):
            train_x = np.concatenate(data_in[0:i])
            train_y = np.concatenate(data_out[0:i])
        else:
            train_x = np.concatenate(np.concatenate(
                (data_in[0:i], data_in[i + 1:]), axis=0))
            train_y = np.concatenate(np.concatenate(
                (data_out[0:i], data_out[i + 1:]), axis=0))

        model.fit(X=train_x, y=train_y)
        y_pred_val = model.predict(val_x)
        y_pred_train = model.predict(train_x)

        lst = []
        for n in range(len(y_pred_val)):
            lst.append(abs(val_y[n] - y_pred_val[n]))
        c_mae.append(np.mean(lst))

        lst = []
        for i in range(len(y_pred_train)):
            lst.append(abs(train_y[i] - y_pred_train[i]))
        t_mae.append(np.mean(lst))

    train_error = np.mean(t_mae)
    cv_error = np.mean(c_mae)
    return cv_error, train_error


def subset(lambdas, f_type, plot):
    s_data = process_data.split_data_ridge_lasso(data_train, 0.75)

    results_train = []
    results_cv = []

    for a in lambdas:
        if (f_type == 'ridge regression'):
            mod = Ridge(alpha=a, fit_intercept=True)
        else:
            mod = Lasso(alpha=a, fit_intercept=False)

        result = kfoldCV(s_data[5], s_data[1], 5, mod)
        results_train.append(np.mean(result[0]))
        results_cv.append(np.mean(result[1]))
    if plot:
        process_data.error_plot(results_train, results_cv,
                                np.log10(lambdas), f_type)
    return min(results_train), min(results_cv)


lambdas_ridge = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2,
                 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
lambdas_lasso = [
    10 ** -2,
    10 ** -1.75,
    10 ** -1.5,
    10 ** -1.25,
    10 ** -1,
    10 ** -0.75,
    10 ** -0.5,
    10 ** -0.25,
    1,
    10 ** 0.25,
    10 ** 0.5,
    10 ** 0.75,
    10 ** 1,
    10 ** 1.25,
    10 ** 1.5,
    10 ** 1.75,
    10 ** 2,
]

nunique_values = [1, 2, 3, 4, 5, 10, 15, 20, 50]

mae_value = [100000000, 100000000, 100000000, 100000000]
string_value = [0, 0, 0, 0]

for value in nunique_values:

    data_train = one_hot_encode.load("datasets/trainingset.csv", value)

    result = subset(lambdas_ridge, "ridge regression", False)

    if (result[0] < mae_value[0]):
        mae_value[0] = result[0]
        string_value[0] = value
    if (result[1] < mae_value[1]):
        mae_value[1] = result[1]
        string_value[1] = value

    result = subset(lambdas_lasso, "the lasso", False)
    if (result[0] < mae_value[2]):
        mae_value[2] = result[0]
        string_value[2] = value
    if (result[1] < mae_value[3]):
        mae_value[3] = result[1]
        string_value[3] = value

print("The lowest training MAE for ridge was " +
      str(mae_value[0]) + " for nunique_values = " + str(string_value[0]))
print("The lowest validation MAE for ridge was " +
      str(mae_value[1]) + " for nunique_values = " + str(string_value[1]))
# print("The lowest training MAE for lasso was " +
#       str(mae_value[2]) + " for nunique_values = " + string_value[2])
# print("The lowest validation MAE for lasso was " +
#       str(mae_value[3]) + " for nunique_values = " + string_value[3])
