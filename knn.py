import process_data

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
import warnings
from sklearn.exceptions import DataConversionWarning
import math

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


def kneigh_kfoldCV(x, y, x_test=0, y_test=0, k=1, test=False):
    cv_val_error = 0
    cv_test_error = 0

    number_in_error = 0

    for i in range(10):
        x_split = np.array_split(x, 10)
        y_split = np.array_split(y, 10)

        val_data_x = x_split[i]
        val_data_y = y_split[i].tolist()

        x_split.pop(i)
        y_split.pop(i)

        train_data_x = np.concatenate(x_split)
        train_data_y = np.concatenate(y_split)

        err_val_cv = 0
        err_test_cv = 0

        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(train_data_x, train_data_y)

        out = neigh.predict(val_data_x)
        for i in range(len(out)):
            if out[i] != val_data_y[i]:
                err_val_cv += 1
        err_val_cv /= len(out)
        cv_val_error += err_val_cv

        if test:
            test_out = neigh.predict(x_test)
            for i in range(len(test_out)):
                if test_out[i] != y_test[i]:
                    err_test_cv += 1
            err_test_cv /= len(test_out)
            cv_test_error += err_test_cv

    cv_val_error /= 10
    cv_test_error /= 10

    return cv_val_error, cv_test_error


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


def subset(lambdas, f_type):
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
    process_data.error_plot(results_train, results_cv,
                            np.log10(lambdas), f_type)


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

data_train = pd.read_csv("datasets/trainingset.csv")

#subset(lambdas_ridge, "ridge regression")
#subset(lambdas_lasso, "the lasso")

k = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]


s_data = process_data.split_data_ridge_lasso(data_train, 0.75)

for i, row in s_data[1].iteritems():
    claim_val = 0
    if row > 0:
        claim_val = 1
    s_data[1][row] = claim_val

results_train = []
results_cv = []

for a in k:
    result = kneigh_kfoldCV(
        s_data[5], s_data[1], x_test=0, y_test=0, k=1, test=False)
    results_train.append(np.mean(result[0]))
    results_cv.append(np.mean(result[1]))
process_data.error_plot(results_train, results_cv,
                        k, "knn")
