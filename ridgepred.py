import pandas as pd
import math
import re
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def ridge_kfoldCV(x, y, K, alph):
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
    type = Ridge(alpha=alph, fit_intercept=True)
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
            training_setin = np.concatenate(np.concatenate((subset_in[0:i], subset_in[i + 1:]), axis=0))
            training_setout = np.concatenate(np.concatenate((subset_out[0:i], subset_out[i + 1:]), axis=0))
        type.fit(X=training_setin, y=training_setout)
        y_pred_val = type.predict(validation_setin)
        y_pred_train = type.predict(training_setin)
        lst = []
        for n in range(len(y_pred_val)):
            tmp = abs(validation_setout[n] - y_pred_val[n])
            lst.append(tmp)
        c_mae.append(np.mean(lst))

        lst = []
        for i in range(len(y_pred_train)):
            tmp = abs(training_setout[i] - y_pred_train[i])
            lst.append(tmp)
        t_mae.append(np.mean(lst))

    train_error = np.mean(t_mae)
    cv_error = np.mean(c_mae)
    return cv_error, train_error

data = pd.read_csv("trainingset.csv")

train_ratio = 0.75
num_rows = data.shape[0]
train_set_size = int(train_ratio * num_rows)

shuffled_indices = list(range(num_rows))

train_indices = shuffled_indices[:train_set_size]
test_indices = shuffled_indices[train_set_size:]

train_data = data.iloc[train_indices, :]
test_data = data.iloc[test_indices, :]

#training_data_in = train_data.loc[:, ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7',
#                                      'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14',
#                                      'feature15', 'feature16', 'feature17', 'feature18']]

#test_data_in = test_data.loc[:, ['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7',
#                                      'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14',
#                                      'feature15', 'feature16', 'feature17', 'feature18']]

training_data_in = train_data
test_data_in = test_data

training_data_in = training_data_in.drop('ClaimAmount', axis=1, inplace=False)
training_data_out = train_data.loc[:, 'ClaimAmount']

test_data_in = test_data_in.drop('ClaimAmount', axis=1, inplace=False)
test_data_out = test_data.loc[:, 'ClaimAmount']

ridge_test = StandardScaler(with_mean=True, with_std=True).fit_transform(test_data_in)
ridge_train = StandardScaler(with_mean=True, with_std=True).fit_transform(training_data_in)

alpha = [10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8, 10**9, 10**10]
alpha_plot = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

c_mae_array = []
t_mae_array = []
for i in range(len(alpha)):
    mae_vals = ridge_kfoldCV(ridge_train, training_data_out, 5, alpha[i])
    c_mae_array.append(mae_vals[0])
    t_mae_array.append(mae_vals[1])


plt.plot(alpha_plot, c_mae_array)
plt.plot(alpha_plot, t_mae_array)
plt.suptitle("Ridge regression on 5 fold-cross validation w/ alphas $10^{-3}$ to $10^{10}$")
plt.xlabel("λ = ")
plt.ylabel("Errors")
plt.legend(['y = cv_error', 'y = train_error'], loc='upper left')
plt.show()

chosenLamb = Ridge(alpha=10**5)
chosenLamb.fit(ridge_train, training_data_out)
coeffs = chosenLamb.coef_
index = np.argpartition(np.abs(coeffs), -17)[-17:]
print("\nTop ten features for Ridge\n: ")
for i in index:
    print(training_data_in.columns[i])
    print(coeffs[i])

print(training_data_in.columns)