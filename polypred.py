import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def poly_kfoldCV(x, y, K, p):
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

        poly = PolynomialFeatures(degree=p)

        x_transf = poly.fit_transform(training_setin)
        valid_transf = poly.fit_transform(validation_setin)

        lin_reg = LinearRegression()
        lin_reg.fit(x_transf, training_setout)
        y_pred_val = lin_reg.predict(valid_transf)
        y_pred_train = lin_reg.predict(x_transf)

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


data = pd.read_csv("datasets/trainingset.csv")

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


pt3_train_arr = []
pt3_valid_arr = []
for i in range(3):
    kfold_result = poly_kfoldCV(training_data_in, training_data_out, 10, i)
    pt3_train_arr.append(kfold_result[1])
    pt3_valid_arr.append(kfold_result[0])
    print(i, "training: ", kfold_result[1], "cv: ", kfold_result[0])

plt.plot(range(1, 10), pt3_train_arr)
plt.plot(range(1, 10), pt3_valid_arr)
plt.suptitle("Learning curve plot for feature # vs. mae")
plt.xlabel("Number of features = ")
plt.ylabel("Error")
plt.legend(['y = train_error', 'y = cv_error'], loc='upper left')
plt.show()


#lst = []
#for i in range(len(price_pred)):
#    tmp = abs(test_data_out.values[i] - price_pred[i])
#    lst.append(tmp)
#mae = np.mean(lst)

#print('Mean Absolute Error = ', mae)

#r2 = r2_score(list(test_data_out), price_pred)
#print("r2 score: ", r2)
