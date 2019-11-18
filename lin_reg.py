import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression

# load automobile price data
# ENSURE "AutomobilePrice.csv" is under your working directory
data_train = pd.read_csv("datasets/trainingset.csv")
data_test = pd.read_csv("datasets/testset.csv")
# data should contain 205 rows and 26 columns
print("\n\ndata_train.info():")
print(data_train.info())
print("\n\ndata_test.info():")
print(data_test.info())

# remove rows that contain missing values
# first replace '?' in the data with special values 'nan'
# 'nan' (not a number) is defined in the numpy module
data_train.replace(to_replace="?", value=np.nan, inplace=True)
data_test.replace(to_replace="?", value=np.nan, inplace=True)
# remove any rows that contain 'nan'
data_train.dropna(axis=0, how="any", inplace=True)
data_test.dropna(axis=0, how="any", inplace=True)

print(len(data_train), "training samples + ", len(data_test), "test samples")

# prepare training features and training labels
# features: all columns except 'price'
# labels: 'price' column
train_features = data_train.drop("ClaimAmount", axis=1, inplace=False)
train_labels = data_train.loc[:, "ClaimAmount"]

# prepare test features and test labels
test_features = data_test
test_labels = train_labels[:30000]

# train a linear regression model using training data
lin_reg = LinearRegression()
# Note: you can ingore warning messages regarding gelsd driver
w = lin_reg.fit(train_features, train_labels)

# predict new prices on test data
price_pred = lin_reg.predict(test_features)

# compute mean absolute error
# formula: MAE = mean of | y_i - y_i_pred|
# where y_i is the i-th element of test_labels
#       y_i_pred is the i-th element of the price_pred
# Ref: https://en.wikipedia.org/wiki/Mean_absolute_error
mae = np.mean(np.abs(test_labels - price_pred))
print("Mean Absolute Error = ", mae)

# compute root means square error
# formula: RMSE = square root of mean of (y_i - y_i_pred)
# Ref: https://en.wikipedia.org/wiki/Root-mean-square_deviation
rmse = np.sqrt(np.mean(np.square(test_labels - price_pred)))
print("Root Mean Squared Error = ", rmse)

# compute coefficient of determination (aka R squared)
# formula: CoD = 1 - SSres/SStot, where
# SSres = sum of squares of ( y_i - y_i_pred )
# SStot = sum of squares of ( y_i - mean of y_i )
# Ref: https://en.wikipedia.org/wiki/Coefficient_of_determination
total_sum_sq = np.sum(np.square(test_labels - np.mean(test_labels)))
res_sum_sq = np.sum(np.square(test_labels - price_pred))
CoD = 1 - (res_sum_sq / total_sum_sq)
print("Coefficient of Determination = ", CoD)
