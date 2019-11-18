import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression

# load automobile price data
# ENSURE "AutomobilePrice.csv" is under your working directory
data_train = pd.read_csv("datasets/trainingset.csv")
data_test = pd.read_csv("datasets/testset.csv")

data_train.replace(to_replace="?", value=np.nan, inplace=True)
data_test.replace(to_replace="?", value=np.nan, inplace=True)

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

mae = np.mean(np.abs(test_labels - price_pred))
print("Mean Absolute Error = ", mae)

rmse = np.sqrt(np.mean(np.square(test_labels - price_pred)))
print("Root Mean Squared Error = ", rmse)

total_sum_sq = np.sum(np.square(test_labels - np.mean(test_labels)))
res_sum_sq = np.sum(np.square(test_labels - price_pred))
CoD = 1 - (res_sum_sq / total_sum_sq)
print("Coefficient of Determination = ", CoD)
