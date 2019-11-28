import xgboost as xgb
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import numpy as np
# Must install graphviz

warnings.simplefilter(action='ignore', category=FutureWarning)
TEST_RATIO = 0.25
RANDOM_SEED = 123
# max_depths = [1, 2, 5, 10, 15, 20, 25, 50, 100]
# alphas = [1, 2, 5, 10, 15, 20, 25, 50, 100]
# colsamples = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# learning_rates = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1]


# Cross-validation
def cv(data_dmatrix):
    params = {"objective": "reg:squarederror", 'colsample_bytree': 0.5, 'learning_rate': 0.01,
              'max_depth': 20, 'alpha': 10}
    cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=10,
                        num_boost_round=50, metrics={'mae'}, early_stopping_rounds=10, as_pandas=True, seed=RANDOM_SEED)

    train_mae = cv_results['train-mae-mean'].values[0]
    print("Training Mean Absolute Error (cv): \t%f" % (train_mae))

    test_mae = cv_results['test-mae-mean'].values[0]
    print("Validation Mean Absolute Error (cv): \t%f" % (test_mae))


# Test using parameters interpreted from cv()
def test(xg_reg, X_train, X_test, y_train, y_test):
    xg_reg.fit(X_train, y_train)
    preds = xg_reg.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print("Test Mean Absolute Error:\t%f" % (mae))

    # inaccurate = 0

    # print(y_test.index)
    # print(preds)

    # for i in range(len(preds)):
    #     if preds[i] != y_test.index[i]:
    #         inaccurate += 1

    # print("Accuracy:\t%f" % (1 - (inaccurate/len(preds))))


# Visualize tree
def plot_tree(x, y):
    params = {"objective": "reg:squarederror", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
              'max_depth': 5, 'alpha': 10}
    xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

    xgb.plot_tree(xg_reg, num_trees=0)
    plt.rcParams['figure.figsize'] = [x, y]
    plt.show()


# Feature Importance
def feature_importance(x, y, xg_reg):
    xgb.plot_importance(xg_reg)
    plt.rcParams['figure.figsize'] = [x, y]
    plt.show()


data = pd.read_csv("datasets/trainingset.csv")
X, y = data.iloc[:, :-1], data.iloc[:, -1]
data_dmatrix = xgb.DMatrix(data=X, label=y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_RATIO, random_state=RANDOM_SEED)

# Find Train and CV MAE using params above
# cv(data_dmatrix)


# Test using found params
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.5, learning_rate=0.01,
                          max_depth=20, alpha=10, n_estimators=10)

test(xg_reg, X_train, X_test, y_train, y_test)


# Visualizations
# plot_tree(100, 40)
feature_importance(15, 15, xg_reg)
