import operator
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm, lognorm
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.simplefilter(action='ignore', category=FutureWarning)


# read data
train = pd.read_csv("datasets/trainingset.csv")
test = pd.read_csv("datasets/testset.csv")

features = [x for x in train.columns if x not in ['rowIndex', 'ClaimAmount']]
# print(features)

cat_features = [x for x in train.select_dtypes(
    include=['object']).columns if x not in ['rowIndex', 'ClaimAmount']]
num_features = [x for x in train.select_dtypes(
    exclude=['object']).columns if x not in ['rowIndex', 'ClaimAmount']]
# print(cat_features)
# print(num_features)


# train['log_ClaimAmount'] = np.log(train['ClaimAmount'])

# # fit the normal distribution on ln(ClaimAmount)
# (mu, sigma) = norm.fit(train['log_ClaimAmount'])

# # the histogram of the ln(ClaimAmount)
# n, bins, patches = plt.hist(
#     train['log_ClaimAmount'], 60, normed=1, facecolor='green', alpha=0.75)

# # add the fitted line
# y = mlab.normpdf(bins, mu, sigma)
# l = plt.plot(bins, y, 'r--', linewidth=2)

# # plot
# plt.xlabel('Ln(ClaimAmount)')
# plt.ylabel('Probability')
# plt.title('Histogram')
# plt.grid(True)

# plt.show()

ntrain = train.shape[0]
ntest = test.shape[0]
train_test = pd.concat(
    (train[features], test[features])).reset_index(drop=True)
for c in range(len(cat_features)):
    train_test[cat_features[c]] = train_test[cat_features[c]].astype(
        'category').cat.codes

train_x = train_test.iloc[:ntrain, :]
test_x = train_test.iloc[ntrain:, :]

xgdmat = xgb.DMatrix(train_x, train['ClaimAmount'])

params = {'eta': 0.01, 'seed': 0, 'subsample': 0.5, 'colsample_bytree': 0.5,
          'objective': 'reg:squarederror', 'max_depth': 6, 'min_child_weight': 3}

num_rounds = 10
bst = xgb.train(params, xgdmat, num_boost_round=num_rounds)


def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()


ceate_feature_map(features)

importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()

plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')

df

test_xgb = xgb.DMatrix(test_x)
submission = pd.read_csv("datasets/submission.csv")
submission.iloc[:, 1] = np.exp(bst.predict(test_xgb))
submission.to_csv('xgb_starter.sub.csv', index=None)
