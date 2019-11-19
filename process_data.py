from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def split_data_ridge_lasso(data, train_ratio):
    num_rows = data.shape[0]
    train_set_size = int(num_rows * train_ratio)

    as_list = list(range(num_rows))

    train_indices = as_list[:train_set_size]
    test_indices = as_list[train_set_size:]

    train_data = data.iloc[train_indices, :]
    test_data = data.iloc[test_indices, :]

    train_features = train_data.drop("ClaimAmount", axis=1, inplace=False)
    train_labels = train_data.loc[:, "ClaimAmount"]

    test_features = test_data.drop("ClaimAmount", axis=1, inplace=False)
    test_labels = test_data.loc[:, "ClaimAmount"]

    ridge_test = StandardScaler(
        with_mean=True, with_std=True).fit_transform(test_features)

    ridge_train = StandardScaler(
        with_mean=True, with_std=True).fit_transform(train_features)

    return train_features, train_labels, test_features, test_labels, ridge_test, ridge_train


def error_plot(results_train, results_cv, function_x, phrase):

    plt.plot(function_x, results_train,
             color="dodgerblue", label="Training error")
    plt.plot(
        function_x,
        results_cv,
        color="springgreen",
        label="Cross-validation estimate of prediction error",
    )
    plt.xlabel("Lambda")
    plt.ylabel("Error")
    plt.title(
        "Training error and cross-validation estimate of prediction error\nas a function of lambda, using "
        + phrase
        + "."
    )
    legend = plt.legend(loc="upper center", fontsize="medium")
    plt.show()
