import pandas as pd
import numpy as np

# load and encode categorical data
# Inputs:
#  filepath: path to the csv file
# Outputs:
#  data: a DataFrame object containing encoded data


def load(filepath, nunique):
    df = pd.read_csv(filepath)
    data = pd.DataFrame(index=df.index)

    for i in range(0, df.shape[1]):
        col = df.iloc[:, [i]]
        if col.values.dtype == np.object or col.nunique().values < nunique:
            dummies = pd.get_dummies(
                col, columns=col.columns, prefix=col.columns, dtype=np.int64
            )
            data = data.join(dummies)
        else:
            data = data.join(col)

    return data
