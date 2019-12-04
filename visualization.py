import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("datasets/trainingset.csv")


def histogram(first):
    x = data.loc[:, first]
    plt.hist(x, 50)
    plt.title("Count of samples for a given claim amount")
    plt.xlabel("Claim amount")
    plt.ylabel("Sample count")
    plt.show()
    return 0


histogram("ClaimAmount")
