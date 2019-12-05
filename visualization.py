import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("datasets/presentation.csv")
print(data)


def line_graph(first):
    y = data.loc[:, first]
    x = range(22)

    plt.plot(x, y)
    plt.title(first + " over time")
    plt.xlabel("Submission number")
    plt.ylabel(first)
    plt.show()
    return 0


line_graph("MAE")
# line_graph("MAEScore")
# line_graph("F1")
line_graph("F1Score")
