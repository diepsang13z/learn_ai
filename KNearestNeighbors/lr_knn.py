import numpy as np
import matplotlib.pyplot as plt

from sklearn import neighbors


def main():
    # Init data
    X = np.array([[2, 9, 7, 9, 11, 16, 25, 23, 22, 29, 29, 35, 37, 40, 46]]).T
    y = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]).T

    plt.plot(X, y, 'ro')

    # Use KNN
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)

    # Visualize liear regression
    x0 = np.linspace(1, 46, 10000).reshape(-1, 1)
    y0 = knn.predict(x0)

    plt.plot(x0, y0)

    plt.show()


if __name__ == '__main__':
    main()
