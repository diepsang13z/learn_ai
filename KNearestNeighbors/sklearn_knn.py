import numpy as np

from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    iris = datasets.load_iris()

    k = 3
    iris_X = iris.data  # petal length, petal width, sepal length, sepal width, label
    iris_y = iris.target  # labels

    X_tran, X_test, y_tran, y_test = train_test_split(
        iris_X, iris_y, train_size=50)

    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_tran, y_tran)

    y_predict = knn.predict(X_test)
    accuracy = accuracy_score(y_predict, y_test)
    print(accuracy)


if __name__ == '__main__':
    main()
