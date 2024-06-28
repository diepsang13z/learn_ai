import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets, neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    digit = datasets.load_digits()

    digit_X = digit.data
    digit_y = digit.target

    k = 10
    training_set_size = int(len(digit_X) * 80 / 100)

    # Shuffle data by index
    rand_index = np.arange(digit_X.shape[0])
    np.random.shuffle(rand_index)

    digit_X = digit_X[rand_index]
    digit_y = digit_y[rand_index]

    # Using KNN
    X_tran, X_test, y_tran, y_test = train_test_split(
        digit_X, digit_y, train_size=training_set_size)

    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_tran, y_tran)

    y_predict = knn.predict(X_test)
    accuracy = accuracy_score(y_predict, y_test)
    print(accuracy)

    # plt.hist(digit_y)
    # plt.show()

    plt.gray()
    plt.imshow(X_test[0].reshape(8, 8))
    print(y_predict[0])
    plt.show()


if __name__ == '__main__':
    main()
