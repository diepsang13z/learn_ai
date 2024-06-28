import numpy as np
import math
import operator

from sklearn import datasets


def calculate_distance(p1, p2):
    dimension = len(p1)
    distance = 0

    for i in range(dimension):
        distance += (p1[i] - p2[i]) ** 2

    return math.sqrt(distance)


def get_k_neighbors(training_X, label_y, point, k):
    distances = []
    neighbors = []

    # Calculate distances with labels
    for i, p in enumerate(training_X):
        distance = calculate_distance(p, point)
        distances.append((distance, label_y[i]))

    # Sort by distance
    distances.sort(key=operator.itemgetter(0))

    # Find neighbors
    for i in range(k):
        neighbors.append(distances[i][1])

    return neighbors


def highest_votes(labels, k):
    labels_count = [0] * k

    for label in labels:
        labels_count[label] += 1

    max_count = max(labels_count)

    return labels_count.index(max_count)


def predict(training_X, label_y, point, k):
    neighbors_labels = get_k_neighbors(training_X, label_y, point, k)
    return highest_votes(neighbors_labels, k)


def accuracy_score(predicts, ground_truths):
    total = len(predicts)
    correct_count = 0
    for i in range(total):
        if predicts[i] == ground_truths[i]:
            correct_count += 1
    return correct_count / total


def main():
    iris = datasets.load_iris()

    k = 3
    iris_X = iris.data  # petal length, petal width, sepal length, sepal width, label
    iris_y = iris.target  # labels

    # Shuffle data by index
    rand_index = np.arange(iris_X.shape[0])
    np.random.shuffle(rand_index)

    iris_X = iris_X[rand_index]
    iris_y = iris_y[rand_index]

    # Spliting 100 data for training and 50 data for testing
    X_tran = iris_X[:100]
    X_test = iris_X[100:]

    y_tran = iris_y[:100]
    y_test = iris_y[100:]

    y_predict = []
    for p in X_test:
        label = predict(X_tran, y_tran, p, k)
        y_predict.append(label)

    accuracy = accuracy_score(y_predict, y_test)
    print(accuracy)


if __name__ == '__main__':
    main()
