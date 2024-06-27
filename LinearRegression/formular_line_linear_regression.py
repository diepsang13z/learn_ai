import numpy as np
import matplotlib.pyplot as plt


def main():
    """ Formular line in Linear Regression:
    Line equation:
        [y1, y2, y3, ...] = a [x1, x2, x3, ...] + b [1, 1, 1, ...]
    Element:
        A = [[x1, x2, x3, ...], [1, 1, 1, ...]]
        y = [y1, y2, y3, ...]
    Formular:
        [a, b] = Invese((A.T dot A)) dot A.T dot y
    """
    # Ramdom data
    low = 1
    high = 50

    np.random.seed(543)
    ra = np.random.randint(low, high, size=14)
    rb = np.random.randint(low, high, size=14)

    # Visualize data
    plt.plot(ra, rb, 'ro')

    # Change row vector to colm vector
    A = np.array([ra]).T
    B = np.array([rb]).T

    # Create vector 1
    ones = np.ones((A.shape[0], A.shape[1]), dtype=np.int8)

    # Combine A and 1
    A = np.hstack((A, ones))

    # x = [a, b]
    (a, b) = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(B)

    # y = ax + b
    x = np.array([low, high]).T
    y = x * a + b

    plt.plot(x, y)

    # Test predicting data
    x_test = 12
    y_test = x_test * a + b
    print(y_test)

    plt.show()


if __name__ == '__main__':
    main()
