import numpy as np
import matplotlib.pyplot as plt


def main():
    """ Formular line in Linear Regression:
    Line equation:
        [y1, y2, y3, ...] =
            a [x1^2, x2^2, x3^2, ...] + b [x1, x2, x3, ...] + c [1, 1, 1, ...]
    Element:
        A = [[x1^2, x2^2, x3^2, ...], [x1, x2, x3, ...], [1, 1, 1, ...]]
        y = [y1, y2, y3, ...]
    Formular:
        [a, b, c] = Invese((A.T dot A)) dot A.T dot y
    """
    # Ramdom data
    low = 1
    high = 50

    np.random.seed(9573)
    ra = np.random.randint(low, high, size=14)
    rb = np.random.randint(low, high, size=14)

    # Change row vector to colm vector
    A = np.array([ra]).T
    B = np.array([rb]).T

    # Create vector 1
    ones = np.ones((A.shape[0], A.shape[1]), dtype=np.int8)

    # Create A square
    x_square = A ** 2

    # Combine A and 1
    A = np.hstack((x_square, A))
    A = np.hstack((A, ones))

    # x = [a, b, c]
    (a, b, c) = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(B)

    # y = ax^2 + bx + c
    x = np.linspace(low, high, 10000)
    y = a * x ** 2 + b * x + c

    # Test predicting data
    x_test = 12
    y_test = a * x_test ** 2 + b * x_test + c
    print(y_test)

    # Visualize data
    plt.plot(ra, rb, 'ro')
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
