import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


def main():
    # Ramdom data
    low = 1
    high = 50

    np.random.seed(543)
    ra = np.random.randint(low, high, size=14)
    rb = np.random.randint(low, high, size=14)

    # Change row vector to colm vector
    A = np.array([ra]).T
    B = np.array([rb]).T

    # Use algorithm
    lr = linear_model.LinearRegression()
    lr.fit(A, B)

    # y = ax + b
    a = lr.coef_
    b = lr.intercept_

    x = np.array([[low, high]]).T
    y = a * x + b

    # Visualize data
    plt.plot(ra, rb, 'ro')
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    main()
