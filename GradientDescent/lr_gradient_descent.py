import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model


def main():

    def gradient_descent(x_init, learning_rate, iteration):

        def cost(x):
            return .5 / m * np.linalg.norm(A.dot(x) - B, 2) ** 2

        def grad(x):
            return 1 / m * A.T.dot(A.dot(x) - B)

        x_list = [x_init]
        m = A.shape[0]

        for _ in range(iteration):
            x_next = x_list[-1] - learning_rate * grad(x_list[-1])

            if np.linalg.norm(grad(x_next)) / m < 0.3:
                break

            x_list.append(x_next)

        # Calculate cost list
        iter_list = []
        const_list = []
        for i, x in enumerate(x_list):
            iter_list.append(i)
            const_list.append(cost(x))

        return x_list, iter_list, const_list

    # Visualize linear regression usign gradient descent
    plt.figure('Gradient descent for Linear Regression')
    plt.axes(xlim=(-10, 60), ylim=(-1, 20))

    # Init data
    A = np.array([[2, 9, 7, 9, 11, 16, 25, 23, 22, 29, 29, 35, 37, 40, 46]]).T
    B = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]).T

    plt.plot(A, B, 'ro')

    # Line created by linear regression formular
    lr = linear_model.LinearRegression()
    lr.fit(A, B)

    x0_gd = np.linspace(1, 46, 2)
    y0_sklearn = lr.intercept_[0] + lr.coef_[0][0] * x0_gd

    plt.plot(x0_gd, y0_sklearn, color='green')

    # Combine A and 1
    ones = np.ones((A.shape[0], A.shape[1]), dtype=np.int8)
    A = np.hstack((A, ones))

    # Random initial line
    x_init = np.array([[1], [2]])
    y0_init = x_init[0][0] + x_init[1][0] * x0_gd

    plt.plot(x0_gd, y0_init, color='black')

    # Run gradient descent
    x_list, iter_list, cost_list = gradient_descent(
        x_init, learning_rate=0.0001, iteration=100)

    for x in x_list:
        y0_gd = x[0] * x0_gd + x[1]

        plt.plot(x0_gd, y0_gd, color='black')

    plt.show()

    # Visualize cost of gradient descent algorithms
    plt.plot(iter_list, cost_list)
    plt.xlabel('Iteration')
    plt.ylabel('Cost value')
    plt.show()


if __name__ == '__main__':
    main()
