import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from sklearn import linear_model
from typing import Callable, List, Tuple


def check_grad(
    x: np.ndarray,
    cost: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Checks the correctness of the gradient function using numerical approximation.

    This function compares the analytical gradient with the numerical gradient approximation
    to ensure that the gradient function is correctly implemented. Else print a WARNING alert.

    Formular:
        f(x) = approximately (f(x - epsilon) - f(x + epsilon)) / 2 * epsilon

    Args:
        x (np.ndarray): Point coordinates at which to evaluate the gradients.
        cost (Callable[[np.ndarray], float]): Function that computes the cost value.
        grad (Callable[[np.ndarray], np.ndarray]): Function that computes the analytical gradient of the cost function.
    """
    eps = 1e-4  # 0.001
    g_approximately = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()

        x1[i] += eps
        x2[i] -= eps

        g_approximately[i] = \
            (cost(x1) - cost(x2)) / (2 * eps)

    g_grap = grad(x)
    if np.linalg.norm(g_grap - g_approximately) > 1e-7:
        print('WARNING: CHECK GRADIENT FUNCTION!')


def gradient_descent(
    x_init: np.ndarray,
    point_coordinates: Tuple[np.ndarray, np.ndarray],
    learning_rate: float,
    iteration: int,
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """ Performs gradient descent to minimize the cost function for linear regression.

    Formular:
        x0 -> x0 - learning_rate * f'(x0)

    Args:
        x_init (np.ndarray): Initial guess for the parameters (x0).
        point_coordinates (Tuple[np.ndarray, np.ndarray]):
            A tuple containing two arrays: the first array consists of the input features (x1, x2, x3, ...),
            and the second array consists of the corresponding target values (y1, y2, y3, ...).
        learning_rate (float): The step size for each iteration of gradient descent.
        iteration (int): The maximum number of iterations to perform.

    Returns:
        Tuple[List[np.ndarray], List[int], List[float]]:
            - List of np.ndarray: The parameter values after each iteration.
            - List of int: The iteration numbers.
            - List of float: The cost function values after each iteration.
    """

    def cost(x: np.ndarray) -> float:
        """Calculates the cost function for linear regression.

        Formular: m/2 * |Ax - b|^2

        Args:
            x (np.ndarray): Current parameter values.

        Returns:
            float: The value of the cost function.
        """
        return .5 / n_point * np.linalg.norm(A.dot(x) - B, 2) ** 2

    def grad(x: np.ndarray) -> np.ndarray:
        """Calculates the gradient of the cost function.

        Formular: 1/m * A.T |Ax - b|

        Args:
            x (np.ndarray): Current parameter values.

        Returns:
            np.ndarray: The gradient of the cost function.
        """
        return 1 / n_point * A.T.dot(A.dot(x) - B)

    x_list = [x_init]
    A, B = point_coordinates
    n_point = A.shape[0]

    check_grad(x_init, cost, grad)

    for _ in range(iteration):
        x_next = x_list[-1] - learning_rate * grad(x_list[-1])

        if np.linalg.norm(grad(x_next)) / n_point < 1e-5:
            break

        x_list.append(x_next)

    # Calculate cost list
    iter_list = []
    const_list = []
    for i, x in enumerate(x_list):
        iter_list.append(i)
        const_list.append(cost(x))

    return x_list, iter_list, const_list


def main():
    # Visualize linear regression usign gradient descent
    fig1 = plt.figure('Gradient descent for Linear Regression')
    ax = plt.axes(xlim=(-10, 60), ylim=(-1, 20))

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
    x_init = np.array([[1.], [2.]])
    y0_init = x_init[0][0] * x0_gd + x_init[1][0]

    plt.plot(x0_gd, y0_init, color='black')

    # Run gradient descent
    x_list, iter_list, cost_list = gradient_descent(
        x_init, (A, B), learning_rate=0.0001, iteration=100)

    # Make animation for line in gradient descent
    line, = ax.plot([], [], color='blue')

    def update(i):
        x = x_list[i]
        y0_gd = x[0][0] * x0_gd + x[1][0]
        line.set_data(x0_gd, y0_gd)
        return line,

    iters = np.arange(1, len(x_list), 1)
    line_animation = animation.FuncAnimation(
        fig1, update, iters, interval=50, blit=True)

    # Show all created line by gradient descent
    for x in x_list:
        y0_gd = x[0][0] * x0_gd + x[1][0]
        plt.plot(x0_gd, y0_gd, color='black', alpha=0.2)

    # Legend for plot
    plt.title('Gradient Descent Animation')
    plt.legend((
        'Points',
        'Solution by formular',
        'Random line'
    ), loc=(0.60, 0.01))
    _ = plt.gca().get_legend().get_texts()

    plt.show()

    # Visualize cost of gradient descent algorithms
    plt.plot(iter_list, cost_list)
    plt.xlabel('Iteration')
    plt.ylabel('Cost value')
    plt.show()


if __name__ == '__main__':
    main()
