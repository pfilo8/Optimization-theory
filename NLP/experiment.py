import time

import numpy as np

from gradient_methods.NewtonMethod import NewtonMethod
from gradient_methods.BFGS import BFGS


def f(x, y):
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


def f_grad(x, y):
    def f_x(x, y):
        return 400 * x ** 3 + 2 * (1 - 200 * y) * x - 2

    def f_y(x, y):
        return 200 * y - 200 * x ** 2

    return np.array([f_x(x, y), f_y(x, y)])


def f_hess(x, y):
    def f_xx(x, y):
        return 1200 * x ** 2 + 2 * (1 - 200 * y)

    def f_xy(x, y):
        return -400 * x

    def f_yy(x, y):
        return 200

    return np.array([[f_xx(x, y), f_xy(x, y)], [f_xy(x, y), f_yy(x, y)]])


def g(x, y, z, t, u):
    return x ** 4 + y ** 4 + z ** 4 + t ** 4 + u ** 4


def g_grad(x, y, z, t, u):
    return np.array([4 * x ** 3, 4 * y ** 3, 4 * z ** 3, 4 * t ** 3, 4 * u ** 3])


def g_hess(x, y, z, t, u):
    return np.array([
        [12 * x ** 2, 0, 0, 0, 0],
        [0, 12 * y ** 2, 0, 0, 0],
        [0, 0, 12 * z ** 2, 0, 0],
        [0, 0, 0, 12 * t ** 2, 0],
        [0, 0, 0, 0, 12 * u ** 2]
    ])


def execute_experiment(method, starting_point, method_kwargs, optimize_kwargs={}):
    method_initialized = method(**method_kwargs)
    start_time = time.time()
    res = method_initialized.optimize(starting_point, **optimize_kwargs)
    print(f'Method: {method}.')
    print(f'Starting point: {starting_point}.')
    print(f'Result: {res}. Nb of iterations: {method_initialized.i}.')
    print(f'Execution time: {time.time() - start_time}.')
    print('-' * 60)


if __name__ == '__main__':
    starting_point_1 = np.array([0.1, 0.1])
    starting_point_2 = np.array([-0.5, 0.7])

    print('Function f')
    print('*' * 100)
    execute_experiment(
        NewtonMethod,
        starting_point_1,
        method_kwargs={
            "f": f,
            "f_grad": f_grad,
            "f_hess": f_hess
        })

    execute_experiment(
        NewtonMethod,
        starting_point_2,
        method_kwargs={
            "f": f,
            "f_grad": f_grad,
            "f_hess": f_hess
        })

    execute_experiment(
        BFGS,
        starting_point_1,
        method_kwargs={
            "f": f,
            "f_grad": f_grad
        },
        optimize_kwargs={
            "max_iter": 1000
        })

    execute_experiment(
        BFGS,
        starting_point_2,
        method_kwargs={
            "f": f,
            "f_grad": f_grad
        },
        optimize_kwargs={
            "max_iter": 1000
        })

    print('Function g')
    print('*' * 100)

    starting_point_1 = np.array([0.1, 0.1, 0.2, -0.7, 1.2])
    starting_point_2 = np.array([-0.5, 0.7, 2.0, 1.0, -7.0])

    execute_experiment(
        NewtonMethod,
        starting_point_1,
        method_kwargs={
            "f": g,
            "f_grad": g_grad,
            "f_hess": g_hess
        })

    execute_experiment(
        NewtonMethod,
        starting_point_2,
        method_kwargs={
            "f": g,
            "f_grad": g_grad,
            "f_hess": g_hess
        })

    execute_experiment(
        BFGS,
        starting_point_1,
        method_kwargs={
            "f": g,
            "f_grad": g_grad
        },
        optimize_kwargs={
            "max_iter": 1000
        })

    execute_experiment(
        BFGS,
        starting_point_2,
        method_kwargs={
            "f": g,
            "f_grad": g_grad
        },
        optimize_kwargs={
            "max_iter": 1000
        })
