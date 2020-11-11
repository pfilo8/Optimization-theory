import numpy as np

from gradient_methods.NewtonMethod import NewtonMethod


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


if __name__ == '__main__':
    newton_method = NewtonMethod(f, f_grad, f_hess)
    res = newton_method.optimize(np.array([0.1, 0.1]))
    print(res)
