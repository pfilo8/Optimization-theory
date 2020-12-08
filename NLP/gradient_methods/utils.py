from functools import partial

import numpy as np


def find_alpha(x, f, f_grad, d):
    """
    Quadratic interpolation algorithm for optimal step size search.
    """

    def __f_alpha(alpha, x, f, f_grad, d):
        """
        F_k(alpha) defined for optimal step size search.
        """
        return f(*(x - alpha * np.dot(d, f_grad(*x))))

    def __find_initial_points(f, delta=0.01, max_iter=20):
        i = 1
        lambda_0 = 0
        lambdas = [lambda_0 - 2 * delta, lambda_0, lambda_0 + 2 * delta]
        f_values = [f(lambdas[0]), f(lambdas[1]), f(lambdas[2])]

        if f_values[1] < min(f_values[0], f_values[2]):
            return lambdas[0], lambdas[1], lambdas[2]

        while i < max_iter:

            lambda_r = lambdas[-1] + 2 ** (i - 1) * delta
            f_value_r = f(lambda_r)

            lambdas.append(lambda_r)
            f_values.append(f_value_r)

            if f_values[-1] > f_values[-2]:
                break

            i += 1

        return lambdas[-3], lambdas[-2], lambdas[-1]

    f_k = partial(__f_alpha, x=x, f=f, f_grad=f_grad, d=d)
    a, b, c = __find_initial_points(f_k)
    eps = 0.01

    x_star = b + eps
    x_bar = c

    while np.linalg.norm(x_star - x_bar) > eps:
        x_star = x_bar
        x_bar_num = ((b ** 2 - c ** 2) * f_k(alpha=a) + (c ** 2 - a ** 2) * f_k(alpha=b) + (a ** 2 - b ** 2) * f_k(
            alpha=c))
        x_bar_den = (2 * ((b - c) * f_k(alpha=a) + (c - a) * f_k(alpha=b) + (a - b) * f_k(alpha=c)))
        x_bar = x_bar_num / x_bar_den

        if x_bar > c:
            d = x_bar
        else:
            d = c
            c = x_bar

        if f_k(c) < f_k(d):
            b = d
        else:
            a = c
            c = d

    x_star = x_bar
    return x_star
