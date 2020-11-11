from functools import partial

import numpy as np


class NewtonMethod:
    def __init__(self, f, f_grad, f_hess):
        """
        Newton method for search of minimum / maximum of the given function.
        :param f: Optimized function.
        :type f: func
        :param f_grad: Gradient of the optimized function.
        :type f: func
        :param f_hess: Hessian matrix of the optimized function.
        :type f: func
        """
        self.f = f
        self.f_grad = f_grad
        self.f_hess = f_hess
        self.points = []

    def optimize(self, x0, eps=0.0001, max_iter=100):
        """
        Find minimum / maximum of the function using Newton method.
        :param x0: Starting point
        :param eps: Stoping criteria. Minimal difference between last two points.
        :param max_iter: Stoping criteria. Maximal number of iterations of algorithm.
        :return: Minimum / Maximum of the function
        """
        x_star = x0
        i = 0

        while True:
            self.points.append(x_star)
            x_star = self.__step(self.points[-1])
            i += 1

            if np.linalg.norm(x_star - self.points[-1]) < eps or i > max_iter:
                break

        return self.points[-1]

    def __step(self, x):
        """
        Gradient method step.
        """
        d = self.__find_descent_direction_matrix(x=x, f_hess=self.f_hess)
        alpha = self.__find_alpha(x=x, f=self.f, f_grad=self.f_grad, d=d)
        return x - alpha * np.dot(d, self.f_grad(*x))

    @staticmethod
    def __find_descent_direction_matrix(x, f_hess):
        """
        Find descent direction matrix. Using Netwon method it is inversion of hessian matrix.
        """
        return np.linalg.inv(f_hess(*x))

    @staticmethod
    def __find_alpha(x, f, f_grad, d):
        """
        Quadratic interpolation algorithm for optimal step size search.
        """

        def __f_alpha(alpha, x, f, f_grad, d):
            """
            F_k(alpha) defined for optimal step size search.
            """
            return f(*(x - alpha * np.dot(d, f_grad(*x))))

        def __find_initial_points(f, max_s=1000):
            """
            Find initial points for limited optimization rule with quadratic interpolation.
            :param f: F_k(alpha)
            :param max_s: Maximum number of iterations
            :return: Initial points which satisfy a < c < b and f(a) > f(c) < f(b)
            """
            s = 1
            while s < max_s:
                a, b = -s, s
                c = np.random.uniform(a, b)
                if f(c) < f(a) and f(c) < f(b):
                    return a, b, c
                s += 1
            raise ValueError('Cannot find proper a, b, c.')

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
