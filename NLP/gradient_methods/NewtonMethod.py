import numpy as np

from .utils import find_alpha


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
        self.i = 0

        while True:
            self.points.append(x_star)
            x_star = self.__step(self.points[-1])
            self.i += 1

            if np.linalg.norm(x_star - self.points[-1]) < eps:
                break
            if self.i > max_iter:
                print('Max iter limit reached.')
                break
        self.points.append(x_star)
        return self.points[-1]

    def __step(self, x):
        """
        Gradient method step.
        """
        d = self.__find_descent_direction_matrix(x=x, f_hess=self.f_hess)
        alpha = find_alpha(x=x, f=self.f, f_grad=self.f_grad, d=d)
        new_x = x - alpha * np.dot(d, self.f_grad(*x))
        # print(f'Gradient norm: {np.linalg.norm(self.f_grad(*new_x))}')
        return new_x

    @staticmethod
    def __find_descent_direction_matrix(x, f_hess):
        """
        Find descent direction matrix. Using Netwon method it is inversion of hessian matrix.
        """
        return np.linalg.inv(f_hess(*x))
