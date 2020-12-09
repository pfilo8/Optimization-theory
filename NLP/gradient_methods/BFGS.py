import numpy as np

from .utils import find_alpha


class BFGS:
    def __init__(self, f, f_grad):
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
        D = np.eye(len(x0))
        g = self.f_grad(*x_star)

        self.i = 0

        while True:
            self.points.append(x_star)
            x_star, D, g = self.__step(self.points[-1], D, g)
            self.i += 1

            if np.linalg.norm(x_star - self.points[-1]) < eps:
                break
            if self.i > max_iter:
                print('Max iter limit reached.')
                break
        self.points.append(x_star)
        return self.points[-1]

    def __step(self, x, D, g):
        """
        Gradient method step.
        """
        d = -np.dot(D, g)
        alpha = find_alpha(x=x, f=self.f, f_grad=self.f_grad, d=d)
        p = d * alpha
        x = x + p
        q = g
        g = self.f_grad(*x)
        q = g - q
        C = self.__find_c(p, q, D)
        D = D + C
        # print(f'Gradient norm: {np.linalg.norm(g)}')
        return x, D, g

    @staticmethod
    def __find_c(p, q, D):
        """
        Find correction matrix for BFGS method.
        """
        p = p.reshape(-1, 1)
        q = q.reshape(-1, 1)
        r = q.T @ D @ q
        v = p / (p.T @ q) - (D @ q) / r
        return ((p @ p.T) / (p.T @ q)) - ((D @ q @ q.T @ D) / (q.T @ D @ q)) + r * (v @ v.T)
