import matplotlib.pyplot as plt
import numpy as np

from gradient_methods.BFGS import BFGS


def f(x, y):
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2


def f_grad(x, y):
    def f_x(x, y):
        return 400 * x ** 3 + 2 * (1 - 200 * y) * x - 2

    def f_y(x, y):
        return 200 * y - 200 * x ** 2

    return np.array([f_x(x, y), f_y(x, y)])


starting_point_1 = np.array([0.1, 0.1])
bfgs = BFGS(f, f_grad)
res = bfgs.optimize(starting_point_1, max_iter=1000)
print(res)

x, y = list(zip(*bfgs.points))
x, y = np.array(x), np.array(y)
xx, yy = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
z = f(xx, yy)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(xx, yy, z, alpha=0.3)
ax.plot(x, y, f(x, y), 'r-*')
plt.show()
