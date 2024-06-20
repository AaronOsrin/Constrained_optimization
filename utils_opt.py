import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_objective_contours(f, x_bounds, y_bounds, paths=None):
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.array([xi, yi])) for xi in x] for yi in y])
    plt.contour(X, Y, Z, levels=50)
    if paths:
        for path, label in paths:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], label=label)
    plt.legend()
    plt.show()

def plot_function_values(*args):
    for values, label in args:
        if values:
            plt.plot(values, label=label)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend()
    plt.show()

def plot_qp_results(path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Quadratic Programming Path")
    plt.show()

def plot_lp_results(path):
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], marker='o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Linear Programming Path")
    plt.show()
