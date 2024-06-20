import numpy as np

def quadratic_programming_example(x):
    f = x[0]**2 + x[1]**2 + (x[2] + 1)**2
    grad = np.array([2*x[0], 2*x[1], 2*(x[2] + 1)])
    hess = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    return f, grad, hess

def linear_programming_example(x):
    f = -(x[0] + x[1])
    grad = np.array([-1, -1])
    hess = np.zeros((2, 2))
    return f, grad, hess

def quadratic_programming_constraints():
    ineq_constraints = [(lambda x: -x[0], lambda x: np.array([-1, 0, 0]), lambda x: np.zeros((3, 3))),
                        (lambda x: -x[1], lambda x: np.array([0, -1, 0]), lambda x: np.zeros((3, 3))),
                        (lambda x: -x[2], lambda x: np.array([0, 0, -1]), lambda x: np.zeros((3, 3)))]
    eq_constraints_mat = np.array([[1, 1, 1]])
    eq_constraints_rhs = np.array([1])
    return ineq_constraints, eq_constraints_mat, eq_constraints_rhs

def linear_programming_constraints():
    ineq_constraints = [(lambda x: x[0] - 2, lambda x: np.array([1, 0]), lambda x: np.zeros((2, 2))),
                        (lambda x: -x[1], lambda x: np.array([0, -1]), lambda x: np.zeros((2, 2))),
                        (lambda x: x[1] - (-x[0] + 1), lambda x: np.array([-1, 1]), lambda x: np.zeros((2, 2))),
                        (lambda x: x[1] - 1, lambda x: np.array([0, 1]), lambda x: np.zeros((2, 2)))]
    eq_constraints_mat = np.array([])  # No equality constraints
    eq_constraints_rhs = np.array([])
    return ineq_constraints, eq_constraints_mat, eq_constraints_rhs
