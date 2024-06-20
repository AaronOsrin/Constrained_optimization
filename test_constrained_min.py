import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from constrained_min import ConstrainedMinimization
from examples import (quadratic_programming_example, linear_programming_example, 
                      quadratic_programming_constraints, linear_programming_constraints)
from utils_opt import plot_qp_results, plot_lp_results

class TestConstrainedMinimization(unittest.TestCase):
    def setUp(self):
        self.obj_tol = 1e-12
        self.param_tol = 1e-8
        self.max_iter = 1000
        self.x0_qp = [0.1, 0.2, 0.7]  # Provided initial point
        self.x0_lp = [0.1, 0.1]  # Ensuring feasibility

    def test_qp(self):
        f = lambda x: quadratic_programming_example(x)[0]
        grad = lambda x: quadratic_programming_example(x)[1]
        hess = lambda x: quadratic_programming_example(x)[2]
        ineq_constraints, eq_constraints_mat, eq_constraints_rhs = quadratic_programming_constraints()

        optimizer = ConstrainedMinimization(f, grad, hess, ineq_constraints, eq_constraints_mat, eq_constraints_rhs)
        result_x, result_f, success, path = optimizer.minimize(self.x0_qp, self.obj_tol, self.param_tol, self.max_iter)

        self.assertTrue(success, "Constrained optimization for QP failed")
        self.assertTrue(np.all(np.array(result_x) >= 0), "Constraints violated for QP")

        plot_qp_results(path)

    def test_lp(self):
        f = lambda x: linear_programming_example(x)[0]
        grad = lambda x: linear_programming_example(x)[1]
        hess = lambda x: linear_programming_example(x)[2]
        ineq_constraints, eq_constraints_mat, eq_constraints_rhs = linear_programming_constraints()

        optimizer = ConstrainedMinimization(f, grad, hess, ineq_constraints, eq_constraints_mat, eq_constraints_rhs)
        result_x, result_f, success, path = optimizer.minimize(self.x0_lp, self.obj_tol, self.param_tol, self.max_iter)

        self.assertTrue(success, "Constrained optimization for LP failed")
        self.assertTrue(np.all(np.array([result_x[0] <= 2, result_x[1] >= -result_x[0] + 1, result_x[1] <= 1, result_x[1] >= 0])), "Constraints violated for LP")

        plot_lp_results(path)

if __name__ == '__main__':
    unittest.main()
