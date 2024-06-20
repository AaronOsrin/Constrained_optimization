import numpy as np

def phi(x, ineq_constraints):
    penalty = -np.sum([np.log(-g(x)) for g, _, _ in ineq_constraints if g(x) < 0])
    if penalty == np.inf:
        return np.inf
    return penalty

def grad_phi(x, ineq_constraints):
    grad = np.zeros_like(x, dtype=np.float64)
    for g, grad_g, _ in ineq_constraints:
        if g(x) >= 0:
            return np.inf * np.ones_like(x, dtype=np.float64)
        grad += grad_g(x) / -g(x)
    return grad

def hess_phi(x, ineq_constraints):
    hess = np.zeros((len(x), len(x)), dtype=np.float64)
    for g, grad_g, hess_g in ineq_constraints:
        if g(x) >= 0:
            return np.inf * np.ones((len(x), len(x)), dtype=np.float64)
        hess += grad_g(x).reshape(-1, 1) @ grad_g(x).reshape(1, -1) / g(x)**2 - hess_g(x) / g(x)
    return hess

def backtracking_line_search(f, grad_f, x, direction, alpha=1.0, beta=0.8, sigma=1e-4, ineq_constraints=None):
    while (f(x + alpha * direction) > f(x) + sigma * alpha * np.dot(grad_f(x), direction)) or \
          (ineq_constraints and any(g(x + alpha * direction) >= 0 for g, _, _ in ineq_constraints)):
        alpha *= beta
        if alpha < 1e-10:
            break  # Prevent infinite loop in case of numerical issues
    return alpha

def newton_step(f, grad_f, hess_f, x, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, t, iteration):
    def lagrangian_grad(x):
        lag_grad = t * grad_f(x) + grad_phi(x, ineq_constraints)
        if eq_constraints_mat is not None and eq_constraints_mat.size > 0:
            lag_grad += eq_constraints_mat.T @ np.linalg.solve(eq_constraints_mat @ eq_constraints_mat.T, eq_constraints_rhs - eq_constraints_mat @ x)
        return lag_grad

    def lagrangian_hess(x):
        lag_hess = t * hess_f(x) + hess_phi(x, ineq_constraints)
        if eq_constraints_mat is not None and eq_constraints_mat.size > 0:
            lag_hess += eq_constraints_mat.T @ np.linalg.solve(eq_constraints_mat @ eq_constraints_mat.T, np.eye(eq_constraints_mat.shape[0])) @ eq_constraints_mat
        return lag_hess

    grad = lagrangian_grad(x).astype(np.float64)
    hess = lagrangian_hess(x).astype(np.float64)
    if not np.isfinite(grad).all() or not np.isfinite(hess).all():
        return x  # Return the current point if grad or hess contains inf or NaN values
    direction = -np.linalg.solve(hess, grad)
    alpha = backtracking_line_search(lambda x: t * f(x) + phi(x, ineq_constraints), lagrangian_grad, x, direction, ineq_constraints=ineq_constraints)
    print(f"Iteration {iteration}: f(x) = {f(x)}, Barrier function value = {t * f(x) + phi(x, ineq_constraints)}")
    return x + alpha * direction

def interior_pt(f, grad_f, hess_f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, t=1, mu=2, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    x = np.array(x0, dtype=np.float64)
    path = [x.copy()]

    if any(g(x) >= 0 for g, _, _ in ineq_constraints):
        raise ValueError("Initial point is not feasible")

    iteration = 0
    for _ in range(max_iter):
        x_prev = x.copy()
        for _ in range(20):  # Inner iterations
            iteration += 1
            x = newton_step(f, grad_f, hess_f, x, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, t, iteration)
            if np.linalg.norm(x - x_prev) < param_tol:
                break
            x_prev = x.copy()
        path.append(x.copy())
        t = min(t * mu, 1e10)  # Limit scaling of t to avoid overflow
        if np.linalg.norm(grad_f(x)) < obj_tol:
            break

    return x, f(x), True, path

class ConstrainedMinimization:
    def __init__(self, f, grad_f, hess_f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs):
        self.f = f
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.ineq_constraints = ineq_constraints
        self.eq_constraints_mat = eq_constraints_mat
        self.eq_constraints_rhs = eq_constraints_rhs

    def minimize(self, x0, obj_tol=1e-12, param_tol=1e-8, max_iter=100):
        return interior_pt(self.f, self.grad_f, self.hess_f, self.ineq_constraints, self.eq_constraints_mat, self.eq_constraints_rhs, x0, obj_tol=obj_tol, param_tol=param_tol, max_iter=max_iter)
