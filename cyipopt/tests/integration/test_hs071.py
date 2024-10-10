import numpy as np
import pytest

import cyipopt


def test_hs071_solve(hs071_initial_guess_fixture,
                     hs071_optimal_solution_fixture,
                     hs071_problem_instance_fixture):
	"""Test hs071 test problem solves to the correct solution."""
	x0 = hs071_initial_guess_fixture
	nlp = hs071_problem_instance_fixture
	x, info = nlp.solve(x0)

	expected_x, expected_J = hs071_optimal_solution_fixture

	np.testing.assert_almost_equal(info["obj_val"], expected_J)
	np.testing.assert_allclose(x, expected_x)


def test_hs071_warm_start(hs071_initial_guess_fixture,
                          hs071_optimal_solution_fixture,
                          hs071_problem_instance_fixture):
    x0 = hs071_initial_guess_fixture
    nlp = hs071_problem_instance_fixture

    _, info = nlp.solve(x0)

    x_opt, _ = hs071_optimal_solution_fixture
    np.testing.assert_allclose(info['x'], x_opt)

    x_init = info['x']
    lagrange = info['mult_g']
    zl = info['mult_x_L']
    zu = info['mult_x_U']

    # Set parameters to avoid push the solution
    # away from the variable bounds
    nlp.add_option('warm_start_init_point', 'yes')
    nlp.add_option("warm_start_bound_frac", 1e-6)
    nlp.add_option("warm_start_bound_push", 1e-6)
    nlp.add_option("warm_start_slack_bound_frac", 1e-6)
    nlp.add_option("warm_start_slack_bound_push", 1e-6)
    nlp.add_option("warm_start_mult_bound_push", 1e-6)

    _, info = nlp.solve(x_init,
                        lagrange=lagrange,
                        zl=zl,
                        zu=zu)

    np.testing.assert_allclose(info['x'], x_opt)


def _make_problem(definition, lb, ub, cl, cu):
    n = len(lb)
    m = len(cl)
    return cyipopt.Problem(
        n=n, m=m, problem_obj=definition, lb=lb, ub=ub, cl=cl, cu=cu
    )


def _solve_and_assert_correct(problem, x0, opt_sol):
    x, info = problem.solve(x0)
    expected_x, _ = opt_sol
    expected_x = np.array(expected_x)
    assert info["status"] == 0
    np.testing.assert_allclose(x, expected_x)


def _assert_solve_fails(problem, x0):
    x, info = problem.solve(x0)
    # The current (Ipopt 3.14.11) return status is "Invalid number".
    assert info["status"] < 0


def test_hs071_objective_eval_error(
    hs071_initial_guess_fixture,
    hs071_optimal_solution_fixture,
    hs071_problem_instance_fixture,
    hs071_definition_instance_fixture,
    hs071_variable_lower_bounds_fixture,
    hs071_variable_upper_bounds_fixture,
    hs071_constraint_lower_bounds_fixture,
    hs071_constraint_upper_bounds_fixture,
):
    class ObjectiveWithError:
        def __init__(self):
            self.n_eval_error = 0

        def __call__(self, x):
            if x[0] > 1.1:
                self.n_eval_error += 1
                raise cyipopt.CyIpoptEvaluationError()
            return x[0] * x[3] * np.sum(x[0:3]) + x[2]

    objective_with_error = ObjectiveWithError()

    x0 = hs071_initial_guess_fixture
    opt = hs071_optimal_solution_fixture
    definition = hs071_definition_instance_fixture
    definition.objective = objective_with_error
    definition.intermediate = None

    lb = hs071_variable_lower_bounds_fixture
    ub = hs071_variable_upper_bounds_fixture
    cl = hs071_constraint_lower_bounds_fixture
    cu = hs071_constraint_upper_bounds_fixture

    problem = _make_problem(definition, lb, ub, cl, cu)
    # Note that the behavior tested here (success or failure of the solve when
    # an evaluation error is raised in each callback) is documented in the
    # CyIpoptEvaluationError class. If this behavior changes (e.g. Ipopt starts
    # handling evaluation errors in the Jacobian), these tests will start to
    # fail. We will need to (a) update these tests and (b) update the
    # CyIpoptEvaluationError documentation, possibly with Ipopt version-specific
    # behavior.
    _solve_and_assert_correct(problem, x0, opt)

    assert objective_with_error.n_eval_error > 0


def test_hs071_grad_obj_eval_error(
    hs071_initial_guess_fixture,
    hs071_problem_instance_fixture,
    hs071_definition_instance_fixture,
    hs071_variable_lower_bounds_fixture,
    hs071_variable_upper_bounds_fixture,
    hs071_constraint_lower_bounds_fixture,
    hs071_constraint_upper_bounds_fixture,
):
    class GradObjWithError:
        def __init__(self):
            self.n_eval_error = 0

        def __call__(self, x):
            if x[0] > 1.1:
                self.n_eval_error += 1
                raise cyipopt.CyIpoptEvaluationError()
            return np.array([
                x[0] * x[3] + x[3] * np.sum(x[0:3]),
                x[0] * x[3],
                x[0] * x[3] + 1.0,
                x[0] * np.sum(x[0:3]),
            ])

    gradient_with_error = GradObjWithError()

    x0 = hs071_initial_guess_fixture
    definition = hs071_definition_instance_fixture
    definition.gradient = gradient_with_error
    definition.intermediate = None

    lb = hs071_variable_lower_bounds_fixture
    ub = hs071_variable_upper_bounds_fixture
    cl = hs071_constraint_lower_bounds_fixture
    cu = hs071_constraint_upper_bounds_fixture

    problem = _make_problem(definition, lb, ub, cl, cu)
    # Solve fails when the evaluation error occurs in objective
    # gradient evaluation.
    _assert_solve_fails(problem, x0)

    # Since we fail at the first evaluation error, we know we only encountered one.
    assert gradient_with_error.n_eval_error == 1


def test_hs071_constraints_eval_error(
    hs071_initial_guess_fixture,
    hs071_optimal_solution_fixture,
    hs071_problem_instance_fixture,
    hs071_definition_instance_fixture,
    hs071_variable_lower_bounds_fixture,
    hs071_variable_upper_bounds_fixture,
    hs071_constraint_lower_bounds_fixture,
    hs071_constraint_upper_bounds_fixture,
):
    class ConstraintsWithError:
        def __init__(self):
            self.n_eval_error = 0

        def __call__(self, x):
            if x[0] > 1.1:
                self.n_eval_error += 1
                raise cyipopt.CyIpoptEvaluationError()
            return np.array((np.prod(x), np.dot(x, x)))

    constraints_with_error = ConstraintsWithError()

    x0 = hs071_initial_guess_fixture
    opt = hs071_optimal_solution_fixture
    definition = hs071_definition_instance_fixture
    definition.constraints = constraints_with_error
    definition.intermediate = None

    lb = hs071_variable_lower_bounds_fixture
    ub = hs071_variable_upper_bounds_fixture
    cl = hs071_constraint_lower_bounds_fixture
    cu = hs071_constraint_upper_bounds_fixture

    problem = _make_problem(definition, lb, ub, cl, cu)
    _solve_and_assert_correct(problem, x0, opt)

    assert constraints_with_error.n_eval_error > 0


def test_hs071_jacobian_eval_error(
    hs071_initial_guess_fixture,
    hs071_problem_instance_fixture,
    hs071_definition_instance_fixture,
    hs071_variable_lower_bounds_fixture,
    hs071_variable_upper_bounds_fixture,
    hs071_constraint_lower_bounds_fixture,
    hs071_constraint_upper_bounds_fixture,
):
    class JacobianWithError:
        def __init__(self):
            self.n_eval_error = 0

        def __call__(self, x):
            if x[0] > 1.1:
                self.n_eval_error += 1
                raise cyipopt.CyIpoptEvaluationError()
            return np.concatenate((np.prod(x) / x, 2 * x))

    jacobian_with_error = JacobianWithError()

    x0 = hs071_initial_guess_fixture
    definition = hs071_definition_instance_fixture
    definition.jacobian = jacobian_with_error
    definition.intermediate = None

    lb = hs071_variable_lower_bounds_fixture
    ub = hs071_variable_upper_bounds_fixture
    cl = hs071_constraint_lower_bounds_fixture
    cu = hs071_constraint_upper_bounds_fixture

    problem = _make_problem(definition, lb, ub, cl, cu)
    # Solve fails when the evaluation error occurs in constraint
    # Jacobian evaluation.
    _assert_solve_fails(problem, x0)

    assert jacobian_with_error.n_eval_error == 1


def test_hs071_hessian_eval_error(
    hs071_initial_guess_fixture,
    hs071_problem_instance_fixture,
    hs071_definition_instance_fixture,
    hs071_variable_lower_bounds_fixture,
    hs071_variable_upper_bounds_fixture,
    hs071_constraint_lower_bounds_fixture,
    hs071_constraint_upper_bounds_fixture,
):
    class HessianWithError:
        def __init__(self):
            self.n_eval_error = 0

        def __call__(self, x, lagrange, obj_factor):
            if x[0] > 1.1:
                self.n_eval_error += 1
                raise cyipopt.CyIpoptEvaluationError()

            H = obj_factor * np.array([
                (2 * x[3], 0, 0, 0),
                (x[3], 0, 0, 0),
                (x[3], 0, 0, 0),
                (2 * x[0] + x[1] + x[2], x[0], x[0], 0),
            ])
            H += lagrange[0] * np.array([
                (0, 0, 0, 0),
                (x[2] * x[3], 0, 0, 0),
                (x[1] * x[3], x[0] * x[3], 0, 0),
                (x[1] * x[2], x[0] * x[2], x[0] * x[1], 0),
            ])
            H += lagrange[1] * 2 * np.eye(4)
            row, col = np.nonzero(np.tril(np.ones((4, 4))))
            return H[row, col]

    hessian_with_error = HessianWithError()

    x0 = hs071_initial_guess_fixture
    definition = hs071_definition_instance_fixture
    definition.hessian = hessian_with_error
    definition.intermediate = None

    lb = hs071_variable_lower_bounds_fixture
    ub = hs071_variable_upper_bounds_fixture
    cl = hs071_constraint_lower_bounds_fixture
    cu = hs071_constraint_upper_bounds_fixture

    problem = _make_problem(definition, lb, ub, cl, cu)
    # Solve fails when the evaluation error occurs in Lagrangian
    # Hessian evaluation.
    _assert_solve_fails(problem, x0)

    assert hessian_with_error.n_eval_error == 1
