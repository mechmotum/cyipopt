import numpy as np
import pytest

import cyipopt


def test_hs071_solve(hs071_initial_guess_fixture, hs071_problem_instance_fixture):
	"""Test hs071 test problem solves to the correct solution."""
	x0 = hs071_initial_guess_fixture
	nlp = hs071_problem_instance_fixture
	x, info = nlp.solve(x0)

	expected_J = 17.01401714021362
	np.testing.assert_almost_equal(info["obj_val"], expected_J)

	expected_x = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
	np.testing.assert_allclose(x, expected_x)


def _make_problem(definition, lb, ub, cl, cu):
    n = len(lb)
    m = len(cl)
    return cyipopt.Problem(
        n=n, m=m, problem_obj=definition, lb=lb, ub=ub, cl=cl, cu=cu
    )


def _solve_and_assert_correct(problem, x0):
    x, info = problem.solve(x0)
    expected_x = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
    assert info["status"] == 0
    np.testing.assert_allclose(x, expected_x)


def _assert_solve_fails(problem, x0):
    x, info = problem.solve(x0)
    # The current (Ipopt 3.14.11) return status is "Invalid number".
    assert info["status"] < 0


def test_hs071_objective_eval_error(
    hs071_initial_guess_fixture,
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
    definition = hs071_definition_instance_fixture
    definition.objective = objective_with_error
    definition.intermediate = None

    lb = hs071_variable_lower_bounds_fixture
    ub = hs071_variable_upper_bounds_fixture
    cl = hs071_constraint_lower_bounds_fixture
    cu = hs071_constraint_upper_bounds_fixture

    problem = _make_problem(definition, lb, ub, cl, cu)
    _solve_and_assert_correct(problem, x0)

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
    _assert_solve_fails(problem, x0)

    # Since we fail at the first evaluation error, we know we only encountered one.
    assert gradient_with_error.n_eval_error == 1
