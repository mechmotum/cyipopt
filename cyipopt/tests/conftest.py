"""Fixture-sharing file for test suite.

This file includes a default instantiation of the hs071 test problem used
elsewhere within this package for demonstration purposes.
"""

import sys

import numpy as np
import pytest

import cyipopt


@pytest.fixture()
def hs071_objective_fixture():
    """Return a function for the hs071 test problem objective."""

    def objective(x):
        return x[0] * x[3] * np.sum(x[0:3]) + x[2]

    return objective


@pytest.fixture()
def hs071_gradient_fixture():
    """Return a function for the hs071 test problem gradient."""

    def gradient(x):
        return np.array([x[0] * x[3] + x[3] * np.sum(x[0:3]),
                         x[0] * x[3],
                         x[0] * x[3] + 1.0,
                         x[0] * np.sum(x[0:3])
                         ])

    return gradient


@pytest.fixture()
def hs071_constraints_fixture():
    """Return a function for the hs071 test problem constraints."""

    def constraints(x):
        return np.array((np.prod(x), np.dot(x, x)))

    return constraints


@pytest.fixture()
def hs071_jacobian_fixture():
    """Return a function for the hs071 test problem jacobian."""

    def jacobian(x):
        return np.concatenate((np.prod(x) / x, 2 * x))

    return jacobian


@pytest.fixture()
def hs071_jacobian_structure_fixture():
    """Return a function for the hs071 test problem jacobian structure."""

    def jacobian_structure(x):
        return np.ones(1, 4)

    return jacobian_structure


@pytest.fixture()
def hs071_hessian_fixture():
    """Return a function for the hs071 test problem hessian."""

    def hessian(x, lagrange, obj_factor):
        H = obj_factor * np.array(((2 * x[3], 0, 0, 0),
                                   (x[3], 0, 0, 0),
                                   (x[3], 0, 0, 0),
                                   (2 * x[0] + x[1] + x[2], x[0], x[0], 0)
                                   ))
        H += lagrange[0] * np.array(((0, 0, 0, 0),
                                     (x[2] * x[3], 0, 0, 0),
                                     (x[1] * x[3], x[0] * x[3], 0, 0),
                                     (x[1] * x[2], x[0] * x[2], x[0] * x[1], 0)
                                     ))
        H += lagrange[1] * 2 * np.eye(4)
        row, col = np.nonzero(np.tril(np.ones((4, 4))))
        return H[row, col]

    return hessian


@pytest.fixture()
def hs071_hessian_structure_fixture():
    """Return a function for the hs071 test problem hessian structure."""

    def hessian_structure(x):
        return np.nonzero(np.tril(np.ones((4, 4))))

    return hessian_structure


@pytest.fixture()
def hs071_intermediate_fixture():
    """Return a function for a default intermediate function."""
    def intermediate(*args):
        iter_count = args[2]
        obj_value = args[3]
        msg = f"Objective value at iteration #{iter_count} is - {obj_value}"
        print(msg)

    return intermediate


@pytest.fixture()
def hs071_defintion_instance_fixture(hs071_objective_fixture,
                                     hs071_gradient_fixture,
                                     hs071_constraints_fixture,
                                     hs071_jacobian_fixture,
                                     hs071_jacobian_structure_fixture,
                                     hs071_hessian_fixture,
                                     hs071_hessian_structure_fixture,
                                     hs071_intermediate_fixture,
                                     ):
    """Return a default implementation of the hs071 test problem."""

    class hs071:
        """The hs071 test problem also found in examples."""

        def __init__(self):
            self.objective = hs071_objective_fixture
            self.gradient = hs071_gradient_fixture
            self.constraints = hs071_constraints_fixture
            self.jacobian = hs071_jacobian_fixture
            self.jacobian_structure = hs071_jacobian_structure_fixture
            self.hessian = hs071_hessian_fixture
            self.hessian_structure = hs071_hessian_structure_fixture
            self.intermediate = hs071_intermediate_fixture

    problem_instance = hs071()
    return problem_instance


@pytest.fixture()
def hs071_initial_guess_fixture():
    """Return a default initial guess for the hs071 test problem."""
    x0 = [1.0, 5.0, 5.0, 1.0]
    return x0


@pytest.fixture()
def hs071_variable_lower_bounds_fixture():
    """Return a default variable lower bounds for the hs071 test problem."""
    lb = [1.0, 1.0, 1.0, 1.0]
    return lb


@pytest.fixture()
def hs071_variable_upper_bounds_fixture():
    """Return a default variable upper bounds for the hs071 test problem."""
    ub = [5.0, 5.0, 5.0, 5.0]
    return ub


@pytest.fixture()
def hs071_constraint_lower_bounds_fixture():
    """Return a default constraint lower bounds for the hs071 test problem."""
    cl = [25.0, 40.0]
    return cl


@pytest.fixture()
def hs071_constraint_upper_bounds_fixture():
    """Return a default constraint upper bounds for the hs071 test problem."""
    cu = [2.0e19, 40.0]
    return cu


@pytest.fixture()
def hs071_problem_instance_fixture(hs071_defintion_instance_fixture,
                                   hs071_initial_guess_fixture,
                                   hs071_variable_lower_bounds_fixture,
                                   hs071_variable_upper_bounds_fixture,
                                   hs071_constraint_lower_bounds_fixture,
                                   hs071_constraint_upper_bounds_fixture,
                                   ):
    """Return a default cyipopt.Problem instance of the hs071 test problem."""
    problem_definition = hs071_defintion_instance_fixture
    x0 = hs071_initial_guess_fixture
    lb = hs071_variable_lower_bounds_fixture
    ub = hs071_variable_upper_bounds_fixture
    cl = hs071_constraint_lower_bounds_fixture
    cu = hs071_constraint_upper_bounds_fixture
    n = len(x0)
    m = len(cl)
    problem = cyipopt.Problem(n=n, m=m, problem_obj=problem_definition, lb=lb,
                              ub=ub, cl=cl, cu=cu)
    return problem


@pytest.fixture
def hs071_problem_instance_with_options_fixture(hs071_problem_instance_fixture):
    nlp = hs071_problem_instance_fixture
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-7)
    return nlp
