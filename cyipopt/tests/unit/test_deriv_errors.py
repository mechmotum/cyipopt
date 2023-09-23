import numpy as np

import cyipopt

import pytest


pre_3_14_13 = (
    cyipopt.IPOPT_VERSION < (3, 14, 13)
)


def full_indices(shape):
    def indices():
        r, c = np.indices(shape)
        return r.flatten(), c.flatten()

    return indices


def tril_indices(size):
    def indices():
        return np.tril_indices(size)

    return indices


def flatten(func):
    def _func(*args):
        return func(*args).flatten()

    return _func


@pytest.fixture
def hs071_sparse_definition_fixture(hs071_variable_lower_bounds_fixture,
                                    hs071_constraint_lower_bounds_fixture,
                                    hs071_definition_instance_fixture):
    problem = hs071_definition_instance_fixture
    n = len(hs071_variable_lower_bounds_fixture)
    m = len(hs071_constraint_lower_bounds_fixture)

    problem.jacobianstructure = full_indices((m, n))
    problem.hessianstructure = tril_indices(n)

    problem.jacobian = flatten(problem.jacobian)
    problem.hessian = flatten(problem.hessian)

    return problem


@pytest.fixture
def hs071_sparse_instance(hs071_initial_guess_fixture,
                          hs071_variable_lower_bounds_fixture,
                          hs071_variable_upper_bounds_fixture,
                          hs071_constraint_lower_bounds_fixture,
                          hs071_constraint_upper_bounds_fixture,
                          hs071_sparse_definition_fixture):

    class Instance:
        pass

    instance = Instance()
    instance.problem_definition = hs071_sparse_definition_fixture
    instance.x0 = hs071_initial_guess_fixture
    instance.lb = hs071_variable_lower_bounds_fixture
    instance.ub = hs071_variable_upper_bounds_fixture
    instance.cl = hs071_constraint_lower_bounds_fixture
    instance.cu = hs071_constraint_upper_bounds_fixture
    instance.n = len(instance.x0)
    instance.m = len(instance.cl)

    return instance


def problem_for_instance(instance):
    return cyipopt.Problem(n=instance.n,
                           m=instance.m,
                           problem_obj=instance.problem_definition,
                           lb=instance.lb,
                           ub=instance.ub,
                           cl=instance.cl,
                           cu=instance.cu)


def test_solve_sparse(hs071_sparse_instance):
    instance = hs071_sparse_instance
    problem = problem_for_instance(instance)

    x, info = problem.solve(instance.x0)

    assert info['status'] == 0


def ensure_solve_status(instance, status):
    problem = problem_for_instance(instance)

    problem.add_option('max_iter', 50)
    x, info = problem.solve(instance.x0)

    assert info['status'] == status


def ensure_invalid_option(instance):
    # -12: Invalid option
    # Thrown in invalid Hessian because "hessian_approximation"
    # is not chosen as "limited-memory"
    ensure_solve_status(instance, -12)


def ensure_invalid_number(instance):
    # -13: Invalid Number Detected
    ensure_solve_status(instance, -13)


def ensure_unrecoverable_exception(instance):
    # -100: Unrecoverable Exception
    # *Should* be returned from errors in initialization
    ensure_solve_status(instance, -100)


@pytest.mark.skipif(pre_3_14_13, reason="Not caught in Ipopt < (3,14,13)")
def test_solve_neg_jac(hs071_sparse_instance):
    n = hs071_sparse_instance.n
    m = hs071_sparse_instance.m
    problem_definition = hs071_sparse_instance.problem_definition

    def jacobianstructure():
        r = np.full((m*n,), fill_value=-1, dtype=int)
        c = np.full((m*n,), fill_value=-1, dtype=int)
        return r, c

    problem_definition.jacobianstructure = jacobianstructure

    ensure_unrecoverable_exception(hs071_sparse_instance)


@pytest.mark.skipif(pre_3_14_13, reason="Not caught in Ipopt < (3,14,13)")
def test_solve_large_jac(hs071_sparse_instance):
    n = hs071_sparse_instance.n
    m = hs071_sparse_instance.m
    problem_definition = hs071_sparse_instance.problem_definition

    import logging
    logging.basicConfig(level=logging.DEBUG)

    def jacobianstructure():
        r = np.full((m*n,), fill_value=(m + n + 100), dtype=int)
        c = np.full((m*n,), fill_value=(m + n + 100), dtype=int)
        return r, c

    problem_definition.jacobianstructure = jacobianstructure

    ensure_unrecoverable_exception(hs071_sparse_instance)


@pytest.mark.skipif(pre_3_14_13, reason="Not caught in Ipopt < (3,14,13)")
def test_solve_wrong_jac_structure_size(hs071_sparse_instance):
    n = hs071_sparse_instance.n
    m = hs071_sparse_instance.m

    problem_definition = hs071_sparse_instance.problem_definition

    problem_definition.jacobianstructure = full_indices((m+1, n+1))

    ensure_unrecoverable_exception(hs071_sparse_instance)


@pytest.mark.skipif(pre_3_14_13, reason="Not caught in Ipopt < (3,14,13)")
def test_solve_wrong_jac_value_size(hs071_sparse_instance):
    n = hs071_sparse_instance.n
    m = hs071_sparse_instance.m

    problem_definition = hs071_sparse_instance.problem_definition

    def jacobian(x):
        return np.zeros((m*n + 10,))

    problem_definition.jacobian = jacobian

    ensure_invalid_number(hs071_sparse_instance)


def test_solve_triu_hess(hs071_sparse_instance):
    n = hs071_sparse_instance.n
    problem_definition = hs071_sparse_instance.problem_definition
    problem_definition.hessianstructure = lambda: np.triu_indices(n)

    ensure_invalid_option(hs071_sparse_instance)


def test_solve_neg_hess_entries(hs071_sparse_instance):
    n = hs071_sparse_instance.n
    problem_definition = hs071_sparse_instance.problem_definition

    def hessianstructure():
        r, c = np.tril_indices(n)
        rneg = np.full_like(r, -1, dtype=int)
        cneg = np.full_like(c, -1, dtype=int)
        return rneg, cneg

    problem_definition.hessianstructure = hessianstructure

    ensure_invalid_option(hs071_sparse_instance)


def test_solve_large_hess_entries(hs071_sparse_instance):
    n = hs071_sparse_instance.n
    problem_definition = hs071_sparse_instance.problem_definition

    def hessianstructure():
        r, c = np.tril_indices(n)
        rlarge = np.full_like(r, n + 100, dtype=int)
        clarge = np.full_like(c, n + 100, dtype=int)
        return rlarge, clarge

    problem_definition.hessianstructure = hessianstructure

    ensure_invalid_option(hs071_sparse_instance)


def test_solve_wrong_hess_struct_size(hs071_sparse_instance):
    n = hs071_sparse_instance.n
    problem_definition = hs071_sparse_instance.problem_definition

    def hessianstructure():
        return np.tril_indices(n + 10)

    problem_definition.hessianstructure = hessianstructure

    ensure_invalid_option(hs071_sparse_instance)


def test_solve_wrong_hess_value_size(hs071_sparse_instance):
    n = hs071_sparse_instance.n
    problem_definition = hs071_sparse_instance.problem_definition

    def hessian(x, lag, obj_factor):
        return np.zeros((n*n + 10,))

    problem_definition.hessian = hessian

    ensure_invalid_number(hs071_sparse_instance)
