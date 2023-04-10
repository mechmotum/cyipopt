"""Test functionality of CyIpopt with/without optional SciPy dependency.

SciPy is an optional dependency of CyIpopt. CyIpopt needs to function without
SciPy installed, but also needs to provide the :func:`minimize_ipopt` function
which requires SciPy.

"""

import re
import sys

import numpy as np
import pytest

import cyipopt

# Hard-code rather than importing from scipy.optimize._minimize in a try/except
MINIMIZE_METHODS = ['nelder-mead', 'powell', 'cg', 'bfgs', 'newton-cg',
                    'l-bfgs-b', 'tnc', 'cobyla', 'slsqp', 'trust-constr',
                    'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']

@pytest.mark.skipif("scipy" in sys.modules,
                    reason="Test only valid if no Scipy available.")
def test_minimize_ipopt_import_error_if_no_scipy():
    """`minimize_ipopt` not callable without SciPy installed."""
    expected_error_msg = re.escape("Install SciPy to use the "
                                   "`minimize_ipopt` function.")
    with pytest.raises(ImportError, match=expected_error_msg):
        cyipopt.minimize_ipopt(None, None)


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
def test_minimize_ipopt_if_scipy():
    """If SciPy is installed `minimize_ipopt` works correctly."""
    from scipy.optimize import rosen, rosen_der
    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    res = cyipopt.minimize_ipopt(rosen, x0, jac=rosen_der)
    assert isinstance(res, dict)
    assert np.isclose(res.get("fun"), 0.0)
    assert res.get("status") == 0
    assert res.get("success") is True
    np.testing.assert_allclose(res.get("x"), np.ones(5))


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
def test_minimize_ipopt_nojac_if_scipy():
    """`minimize_ipopt` works without Jacobian."""
    from scipy.optimize import rosen
    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    options = {"tol": 1e-7}
    res = cyipopt.minimize_ipopt(rosen, x0, options=options)

    assert isinstance(res, dict)
    assert np.isclose(res.get("fun"), 0.0)
    assert res.get("status") == 0
    assert res.get("success") is True
    np.testing.assert_allclose(res.get("x"), np.ones(5), rtol=1e-5)


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
def test_minimize_ipopt_nojac_constraints_if_scipy():
    """ `minimize_ipopt` works without Jacobian and with constraints"""
    from scipy.optimize import rosen
    x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
    constr = {"fun": lambda x: rosen(x) - 1.0, "type": "ineq"}
    res = cyipopt.minimize_ipopt(rosen, x0, constraints=constr)
    assert isinstance(res, dict)
    assert np.isclose(res.get("fun"), 1.0)
    assert res.get("status") == 0
    assert res.get("success") is True
    expected_res = np.array([1.001867, 0.99434067, 1.05070075, 1.17906312,
                             1.38103001])
    np.testing.assert_allclose(res.get("x"), expected_res)


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
def test_minimize_ipopt_jac_and_hessians_constraints_if_scipy(
):
    """`minimize_ipopt` works with objective gradient and Hessian
       and constraint jacobians and Hessians."""
    from scipy.optimize import rosen, rosen_der, rosen_hess
    x0 = [0.0, 0.0]
    constr = {
        "type": "ineq",
        "fun": lambda x: -x[0]**2 - x[1]**2 + 2,
        "jac": lambda x: np.array([-2 * x[0], -2 * x[1]]),
        "hess": lambda x, v: -2 * np.eye(2) * v[0]
    }
    bounds = [(-1.5, 1.5), (-1.5, 1.5)]
    res = cyipopt.minimize_ipopt(rosen, x0, jac=rosen_der, hess=rosen_hess,
                                 constraints=constr)
    assert isinstance(res, dict)
    assert np.isclose(res.get("fun"), 0.0)
    assert res.get("status") == 0
    assert res.get("success") is True
    expected_res = np.array([1.0, 1.0])
    np.testing.assert_allclose(res.get("x"), expected_res, rtol=1e-5)


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
def test_minimize_ipopt_jac_hessians_constraints_with_arg_kwargs():
    """Makes sure that args and kwargs can be passed to all user defined
    functions in minimize_ipopt."""
    from scipy.optimize import rosen, rosen_der, rosen_hess

    rosen2 = lambda x, a, b=None: rosen(x)
    rosen_der2 = lambda x, a, b=None: rosen_der(x)
    rosen_hess2 = lambda x, a, b=None: rosen_hess(x)

    x0 = [0.0, 0.0]
    constr = {
        "type": "ineq",
        "fun": lambda x, a, b=None: -x[0]**2 - x[1]**2 + 2,
        "jac": lambda x, a, b=None: np.array([-2 * x[0], -2 * x[1]]),
        "hess": lambda x, v, a, b=None: -2 * np.eye(2) * v[0],
        "args": (1.0, ),
        "kwargs": {'b': 1.0},
    }
    res = cyipopt.minimize_ipopt(rosen2, x0,
                                 jac=rosen_der2,
                                 hess=rosen_hess2,
                                 args=constr['args'],
                                 kwargs=constr['kwargs'],
                                 constraints=constr)
    assert isinstance(res, dict)
    assert np.isclose(res.get("fun"), 0.0)
    assert res.get("status") == 0
    assert res.get("success") is True
    expected_res = np.array([1.0, 1.0])
    np.testing.assert_allclose(res.get("x"), expected_res, rtol=1e-5)


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
@pytest.mark.parametrize('method', MINIMIZE_METHODS)
def test_minimize_ipopt_jac_with_scipy_methods(method):
    x0 = [0] * 4
    a0, b0, c0, d0 = 1, 2, 3, 4

    def fun(x, a=0, e=0, b=0):
        assert a == a0
        assert b == b0
        fun.count += 1
        return (x[0] - a) ** 2 + (x[1] - b) ** 2 + x[2] ** 2 + x[3] ** 2

    def grad(x, a=0, e=0, b=0):
        assert a == a0
        assert b == b0
        grad.count += 1
        return [2 * (x[0] - a), 2 * (x[1] - b), 2 * x[2], 2 * x[3]]

    def hess(x, a=0, e=0, b=0):
        assert a == a0
        assert b == b0
        hess.count += 1
        return 2 * np.eye(4)

    def fun_constraint(x, c=0, e=0, d=0):
        assert c == c0
        assert d == d0
        fun_constraint.count += 1
        return [(x[2] - c) ** 2, (x[3] - d) ** 2]

    def grad_constraint(x, c=0, e=0, d=0):
        assert c == c0
        assert d == d0
        grad_constraint.count += 1
        return np.hstack((np.zeros((2, 2)),
                         np.diag([2 * (x[2] - c), 2 * (x[3] - d)])))

    fun.count = 0
    grad.count = 0
    hess.count = 0
    fun_constraint.count = 0
    grad_constraint.count = 0

    constr = {
        "type": "eq",
        "fun": fun_constraint,
        "jac": grad_constraint,
        "args": (c0,),
        "kwargs": {'d': d0},
    }

    kwargs = {}
    jac_methods = {'cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'slsqp,',
                   'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact',
                   'trust-constr'}
    hess_methods = {'newton-cg', 'dogleg', 'trust-ncg', 'trust-krylov',
                   'trust-exact', 'trust-constr'}
    constr_methods = {'slsqp', 'trust-constr'}

    if method in jac_methods:
        kwargs['jac'] = grad
    if method in hess_methods:
        kwargs['hess'] = hess
    if method in constr_methods:
        kwargs['constraints'] = constr

    res = cyipopt.minimize_ipopt(fun, x0, method=method, args=(a0,),
                                 kwargs={'b': b0}, **kwargs)

    assert res.success
    np.testing.assert_allclose(res.x[:2], [a0, b0], rtol=1e-3)

    # confirm that the test covers what we think it does: all the functions
    # that we provide are actually being executed; that is, the assertions
    # are *passing*, not being skipped
    assert fun.count > 0
    if method in jac_methods:
        assert grad.count > 0
    if method in hess_methods:
        assert hess.count > 0
    if method in constr_methods:
        assert fun_constraint.count > 0
        assert grad_constraint.count > 0
        np.testing.assert_allclose(res.x[2:], [c0, d0], rtol=1e-3)
    else:
        np.testing.assert_allclose(res.x[2:], 0, atol=1e-3)


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid of Scipy available")
def test_minimize_ipopt_sparse_jac_if_scipy():
    """ `minimize_ipopt` works with objective gradient, and sparse
        constraint jacobian. Solves
        Hock & Schittkowski's test problem 71:

        min x0*x3*(x0+x1+x2)+x2
        s.t. x0**2 + x1**2 + x2**2 + x3**2 - 40  = 0
                         x0 * x1 * x2 * x3 - 25 >= 0
                               1 <= x0,x1,x2,x3 <= 5
    """
    try:
        from scipy.sparse import coo_array
    except ImportError:
        from scipy.sparse import coo_matrix as coo_array

    def obj(x):
        return x[0] * x[3] * np.sum(x[:3]) + x[2]

    def grad(x):
        return np.array([
            x[0] * x[3] + x[3] * np.sum(x[0:3]), x[0] * x[3],
            x[0] * x[3] + 1.0, x[0] * np.sum(x[0:3])
        ])

    # Note:
    # coo_array(dense_jac_val(x)) only works if dense_jac_val(x0)
    # doesn't contain any zeros for the initial guess x0

    con_eq = {
        "type": "eq",
        "fun": lambda x: np.sum(x**2) - 40,
        "jac": lambda x: coo_array(2 * x)
    }
    con_ineq = {
        "type": "ineq",
        "fun": lambda x: np.prod(x) - 25,
        "jac": lambda x: coo_array(np.prod(x) / x),
    }
    constrs = (con_eq, con_ineq)

    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    bnds = [(1, 5) for _ in range(x0.size)]

    res = cyipopt.minimize_ipopt(obj, jac=grad, x0=x0,
                                 bounds=bnds, constraints=constrs)
    assert isinstance(res, dict)
    assert np.isclose(res.get("fun"), 17.01401727277449)
    assert res.get("status") == 0
    assert res.get("success") is True
    expected_res = np.array([0.99999999, 4.74299964, 3.82114998, 1.3794083])
    np.testing.assert_allclose(res.get("x"), expected_res)


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid of Scipy available")
def test_minimize_ipopt_sparse_and_dense_jac_if_scipy():
    """ `minimize_ipopt` works with objective gradient, and sparse
        constraint jacobian. Solves
        Hock & Schittkowski's test problem 71:

        min x0*x3*(x0+x1+x2)+x2
        s.t. x0**2 + x1**2 + x2**2 + x3**2 - 40  = 0
                         x0 * x1 * x2 * x3 - 25 >= 0
                               1 <= x0,x1,x2,x3 <= 5
    """
    try:
        from scipy.sparse import coo_array
    except ImportError:
        from scipy.sparse import coo_matrix as coo_array

    def obj(x):
        return x[0] * x[3] * np.sum(x[:3]) + x[2]

    def grad(x):
        return np.array([
            x[0] * x[3] + x[3] * np.sum(x[0:3]), x[0] * x[3],
            x[0] * x[3] + 1.0, x[0] * np.sum(x[0:3])
        ])

    # Note:
    # coo_array(dense_jac_val(x)) only works if dense_jac_val(x0)
    # doesn't contain any zeros for the initial guess x0

    con_eq_dense = {
        "type": "eq",
        "fun": lambda x: np.sum(x**2) - 40,
        "jac": lambda x: 2 * x
    }
    con_ineq_sparse = {
        "type": "ineq",
        "fun": lambda x: np.prod(x) - 25,
        "jac": lambda x: coo_array(np.prod(x) / x),
    }
    constrs = (con_eq_dense, con_ineq_sparse)

    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    bnds = [(1, 5) for _ in range(x0.size)]

    res = cyipopt.minimize_ipopt(obj, jac=grad, x0=x0,
                                 bounds=bnds, constraints=constrs)
    assert isinstance(res, dict)
    assert np.isclose(res.get("fun"), 17.01401727277449)
    assert res.get("status") == 0
    assert res.get("success") is True
    expected_res = np.array([0.99999999, 4.74299964, 3.82114998, 1.3794083])
    np.testing.assert_allclose(res.get("x"), expected_res)


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
def test_minimize_ipopt_hs071():
    """ `minimize_ipopt` works with objective gradient and Hessian and
         constraint jacobians and Hessians.

        The objective and the constraints functions return a tuple containing
        the function value and the evaluated gradient or jacobian. Solves
        Hock & Schittkowski's test problem 71:

        min x0*x3*(x0+x1+x2)+x2
        s.t. x0**2 + x1**2 + x2**2 + x3**2 - 40  = 0
                         x0 * x1 * x2 * x3 - 25 >= 0
                               1 <= x0,x1,x2,x3 <= 5
    """

    def obj_and_grad(x):
        obj = x[0] * x[3] * np.sum(x[:3]) + x[2]
        grad = np.array([
            x[0] * x[3] + x[3] * np.sum(x[0:3]), x[0] * x[3],
            x[0] * x[3] + 1.0, x[0] * np.sum(x[0:3])
        ])
        return obj, grad

    def obj_hess(x):
        return np.array([[2 * x[3], 0.0, 0, 0], [x[3], 0, 0, 0],
                         [x[3], 0, 0, 0],
                         [2 * x[0] + x[1] + x[2], x[0], x[0], 0]])

    def con_eq_and_jac(x):
        value = np.sum(x**2) - 40
        jac = np.array([2 * x])
        return value, jac

    def con_eq_hess(x, v):
        return v[0] * 2.0 * np.eye(4)

    def con_ineq_and_jac(x):
        value = np.prod(x) - 25
        jac = np.array([np.prod(x) / x])
        return value, jac

    def con_ineq_hess(x, v):
        return v[0] * np.array([[0, 0, 0, 0], [x[2] * x[3], 0, 0, 0],
                                [x[1] * x[3], x[0] * x[3], 0, 0],
                                [x[1] * x[2], x[0] * x[2], x[0] * x[1], 0]])

    con1 = {
        "type": "eq",
        "fun": con_eq_and_jac,
        "jac": True,
        "hess": con_eq_hess
    }
    con2 = {
        "type": "ineq",
        "fun": con_ineq_and_jac,
        "jac": True,
        "hess": con_ineq_hess
    }
    constrs = (con1, con2)

    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    bnds = [(1, 5) for _ in range(x0.size)]

    res = cyipopt.minimize_ipopt(obj_and_grad, jac=True, hess=obj_hess, x0=x0,
                                 bounds=bnds, constraints=constrs)
    assert isinstance(res, dict)
    assert np.isclose(res.get("fun"), 17.01401727277449)
    assert res.get("status") == 0
    assert res.get("success") is True
    expected_res = np.array([0.99999999, 4.74299964, 3.82114998, 1.3794083])
    np.testing.assert_allclose(res.get("x"), expected_res)
