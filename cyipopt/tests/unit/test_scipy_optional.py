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
try:
    from scipy.optimize._minimize import MINIMIZE_METHODS
except ImportError:
    MINIMIZE_METHODS = []

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
def test_tol_type_issue_235():
    # from: https://github.com/mechmotum/cyipopt/issues/235

    def fun(x):
        return np.sum(x ** 2)

    # tol should not raise an error
    cyipopt.minimize_ipopt(
        fun=fun,
        x0=np.zeros(2),
        tol=1e-9,
    )


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
def test_minimize_ipopt_input_validation():
    from scipy import optimize

    x0 = 1
    def f(x):
        return x @ x

    message = "`fun` must be callable."
    with pytest.raises(ValueError, match=message):
        cyipopt.minimize_ipopt('migratory coconuts', x0)

    message = "`x0` must be a numeric array."
    with pytest.raises(ValueError, match=message):
        cyipopt.minimize_ipopt(f, 'spamalot')

    message = "`kwargs` must be a dictionary."
    with pytest.raises(ValueError, match=message):
        cyipopt.minimize_ipopt(f, x0, kwargs='elderberries')

    message = "Unknown solver..."
    with pytest.raises(ValueError, match=message):
        cyipopt.minimize_ipopt(f, x0, method='a newt')

    message = "`jac` must be callable or boolean."
    with pytest.raises(ValueError, match=message):
        cyipopt.minimize_ipopt(f, x0, jac='self-perpetuating autocracy')

    message = "`hess` must be callable."
    with pytest.raises(ValueError, match=message):
        cyipopt.minimize_ipopt(f, x0, hess='farcical aquatic ceremony')

    message = "`hessp` is not yet supported by Ipopt."
    with pytest.raises(NotImplementedError, match=message):
        cyipopt.minimize_ipopt(f, x0, hessp='shrubbery')

    message = "`bounds` must specify both lower and upper..."
    with pytest.raises(ValueError, match=message):
        cyipopt.minimize_ipopt(f, [1, 2], bounds=1)

    message = "The number of lower bounds, upper bounds..."
    with pytest.raises(ValueError, match=message):
        cyipopt.minimize_ipopt(f, [1, 2], bounds=[(1, 2), (3, 4), (5, 6)])

    message = "The bounds must be numeric."
    with pytest.raises(ValueError, match=message):
        cyipopt.minimize_ipopt(f, x0, bounds=[['low', 'high']])

    message = "`callback` is not yet supported by Ipopt."
    with pytest.raises(NotImplementedError, match=message):
        cyipopt.minimize_ipopt(f, x0, callback='a duck')

    message = "`tol` must be a positive scalar."
    with pytest.raises(ValueError, match=message):
        cyipopt.minimize_ipopt(f, x0, tol=[1, 2, 3])
    with pytest.raises(ValueError, match=message):
        cyipopt.minimize_ipopt(f, x0, tol='ni')
    with pytest.raises(ValueError, match=message):
        cyipopt.minimize_ipopt(f, x0, tol=-1)

    message = "`options` must be a dictionary."
    with pytest.raises(ValueError, match=message):
        cyipopt.minimize_ipopt(f, x0, options='supreme executive power')


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

    rosen2 = lambda x, a, b=None: rosen(x)*a*b
    rosen_der2 = lambda x, a, b=None: rosen_der(x)*a*b
    rosen_hess2 = lambda x, a, b=None: rosen_hess(x)*a*b

    x0 = [0.0, 0.0]
    constr = {
        "type": "ineq",
        "fun": lambda x, a, b=None: -x[0]**2 - x[1]**2 + 2*a*b,
        "jac": lambda x, a, b=None: np.array([-2 * x[0], -2 * x[1]])*a*b,
        "hess": lambda x, v, a, b=None: -2 * np.eye(2) * v[0]*a*b,
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
@pytest.mark.parametrize('method', [None] + MINIMIZE_METHODS)
def test_minimize_ipopt_jac_with_scipy_methods(method):
    x0 = [0] * 4
    a0, b0, c0, d0 = 1, 2, 3, 4
    atol, rtol = 5e-5, 5e-5

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

    def hessp(x, p, a=0, e=0, b=0):
        assert a == a0
        assert b == b0
        hessp.count += 1
        return 2 * np.eye(4) @ p

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

    def callback(*args, **kwargs):
        callback.count += 1

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
    hessp_methods = hess_methods - {'dogleg', 'trust-exact'}
    constr_methods = {'slsqp', 'trust-constr'}

    if method in jac_methods:
        kwargs['jac'] = grad
    if method in hess_methods:
        kwargs['hess'] = hess
    if method in constr_methods:
        kwargs['constraints'] = constr
    if method in MINIMIZE_METHODS:
        kwargs['callback'] = callback

    fun.count = 0
    grad.count = 0
    hess.count = 0
    hessp.count = 0
    fun_constraint.count = 0
    grad_constraint.count = 0
    callback.count = 0

    res = cyipopt.minimize_ipopt(fun, x0, method=method, args=(a0,),
                                 kwargs={'b': b0}, **kwargs)

    assert res.success
    np.testing.assert_allclose(res.x[:2], [a0, b0], rtol=rtol)

    # confirm that the test covers what we think it does: all the functions
    # that we provide are actually being executed; that is, the assertions
    # are *passing*, not being skipped
    assert fun.count > 0
    if method in MINIMIZE_METHODS:
        assert callback.count > 0
    if method in jac_methods:
        assert grad.count > 0
    if method in hess_methods:
        assert hess.count > 0
    if method in constr_methods:
        assert fun_constraint.count > 0
        assert grad_constraint.count > 0
        np.testing.assert_allclose(res.x[2:], [c0, d0], rtol=rtol)
    else:
        np.testing.assert_allclose(res.x[2:], 0, atol=atol)

    # For methods that support `hessp`, check that it works, too.
    if method in hessp_methods:
        del kwargs['hess']
        kwargs['hessp'] = hessp

        res = cyipopt.minimize_ipopt(fun, x0, method=method, args=(a0,),
                                     kwargs={'b': b0}, **kwargs)
        assert res.success
        assert hessp.count > 0
        np.testing.assert_allclose(res.x[:2], [a0, b0], rtol=rtol)
        if method in constr_methods:
            np.testing.assert_allclose(res.x[2:], [c0, d0], rtol=rtol)
        else:
            np.testing.assert_allclose(res.x[2:], 0, atol=atol)


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid of Scipy available")
def test_minimize_ipopt_bounds_tol_options():
    # spot check additional cases not tested above
    def fun(x):
        return x**2

    x0 = 2.

    # make sure `bounds` is passed to SciPy methods
    bounds = (1, 3)
    res = cyipopt.minimize_ipopt(fun, x0, method='slsqp', bounds=[(1, 3)])
    np.testing.assert_allclose(res.x, bounds[0])

    # make sure `tol` is passed to SciPy methods
    tol1, tol2 = 1e-3, 1e-9
    res1 = cyipopt.minimize_ipopt(fun, x0, method='cobyla', tol=tol1)
    res2 = cyipopt.minimize_ipopt(fun, x0, method='cobyla', tol=tol2)
    assert abs(res2.x[0]) <= tol2 < abs(res1.x[0]) <= tol1

    # make sure `options` is passed to SciPy methods
    res = cyipopt.minimize_ipopt(fun, x0, method='slsqp')
    assert res.nit > 1
    options = dict(maxiter=1)
    res = cyipopt.minimize_ipopt(fun, x0, method='slsqp', options=options)
    assert res.nit == 1


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
        "jac": lambda x: coo_array([2 * x])
    }
    con_ineq = {
        "type": "ineq",
        "fun": lambda x: np.prod(x) - 25,
        "jac": lambda x: coo_array([np.prod(x) / x]),
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
        "jac": lambda x: coo_array([np.prod(x) / x]),
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


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
def test_minimize_ipopt_bounds():
    # Test that `minimize_ipopt` accepts  bounds sequences or `optimize.Bounds`
    from scipy import optimize

    def f(x):
        return x @ x

    # accept size 2 sequence containing same bounds for all variables
    res = cyipopt.minimize_ipopt(f, [2, 3], bounds=[1, 10])
    np.testing.assert_allclose(res.x, [1, 1])

    res = cyipopt.minimize_ipopt(f, [2, 3], bounds=[None, None])
    np.testing.assert_allclose(res.x, [0, 0], atol=1e-6)

    # accept instance of Bounds
    bounds = optimize.Bounds(lb=0.5, ub=[1, 2])
    res = cyipopt.minimize_ipopt(f, [2, 3], bounds=bounds)
    np.testing.assert_allclose(res.x, [0.5, 0.5])


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
def test_minimize_late_binding_bug():
    # `IpoptProblemWrapper` had a late binding bug when constraint Jacobians
    # were defined with `optimize.approx_fprime`. Check that this is resolved.
    from scipy.optimize import minimize

    fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
    cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
            {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
            {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
    bnds = ((0, None), (0, None))

    res = cyipopt.minimize_ipopt(fun, (2, 0), bounds=bnds, constraints=cons)
    ref = minimize(fun, (2, 0), bounds=bnds, constraints=cons)
    assert res.success
    np.testing.assert_allclose(res.x, ref.x)
    np.testing.assert_allclose(res.fun, ref.fun)


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
def test_gh115_eps_option():
    # gh-115 requested that the `eps` argument be exposed as an option. Verify
    # that it is working as advertised (at least for the objective function).
    def f(x):
        if f.x is None:
            f.x = x
        elif f.dx is None:
            f.dx = x - f.x
        return x ** 2

    f.x, f.dx = None, None
    cyipopt.minimize_ipopt(f, x0=0)
    np.testing.assert_equal(f.dx, 1e-8)

    f.x, f.dx = None, None
    eps = 1e-9
    cyipopt.minimize_ipopt(f, x0=0, options={'eps': eps})
    np.testing.assert_equal(f.dx, eps)
