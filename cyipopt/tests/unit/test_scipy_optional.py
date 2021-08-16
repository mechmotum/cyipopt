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