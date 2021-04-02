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
