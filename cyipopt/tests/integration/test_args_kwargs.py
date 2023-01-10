"""Test passing extra arguments (*args and **kwargs) to objective function and its jac and hess.

Here, these extra arguments are NumPy arrays `dataX` and `dataY`
used to fit a linear regression `Y ~ a + b * X` using least squares.

NOTE: The mandatory `kwarg` keyword argument in functions below exists
solely to trigger an exception if not passed from `cyipopt.minimize_ipopt`.
"""
import pytest

import numpy as np
import cyipopt

def objective_with_args_kwargs(x: np.ndarray, dataX: np.ndarray, dataY: np.ndarray, *, kwarg):
    "Least squares loss for a linear regression `Y ~ a + b * X`"
    a, b = x
    return ( (dataY - a * dataX - b)**2 ).mean()

def jac_with_args_kwargs(x: np.ndarray, dataX: np.ndarray, dataY: np.ndarray, *, kwarg):
    # BOTH `dataX` and `dataY` are used!
    a, b = x
    return np.array([
        ( 2 * (dataY - a * dataX - b) * -dataX ).mean(),
        ( 2 * (dataY - a * dataX - b) * -1     ).mean()
    ])

def hess_with_args_kwargs(x: np.ndarray, dataX: np.ndarray, dataY: np.ndarray, *, kwarg):
    # Only `dataX` is used, but `dataY` must be provided as well
    # to be consistent with `objective_with_args_kwargs` and `jac_with_args_kwargs`.
    dataX_mean = dataX.mean()
    return np.array([
        [2 * (dataX**2).mean(), 2 * dataX_mean],
        [2 * dataX_mean, 2.0]
    ])

def test_args_kwargs():
    """Practical example of when args and kwargs are useful.
    """
    # 1. Define data points.
    dataX = np.asfarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dataY = np.asfarray([1, 2, 5, 3, 5, 4, 7, 5, 3, 8])

    # 2. Minimize loss function.
    x0 = np.asfarray([2.0, 4.0])
    res = cyipopt.minimize_ipopt(
        objective_with_args_kwargs, x0,
        args=(dataX, dataY), kwargs={'kwarg': 0},
        jac=jac_with_args_kwargs,
        hess=hess_with_args_kwargs
    )
    print(res)

    # 3. Check solution against known values.
    # Expected values calculated using Mathematica's `NMinimize`.
    assert res.success is True
    expected_fun = 2.22181818181818181818181818182
    expected_x = np.asfarray([0.490909090909090909090909090909, 1.6])
    np.testing.assert_allclose(res.fun, expected_fun)
    np.testing.assert_allclose(res.x, expected_x)

class TestMissingArgsKwargs:
    x0 = np.asfarray([2.0, 4.0])

    def test_missing_everything(self):
        with pytest.raises(TypeError):
            cyipopt.minimize_ipopt(
                objective_with_args_kwargs, self.x0,
                # Missing args AND mandatory kwargs.
                # args=(dataX, dataY), kwargs={'kwarg': 0},
                jac=jac_with_args_kwargs,
                hess=hess_with_args_kwargs
            )

    def test_missing_args(self):
        with pytest.raises(TypeError):
            cyipopt.minimize_ipopt(
                objective_with_args_kwargs, self.x0,
                # Missing args.
                args=(), kwargs={'kwarg': 0},
                jac=jac_with_args_kwargs,
                hess=hess_with_args_kwargs
            )

    def test_missing_kwargs(self):
        with pytest.raises(TypeError):
            cyipopt.minimize_ipopt(
                objective_with_args_kwargs, self.x0,
                # Missing kwargs.
                args=(2, 3), kwargs={},
                jac=jac_with_args_kwargs,
                hess=hess_with_args_kwargs
            )

    def test_incorrect_number_of_args(self):
        with pytest.raises(TypeError):
            cyipopt.minimize_ipopt(
                objective_with_args_kwargs, self.x0,
                # Need two args, but passed only one.
                args=(2, ), kwargs={'kwarg': 0},
                jac=jac_with_args_kwargs,
                hess=hess_with_args_kwargs
            )
