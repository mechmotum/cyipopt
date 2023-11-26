"""
Unit tests written for scipy.optimize.minimize method='SLSQP' and
'method='trust-constr'; adapted for `minimize_ipopt`. Original license
from scipy/LICENSE.txt, 79f5cd1ec0a1ba37ad4c295dedb5543d368be438:

Copyright (c) 2001-2002 Enthought, Inc. 2003-2023, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
import sys
import pytest
import numpy as np
from numpy.testing import (TestCase, assert_array_almost_equal,
                           assert_, assert_allclose,
                           suppress_warnings)

try:
    from scipy.optimize import (NonlinearConstraint,
                                LinearConstraint,
                                Bounds)
    from scipy.linalg import block_diag
    from scipy.sparse import csc_matrix
except ImportError:
    pass

from cyipopt import minimize_ipopt as minimize

### Adapted from scipy/scipy/optimize/tests/test_slsqp.py,
### 6c4a3f3551f9ee52a75c8c3999a57bed8ea67deb

class MyCallBack:
    """pass a custom callback function

    This makes sure it's being used.
    """
    def __init__(self):
        self.been_called = False
        self.ncalls = 0

    def __call__(self, x):
        self.been_called = True
        self.ncalls += 1


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
class TestSLSQP:
    """
    Test SLSQP algorithm using Example 14.4 from Numerical Methods for
    Engineers by Steven Chapra and Raymond Canale.
    This example maximizes the function f(x) = 2*x*y + 2*x - x**2 - 2*y**2,
    which has a maximum at x=2, y=1.
    """
    atol = 1e-7

    def setup_method(self):
        self.opts = {'disp': False}

    def fun(self, d, sign=1.0):
        """
        Arguments:
        d     - A list of two elements, where d[0] represents x and d[1] represents y
                 in the following equation.
        sign - A multiplier for f. Since we want to optimize it, and the SciPy
               optimizers can only minimize functions, we need to multiply it by
               -1 to achieve the desired solution
        Returns:
        2*x*y + 2*x - x**2 - 2*y**2

        """
        x = d[0]
        y = d[1]
        return sign*(2*x*y + 2*x - x**2 - 2*y**2)

    def jac(self, d, sign=1.0):
        """
        This is the derivative of fun, returning a NumPy array
        representing df/dx and df/dy.

        """
        x = d[0]
        y = d[1]
        dfdx = sign*(-2*x + 2*y + 2)
        dfdy = sign*(2*x - 4*y)
        return np.array([dfdx, dfdy], float)

    def fun_and_jac(self, d, sign=1.0):
        return self.fun(d, sign), self.jac(d, sign)

    def f_eqcon(self, x, sign=1.0):
        """ Equality constraint """
        return np.array([x[0] - x[1]])

    def fprime_eqcon(self, x, sign=1.0):
        """ Equality constraint, derivative """
        return np.array([[1, -1]])

    def f_eqcon_scalar(self, x, sign=1.0):
        """ Scalar equality constraint """
        return self.f_eqcon(x, sign)[0]

    def fprime_eqcon_scalar(self, x, sign=1.0):
        """ Scalar equality constraint, derivative """
        return self.fprime_eqcon(x, sign)[0].tolist()

    def f_ieqcon(self, x, sign=1.0):
        """ Inequality constraint """
        return np.array([x[0] - x[1] - 1.0])

    def fprime_ieqcon(self, x, sign=1.0):
        """ Inequality constraint, derivative """
        return np.array([[1, -1]])

    def f_ieqcon2(self, x):
        """ Vector inequality constraint """
        return np.asarray(x)

    def fprime_ieqcon2(self, x):
        """ Vector inequality constraint, derivative """
        return np.identity(x.shape[0])

    # minimize
    def test_minimize_unbounded_approximated(self):
        # Minimize, method=None: unbounded, approximated jacobian.
        jacs = [None, False]
        for jac in jacs:
            res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
                           jac=jac, method=None,
                           options=self.opts)
            # NOTE : This test fails on some MacOSX builds with the error:
            # AssertionError: b'Algorithm stopped at a point that was
            # converged, not to "desired" tolerances, but to "acceptable"
            # tolerances (see the acceptable-... options).'
            if not res['success']:
                assert_('"acceptable" tolerances' in
                        res['message'].decode("utf-8"))
            else:
                assert_(res['success'], res['message'])

            assert_allclose(res.x, [2, 1])

    def test_minimize_unbounded_given(self):
        # Minimize, method=None: unbounded, given Jacobian.
        res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
                       jac=self.jac, method=None, options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1])

    def test_minimize_bounded_approximated(self):
        # Minimize, method=None: bounded, approximated jacobian.
        jacs = [None, False]
        for jac in jacs:
            with np.errstate(invalid='ignore'):
                res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
                               jac=jac,
                               bounds=((2.5, None), (None, 0.5)),
                               method=None, options=self.opts)
            assert_(res['success'], res['message'])
            assert_allclose(res.x, [2.5, 0.5])
            assert_(2.5 - self.atol <= res.x[0])
            assert_(res.x[1] - self.atol <= 0.5)

    def test_minimize_unbounded_combined(self):
        # Minimize, method=None: unbounded, combined function and Jacobian.
        res = minimize(self.fun_and_jac, [-1.0, 1.0], args=(-1.0, ),
                       jac=True, method=None, options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1])

    def test_minimize_equality_approximated(self):
        # Minimize with method=None: equality constraint, approx. jacobian.
        jacs = [None, False]
        for jac in jacs:
            res = minimize(self.fun, [-1.0, 1.0], args=(-1.0, ),
                           jac=jac,
                           constraints={'type': 'eq',
                                        'fun': self.f_eqcon,
                                        'args': (-1.0, )},
                           method=None, options=self.opts)
            assert_(res['success'], res['message'])
            assert_allclose(res.x, [1, 1])

    def test_minimize_equality_given(self):
        # Minimize with method=None: equality constraint, given Jacobian.
        res = minimize(self.fun, [-1.0, 1.0], jac=self.jac,
                       method=None, args=(-1.0,),
                       constraints={'type': 'eq', 'fun':self.f_eqcon,
                                    'args': (-1.0, )},
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [1, 1])

    def test_minimize_equality_given2(self):
        # Minimize with method=None: equality constraint, given Jacobian
        # for fun and const.
        res = minimize(self.fun, [-1.0, 1.0], method=None,
                       jac=self.jac, args=(-1.0,),
                       constraints={'type': 'eq',
                                    'fun': self.f_eqcon,
                                    'args': (-1.0, ),
                                    'jac': self.fprime_eqcon},
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [1, 1])

    def test_minimize_equality_given_cons_scalar(self):
        # Minimize with method=None: scalar equality constraint, given
        # Jacobian for fun and const.
        res = minimize(self.fun, [-1.0, 1.0], method=None,
                       jac=self.jac, args=(-1.0,),
                       constraints={'type': 'eq',
                                    'fun': self.f_eqcon_scalar,
                                    'args': (-1.0, ),
                                    'jac': self.fprime_eqcon_scalar},
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [1, 1])

    def test_minimize_inequality_given(self):
        # Minimize with method=None: inequality constraint, given Jacobian.
        res = minimize(self.fun, [-1.0, 1.0], method=None,
                       jac=self.jac, args=(-1.0, ),
                       constraints={'type': 'ineq',
                                    'fun': self.f_ieqcon,
                                    'args': (-1.0, )},
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1], atol=1e-3)

    def test_minimize_inequality_given_vector_constraints(self):
        # Minimize with method=None: vector inequality constraint, given
        # Jacobian.
        res = minimize(self.fun, [-1.0, 1.0], jac=self.jac,
                       method=None, args=(-1.0,),
                       constraints={'type': 'ineq',
                                    'fun': self.f_ieqcon2,
                                    'jac': self.fprime_ieqcon2},
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1])

    def test_minimize_bound_equality_given2(self):
        # Minimize with method=None: bounds, eq. const., given jac. for
        # fun. and const.
        res = minimize(self.fun, [-1.0, 1.0], method=None,
                       jac=self.jac, args=(-1.0, ),
                       bounds=[(-0.8, 1.), (-1, 0.8)],
                       constraints={'type': 'eq',
                                    'fun': self.f_eqcon,
                                    'args': (-1.0, ),
                                    'jac': self.fprime_eqcon},
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [0.8, 0.8], atol=1e-3)
        assert_(-0.8 - self.atol <= res.x[0] <= 1 + self.atol)
        assert_(-1 - self.atol <= res.x[1] <= 0.8 + self.atol)

    def test_inconsistent_linearization(self):
        # SLSQP must be able to solve this problem, even if the
        # linearized problem at the starting point is infeasible.

        # Linearized constraints are
        #
        #    2*x0[0]*x[0] >= 1
        #
        # At x0 = [0, 1], the second constraint is clearly infeasible.
        # This triggers a call with n2==1 in the LSQ subroutine.
        x = [0, 1]
        def f1(x):
            return x[0] + x[1] - 2
        def f2(x):
            return x[0] ** 2 - 1
        sol = minimize(
            lambda x: x[0]**2 + x[1]**2,
            x,
            constraints=({'type':'eq','fun': f1},
                         {'type':'ineq','fun': f2}),
            bounds=((0,None), (0,None)),
            method=None)
        x = sol.x

        assert_allclose(f1(x), 0, atol=1e-8)
        assert_(f2(x) >= -1e-8)
        # assert_(sol.success, sol)
        # "Algorithm stopped at a point that was converged, not to "desired"
        # tolerances, but to "acceptable" tolerances

    def test_regression_5743(self):
        # SLSQP must not indicate success for this problem,
        # which is infeasible.
        x = [1, 2]
        sol = minimize(
            lambda x: x[0]**2 + x[1]**2,
            x,
            constraints=({'type':'eq','fun': lambda x: x[0]+x[1]-1},
                         {'type':'ineq','fun': lambda x: x[0]-2}),
            bounds=((0,None), (0,None)),
            method=None)
        assert_(not sol.success, sol)

    def test_gh_6676(self):
        def func(x):
            return (x[0] - 1)**2 + 2*(x[1] - 1)**2 + 0.5*(x[2] - 1)**2

        sol = minimize(func, [0, 0, 0], method=None)
        # assert_(sol.jac.shape == (3,))
        # minimize_ipopt doesn't return Jacobian

    def test_invalid_bounds(self):
        # Raise correct error when lower bound is greater than upper bound.
        # See Github issue 6875.
        bounds_list = [
            ((1, 2), (2, 1)),
            ((2, 1), (1, 2)),
            ((2, 1), (2, 1)),
            ((np.inf, 0), (np.inf, 0)),
            ((1, -np.inf), (0, 1)),
        ]
        for bounds in bounds_list:
            res = minimize(self.fun, [-1.0, 1.0], bounds=bounds, method=None)
            # "invalid problem definition" or "problem may be infeasible"
            assert res.status == -11 or res.status == 2

    def test_bounds_clipping(self):
        #
        # SLSQP returns bogus results for initial guess out of bounds, gh-6859
        #
        def f(x):
            return (x[0] - 1)**2

        sol = minimize(f, [10], method=None, bounds=[(None, 0)])
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=self.atol)

        sol = minimize(f, [-10], method=None, bounds=[(2, None)])
        assert_(sol.success)
        assert_allclose(sol.x, 2, atol=self.atol)

        sol = minimize(f, [-10], method=None, bounds=[(None, 0)])
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=self.atol)

        sol = minimize(f, [10], method=None, bounds=[(2, None)])
        assert_(sol.success)
        assert_allclose(sol.x, 2, atol=self.atol)

        sol = minimize(f, [-0.5], method=None, bounds=[(-1, 0)])
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=self.atol)

        sol = minimize(f, [10], method=None, bounds=[(-1, 0)])
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=self.atol)

    def test_infeasible_initial(self):
        # Check SLSQP behavior with infeasible initial point
        def f(x):
            x, = x
            return x*x - 2*x + 1

        cons_u = [{'type': 'ineq', 'fun': lambda x: 0 - x}]
        cons_l = [{'type': 'ineq', 'fun': lambda x: x - 2}]
        cons_ul = [{'type': 'ineq', 'fun': lambda x: 0 - x},
                   {'type': 'ineq', 'fun': lambda x: x + 1}]

        sol = minimize(f, [10], method=None, constraints=cons_u)
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=self.atol)

        sol = minimize(f, [-10], method=None, constraints=cons_l)
        assert_(sol.success)
        assert_allclose(sol.x, 2, atol=self.atol)

        sol = minimize(f, [-10], method=None, constraints=cons_u)
        assert_(sol.success)
        assert_allclose(sol.x, 0, atol=self.atol)

        sol = minimize(f, [10], method=None, constraints=cons_l)
        assert_(sol.success)
        assert_allclose(sol.x, 2, atol=self.atol)

    def test_inconsistent_inequalities(self):
        # gh-7618

        def cost(x):
            return -1 * x[0] + 4 * x[1]

        def ineqcons1(x):
            return x[1] - x[0] - 1

        def ineqcons2(x):
            return x[0] - x[1]

        # The inequalities are inconsistent, so no solution can exist:
        #
        # x1 >= x0 + 1
        # x0 >= x1

        x0 = (1,5)
        bounds = ((-5, 5), (-5, 5))
        cons = (dict(type='ineq', fun=ineqcons1), dict(type='ineq', fun=ineqcons2))
        res = minimize(cost, x0, method=None, bounds=bounds, constraints=cons)

        assert_(not res.success)

    def test_new_bounds_type(self):
        def f(x):
            return x[0] ** 2 + x[1] ** 2
        bounds = Bounds([1, 0], [np.inf, np.inf])
        sol = minimize(f, [0, 0], method=None, bounds=bounds)
        assert_(sol.success)
        assert_allclose(sol.x, [1, 0], atol=5e-5)

    def test_nested_minimization(self):
        atol = self.atol

        class NestedProblem():

            def __init__(self):
                self.F_outer_count = 0

            def F_outer(self, x):
                self.F_outer_count += 1
                if self.F_outer_count > 1000:
                    raise Exception("Nested minimization failed to terminate.")
                inner_res = minimize(self.F_inner, (3, 4), method=None)
                assert_(inner_res.success)
                assert_allclose(inner_res.x, [1, 1])
                return x[0]**2 + x[1]**2 + x[2]**2

            def F_inner(self, x):
                return (x[0] - 1)**2 + (x[1] - 1)**2

            def solve(self):
                outer_res = minimize(self.F_outer, (5, 5, 5), method=None)
                assert_(outer_res.success)
                assert_allclose(outer_res.x, [0, 0, 0], atol=atol)

        problem = NestedProblem()
        problem.solve()

    def test_gh1758(self):
        # minimize_ipopt finds this to be infeasible

        # the test suggested in gh1758
        # https://nlopt.readthedocs.io/en/latest/NLopt_Tutorial/
        # implement two equality constraints, in R^2.
        def fun(x):
            return np.sqrt(x[1])

        def f_eqcon(x):
            """ Equality constraint """
            return x[1] - (2 * x[0]) ** 3

        def f_eqcon2(x):
            """ Equality constraint """
            return x[1] - (-x[0] + 1) ** 3

        c1 = {'type': 'eq', 'fun': f_eqcon}
        c2 = {'type': 'eq', 'fun': f_eqcon2}

        res = minimize(fun, [8, 0.25], method=None,
                       constraints=[c1, c2], bounds=[(-0.5, 1), (0, 8)])

        np.testing.assert_allclose(res.fun, 0.5443310539518)
        np.testing.assert_allclose(res.x, [0.33333333, 0.2962963])
        assert res.success

    def test_gh9640(self):
        np.random.seed(10)
        cons = ({'type': 'ineq', 'fun': lambda x: -x[0] - x[1] - 3},
                {'type': 'ineq', 'fun': lambda x: x[1] + x[2] - 2})
        bnds = ((-2, 2), (-2, 2), (-2, 2))

        def target(x):
            return 1
        x0 = [-1.8869783504471584, -0.640096352696244, -0.8174212253407696]
        res = minimize(target, x0, method=None, bounds=bnds, constraints=cons,
                       options={'disp':False, 'maxiter':10000})

        # The problem is infeasible, so it cannot succeed
        assert not res.success


### Adapted from scipy/scipy/optimize/tests/test_minimize_constrained.py,
### 79f5cd1ec0a1ba37ad4c295dedb5543d368be438

class Maratos:
    """Problem 15.4 from Nocedal and Wright

    The following optimization problem:
        minimize 2*(x[0]**2 + x[1]**2 - 1) - x[0]
        Subject to: x[0]**2 + x[1]**2 - 1 = 0
    """

    def __init__(self, degrees=60, constr_jac=None, constr_hess=None):
        rads = degrees/180*np.pi
        self.x0 = [np.cos(rads), np.sin(rads)]
        self.x_opt = np.array([1.0, 0.0])
        self.constr_jac = constr_jac
        self.constr_hess = constr_hess
        self.bounds = None

    def fun(self, x):
        return 2*(x[0]**2 + x[1]**2 - 1) - x[0]

    def grad(self, x):
        return np.array([4*x[0]-1, 4*x[1]])

    def hess(self, x):
        return 4*np.eye(2)

    @property
    def constr(self):
        def fun(x):
            return x[0]**2 + x[1]**2

        if self.constr_jac is None:
            def jac(x):
                return [[2*x[0], 2*x[1]]]
        else:
            jac = self.constr_jac

        if self.constr_hess is None:
            def hess(x, v):
                return 2*v[0]*np.eye(2)
        else:
            hess = self.constr_hess

        return NonlinearConstraint(fun, 1, 1, jac, hess)


class MaratosTestArgs:
    """Problem 15.4 from Nocedal and Wright

    The following optimization problem:
        minimize 2*(x[0]**2 + x[1]**2 - 1) - x[0]
        Subject to: x[0]**2 + x[1]**2 - 1 = 0
    """

    def __init__(self, a, b, degrees=60, constr_jac=None, constr_hess=None):
        rads = degrees/180*np.pi
        self.x0 = [np.cos(rads), np.sin(rads)]
        self.x_opt = np.array([1.0, 0.0])
        self.constr_jac = constr_jac
        self.constr_hess = constr_hess
        self.a = a
        self.b = b
        self.bounds = None

    def _test_args(self, a, b):
        if self.a != a or self.b != b:
            raise ValueError()

    def fun(self, x, a, b):
        self._test_args(a, b)
        return 2*(x[0]**2 + x[1]**2 - 1) - x[0]

    def grad(self, x, a, b):
        self._test_args(a, b)
        return np.array([4*x[0]-1, 4*x[1]])

    def hess(self, x, a, b):
        self._test_args(a, b)
        return 4*np.eye(2)

    @property
    def constr(self):
        def fun(x):
            return x[0]**2 + x[1]**2

        if self.constr_jac is None:
            def jac(x):
                return [[4*x[0], 4*x[1]]]
        else:
            jac = self.constr_jac

        if self.constr_hess is None:
            def hess(x, v):
                return 2*v[0]*np.eye(2)
        else:
            hess = self.constr_hess

        return NonlinearConstraint(fun, 1, 1, jac, hess)


class MaratosGradInFunc:
    """Problem 15.4 from Nocedal and Wright

    The following optimization problem:
        minimize 2*(x[0]**2 + x[1]**2 - 1) - x[0]
        Subject to: x[0]**2 + x[1]**2 - 1 = 0
    """

    def __init__(self, degrees=60, constr_jac=None, constr_hess=None):
        rads = degrees/180*np.pi
        self.x0 = [np.cos(rads), np.sin(rads)]
        self.x_opt = np.array([1.0, 0.0])
        self.constr_jac = constr_jac
        self.constr_hess = constr_hess
        self.bounds = None

    def fun(self, x):
        return (2*(x[0]**2 + x[1]**2 - 1) - x[0],
                np.array([4*x[0]-1, 4*x[1]]))

    @property
    def grad(self):
        return True

    def hess(self, x):
        return 4*np.eye(2)

    @property
    def constr(self):
        def fun(x):
            return x[0]**2 + x[1]**2

        if self.constr_jac is None:
            def jac(x):
                return [[4*x[0], 4*x[1]]]
        else:
            jac = self.constr_jac

        if self.constr_hess is None:
            def hess(x, v):
                return 2*v[0]*np.eye(2)
        else:
            hess = self.constr_hess

        return NonlinearConstraint(fun, 1, 1, jac, hess)


class HyperbolicIneq:
    """Problem 15.1 from Nocedal and Wright

    The following optimization problem:
        minimize 1/2*(x[0] - 2)**2 + 1/2*(x[1] - 1/2)**2
        Subject to: 1/(x[0] + 1) - x[1] >= 1/4
                                   x[0] >= 0
                                   x[1] >= 0
    """
    def __init__(self, constr_jac=None, constr_hess=None):
        self.x0 = [0, 0]
        self.x_opt = [1.952823, 0.088659]
        self.constr_jac = constr_jac
        self.constr_hess = constr_hess
        self.bounds = (0, np.inf)

    def fun(self, x):
        return 1/2*(x[0] - 2)**2 + 1/2*(x[1] - 1/2)**2

    def grad(self, x):
        return [x[0] - 2, x[1] - 1/2]

    def hess(self, x):
        return np.eye(2)

    @property
    def constr(self):
        def fun(x):
            return 1/(x[0] + 1) - x[1]

        if self.constr_jac is None:
            def jac(x):
                return [[-1/(x[0] + 1)**2, -1]]
        else:
            jac = self.constr_jac

        if self.constr_hess is None:
            def hess(x, v):
                return 2*v[0]*np.array([[1/(x[0] + 1)**3, 0],
                                        [0, 0]])
        else:
            hess = self.constr_hess

        return NonlinearConstraint(fun, 0.25, np.inf, jac, hess)


class Rosenbrock:
    """Rosenbrock function.

    The following optimization problem:
        minimize sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    """

    def __init__(self, n=2, random_state=0):
        rng = np.random.RandomState(random_state)
        self.x0 = rng.uniform(-1, 1, n)
        self.x_opt = np.ones(n)
        self.bounds = None

    def fun(self, x):
        x = np.asarray(x)
        r = np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0,
                   axis=0)
        return r

    def grad(self, x):
        x = np.asarray(x)
        xm = x[1:-1]
        xm_m1 = x[:-2]
        xm_p1 = x[2:]
        der = np.zeros_like(x)
        der[1:-1] = (200 * (xm - xm_m1**2) -
                     400 * (xm_p1 - xm**2) * xm - 2 * (1 - xm))
        der[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
        der[-1] = 200 * (x[-1] - x[-2]**2)
        return der

    def hess(self, x):
        x = np.atleast_1d(x)
        H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
        diagonal = np.zeros(len(x), dtype=x.dtype)
        diagonal[0] = 1200 * x[0]**2 - 400 * x[1] + 2
        diagonal[-1] = 200
        diagonal[1:-1] = 202 + 1200 * x[1:-1]**2 - 400 * x[2:]
        H = H + np.diag(diagonal)
        return H

    @property
    def constr(self):
        return ()


class IneqRosenbrock(Rosenbrock):
    """Rosenbrock subject to inequality constraints.

    The following optimization problem:
        minimize sum(100.0*(x[1] - x[0]**2)**2.0 + (1 - x[0])**2)
        subject to: x[0] + 2 x[1] <= 1

    Taken from matlab ``fmincon`` documentation.
    """
    def __init__(self, random_state=0):
        Rosenbrock.__init__(self, 2, random_state)
        self.x0 = [-1, -0.5]
        self.x_opt = [0.5022, 0.2489]
        self.bounds = None

    @property
    def constr(self):
        A = [[1, 2]]
        b = 1
        return LinearConstraint(A, -np.inf, b)


class BoundedRosenbrock(Rosenbrock):
    """Rosenbrock subject to inequality constraints.

    The following optimization problem:
        minimize sum(100.0*(x[1] - x[0]**2)**2.0 + (1 - x[0])**2)
        subject to:  -2 <= x[0] <= 0
                      0 <= x[1] <= 2

    Taken from matlab ``fmincon`` documentation.
    """
    def __init__(self, random_state=0):
        Rosenbrock.__init__(self, 2, random_state)
        self.x0 = [-0.2, 0.2]
        self.x_opt = None
        self.bounds = [(-2, 0), (0, 2)]


class EqIneqRosenbrock(Rosenbrock):
    """Rosenbrock subject to equality and inequality constraints.

    The following optimization problem:
        minimize sum(100.0*(x[1] - x[0]**2)**2.0 + (1 - x[0])**2)
        subject to: x[0] + 2 x[1] <= 1
                    2 x[0] + x[1] = 1

    Taken from matlab ``fimincon`` documentation.
    """
    def __init__(self, random_state=0):
        Rosenbrock.__init__(self, 2, random_state)
        self.x0 = [-1, -0.5]
        self.x_opt = [0.41494, 0.17011]
        self.bounds = None

    @property
    def constr(self):
        A_ineq = [[1, 2]]
        b_ineq = 1
        A_eq = [[2, 1]]
        b_eq = 1
        return (LinearConstraint(A_ineq, -np.inf, b_ineq),
                LinearConstraint(A_eq, b_eq, b_eq))


class Elec:
    """Distribution of electrons on a sphere.

    Problem no 2 from COPS collection [2]_. Find
    the equilibrium state distribution (of minimal
    potential) of the electrons positioned on a
    conducting sphere.

    References
    ----------
    .. [1] E. D. Dolan, J. J. Mor\'{e}, and T. S. Munson,
           "Benchmarking optimization software with COPS 3.0.",
            Argonne National Lab., Argonne, IL (US), 2004.
    """
    def __init__(self, n_electrons=200, random_state=0,
                 constr_jac=None, constr_hess=None):
        self.n_electrons = n_electrons
        self.rng = np.random.RandomState(random_state)
        # Initial Guess
        phi = self.rng.uniform(0, 2 * np.pi, self.n_electrons)
        theta = self.rng.uniform(-np.pi, np.pi, self.n_electrons)
        x = np.cos(theta) * np.cos(phi)
        y = np.cos(theta) * np.sin(phi)
        z = np.sin(theta)
        self.x0 = np.hstack((x, y, z))
        self.x_opt = None
        self.constr_jac = constr_jac
        self.constr_hess = constr_hess
        self.bounds = None

    def _get_cordinates(self, x):
        x_coord = x[:self.n_electrons]
        y_coord = x[self.n_electrons:2 * self.n_electrons]
        z_coord = x[2 * self.n_electrons:]
        return x_coord, y_coord, z_coord

    def _compute_coordinate_deltas(self, x):
        x_coord, y_coord, z_coord = self._get_cordinates(x)
        dx = x_coord[:, None] - x_coord
        dy = y_coord[:, None] - y_coord
        dz = z_coord[:, None] - z_coord
        return dx, dy, dz

    def fun(self, x):
        dx, dy, dz = self._compute_coordinate_deltas(x)
        with np.errstate(divide='ignore'):
            dm1 = (dx**2 + dy**2 + dz**2) ** -0.5
        dm1[np.diag_indices_from(dm1)] = 0
        return 0.5 * np.sum(dm1)

    def grad(self, x):
        dx, dy, dz = self._compute_coordinate_deltas(x)

        with np.errstate(divide='ignore'):
            dm3 = (dx**2 + dy**2 + dz**2) ** -1.5
        dm3[np.diag_indices_from(dm3)] = 0

        grad_x = -np.sum(dx * dm3, axis=1)
        grad_y = -np.sum(dy * dm3, axis=1)
        grad_z = -np.sum(dz * dm3, axis=1)

        return np.hstack((grad_x, grad_y, grad_z))

    def hess(self, x):
        dx, dy, dz = self._compute_coordinate_deltas(x)
        d = (dx**2 + dy**2 + dz**2) ** 0.5

        with np.errstate(divide='ignore'):
            dm3 = d ** -3
            dm5 = d ** -5

        i = np.arange(self.n_electrons)
        dm3[i, i] = 0
        dm5[i, i] = 0

        Hxx = dm3 - 3 * dx**2 * dm5
        Hxx[i, i] = -np.sum(Hxx, axis=1)

        Hxy = -3 * dx * dy * dm5
        Hxy[i, i] = -np.sum(Hxy, axis=1)

        Hxz = -3 * dx * dz * dm5
        Hxz[i, i] = -np.sum(Hxz, axis=1)

        Hyy = dm3 - 3 * dy**2 * dm5
        Hyy[i, i] = -np.sum(Hyy, axis=1)

        Hyz = -3 * dy * dz * dm5
        Hyz[i, i] = -np.sum(Hyz, axis=1)

        Hzz = dm3 - 3 * dz**2 * dm5
        Hzz[i, i] = -np.sum(Hzz, axis=1)

        H = np.vstack((
            np.hstack((Hxx, Hxy, Hxz)),
            np.hstack((Hxy, Hyy, Hyz)),
            np.hstack((Hxz, Hyz, Hzz))
        ))

        return H

    @property
    def constr(self):
        def fun(x):
            x_coord, y_coord, z_coord = self._get_cordinates(x)
            return x_coord**2 + y_coord**2 + z_coord**2 - 1

        if self.constr_jac is None:
            def jac(x):
                x_coord, y_coord, z_coord = self._get_cordinates(x)
                Jx = 2 * np.diag(x_coord)
                Jy = 2 * np.diag(y_coord)
                Jz = 2 * np.diag(z_coord)
                return csc_matrix(np.hstack((Jx, Jy, Jz)))
        else:
            jac = self.constr_jac

        if self.constr_hess is None:
            def hess(x, v):
                D = 2 * np.diag(v)
                return block_diag(D, D, D)
        else:
            hess = self.constr_hess

        return NonlinearConstraint(fun, -np.inf, 0, jac, hess)


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
class TestTrustRegionConstr():
    list_of_problems = [Maratos(),
                        MaratosGradInFunc(),
                        HyperbolicIneq(),
                        Rosenbrock(),
                        IneqRosenbrock(),
                        EqIneqRosenbrock(),
                        BoundedRosenbrock(),
                        Elec(n_electrons=2)]
    @pytest.mark.slow
    @pytest.mark.parametrize("prob", list_of_problems)
    def test_list_of_problems(self, prob):

        for grad in (prob.grad, False):
            if prob == self.list_of_problems[1]:  # MaratosGradInFunc
                grad = True
            for hess in (None,):
                result = minimize(prob.fun, prob.x0,
                                  method=None,
                                  jac=grad, hess=hess,
                                  bounds=prob.bounds,
                                  constraints=prob.constr)

                if prob.x_opt is None:
                    ref = minimize(prob.fun, prob.x0,
                                   method='trust-constr',
                                   jac=grad, hess=hess,
                                   bounds=prob.bounds,
                                   constraints=prob.constr)
                    ref_x_opt = ref.x
                else:
                    ref_x_opt = prob.x_opt

                try:  # figure out how to clean this up
                    res_f_opt = prob.fun(result.x)[0]
                    ref_f_opt = prob.fun(ref_x_opt)[0]
                except IndexError:
                    res_f_opt = prob.fun(result.x)
                    ref_f_opt = prob.fun(ref_x_opt)
                ref_f_opt = ref_f_opt if np.size(ref_f_opt) == 1 else ref_f_opt[0]
                pass1 = np.allclose(result.x, ref_x_opt, atol=5e-4)
                pass2 =  res_f_opt < ref_f_opt + 1e-6
                assert pass1 or pass2

    def test_default_jac_and_hess(self):
        def fun(x):
            return (x - 1) ** 2
        bounds = [(-2, 2)]
        res = minimize(fun, x0=[-1.5], bounds=bounds, method=None)
        assert_array_almost_equal(res.x, 1, decimal=5)

    def test_default_hess(self):
        def fun(x):
            return (x - 1) ** 2

        def jac(x):
            return 2*(x-1)

        bounds = [(-2, 2)]
        res = minimize(fun, x0=[-1.5], bounds=bounds, method=None,
                       jac=jac)
        assert_array_almost_equal(res.x, 1, decimal=5)

    def test_no_constraints(self):
        prob = Rosenbrock()
        result = minimize(prob.fun, prob.x0,
                          method=None,
                          jac=prob.grad)
        assert_array_almost_equal(result.x, prob.x_opt, decimal=5)

    def test_args(self):
        prob = MaratosTestArgs("a", 234)

        result = minimize(prob.fun, prob.x0, ("a", 234),
                          method=None,
                          jac=prob.grad,
                          bounds=prob.bounds,
                          constraints=prob.constr)

        assert_array_almost_equal(result.x, prob.x_opt, decimal=2)

        assert result.success

    def test_issue_9044(self):
        # https://github.com/scipy/scipy/issues/9044
        # Test the returned `OptimizeResult` contains keys consistent with
        # other solvers.

        result = minimize(lambda x: x**2, [0], jac=lambda x: 2*x,
                          hess=lambda x: 2, method=None)
        assert_(result.get('success'))
        assert result.get('nit', -1) == 0


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
class TestEmptyConstraint(TestCase):
    """
    Here we minimize x^2+y^2 subject to x^2-y^2>1.
    The actual minimum is at (0, 0) which fails the constraint.
    Therefore we will find a minimum on the boundary at (+/-1, 0).

    When minimizing on the boundary, optimize uses a set of
    constraints that removes the constraint that sets that
    boundary.  In our case, there's only one constraint, so
    the result is an empty constraint.

    This tests that the empty constraint works.
    """
    def test_empty_constraint(self):

        def function(x):
            return x[0]**2 + x[1]**2

        def functionjacobian(x):
            return np.array([2.*x[0], 2.*x[1]])

        def constraint(x):
            return np.array([x[0]**2 - x[1]**2])

        def constraintjacobian(x):
            return np.array([[2*x[0], -2*x[1]]])

        def constraintlcoh(x, v):
            return np.array([[2., 0.], [0., -2.]]) * v[0]

        constraint = NonlinearConstraint(constraint, 1., np.inf, constraintjacobian, constraintlcoh)

        startpoint = [1., 2.]

        bounds = Bounds([-np.inf, -np.inf], [np.inf, np.inf])

        result = minimize(
          function,
          startpoint,
          method=None,
          jac=functionjacobian,
          constraints=[constraint],
          bounds=bounds,
        )

        assert_array_almost_equal(abs(result.x), np.array([1, 0]), decimal=4)


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
def test_bug_11886():
    def opt(x):
        return x[0]**2+x[1]**2

    with np.testing.suppress_warnings() as sup:
        sup.filter(PendingDeprecationWarning)
        A = np.matrix(np.diag([1, 1]))
    lin_cons = LinearConstraint(A, -1, np.inf)
    minimize(opt, 2*[1], constraints = lin_cons)  # just checking that there are no errors


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
def test_gh11649():
    bnds = Bounds(lb=[-1, -1], ub=[1, 1], keep_feasible=True)

    def assert_inbounds(x):
        assert np.all(x >= bnds.lb)
        assert np.all(x <= bnds.ub)

    def obj(x):
        return np.exp(x[0])*(4*x[0]**2 + 2*x[1]**2 + 4*x[0]*x[1] + 2*x[1] + 1)

    def nce(x):
        return x[0]**2 + x[1]

    def nci(x):
        return x[0]*x[1]

    x0 = np.array((0.99, -0.99))
    nlcs = [NonlinearConstraint(nci, -10, np.inf),
            NonlinearConstraint(nce, 1, 1)]

    res = minimize(fun=obj, x0=x0, method=None,
                   bounds=bnds, constraints=nlcs)
    assert res.success
    assert_inbounds(res.x)
    assert nlcs[0].lb < nlcs[0].fun(res.x) < nlcs[0].ub
    assert_allclose(nce(res.x), nlcs[1].ub)

    ref = minimize(fun=obj, x0=x0, method='slsqp',
                   bounds=bnds, constraints=nlcs)
    assert ref.success
    assert res.fun <= ref.fun  # Ipopt legitimately does better than slsqp here

    # If we give SLSQP a good guess, it agrees with Ipopt
    ref2 = minimize(fun=obj, x0=res.x, method='slsqp',
                   bounds=bnds, constraints=nlcs)
    assert ref2.success
    assert_allclose(ref2.fun, res.fun)


@pytest.mark.skipif("scipy" not in sys.modules,
                    reason="Test only valid if Scipy available.")
class TestBoundedNelderMead:
    atol = 1e-7

    @pytest.mark.parametrize('case_no', range(4))
    def test_rosen_brock_with_bounds(self, case_no):
        cases = [(Bounds(-np.inf, np.inf), Rosenbrock().x_opt),
                 (Bounds(-np.inf, -0.8), [-0.8, -0.8]),
                 (Bounds(3.0, np.inf), [3.0, 9.0]),
                 (Bounds([3.0, 1.0], [4.0, 5.0]), [3., 5.])]
        bounds, x_opt = cases[case_no]
        prob = Rosenbrock()
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "Initial guess is not within "
                                    "the specified bounds")
            result = minimize(prob.fun, [-10, -10],
                              method=None,
                              bounds=bounds)
            assert np.less_equal(bounds.lb - self.atol, result.x).all()
            assert np.less_equal(result.x, bounds.ub + self.atol).all()
            assert np.allclose(prob.fun(result.x), result.fun)
            assert np.allclose(result.x, x_opt, atol=1.e-3)

    @pytest.mark.xfail
    def test_equal_all_bounds(self):
        prob = Rosenbrock()
        bounds = Bounds([4.0, 5.0], [4.0, 5.0])
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "Initial guess is not within "
                                    "the specified bounds")
            result = minimize(prob.fun, [-10, 8],
                              method=None,
                              bounds=bounds)
            assert np.allclose(result.x, [4.0, 5.0])

    def test_equal_one_bounds(self):
        prob = Rosenbrock()
        bounds = Bounds([4.0, 5.0], [4.0, 20.0])
        with suppress_warnings() as sup:
            sup.filter(UserWarning, "Initial guess is not within "
                                    "the specified bounds")
            result = minimize(prob.fun, [-10, 8],
                              method=None,
                              bounds=bounds)
            assert np.allclose(result.x, [4.0, 16.0])

    def test_invalid_bounds(self):
        prob = Rosenbrock()
        message = 'An upper bound is less than the corresponding lower bound.'
        bounds = Bounds([-np.inf, 1.0], [4.0, -5.0])
        res = minimize(prob.fun, [-10, 3], method=None, bounds=bounds)
        assert res.status == -11 or res.status == 2

    def test_outside_bounds_warning(self):
        prob = Rosenbrock()
        bounds = Bounds([-np.inf, 1.0], [4.0, 5.0])
        res = minimize(prob.fun, [-10, 8],
                 method=None,
                 bounds=bounds)
        assert res.success
        assert_allclose(res.x, prob.x_opt, rtol=5e-4)
