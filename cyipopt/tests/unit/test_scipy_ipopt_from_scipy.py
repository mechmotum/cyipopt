"""
Unit tests written for scipy.optimize.minimize method='SLSQP';
adapted for minimize_ipopt
"""
import sys
import pytest
from numpy.testing import (assert_, assert_allclose)
import numpy as np

try:
    from scipy.optimize import Bounds
except ImportError:
    pass

from cyipopt import minimize_ipopt as minimize


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

    # def test_gh1758(self):
    #     # minimize_ipopt finds this to be infeasible
    #
    #     # the test suggested in gh1758
    #     # https://nlopt.readthedocs.io/en/latest/NLopt_Tutorial/
    #     # implement two equality constraints, in R^2.
    #     def fun(x):
    #         return np.sqrt(x[1])
    #
    #     def f_eqcon(x):
    #         """ Equality constraint """
    #         return x[1] - (2 * x[0]) ** 3
    #
    #     def f_eqcon2(x):
    #         """ Equality constraint """
    #         return x[1] - (-x[0] + 1) ** 3
    #
    #     c1 = {'type': 'eq', 'fun': f_eqcon}
    #     c2 = {'type': 'eq', 'fun': f_eqcon2}
    #
    #     res = minimize(fun, [8, 0.25], method=None,
    #                    constraints=[c1, c2], bounds=[(-0.5, 1), (0, 8)])
    #
    #     np.testing.assert_allclose(res.fun, 0.5443310539518)
    #     np.testing.assert_allclose(res.x, [0.33333333, 0.2962963])
    #     assert res.success

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
