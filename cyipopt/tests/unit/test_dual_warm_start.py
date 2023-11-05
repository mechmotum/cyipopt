import pytest
import numpy as np
from numpy.testing import assert_, assert_allclose
from cyipopt import minimize_ipopt


class TestDualWarmStart:
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
    
    def f_eqcon(self, x, sign=1.0):
        """ Equality constraint """
        return np.array([x[0] - x[1]])

    def f_ieqcon(self, x, sign=1.0):
        """ Inequality constraint """
        return np.array([x[0] - x[1] - 1.0])

    def f_ieqcon2(self, x):
        """ Vector inequality constraint """
        return np.asarray(x)

    def fprime_ieqcon2(self, x):
        """ Vector inequality constraint, derivative """
        return np.identity(x.shape[0])
    
    # minimize
    def test_dual_warm_start_unconstrained_without(self):
        # unconstrained, without warm start.
        res = minimize_ipopt(self.fun, [-1.0, 1.0], args=(-1.0, ),
                       jac=self.jac, method=None, options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1])

    @pytest.mark.xfail(raises=(ValueError,), reason="Initial guesses for dual variables have wrong shape")
    def test_dual_warm_start_unconstrained_with(self):
        # unconstrained, with warm start.
        res = minimize_ipopt(self.fun, [-1.0, 1.0], args=(-1.0, ),
                       jac=self.jac, method=None, options=self.opts, mult_g=[1, 1])
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1])

    def test_dual_warm_start_equality_without(self):
        # equality constraint, without warm start.
        res = minimize_ipopt(self.fun, [-1.0, 1.0], jac=self.jac,
                       method=None, args=(-1.0,),
                       constraints={'type': 'eq', 'fun':self.f_eqcon,
                                    'args': (-1.0, )},
                       options=self.opts)
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [1, 1])

    def test_dual_warm_start_equality_with_right(self):
        # equality constraint, with right warm start.
        res = minimize_ipopt(self.fun, [-1.0, 1.0], jac=self.jac,
                       method=None, args=(-1.0,),
                       constraints={'type': 'eq', 'fun':self.f_eqcon,
                                    'args': (-1.0, )},
                       options=self.opts, mult_g=[1], mult_x_L=[1, 1], mult_x_U=[-1, -1])
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [1, 1])
    
    @pytest.mark.xfail(raises=(ValueError,), reason="Initial guesses for dual variables have wrong shape")
    def test_dual_warm_start_equality_with_wrong_shape(self):
        # equality constraint, with wrong warm start shape.
        res = minimize_ipopt(self.fun, [-1.0, 1.0], jac=self.jac,
                       method=None, args=(-1.0,),
                       constraints={'type': 'eq', 'fun':self.f_eqcon,
                                    'args': (-1.0, )},
                       options=self.opts, mult_g=[1], mult_x_U=[1])
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [1, 1])

    @pytest.mark.xfail(raises=(TypeError,), reason="Initial guesses for dual variables have wrong type")
    def test_dual_warm_start_equality_with_wrong_type(self):
        # equality constraint, with wrong warm start type.
        res = minimize_ipopt(self.fun, [-1.0, 1.0], jac=self.jac,
                       method=None, args=(-1.0,),
                       constraints={'type': 'eq', 'fun':self.f_eqcon,
                                    'args': (-1.0, )},
                       options=self.opts, mult_x_L=[1, 1], mult_x_U=np.array([1, 1]))
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [1, 1])

    def test_dual_warm_start_inequality_with_right(self):
        # inequality constraint, with right warm start.
        res = minimize_ipopt(self.fun, [-1.0, 1.0], method=None,
                       jac=self.jac, args=(-1.0, ),
                       constraints={'type': 'ineq',
                                    'fun': self.f_ieqcon,
                                    'args': (-1.0, )},
                       options=self.opts, mult_x_L=[-1, 1], mult_x_U=[1, 1])
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1], atol=1e-3)

    @pytest.mark.xfail(raises=(ValueError,), reason="Initial guesses for dual variables have wrong shape")
    def test_dual_warm_start_inequality_vec_with_wrong_shape(self):
        # vector inequality constraint, with wrong warm start shape.
        res = minimize_ipopt(self.fun, [-1.0, 1.0], jac=self.jac,
                       method=None, args=(-1.0,),
                       constraints={'type': 'ineq',
                                    'fun': self.f_ieqcon2,
                                    'jac': self.fprime_ieqcon2},
                       options=self.opts, mult_g=[1])
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1])

    @pytest.mark.xfail(raises=(AssertionError,), reason="Initial guesses for dual variables have wrong type")
    def test_dual_warm_start_inequality_vec_with_wrong_element_type(self):
        # vector inequality constraint, with wrong warm start element type.
        res = minimize_ipopt(self.fun, [-1.0, 1.0], jac=self.jac,
                       method=None, args=(-1.0,),
                       constraints={'type': 'ineq',
                                    'fun': self.f_ieqcon2,
                                    'jac': self.fprime_ieqcon2},
                       options=self.opts, mult_g=['1', 1])
        assert_(res['success'], res['message'])
        assert_allclose(res.x, [2, 1])