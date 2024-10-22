# -*- coding: utf-8 -*-
"""
cyipopt: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012-2015 Amit Aides
Copyright (C) 2015-2017 Matthias Kümmerer
Copyright (C) 2017-2024 cyipopt developers

License: EPL 2.0
"""

import sys

import numpy as np
try:
    import scipy
except ImportError:  # scipy is not installed
    SCIPY_INSTALLED = False
else:
    SCIPY_INSTALLED = True
    del scipy
    from scipy import optimize
    import scipy.sparse
    try:
        from scipy.optimize import OptimizeResult
    except ImportError:
        # in scipy 0.14 Result was renamed to OptimizeResult
        from scipy.optimize import Result
        OptimizeResult = Result
    try:
        # MemoizeJac has been made a private class, see
        # https://github.com/scipy/scipy/issues/17572
        from scipy.optimize._optimize import MemoizeJac
    except ImportError:
        from scipy.optimize.optimize import MemoizeJac
    try:
        from scipy.sparse import coo_array
    except ImportError:
        # coo_array was introduced with scipy 1.8
        from scipy.sparse import coo_matrix as coo_array

import cyipopt


class IpoptProblemWrapper(object):
    """Class used to map a scipy minimize definition to a cyipopt problem.

    Parameters
    ==========
    fun : callable
        The objective function to be minimized: ``fun(x, *args, **kwargs) ->
        float``.
    args : tuple, optional
        Extra arguments passed to the objective function and its derivatives
        (``fun``, ``jac``, ``hess``).
    kwargs : :py:class:`dict`, optional
        Extra keyword arguments passed to the objective function and its
        derivatives (``fun``, ``jac``, ``hess``).
    jac : callable, optional
        The Jacobian (gradient) of the objective function:
        ``jac(x, *args, **kwargs) -> ndarray, shape(n, )``.
        If ``None``, SciPy's ``approx_fprime`` is used.
    hess : callable, optional
        If ``None``, the Hessian is computed using IPOPT's numerical methods.
        Explicitly defined Hessians are not yet supported for this class.
    hessp : callable, optional
        If ``None``, the Hessian is computed using IPOPT's numerical methods.
        Explicitly defined Hessians are not yet supported for this class.
    constraints : {Constraint, :py:class:`dict`} or List of {Constraint, :py:class:`dict`}, optional
        See :py:func:`scipy.optimize.minimize` for more information. Note that
        the jacobian of each constraint corresponds to the `'jac'` key and must
        be a callable function with signature ``jac(x) -> {ndarray,
        coo_array}``. If the constraint's value of `'jac'` is a boolean and
        True, the constraint function `fun` is expected to return a tuple
        `(con_val, con_jac)` consisting of the evaluated constraint `con_val`
        and the evaluated jacobian `con_jac`.
    eps : float, optional
        Step size used in finite difference approximations of the objective
        function gradient and constraint Jacobian.
    con_dims : array_like, optional
        Dimensions p_1, ..., p_m of the m constraint functions
        g_1, ..., g_m : R^n -> R^(p_i).
    sparse_jacs: array_like, optional
        If sparse_jacs[i] = True, the i-th constraint's jacobian is sparse.
        Otherwise, the i-th constraint jacobian is assumed to be dense.
    jac_nnz_row: array_like, optional
        The row indices of the nonzero elements in the stacked
        constraint jacobian matrix
    jac_nnz_col: array_like, optional
        The column indices of the nonzero elements in the stacked
        constraint jacobian matrix
    """

    def __init__(self,
                 fun,
                 args=(),
                 kwargs=None,
                 jac=None,
                 hess=None,
                 hessp=None,
                 constraints=(),
                 eps=1e-8,
                 con_dims=(),
                 sparse_jacs=(),
                 jac_nnz_row=(),
                 jac_nnz_col=()):
        if not SCIPY_INSTALLED:
            msg = 'Install SciPy to use the `IpoptProblemWrapper` class.'
            raise ImportError()
        self.obj_hess = None
        self.last_x = None

        # Input validation of user-provided arguments
        if fun is not None and not callable(fun):
            raise ValueError('`fun` must be callable.')
        if not isinstance(args, tuple):
            args = (args,)
        kwargs = dict() if kwargs is None else kwargs
        if not isinstance(kwargs, dict):
            raise ValueError('`kwargs` must be a dictionary.')
        if jac is not None and jac not in {True, False} and not callable(jac):
            raise ValueError('`jac` must be callable or boolean.')
        if hess is not None and not callable(hess):
            raise ValueError('`hess` must be callable.')
        if hessp is not None:
            raise NotImplementedError(
                '`hessp` is not yet supported by Ipopt.`')
        # TODO: add input validation for `constraints` when adding
        #  support for instances of new-style constraints (e.g.
        #  `NonlinearConstraint`) and sequences of constraints.

        if hess is not None:
            self.obj_hess = hess
        if not jac:
            def jac(x, *args, **kwargs):
                def wrapped_fun(x):
                    return fun(x, *args, **kwargs)
                return optimize.approx_fprime(x, wrapped_fun, eps)
        elif jac is True:
            fun = MemoizeJac(fun)
            jac = fun.derivative

        self.fun = fun
        self.jac = jac
        self.args = args
        self.kwargs = kwargs or {}
        self._constraint_funs = []
        self._constraint_jacs = []
        self._constraint_hessians = []
        self._constraint_dims = np.asarray(con_dims)
        self._constraint_args = []
        self._constraint_kwargs = []
        self._constraint_jac_is_sparse = sparse_jacs
        self._constraint_jacobian_structure = (jac_nnz_row, jac_nnz_col)
        if isinstance(constraints, dict):
            constraints = (constraints, )
        for con in constraints:
            con_fun = con['fun']
            con_jac = con.get('jac', None)
            con_args = con.get('args', [])
            con_hessian = con.get('hess', None)
            con_kwargs = con.get('kwargs', {})
            if con_jac is None:
                # beware of late binding!
                def con_jac(x, *args, con_fun=con_fun, **kwargs):
                    def wrapped(x):
                        return con_fun(x, *args, **kwargs)
                    return optimize.approx_fprime(x, wrapped, eps)
            elif con_jac is True:
                con_fun = MemoizeJac(con_fun)
                con_jac = con_fun.derivative
            elif not callable(con_jac):
                raise NotImplementedError('jac has to be bool or a function')
            if (self.obj_hess is not None
                    and con_hessian is None) or (self.obj_hess is None
                                                 and con_hessian is not None):
                msg = "hessian has to be provided for the objective and all constraints"
                raise NotImplementedError(msg)
            self._constraint_funs.append(con_fun)
            self._constraint_jacs.append(con_jac)
            self._constraint_hessians.append(con_hessian)
            self._constraint_args.append(con_args)
            self._constraint_kwargs.append(con_kwargs)
        # Set up evaluation counts
        self.nfev = 0
        self.njev = 0
        self.nit = 0

    def evaluate_fun_with_grad(self, x):
        """ For backwards compatibility. """
        return (self.objective(x), self.gradient(x, **self.kwargs))

    def objective(self, x):
        self.nfev += 1
        return self.fun(x, *self.args, **self.kwargs)

    # TODO : **kwargs is ignored, not sure why it is here.
    def gradient(self, x, **kwargs):
        self.njev += 1
        return self.jac(x, *self.args, **self.kwargs)  # .T

    def constraints(self, x):
        con_values = []
        for fun, args, kwargs in zip(self._constraint_funs,
                                     self._constraint_args,
                                     self._constraint_kwargs):
            con_values.append(fun(x, *args, **kwargs))
        return np.hstack(con_values)

    def jacobianstructure(self):
        return self._constraint_jacobian_structure

    def jacobian(self, x):
        # Convert all dense constraint jacobians to sparse ones.
        # The structure ( = row and column indices) is already known at this point,
        # so we only need to stack the evaluated jacobians
        jac_values = []
        for i, (jac, args, kwargs) in enumerate(zip(self._constraint_jacs,
                                                    self._constraint_args,
                                                    self._constraint_kwargs)):
            if self._constraint_jac_is_sparse[i]:
                jac_val = jac(x, *args, **kwargs)
                jac_values.append(jac_val.data)
            else:
                dense_jac_val = np.atleast_2d(jac(x, *args, **kwargs))
                jac_values.append(dense_jac_val.ravel())
        return np.hstack(jac_values)

    def hessian(self, x, lagrange, obj_factor):
        H = obj_factor * self.obj_hess(x, *self.args, **self.kwargs)  # type: ignore
        # split the lagrangian multipliers for each constraint hessian
        lagrs = np.split(lagrange, np.cumsum(self._constraint_dims[:-1]))
        for hessian, args, kwargs, lagr in zip(self._constraint_hessians,
                                               self._constraint_args,
                                               self._constraint_kwargs, lagrs):
            H += hessian(x, lagr, *args, **kwargs)
        return H[np.tril_indices(x.size)]

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                     d_norm, regularization_size, alpha_du, alpha_pr,
                     ls_trials):

        self.nit = iter_count


def get_bounds(bounds):
    if bounds is None:
        return None, None
    else:
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]
        return lb, ub


def _get_sparse_jacobian_structure(constraints, x0):
    con_jac_is_sparse = []
    jacobians = []
    x0 = np.asarray(x0)
    if isinstance(constraints, dict):
        constraints = (constraints, )
    if len(constraints) == 0:
        return [], [], []
    for con in constraints:
        con_jac = con.get('jac', False)
        if con_jac:
            if isinstance(con_jac, bool):
                _, jac_val = con['fun'](x0, *con.get('args', []),
                                        **con.get('kwargs', {}))
            else:
                jac_val = con_jac(x0, *con.get('args', []),
                                  **con.get('kwargs', {}))
            # check if dense or sparse
            if isinstance(jac_val, coo_array):
                jacobians.append(jac_val)
                con_jac_is_sparse.append(True)
            else:
                # Creating the coo_array from jac_val would yield to
                # wrong dimensions if some values in jac_val are zero,
                # so we assume all values in jac_val are nonzero
                jacobians.append(coo_array(np.ones_like(np.atleast_2d(jac_val))))
                con_jac_is_sparse.append(False)
        else:
            # we approximate this jacobian later (=dense)
            con_val = np.atleast_1d(con['fun'](x0, *con.get('args', []),
                                               **con.get('kwargs', {})))
            jacobians.append(coo_array(np.ones((con_val.size, x0.size))))
            con_jac_is_sparse.append(False)
    J = scipy.sparse.vstack(jacobians)
    return con_jac_is_sparse, J.row, J.col


def get_constraint_dimensions(constraints, x0):
    con_dims = []
    if isinstance(constraints, dict):
        constraints = (constraints, )
    for con in constraints:
        if con.get('jac', False) is True:
            m = len(np.atleast_1d(con['fun'](x0, *con.get('args', []),
                                             **con.get('kwargs', {}))[0]))
        else:
            m = len(np.atleast_1d(con['fun'](x0, *con.get('args', []),
                                             **con.get('kwargs', {}))))
        con_dims.append(m)
    return np.array(con_dims)


def get_constraint_bounds(constraints, x0, INF=1e19):
    cl = []
    cu = []
    if isinstance(constraints, dict):
        constraints = (constraints, )
    for con in constraints:
        if con.get('jac', False) is True:
            m = len(np.atleast_1d(con['fun'](x0, *con.get('args', []),
                                             **con.get('kwargs', {}))[0]))
        else:
            m = len(np.atleast_1d(con['fun'](x0, *con.get('args', []),
                                             **con.get('kwargs', {}))))
        cl.extend(np.zeros(m))
        if con['type'] == 'eq':
            cu.extend(np.zeros(m))
        elif con['type'] == 'ineq':
            cu.extend(INF * np.ones(m))
        else:
            raise ValueError(con['type'])
    cl = np.array(cl)
    cu = np.array(cu)

    return cl, cu


def replace_option(options, oldname, newname):
    if oldname in options:
        if newname not in options:
            options[newname] = options.pop(oldname)


def convert_to_bytes(options):
    if sys.version_info >= (3, 0):
        for key in list(options.keys()):
            try:
                if bytes(key, 'utf-8') != key:
                    options[bytes(key, 'utf-8')] = options[key]
                    options.pop(key)
            except TypeError:
                pass


def _wrap_fun(fun, kwargs):
    if callable(fun) and kwargs:
        def new_fun(x, *args):
            return fun(x, *args, **kwargs)
    else:
        new_fun = fun
    return new_fun

def _wrap_funs(fun, jac, hess, hessp, constraints, kwargs):
    wrapped_fun = _wrap_fun(fun, kwargs)
    wrapped_jac = _wrap_fun(jac, kwargs)
    wrapped_hess = _wrap_fun(hess, kwargs)
    wrapped_hessp = _wrap_fun(hessp, kwargs)
    if isinstance(constraints, dict):
        constraints = (constraints,)
    wrapped_constraints = []
    for constraint in constraints:
        constraint = constraint.copy()
        ckwargs = constraint.pop('kwargs', {})
        constraint['fun'] = _wrap_fun(constraint.get('fun', None), ckwargs)
        constraint['jac'] = _wrap_fun(constraint.get('jac', None), ckwargs)
        wrapped_constraints.append(constraint)
    return (wrapped_fun, wrapped_jac, wrapped_hess, wrapped_hessp,
            wrapped_constraints)



def minimize_ipopt(fun,
                   x0,
                   args=(),
                   kwargs=None,
                   method=None,
                   jac=None,
                   hess=None,
                   hessp=None,
                   bounds=None,
                   constraints=(),
                   tol=None,
                   callback=None,
                   options=None):
    r"""
    Minimization using Ipopt with an interface like
    :py:func:`scipy.optimize.minimize`.

    Differences compared to :py:func:`scipy.optimize.minimize` include:

    - A different default `method`: when `method` is not provided, Ipopt is
      used to solve the problem.
    - Support for parameter `kwargs`: additional keyword arguments to be
      passed to the objective function, constraints, and their derivatives.
    - Lack of support for `callback` and `hessp` with the default `method`.

    This function can be used to solve general nonlinear programming problems
    of the form:

    .. math::

       \min_ {x \in R^n} f(x)

    subject to

    .. math::

       g_L \leq g(x) \leq g_U

       x_L \leq  x  \leq x_U

    where :math:`x` are the optimization variables, :math:`f(x)` is the
    objective function, :math:`g(x)` are the general nonlinear constraints,
    and :math:`x_L` and :math:`x_U` are the upper and lower bounds
    (respectively) on the decision variables. The constraints, :math:`g(x)`,
    have lower and upper bounds :math:`g_L` and :math:`g_U`. Note that equality
    constraints can be specified by setting :math:`g^i_L = g^i_U`.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized: ``fun(x, *args, **kwargs) ->
        float``.
    x0 : array-like, shape(n, )
        Initial guess. Array of real elements of shape (n,),
        where ``n`` is the number of independent variables.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (``fun``, ``jac``, and ``hess``).
    kwargs : :py:class:`dict`, optional
        Extra keyword arguments passed to the objective function and its
        derivatives (``fun``, ``jac``, ``hess``).
    method : :py:class:`str`, optional
        If unspecified (default), Ipopt is used.
        :py:func:`scipy.optimize.minimize` methods can also be used.
    jac : callable, optional
        The Jacobian (gradient) of the objective function:
        ``jac(x, *args, **kwargs) -> ndarray, shape(n, )``.
        If ``None``, SciPy's ``approx_fprime`` is used.
    hess : callable, optional
        The Hessian of the objective function:
        ``hess(x) -> ndarray, shape(n, )``.
        If ``None``, the Hessian is computed using IPOPT's numerical methods.
    hessp : callable, optional
        If `method` is one of the SciPy methods, this is a callable that
        produces the inner product of the Hessian and a vector. Otherwise, an
        error will be raised if a value other than ``None`` is provided.
    bounds : sequence of `array-like` shape(n, ) or :py:class:`scipy.optimize.Bounds`, optional
        Simple bounds on decision variables. There are two ways to specify the
        bounds:

            1. Instance of :py:class:`scipy.optimize.Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`. Use
               ``None`` to specify an infinite bound (i.e., no bound).

    constraints : {Constraint, dict}, optional
        Linear or nonlinear constraint specified by a dictionary,
        :py:class:`scipy.optimize.LinearConstraint`, or
        :py:class:`scipy.optimize.NonlinearConstraint`.
        See :py:func:`scipy.optimize.minimize` for more information. Note that
        the Jacobian of each constraint corresponds to the ``'jac'`` key and
        must be a callable function with signature ``jac(x) -> {ndarray,
        coo_array}``. If the constraint's value of ``'jac'`` is ``True``, the
        constraint function ``fun`` must return a tuple ``(con_val, con_jac)``
        consisting of the evaluated constraint ``con_val`` and the evaluated
        Jacobian ``con_jac``.
    tol : :py:class:`float`, optional (default=1e-8)
        The desired relative convergence tolerance, passed as an option to
        Ipopt. See [1]_ for details.
    options : :py:class:`dict`, optional
        A dictionary of solver options.

        When `method` is unspecified (default: Ipopt), the options
        ``disp`` and ``maxiter`` are automatically mapped to their Ipopt
        equivalents ``print_level`` and ``max_iter``, and ``eps`` is used to
        control the step size of finite difference gradient and constraint
        Jacobian approximations. All other options are passed directly
        to Ipopt. See [1]_ for details.

        For other values of `method`, `options` is passed to the SciPy solver.
        See [2]_ for details.
    callback : callable, optional
        This argument is ignored by the default `method` (Ipopt).
        If `method` is one of the SciPy methods, this is a callable that is
        called once per iteration. See [2]_ for details.

    References
    ----------
    .. [1] COIN-OR Project. "Ipopt: Ipopt Options".
           https://coin-or.github.io/Ipopt/OPTIONS.html
    .. [2] The SciPy Developers. "scipy.optimize.minimize".
           https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Examples
    --------
    Consider the problem of minimizing the Rosenbrock function. The Rosenbrock
    function and its derivatives are implemented in
    :py:func:`scipy.optimize.rosen`, :py:func:`scipy.optimize.rosen_der`, and
    :py:func:`scipy.optimize.rosen_hess`.

    >>> from cyipopt import minimize_ipopt
    >>> from scipy.optimize import rosen, rosen_der
    >>> x0 = [1.3, 0.7, 0.8, 1.9, 1.2]  # initial guess

    If we provide the objective function but no derivatives, Ipopt finds the
    correct minimizer (``[1, 1, 1, 1, 1]``) with a minimum objective value of
    0. However, it does not report success, and it requires many iterations
    and function evaluations before termination. This is because SciPy's
    ``approx_fprime`` requires many objective function evaluations to
    approximate the gradient, and still the approximation is not very accurate,
    delaying convergence.

    >>> res = minimize_ipopt(rosen, x0, jac=rosen_der)
    >>> res.success
    False
    >>> res.x
    array([1., 1., 1., 1., 1.])
    >>> res.nit, res.nfev, res.njev
    (46, 528, 48)

    To improve performance, provide the gradient using the `jac` keyword.
    In this case, Ipopt recognizes its own success, and requires fewer function
    evaluations to do so.

    >>> res = minimize_ipopt(rosen, x0, jac=rosen_der)
    >>> res.success
    True
    >>> res.nit, res.nfev, res.njev
    (37, 200, 39)

    For best results, provide the Hessian, too.

    >>> res = minimize_ipopt(rosen, x0, jac=rosen_der, hess=rosen_hess)
    >>> res.success
    True
    >>> res.nit, res.nfev, res.njev
    (17, 29, 19)
    """
    if not SCIPY_INSTALLED:
        msg = 'Install SciPy to use the `minimize_ipopt` function.'
        raise ImportError(msg)

    res = _minimize_ipopt_iv(fun, x0, args, kwargs, method, jac, hess, hessp,
                             bounds, constraints, tol, callback, options)
    (fun, x0, args, kwargs, method, jac, hess, hessp,
     bounds, constraints, tol, callback, options) = res

    if method is not None:
        funs = _wrap_funs(fun, jac, hess, hessp, constraints, kwargs)
        fun, jac, hess, hessp, constraints = funs
        bounds = optimize.Bounds(*bounds)
        res = optimize.minimize(fun, x0, args, method, jac, hess, hessp,
                                bounds, constraints, tol, callback, options)
        return res

    _x0 = np.atleast_1d(x0)

    lb, ub = bounds
    cl, cu = get_constraint_bounds(constraints, _x0)
    con_dims = get_constraint_dimensions(constraints, _x0)
    sparse_jacs, jac_nnz_row, jac_nnz_col = _get_sparse_jacobian_structure(
        constraints, x0)

    if options is None:
        options = {}
    eps = options.pop('eps', 1e-8)

    problem = IpoptProblemWrapper(fun,
                                  args=args,
                                  kwargs=kwargs,
                                  jac=jac,
                                  hess=hess,
                                  hessp=hessp,
                                  constraints=constraints,
                                  eps=eps,
                                  con_dims=con_dims,
                                  sparse_jacs=sparse_jacs,
                                  jac_nnz_row=jac_nnz_row,
                                  jac_nnz_col=jac_nnz_col)

    nlp = cyipopt.Problem(n=len(x0),
                          m=len(cl),
                          problem_obj=problem,
                          lb=lb,
                          ub=ub,
                          cl=cl,
                          cu=cu)

    # python3 compatibility
    convert_to_bytes(options)

    # Rename some default scipy options
    replace_option(options, b'disp', b'print_level')
    replace_option(options, b'maxiter', b'max_iter')
    options[b'print_level'] = int(options.get(b'print_level', 0))

    if b'tol' not in options:
        options[b'tol'] = tol or 1e-8
    if b'mu_strategy' not in options:
        options[b'mu_strategy'] = b'adaptive'
    if b'hessian_approximation' not in options:
        if hess is None and hessp is None:
            options[b'hessian_approximation'] = b'limited-memory'
    for option, value in options.items():
        try:
            nlp.add_option(option, value)
        except TypeError as e:
            msg = 'Invalid option for IPOPT: {0}: {1} (Original message: "{2}")'
            raise TypeError(msg.format(option, value, e))

    x, info = nlp.solve(x0)

    return OptimizeResult(x=x,
                          success=info['status'] == 0,
                          status=info['status'],
                          message=info['status_msg'],
                          fun=info['obj_val'],
                          info=info,
                          nfev=problem.nfev,
                          njev=problem.njev,
                          nit=problem.nit)


def _minimize_ipopt_iv(fun, x0, args, kwargs, method, jac, hess, hessp,
                       bounds, constraints, tol, callback, options):
    # basic input validation for minimize_ipopt that is not included in
    # IpoptProblemWrapper
    x0 = np.atleast_1d(x0)
    if not np.issubdtype(x0.dtype, np.number):
        raise ValueError('`x0` must be a numeric array.')

    # `method` does not need input validation. If `method is not None`, it is
    # passed to `scipy.optimize.minimize`, which raises a readable error if
    # the value isn't recognized.

    # Handle bounds that are either sequences (of sequences) or instances of
    # `optimize.Bounds`.
    if bounds is None:
        bounds = [-np.inf, np.inf]

    if isinstance(bounds, optimize.Bounds):
        lb, ub = bounds.lb, bounds.ub
    else:
        bounds = np.atleast_2d(bounds)
        if bounds.shape[1] != 2:
            raise ValueError("`bounds` must specify both lower and upper "
                             "limits for each decision variable.")
        lb, ub = bounds.T

    try:
        lb, ub, x0 = np.broadcast_arrays(lb, ub, x0)
    except ValueError:
        raise ValueError("The number of lower bounds, upper bounds, and "
                         "decision variables must be equal or broadcastable.")

    try:
        lb = lb.astype(np.float64)
        ub = ub.astype(np.float64)
    except ValueError:
        raise ValueError("The bounds must be numeric.")

    # Nones turn into NaNs above. Previously, NaNs caused Ipopt to hang, so
    # I'm not concerned about turning them into infs.
    lb[np.isnan(lb)] = -np.inf
    ub[np.isnan(ub)] = np.inf

    bounds = lb, ub

    constraints = optimize._minimize.standardize_constraints(constraints, x0,
                                                             'old')

    if method is None and callback is not None:
        raise NotImplementedError('`callback` is not yet supported by Ipopt.`')

    if tol is not None:
        tol = np.asarray(tol)[()]
        if tol.ndim != 0 or not np.issubdtype(tol.dtype, np.number) or tol <= 0:
            raise ValueError('`tol` must be a positive scalar.')
        else:  # tol should be a float, not an array
            tol = float(tol)

    options = dict() if options is None else options
    if not isinstance(options, dict):
        raise ValueError('`options` must be a dictionary.')

    return (fun, x0, args, kwargs, method, jac, hess, hessp,
            bounds, constraints, tol, callback, options)
