# -*- coding: utf-8 -*-
"""
cyipopt: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012-2015 Amit Aides
Copyright (C) 2015-2017 Matthias Kümmerer
Copyright (C) 2017-2023 cyipopt developers

License: EPL 2.0
"""

import logging
import sys
import warnings
import inspect

import numpy as np
cimport numpy as np

from cyipopt.exceptions import CyIpoptEvaluationError
from ipopt cimport *

__all__ = [
    "set_logging_level", "Problem", "IPOPT_VERSION"
]

IPOPT_VERSION = (IPOPT_VERSION_MAJOR, IPOPT_VERSION_MINOR, IPOPT_VERSION_RELEASE)

DTYPEi = np.int32
ctypedef np.int32_t DTYPEi_t
DTYPEd = np.double
ctypedef np.double_t DTYPEd_t

#
# Logging mechanism.
#
cdef int verbosity = logging.DEBUG


def set_logging_level(level=None):
    """Set the logger verbosity to the specified level.

    Parameters
    ----------
    level : int
        The verbosity of the logger. This threshold is used to determine which
        logging messages are logged by this module's :func:`log` function.

    """
    global verbosity

    if not level:
        logger = logging.getLogger('cyipopt')
        verbosity = logger.getEffectiveLevel()
    else:
        verbosity = level


set_logging_level()

cdef inline void log(char* msg, int level):
     if level >= verbosity:
         logging.getLogger('cyipopt').log(level, msg)

STATUS_MESSAGES = {
    Solve_Succeeded: (b"Algorithm terminated successfully at a locally "
                      b"optimal point, satisfying the convergence tolerances "
                      b"(can be specified by options)."),
    Solved_To_Acceptable_Level: (b"Algorithm stopped at a point that was "
                                 b"converged, not to \"desired\" tolerances, "
                                 b"but to \"acceptable\" tolerances (see the "
                                 b"acceptable-... options)."),
    Infeasible_Problem_Detected: (b"Algorithm converged to a point of local "
                                  b"infeasibility. Problem may be "
                                  b"infeasible."),
    Search_Direction_Becomes_Too_Small: (b"Algorithm proceeds with very "
                                         b"little progress."),
    Diverging_Iterates: b"It seems that the iterates diverge.",
    User_Requested_Stop: (b"The user call-back function intermediate_callback "
                          b"(see Section 3.3.4 in the documentation) returned "
                          b"false, i.e., the user code requested a premature "
                          b"termination of the optimization."),
    Feasible_Point_Found: b"Feasible point for square problem found.",
    Maximum_Iterations_Exceeded: (b"Maximum number of iterations exceeded "
                                  b"(can be specified by an option)."),
    Restoration_Failed: (b"Restoration phase failed, algorithm doesn\'t know "
                         b"how to proceed."),
    Error_In_Step_Computation: (b"An unrecoverable error occurred while Ipopt "
                                b"tried to compute the search direction."),
    Maximum_CpuTime_Exceeded: b"Maximum CPU time exceeded.",
    Not_Enough_Degrees_Of_Freedom: b"Problem has too few degrees of freedom.",
    Invalid_Problem_Definition: b"Invalid problem definition.",
    Invalid_Option: b"Invalid option encountered.",
    Invalid_Number_Detected: (b"Algorithm received an invalid number (such as "
                              b"NaN or Inf) from the NLP; see also option "
                              b"check_derivatives_for_naninf."),
    Unrecoverable_Exception: b"Some uncaught Ipopt exception encountered.",
    NonIpopt_Exception_Thrown: b"Unknown Exception caught in Ipopt.",
    Insufficient_Memory: b"Not enough memory.",
    Internal_Error: (b"An unknown internal error occurred. Please contact the "
                     b"Ipopt authors through the mailing list."),
}

INF = 10**19

CREATE_PROBLEM_MSG = """
----------------------------------------------------
Creating Ipopt problem with the following parameters
n = %s
m = %s
jacobian elements num = %s
hessian elements num = %s
"""


cdef class Problem:
    """
    Wrapper class for solving optimization problems using the C interface of
    the Ipopt package.

    It can be used to solve general nonlinear programming problems of the form:

    .. math::

       \min_ {x \in R^n} f(x)

    subject to

    .. math::

       g_L \leq g(x) \leq g_U

       x_L \leq  x  \leq x_U

    Where :math:`x` are the optimization variables (possibly with upper an
    lower bounds), :math:`f(x)` is the objective function and :math:`g(x)` are
    the general nonlinear constraints. The constraints, :math:`g(x)`, have
    lower and upper bounds. Note that equality constraints can be specified by
    setting :math:`g^i_L = g^i_U`.

    Parameters
    ----------
    n : :py:class:`int`
        Number of primal variables.
    m : :py:class:`int`
        Number of constraints.
    problem_obj: :py:class:`object`, optional (default=None)
        An object holding the problem's callbacks. If `None`, cyipopt will use
        ``self``, this is useful when subclassing ``Problem``. The object is
        required to have the following attributes and methods (some are
        optional):

            - .. py:function:: objective(x)

                Callback function for evaluating objective function
                at the given point `x`.

                **Parameters**

                x : :py:class:`numpy.ndarray`, shape `(n, )`
                    Value of the optimization variables at which the objective
                    is to be evaluated.

                **Returns**

                :py:class:`float`
                    Objective function value at the point x.

            - .. py:function:: constraints(x)

                Callback function for evaluating constraint functions at the
                given point `x`.

                **Parameters**

                x : :py:class:`numpy.ndarray`, shape `(n, )`
                    Value of the optimization variables at which the constraints
                    are to be evaluated.

                **Returns**

                :py:class:`numpy.ndarray`, shape `(m, )`
                    Constraint values at the point x.

            - .. py:function:: gradient(x)

                Callback function for evaluating gradient of objective function
                at the given point `x`.

                **Parameters**

                x : :py:class:`numpy.ndarray`, shape `(n, )`
                    Value of the optimization variables at which the gradient
                    is to be evaluated.

                **Returns**

                :py:class:`numpy.ndarray`, shape `(n, )`
                    Gradient of the objective function at the point x.

            - .. py:function:: jacobian(x)

                Callback function for evaluating Jacobian of constraint functions
                at the given point `x`. The values should be returned
                as a 1-dim numpy array (using the same order as you used when
                specifying the sparsity structure)

                **Parameters**

                x : :py:class:`numpy.ndarray`, shape `(n, )`
                    Value of the optimization variables at which the Jacobian is
                    to be evaluated.

                **Returns**

                :py:class:`numpy.ndarray`, shape `(jac_nnz, )`
                    Jacobian of the constraint functions at the point x.

            - .. py:function:: jacobianstructure()

                Optional callback function that returns the sparsity structure of the
                Jacobian of the constraints. The function should return a tuple
                of two :py:class:`numpy.ndarray`\ s of integers,
                the first array contains the row indices
                and the second array contains the column indices of the non-zero
                elements of the Jacobian. Both arrays should have the same
                length of `jac_nnz`, and indexing is zero-based.

                If this callback is not provided, the Jacobian is assumed
                to be dense (`jac_nnz` = `m` * `n`) using row-major (C-style) order.

                **Returns**

                :py:class:`tuple`
                    Tuple containing two numpy arrays, the first array contains
                    the row indices and the second array contains the column
                    indices of the non-zero elements of the Jacobian.

            - .. py:function:: hessian(x, lambda, objective_factor)

                Optional callback function for evaluating the Hessian of the Lagrangian
                function with respect to the primal variables :math:`x`. The Lagrangian
                is given by

                .. math:: \mathcal{L}(x, \lambda) = \lambda_0 f(x) + \sum_{i=1}^{m} \lambda_i g_i(x)

                where :math:`\lambda_1, \ldots, \lambda_m` are the Lagrange multipliers
                and :math:`\lambda_0` is the objective factor. The
                function should return the values of the Hessian as calculated
                using `x`, `lambda` and `objective_factor`. The values should be
                returned as a 1-dimensional array-like structure (using the same order as you
                used when specifying the sparsity structure).

                If this callback is not provided, the Hessian is calculated numerically.

                **Parameters**

                x : :py:class:`numpy.ndarray`, shape `(n, )`
                    Value of the optimization variables at which the Hessian is
                    to be evaluated.
                lambda : :py:class:`numpy.ndarray`, shape `(m, )`
                    Values for the Lagrange multipliers.
                objective_factor : :py:class:`float`
                    Factor for the objective term of the Hessian (usually 1).

                **Returns**

                :py:class:`numpy.ndarray`, shape `(hess_nnz, )`
                    Hessian of the Lagrangian function at the point x.

            - .. py:function:: hessianstructure()

                Optional callback function that returns the sparsity structure of the
                Hessian of the Lagrangian. The function should return a tuple
                of two :py:class:`numpy.ndarray`\ s, the first array contains the row indices
                and the second array contains the column indices of the non-zero
                elements of the Hessian. Both arrays should have the same
                length of `hess_nnz`, and indexing is zero-based. Furthermore,
                since the Hessian is symmetric, the indices must only correspond
                to the lower triangular part of the Hessian (i.e., `row >= col`).

                If this callback is not provided, the Hessian is assumed
                to be dense (`hess_nnz` = `n` * (`n` + 1) / 2) using row-major
                (C-style) order.

                **Returns**

                :py:class:`tuple`
                    Tuple containing two numpy arrays, the first array contains
                    the row indices and the second array contains the column
                    indices of the non-zero elements of the Hessian.

            - ``intermediate`` : function pointer, optional (default=None)
                Optional. Callback function that is called once per iteration
                (during the convergence check), and can be used to obtain
                information about the optimization status while Ipopt solves
                the problem. If this callback returns False, Ipopt will
                terminate with the ``User_Requested_Stop status``. The
                information below corresponeds to the argument list passed to
                this callback:

                    ``alg_mod``:
                        Algorithm phase: 0 is for regular, 1 is restoration.
                    ``iter_count``:
                        The current iteration count.
                    ``obj_value``:
                        The unscaled objective value at the current point
                    ``inf_pr``:
                        The scaled primal infeasibility at the current point.
                    ``inf_du``:
                        The scaled dual infeasibility at the current point.
                    ``mu``:
                        The value of the barrier parameter.
                    ``d_norm``:
                        The infinity norm (max) of the primal step.
                    ``regularization_size``:
                        The value of the regularization term for the Hessian
                        of the Lagrangian in the augmented system.
                    ``alpha_du``:
                        The stepsize for the dual variables.
                    ``alpha_pr``:
                        The stepsize for the primal variables.
                    ``ls_trials``:
                        The number of backtracking line search steps.

                more information can be found in the following link:
                https://coin-or.github.io/Ipopt/OUTPUT.html

    lb : array-like, shape(n, )
        Lower bounds on variables, where n is the dimension of x. To assume no
        lower bounds pass values lower then ``10^-19``.
    ub : array-like, shape(n, )
        Upper bounds on variables, where n is the dimension of x. To assume no
        upper bounds pass values higher then ``10^-19``.
    cl : array-like, shape(m, )
        Lower bounds on constraints, where m is the number of constraints.
        Equality constraints can be specified by setting ``cl[i] = cu[i]``.
    cu : array-like, shape(m, )
        Upper bounds on constraints, where m is the number of constraints.
        Equality constraints can be specified by setting ``cl[i] = cu[i]``.

    """

    cdef IpoptProblem __nlp
    cdef public object __objective
    cdef public object __constraints
    cdef public object __gradient
    cdef public object __jacobian
    cdef public object __jacobianstructure
    cdef public object __hessian
    cdef public object __hessianstructure
    cdef public object __intermediate
    cdef public Index __n
    cdef public Index __m

    cdef public object __exception
    cdef Bool __in_ipopt_solve

    def __init__(self, n, m, problem_obj=None, lb=None, ub=None, cl=None,
                 cu=None):

        if not isinstance(n, int) or not n > 0:
            raise TypeError("n must be a positive integer")

        if problem_obj is None:
            log(b"problem_obj is not defined, using self", logging.INFO)
            problem_obj = self

        self.__n = n

        if lb is None:
            lb = -INF * np.ones(n)

        if ub is None:
            ub = INF * np.ones(n)

        if len(lb) != len(ub) or len(lb) != n:
            raise ValueError("lb and ub must either be None or have length n.")

        cdef np.ndarray[DTYPEd_t, ndim=1]  np_lb = np.array(lb, dtype=DTYPEd).flatten()
        cdef np.ndarray[DTYPEd_t, ndim=1]  np_ub = np.array(ub, dtype=DTYPEd).flatten()

        #
        # Handle the constraints
        #
        if not isinstance(m, int) or not m >= 0:
            raise TypeError("m must be zero or a positive integer")

        if m < 1:
            m = 0
            cl = np.zeros(0)
            cu = np.zeros(0)
        else:
            if cl is None and cu is None:
                msg = ("Neither cl nor cu defined. At least one should be "
                       "defined.")
                raise ValueError(msg)
            elif cl is None:
                cl = -INF * np.ones(m)
            elif cu is None:
                cu = INF * np.ones(m)

        if len(cl) != len(cu) or len(cl) != m:
            msg = ("cl an cu must either be None (but not both) or have "
                   "length m.")
            raise ValueError(msg)

        self.__m = m

        cdef np.ndarray[DTYPEd_t, ndim=1]  np_cl = np.array(cl, dtype=DTYPEd).flatten()
        cdef np.ndarray[DTYPEd_t, ndim=1]  np_cu = np.array(cu, dtype=DTYPEd).flatten()

        #
        # Handle the callbacks
        #
        self.__objective = getattr(problem_obj, "objective", None)
        self.__constraints = getattr(problem_obj, "constraints", None)
        self.__gradient = getattr(problem_obj, "gradient", None)
        self.__jacobian = getattr(problem_obj, "jacobian", None)
        self.__jacobianstructure = getattr(problem_obj,
                                           "jacobianstructure",
                                           None)
        self.__hessian = getattr(problem_obj, "hessian", None)
        self.__hessianstructure = getattr(problem_obj,
                                          "hessianstructure",
                                          None)
        self.__intermediate = getattr(problem_obj, "intermediate", None)

        #
        # Verify that the objective and gradient callbacks are defined
        #
        if self.__objective is None or self.__gradient is None:
            msg = ("Both the \"objective\" and \"gradient\" callbacks must be "
                   "defined.")
            raise ValueError(msg)

        #
        # Verify that the constraints and jacobian callbacks are defined
        #
        if m > 0 and (self.__constraints is None or self.__jacobian is None):
            msg = ("Both the \"constraints\" and \"jacobian\" callbacks must "
                   "be defined.")
            raise ValueError(msg)

        cdef Index nele_jac = self.__m * self.__n
        cdef Index nele_hess = <Index>(<long>self.__n * (<long>self.__n + 1) / 2)

        if self.__jacobianstructure:
            ret_val = self.__jacobianstructure()
            nele_jac = len(ret_val[0])

        if self.__hessianstructure:
            ret_val = self.__hessianstructure()
            nele_hess = len(ret_val[0])
        else:
            if self.__hessian is None:
                msg = b"Hessian callback not given, setting nele_hess to 0"
                log(msg, logging.INFO)
                nele_hess = 0
            elif self.__n > 2**16:
                msg = ("Number of varialbes is too large for using dense "
                       "Hessian")
                raise ValueError(msg)

        #
        # Some input checking
        #
        if self.__m == 0 and nele_jac != 0:
            raise ValueError("m == 0 and number of jacobian elements != 0")

        if self.__m > 0 and nele_jac < 0:
            raise ValueError("m > 0 and number of jacobian elements < 1")

        if nele_hess < 0:
            raise ValueError("number of hessian elements < 0")

        creation_msg = CREATE_PROBLEM_MSG % (repr(self.__n),
                                             repr(self.__m),
                                             repr(nele_jac),
                                             repr(nele_hess)
                                             )
        creation_msg = creation_msg.encode("utf8")

        log(creation_msg, logging.DEBUG)

        # TODO: verify that the numpy arrays use C order
        self.__nlp = CreateIpoptProblem(self.__n,
                                        <Number*>np_lb.data,
                                        <Number*>np_ub.data,
                                        self.__m,
                                        <Number*>np_cl.data,
                                        <Number*>np_cu.data,
                                        nele_jac,
                                        nele_hess,
                                        0,
                                        objective_cb,
                                        constraints_cb,
                                        gradient_cb,
                                        jacobian_cb,
                                        hessian_cb
                                        )

        if self.__nlp == NULL:
            msg = "Failed to create NLP problem. Make sure inputs are okay!"
            raise RuntimeError(msg)

        SetIntermediateCallback(self.__nlp, intermediate_cb)

        if self.__hessian is None:
            msg = b"Hessian callback not given, using approximation"
            log(msg, logging.INFO)
            self.add_option(b"hessian_approximation", b"limited-memory")

        self.__exception = None

        # This flag is necessary to prevent segfaults in Ipopt <=3.14.11 due
        # to the lack of guard for __nlp->tnlp being NULL or undefined.
        self.__in_ipopt_solve = False

    def __dealloc__(self):
        if self.__nlp != NULL:
            FreeIpoptProblem(self.__nlp)

        self.__nlp = NULL

    def close(self):
        """Deallocate memory resources used by the Ipopt package.

        Called implicitly by the :class:`Problem` class destructor.

        """

        if self.__nlp != NULL:
            FreeIpoptProblem(self.__nlp)

        self.__nlp = NULL

    def add_option(self, keyword, val):
        """Add a keyword/value option pair to the problem.

        See the Ipopt documentaion for details on available options.

        Parameters
        ----------
        keyword : str
            Option name.
        val : str, int or float
            Value of the option. The type of val should match the option
            definition as described in the Ipopt documentation.

        """
        if isinstance(keyword, type("")):
            keyword = bytes(keyword, "utf-8")

        if isinstance(val, type("")):
            val = bytes(val, "utf-8")

        if isinstance(val, bytes):
            ret_val = AddIpoptStrOption(self.__nlp, keyword, val)
        elif type(val) == float:
            ret_val = AddIpoptNumOption(self.__nlp, keyword, val)
        elif type(val) == int:
            ret_val = AddIpoptIntOption(self.__nlp, keyword, val)
        else:
            raise TypeError("Invalid option type")

        if not ret_val:
            raise TypeError("Error while assigning an option")

    def set_problem_scaling(self, obj_scaling=1.0, x_scaling=None,
                            g_scaling=None):
        """Optional function for setting scaling parameters for the problem.

        To use the scaling parameters set the option ``nlp_scaling_method`` to
        ``user-scaling``.

        Parameters
        ----------
        obj_scaling : float
            Determines, how Ipopt should internally scale the objective
            function. For example, if this number is chosen to be 10, then
            Ipopt solves internally an optimization problem that has 10 times
            the value of the original objective. In particular, if this value
            is negative, then Ipopt will maximize the objective function
            instead of minimizing it.
        x_scaling : array-like, shape(n, )
            The scaling factors for the variables. If ``None``, no scaling is
            done.
        g_scaling : array-like, shape(m, )
            The scaling factors for the constrains. If ``None``, no scaling is
            done.

        """

        try:
            obj_scaling = float(obj_scaling)
        except (TypeError, ValueError):
            msg = "obj_scaling should be convertible to float type."
            raise ValueError(msg)

        cdef Number *x_scaling_p
        cdef Number *g_scaling_p
        cdef np.ndarray[DTYPEd_t, ndim=1] np_x_scaling
        cdef np.ndarray[DTYPEd_t, ndim=1] np_g_scaling

        if x_scaling is None:
            x_scaling_p = NULL
        else:
            if len(x_scaling) != self.__n:
                msg = "x_scaling must either be None or have length n."
                raise ValueError(msg)

            np_x_scaling = np.array(x_scaling, dtype=DTYPEd).flatten()
            x_scaling_p = <Number*>np_x_scaling.data

        if g_scaling is None:
            g_scaling_p = NULL
        else:
            if len(g_scaling) != self.__m:
                msg = "g_scaling must either be None or have length m."
                raise ValueError(msg)

            np_g_scaling = np.array(g_scaling, dtype=DTYPEd).flatten()
            g_scaling_p = <Number*>np_g_scaling.data

        ret_val = SetIpoptProblemScaling(self.__nlp,
                                         obj_scaling,
                                         x_scaling_p,
                                         g_scaling_p
                                         )

        if not ret_val:
            raise TypeError("Error while setting the scaling of the problem.")

    def solve(self, x, lagrange=[], zl=[], zu=[]):
        """Returns the optimal solution and an info dictionary.

        Solves the posed optimization problem starting at point x.

        Parameters
        ----------
        x : array-like, shape(n, )
            Initial guess.

        lagrange : array-like, shape(m, ), optional (default=[])
            Initial values for the constraint multipliers (only if warm start option is chosen).

        zl : array-like, shape(n, ), optional (default=[])
            Initial values for the multipliers for lower variable bounds (only if warm start option is chosen).

        zu : array-like, shape(n, ), optional (default=[])
            Initial values for the multipliers for upper variable bounds (only if warm start option is chosen).

        Returns
        -------
        x : :py:class:`numpy.ndarray`, shape `(n, )`
            Optimal solution.
        info: :py:class:`dict` with the following entries

            ``x``: :py:class:`numpy.ndarray`, shape `(n, )`
                optimal solution
            ``g``: :py:class:`numpy.ndarray`, shape `(m, )`
                constraints at the optimal solution
            ``obj_val``: :py:class:`float`
                objective value at optimal solution
            ``mult_g``: :py:class:`numpy.ndarray`, shape `(m, )`
                final values of the constraint multipliers
            ``mult_x_L``: :py:class:`numpy.ndarray`, shape `(n, )`
                bound multipliers at the solution
            ``mult_x_U``: :py:class:`numpy.ndarray`, shape `(n, )`
                bound multipliers at the solution
            ``status``: :py:class:`int`
                gives the status of the algorithm
            ``status_msg``: :py:class:`str`
                gives the status of the algorithm as a message

        """

        if self.__n != len(x):
            raise ValueError("Wrong length of x0.")

        cdef np.ndarray[DTYPEd_t, ndim=1]  np_x = np.array(x, dtype=DTYPEd).flatten()

        cdef ApplicationReturnStatus stat
        cdef np.ndarray[DTYPEd_t, ndim=1] g = np.zeros((self.__m,), dtype=DTYPEd)

        lagrange = np.atleast_1d(lagrange)
        zl = np.atleast_1d(zl)
        zu = np.atleast_1d(zu)

        if len(lagrange) == 0:
            lagrange = np.zeros((self.__m,), dtype=DTYPEd)
        elif self.__m != len(lagrange):
            raise ValueError("Wrong length of lagrange.")

        cdef np.ndarray[DTYPEd_t, ndim=1] mult_g = np.array(lagrange, dtype=DTYPEd).flatten()

        if len(zl) == 0:
            zl = np.zeros((self.__n,), dtype=DTYPEd)
        elif self.__n != len(zl):
            raise ValueError("Wrong length of zl.")

        if len(zu) == 0:
            zu = np.zeros((self.__n,), dtype=DTYPEd)
        elif self.__n != len(zu):
            raise ValueError("Wrong length of zu.")

        cdef np.ndarray[DTYPEd_t, ndim=1] mult_x_L = np.array(zl, dtype=DTYPEd).flatten()
        cdef np.ndarray[DTYPEd_t, ndim=1] mult_x_U = np.array(zu, dtype=DTYPEd).flatten()


        cdef Number obj_val = 0

        # Set flag that we are in a solve, so __nlp->tnlp references (e.g. in
        # get_current_iterate) are valid.
        self.__in_ipopt_solve = True
        stat = IpoptSolve(self.__nlp,
                          <Number*>np_x.data,
                          <Number*>g.data,
                          &obj_val,
                          <Number*>mult_g.data,
                          <Number*>mult_x_L.data,
                          <Number*>mult_x_U.data,
                          <UserDataPtr>self
                          )
        # Unset flag
        self.__in_ipopt_solve = False

        if self.__exception:
            raise self.__exception[0], self.__exception[1], self.__exception[2]


        info = {"x": np_x,
                "g": g,
                "obj_val": obj_val,
                "mult_g": mult_g,
                "mult_x_L": mult_x_L,
                "mult_x_U": mult_x_U,
                "status": stat,
                "status_msg": STATUS_MESSAGES.get(stat, b"Unknown status")
                }

        return np_x, info

    def get_current_iterate(self, scaled=False):
        """Return the current iterate vectors during an Ipopt solve

        The iterate contains vectors for primal variables, bound multipliers,
        constraint function values, and constraint multipliers. Here, the
        constraints are treated as a single function rather than separating
        equality and inequality constraints. This method can only be called
        during an intermediate callback.

        **Only supports Ipopt >=3.14.0**

        Parameters
        ----------
        scaled: :py:class:`bool`
            Whether the scaled iterate vectors should be returned

        Returns
        -------
        :py:class:`dict` or `None`
            A dict containing the iterate vector with keys ``"x"``,
            ``"mult_x_L"``, ``"mult_x_U"``, ``"g"``, and ``"mult_g"``.
            If iterate vectors cannot be obtained, ``None`` is returned.

        """
        # Check that we are using an Ipopt version that supports this
        # functionality
        major, minor, release = IPOPT_VERSION
        if major < 3 or (major == 3 and minor < 14):
            raise RuntimeError(
                "get_current_iterate only supports Ipopt version >=3.14.0"
                " CyIpopt is compiled with version %s.%s.%s"
                % (major, minor, release)
            )
        # Check that we are in a solve. This is necessary to prevent segfaults
        # pre-Ipopt 3.14.12
        if not self.__in_ipopt_solve:
            raise RuntimeError(
                "get_current_iterate can only be called during a call to solve,"
                " e.g. in an intermediate callback."
            )
        # Allocate arrays to hold the current iterate
        cdef np.ndarray[DTYPEd_t, ndim=1] np_x
        cdef np.ndarray[DTYPEd_t, ndim=1] np_mult_x_L
        cdef np.ndarray[DTYPEd_t, ndim=1] np_mult_x_U
        cdef np.ndarray[DTYPEd_t, ndim=1] np_g
        cdef np.ndarray[DTYPEd_t, ndim=1] np_mult_g
        np_x = np.zeros((self.__n,), dtype=DTYPEd).flatten()
        np_mult_x_L = np.zeros((self.__n,), dtype=DTYPEd).flatten()
        np_mult_x_U = np.zeros((self.__n,), dtype=DTYPEd).flatten()
        np_g = np.zeros((self.__m,), dtype=DTYPEd).flatten()
        np_mult_g = np.zeros((self.__m,), dtype=DTYPEd).flatten()

        # Cast to C data types
        x = <Number*>np_x.data
        mult_x_L = <Number*>np_mult_x_L.data
        mult_x_U = <Number*>np_mult_x_U.data
        g = <Number*>np_g.data
        mult_g = <Number*>np_mult_g.data

        successful = CyGetCurrentIterate(
            self.__nlp,
            scaled,
            self.__n,
            x,
            mult_x_L,
            mult_x_U,
            self.__m,
            g,
            mult_g,
        )
        if successful:
            # Return values to user
            return {
                "x": np_x,
                "mult_x_L": np_mult_x_L,
                "mult_x_U": np_mult_x_U,
                "g": np_g,
                "mult_g": np_mult_g,
            }
        else:
            # This happens if this method is called during IpoptSolve,
            # but outside of an intermediate callback.
            return None

    def get_current_violations(self, scaled=False):
        """Return the current violation vectors during an Ipopt solve

        Violations returned are primal variable bound violations, bound
        complementarities, the gradient of the Lagrangian, constraint
        violation, and constraint complementarity. Here, the constraints
        are treated as a single function rather than separating equality
        and inequality constraints. This method can only be called during
        an intermediate callback.

        **Only supports Ipopt >=3.14.0**

        Parameters
        ----------
        scaled: :py:class:`bool`
            Whether to scale the returned violations

        Returns
        -------
        :py:class:`dict` or `None`
            A dict containing the violation vector with keys
            ``"x_L_violation"``, ``"x_U_violation"``, ``"compl_x_L"``,
            ``"compl_x_U"``, ``"grad_lag_x"``, ``"g_violation"``,
            and ``"compl_g"``. If violation vectors cannot be obtained,
            ``None`` is returned.

        """
        major, minor, release = IPOPT_VERSION
        if major < 3 or (major == 3 and minor < 14):
            raise RuntimeError(
                "get_current_violations only supports Ipopt version >=3.14.0"
                " CyIpopt is compiled with version %s.%s.%s"
                % (major, minor, release)
            )
        # Check that we are in a solve. This is necessary to prevent segfaults
        # pre-Ipopt 3.14.12
        if not self.__in_ipopt_solve:
            raise RuntimeError(
                "get_current_violations can only be called during a call to solve,"
                " e.g. in an intermediate callback."
            )
        # Allocate arrays to hold current violations
        cdef np.ndarray[DTYPEd_t, ndim=1] np_x_L_viol
        cdef np.ndarray[DTYPEd_t, ndim=1] np_x_U_viol
        cdef np.ndarray[DTYPEd_t, ndim=1] np_compl_x_L
        cdef np.ndarray[DTYPEd_t, ndim=1] np_compl_x_U
        cdef np.ndarray[DTYPEd_t, ndim=1] np_grad_lag_x
        cdef np.ndarray[DTYPEd_t, ndim=1] np_g_viol
        cdef np.ndarray[DTYPEd_t, ndim=1] np_compl_g
        np_x_L_viol = np.zeros((self.__n,), dtype=DTYPEd).flatten()
        np_x_U_viol = np.zeros((self.__n,), dtype=DTYPEd).flatten()
        np_compl_x_L = np.zeros((self.__n,), dtype=DTYPEd).flatten()
        np_compl_x_U = np.zeros((self.__n,), dtype=DTYPEd).flatten()
        np_grad_lag_x = np.zeros((self.__n,), dtype=DTYPEd).flatten()
        np_g_viol = np.zeros((self.__m,), dtype=DTYPEd).flatten()
        np_compl_g = np.zeros((self.__m,), dtype=DTYPEd).flatten()

        # Cast to C data types
        x_L_viol = <Number*>np_x_L_viol.data
        x_U_viol = <Number*>np_x_U_viol.data
        compl_x_L = <Number*>np_compl_x_L.data
        compl_x_U = <Number*>np_compl_x_U.data
        grad_lag_x = <Number*>np_grad_lag_x.data
        g_viol = <Number*>np_g_viol.data
        compl_g = <Number*>np_compl_g.data

        successful = CyGetCurrentViolations(
            self.__nlp,
            scaled,
            self.__n,
            x_L_viol,
            x_U_viol,
            compl_x_L,
            compl_x_U,
            grad_lag_x,
            self.__m,
            g_viol,
            compl_g,
        )
        if successful:
            # Return values to the user
            return {
                "x_L_violation": np_x_L_viol,
                "x_U_violation": np_x_U_viol,
                "compl_x_L": np_compl_x_L,
                "compl_x_U": np_compl_x_U,
                "grad_lag_x": np_grad_lag_x,
                "g_violation": np_g_viol,
                "compl_g": np_compl_g,
            }
        else:
            # This happens if this method is called during IpoptSolve,
            # but outside of an intermediate callback.
            return None


#
# Callback functions
#
cdef Bool objective_cb(Index n,
                       Number* x,
                       Bool new_x,
                       Number* obj_value,
                       UserDataPtr user_data
                       ) noexcept:
    cdef Problem self
    cdef Index i
    cdef np.ndarray[DTYPEd_t, ndim=1] _x

    try:
        log(b"objective_cb", logging.INFO)

        self = <Problem>user_data
        _x = np.zeros((n,), dtype=DTYPEd)

        for i in range(n):
            _x[i] = x[i]

        obj_value[0] = self.__objective(_x)
    except CyIpoptEvaluationError:
        return False
    except:
        self.__exception = sys.exc_info()
    return True


cdef Bool gradient_cb(Index n,
                      Number* x,
                      Bool new_x,
                      Number* grad_f,
                      UserDataPtr user_data
                      ) noexcept:

    cdef Problem self
    cdef Index i
    cdef np.ndarray[DTYPEd_t, ndim=1] _x
    cdef np.ndarray[DTYPEd_t, ndim=1] np_grad_f

    try:
        log(b"gradient_cb", logging.INFO)

        self = <Problem>user_data
        _x = np.zeros((n,), dtype=DTYPEd)

        for i in range(n):
            _x[i] = x[i]

        ret_val = self.__gradient(_x)

        np_grad_f = np.array(ret_val, dtype=DTYPEd).flatten()

        for i in range(n):
            grad_f[i] = np_grad_f[i]

    except CyIpoptEvaluationError:
        return False
    except:
        self.__exception = sys.exc_info()
        return True

    return True


cdef Bool constraints_cb(Index n,
                         Number* x,
                         Bool new_x,
                         Index m,
                         Number* g,
                         UserDataPtr user_data
                         ) noexcept:
    cdef Problem self
    cdef Index i
    cdef np.ndarray[DTYPEd_t, ndim=1] _x
    cdef np.ndarray[DTYPEd_t, ndim=1] np_g

    try:
        log(b"constraints_cb", logging.INFO)

        self = <Problem>user_data
        _x = np.zeros((n,), dtype=DTYPEd)

        if not self.__constraints:
            log(b"Constraints callback not defined", logging.DEBUG)
            return True

        for i in range(n):
            _x[i] = x[i]

        ret_val = self.__constraints(_x)

        np_g = np.array(ret_val, dtype=DTYPEd).flatten()

        for i in range(m):
            g[i] = np_g[i]

    except CyIpoptEvaluationError:
        return False
    except:
        self.__exception = sys.exc_info()
        return True

    return True


cdef Bool jacobian_struct_cb(Index n,
                             Index m,
                             Index nele_jac,
                             Index *iRow,
                             Index *jCol,
                             UserDataPtr user_data):
    cdef Problem self = <Problem>user_data
    cdef Index i

    if not self.__jacobianstructure:
        msg = b"Jacobian callback not defined. assuming a dense jacobian"
        log(msg, logging.INFO)

        #
        # Assuming a dense Jacobian
        #
        s = np.unravel_index(np.arange(self.__m * self.__n),
                             (self.__m, self.__n))
        np_iRow = np.array(s[0], dtype=DTYPEi)
        np_jCol = np.array(s[1], dtype=DTYPEi)
    else:
        #
        # Sparse Jacobian
        #
        ret_val = self.__jacobianstructure()

        np_iRow = np.array(ret_val[0], dtype=DTYPEi).flatten()
        np_jCol = np.array(ret_val[1], dtype=DTYPEi).flatten()

        if (np_iRow.size != nele_jac) or (np_jCol.size != nele_jac):
            msg = b"Invalid number of indices returned from jacobianstructure"
            log(msg, logging.ERROR)
            return False

        if (np_iRow < 0).any() or (np_iRow >= m).any():
            msg = b"Invalid row indices returned from jacobianstructure"
            log(msg, logging.ERROR)
            return False

        if (np_jCol < 0).any() or (np_jCol >= n).any():
            msg = b"Invalid column indices returned from jacobianstructure"
            log(msg, logging.ERROR)
            return False

    for i in range(nele_jac):
        iRow[i] = np_iRow[i]
        jCol[i] = np_jCol[i]

    return True


cdef Bool jacobian_value_cb(Index n,
                            Number* x,
                            Bool new_x,
                            Index m,
                            Index nele_jac,
                            Number *values,
                            UserDataPtr user_data
                            ):
    cdef Problem self = <Problem>user_data
    cdef Index i
    cdef np.ndarray[DTYPEd_t, ndim=1] _x = np.zeros((n,), dtype=DTYPEd)

    if not self.__jacobian:
        log(b"Jacobian callback not defined", logging.DEBUG)
        return True

    for i in range(n):
        _x[i] = x[i]

    try:
        ret_val = self.__jacobian(_x)
    except CyIpoptEvaluationError:
            return False

    np_jac_g = np.array(ret_val, dtype=DTYPEd).flatten()

    if (np_jac_g.size != nele_jac):
        msg = b"Invalid number of indices returned from jacobian"
        log(msg, logging.ERROR)
        return False

    for i in range(nele_jac):
        values[i] = np_jac_g[i]

    return True


cdef Bool jacobian_cb(Index n,
                      Number* x,
                      Bool new_x,
                      Index m,
                      Index nele_jac,
                      Index *iRow,
                      Index *jCol,
                      Number *values,
                      UserDataPtr user_data
                      ) noexcept:
    cdef Problem self
    cdef object ret_val

    try:
        log(b"jacobian_cb", logging.INFO)
        ret_val = True
        self = <Problem>user_data
        if values == NULL:
            log(b"Querying for iRow/jCol indices of the jacobian", logging.INFO)
            ret_val = jacobian_struct_cb(n, m, nele_jac, iRow, jCol, user_data)
        else:
            log(b"Querying for jacobian", logging.INFO)
            ret_val = jacobian_value_cb(n, x, new_x, m, nele_jac, values, user_data)

    except:
        self.__exception = sys.exc_info()
    finally:
        return ret_val


cdef Bool hessian_struct_cb(Index n,
                            Index m,
                            Index nele_hess,
                            Index *iRow,
                            Index *jCol,
                            UserDataPtr user_data
                            ):
    cdef Problem self = <Problem>user_data
    cdef Index i
    cdef np.ndarray[DTYPEi_t, ndim=1] np_iRow
    cdef np.ndarray[DTYPEi_t, ndim=1] np_jCol

    msg = b"Querying for iRow/jCol indices of the Hessian"
    log(msg, logging.INFO)

    if not self.__hessianstructure:
        msg = (b"Hessian callback not defined. assuming a lower triangle "
               b"Hessian")
        log(msg, logging.INFO)

        #
        # Assuming a lower triangle Hessian
        # Note:
        # There is a need to reconvert the s.col and s.row to arrays
        # because they have the wrong stride
        #
        row, col = np.nonzero(np.tril(np.ones((self.__n, self.__n))))
        np_iRow = np.array(col, dtype=DTYPEi)
        np_jCol = np.array(row, dtype=DTYPEi)
    else:
        #
        # Sparse Hessian
        #
        ret_val = self.__hessianstructure()

        np_iRow = np.array(ret_val[0], dtype=DTYPEi).flatten()
        np_jCol = np.array(ret_val[1], dtype=DTYPEi).flatten()

        if (np_iRow.size != nele_hess) or (np_jCol.size != nele_hess):
            msg = b"Invalid number of indices returned from hessianstructure"
            log(msg, logging.ERROR)
            return False

        if not(np_iRow >= np_jCol).all():
            msg = b"Indices are not lower triangular in hessianstructure"
            log(msg, logging.ERROR)
            return False

        if (np_jCol < 0).any():
            msg = b"Invalid column indices returned from hessianstructure"
            log(msg, logging.ERROR)
            return False

        if (np_iRow >= n).any():
            msg = b"Invalid row indices returned from hessianstructure"
            log(msg, logging.ERROR)
            return False

    for i in range(nele_hess):
        iRow[i] = np_iRow[i]
        jCol[i] = np_jCol[i]

    return True


cdef Bool hessian_value_cb(Index n,
                           Number* x,
                           Bool new_x,
                           Number obj_factor,
                           Index m,
                           Number *lambd,
                           Bool new_lambda,
                           Index nele_hess,
                           Number *values,
                           UserDataPtr user_data
                           ):
    cdef Index i
    cdef Problem self = <Problem>user_data
    cdef np.ndarray[DTYPEd_t, ndim=1] _x = np.zeros((n,), dtype=DTYPEd)
    cdef np.ndarray[DTYPEd_t, ndim=1] _lambda = np.zeros((m,), dtype=DTYPEd)

    if not self.__hessian:
        msg = (b"Hessian callback not defined but called by the Ipopt "
               b"algorithm")
        log(msg, logging.ERROR)
        return False

    for i in range(n):
        _x[i] = x[i]

    for i in range(m):
        _lambda[i] = lambd[i]

    try:
        ret_val = self.__hessian(_x, _lambda, obj_factor)
    except CyIpoptEvaluationError:
        return False

    np_h = np.array(ret_val, dtype=DTYPEd).flatten()

    if (np_h.size != nele_hess):
        msg = b"Invalid number of indices returned from hessian"
        log(msg, logging.ERROR)
        return False

    for i in range(nele_hess):
        values[i] = np_h[i]

    return True


cdef Bool hessian_cb(Index n,
                     Number* x,
                     Bool new_x,
                     Number obj_factor,
                     Index m,
                     Number *lambd,
                     Bool new_lambda,
                     Index nele_hess,
                     Index *iRow,
                     Index *jCol,
                     Number *values,
                     UserDataPtr user_data
                     ) noexcept:
    cdef object self
    cdef object ret_val

    try:
        log(b"hessian_cb", logging.INFO)
        ret_val = True
        self = <object>user_data

        if values == NULL:
            ret_val = hessian_struct_cb(n,
                                        m,
                                        nele_hess,
                                        iRow,
                                        jCol,
                                        user_data)

        else:
            ret_val = hessian_value_cb(n,
                                       x,
                                       new_x,
                                       obj_factor,
                                       m,
                                       lambd,
                                       new_lambda,
                                       nele_hess,
                                       values,
                                       user_data)
    except:
        self.__exception = sys.exc_info()
    finally:
        return ret_val


cdef Bool intermediate_cb(Index alg_mod,
                          Index iter_count,
                          Number obj_value,
                          Number inf_pr,
                          Number inf_du,
                          Number mu,
                          Number d_norm,
                          Number regularization_size,
                          Number alpha_du,
                          Number alpha_pr,
                          Index ls_trials,
                          UserDataPtr user_data
                          ) noexcept:
    cdef Problem self

    try:
        log(b"intermediate_cb", logging.INFO)

        self = <Problem>user_data

        if self.__exception:
            return False

        if not self.__intermediate:
            return True

        ret_val = self.__intermediate(alg_mod,
                                      iter_count,
                                      obj_value,
                                      inf_pr,
                                      inf_du,
                                      mu,
                                      d_norm,
                                      regularization_size,
                                      alpha_du,
                                      alpha_pr,
                                      ls_trials
                                      )

        if ret_val is None:
            return True
    except:
        self.__exception = sys.exc_info()
        return True

    return ret_val
