# -*- coding: utf-8 -*-
"""
cyipot: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012 Amit Aides
Author: Amit Aides <amitibo@tx.technion.ac.il>
URL: <http://http://code.google.com/p/cyipopt/>
License: EPL 1.0
"""
import numpy as np
cimport numpy as np
from ipopt cimport *
import logging
import scipy.sparse as sps
import types

__all__ = ['setLoggingLevel', 'problem']

DTYPEi = np.int32
ctypedef np.int32_t DTYPEi_t
DTYPEd = np.double
ctypedef np.double_t DTYPEd_t

#
# Logging mechanism.
#
cdef int verbosity = logging.DEBUG

def setLoggingLevel(level=None):
    global verbosity
    
    if not level:
        logger = logging.getLogger()
        verbosity = logger.getEffectiveLevel()
    else:
        verbosity = level
        
setLoggingLevel()

cdef inline void log(char* msg, int level):
     if level >= verbosity:
         logging.log(level, msg)

STATUS_MESSAGES = {
    Solve_Succeeded: 'Algorithm terminated successfully at a locally optimal point, satisfying the convergence tolerances (can be specified by options).',
    Solved_To_Acceptable_Level: 'Algorithm stopped at a point that was converged, not to "desired" tolerances, but to "acceptable" tolerances (see the acceptable-... options).',
    Infeasible_Problem_Detected: 'Algorithm converged to a point of local infeasibility. Problem may be infeasible.',
    Search_Direction_Becomes_Too_Small: 'Algorithm proceeds with very little progress.',
    Diverging_Iterates: 'It seems that the iterates diverge.',
    User_Requested_Stop: 'The user call-back function intermediate_callback (see Section 3.3.4 in the documentation) returned false, i.e., the user code requested a premature termination of the optimization.',
    Feasible_Point_Found: 'Feasible point for square problem found.',
    Maximum_Iterations_Exceeded: 'Maximum number of iterations exceeded (can be specified by an option).',
    Restoration_Failed: 'Restoration phase failed, algorithm doesn\'t know how to proceed.',
    Error_In_Step_Computation: 'An unrecoverable error occurred while Ipopt tried to compute the search direction.',
    Maximum_CpuTime_Exceeded: 'Maximum CPU time exceeded.',
    Not_Enough_Degrees_Of_Freedom: 'Problem has too few degrees of freedom.',
    Invalid_Problem_Definition: 'Invalid problem definition.',
    Invalid_Option: 'Invalid option encountered.',
    Invalid_Number_Detected: 'Algorithm received an invalid number (such as NaN or Inf) from the NLP; see also option check_derivatives_for_naninf',
    Unrecoverable_Exception: 'Some uncaught Ipopt exception encountered.',
    NonIpopt_Exception_Thrown: 'Unknown Exception caught in Ipopt',
    Insufficient_Memory: 'Not enough memory.',
    Internal_Error: 'An unknown internal error occurred. Please contact the Ipopt authors through the mailing list.'
}

INF = 10**19

cdef class problem:
    """
    Wrapper class for solving optimization problems using the C interface of
    the IPOPT package.
    
    It can be used to solve general nonlinear programming problems of the form:

    .. math::

           \min_ {x \in R^n} f(x)

    subject to

    .. math::

           g_L \leq g(x) \leq g_U

           x_L \leq  x  \leq x_U
    
    Where :math:`x` are the optimization variables (possibly with upper an lower
    bounds), :math:`f(x)` is the objective function and :math:`g(x)` are the general nonlinear
    constraints. The constraints, :math:`g(x)`, have lower and upper bounds. Note that
    equality constraints can be specified by setting :math:`g^i_L = g^i_U`.

    Parameters
    ----------
    n : integer
        Number of primal variables.
    m : integer
        Number of constraints    
    problem_obj: object
        An object with the following attributes (holding the problam's callbacks)
            
            - 'objective' : function pointer
                Callback function for evaluating objective function.
                The callback functions accepts one parameter: x (value of the
                optimization variables at which the objective is to be evaluated).
                The function should return the objective function value at the point x.
            - 'constraints' : function pointer
                Callback function for evaluating constraint functions.
                The callback functions accepts one parameter: x (value of the
                optimization variables at which the constraints are to be evaluated).
                The function should return the constraints values at the point x.
            - 'gradient' : function pointer
                Callback function for evaluating gradient of objective function.
                The callback functions accepts one parameter: x (value of the
                optimization variables at which the gradient is to be evaluated).
                The function should return the gradient of the objective function at the
                point x.
            - 'jacobian' : function pointer
                Callback function for evaluating Jacobian of constraint functions.
                The callback functions accepts one parameter: x (value of the
                optimization variables at which the jacobian is to be evaluated).
                The function should return the values of the jacobian as calculated
                using x. The values should be returned as a 1-dim numpy array (using
                the same order as you used when specifying the sparsity structure)
            - 'jacobianstructure' : function pointer
                Optional. Callback function that accepts no parameters and returns the
                sparsity structure of the Jacobian (the row and column indices only).
                If None, the Jacobian is assumed to be dense.
            - 'hessian' : function pointer
                Optional. Callback function for evaluating Hessian of the Lagrangian
                function.
                The callback functions accepts three parameters x (value of the
                optimization variables at which the hessian is to be evaluated), lambda
                (values for the constraint multipliers at which the hessian is to be
                evaluated) objective_factor the factor in front of the objective term
                in the Hessian.
                The function should return the values of the Hessian as calculated using
                x, lambda and objective_factor. The values should be returned as a 1-dim
                numpy array (using the same order as you used when specifying the
                sparsity structure).
                If None, the Hessian is calculated numerically.
            - 'hessianstructure' : function pointer
                Optional. Callback function that accepts no parameters and returns the
                sparsity structure of the Hessian of the lagrangian (the row and column
                indices only). If None, the Hessian is assumed to be dense.
            - 'intermediate' : function pointer
                Optional. Callback function that is called once per iteration (during
                the convergence check), and can be used to obtain information about the
                optimization status while IPOPT solves the problem.
                If this callback returns False, IPOPT will terminate with the
                User_Requested_Stop status.
                The information below corresponeds to the argument list passed to this
                callback:

                    'alg_mod':
                        Algorithm phase: 0 is for regular, 1 is restoration.
                    'iter_count':
                        The current iteration count.
                    'obj_value':
                        The unscaled objective value at the current point
                    'inf_pr':
                        The scaled primal infeasibility at the current point. 
                    'inf_du':
                        The scaled dual infeasibility at the current point.
                    'mu':
                        The value of the barrier parameter.
                    'd_norm':
                        The infinity norm (max) of the primal step.
                    'regularization_size':
                        The value of the regularization term for the Hessian
                        of the Lagrangian in the augmented system.
                    'alpha_du':
                        The stepsize for the dual variables. 
                    'alpha_pr':
                        The stepsize for the primal variables.
                    'ls_trials':
                        The number of backtracking line search steps.

                more information can be found in the following link:
                http://www.coin-or.org/Ipopt/documentation/node56.html#sec:output

    lb : array-like, shape = [n]
        Lower bounds on variables, where n is the dimension of x.
        To assume no lower bounds pass values lower then 10^-19.    
    ub : array-like, shape = [n]
        Upper bounds on variables, where n is the dimension of x..
        To assume no upper bounds pass values higher then 10^-19.
    cl : array-like, shape = [m]
        Lower bounds on constraints, where m is the number of constraints.
        Equality constraints can be specified by setting cl[i] = cu[i].
    cu : array-like, shape = [m]
        Upper bounds on constraints, where m is the number of constraints.
        Equality constraints can be specified by setting cl[i] = cu[i].

"""

    cdef IpoptProblem _nlp
    cdef public object _objective
    cdef public object _constraints
    cdef public object _gradient
    cdef public object _jacobian
    cdef public object _jacobianstructure
    cdef public object _hessian   
    cdef public object _hessianstructure
    cdef public object _intermediate
    cdef public Index _n
    cdef public Index _m
    
    def __cinit__(
            self,
            n,
            m,
            problem_obj,
            lb=None,
            ub=None,
            cl=None,
            cu=None
            ):

        assert(n in (types.IntType, types.LongType) and n > 0, 'n must be a positive integer.')
            
        self._n = n
        
        if lb == None:
            lb = -INF*np.ones(n)
        
        if ub is None:
            ub = INF*np.ones(n)
            
        if len(lb) != len(ub) or len(lb) != n:
            raise ValueError('lb and ub must either be None or have length n.')
            
        cdef np.ndarray[DTYPEd_t, ndim=1]  np_lb = np.array(lb, dtype=DTYPEd).flatten()
        cdef np.ndarray[DTYPEd_t, ndim=1]  np_ub = np.array(ub, dtype=DTYPEd).flatten()
        
        #
        # Handle the constraints
        #
        assert(m in (types.IntType, types.LongType) and m > 0, 'm must be a positive integer.')
            
        if m < 1:
            m = 0
            cl = np.zeros(0)
            cu = np.zeros(0)
        else:
            if cl == None and cu == None:
                raise ValueError('Neither cl nor cu defined. At least one should be defined.')
            elif cl == None:
                cl = -INF*np.ones(m)
            elif cu == None:
                cu = INF*np.ones(m)
                
        if len(cl) != len(cu) or len(cl) != m:
            raise ValueError('cl an cu must either be None (but not both) or have length m.')

        self._m = m
        
        cdef np.ndarray[DTYPEd_t, ndim=1]  np_cl = np.array(cl, dtype=DTYPEd).flatten()
        cdef np.ndarray[DTYPEd_t, ndim=1]  np_cu = np.array(cu, dtype=DTYPEd).flatten()
        
        #
        # Handle the callbacks
        #
        self._objective = problem_obj.objective
        self._constraints = problem_obj.constraints
        self._gradient = problem_obj.gradient
        self._jacobian = problem_obj.jacobian
        self._jacobianstructure = getattr(problem_obj, 'jacobianstructure', None)
        self._hessian = getattr(problem_obj, 'hessian', None)
        self._hessianstructure = getattr(problem_obj, 'hessianstructure', None)
        self._intermediate = getattr(problem_obj, 'intermediate', None)

        cdef Index nele_jac = self._m * self._n
        cdef Index nele_hess = int(self._n * (self._n - 1) / 2)

        if self._jacobianstructure:
            ret_val = self._jacobianstructure()
            nele_jac = len(ret_val[0])

        if self._hessianstructure:
            ret_val = self._hessianstructure()
            nele_hess = len(ret_val[0])

        # TODO: verify that the numpy arrays use C order
        self._nlp = CreateIpoptProblem(
                            n,
                            <Number*>np_lb.data,
                            <Number*>np_ub.data,
                            m,
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
        
        if self._nlp == NULL:
            raise RuntimeError('Failed to create NLP problem. Possibly memory error.')

        if self._intermediate:
            SetIntermediateCallback(self._nlp, intermediate_cb)
        
    def __dealloc__(self):
        if self._nlp != NULL:
            FreeIpoptProblem(self._nlp)
    
        self._nlp = NULL
        
    def close(self):
        """
        Deallcate memory resources used by the IPOPT package. Called implicitly
        by the 'problem' class destructor.

        Parameters
        ----------
            None

        Returns
        -------
            None
        """

        if self._nlp != NULL:
            FreeIpoptProblem(self._nlp)
    
        self._nlp = NULL    
        
    def addOption(self, char* keyword, val):
        """
        Add a keyword/value option pair to the problem. See the IPOPT
        documentaion for details on available options.

        Parameters
        ----------
        keyword : string,
            Option name.

        val : string / int / float
            Value of the option. The type of val should match the option
            definition as described in the IPOPT documentation.

        Returns
        -------
            None
        """
        
        if type(val) == str:
            ret_val = AddIpoptStrOption(self._nlp, keyword, val)
        elif type(val) == float:
            ret_val = AddIpoptNumOption(self._nlp, keyword, val)
        elif type(val) == int:
            ret_val = AddIpoptIntOption(self._nlp, keyword, val)
        else:
            raise TypeError("Invalid option type")

        if not ret_val:
            raise TypeError("Error while assigning an option")
            
    def setProblemScaling(self, obj_scaling=1.0, x_scaling=None, g_scaling=None):
        """
        Optional function for setting scaling parameters for the problem.
        To use the scaling parameters set the option 'nlp_scaling_method' to
        'user-scaling'.

        Parameters
        ----------
        obj_scaling : float,
            Determines, how IPOPT should internally scale the objective function.
            For example, if this number is chosen to be 10, then IPOPT solves
            internally an optimization problem that has 10 times the value of
            the original objective. In particular, if this value is negative,
            then IPOPT will maximize the objective function instead of minimizing
            it. 

        x_scaling : array-like, shape = [n]
            The scaling factors for the variables. If None, no scaling is done.

        g_scaling : array-like, shape = [m]
            The scaling factors for the constrains. If None, no scaling is done.

        Returns
        -------
            None
        """
        
        try:
            obj_scaling = float(obj_scaling)
        except:
            raise ValueError('obj_scaling should be convertible to float type.')

        cdef Number *x_scaling_p
        cdef Number *g_scaling_p
        cdef np.ndarray[DTYPEd_t, ndim=1] np_x_scaling
        cdef np.ndarray[DTYPEd_t, ndim=1] np_g_scaling
        
        if x_scaling == None:
            x_scaling_p = NULL
        else:
            if len(x_scaling) != self._n:
                raise ValueError('x_scaling must either be None or have length n.')
            
            np_x_scaling = np.array(x_scaling, dtype=DTYPEd).flatten()
            x_scaling_p = <Number*>np_x_scaling.data
            

        if g_scaling == None:
            g_scaling_p = NULL
        else:
            if len(g_scaling) != self._m:
                raise ValueError('g_scaling must either be None or have length n.')

            np_g_scaling = np.array(g_scaling, dtype=DTYPEd).flatten()
            g_scaling_p = <Number*>np_g_scaling.data
        
        ret_val = SetIpoptProblemScaling(
            self._nlp,
            obj_scaling,
            x_scaling_p,
            g_scaling_p
            )

        if not ret_val:
            raise TypeError("Error while setting the scaling of the problem.")
            
    def solve(
            self,
            x
            ):
        """
        Solve the posed optimization problem starting at point x.
        Returns the optimal solution and an info dictionary.

        Parameters
        ----------
        x : array-like, shape = [n]
            Starting point.

        Returns
        -------
        x : array, shape = [n]
            Optimal solution.
        
        info: dictionary, with following keys

            'x':
                optimal solution
            'g':
                constraints at the optimal solution
            'obj_val':
                objective value at optimal solution
            'mult_g':
                final values of the constraint multipliers
            'mult_x_L':
                bound multipliers at the solution
            'mult_x_U':
                bound multipliers at the solution
            'status':
                gives the status of the algorithm
            'status_msg':
                gives the status of the algorithm as a message

        """
                
        if self._n != len(x):
            raise ValueError('Wrong length of x0')
            
        cdef np.ndarray[DTYPEd_t, ndim=1]  np_x = np.array(x, dtype=DTYPEd).flatten()

        cdef ApplicationReturnStatus stat
        cdef np.ndarray[DTYPEd_t, ndim=1] g = np.zeros((self._m,), dtype=DTYPEd)
        cdef np.ndarray[DTYPEd_t, ndim=1] mult_g = np.zeros((self._m,), dtype=DTYPEd)
        cdef np.ndarray[DTYPEd_t, ndim=1] mult_x_L = np.zeros((self._n,), dtype=DTYPEd)
        cdef np.ndarray[DTYPEd_t, ndim=1] mult_x_U = np.zeros((self._n,), dtype=DTYPEd)
        
        cdef Number obj_val = 0
        
        stat = IpoptSolve(
                    self._nlp,
                    <Number*>np_x.data,
                    <Number*>g.data,
                    &obj_val,
                    <Number*>mult_g.data,
                    <Number*>mult_x_L.data,
                    <Number*>mult_x_U.data,
                    <UserDataPtr>self
                    )
        
        info = {
            'x': np_x,
            'g': g,
            'obj_val': obj_val,
            'mult_g': mult_g,
            'mult_x_L': mult_x_L,
            'mult_x_U': mult_x_U,
            'status': stat,
            'status_msg': STATUS_MESSAGES[stat]
            }
            
        return np_x, info
        
#
# Callback functions
#
cdef Bool objective_cb(
            Index n,
            Number* x,
            Bool new_x,
            Number* obj_value,
            UserDataPtr user_data
            ):

    log('objective_cb', logging.INFO)
    
    cdef object self = <object>user_data
    cdef Index i
    cdef np.ndarray[DTYPEd_t, ndim=1] _x = np.zeros((n,), dtype=DTYPEd)
    for i in range(n):
        _x[i] = x[i]
        
    obj_value[0] = self._objective(_x)
    
    return True
    
cdef Bool gradient_cb(
            Index n,
            Number* x,
            Bool new_x,
            Number* grad_f,
            UserDataPtr user_data
            ):

    log('gradient_cb', logging.INFO)
    
    cdef object self = <object>user_data
    cdef Index i
    cdef np.ndarray[DTYPEd_t, ndim=1] _x = np.zeros((n,), dtype=DTYPEd)
    cdef np.ndarray[DTYPEd_t, ndim=1] np_grad_f
    
    for i in range(n):
        _x[i] = x[i]
        
    ret_val = self._gradient(_x)
    
    np_grad_f = np.array(ret_val, dtype=DTYPEd).flatten()
    
    for i in range(n):
        grad_f[i] = np_grad_f[i]
    
    return True
    
cdef Bool constraints_cb(
            Index n,
            Number* x,
            Bool new_x,
            Index m,
            Number* g,
            UserDataPtr user_data
            ):

    log('constraints_cb', logging.INFO)

    cdef object self = <object>user_data
    cdef Index i
    cdef np.ndarray[DTYPEd_t, ndim=1] _x = np.zeros((n,), dtype=DTYPEd)
    cdef np.ndarray[DTYPEd_t, ndim=1] np_g
    
    for i in range(n):
        _x[i] = x[i]
        
    ret_val = self._constraints(_x)
    
    np_g = np.array(ret_val, dtype=DTYPEd).flatten()
    
    for i in range(m):
        g[i] = np_g[i]
    
    return True

cdef Bool jacobian_cb(
            Index n,
            Number* x,
            Bool new_x,
            Index m,
            Index nele_jac,
            Index *iRow,
            Index *jCol,
            Number *values,
            UserDataPtr user_data
            ):

    log('jacobian_cb', logging.INFO)

    cdef object self = <object>user_data
    cdef Index i
    cdef np.ndarray[DTYPEd_t, ndim=1] _x = np.zeros((n,), dtype=DTYPEd)
    cdef np.ndarray[DTYPEi_t, ndim=1] np_iRow
    cdef np.ndarray[DTYPEi_t, ndim=1] np_jCol
    cdef np.ndarray[DTYPEd_t, ndim=1] np_jac_g

    if values == NULL:
        log('Querying for iRow/jCol indices of the jacobian', logging.INFO)
        
        if not self._jacobianstructure:
            #
            # Assuming a dense Jacobian
            #
            s = np.unravel_index(np.arange(self._m*self._n), (self._m, self._n))
            np_iRow = np.array(s[0], dtype=DTYPEi)
            np_jCol = np.array(s[1], dtype=DTYPEi)
        else:
            #
            # Sparse Jacobian
            #
            ret_val = self._jacobianstructure()
            
            np_iRow = np.array(ret_val[0], dtype=DTYPEi).flatten()
            np_jCol = np.array(ret_val[1], dtype=DTYPEi).flatten()
        
        for i in range(nele_jac):
            iRow[i] = np_iRow[i]
            jCol[i] = np_jCol[i]
    else:
        log('Querying for jacobian', logging.INFO)
        
        for i in range(n):
            _x[i] = x[i]
        
        ret_val = self._jacobian(_x)
        
        np_jac_g = np.array(ret_val, dtype=DTYPEd).flatten()
        
        for i in range(nele_jac):
            values[i] = np_jac_g[i]
        
    return True

cdef Bool hessian_cb(
            Index n,
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
            ):

    log('hessian_cb', logging.INFO)

    cdef object self = <object>user_data
    cdef Index i
    cdef np.ndarray[DTYPEd_t, ndim=1] _x = np.zeros((n,), dtype=DTYPEd)
    cdef np.ndarray[DTYPEd_t, ndim=1] _lambda = np.zeros((m,), dtype=DTYPEd)    
    cdef np.ndarray[DTYPEi_t, ndim=1] np_iRow
    cdef np.ndarray[DTYPEi_t, ndim=1] np_jCol
    cdef np.ndarray[DTYPEd_t, ndim=1] np_h
    
    if values == NULL:
        log('Querying for iRow/jCol indices of the hessian', logging.INFO)
        
        if not self._hessianstructure:
            #
            # Assuming a lower triangle Hessian
            # Note:
            # There is a need to reconvert the s.col and s.row to arrays
            # because they have the wrong stride
            #
            s = sps.coo_matrix(np.tril(np.ones((self._n, self._n))))
            np_iRow = np.array(s.col, dtype=DTYPEi)
            np_jCol = np.array(s.row, dtype=DTYPEi)
        else:
            #
            # Sparse Hessian
            #
            ret_val = self._hessianstructure()
            
            np_iRow = np.array(ret_val[0], dtype=DTYPEi).flatten()
            np_jCol = np.array(ret_val[1], dtype=DTYPEi).flatten()
        
        for i in range(nele_hess):
            iRow[i] = np_iRow[i]
            jCol[i] = np_jCol[i]
    else:
        if not self._hessian:
            return False
        
        for i in range(n):
            _x[i] = x[i]
        
        for i in range(m):
            _lambda[i] = lambd[i]
            
        ret_val = self._hessian(_x, _lambda, obj_factor)
        
        np_h = np.array(ret_val, dtype=DTYPEd).flatten()

        for i in range(nele_hess):
            values[i] = np_h[i]
        
    return True


cdef Bool intermediate_cb(
            Index alg_mod,
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
            ):

    log('intermediate_cb', logging.INFO)

    cdef object self = <object>user_data

    ret_val = self._intermediate(
        alg_mod,
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

    if ret_val == None:
        return True
    
    return ret_val
