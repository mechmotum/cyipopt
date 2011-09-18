# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 12:55:23 2011

@author: amitibo
"""
import numpy as np
cimport numpy as np
from ipopt cimport *
import logging

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


cdef class problem:
    """
    Wrapper class for solving optimization problems using the C interface of
    the IPOPT package.
    
    It can be used to solve general nonlinear programming problems of the form:
        \min_{x \in \Real^n} f(x)
        s.t.
            g^L \leq g(x) \leq g^U
            x^L \leq x \leq x^U
    
    Where x are the optimization variables (possibly with upper an lower
    bounds), f(x) is the objective function and g(x) are the general nonlinear
    constraints. The constraints, g(x), have lower and upper bounds. Note that
    equality constraints can be specified by setting g_i^L = g_i^U.

    Parameters
    ----------
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
        
    objective : function pointer
        Callback function for evaluating objective function.
        The callback functions accepts two parameters x (value of the
        optimization variables at which the objective is to be evaluated) and
        user_data.
        The function should return the objective function at the point x.
        
    constraints : function pointer
        Callback function for evaluating constraint functions.
        
    gradient : function pointer
        Callback function for evaluating gradient of objective function.
        The callback functions accepts two parameters x (value of the
        optimization variables at which the gradient is to be evaluated) and
        user_data.
        The function should return the gradient of the objective function at the
        point x.
        
    jacobian : function pointer
        Callback function for evaluating Jacobian of constraint functions.
        The callback functions accepts two parameters x (value of the
        optimization variables at which the jacobian is to be evaluated) and
        user_data.
        The function should return the values of the jacobian as calculated
        using x. The values should be returned as a 1-dim numpy array (using
        the same order as you used when specifying the sparsity structure)
    
    jacobianstructure : function pointer
        Optional. Callback function that accepts no parameters and returns the
        sparsity structure of the Jacobian (the row and column indices only).
        If None, the Jacobian is assumed to be dense.
    
    hessian : function pointer
        Optional. Callback function for evaluating Hessian of the Lagrangian
        function.
        The callback functions accepts four parameters x (value of the
        optimization variables at which the hessian is to be evaluated), lambda
        (values for the constraint multipliers at which the hessian is to be
        evaluated) objective_factor the factor in front of the objective term
        in the Hessian and user_data.
        The function should return the values of the Hessian as calculated using
        x and lambda. The values should be returned as a 1-dim numpy array
        (using the same order as you used when specifying the sparsity 
        structure).
        If None, the Hessian is calculated numerically.
    
    hessianstructure : function pointer
        Optional. Callback function that accepts no parameters and returns the
        sparsity structure of the Hessian of the lagrangian (the row and column
        indices only). If None, the Hessian is assumed to be dense.
        
    Methods
    -------
    addOption(keyword, val) : None
        Add a keyword/value option pair to the problem. See the IPOPT
        documentaion for details on available options.

    solve(x, user_data) : array, dict
        Solve the posed optimization problem starting at point x. user_data
        is passed to all callback functions.
        Returns the optimial solution and an info dictionary with the following
        fields:
            'x': optimal solution
            'g': constraints at the optimal solution
            'obj_val': objective value at optimal solution
            'mult_g': final values of the constraint multipliers
            'mult_x_L': bound multipliers at the solution
            'mult_x_U': bound multipliers at the solution
            'status':  gives the status of the algorithm
            'status_msg':  gives the status of the algorithm as a message

    close() : None
        Deallcate memory resources used by the IPOPT package. Called implicitly
        by the 'problem' class destructor.
    """

    cdef IpoptProblem _nlp
    cdef public object _objective
    cdef public object _constraints
    cdef public object _gradient
    cdef public object _jacobian
    cdef public object _jacobianstructure
    cdef public object _hessian   
    cdef public object _hessianstructure
    cdef public object _user_data
    cdef public Index _n
    cdef public Index _m
    
    def __cinit__(
            self,
            lb,
            ub,
            cl,
            cu,
            objective,
            constraints,
            gradient,
            jacobian,
            jacobianstructure=None,
            hessian=None,
            hessianstructure=None
            ):
        
        cdef Index n
        n = len(lb)
        assert n == len(ub)
        self._n = n
        
        cdef np.ndarray[DTYPEd_t, ndim=1]  np_lb = np.array(lb)
        cdef np.ndarray[DTYPEd_t, ndim=1]  np_ub = np.array(ub)
        
        cdef Index m
        m = len(cl)
        assert m == len(cu)
        self._m = m
        
        cdef np.ndarray[DTYPEd_t, ndim=1]  np_cl = np.array(cl)
        cdef np.ndarray[DTYPEd_t, ndim=1]  np_cu = np.array(cu)
        
        self._objective = objective
        self._constraints = constraints
        self._gradient = gradient
        self._jacobian = jacobian
        self._jacobianstructure = jacobianstructure
        self._hessian = hessian
        self._hessianstructure = hessianstructure
        
        cdef Index nele_jac = self._m * self._n
        cdef Index nele_hess = self._n * self._n
        
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
            
    def solve(
            self,
            np.ndarray[DTYPEd_t, ndim=1] x,
            user_data
            ):
        """
        Solve the posed optimization problem starting at point x. user_data
        is passed to all callback functions.

        fields:
            'x': optimal solution
            'g': constraints at the optimal solution
            'obj_val': objective value at optimal solution
            'mult_g': final values of the constraint multipliers
            'mult_x_L': bound multipliers at the solution
            'mult_x_U': bound multipliers at the solution
            'status':  gives the status of the algorithm
            'status_msg':  gives the status of the algorithm as a message


        Parameters
        ----------
        x : array-like, shape = [n]
        
        user_data: python object
        
        Returns
        -------
        x : array, shape = [n]
            Optimal solution.
        
        info: dictionary, with following values
            'x': optimal solution
            'g': constraints at the optimal solution
            'obj_val': objective value at optimal solution
            'mult_g': final values of the constraint multipliers
            'mult_x_L': bound multipliers at the solution
            'mult_x_U': bound multipliers at the solution
            'status':  gives the status of the algorithm
            'status_msg':  gives the status of the algorithm as a message
        """
                
        self._user_data = user_data
        
        cdef ApplicationReturnStatus stat
        cdef np.ndarray[DTYPEd_t, ndim=1] g = np.zeros((self._m,), dtype=DTYPEd)
        cdef np.ndarray[DTYPEd_t, ndim=1] mult_g = np.zeros((self._m,), dtype=DTYPEd)
        cdef np.ndarray[DTYPEd_t, ndim=1] mult_x_L = np.zeros((self._n,), dtype=DTYPEd)
        cdef np.ndarray[DTYPEd_t, ndim=1] mult_x_U = np.zeros((self._n,), dtype=DTYPEd)
        
        cdef Number obj_val = 0
        
        stat = IpoptSolve(
                    self._nlp,
                    <Number*>x.data,
                    <Number*>g.data,
                    &obj_val,
                    <Number*>mult_g.data,
                    <Number*>mult_x_L.data,
                    <Number*>mult_x_U.data,
                    <UserDataPtr>self
                    )
        
        info = {
            'x': x,
            'g': g,
            'obj_val': obj_val,
            'mult_g': mult_g,
            'mult_x_L': mult_x_L,
            'mult_x_U': mult_x_U,
            'status': stat,
            'status_msg': STATUS_MESSAGES[stat]
            }
            
        return x, info
        
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
        
    obj_value[0] = self._objective(_x, <object>self._user_data)
    
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
        
    ret_val = self._gradient(_x, <object>self._user_data)
    
    np_grad_f = ret_val
    
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
        
    ret_val = self._constraints(_x, <object>self._user_data)
    
    np_g = ret_val
    
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
        log('Querying for iRow/jCol values', logging.INFO)
        
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
            
            np_iRow = np.array(ret_val[0], dtype=DTYPEi)
            np_jCol = np.array(ret_val[1], dtype=DTYPEi)
        
        for i in range(nele_jac):
            iRow[i] = np_iRow[i]
            jCol[i] = np_jCol[i]
    else:
        log('Querying for jacobian', logging.INFO)
        
        for i in range(n):
            _x[i] = x[i]
        
        ret_val = self._jacobian(_x, <object>self._user_data)
        
        np_jac_g = ret_val
        
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
        if not self._hessianstructure:
            #
            # Assuming a dense Hessian
            #
            s = np.unravel_index(np.arange(self._n*self._n), (self._n, self._n))
            np_iRow = np.array(s[0], dtype=DTYPEi)
            np_jCol = np.array(s[1], dtype=DTYPEi)
        else:
            #
            # Sparse Hessian
            #
            ret_val = self._hessianstructure()
            
            np_iRow = np.array(ret_val[0], dtype=DTYPEi)
            np_jCol = np.array(ret_val[1], dtype=DTYPEi)
        
        for i in range(nele_hess):
            iRow[i] = np_iRow[i]
            jCol[i] = np_jCol[i]
    else:
        for i in range(n):
            _x[i] = x[i]
        
        for i in range(m):
            _lambda[i] = lambd[i]
            
        ret_val = self._hessian(_x, _lambda, obj_factor, <object>self._user_data)
        
        np_h = ret_val

        for i in range(nele_hess):
            values[i] = np_h[i]
        
    return True
