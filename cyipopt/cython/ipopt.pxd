# -*- coding: utf-8 -*-
"""
cyipopt: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012-2015 Amit Aides
Copyright (C) 2015-2017 Matthias Kümmerer
Copyright (C) 2017-2023 cyipopt developers

License: EPL 2.0
"""

cdef extern from "IpoptConfig.h":

    int IPOPT_VERSION_MAJOR
    
    int IPOPT_VERSION_MINOR

    int IPOPT_VERSION_RELEASE


cdef extern from "IpStdCInterface.h":
    """
    #define VERSION_LT_3_14_0\
        (IPOPT_VERSION_MAJOR < 3\
            || (IPOPT_VERSION_MAJOR == 3 && IPOPT_VERSION_MINOR < 14))

    #if VERSION_LT_3_14_0
        // If not defined, define dummy versions of these functions
        Bool GetIpoptCurrentIterate(
                        IpoptProblem ipopt_problem,
                        Bool scaled,
                        Index n,
                        Number* x,
                        Number* z_L,
                        Number* z_U,
                        Index m,
                        Number* g,
                        Number* lambd
                        ){
            return 0;
        }
        Bool GetIpoptCurrentViolations(
                        IpoptProblem ipopt_problem,
                        Bool scaled,
                        Index n,
                        Number* x_L_violation,
                        Number* x_U_violation,
                        Number* compl_x_L,
                        Number* compl_x_U,
                        Number* grad_lag_x,
                        Index m,
                        Number* nlp_constraint_violation,
                        Number* compl_g
                        ){
            return 0;
        }
        #define _ip_get_iter(\
                problem, scaled, n, x, z_L, z_U, m, g, lambd\
            )\
            GetIpoptCurrentIterate(\
                problem, scaled, n, x, z_L, z_U, m, g, lambd\
            )
        #define _ip_get_viol(\
                problem, scaled, n, xL, xU, complxL, complxU, glx, m, cviol, complg\
            )\
            GetIpoptCurrentViolations(\
                problem, scaled, n, xL, xU, complxL, complxU, glx, m, cviol, complg\
            )
    #else
        #define _ip_get_iter(\
                problem, scaled, n, x, z_L, z_U, m, g, lambd\
            )\
            GetIpoptCurrentIterate(\
                problem, scaled, n, x, z_L, z_U, m, g, lambd\
            )
        #define _ip_get_viol(\
                problem, scaled, n, xL, xU, complxL, complxU, glx, m, cviol, complg\
            )\
            GetIpoptCurrentViolations(\
                problem, scaled, n, xL, xU, complxL, complxU, glx, m, cviol, complg\
            )
    #endif
    """

    ctypedef double Number

    ctypedef int Index

    ctypedef int Int

    ctypedef struct IpoptProblemInfo:
        pass

    ctypedef IpoptProblemInfo* IpoptProblem

    ctypedef int Bool

    ctypedef void* UserDataPtr

    cdef enum ApplicationReturnStatus:
        Solve_Succeeded=0
        Solved_To_Acceptable_Level=1
        Infeasible_Problem_Detected=2
        Search_Direction_Becomes_Too_Small=3
        Diverging_Iterates=4
        User_Requested_Stop=5
        Feasible_Point_Found=6
        Maximum_Iterations_Exceeded=-1
        Restoration_Failed=-2
        Error_In_Step_Computation=-3
        Maximum_CpuTime_Exceeded=-4
        Not_Enough_Degrees_Of_Freedom=-10
        Invalid_Problem_Definition=-11
        Invalid_Option=-12
        Invalid_Number_Detected=-13
        Unrecoverable_Exception=-100
        NonIpopt_Exception_Thrown=-101
        Insufficient_Memory=-102
        Internal_Error=-199

    cdef enum AlgorithmMode:
        RegularMode=0
        RestorationPhaseMode=1

    ctypedef Bool (*Eval_F_CB)(
                    Index n,
                    Number* x,
                    Bool new_x,
                    Number* obj_value,
                    UserDataPtr user_data
                    )

    ctypedef Bool (*Eval_Grad_F_CB)(
                    Index n,
                    Number* x,
                    Bool new_x,
                    Number* grad_f,
                    UserDataPtr user_data
                    )

    ctypedef Bool (*Eval_G_CB)(
                    Index n,
                    Number* x,
                    Bool new_x,
                    Index m,
                    Number* g,
                    UserDataPtr user_data
                    )

    ctypedef Bool (*Eval_Jac_G_CB)(
                    Index n,
                    Number *x,
                    Bool new_x,
                    Index m,
                    Index nele_jac,
                    Index *iRow,
                    Index *jCol,
                    Number *values,
                    UserDataPtr user_data
                    )

    ctypedef Bool (*Eval_H_CB)(
                    Index n,
                    Number *x,
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
                    )

    ctypedef Bool (*Intermediate_CB)(
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
                    )

    IpoptProblem CreateIpoptProblem(
                            Index n,
                            Number* x_L,
                            Number* x_U,
                            Index m,
                            Number* g_L,
                            Number* g_U,
                            Index nele_jac,
                            Index nele_hess,
                            Index index_style,
                            Eval_F_CB eval_f,
                            Eval_G_CB eval_g,
                            Eval_Grad_F_CB eval_grad_f,
                            Eval_Jac_G_CB eval_jac_g,
                            Eval_H_CB eval_h
                            )

    void FreeIpoptProblem(IpoptProblem ipopt_problem)

    Bool AddIpoptStrOption(IpoptProblem ipopt_problem, char* keyword, char* val)

    Bool AddIpoptNumOption(IpoptProblem ipopt_problem, char* keyword, Number val)

    Bool AddIpoptIntOption(IpoptProblem ipopt_problem, char* keyword, Int val)

    Bool OpenIpoptOutputFile(IpoptProblem ipopt_problem, char* file_name, Int print_level)

    Bool SetIpoptProblemScaling(
                    IpoptProblem ipopt_problem,
                    Number obj_scaling,
                    Number* x_scaling,
                    Number* g_scaling
                    )

    Bool SetIntermediateCallback(
                    IpoptProblem ipopt_problem,
                    Intermediate_CB intermediate_cb
                    )

    ApplicationReturnStatus IpoptSolve(
                    IpoptProblem ipopt_problem,
                    Number* x,
                    Number* g,
                    Number* obj_val,
                    Number* mult_g,
                    Number* mult_x_L,
                    Number* mult_x_U,
                    UserDataPtr user_data
                    )

    # Wrapper around GetIpoptCurrentIterate with a dummy implementation in
    # case it is not defined (i.e. Ipopt < 3.14.0)
    Bool CyGetCurrentIterate "_ip_get_iter" (
                    IpoptProblem ipopt_problem,
                    Bool scaled,
                    Index n,
                    Number* x,
                    Number* z_L,
                    Number* z_U,
                    Index m,
                    Number* g,
                    Number* lambd
                    )

    # Wrapper around GetIpoptCurrentViolations with a dummy implementation in
    # case it is not defined (i.e. Ipopt < 3.14.0)
    Bool CyGetCurrentViolations "_ip_get_viol" (
                    IpoptProblem ipopt_problem,
                    Bool scaled,
                    Index n,
                    Number* x_L_violation,
                    Number* x_U_violation,
                    Number* compl_x_L,
                    Number* compl_x_U,
                    Number* grad_lag_x,
                    Index m,
                    Number* nlp_constraint_violation,
                    Number* compl_g
                    )
