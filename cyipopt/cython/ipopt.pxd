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
                        int n,
                        ipnumber* x,
                        ipnumber* z_L,
                        ipnumber* z_U,
                        ipindex m,
                        ipnumber* g,
                        ipnumber* lambd
                        ){
            return 0;
        }
        Bool GetIpoptCurrentViolations(
                        IpoptProblem ipopt_problem,
                        Bool scaled,
                        ipindex n,
                        ipnumber* x_L_violation,
                        ipnumber* x_U_violation,
                        ipnumber* compl_x_L,
                        ipnumber* compl_x_U,
                        ipnumber* grad_lag_x,
                        ipindex m,
                        ipnumber* nlp_constraint_violation,
                        ipnumber* compl_g
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

    ctypedef double ipnumber

    ctypedef int ipindex

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
                    ipindex n,
                    ipnumber* x,
                    Bool new_x,
                    ipnumber* obj_value,
                    UserDataPtr user_data
                    )

    ctypedef Bool (*Eval_Grad_F_CB)(
                    ipindex n,
                    ipnumber* x,
                    Bool new_x,
                    ipnumber* grad_f,
                    UserDataPtr user_data
                    )

    ctypedef Bool (*Eval_G_CB)(
                    ipindex n,
                    ipnumber* x,
                    Bool new_x,
                    ipindex m,
                    ipnumber* g,
                    UserDataPtr user_data
                    )

    ctypedef Bool (*Eval_Jac_G_CB)(
                    ipindex n,
                    ipnumber *x,
                    Bool new_x,
                    ipindex m,
                    ipindex nele_jac,
                    ipindex *iRow,
                    ipindex *jCol,
        ipnumber *values,
                    UserDataPtr user_data
                    )

    ctypedef Bool (*Eval_H_CB)(
                    ipindex n,
                    ipnumber *x,
                    Bool new_x,
                    ipnumber obj_factor,
                    ipindex m,
                    ipnumber *lambd,
                    Bool new_lambda,
                    ipindex nele_hess,
                    ipindex *iRow,
                    ipindex *jCol,
                    ipnumber *values,
                    UserDataPtr user_data
                    )

    ctypedef Bool (*Intermediate_CB)(
                    ipindex alg_mod,
                    ipindex iter_count,
                    ipnumber obj_value,
                    ipnumber inf_pr,
                    ipnumber inf_du,
                    ipnumber mu,
                    ipnumber d_norm,
                    ipnumber regularization_size,
                    ipnumber alpha_du,
                    ipnumber alpha_pr,
                    ipindex ls_trials,
                    UserDataPtr user_data
                    )

    IpoptProblem CreateIpoptProblem(
                            ipindex n,
                            ipnumber* x_L,
                            ipnumber* x_U,
                            ipindex m,
                            ipnumber* g_L,
                            ipnumber* g_U,
                            ipindex nele_jac,
                            ipindex nele_hess,
                            ipindex index_style,
                            Eval_F_CB eval_f,
                            Eval_G_CB eval_g,
                            Eval_Grad_F_CB eval_grad_f,
                            Eval_Jac_G_CB eval_jac_g,
                            Eval_H_CB eval_h
                            )

    void FreeIpoptProblem(IpoptProblem ipopt_problem)

    Bool AddIpoptStrOption(IpoptProblem ipopt_problem, char* keyword, char* val)

    Bool AddIpoptNumOption(IpoptProblem ipopt_problem, char* keyword, ipnumber val)

    Bool AddIpoptIntOption(IpoptProblem ipopt_problem, char* keyword, int val)

    Bool OpenIpoptOutputFile(IpoptProblem ipopt_problem, char* file_name, int print_level)

    Bool SetIpoptProblemScaling(
                    IpoptProblem ipopt_problem,
                    ipnumber obj_scaling,
                    ipnumber* x_scaling,
                    ipnumber* g_scaling
                    )

    Bool SetIntermediateCallback(
                    IpoptProblem ipopt_problem,
                    Intermediate_CB intermediate_cb
                    )

    ApplicationReturnStatus IpoptSolve(
                    IpoptProblem ipopt_problem,
                    ipnumber* x,
                    ipnumber* g,
                    ipnumber* obj_val,
                    ipnumber* mult_g,
                    ipnumber* mult_x_L,
                    ipnumber* mult_x_U,
                    UserDataPtr user_data
                    )

    # Wrapper around GetIpoptCurrentIterate with a dummy implementation in
    # case it is not defined (i.e. Ipopt < 3.14.0)
    Bool CyGetCurrentIterate "_ip_get_iter" (
                    IpoptProblem ipopt_problem,
                    Bool scaled,
                    ipindex n,
                    ipnumber* x,
                    ipnumber* z_L,
                    ipnumber* z_U,
                    ipindex m,
                    ipnumber* g,
                    ipnumber* lambd
                    )

    # Wrapper around GetIpoptCurrentViolations with a dummy implementation in
    # case it is not defined (i.e. Ipopt < 3.14.0)
    Bool CyGetCurrentViolations "_ip_get_viol" (
                    IpoptProblem ipopt_problem,
                    Bool scaled,
                    ipindex n,
                    ipnumber* x_L_violation,
                    ipnumber* x_U_violation,
                    ipnumber* compl_x_L,
                    ipnumber* compl_x_U,
                    ipnumber* grad_lag_x,
                    ipindex m,
                    ipnumber* nlp_constraint_violation,
                    ipnumber* compl_g
                    )
