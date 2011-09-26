# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 12:55:23 2011

@author: amitibo
"""
#
# Test the "ipopt" Matlab interface on the Hock & Schittkowski test problem
# #71. See: Willi Hock and Klaus Schittkowski. (1981) Test Examples for
# Nonlinear Programming Codes. Lecture Notes in Economics and Mathematical
# Systems Vol. 187, Springer-Verlag.
#
# Based on matlab code by Peter Carbonetto.
#

import numpy as np
import scipy.sparse as sps
import ipopt


class hs071(object):
    def __init__(self):
        pass
    
    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        return x[0] * x[3] * np.sum(x[0:3]) + x[2]
    
    
    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        return np.array([
                    x[0] * x[3] + x[3] * np.sum(x[0:3]), 
                    x[0] * x[3],
                    x[0] * x[3] + 1.0,
                    x[0] * np.sum(x[0:3])
                    ])
    
    
    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        return np.array((np.prod(x), np.dot(x, x)))
    
    
    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        return np.concatenate((np.prod(x) / x, 2*x))
    
    
    def hessianstructure(self):
        #
        # The structure of the Hessian
        # Note:
        # The default hessian structure is of a lower triangular matrix. Therefore
        # this function is redundant. I include it as an example for structure
        # callback.
        # 
        global hs
        
        hs = sps.coo_matrix(np.tril(np.ones((4, 4))))
        return (hs.col, hs.row)
    
    
    def hessian(self, x, lagrange, obj_factor):
        #
        # The callback for calculating the Hessian
        #
        H = obj_factor*np.array((
                (2*x[3], 0, 0, 0),
                (x[3],   0, 0, 0),
                (x[3],   0, 0, 0),
                (2*x[0]+x[1]+x[2], x[0], x[0], 0)))
                
        H += lagrange[0]*np.array((
                (0, 0, 0, 0),
                (x[2]*x[3], 0, 0, 0),
                (x[1]*x[3], x[0]*x[3], 0, 0),
                (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))
                
        H += lagrange[1]*2*np.eye(4)
    
        #
        # Note:
        # 
        #
        return H[hs.row, hs.col]


def main():
    #
    # Define the problem
    #
    x0 = [1.0, 5.0, 5.0, 1.0]
    
    lb = [1.0, 1.0, 1.0, 1.0]
    ub = [5.0, 5.0, 5.0, 5.0]
    
    cl = [25.0, 40.0]
    cu = [2.0e19, 40.0]

    nlp = ipopt.problem(
                4,
                2,
                hs071(),
                lb,
                ub,
                cl,
                cu
                )
    
    #
    # Set solver options
    #
    nlp.addOption('derivative_test', 'second-order')
    nlp.addOption('mu_strategy', 'adaptive')
    nlp.addOption('tol', 1e-7)
    
    #
    # Solve the problem
    #
    x, info = nlp.solve(x0)
    
    print "Solution of the primal variables: x=%s\n" % repr(x)
    
    print "Solution of the dual variables: lambda=%s\n" % repr(info['mult_g'])
    
    print "Objective=%s\n" % repr(info['obj_val'])


if __name__ == '__main__':
    main()
