# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 12:55:23 2011

@author: amitibo
"""
#
# Implementation of example problem Ipopt/examples/hs071
# based on the work of Eric Xu author of pyipopt.
#

import numpy as np
import ipopt

lb = [1.0, 1.0, 1.0, 1.0]
ub = [5.0, 5.0, 5.0, 5.0]

cl = [25.0, 40.0]
cu = [2.0e19, 40.0]

def objective(x, user_data=None):
    return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

def gradient(x, user_data=None):
    return np.array([
                x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]) , 
                x[0] * x[3],
                x[0] * x[3] + 1.0,
                x[0] * (x[0] + x[1] + x[2])
                ])
                	
def constraints(x, user_data=None):
    return np.array([
                x[0] * x[1] * x[2] * x[3], 
                x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]
                ])


def jacobianstructure():
    return (np.array([0, 0, 0, 0, 1, 1, 1, 1]), 
            np.array([0, 1, 2, 3, 0, 1, 2, 3]))

def jacobian(x, user_data = None):
    return np.array([
                x[1]*x[2]*x[3],
                x[0]*x[2]*x[3], 
                x[0]*x[1]*x[3], 
                x[0]*x[1]*x[2],
                2.0*x[0], 
                2.0*x[1],
                2.0*x[2], 
                2.0*x[3]
                ])
		
def hessianstructure():
    hrow = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    hcol = [0, 0, 1, 0, 1, 2, 0, 1, 2, 3]
    return (np.array(hcol), np.array(hrow))
        
def hessian(x, lagrange, obj_factor, user_data=None):

    values = np.zeros((10,))
    
    values[0] = obj_factor * (2*x[3])
    values[1] = obj_factor * (x[3])
    values[2] = 0
    values[3] = obj_factor * (x[3])
    values[4] = 0
    values[5] = 0
    values[6] = obj_factor * (2*x[0] + x[1] + x[2])
    values[7] = obj_factor * (x[0])
    values[8] = obj_factor * (x[0])
    values[9] = 0
    
    values[1] += lagrange[0] * (x[2] * x[3])
    values[3] += lagrange[0] * (x[1] * x[3])
    values[4] += lagrange[0] * (x[0] * x[3])
    values[6] += lagrange[0] * (x[1] * x[2])
    values[7] += lagrange[0] * (x[0] * x[2])
    values[8] += lagrange[0] * (x[0] * x[1])
    
    values[0] += lagrange[1] * 2
    values[2] += lagrange[1] * 2
    values[5] += lagrange[1] * 2
    values[9] += lagrange[1] * 2
    
    return values

def main():
    nlp = ipopt.problem(
                lb,
                ub,
                cl,
                cu,
                objective,
                constraints,
                gradient,
                jacobian,
                jacobianstructure,
                hessian,
                hessianstructure
                )
                
    nlp.addOption('derivative_test', 'second-order')
    
    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    
    x, info = nlp.solve(x0, None)
    
    print "Solution of the primal variables: x=%s\n" % repr(x)
    
    print "Solution of the dual variables: lambda=%s\n" % repr(info['mult_g'])
    
    print "Objective=%s\n" % repr(info['obj_val'])
        

if __name__ == '__main__':
    main()
