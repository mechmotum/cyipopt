# -*- coding: utf-8 -*-
"""
cyipot: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012 Amit Aides
Author: Amit Aides <amitibo@tx.technion.ac.il>
URL: <http://http://code.google.com/p/cyipopt/>
License: EPL 1.0
"""
#
# This function executes IPOPT to find the maximum likelihood solution to
# least squares regression with L1 regularization or the "Lasso". The inputs
# are the data matrix A (in which each row is an example vector), the vector
# of regression outputs y, and the penalty parameter lambda, a number
# greater than zero. The output is the estimated vector of regression
# coefficients.
#
# Adapted from matlab code by Peter Carbonetto.
#
from __future__ import division
import numpy as np
import scipy.sparse as sps
import ipopt


class lasso(object):
    def __init__(self, A, y):
        
        self._A = A
        self._y = y
        self._m = A.shape[1]
            
        #
        # The constraint functions are bounded from below by zero.
        #
        cl = np.zeros(2*self._m)
      
        self._nlp = ipopt.problem(
                            2*self._m,
                            2*self._m,
                            self,
                            cl=cl
                            )
        
        #
        # Set solver options
        #
        self._nlp.addOption('derivative_test', 'second-order')
        self._nlp.addOption('jac_d_constant', 'yes')
        self._nlp.addOption('hessian_constant', 'yes')
        self._nlp.addOption('mu_strategy', 'adaptive')
        self._nlp.addOption('max_iter', 100)
        self._nlp.addOption('tol', 1e-8)
    
    def solve(self, _lambda):

        x0 = np.concatenate((np.zeros(m), np.ones(m)))
        self._lambda = _lambda
        x, info = self._nlp.solve(x0)
        
        return x[:self._m]
        
    def objective(self, x):

        w = x[:self._m].reshape((-1, 1))
        u = x[self._m:].reshape((-1, 1))
        
        return np.linalg.norm(self._y - np.dot(self._A, w))**2/2 + self._lambda * np.sum(u)
    
    
    def constraints(self, x):

        w = x[:self._m].reshape((-1, 1))
        u = x[self._m:].reshape((-1, 1))
        
        return np.vstack((u + w,  u - w))
    
    
    def gradient(self, x):

        w = x[:self._m].reshape((-1, 1))
        
        g = np.vstack((np.dot(-self._A.T, self._y - np.dot(self._A, w)), self._lambda*np.ones((self._m, 1))))
    
        return g
    
      
    def jacobianstructure(self):

        global js
        
        js = sps.coo_matrix(np.tile(np.eye(self._m), (2, 2)))
        return (js.row, js.col)
        
    
    def jacobian(self, x): 

        I = np.eye(self._m)
        
        J = np.vstack((np.hstack((I, I)), np.hstack((-I, I))))
        return J[js.row, js.col]   
    
    
    def hessianstructure(self):

        h = np.zeros((2*self._m, 2*self._m))
        h[:self._m,:self._m] = np.tril(np.ones((self._m, self._m)))
        
        global hs
        hs = sps.coo_matrix(h)
        
        return (hs.row, hs.col)
    
    
    def hessian(self, x, lagrange, obj_factor):

        H = np.zeros((2*self._m, 2*self._m))
        H[:self._m,:self._m] = np.tril(np.tril(np.dot(self._A.T, self._A)))
    
        return obj_factor*H[hs.row, hs.col]   


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    #
    # _lambda - Level of L1 regularization
    # n - Number of training examples
    # e - Std. dev. in noise of outputs
    # beta - "True" regression coefficients
    #
    n = 100
    e = 1
    beta = np.array((0, 0, 2, -4, 0, 0, -1, 3), dtype=np.float).reshape((-1, 1))
    
    #
    # Set the random number generator seed.
    #
    seed = 7
    np.random.seed(seed)
    
    #
    # CREATE DATA SET.
    # Generate the input vectors from the standard normal, and generate the
    # binary responses from the regression with some additional noise, and then
    # transform the results using the logistic function. The variable "beta" is
    # the set of true regression coefficients.
    #
    # A - The n x m matrix of examples
    # noise - Noise in outputs
    # y - The binary outputs
    #
    m = len(beta)
    A = np.random.randn(n, m)
    noise = e * np.random.randn(n, 1)
    y = np.dot(A, beta) + noise
    
    #
    # COMPUTE SOLUTION WITH IPOPT.
    # Compute the L1-regularized maximum likelihood estimator.
    #
    problem = lasso(A, y)
    
    LAMBDA_SAMPLES = 30
    _lambdas = np.logspace(0, 2.6, LAMBDA_SAMPLES)
    estim_betas = np.zeros((LAMBDA_SAMPLES, beta.size))
    
    for i, _lambda in enumerate(_lambdas):
        estim_betas[i, :] = problem.solve(_lambda)
        
    plt.plot(_lambdas, estim_betas)
    plt.xlabel('L1 Regularization Parameter')
    plt.ylabel('Estimated Regression Coefficients')
    plt.title('Lasso Solution as Function of The Regularization Parameter')
    plt.show()

