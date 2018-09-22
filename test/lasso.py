#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cyipopt: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012-2015 Amit Aides
Copyright (C) 2015-2018 Matthias Kümmerer

Author: Matthias Kümmerer <matthias.kuemmerer@bethgelab.org>
(original Author: Amit Aides <amitibo@tx.technion.ac.il>)
URL: https://github.com/matthias-k/cyipopt
License: EPL 1.0

Usage::

   $ python lasso.py
   $ python lasso.py -p
   $ python lasso.py --plot
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
import ipopt


class lasso(ipopt.problem):
    def __init__(self, A, y):

        self._A = A
        self._y = y
        self._m = A.shape[1]

        #
        # The constraint functions are bounded from below by zero.
        #
        cl = np.zeros(2*self._m)

        super(lasso, self).__init__(
                            2*self._m,
                            2*self._m,
                            cl=cl
                            )

        #
        # Set solver options
        #
        self.addOption('derivative_test', 'second-order')
        self.addOption('jac_d_constant', 'yes')
        self.addOption('hessian_constant', 'yes')
        self.addOption('mu_strategy', 'adaptive')
        self.addOption('max_iter', 100)
        self.addOption('tol', 1e-8)

    def solve(self, _lambda):

        x0 = np.concatenate((np.zeros(m), np.ones(m)))
        self._lambda = _lambda
        x, info = super(lasso, self).solve(x0)

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

        #
        # Create a sparse matrix to hold the jacobian structure
        #
        return np.nonzero(np.tile(np.eye(self._m), (2, 2)))

    def jacobian(self, x):

        I = np.eye(self._m)

        J = np.vstack((np.hstack((I, I)), np.hstack((-I, I))))

        row, col = self.jacobianstructure()

        return J[row, col]

    def hessianstructure(self):

        h = np.zeros((2*self._m, 2*self._m))
        h[:self._m, :self._m] = np.tril(np.ones((self._m, self._m)))

        #
        # Create a sparse matrix to hold the hessian structure
        #
        return np.nonzero(h)

    def hessian(self, x, lagrange, obj_factor):

        H = np.zeros((2*self._m, 2*self._m))
        H[:self._m, :self._m] = np.tril(np.tril(np.dot(self._A.T, self._A)))

        row, col = self.hessianstructure()

        return obj_factor*H[row, col]


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--plot', help='Plot results with matplotlib',
                        action='store_true')
    args = parser.parse_args()


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

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(_lambdas, estim_betas)
        plt.xlabel('L1 Regularization Parameter')
        plt.ylabel('Estimated Regression Coefficients')
        plt.title('Lasso Solution as Function of The Regularization Parameter')
        plt.show()
