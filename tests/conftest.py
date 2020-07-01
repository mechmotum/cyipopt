"""Fixture-sharing file for test suite.

Fixtures within this file can be shared across both unit and functional test 
directories to speed up test setup.
"""

import numpy as np
import pytest

import ipopt


@pytest.fixture(scope=module)
def hs071_fixture():
	"""Return a default implementation of the hs071 test problem.

	Scope limited to module to speed up tests but allow for modification of 
	fixture within different modules.
	"""

	class hs071:
		"""The hs071 test problem also found in examples."""

		def __init__(self):
			pass

		def objective(self, x):
			return x[0] * x[3] * np.sum(x[0:3]) + x[2]

		def gradient(self, x):
			return np.array([
				x[0] * x[3] + x[3] * np.sum(x[0:3]),
				x[0] * x[3],
				x[0] * x[3] + 1.0,
				x[0] * np.sum(x[0:3])
				])

		def constraints(self, x):
			return np.array((np.prod(x), np.dot(x, x)))

		def jacobian(self, x):
			return np.concatenate((np.prod(x) / x, 2*x))

		def hessian_structure(self):
			return np.nonzero(np.tril(np.ones((4, 4))))

		def hessian(self, x, lagrange, obj_factor):
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
			row, col = self.hessianstructure()
			return H[row, col]

		def intermediate(*args):
			iter_count = args[2]
			obj_value = args[3]
			msg = f"Objective value at iteration #{iter_count} is - {obj_value}"
			print(msg)


	problem_instance = hs071()
	return problem_instance
