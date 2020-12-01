import numpy as np
import pytest

import cyipopt


def test_hs071_solve(hs071_initial_guess_fixture, hs071_problem_instance_fixture):
	"""Test hs071 test problem solves to the correct solution."""
	x0 = hs071_initial_guess_fixture
	nlp = hs071_problem_instance_fixture
	x, info = nlp.solve(x0)

	expected_J = 17.01401714021362
	np.testing.assert_almost_equal(info["obj_val"], expected_J)

	expected_x = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
	np.testing.assert_allclose(x, expected_x)
