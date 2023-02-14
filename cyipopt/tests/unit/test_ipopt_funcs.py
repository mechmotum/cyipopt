import numpy as np
import pytest

import cyipopt

pre_3_14_0 = (
    cyipopt.IPOPT_VERSION[0] < 3
    or (cyipopt.IPOPT_VERSION[0] == 3 and cyipopt.IPOPT_VERSION[1] < 14)
)


@pytest.mark.skipif(True, reason="This segfaults. Ideally, it fails gracefully")
def test_get_iterate_uninit(hs071_problem_instance_fixture):
    """Test that we can call get_current_iterate on an uninitialized problem
    """
    nlp = hs071_problem_instance_fixture
    x, zL, zU, g, lam = nlp.get_current_iterate()


@pytest.mark.skipif(True, reason="This also segfaults")
def test_get_iterate_postsolve(
    hs071_initial_guess_fixture,
    hs071_problem_instance_fixture,
):
    x0 = hs071_initial_guess_fixture
    nlp = hs071_problem_instance_fixture
    x, info = nlp.solve(x0)

    x_iter, zL, zU, g, lam = nlp.get_current_iterate()
    expected_x = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
    np.testing.assert_allclose(x, expected_x)


@pytest.mark.skipif(True, reason="Segfaults")
def test_get_violations_uninit(hs071_problem_instance_fixture):
    nlp = hs071_problem_instance_fixture
    violations = nlp.get_current_violations()


@pytest.mark.skipif(True, reason="Segfaults")
def test_get_violations_postsolve(
    hs071_initial_guess_fixture,
    hs071_problem_instance_fixture,
):
    x0 = hs071_initial_guess_fixture
    nlp = hs071_problem_instance_fixture
    x, info = nlp.solve(x0)

    violations = nlp.get_current_violations()
    expected_x = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
    np.testing.assert_allclose(x, expected_x)


@pytest.mark.skipif(
    not pre_3_14_0,
    reason="GetIpoptCurrentIterate was introduced in Ipopt v3.14.0",
)
def test_get_iterate_fail_pre_3_14_0(hs071_problem_instance_fixture):
    nlp = hs071_problem_instance_fixture
    with pytest.raises(RuntimeError):
        # TODO: Test error message
        x, zL, zU, g, lam = nlp.get_current_iterate()


@pytest.mark.skipif(
    not pre_3_14_0,
    reason="GetIpoptCurrentViolations was introduced in Ipopt v3.14.0",
)
def test_get_violations_fail_pre_3_14_0(hs071_problem_instance_fixture):
    nlp = hs071_problem_instance_fixture
    with pytest.raises(RuntimeError):
        # TODO: Test error message
        violations = nlp.get_current_violations()


@pytest.mark.skipif(
    pre_3_14_0,
    reason="GetIpoptCurrentIterate was introduced in Ipopt v3.14.0",
)
def test_get_iterate_hs071(
    hs071_initial_guess_fixture,
    hs071_definition_instance_fixture,
    hs071_variable_lower_bounds_fixture,
    hs071_variable_upper_bounds_fixture,
    hs071_constraint_lower_bounds_fixture,
    hs071_constraint_upper_bounds_fixture,
):
    x0 = hs071_initial_guess_fixture
    lb = hs071_variable_lower_bounds_fixture
    ub = hs071_variable_upper_bounds_fixture
    cl = hs071_constraint_lower_bounds_fixture
    cu = hs071_constraint_upper_bounds_fixture
    n = len(x0)
    m = len(cl)

    problem_definition = hs071_definition_instance_fixture

    #
    # Define a callback that uses some "global" information to call
    # get_current_iterate and store the result
    #
    x_iterates = []
    def intermediate(
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
        ls_trials,
    ):
        # CyIpopt's C wapper expects a callback with this signature. If we
        # implemented this as a method on problem_definition, we could store
        # and access global information on self.

        # This callback must be defined before constructing the Problem, but can
        # be defined after (or as part of) problem_definition. If we attach the
        # Problem to the "definition", then we can call get_current_iterate
        # from this callback.
        iterate = problem_definition.nlp.get_current_iterate(scaled=False)
        x, zL, zU, g, lam = iterate
        x_iterates.append(x)

        # Hack so we may get the number of iterations after the solve
        problem_definition.iter_count = iter_count

    # Replace "intermediate" attribute with our callback, which knows
    # about the "Problem", and therefore can call get_current_iterate.
    problem_definition.intermediate = intermediate

    nlp = cyipopt.Problem(
        n=n,
        m=m,
        problem_obj=problem_definition,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )
    # Add nlp (the "Problem") as an attribute on our "problem definition".
    # This way we can call methods on the Problem, like get_current_iterate,
    # during the solve.
    problem_definition.nlp = nlp

    # Disable bound push to make testing easier
    nlp.add_option("bound_push", 1e-9)
    x, info = nlp.solve(x0)

    # Assert correct solution
    expected_x = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
    np.testing.assert_allclose(x, expected_x)

    #
    # Assert some very basic information about the collected primal iterates
    #
    assert len(x_iterates) == (1 + problem_definition.iter_count)

    # These could be different due to bound_push (and scaling)
    np.testing.assert_allclose(x_iterates[0], x0)

    # These could be different due to honor_original_bounds (and scaling)
    np.testing.assert_allclose(x_iterates[-1], x)


@pytest.mark.skipif(
    pre_3_14_0,
    reason="GetIpoptCurrentViolations was introduced in Ipopt v3.14.0",
)
def test_get_violations_hs071(
    hs071_initial_guess_fixture,
    hs071_definition_instance_fixture,
    hs071_variable_lower_bounds_fixture,
    hs071_variable_upper_bounds_fixture,
    hs071_constraint_lower_bounds_fixture,
    hs071_constraint_upper_bounds_fixture,
):
    x0 = hs071_initial_guess_fixture
    lb = hs071_variable_lower_bounds_fixture
    ub = hs071_variable_upper_bounds_fixture
    cl = hs071_constraint_lower_bounds_fixture
    cu = hs071_constraint_upper_bounds_fixture
    n = len(x0)
    m = len(cl)

    problem_definition = hs071_definition_instance_fixture

    pr_violations = []
    du_violations = []
    def intermediate(
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
        ls_trials,
    ):
        violations = problem_definition.nlp.get_current_violations(scaled=True)
        (
            xL_viol, xU_viol, xL_compl, xU_compl, grad_lag, g_viol, g_compl
        ) = violations
        pr_violations.append(g_viol)
        du_violations.append(grad_lag)

        # Hack so we may get the number of iterations after the solve
        problem_definition.iter_count = iter_count

    problem_definition.intermediate = intermediate
    nlp = cyipopt.Problem(
        n=n,
        m=m,
        problem_obj=problem_definition,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )
    problem_definition.nlp = nlp

    nlp.add_option("tol", 1e-8)
    x, info = nlp.solve(x0)

    # Assert correct solution
    expected_x = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
    np.testing.assert_allclose(x, expected_x)

    #
    # Assert some very basic information about the collected violations
    #
    assert len(pr_violations) == (1 + problem_definition.iter_count)
    assert len(du_violations) == (1 + problem_definition.iter_count)

    #
    # With atol=1e-8, this check fails. This differs from what I see in the
    # Ipopt log, where inf_pr is 1.77e-11 at the final iteration. I see
    # final primal violations: [2.455637e-07, 1.770672e-11]
    # Not sure if a bug or not...
    #
    np.testing.assert_allclose(pr_violations[-1], np.zeros(m), atol=1e-6)

    np.testing.assert_allclose(du_violations[-1], np.zeros(n), atol=1e-8)
