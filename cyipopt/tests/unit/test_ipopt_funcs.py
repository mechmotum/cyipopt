import numpy as np
import pytest

import cyipopt

pre_3_14_0 = (
    cyipopt.IPOPT_VERSION[0] < 3
    or (cyipopt.IPOPT_VERSION[0] == 3 and cyipopt.IPOPT_VERSION[1] < 14)
)


@pytest.mark.skipif(
    pre_3_14_0,
    reason="GetIpoptCurrentViolations was introduced in Ipopt v3.14.0",
    # skip these tests in old versions as the version check happens before
    # the __in_ipopt_solve check
)
def test_get_iterate_uninit(hs071_problem_instance_fixture):
    """Test that we can call get_current_iterate on an uninitialized problem
    """
    nlp = hs071_problem_instance_fixture
    msg = "can only be called during a call to solve"
    with pytest.raises(RuntimeError, match=msg):
        iterate = nlp.get_current_iterate()


@pytest.mark.skipif(
    pre_3_14_0,
    reason="GetIpoptCurrentViolations was introduced in Ipopt v3.14.0",
)
def test_get_iterate_postsolve(
    hs071_initial_guess_fixture,
    hs071_problem_instance_fixture,
):
    x0 = hs071_initial_guess_fixture
    nlp = hs071_problem_instance_fixture
    x, info = nlp.solve(x0)

    msg = "can only be called during a call to solve"
    with pytest.raises(RuntimeError, match=msg):
        iterate = nlp.get_current_iterate()
    expected_x = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
    np.testing.assert_allclose(x, expected_x)


@pytest.mark.skipif(
    pre_3_14_0,
    reason="GetIpoptCurrentViolations was introduced in Ipopt v3.14.0",
)
def test_get_violations_uninit(hs071_problem_instance_fixture):
    nlp = hs071_problem_instance_fixture
    msg = "can only be called during a call to solve"
    with pytest.raises(RuntimeError, match=msg):
        violations = nlp.get_current_violations()


@pytest.mark.skipif(
    pre_3_14_0,
    reason="GetIpoptCurrentViolations was introduced in Ipopt v3.14.0",
)
def test_get_violations_postsolve(
    hs071_initial_guess_fixture,
    hs071_problem_instance_fixture,
):
    x0 = hs071_initial_guess_fixture
    nlp = hs071_problem_instance_fixture
    x, info = nlp.solve(x0)

    msg = "can only be called during a call to solve"
    with pytest.raises(RuntimeError, match=msg):
        violations = nlp.get_current_violations()
    expected_x = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
    np.testing.assert_allclose(x, expected_x)


@pytest.mark.skipif(
    not pre_3_14_0,
    reason="GetIpoptCurrentIterate was introduced in Ipopt v3.14.0",
)
def test_get_iterate_fail_pre_3_14_0(hs071_problem_instance_fixture):
    nlp = hs071_problem_instance_fixture
    # Note that the version check happens before the __in_ipopt_solve
    # check, so we don't need to call solve to test this.
    msg = "only supports Ipopt version >=3.14.0"
    with pytest.raises(RuntimeError, match=msg):
        iterate = nlp.get_current_iterate()


@pytest.mark.skipif(
    not pre_3_14_0,
    reason="GetIpoptCurrentViolations was introduced in Ipopt v3.14.0",
)
def test_get_violations_fail_pre_3_14_0(hs071_problem_instance_fixture):
    nlp = hs071_problem_instance_fixture
    # Note that the version check happens before the __in_ipopt_solve
    # check, so we don't need to call solve to test this.
    msg = "only supports Ipopt version >=3.14.0"
    with pytest.raises(RuntimeError, match=msg):
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
    """This test demonstrates a hacky way to call get_current_iterate from an
    intermediate callback without subclassing Problem. This is not recommended.
    """
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
        # test_get_iterate_hs071_subclass_Problem tests this below.

        # This callback must be defined before constructing the Problem, but can
        # be defined after (or as part of) problem_definition. If we attach the
        # Problem to the "definition", then we can call get_current_iterate
        # from this callback.
        iterate = problem_definition.nlp.get_current_iterate(scaled=False)
        x_iterates.append(iterate["x"])

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
    reason="GetIpoptCurrentIterate was introduced in Ipopt v3.14.0",
)
def test_get_iterate_hs071_subclass_Problem(
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

    x_iterates = []
    class MyProblem(cyipopt.Problem):

        def objective(self, x):
            return problem_definition.objective(x)

        def gradient(self, x):
            return problem_definition.gradient(x)

        def constraints(self, x):
            return problem_definition.constraints(x)

        def jacobian(self, x):
            return problem_definition.jacobian(x)

        def jacobian_structure(self, x):
            return problem_definition.jacobian_structure(x)

        def hessian(self, x, lagrange, obj_factor):
            return problem_definition.hessian(x, lagrange, obj_factor)

        def hessian_structure(self, x):
            return problem_definition.hessian_structure(x)

        def intermediate(
            self,
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
            # By subclassing Problem, we can call get_current_iterate
            # without any "global" information
            iterate = self.get_current_iterate(scaled=False)
            x_iterates.append(iterate["x"])

            # Hack so we may get the number of iterations after the solve
            self.iter_count = iter_count

    nlp = MyProblem(
        n=n,
        m=m,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )

    # Disable bound push to make testing easier
    nlp.add_option("bound_push", 1e-9)
    x, info = nlp.solve(x0)

    # Assert correct solution
    expected_x = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
    np.testing.assert_allclose(x, expected_x)

    #
    # Assert some very basic information about the collected primal iterates
    #
    assert len(x_iterates) == (1 + nlp.iter_count)

    # These could be different due to bound_push (and scaling)
    np.testing.assert_allclose(x_iterates[0], x0)

    # These could be different due to honor_original_bounds (and scaling)
    np.testing.assert_allclose(x_iterates[-1], x)


@pytest.mark.skipif(
    pre_3_14_0,
    reason="GetIpoptCurrentViolations was introduced in Ipopt v3.14.0",
)
def test_get_violations_hs071_subclass_Problem(
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
    iter_counts = []
    class MyProblem(cyipopt.Problem):

        def objective(self, x):
            return problem_definition.objective(x)

        def gradient(self, x):
            return problem_definition.gradient(x)

        def constraints(self, x):
            return problem_definition.constraints(x)

        def jacobian(self, x):
            return problem_definition.jacobian(x)

        def jacobian_structure(self, x):
            return problem_definition.jacobian_structure(x)

        def hessian(self, x, lagrange, obj_factor):
            return problem_definition.hessian(x, lagrange, obj_factor)

        def hessian_structure(self, x):
            return problem_definition.hessian_structure(x)

        def intermediate(
            self,
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
            violations = self.get_current_violations(scaled=True)
            pr_violations.append(violations["g_violation"])
            du_violations.append(violations["grad_lag_x"])

            # Hack so we may get the number of iterations after the solve
            iter_counts.append(iter_count)

    nlp = MyProblem(
        n=n,
        m=m,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )

    nlp.add_option("tol", 1e-8)
    # Note that Ipopt appears to check tolerance in the scaled, bound-relaxed
    # NLP. To ensure our intermediate infeasibilities, which are in the user's
    # original NLP, are less than the above tolerance at the final iteration,
    # we must turn off bound relaxation.
    nlp.add_option("bound_relax_factor", 0.0)
    x, info = nlp.solve(x0)

    # Assert correct solution
    expected_x = np.array([1.0, 4.74299964, 3.82114998, 1.37940829])
    np.testing.assert_allclose(x, expected_x)

    #
    # Assert some very basic information about the collected violations
    #
    iter_count = iter_counts[-1]
    assert len(pr_violations) == (1 + iter_count)
    assert len(du_violations) == (1 + iter_count)

    np.testing.assert_allclose(pr_violations[-1], np.zeros(m), atol=1e-8)
    np.testing.assert_allclose(du_violations[-1], np.zeros(n), atol=1e-8)


def test_intermediate_cb(
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
        return False

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
    x, info = nlp.solve(x0)
    assert b'premature termination' in info['status_msg']
