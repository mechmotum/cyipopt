import pytest
import cyipopt


def test_hs071_extra_arg_intermediate_with_varargs(
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

    obj_values = []
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
        problem,
        extra_arg,
        *args,
    ):
        obj_values.append(obj_value)

    problem_definition = hs071_definition_instance_fixture
    # Replace "intermediate" attribute with our callback
    problem_definition.intermediate = intermediate

    msg = "More than 12 positional arguments"
    with pytest.raises(RuntimeError, match=msg):
        nlp = cyipopt.Problem(
            n=n,
            m=m,
            problem_obj=problem_definition,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )


def test_hs071_toofew_arg_intermediate(
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

    obj_values = []
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
    ):
        obj_values.append(obj_value)

    problem_definition = hs071_definition_instance_fixture
    # Replace "intermediate" attribute with our callback
    problem_definition.intermediate = intermediate

    msg = "Invalid intermediate callback call signature"
    with pytest.raises(RuntimeError, match=msg):
        nlp = cyipopt.Problem(
            n=n,
            m=m,
            problem_obj=problem_definition,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )


def test_hs071_intermediate_with_kwds(
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

    obj_values = []
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
        problem,
        **kwds,
    ):
        obj_values.append(obj_value)

    problem_definition = hs071_definition_instance_fixture
    # Replace "intermediate" attribute with our callback
    problem_definition.intermediate = intermediate

    msg = "Variable keyword arguments are not allowed"
    with pytest.raises(RuntimeError, match=msg):
        nlp = cyipopt.Problem(
            n=n,
            m=m,
            problem_obj=problem_definition,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )
