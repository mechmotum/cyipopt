"""Tests for deprecation of old, non-PEP8 API.

These tests can be removed when the deprecated classes/methods/functions are
removed from CyIpopt.

"""


import pytest

import cyipopt


def test_ipopt_import_deprecation():
    """Ensure that old module name import raises FutureWarning to user."""
    expected_warning_msg = ("The module has been renamed to 'cyipopt' from "
                            "'ipopt'. Please import using 'import cyipopt' "
                            "and remove all uses of 'import ipopt' in your "
                            "code as this will be deprecated in a future "
                            "release.")
    with pytest.warns(FutureWarning, match=expected_warning_msg):
        import ipopt


def test_non_pep8_class_name_deprecation(hs071_defintion_instance_fixture,
                                         hs071_initial_guess_fixture,
                                         hs071_variable_lower_bounds_fixture,
                                         hs071_variable_upper_bounds_fixture,
                                         hs071_constraint_lower_bounds_fixture,
                                         hs071_constraint_upper_bounds_fixture,
                                         ):
    """Ensure use of old non-PEP8 classes API raises FutureWarning to user."""
    expected_warning_msg = ("The class named 'problem' will soon be "
                            "deprecated in CyIpopt. Please replace all uses "
                            "and use 'Problem' going forward.")
    with pytest.warns(FutureWarning, match=expected_warning_msg):
        _ = cyipopt.problem(n=len(hs071_initial_guess_fixture),
                            m=len(hs071_constraint_lower_bounds_fixture),
                            problem_obj=hs071_defintion_instance_fixture,
                            lb=hs071_variable_lower_bounds_fixture,
                            ub=hs071_variable_upper_bounds_fixture,
                            cl=hs071_constraint_lower_bounds_fixture,
                            cu=hs071_constraint_upper_bounds_fixture,
                            )


def test_non_pep8_set_logging_level_deprecation():
    """Ensure use of old non-PEP8 classes API raises FutureWarning to user."""
    expected_warning_msg = ("The function named 'setLoggingLevel' will soon "
                            "be deprecated in CyIpopt. Please replace all "
                            "uses and use 'set_logging_level' going forward.")
    with pytest.warns(FutureWarning, match=expected_warning_msg):
        cyipopt.setLoggingLevel()


def test_non_pep8_method_names_deprecation(hs071_problem_instance_fixture):
    """Ensure use of old non-PEP8 methods API raises FutureWarning to user."""
    nlp = hs071_problem_instance_fixture

    assert isinstance(nlp, cyipopt.Problem)

    expected_warning_msg = ""
    with pytest.warns(FutureWarning, match=expected_warning_msg):
        nlp.addOption("mu_strategy", "adaptive")

    expected_warning_msg = ""
    with pytest.warns(FutureWarning, match=expected_warning_msg):
        nlp.setProblemScaling(obj_scaling=2.0)


def test_deprecated_problem_can_be_subclassed():
    """`problem` can be subclassed and its args/kwargs changed."""

    class SubclassedProblem(cyipopt.problem):

        def __init__(self, *args, **kwargs):
            n = args[0]
            m = args[1]
            problem_obj = kwargs.get("problem_obj")
            lb = kwargs.get("lb")
            ub = kwargs.get("ub")
            cl = kwargs.get("cl")
            cu = kwargs.get("cu")
            super(SubclassedProblem, self).__init__(n,
                                                    m,
                                                    problem_obj=problem_obj,
                                                    lb=lb,
                                                    ub=ub,
                                                    cl=cl,
                                                    cu=cu)

        def objective(self):
            pass

        def gradient(self):
            pass

        def constraints(self):
            pass

        def jacobian(self):
            pass

    expected_warning_msg = ("The class named 'problem' will soon be "
                            "deprecated in CyIpopt. Please replace all uses "
                            "and use 'Problem' going forward.")
    with pytest.warns(FutureWarning, match=expected_warning_msg):
        _ = SubclassedProblem(2, 2, None, None, problem_obj=None, lb=None,
                              ub=None, cl=[0, 0], cu=[0, 0], other_kwarg=None)
