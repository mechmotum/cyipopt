def test_maximum_cpu_time_exceeded(
    hs071_initial_guess_fixture, hs071_problem_instance_fixture
):
    """Test whether cyipopt properly can handle the
    "Maximum_CpuTime_Exceeded" status from IPOPT."""
    # Initialize a reference problem.
    x0 = hs071_initial_guess_fixture
    nlp = hs071_problem_instance_fixture

    # Set a ridiculously low wall time limit.
    nlp.add_option("max_wall_time", 1e-10)

    # Solve the problem and ensure that the correct IPOPT return status
    # is set.
    _, info = nlp.solve(x0)
    assert info["status"] == -5
