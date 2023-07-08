class CyIpoptEvaluationError(ArithmeticError):
    """An exception that should be raised in evaluation callbacks to signal
    to CyIpopt that a numerical error occured during function evaluation.
    Whereas most exceptions that occur in callbacks are re-raised, exceptions
    of this type are ignored other than communicating to Ipopt that an error
    occurred.

    """

    pass
