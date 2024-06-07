# -*- coding: utf-8 -*-
"""
cyipopt: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012-2015 Amit Aides
Copyright (C) 2015-2017 Matthias Kümmerer
Copyright (C) 2017-2024 cyipopt developers

License: EPL 2.0
"""

class CyIpoptEvaluationError(ArithmeticError):
    """An exception that should be raised in evaluation callbacks to signal
    to CyIpopt that a numerical error occured during function evaluation.

    Whereas most exceptions that occur in callbacks are re-raised, exceptions
    of this type are ignored other than to communicate to Ipopt that an error
    occurred.

    Ipopt handles evaluation errors differently depending on where they are
    raised (which evaluation callback returns ``false`` to Ipopt).
    When evaluation errors are raised in the following callbacks, Ipopt
    attempts to recover by cutting the step size. This is usually the desired
    behavior when an undefined value is encountered.

    - ``objective``
    - ``constraints``

    When raised in the following callbacks, Ipopt fails with an "Invalid number"
    return status.

    - ``gradient``
    - ``jacobian``
    - ``hessian``

    Raising an evaluation error in the following callbacks results is not
    supported.

    - ``jacobianstructure``
    - ``hessianstructure``
    - ``intermediate``

    """

    pass
