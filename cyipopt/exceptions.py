# -*- coding: utf-8 -*-
"""
cyipopt: Python wrapper for the Ipopt optimization package, written in Cython.

Copyright (C) 2012-2015 Amit Aides
Copyright (C) 2015-2017 Matthias Kümmerer
Copyright (C) 2017-2023 cyipopt developers

License: EPL 2.0
"""

class CyIpoptEvaluationError(ArithmeticError):
    """An exception that should be raised in evaluation callbacks to signal
    to CyIpopt that a numerical error occured during function evaluation.
    Whereas most exceptions that occur in callbacks are re-raised, exceptions
    of this type are ignored other than communicating to Ipopt that an error
    occurred.

    """

    pass
