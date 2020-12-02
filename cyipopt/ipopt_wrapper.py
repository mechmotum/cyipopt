import warnings

from .minimize import minimize_ipopt as cyipopt_minimize_ipopt


def minimize_ipopt(*args, **kwargs):
    msg = ("'minimize_ipopt' from 'ipopt.ipopt_wrapper' has been replaced by 'minimize_ipopt' from 'cyipopt.minimize'. Please "
           "import using 'from cyipopt.minimize import minimize_ipopt' (or similar) and remove all references to "
           "'ipopt.ipopt_wrapper.minimize_ipopt' in your code as this will be deprecated in a "
           "future release.")
    warnings.warn(msg, FutureWarning)
    return cyipopt_minimize_ipopt(*args, **kwargs)
