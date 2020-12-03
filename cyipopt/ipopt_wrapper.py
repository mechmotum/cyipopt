"""Backward-compatible module."""

import warnings

from .scipy_interface import IpoptProblemWrapper as CyipoptIpoptProblemWrapper
from .scipy_interface import convert_to_bytes as cyipopt_convert_to_bytes
from .scipy_interface import get_bounds as cyipopt_get_bounds
from .scipy_interface import get_constraint_bounds as cyipopt_get_constraint_bounds
from .scipy_interface import minimize_ipopt as cyipopt_minimize_ipopt


__all__ = ["get_bounds", "minimize_ipopt"]


def make_future_warning_error_msg(func_name):
    msg = (f"'{func_name}' from 'ipopt.ipopt_wrapper' has been replaced by "
           f"'{func_name}' from 'cyipopt.scipy_interface'. Please import using "
           f"'from cyipopt.scipy_interface import {func_name}' (or similar) and "
           f"remove all references to 'ipopt.ipopt_wrapper.{func_name}' in "
           f"your code as this will be deprecated in a future release.")
    return msg


def convert_to_bytes(*args, **kwargs):
    """Wrapper around `convert_to_bytes` for backwards compatibility."""
    msg = make_future_warning_error_msg("convert_to_bytes")
    warnings.warn(msg, FutureWarning)
    return cyipopt_convert_to_bytes(*args, **kwargs)


def get_bounds(*args, **kwargs):
    """Wrapper around `get_bounds` for backwards compatibility."""
    msg = make_future_warning_error_msg("get_bounds")
    warnings.warn(msg, FutureWarning)
    return cyipopt_get_bounds(*args, **kwargs)


def get_constraint_bounds(*args, **kwargs):
    """Wrapper around `get_constraint_bounds` for backwards compatibility."""
    msg = make_future_warning_error_msg("get_constraint_bounds")
    warnings.warn(msg, FutureWarning)
    return cyipopt_get_constraint_bounds(*args, **kwargs)


def minimize_ipopt(*args, **kwargs):
    """Wrapper around `minimize_ipopt` for backwards compatibility."""
    msg = make_future_warning_error_msg("minimize_ipopt")
    warnings.warn(msg, FutureWarning)
    return cyipopt_minimize_ipopt(*args, **kwargs)


def replace_option(options, oldname, newname):
    """Wrapper around `replace_option` for backwards compatibility."""
    msg = make_future_warning_error_msg("replace_option")
    warnings.warn(msg, FutureWarning)
    return cyipopt_replace_option(*args, **kwargs)


class IpoptProblemWrapper:
    """Wrapper around `IpoptProblemWrapper` for backwards compatibility."""

    def __new__(self, *args, **kwargs):
        msg = make_future_warning_error_msg("IpoptProblemWrapper")
        warnings.warn(msg, FutureWarning)
        return CyipoptIpoptProblemWrapper(*args, **kwargs)
