"""Module with utilities for use within CyIpopt.

Currently contains functions to aid with deprecation within CyIpopt.

"""

import warnings
from functools import wraps

from ipopt_wrapper import Problem


def deprecated_warning(new_name):
    """Decorator that issues a FutureWarning for deprecated functionality.

    Parameters
    ----------
    new_name : :obj:`str`
        The name of the object replacing the deprecated one.

    Returns
    -------
    :obj:`decorator`
        For decorating functions that will soon be deprecated.

    """

    def decorate(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if hasattr(func, "__objclass__"):
                what = "method"
                class_name = getattr(func, "__objclass__").__name__
                msg = generate_deprecation_warning_msg(what,
                                                       old_name,
                                                       new_name,
                                                       class_name=class_name)
            else:
                what = "function"
                msg = generate_deprecation_warning_msg(
                    what, old_name, new_name)
            warnings.warn(msg, FutureWarning)
            return func(*args, **kwargs)

        old_name = getattr(func, "__name__")
        return wrapper

    return decorate


def generate_deprecation_warning_msg(what,
                                     old_name,
                                     new_name,
                                     class_name=None):
    """Helper function to create user-friendly deprecation messages.

    Parameters
    ----------
    what : str
        The type of object that is being deprecated. Expected values are
        :str:`class`, :str:`function` and :str:`method`.
    old_name : str
        The name of the object being deprecated.
    new_name : str
        The name of the object replacing the deprecated object.
    class_name : str, optional
        The class name if the object being deprecated is a :obj:`class`.
        Default value is :obj:`None`.

    Returns
    -------
    str
        The nicely formatted informative :obj:`str` to be outputted as the
        warning message.

    Raises
    ------
    ValueError
        If a :arg:`class_name` is supplied but :arg:`what` does not equal
        :str:`"class"`.
    """
    if what == "class" and class_name is not None:
        msg = "Incorrect use of function arguments."
        raise ValueError(msg)
    if class_name is not None:
        class_name_msg = f"in class '{str(class_name)}' "
    else:
        class_name_msg = ""
    msg = (f"The {what} named '{old_name}' {class_name_msg}will soon be "
           f"deprecated in CyIpopt. Please replace all uses and use "
           f"'{new_name}' going forward.")
    return msg
