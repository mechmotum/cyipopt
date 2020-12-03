import sys
import warnings

sys.modules[__name__] = __import__("cyipopt")


def issue_deprecation_warning():
    """Warn user of deprication of 'ipopt' as module name."""
    msg = ("The module has been renamed to 'cyipopt' from 'ipopt'. Please "
           "import using 'import cyipopt' and remove all uses of "
           "'import ipopt' in your code as this will be deprecated in a "
           "future release.")
    warnings.warn(msg, FutureWarning)


issue_deprecation_warning()
