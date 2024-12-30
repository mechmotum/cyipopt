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
