import warnings

from ipopt_cython_wrapper import Problem


__all__ = ["problem"]


class problem:

	def __new__(self, *args, **kwargs):
		msg = generate_deprication_warning_msg("class", "problem", "Problem")
		warnings.warn(msg, FutureWarning)
		return Problem(*args, **kwargs)


def generate_deprication_warning_msg(what, old_name, new_name):
	msg = (f"The {what} named '{old_name}' will soon be depricated in CyIpopt. "
		f"Please replace all uses and use '{new_name}' going forward.")
	return msg