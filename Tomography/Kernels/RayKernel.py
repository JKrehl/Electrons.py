import numpy
import numexpr


from ...Utilities import Progress
from ...Utilities.Magic import apply_if

from .Kernel import Kernel

class RayKernel(Kernel):
	_arrays = dict(y=0, x=0, t=0, d=0)
