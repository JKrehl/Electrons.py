from __future__ import division, print_function

import numpy
import numexpr

class RayProjection2D:
	def __init__(self, y, x, t, d):
		self.__dict__.update(dict(y=y, x=x, t=t, d=d))
