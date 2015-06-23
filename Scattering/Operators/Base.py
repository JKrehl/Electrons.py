from __future__ import absolute_import

class PlaneOperator:
	z = None
	
	def __init__(self):
		pass
	def apply(self, wave):
		raise NotImplemented

class IntervalOperator:
	zi = None
	zf = None
	
	def __init__(self):
		pass

	def apply(self, wave):
		raise NotImplemented
	
	def split(self, z):
		raise NotImplemented
