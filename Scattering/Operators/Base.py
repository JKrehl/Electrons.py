from __future__ import absolute_import

class PlaneOperator:
	def __init__(self):
		pass
	def apply(self, wave):
		raise NotImplemented

class IntervalOperator:
	def __init__(self):
		pass

	def apply(self, wave):
		raise NotImplemented
	
	def split(self, z):
		raise NotImplemented
