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

# class OperatorFactory:
# 	def __init__(self, cls, *args, **kwargs):
# 		self.cls = cls

# 		self.cls.factory_init(self, *args, **kwargs)
		
# 	def create(self, *args, **kwargs):
# 		self.cls.factory_new(self, *args, **kwargs)
		
