import pyximport
pyximport.install()

def lazy_property(fn):
	decorated_name = '__'+fn.__name__
	@property
	def __lazy_property(self):
		if not hasattr(self, decorated_name):
			setattr(self, decorated_name, fn(self))
		return getattr(self, decorated_name)
	return __lazy_property

from .Magic_cy import *
