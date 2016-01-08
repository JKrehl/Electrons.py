import pyximport
pyximport.install()

import math

def humanize_filesize(size):
	if size==0: return (0,"B")
	suffixes=['B','KiB','MiB','GiB','TiB','PiB','EiB','ZiB','YiB']
	
	po1024 = int(math.floor(math.log(size, 1024)))
	
	return (size/(1024**po1024), suffixes[po1024])

def lazy_property(fn):
	decorated_name = '__'+fn.__name__
	@property
	def __lazy_property(self):
		if not hasattr(self, decorated_name):
			setattr(self, decorated_name, fn(self))
		return getattr(self, decorated_name)
	return __lazy_property

def apply_if(obj, fun, cond, *args, **kwargs):
	if cond:
		return fun(obj, *args, **kwargs)
	else:
		return obj

from .Magic_cy import *

class MemberAccessDictionary(dict):
	def __getattr__(self, key):
		return self.__getitem__(key)
	def __setattr__(self, key, value):
		if key in self.__dict__:
			self.__dict__[key] = value
		else:
			self.__setitem__(key, value)
