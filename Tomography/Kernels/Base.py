from __future__ import print_function

import scipy.sparse

class Kernel:
	ndims = None
	fshape = None
	shape = None

	dat = None
	idx = None
	status = -1

	@property
	def as_coo(self):
		return scipy.sparse.coo_matrix((self.dat, (self.col, self.row)), shape=self.fshape)
	
	def __init__(self):
		pass

	def calc(self):
		raise NotImplemented

