import numpy
import scipy.sparse

from ..Kernels import Kernel

from reikna.cluda.api import Thread

class FlatProjector(scipy.sparse.linalg.LinearOperator):
	def __init__(self, kernel, shape = None, thread=None):
		
		if isinstance(kernel, tuple) or isinstance(kernel, list):
			if shape is not None:
				self.shape = shape
			else:
				raise AttributeError("The shape needs to be supplied to the Projector constructor")
			
			self.dat = kernel[0]
			self.nnz = self.dat.size
			self.dtype = self.dat.dtype
			if len(kernel) == 2:
				assert kernel[1].ndims==2 and kernel[1].shape[0]==2
				self.idx = tuple(kernel[1])
			elif len(kernel)==3:
				self.idx = kernel[1:]
			else:
				raise AttributeError
		elif isinstance(kernel, Kernel):
			assert kernel.ndims==2

			if kernel.status == -1:
				kernel.calc()
				
			self.shape = kernel.fshape
			self.dat = kernel.dat
			self.idx = kernel.idx
			self.nnz = self.dat.size
			self.dtype = self.dat.dtype
		else:
			raise NotImplementedError

		if isinstance(thread, Thread):
			self.thread = thread
		else:
			if thread=='cuda':
				self.thread = reikna.cluda.cuda_api().Thread.create()
			else:
				self.thread = reikna.ocl.cuda_api().Thread.create()

		
