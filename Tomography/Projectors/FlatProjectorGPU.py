import numpy
import scipy.sparse

from ..Kernels import Kernel

from reikna.cluda.api import Thread
import reikna.core
import reikna.cluda

class GPU_matvec(reikna.core.Computation):
	def __init__(self, vec, res, dat, bounds, indices):
		super().__init__([reikna.core.Parameter('vec', reikna.core.Annotation(vec, 'i')),
						  reikna.core.Parameter('res', reikna.core.Annotation(res, 'o')),
						  reikna.core.Parameter('dat', reikna.core.Annotation(dat, 'i')),
						  reikna.core.Parameter('bounds', reikna.core.Annotation(bounds, 'i')),
						  reikna.core.Parameter('indices', reikna.core.Annotation(indices, 'i'))])
		
	def _build_plan(self, plan_factory, device_params, vec, res, dat, bounds, indices):
		plan = plan_factory()
		
		template = reikna.helpers.template_from(
			"""
			<%def name='function(kernel_declaration, vec, res, dat, bounds, indices)'>
				${kernel_declaration}
				{
				VIRTUAL_SKIP_THREADS;
				const VSIZE_T idx = virtual_global_id(0);
				${res.ctype} tmp = 0;
			
				for(VSIZE_T jdx = ${bounds.load_idx}(idx); jdx < ${bounds.load_idx}(idx+1); jdx++)
				{
					tmp += ${dat.load_idx}(jdx)*${vec.load_idx}(${indices.load_idx}(jdx));
				}
			
				${res.store_idx}(idx, tmp);
				}
			</%def>
			""")
		plan.kernel_call(template.get_def('function'),
						 [vec, res, dat, bounds, indices],
						 global_size=bounds.size-1,
						 render_kwds=dict())
		return plan
		

class FlatProjectorGPU(scipy.sparse.linalg.LinearOperator):
	def __init__(self, kernel, shape = None, thread=None):
		
		if isinstance(kernel, tuple) or isinstance(kernel, list):
			if shape is not None:
				self.shape = shape
			else:
				raise AttributeError("The shape needs to be supplied to the Projector constructor")
			
			dat = kernel[0]
			self.nnz = dat.size
			self.dtype = dat.dtype
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
			dat = kernel.dat
			idx = kernel.idx
			self.nnz = dat.size
			self.dtype = dat.dtype
		else:
			raise NotImplementedError
		
		if isinstance(thread, Thread):
			self.thread = thread
		else:
			if thread=='cuda':
				self.thread = reikna.cluda.cuda_api().Thread.create()
			else:
				self.thread = reikna.cluda.ocl_api().Thread.create()

		csort = numpy.argsort(idx[1])
		self.c_idxr = self.thread.to_device(idx[0][csort])
		self.c_idxc = self.thread.to_device(numpy.hstack((0, numpy.cumsum(numpy.bincount(idx[1], minlength=self.shape[0])))).astype(idx[1].dtype))
		self.c_dat = self.thread.to_device(dat[csort])

		rsort = numpy.argsort(idx[0])
		self.r_idxc = self.thread.to_device(idx[1][rsort])
		self.r_idxr = self.thread.to_device(numpy.hstack((0, numpy.cumsum(numpy.bincount(idx[0], minlength=self.shape[1])))).astype(idx[0].dtype))
		self.r_dat = self.thread.to_device(dat[rsort])

		self.c_arr = self.thread.array(self.shape[1], self.dtype)
		self.r_arr = self.thread.array(self.shape[0], self.dtype)
		
		self.gpu_matvec = GPU_matvec(self.c_arr, self.r_arr, self.c_dat, self.c_idxc, self.c_idxr).compile(self.thread)
		self.gpu_rmatvec = GPU_matvec(self.r_arr, self.c_arr, self.r_dat, self.r_idxr, self.r_idxc).compile(self.thread)
		
	def matvec(self, v):
		self.thread.to_device(v.reshape(self.shape[1]), self.c_arr)

		self.gpu_matvec(self.c_arr, self.r_arr, self.c_dat, self.c_idxc, self.c_idxr)

		return self.r_arr.get()
	
	def rmatvec(self, v):
		self.thread.to_device(v.reshape(self.shape[0]), self.r_arr)

		self.gpu_rmatvec(self.r_arr, self.c_arr, self.r_dat, self.r_idxr, self.r_idxc)

		return self.c_arr.get()
