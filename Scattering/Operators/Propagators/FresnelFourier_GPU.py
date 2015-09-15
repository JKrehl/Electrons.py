import numpy
import numexpr
from scipy import ndimage
from ....Mathematics import FourierTransforms as FT
from ....Utilities import Physics

import reikna.core
import reikna.fft
from reikna.cluda.api import Thread

from ..Base import IntervalOperator

compiled_phase_parabola = {}
compiled_fft_gpu = {}

class PhaseParabola(reikna.core.Computation):
	def __init__(self, wave, kk, factor):
		super().__init__([
			 reikna.core.Parameter('wave', reikna.core.Annotation(wave, 'io')), 
			 reikna.core.Parameter('kk', reikna.core.Annotation(kk, 'i')),
			 reikna.core.Parameter('factor', reikna.core.Annotation(factor)),
			])
		
	def _build_plan(self, plan_factory, device_params, wave, kk, factor):
		plan = plan_factory()
		
		template = reikna.helpers.template_from(
			"""
			<%def name='test(kernel_declaration, k_wave, k_kk, k_factor)'>
				${kernel_declaration}
				{
				VIRTUAL_SKIP_THREADS;
				const VSIZE_T idy = virtual_global_id(0);
				const VSIZE_T idx = virtual_global_id(1);
				${k_wave.store_idx}(idy, idx, ${mul}(${k_wave.load_idx}(idy, idx), ${exp}(${exmul}(${k_factor}, ${k_kk.load_idx}(idy, idx)))));
				}
			</%def>
			""")
		
		extype = numpy.result_type(factor.dtype, kk.dtype)
		
		plan.kernel_call(template.get_def('test'), [wave, kk, factor], global_size=wave.shape, 
						 render_kwds=dict(exp=reikna.cluda.functions.exp(extype), mul=reikna.cluda.functions.mul(wave.dtype, extype), exmul=reikna.cluda.functions.mul(factor.dtype, kk.dtype, out_dtype=extype)))
		
		return plan

class FresnelFourier_GPU(IntervalOperator):
	def __init__(self, zi, zf, thread=None, k=None, kk=None, ky=None, kx=None, y=None, x=None, compiled_phase_parabola=None, compiled_fft_gpu=None):
		self.__dict__.update(dict(thread=thread, zi=zi,zf=zf, k=k, kk=kk, compiled_phase_parabola=compiled_phase_parabola, compiled_fft_gpu=compiled_fft_gpu))

		if self.compiled_phase_parabola is None: self.compiled_phase_parabola={}
		if self.compiled_fft_gpu is None: self.compiled_fft_gpu={}

		if self.kk is None:
			if ky is None:
				ky = FT.reciprocal_coords(y)
			if kx is None:
				kx = FT.reciprocal_coords(x)
				
			self.kk = self.thread.to_device(numpy.add.outer(ky**2, kx**2))

		if not hasattr(self.kk, 'thread'):
			self.kk = self.thread.to_device(self.kk)
		elif self.kk.thread!=self.thread:
			self.kk = self.thread.to_device(self.kk.get())
			
	@staticmethod
	def inherit(parent, zi, zf, **kwargs):
		args = {}

		args.update({k:v for k,v in parent.propagator_args.items() if v is not None})
		args.update({k:v for k,v in kwargs.items() if v is not None})

		if 'thread' in parent.transfer_function_args and isinstance(parent.transfer_function_args['thread'], Thread):
			args['thread'] = parent.transfer_function_args['thread']
		elif not 'thread' in args or args['thread'] is None:
			args['thread'] = reikna.cluda.any_api().Thread.create()
		elif isinstance(args['thread'], Thread):
			pass
		elif args['thread'] == 'cuda':
			args['thread'] = reikna.cluda.cuda_api().Thread.create()
		elif args['thread'] == 'opencl':
			args['thread'] = reikna.cluda.ocl_api().Thread.create()
		else:
			raise ValueError
		thread = args['thread']

		if 'compiled_phase_parabola' not in args or args['compiled_phase_parabola'] is None:
			args['compiled_phase_parabola'] = {}

		if 'compiled_fft_gpu' not in args or args['compiled_fft_gpu'] is None:
			args['compiled_fft_gpu'] = {}
		
		args.update({s:thread.to_device(parent.__dict__[s]) for s in ['kk'] if s not in args or args[s] is None})

		if not 'kk' in args or args['kk'] is None or not hasattr(args['kk'], 'thread') or args['kk'].thread != args['thread']:
			args['kk'] = args['thread'].todevice(parent.kk)

		if not 'k' in args or args['k'] is None:
			args['k'] = Physics.wavenumber(parent.energy)
			
		parent.propagator_args.update(args)
			
		return FresnelFourier_GPU(zi, zf, **args)

	def apply(self, wave, tmp=None):
		if not hasattr(wave, 'thread'):
			wave = self.thread.to_device(wave)
		elif wave.thread != self.thread:
			wave = self.thread.to_device(wave.get())

		if tmp is None or not hasattr(tmp, 'thread') or tmp.thread != self.thread:
			tmp = self.thread.array(wave.shape, wave.dtype, wave.strides)
		
		factor = -1j*numpy.pi*(self.zf-self.zi)/self.k

		psignature = (self.thread, reikna.core.Type.from_value(wave).__repr__(), reikna.core.Type.from_value(self.kk).__repr__(), reikna.core.Type.from_value(factor).__repr__())

		if psignature in self.compiled_phase_parabola:
			phase_parab = self.compiled_phase_parabola[psignature]
		else:
			phase_parab = PhaseParabola(wave, self.kk, factor).compile(self.thread)
			self.compiled_phase_parabola.update({psignature:phase_parab})

		fsignature = (self.thread, reikna.core.Type.from_value(wave).__repr__())

		if fsignature in self.compiled_fft_gpu:
			fft_gpu = self.compiled_fft_gpu[fsignature]
		else:
			fft_gpu = reikna.fft.FFT(wave).compile(self.thread)
			self.compiled_fft_gpu.update({fsignature:fft_gpu})

		fft_gpu(tmp, wave, 1)
		phase_parab(tmp, self.kk, factor)
		fft_gpu(wave, tmp, 0)

		return wave
			
	def split(self, z):
		return FresnelFourier(self.zi, z, self.k, self.kk), FresnelFourier(z, self.zf, self.k, self.kk)
