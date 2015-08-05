from __future__ import absolute_import, division

import numpy
import numexpr
from scipy import ndimage
from ....Mathematics import FourierTransforms as FT

import reikna.core
import reikna.fft

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
	def __init__(self, thread, zi, zf, k, kk=None, ky=None, kx=None, y=None, x=None):
		self.__dict__.update(dict(thread=thread, zi=zi,zf=zf, k=k, kk=kk))

		if self.kk is None:
			if ky is None:
				ky = FT.reciprocal_coords(y)
			if kx is None:
				kx = FT.reciprocal_coords(x)
			
			self.kk = numpy.add.outer(ky**2, kx**2)

		if not hasattr(self.kk, 'thread'):
			self.kk = self.thread.to_device(self.kk)
		elif self.kk.thread!=self.thread:
			self.kk = self.thread.to_device(self.kk.get())
			
	@classmethod
	def inherit(cls, parent, zi, zf, **kwargs):
		thread = parent.thread
		k = parent.k
		args = {}

		if hasattr(parent, 'g_kk'):
			args.update(dict(kk = parent.g_kk))
		if hasattr(parent, 'g_ky') and hasattr(parent, 'g_kx'):
			args.update(dict(ky = parent.g_ky, kx = parent.g_kx))
		#elif hasattr(parent, 'g_y') and hasattr(parent, 'g_x'):
		#	args.update(dict(y = parent.g_y, x = parent.g_x))

		args.update(kwargs)
			
		return cls(thread, zi, zf, k, **args)

	def apply(self, wave, tmp=None):
		if not hasattr(wave, 'thread'):
			wave = self.thread.to_device(wave)
		elif wave.thread != self.thread:
			wave = self.thread.to_device(wave.get())

		if tmp is None or not hasattr(tmp, 'thread') or tmp.thread != self.thread:
			tmp = self.thread.array(wave.shape, wave.dtype, wave.strides)
		
		factor = -1j*numpy.pi*(self.zf-self.zi)/self.k

		psignature = (self.thread, reikna.core.Type.from_value(wave).__repr__(), reikna.core.Type.from_value(self.kk).__repr__(), reikna.core.Type.from_value(factor).__repr__())

		if psignature in compiled_phase_parabola:
			phase_parab = compiled_phase_parabola[psignature]
		else:
			phase_parab = PhaseParabola(wave, self.kk, factor).compile(self.thread)
			compiled_phase_parabola.update({psignature:phase_parab})

		fsignature = (self.thread, reikna.core.Type.from_value(wave).__repr__())

		if fsignature in compiled_fft_gpu:
			fft_gpu = compiled_fft_gpu[fsignature]
		else:
			fft_gpu = reikna.fft.FFT(wave).compile(self.thread)
			compiled_fft_gpu.update({fsignature:fft_gpu})

		fft_gpu(tmp, wave, 1)
		phase_parab(tmp, self.kk, factor)
		fft_gpu(wave, tmp, 0)

		return wave
			
	def split(self, z):
		return FresnelFourier(self.zi, z, self.k, self.kk), FresnelFourier(z, self.zf, self.k, self.kk)
