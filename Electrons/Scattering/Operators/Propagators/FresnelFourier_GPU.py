#!/usr/bin/env python
"""
Copyright (c) 2015 Jonas Krehl <Jonas.Krehl@triebenberg.de>

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""

import numpy
import numexpr
from scipy import ndimage
from ....Mathematics import FourierTransforms as FT
from ....Utilities import Physics

import reikna.core
import reikna.fft
from ...Operators import IntervalOperator
from ...Operators import AbstractArray

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
	def __init__(self, zi, zf,
	             kk = None, ky = None, kx = None,
	             k = None, energy = None,
	             y = None, x = None,
	             factory = False,
				 mode = "cuda", thread = None):
		super().__init__(zi, zf)

		self.mode = mode

		if kk is None:
			if ky is None: ky = FT.reciprocal_coords(y)
			if kx is None: kx = FT.reciprocal_coords(x)
			kk = numpy.add.outer(ky**2, kx**2)

		if isinstance(kk, AbstractArray):
			self.kk = kk._as(self.mode, thread)
		else:
			self.kk = AbstractArray(kk, mode=self.mode, thread=thread)

		self.thread = self.kk.thread

		if k is None: self.k = Physics.wavenumber(energy)
		else: self.k = k

	def derive(self, zi, zf, **kwargs):
		args = dict(k=self.k, kk=self.kk, mode=self.mode, thread=self.thread)
		args.update(kwargs)

		return self.__class__(zi, zf, **args)

	def apply(self, wave):
		if isinstance(wave, AbstractArray):
			wave = wave._as(self.mode, self.thread)
		else:
			wave = AbstractArray(wave, self.mode, self.thread)

		factor = numpy.obj2sctype(wave.dtype)(-1j*(self.zf-self.zi)/(2*self.k))

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

		tmp = self.thread.temp_array(wave.shape, wave.dtype)

		fft_gpu(tmp, wave, 0)
		phase_parab(tmp, self.kk, factor)
		fft_gpu(wave, tmp, 1)

		return wave


	def split(self, z):
		return self.derive(self.zi, z), self.derive(z, self.zf)
