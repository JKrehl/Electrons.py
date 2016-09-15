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

from ....Mathematics import FourierTransforms as FT
from ...AtomPotentials import WeickenmeierKohl

import reikna.core
import reikna.fft
from ...Operators import PlaneOperator
from ...Operators import AbstractArray

compiled_atom_phaseshift = {}


class AtomPhaseshift(reikna.core.Computation):
	def __init__(self, tmp, ps, ys, xs, B, ky, kx, kk):
		super().__init__([reikna.core.Parameter('tmp', reikna.core.Annotation(tmp, 'o')),
		                  reikna.core.Parameter('ps', reikna.core.Annotation(ps, 'i')),
		                  reikna.core.Parameter('ys', reikna.core.Annotation(ys, 's')),
		                  reikna.core.Parameter('xs', reikna.core.Annotation(xs, 's')),
		                  reikna.core.Parameter('B', reikna.core.Annotation(B, 's')),
		                  reikna.core.Parameter('ky', reikna.core.Annotation(ky, 'i')),
		                  reikna.core.Parameter('kx', reikna.core.Annotation(kx, 'i')),
		                  reikna.core.Parameter('kk', reikna.core.Annotation(kk, 'i'))])

	def _build_plan(self, plan_factory, device_params, tmp, ps, ys, xs, B, ky, kx, kk):
		plan = plan_factory()

		template = reikna.helpers.template_from(
			"""
			<%def name='function(kernel_declaration, tmp, ps, ys, xs, B, ky, kx, kk)'>
				${kernel_declaration}
				{
				VIRTUAL_SKIP_THREADS;
				const VSIZE_T idy = virtual_global_id(0);
				const VSIZE_T idx = virtual_global_id(1);
				${tmp.store_idx}(idy, idx, ${mulp}(${ps.load_idx}(idy, idx), ${expx}(${addx}(${muli}(COMPLEX_CTR(float2)(0,-1), ${adds}(${muly}(${ys}, ${ky.load_idx}(idy)), ${mulx}(${xs}, ${kx.load_idx}(idx)))), ${mulB}(-${B}/8, ${kk.load_idx}(idy,idx))))));
				}
			</%def>
			""")

		Bkk_dtype = reikna.cluda.dtypes.result_type(B.dtype, kk.dtype)
		kxs_dtype = reikna.cluda.dtypes.result_type(xs.dtype, kx.dtype)
		kys_dtype = reikna.cluda.dtypes.result_type(ys.dtype, ky.dtype)
		ks_dtype = reikna.cluda.dtypes.result_type(kys_dtype, kxs_dtype)
		cks_dtype = reikna.cluda.dtypes.result_type(numpy.complex64, ks_dtype)
		Bcks_dtype = reikna.cluda.dtypes.result_type(cks_dtype, Bkk_dtype)

		plan.kernel_call(template.get_def('function'),
		                 [tmp, ps, ys, xs, B, ky, kx, kk],
		                 global_size=tmp.shape,
		                 render_kwds=dict(mulB=reikna.cluda.functions.mul(B.dtype, kk.dtype),
		                                  mulx=reikna.cluda.functions.mul(xs.dtype, kx.dtype),
		                                  muly=reikna.cluda.functions.mul(ys.dtype, ky.dtype),
		                                  adds=reikna.cluda.functions.add(kys_dtype, kxs_dtype),
		                                  muli=reikna.cluda.functions.mul(numpy.complex64, ks_dtype),
		                                  addx=reikna.cluda.functions.add(cks_dtype, Bkk_dtype),
		                                  expx=reikna.cluda.functions.exp(Bcks_dtype),
		                                  mulp=reikna.cluda.functions.mul(ps.dtype, Bcks_dtype)
		                                  ))
		return plan


compiled_expi = {}


class Expi(reikna.core.Computation):
	def __init__(self, arg):
		super().__init__([reikna.core.Parameter('res', reikna.core.Annotation(arg, 'io')),
		                  reikna.core.Parameter('arg', reikna.core.Annotation(arg, 'io'))])

	def _build_plan(self, plan_factory, device_params, res, arg):
		plan = plan_factory()

		template = reikna.helpers.template_from(
			"""
			<%def name='function(kernel_declaration, res, arg)'>
				${kernel_declaration}
				{
				VIRTUAL_SKIP_THREADS;
				const VSIZE_T idy = virtual_global_id(0);
				const VSIZE_T idx = virtual_global_id(1);
				${res.store_idx}(idy, idx, ${exp}(${mul}(COMPLEX_CTR(${arg.ctype})(0,1), ${arg.load_idx}(idy, idx))));
				}
			</%def>
			""")

		plan.kernel_call(template.get_def('function'),
		                 [res, arg],
		                 global_size=arg.shape,
		                 render_kwds=dict(mul=reikna.cluda.functions.mul(arg.dtype, arg.dtype),
		                                  exp=reikna.cluda.functions.exp(arg.dtype)))
		return plan

compiled_ipmul = {}

class InplaceMul(reikna.core.Computation):
	def __init__(self, A, B):
		super().__init__([reikna.core.Parameter('A', reikna.core.Annotation(A, 'io')),
		                  reikna.core.Parameter('B', reikna.core.Annotation(B, 'i'))])

	def _build_plan(self, plan_factory, device_params, A, B):
		plan = plan_factory()

		template = reikna.helpers.template_from(
			"""
			<%def name='function(kernel_declaration, A, B)'>
				${kernel_declaration}
				{
				VIRTUAL_SKIP_THREADS;
				const VSIZE_T idy = virtual_global_id(0);
				const VSIZE_T idx = virtual_global_id(1);
				${A.store_idx}(idy, idx, ${mul}(${A.load_idx}(idy, idx), ${B.load_idx}(idy, idx)));
				}
			</%def>
			""")

		plan.kernel_call(template.get_def('function'),
		                 [A, B],
		                 global_size=A.shape,
		                 render_kwds=dict(mul=reikna.cluda.functions.mul(A.dtype, B.dtype)))
		return plan



compiled_fft_gpu = {}

class FlatAtomDW_GPU(PlaneOperator):
	def __init__(self, atoms,
	             z = None,
	             ky = None, kx=None, kk=None,
	             phaseshifts_tf = None,
	             y = None, x = None,
	             atom_potential = WeickenmeierKohl, energy = None,
	             dtype = numpy.complex,
	             lazy = True, forgetful = True, factory = False,
	             mode ="cuda", thread = None):

		super().__init__(None)

		self.mode = mode
		self.thread = thread
		if self.thread is None:
			self.thread = AbstractArray._modes[self.mode].get_thread()

		self.atoms = atoms

		self.dtype = dtype
		self.lazy, self.forgetful = lazy, forgetful
		self.factory = factory

		if ky is None: ky = FT.reciprocal_coords(y)
		if kx is None: kx = FT.reciprocal_coords(x)
		if kk is None: kk = numpy.add.outer(ky**2, kx**2)

		self.ky = AbstractArray(ky, self.mode, self.thread)
		self.kx = AbstractArray(kx, self.mode, self.thread)
		self.kk = AbstractArray(kk, self.mode, self.thread)

		if phaseshifts_tf is None:
			phaseshifts_tf = {}

		self.phaseshifts_tf = phaseshifts_tf
		if not all(Z in self.phaseshifts_tf for Z in numpy.unique(self.atoms['Z'])):
			if self.lazy:
				self.atom_potential = atom_potential
				self.energy = energy
				self.y, self.x = y, x
			else:
				for Z in numpy.unique(self.atoms['Z']):
					if Z not in self.phaseshifts_tf:
						self.phaseshifts_tf[Z] = atom_potential.cis_phaseshift_f(Z, energy, y, x)

		for i,j in self.phaseshifts_tf.items():
			self.phaseshifts_tf[i] = AbstractArray(j, self.mode, self.thread)

		self.transmission_function = None
		if self.lazy == 0:
			self.transmission_function = self.generate_transmission_function()

		self.z = z
		if self.z is None:
			self.z = numpy.mean(self.atoms['zyx'][:,0])

	def derive(self, atoms, **kwargs):
		args = dict(ky = self.ky, kx = self.kx, kk = self.kk,
		            phaseshifts_tf = self.phaseshifts_tf,
		            dtype = self.dtype,
		            lazy = self. lazy, forgetful = self.forgetful, factory = False,
		            mode = self.mode, thread = self.thread)

		if hasattr(self, "atom_potential"):
			args.update(atom_potential=self.atom_potential, energy=self.energy, y=self.y, x=self.x)

		args.update(kwargs)

		return self.__class__(atoms, **args)

	def generate_transmission_function(self):
		if self.factory: return None

		if self.forgetful:
			phaseshifts_tf = self.phaseshifts_tf.copy()
		else:
			phaseshifts_tf = self.phaseshifts_tf

		for Z in numpy.unique(self.atoms['Z']):
			if Z not in phaseshifts_tf:
				phaseshifts_tf[Z] = AbstractArray(self.atom_potential.cis_phaseshift_f(Z, self.energy, self.y, self.x), self.mode, self.thread)

		transmission_function = AbstractArray(numpy.ones(self.kk.shape, dtype=self.dtype), self.mode, self.thread)

		sign = (self.thread,
				reikna.core.Type.from_value(transmission_function).__repr__(),
				reikna.core.Type.from_value(list(phaseshifts_tf.values())[0]).__repr__(),
				self.atoms[0]['zyx'].dtype.__repr__(),
				self.atoms[0]['B'].dtype.__repr__(),
				reikna.core.Type.from_value(self.ky).__repr__(),
				reikna.core.Type.from_value(self.kx).__repr__(),
				reikna.core.Type.from_value(self.kk).__repr__())

		if sign in compiled_atom_phaseshift:
			atom_phaseshift = compiled_atom_phaseshift[sign]
		else:
			atom_phaseshift =  AtomPhaseshift(transmission_function, list(phaseshifts_tf.values())[0], self.atoms[0]['zyx'][1], self.atoms[0]['zyx'][2], self.atoms[0]['B'], self.ky, self.kx, self.kk).compile(self.thread)
			compiled_atom_phaseshift[sign] = atom_phaseshift

		xsign = (self.thread, reikna.core.Type.from_value(transmission_function).__repr__())

		if xsign in compiled_expi:
			expi = compiled_expi[xsign]
		else:
			expi = Expi(transmission_function).compile(self.thread)
			compiled_expi[xsign] = expi

		if xsign in compiled_fft_gpu:
			fft_gpu = compiled_fft_gpu[xsign]
		else:
			fft_gpu = reikna.fft.FFT(transmission_function).compile(self.thread)
			compiled_fft_gpu[xsign] = fft_gpu

		ipmsign = (self.thread, reikna.core.Type.from_value(transmission_function).__repr__(), reikna.core.Type.from_value(transmission_function).__repr__())
		if ipmsign in compiled_ipmul:
			ipmul = compiled_ipmul[ipmsign]
		else:
			ipmul = InplaceMul(transmission_function, transmission_function).compile(self.thread)
			compiled_ipmul[ipmsign] = ipmul

		tmp = self.thread.temp_array(transmission_function.shape, transmission_function.dtype, transmission_function.strides)
		tmp2 = self.thread.temp_array(transmission_function.shape, transmission_function.dtype, transmission_function.strides)

		for a in self.atoms:
			atom_phaseshift(tmp, phaseshifts_tf[a['Z']], a['zyx'][1], a['zyx'][2], a['B']/(4*numpy.pi**2), self.ky, self.kx, self.kk)
			fft_gpu(tmp, tmp2, 1)
			ipmul(transmission_function, tmp2)

		self.transmission_function = transmission_function

		return transmission_function

	def apply(self, wave):
		if self.transmission_function is None:
			self.transmission_function = self.generate_transmission_function()

		wave = wave._as(self.mode, self.thread)

		wave *= self.transmission_function

		if self.forgetful:
			self.transmission_function = None

		return wave
