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
import scipy.interpolate

import reikna.core
import reikna.cluda
import reikna.helpers
from reikna.cluda.api import Thread

from ....Mathematics import FourierTransforms as FT
from ...Potentials.AtomPotentials import WeickenmeierKohl

from ..Base import PlaneOperator

compiled_atom_phaseshift = {}
	
class AtomPhaseshift(reikna.core.Computation):
	def __init__(self, tf, ps, ys, xs, B, ky, kx, kk):
		super().__init__([reikna.core.Parameter('tf', reikna.core.Annotation(tf, 'io')),
						  reikna.core.Parameter('ps', reikna.core.Annotation(ps, 'i')),
						  reikna.core.Parameter('ys', reikna.core.Annotation(ys, 's')),
						  reikna.core.Parameter('xs', reikna.core.Annotation(xs, 's')),
						  reikna.core.Parameter('B', reikna.core.Annotation(B, 's')), 
						  reikna.core.Parameter('ky', reikna.core.Annotation(ky, 'i')),
						  reikna.core.Parameter('kx', reikna.core.Annotation(kx, 'i')),
						  reikna.core.Parameter('kk', reikna.core.Annotation(kk, 'i'))])
		
	def _build_plan(self, plan_factory, device_params, tf, ps, ys, xs, B, ky, kx, kk):
		plan = plan_factory()

		template = reikna.helpers.template_from(
			"""
			<%def name='function(kernel_declaration, tf, ps, ys, xs, B, ky, kx, kk)'>
				${kernel_declaration}
				{
				VIRTUAL_SKIP_THREADS;
				const VSIZE_T idy = virtual_global_id(0);
				const VSIZE_T idx = virtual_global_id(1);
				${tf.store_idx}(idy, idx, ${tf.load_idx}(idy, idx)+${mulp}(${ps.load_idx}(idy, idx), ${expx}(${addx}(${muli}(COMPLEX_CTR(float2)(0,-1), ${adds}(${muly}(${ys}, ${ky.load_idx}(idy)), ${mulx}(${xs}, ${kx.load_idx}(idx)))), ${mulB}(-${B}/8, ${kk.load_idx}(idy,idx))))));
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
						 [tf, ps, ys, xs, B, ky, kx, kk],
						 global_size=tf.shape,
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
						 render_kwds=dict(mul=reikna.cluda.functions.mul(arg.dtype, arg.dtype), exp=reikna.cluda.functions.exp(arg.dtype)))
		return plan

compiled_fft_gpu = {}
	
class FlatAtomDW_GPU(PlaneOperator):
	def __init__(self, atoms, thread=None, phaseshifts_f=None,
				 ky=None, kx=None, kk=None,
				 atom_potential_generator=WeickenmeierKohl, energy=None, y=None, x=None,
				 dtype=numpy.complex,
				 lazy=True, forgetful=True):
		self.__dict__.update(dict(thread=thread, atoms=atoms,
								  phaseshifts_f=phaseshifts_f,
								  ky=ky, kx=kx, kk=kk,
								  atom_potential_generator=atom_potential_generator, energy=energy, y=y, x=x,
								  dtype=dtype,
								  lazy=lazy, forgetful=forgetful))
		
		self.transmission_function = None
		if not self.lazy:
			self.generate_tf()
		
		self.z = numpy.mean(self.atoms['zyx'][:,0])
		
	@classmethod
	def inherit(cls, parent, atoms, **kwargs):
		args = {}
	
		args.update({k:v for k,v in parent.transmission_function_args.items() if v is not None})
		args.update({k:v for k,v in kwargs.items() if v is not None})

		if 'thread' in parent.propagator_args and isinstance(parent.propagator_args['thread'], Thread):
			args['thread'] = parent.propagator_args['thread']
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
		
		args.update({s:parent.__dict__[s] for s in ['y', 'x'] if s not in args or args[s] is None})
		args.update({s:thread.to_device(parent.__dict__[s]) for s in ['ky', 'kx', 'kk'] if s not in args or args[s] is None})
		
		if 'phaseshifts_f' not in args or args['phaseshifts_f'] is None or not set(numpy.unique(atoms['Z'])).issubset(set(args['phaseshifts_f'].keys())):
			if hasattr(parent, 'phaseshifts_f') and parent.phaseshifts_f is not None:
				args['phaseshifts_f'] = parent.phaseshifts_f
			else:
				if 'energy' not in args or args['energy'] is None:
					args['energy'] = parent.energy
				if 'atom_potential_generator' not in args or args['atom_potential_generator'] is None:
					args['atom_potential_generator'] = parent.atom_potential_generator
				if 'phaseshifts_f' not in args or args['phaseshifts_f'] is None:
					args['phaseshifts_f'] = {}
				args['phaseshifts_f'].update({i: thread.to_device(args['atom_potential_generator'].cis_phaseshift_f(i, args['energy'], args['y'], args['x'])) for i in set(numpy.unique(atoms['Z'])).difference(set(args['phaseshifts_f'].keys()))})

		parent.transmission_function_args.update(args)
		return cls(atoms, **args)
			
	def generate_tf(self):
		
		if self.phaseshifts_f is None:
			phaseshifts_f = {i: self.thread.to_device(self.atom_potential_generator.cis_phaseshift_f(i, self.energy, self.y, self.x)) for i in numpy.unique(self.atoms['Z'])}
		else:
			phaseshifts_f = self.phaseshifts_f
			
		if self.ky is None:
			ky = self.thread.to_device(FT.reciprocal_coords(self.y))
		else:
			ky = self.ky
			
		if self.kx is None:
			kx = self.thread.to_device(FT.reciprocal_coords(self.x))
		else:
			kx = self.kx

		if self.kk is None:
			kk = self.thread.to_device(numpy.add.outer(ky.get()**2, kx.get()**2))
		else:
			kk = self.kk

		if not hasattr(ky, 'thread') or ky.thread != self.thread:
			ky = self.thread.to_device(ky)
		if not hasattr(kx, 'thread') or kx.thread != self.thread:
			kx = self.thread.to_device(kx)
		if not hasattr(kk, 'thread') or kk.thread != self.thread:
			kk = self.thread.to_device(kk)

			
		tf = self.thread.array(kk.shape, self.dtype)

		sign = (self.thread,
				reikna.core.Type.from_value(tf).__repr__(),
				reikna.core.Type.from_value(list(phaseshifts_f.values())[0]).__repr__(),
				self.atoms[0]['zyx'].dtype.__repr__(),
				self.atoms[0]['B'].dtype.__repr__(),
				reikna.core.Type.from_value(ky).__repr__(),
				reikna.core.Type.from_value(kx).__repr__(),
				reikna.core.Type.from_value(kk).__repr__())

		if sign in compiled_atom_phaseshift:
			atom_phaseshift = compiled_atom_phaseshift[sign]
		else:
			atom_phaseshift =  AtomPhaseshift(tf, list(phaseshifts_f.values())[0], self.atoms[0]['zyx'][1], self.atoms[0]['zyx'][2], self.atoms[0]['B'], ky, kx, kk).compile(self.thread)
			compiled_atom_phaseshift[sign] = atom_phaseshift
			
		xsign = (self.thread, reikna.core.Type.from_value(tf).__repr__())

		if xsign in compiled_expi:
			expi = compiled_expi[xsign]
		else:
			expi = Expi(tf).compile(self.thread)
			compiled_expi[xsign] = expi
			
		for a in self.atoms:
			atom_phaseshift(tf, phaseshifts_f[a['Z']], a['zyx'][1], a['zyx'][2], a['B']/(4*numpy.pi**2), ky, kx, kk)

		if xsign in compiled_fft_gpu:
			fft_gpu = compiled_fft_gpu[xsign]
		else:
			fft_gpu = reikna.fft.FFT(tf).compile(self.thread)
			compiled_fft_gpu[xsign] = fft_gpu

			
		tmp = self.thread.temp_array(tf.shape, tf.dtype, tf.strides)
		fft_gpu(tmp, tf, 1)

		#self.thread.copy_array(tf, tmp)

		self.transmission_function = tmp

	def apply(self, wave):
		if not hasattr(wave, 'thread') or wave.thread != self.thread:
			wave = self.thread.to_device(wave)
		
		if self.transmission_function is None:
			self.generate_tf()

		wave *= self.transmission_function
		
		if self.forgetful:
			self.transmission_function = None
		return wave
