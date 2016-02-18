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
				${tf.store_idx}(idy, idx, ${mulp}(${ps.load_idx}(idy, idx), ${expx}(${addx}(${muli}(COMPLEX_CTR(float2)(0,1), ${adds}(${muly}(${ys}, ${ky.load_idx}(idy)), ${mulx}(${xs}, ${kx.load_idx}(idx)))), ${mulB}(-${B}/8., ${kk.load_idx}(idy,idx))))));
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
	
class Expi(reikna.core.Computation):
	def __init__(self, arg):
		super().__init__([reikna.core.Parameter('res', reikna.core.Annotation(arg, 'o')),
						  reikna.core.Parameter('arg', reikna.core.Annotation(arg, 'i'))])
		
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
				${res.store_idx}(idy, idx, ${exp}(${mul}(COMPLEX_CTR(${arg.ctype})(0,1), ${div}(${arg.load_idx}(idy, idx), ${f}))));
				}
			</%def>
			""")
			
		plan.kernel_call(template.get_def('function'),
						 [res, arg],
						 global_size=arg.shape,
						 render_kwds=dict(mul=reikna.cluda.functions.mul(arg.dtype, arg.dtype, out_dtype=res.dtype),
										  exp=reikna.cluda.functions.exp(res.dtype),
										  div = reikna.cluda.functions.div(arg.dtype, int, out_dtype=arg.dtype),
										  f=arg.size))
		return plan

class RegionMult(reikna.core.Computation):
	def __init__(self, A, B):
		super().__init__([reikna.core.Parameter('A', reikna.core.Annotation(A, 'io')),
						  reikna.core.Parameter('B', reikna.core.Annotation(B, 'i')),
						  reikna.core.Parameter('Aoff0', reikna.core.Annotation(numpy.uint, 's')),
						  reikna.core.Parameter('Aoff1', reikna.core.Annotation(numpy.uint, 's')),
						  reikna.core.Parameter('Boff0', reikna.core.Annotation(numpy.uint, 's')),
						  reikna.core.Parameter('Boff1', reikna.core.Annotation(numpy.uint, 's')),
						  reikna.core.Parameter('len0', reikna.core.Annotation(numpy.uint, 's')),
						  reikna.core.Parameter('len1', reikna.core.Annotation(numpy.uint, 's')),
						 ])
		
	def _build_plan(self, plan_factory, device_params, A, B, Aoff0, Aoff1, Boff0, Boff1, len0, len1):
		plan = plan_factory()

		template = reikna.helpers.template_from(
			"""
			<%def name='function(kernel_declaration, A, B, Aoff0, Aoff1, Boff0, Boff1, len0, len1)'>
				${kernel_declaration}
				{
				VIRTUAL_SKIP_THREADS;
				const VSIZE_T idx0 = virtual_global_id(0);
				const VSIZE_T idx1 = virtual_global_id(1);
				//if(idx0<${len0} && idx1<${len1}){
					//${A.store_idx}(idx0+${Aoff0}, idx1+${Aoff1}, ${mul}(${A.load_idx}(idx0+${Aoff0}, idx1+${Aoff1}), ${B.load_idx}(idx0+${Boff0}, idx1+${Boff1})));
				${A.store_idx}(${add}(idx0, ${Aoff0}), ${add}(idx1, ${Aoff1}), ${mul}(${A.load_idx}(${add}(idx0, ${Aoff0}), ${add}(idx1, ${Aoff1})), ${B.load_idx}(${add}(idx0, ${Boff0}), ${add}(idx1, ${Boff1}))));
				//}
				}
			</%def>
			""")
		
		plan.kernel_call(template.get_def('function'),
						 [A,B, Aoff0, Aoff1, Boff0, Boff1, len0, len1],
						 global_size=tuple(min(a,b) for a,b in zip(A.shape, B.shape)),
						 render_kwds=dict(mul=reikna.cluda.functions.mul(A.dtype, B.dtype, out_dtype=A.dtype),
										 add=reikna.cluda.functions.add(numpy.uint, numpy.uint)))
		return plan
	
class FlatAtomDW_ROI_GPU(PlaneOperator):
	
	def __init__(self, atoms, thread=None, phaseshifts_f=None,
				 ky=None, kx=None, kk=None,
				 roi=None, roi_x=None, roi_y=None,
				 roi_ky=None, roi_kx=None, roi_kk=None,
				 atom_potential_generator=WeickenmeierKohl, energy=None, y=None, x=None,
				 dtype=numpy.complex,
				 lazy=True, forgetful=True,
				 compiled_atom_phaseshift=None,
				 compiled_expi=None,
				 compiled_fft_gpu=None,
				 compiled_mult=None,
	):
		self.__dict__.update(dict(thread=thread, atoms=atoms,
								  phaseshifts_f=phaseshifts_f,
								  ky=ky, kx=kx, kk=kk,
								  roi=roi, roi_y=roi_y, roi_x=roi_x,
								  roi_ky=roi_ky, roi_kx=roi_kx, roi_kk=roi_kk,
								  atom_potential_generator=atom_potential_generator, energy=energy, y=y, x=x,
								  dtype=dtype,
								  lazy=lazy, forgetful=forgetful,
								  compiled_atom_phaseshift=compiled_atom_phaseshift,
								  compiled_expi=compiled_expi,
								  compiled_fft_gpu=compiled_fft_gpu,
								  compiled_mult=compiled_mult,
							  ))

		if self.compiled_atom_phaseshift is None: self.compiled_atom_phaseshift={}
		if self.compiled_expi is None: self.compiled_expi={}
		if self.compiled_fft_gpu is None: self.compiled_fft_gpu={}
		if self.compiled_mult is None: self.compiled_mult={}
		
		self.transfer_function = None
		if not self.lazy:
			self.generate_tf()
		
		self.z = numpy.mean(self.atoms['zyx'][:,0])
		
	@classmethod
	def inherit(cls, parent, atoms, **kwargs):
		args = {}
	
		args.update({k:v for k,v in parent.transfer_function_args.items() if v is not None})
		args.update({k:v for k,v in kwargs.items() if v is not None})

		if hasattr(parent, 'propagator_args') and 'thread' in parent.propagator_args and isinstance(parent.propagator_args['thread'], Thread):
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

		if 'roi' not in args or args['roi'] is None:
			args['roi'] = 1e-9
		
		if 'roi_y' not in args or args['roi_y'] is None:
			dy = args['y'][1]-args['y'][0]
			ay = numpy.ceil(args['roi']/dy)
			args['roi_y'] = dy*numpy.arange(-ay,ay+1)
			
		if 'roi_x' not in args or args['roi_x'] is None:
			dx = args['x'][1]-args['x'][0]
			ax = numpy.ceil(args['roi']/dx)
			args['roi_x'] = dx*numpy.arange(-ax,ax+1)
			
		if 'roi_ky' not in args or args['roi_ky'] is None:
			args['roi_ky'] = thread.to_device(FT.reciprocal_coords(args['roi_y']))

		if 'roi_kx' not in args or args['roi_kx'] is None:
			args['roi_kx'] = thread.to_device(FT.reciprocal_coords(args['roi_x']))

		if 'roi_kk' not in args or args['roi_kk'] is None:
			args['roi_kk'] = thread.to_device(numpy.add.outer(args['roi_ky'].get()**2, args['roi_kx'].get()**2))

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
				args['phaseshifts_f'].update({i: thread.to_device(args['atom_potential_generator'].phaseshift_f(i, args['energy'], args['roi_y'], args['roi_x'])) for i in set(numpy.unique(atoms['Z'])).difference(set(args['phaseshifts_f'].keys()))})

		if 'compiled_atom_phaseshift' not in args or args['compiled_atom_phaseshift'] is None:
			args['compiled_atom_phaseshift'] = {}
		if 'compiled_expi' not in args or args['compiled_expi'] is None:
			args['compiled_expi'] = {}
		if 'compiled_fft_gpu' not in args or args['compiled_fft_gpu'] is None:
			args['compiled_fft_gpu'] = {}
		if 'compiled_mult' not in args or args['compiled_mult'] is None:
			args['compiled_mult'] = {}

			
		
		parent.transfer_function_args.update(args)
		return cls(atoms, **args)

	# noinspection PyUnusedLocal,PyUnusedLocal
	def generate_tf(self):
		
		if self.phaseshifts_f is None:
			phaseshifts_f = {i: self.thread.to_device(self.atom_potential_generator.phaseshift_f(i, self.energy, self.roi_y, self.roi_x)) for i in numpy.unique(self.atoms['Z'])}
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

		if self.roi_y is None:
			roi_yl = -numpy.ceil(self.roi/dy)-1
			roi_yu = numpy.ceil(self.roi/dy)+1
			roi_y = dy*numpy.arange(roi_yl, roi_yu)
		else:
			roi_y = self.roi_y
			roi_yl = -(roi_y.size-1)//2
			roi_yu = (roi_y.size+1)//2
			
		if self.roi_x is None:
			roi_xl = -numpy.ceil(self.roi/dx)-1
			roi_xu = numpy.ceil(self.roi/dx)+1
			roi_x = dx*numpy.arange(roi_xl, roi_xu)
		else:
			roi_x = self.roi_x
			roi_xl = -(roi_x.size-1)//2
			roi_xu = (roi_x.size+1)//2
			
		if self.roi_ky is None:
			roi_ky = self.thread.to_device(FT.reciprocal_coords(roi_y))
		else:
			roi_ky = self.roi_ky

		if self.roi_kx is None:
			roi_kx = self.thread.to_device(FT.reciprocal_coords(roi_x))
		else:
			roi_kx = self.roi_kx

		if self.roi_kk is None:
			roi_kk =self.thread.to_device(numpy.add.outer(roi_ky.get()**2, roi_kx.get()**2))
		else:
			roi_kk = self.roi_kk

		if not hasattr(ky, 'thread') or ky.thread != self.thread:
			ky = self.thread.to_device(ky)
		if not hasattr(kx, 'thread') or kx.thread != self.thread:
			kx = self.thread.to_device(kx)
		if not hasattr(kk, 'thread') or kk.thread != self.thread:
			kk = self.thread.to_device(kk)

			
		self.transfer_function = self.thread.to_device(numpy.ones(kk.shape, self.dtype))
		itf = self.thread.array(roi_kk.shape, self.dtype)
		tmp = self.thread.temp_array(itf.shape, itf.dtype, itf.strides)

		sign = (self.thread,
				reikna.core.Type.from_value(itf).__repr__(),
				reikna.core.Type.from_value(list(phaseshifts_f.values())[0]).__repr__(),
				self.atoms[0]['zyx'].dtype.__repr__(),
				self.atoms[0]['B'].dtype.__repr__(),
				reikna.core.Type.from_value(roi_ky).__repr__(),
				reikna.core.Type.from_value(roi_kx).__repr__(),
				reikna.core.Type.from_value(roi_kk).__repr__())

		if sign in self.compiled_atom_phaseshift:
			atom_phaseshift = self.compiled_atom_phaseshift[sign]
		else:
			atom_phaseshift =  AtomPhaseshift(itf, list(phaseshifts_f.values())[0], self.atoms['zyx'][0,1], self.atoms['zyx'][0,2], self.atoms[0]['B'], roi_ky, roi_kx, roi_kk).compile(self.thread)
			self.compiled_atom_phaseshift[sign] = atom_phaseshift
			
		xsign = (self.thread, reikna.core.Type.from_value(itf).__repr__())

		if xsign in self.compiled_expi:
			expi = self.compiled_expi[xsign]
		else:
			expi = Expi(itf).compile(self.thread)
			self.compiled_expi[xsign] = expi

		if xsign in self.compiled_fft_gpu:
			fft_gpu = self.compiled_fft_gpu[xsign]
		else:
			fft_gpu = reikna.fft.FFT(itf).compile(self.thread)
			self.compiled_fft_gpu[xsign] = fft_gpu

		msign = (self.thread, reikna.core.Type.from_value(self.transfer_function).__repr__(), reikna.core.Type.from_value(itf).__repr__())
		
		if msign in self.compiled_fft_gpu:
			mult = self.compiled_mult[msign]
		else:
			mult = RegionMult(self.transfer_function, itf).compile(self.thread)
			self.compiled_mult[msign] = mult

		dy = self.y[1]-self.y[0]
		dx = self.x[1]-self.x[0]
		
		for a in self.atoms:
			py, px = a['zyx'][1], a['zyx'][2]
				
			rpy, ipy = numpy.modf((py-self.y[0])/dy)
			rpx, ipx = numpy.modf((px-self.x[0])/dx)
			ipy = int(ipy)
			ipx = int(ipx)
			
			atom_phaseshift(itf, phaseshifts_f[a['Z']], rpy*dy, rpx*dx, a['B'], roi_ky, roi_kx, roi_kk)
				
			fft_gpu(tmp, itf, 0)
			#tmp /= itf.size
			expi(itf, tmp)
				
			select = numpy.s_[ipy+roi_yl if ipy+roi_yl>=0 else 0:ipy+roi_yu if ipy+roi_yu<=self.y.size else self.y.size,
							  ipx+roi_xl if ipx+roi_xl>=0 else 0:ipx+roi_xu if ipx+roi_xu<=self.x.size else self.x.size]

			iselect = numpy.s_[0 if ipy+roi_yl>=0 else -(ipy+roi_yl): roi_y.size if ipy+roi_yu<=self.y.size else self.y.size+self.roi_y.size-ipy-roi_yu,
							   0 if ipx+roi_xl>=0 else -(ipx+roi_xl): roi_x.size if ipx+roi_xu<=self.x.size else self.x.size+self.roi_x.size-ipx-roi_xu]
				
			assert select[0].stop-select[0].start==iselect[0].stop-iselect[0].start and select[1].stop-select[1].start==iselect[1].stop-iselect[1].start, (py, self.y[0], dy, select, iselect)
			if (select[0].stop-select[0].start)>0 and (select[1].stop-select[1].start)>0:
				mult(self.transfer_function, itf, select[0].start, select[1].start, iselect[0].start, iselect[1].start, select[0].stop-select[0].start, select[1].stop-select[1].start)

		del itf

	def apply(self, wave):
		if not hasattr(wave, 'thread') or wave.thread != self.thread:
			wave = self.thread.to_device(wave)
		
		if self.transfer_function is None:
			self.generate_tf()
			
		wave *= self.transfer_function
		
		if self.forgetful:
			self.transfer_function = None
		return wave
