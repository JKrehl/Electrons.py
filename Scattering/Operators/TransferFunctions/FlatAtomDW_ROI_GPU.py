 
from __future__ import division, print_function

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
				${tf.store_idx}(idy, idx, ${mulp}(${ps.load_idx}(idy, idx), ${expx}(${addx}(${muli}(COMPLEX_CTR(float2)(0,-1), ${adds}(${muly}(${ys}, ${ky.load_idx}(idy)), ${mulx}(${xs}, ${kx.load_idx}(idx)))), ${mulB}(-${B}/8, ${kk.load_idx}(idy,idx))))));
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
		super().__init__([reikna.core.Parameter('arg', reikna.core.Annotation(arg, 'io'))])
		
	def _build_plan(self, plan_factory, device_params, arg):
		plan = plan_factory()

		template = reikna.helpers.template_from(
			"""
			<%def name='function(kernel_declaration, arg)'>
				${kernel_declaration}
				{
				VIRTUAL_SKIP_THREADS;
				const VSIZE_T idy = virtual_global_id(0);
				const VSIZE_T idx = virtual_global_id(1);
				${arg.store_idx}(idy, idx, ${exp}(${mul}(COMPLEX_CTR(${arg.ctype})(0,1), ${arg.load_idx}(idy, idx))));
				}
			</%def>
			""")
			
		plan.kernel_call(template.get_def('function'),
						 [arg],
						 global_size=arg.shape,
						 render_kwds=dict(mul=reikna.cluda.functions.mul(arg.dtype, arg.dtype), exp=reikna.cluda.functions.exp(arg.dtype)))
		return plan

compiled_fft_gpu = {}
	
class FlatAtomDW_ROI_GPU(PlaneOperator):
	def __init__(self, atoms, thread=None, phaseshifts_f=None,
				 ky=None, kx=None, kk=None,
				 roi=None, roi_x=None, roi_y=None,
				 roi_ky=None, roi_kx=None, roi_kk=None,
				 atom_potential_generator=WeickenmeierKohl, energy=None, y=None, x=None,
				 dtype=numpy.complex,
				 lazy=True, forgetful=True):
		self.__dict__.update(dict(thread=thread, atoms=atoms,
								  phaseshifts_f=phaseshifts_f,
								  ky=ky, kx=kx, kk=kk,
								  roi=roi, roi_y=roi_y, roi_x=roi_x,
								  roi_ky=roi_ky, roi_kx=roi_kx, roi_kk=roi_kk,
								  atom_potential_generator=atom_potential_generator, energy=energy, y=y, x=x,
								  dtype=dtype,
								  lazy=lazy, forgetful=forgetful))
		
		self.transfer_function = None
		if not self.lazy:
			self.generate_tf()
		
		self.z = numpy.mean(self.atoms['zyx'][:,0])
		
	@classmethod
	def inherit(cls, parent, atoms, **kwargs):
		args = {}
	
		args.update({k:v for k,v in parent.transfer_function_args.items() if v is not None})
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

		if 'phaseshifts_f' not in args or args['phaseshifts_f'] is None:
			if hasattr(parent, 'phaseshifts_f') and parent.phaseshifts_f is not None:
				args['phaseshifts_f'] = {k:thread.to_device(v) for k,v in parent.phaseshifts_f.items()}
			else:
				if 'energy' not in args or args['energy'] is None:
					args['energy'] = parent.energy
				if 'atom_potential_generator' not in args or args['atom_potential_generator'] is None:
					args['atom_potential_generator'] = parent.atom_potential_generator
				args['phaseshifts_f'] = {i: thread.to_device(args['atom_potential_generator'].phaseshift_f(i, args['energy'], args['roi_y'], args['roi_x'])) for i in numpy.unique(atoms['Z'])}

		parent.transfer_function_args.update(args)
		return cls(atoms, **args)
			
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
			roi_yl = -numpy.ceil(self.roi/dy)
			roi_yu = numpy.ceil(self.roi/dy)+1
			roi_y = dy*numpy.arange(roi_yl, roi_yu)
		else:
			roi_y = self.roi_y
			roi_yl = -(roi_y.size-1)//2
			roi_yu = (roi_y.size+1)//2
			
		if self.roi_x is None:
			roi_xl = -numpy.ceil(self.roi/dx)
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

			
		tf = numpy.ones(kk.shape, self.dtype)
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

		if sign in compiled_atom_phaseshift:
			atom_phaseshift = compiled_atom_phaseshift[sign]
		else:
			atom_phaseshift =  AtomPhaseshift(itf, list(phaseshifts_f.values())[0], self.atoms['zyx'][0,1], self.atoms['zyx'][0,2], self.atoms[0]['B'], roi_ky, roi_kx, roi_kk).compile(self.thread)
			compiled_atom_phaseshift[sign] = atom_phaseshift
			
		xsign = (self.thread, reikna.core.Type.from_value(itf).__repr__())

		if xsign in compiled_expi:
			expi = compiled_expi[xsign]
		else:
			expi = Expi(itf).compile(self.thread)
			compiled_expi[xsign] = expi

		if xsign in compiled_fft_gpu:
			fft_gpu = compiled_fft_gpu[xsign]
		else:
			fft_gpu = reikna.fft.FFT(itf).compile(self.thread)
			compiled_fft_gpu[xsign] = fft_gpu

		dy = self.y[1]-self.y[0]
		dx = self.x[1]-self.x[0]
		
		for a in self.atoms:
			py, px = a['zyx'][1], a['zyx'][2]
			rpy, ipy = numpy.modf((py-self.y[0])/dy)
			rpx, ipx = numpy.modf((px-self.x[0])/dx)
			ipy = int(ipy)
			ipx = int(ipx)
			
			atom_phaseshift(itf, phaseshifts_f[a['Z']], rpy*dy, rpx*dx, a['B'], roi_ky, roi_kx, roi_kk)
			
			fft_gpu(tmp, itf, 1)
			itf[...] = tmp[...]
			expi(itf)
			
			select = numpy.s_[ipy+roi_yl if ipy+roi_yl>=0 else 0:ipy+roi_yu if ipy+roi_yu<=self.y.size else self.y.size,
							  ipx+roi_xl if ipx+roi_xl>=0 else 0:ipx+roi_xu if ipx+roi_xu<=self.x.size else self.x.size]

			iselect = numpy.s_[0 if ipy+roi_yl>0 else ipy+roi_yl: roi_y.size if ipy+roi_yu<self.y.size else self.y.size-ipy-roi_yu,
							   0 if ipx+roi_xl>0 else ipx+roi_xl: roi_x.size if ipx+roi_xu<self.x.size else self.x.size-ipx-roi_xu]

			tf[select] *= itf.get()[iselect]
	
		self.transfer_function = self.thread.to_device(tf)

	def apply(self, wave):
		if not hasattr(wave, 'thread') or wave.thread != self.thread:
			wave = self.thread.to_device(wave)
		
		if self.transfer_function is None:
			self.generate_tf()
		
		wave *= self.transfer_function
		
		if self.forgetful:
			self.transfer_function = None
		return wave
