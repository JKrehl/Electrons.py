 
from __future__ import division, print_function

import numpy
import numexpr
import scipy.interpolate

import reikna.core
import reikna.cluda
import reikna.helpers

from ....Mathematics import FourierTransforms as FT
from ...Potentials.AtomPotentials import WeickenmeierKohl

from ..Base import PlaneOperator

compiled_atom_phaseshift = {}

class AtomPhaseshift2(reikna.core.Computation):
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
				${tf.store_idx}(idy, idx, ${tf.load_idx}(idy, idx)+${mulp}(${ps.load_idx}(idy, idx), ${pus}(${adds}(${muly}(${ys}, ${ky.load_idx}(idy)), ${mulx}(${xs}, ${kx.load_idx}(idx)))), ${expB}(${mulB}(-${B}/8, ${kk.load_idx}(idy,idx)))));
				}
			</%def>
			""")

		Bkk_dtype = reikna.cluda.dtypes.result_type(B.dtype, kk.dtype)
		kxs_dtype = reikna.cluda.dtypes.result_type(xs.dtype, kx.dtype)
		kys_dtype = reikna.cluda.dtypes.result_type(ys.dtype, ky.dtype)
		ks_dtype = reikna.cluda.dtypes.result_type(kys_dtype, kxs_dtype)
		cks_dtype = reikna.cluda.dtypes.complex_for(ks_dtype)
		
		plan.kernel_call(template.get_def('function'),
						 [tf, ps, ys, xs, B, ky, kx, kk],
						 global_size=tf.shape,
						 render_kwds=dict(mulB=reikna.cluda.functions.mul(B.dtype, kk.dtype, out_dtype=Bkk_dtype),
										  expB=reikna.cluda.functions.exp(Bkk_dtype),
										  mulx=reikna.cluda.functions.mul(xs.dtype, kx.dtype, out_dtype=kxs_dtype),
										  muly=reikna.cluda.functions.mul(ys.dtype, ky.dtype, out_dtype=kys_dtype),
										  adds=reikna.cluda.functions.add(kys_dtype, kxs_dtype, out_dtype=ks_dtype),
										  pus=reikna.cluda.functions.polar_unit(ks_dtype),
										  mulp=reikna.cluda.functions.mul(ps.dtype, cks_dtype, Bkk_dtype, out_dtype=tf.dtype)
									  ))
		return plan
	
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
				${tf.store_idx}(idy, idx, ${tf.load_idx}(idy, idx)+${mulp}(${ps.load_idx}(idy, idx), ${expx}(${addx}(${muli}(COMPLEX_CTR(float2)(0,1), ${adds}(${muly}(${ys}, ${ky.load_idx}(idy)), ${mulx}(${xs}, ${kx.load_idx}(idx)))), ${mulB}(-${B}/8, ${kk.load_idx}(idy,idx))))));
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
	
class FlatAtomDW_GPU(PlaneOperator):
	def __init__(self, thread, atoms, phaseshifts_f=None,
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
		
		self.transfer_function = None
		if not self.lazy:
			self.generate_tf()
		
		self.z = numpy.mean(self.atoms['zyx'][:,0])

	@classmethod
	def inherit(cls, parent, atoms, **kwargs):
		thread = parent.thread
		args = {}

		if hasattr(parent, 'g_kk'):
			args.update(dict(kk=parent.g_kk))
		if hasattr(parent, 'g_ky') and hasattr(parent, 'g_kx'):
			args.update(dict(ky=parent.g_ky, kx=parent.g_kx))
		if hasattr(parent, 'y') and hasattr(parent, 'x'):
			args.update(dict(y=parent.y, x=parent.x))
		

		if hasattr(parent, 'phaseshifts_f'):
			args.update(dict(phaseshifts_f=parent.phaseshifts_f))
		else:
			args.update(dict(energy=parent.energy, x=parent.x, y=parent.y))
			if hasattr(parent, 'atom_potential_generator'):
				args.update(dict(atom_potential_generator=parent.atom_potential_generator))

		args.update(kwargs)

		return cls(thread, atoms, **args)
			
	def generate_tf(self):
		
		if self.phaseshifts_f is None:
			phaseshifts_f = {i: self.thread.to_device(self.atom_potential_generator.phaseshift_f(i, self.energy, self.y, self.x)) for i in numpy.unique(self.atoms['Z'])}
		else:
			phaseshifts_f = self.phaseshifts_f
			
		if self.ky is None:
			ky = FT.reciprocal_coords(self.y)
		else:
			ky = self.ky
			
		if self.kx is None:
			kx = FT.reciprocal_coords(self.x)
		else:
			kx = self.kx

		if self.kk is None:
			kk = numpy.add.outer(ky**2, kx**2)
		else:
			kk = self.kk

		
	
		if not hasattr(kk, 'thread'):
			kk = self.thread.to_device(kk)
		if not hasattr(ky, 'thread'):
			ky = self.thread.to_device(ky)
		if not hasattr(kx, 'thread'):
			kx = self.thread.to_device(kx)

		tf = self.thread.array(kk.shape, self.dtype)
		tf[...] = 0

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
			atom_phaseshift(tf, phaseshifts_f[a['Z']], a['zyx'][2], a['zyx'][1], a['B'], ky, kx, kk)

		if xsign in compiled_fft_gpu:
			fft_gpu = compiled_fft_gpu[xsign]
		else:
			fft_gpu = reikna.fft.FFT(tf).compile(self.thread)
			compiled_fft_gpu[xsign] = fft_gpu

			
		tmp = self.thread.temp_array(tf.shape, tf.dtype, tf.strides)
		fft_gpu(tmp, tf, 1)
		tf[...] = tmp[...]
		expi(tf)
	
		self.transfer_function = tf

	def apply(self, wave):
		if self.transfer_function is None:
			self.generate_tf()
		
		wave *= self.transfer_function
		
		if self.forgetful:
			self.transfer_function = None
		return wave
