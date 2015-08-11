 
from __future__ import division, print_function

import numpy
import numexpr
import scipy.interpolate

from matplotlib.pyplot import *

from ....Mathematics import FourierTransforms as FT
from ...Potentials.AtomPotentials import WeickenmeierKohl

from ..Base import PlaneOperator

class FlatAtomDW_ROI(PlaneOperator):
	def __init__(self, atoms, roi=None,
				 roi_x=None, roi_y=None,
				 roi_kx=None, roi_ky=None, roi_kk=None,
				 phaseshifts_f=None,
				 ky=None, kx=None, kk=None,
				 atom_potential_generator=WeickenmeierKohl, energy=None, y=None, x=None,
				 dtype=numpy.complex,
				 lazy=True, forgetful=True):
		self.__dict__.update(dict(atoms=atoms,
								  roi=roi, roi_x=roi_x, roi_y=roi_y,
								  roi_kx=roi_kx, roi_ky=roi_ky, roi_kk=roi_kk,
								  phaseshifts_f=phaseshifts_f,
								  ky=ky, kx=kx, kk=kk,
								  atom_potential_generator=atom_potential_generator, energy=energy, y=y, x=x,
								  dtype=dtype,
								  lazy=lazy, forgetful=forgetful))
		
		self.transfer_function = None
		if not self.lazy:
			self.generate_tf()
		
		self.z = numpy.mean(self.atoms['zyx'][:,0])

	@staticmethod
	def ms_prep(parent):
		if hasattr(parent, 'roi_y') and hasattr(parent, 'roi_x'):
			roi_y = parent.roi_y
			roi_x = parent.roi_x
		elif hasattr(parent, 'transfer_function_args') and ('roi_y' in parent.transfer_function_args and 'roi_x' in parent.transfer_function_args):
			roi_y = parent.parent.transfer_function_args['roi_y']
			roi_x = parent.parent.transfer_function_args['roi_x']
		elif hasattr(parent, 'roi') or (hasattr(parent, 'transfer_function_args') and 'roi' in parent.transfer_function_args):
			if hasattr(parent, 'roi'):
				roi = parent.roi
			else:
				roi = parent.transfer_function_args['roi']

			dy = parent.y[1]-parent.y[0]
			dx = parent.x[1]-parent.x[0]
			ay = numpy.ceil(roi/dy)
			ax = numpy.ceil(roi/dx)
			roi_y = dy*numpy.arange(-ay,ay+1)
			roi_x = dx*numpy.arange(-ax,ax+1)
		else:
			raise ValueError
		parent.phaseshifts_f = {i: parent.atom_potential_generator.phaseshift_f(i, parent.energy, roi_y, roi_x) for i in numpy.unique(parent.potential.atoms['Z'])}

	@classmethod
	def inherit(cls, parent, atoms, **kwargs):
		args = {}

		if hasattr(parent, 'kk'):
			args.update(dict(kk=parent.kk))
		if hasattr(parent, 'ky') and hasattr(parent, 'kx'):
			args.update(dict(ky=parent.ky, kx=parent.kx))
		if hasattr(parent, 'y') and hasattr(parent, 'x'):
			args.update(dict(y=parent.y, x=parent.x))
			
		if hasattr(parent, 'phaseshifts_f'):
			args.update(dict(phaseshifts_f=parent.phaseshifts_f))
		else:
			args.update(dict(energy=parent.energy, x=parent.x, y=parent.y))
			if hasattr(parent, 'atom_potential_generator'):
				args.update(dict(atom_potential_generator=parent.atom_potential_generator))

		if hasattr(parent, 'transfer_function_args'):
			args.update(parent.transfer_function_args)

		args.update(kwargs)

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
			args['roi_ky'] = FT.reciprocal_coords(args['roi_y'])

		if 'roi_kx' not in args or args['roi_kx'] is None:
			args['roi_kx'] = FT.reciprocal_coords(args['roi_x'])
			
		return cls(atoms, **args)
			
	def generate_tf(self):
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

		dy = self.y[1]-self.y[0]
		dx = self.x[1]-self.x[0]
		
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
			roi_ky = FT.reciprocal_coords(roi_y)
		else:
			roi_ky = self.roi_ky

		if self.roi_kx is None:
			roi_kx = FT.reciprocal_coords(roi_x)
		else:
			roi_kx = self.roi_kx

		if self.roi_kk is None:
			roi_kk = numpy.add.outer(roi_ky**2, roi_kx**2)
		else:
			roi_kk = self.roi_kk
		
		if self.phaseshifts_f is None:
			self.phaseshifts_f = {i: self.atom_potential_generator.phaseshift_f(i, self.energy, yoi, xoi) for i in numpy.unique(self.atoms['Z'])}
		
		tf = numpy.ones(kk.shape, dtype=self.dtype)
		itf = numpy.empty(roi_kk.shape, dtype=self.dtype)
		
		for a in self.atoms:
			py, px = a['zyx'][1], a['zyx'][2]
			rpy, ipy = numpy.modf((py-self.y[0])/dy)
			rpx, ipx = numpy.modf((px-self.x[0])/dx)
			
			select = numpy.s_[ipy+roi_yl if ipy+roi_yl>=0 else 0:ipy+roi_yu if ipy+roi_yu<=self.y.size else self.y.size,
							  ipx+roi_xl if ipx+roi_xl>=0 else 0:ipx+roi_xu if ipx+roi_xu<=self.x.size else self.x.size]

			iselect = numpy.s_[0 if ipy+roi_yl>0 else ipy+roi_yl: roi_y.size if ipy+roi_yu<self.y.size else self.y.size-ipy-roi_yu,
							   0 if ipx+roi_xl>0 else ipx+roi_xl: roi_x.size if ipx+roi_xu<self.x.size else self.x.size-ipx-roi_xu]

			itf =  numexpr.evaluate('ps*exp(1j*(xs*kx+ys*ky)-kk*B/8)',
								   local_dict={'ps':self.phaseshifts_f[a['Z']],
											   'ys':0*dy*rpy, 'xs':0*dx*rpx,
											   'kx':roi_kx[:,None], 'ky':roi_ky[None,:],
											   'kk':roi_kk, 'B':a['B']})
			
			tf[select] *= numpy.exp(1j*FT.ifft(itf))[iselect]
			
		self.transfer_function =tf

	def apply(self, wave):
		if self.transfer_function is None:
			self.generate_tf()
		res = numexpr.evaluate("tf*wave", local_dict=dict(tf=self.transfer_function, wave=wave))
		if self.forgetful:
			self.transfer_function = None
		return res
