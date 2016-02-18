import numpy
import numexpr
import scipy.interpolate

from ....Mathematics import FourierTransforms as FT
from ...Potentials.AtomPotentials import WeickenmeierKohl

from ..Base import PlaneOperator

class FlatAtomDW_ROI(PlaneOperator):
	def __init__(self, atoms, roi=None,
				 roi_y=None, roi_x=None,
				 roi_ky=None, roi_kx=None, roi_kk=None,
				 phaseshifts_f=None,
				 ky=None, kx=None, kk=None,
				 atom_potential_generator=WeickenmeierKohl, energy=None, y=None, x=None,
				 dtype=numpy.complex,
				 lazy=True, forgetful=True):
		self.__dict__.update(dict(atoms=atoms,
								  roi=roi, roi_y=roi_y, roi_x=roi_x,
								  roi_ky=roi_ky, roi_kx=roi_kx, roi_kk=roi_kk,
								  phaseshifts_f=phaseshifts_f,
								  ky=ky, kx=kx, kk=kk,
								  atom_potential_generator=atom_potential_generator, energy=energy, y=y, x=x,
								  dtype=dtype,
								  lazy=lazy, forgetful=forgetful))

		self.phaseshifts_f = None
		self.transfer_function = None
		if not self.lazy:
			self.generate_tf()
		
		self.z = numpy.mean(self.atoms['zyx'][:,0])
		
	@classmethod
	def inherit(cls, parent, atoms, **kwargs):
		args = {}
		
		args.update(parent.transfer_function_args)
		args.update(kwargs)
		
		args.update({s:parent.__dict__[s] for s in ['y', 'x', 'ky', 'kx', 'kk'] if s not in args or args[s] is None})
		
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
			
		if 'roi_kk' not in args or args['roi_kk'] is None:
			args['roi_kk'] = numpy.add.outer(args['roi_ky']**2, args['roi_kx']**2)
			
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
				args['phaseshifts_f'].update({i: args['atom_potential_generator'].phaseshift_f(i, args['energy'], args['roi_y'], args['roi_x']) for i in set(numpy.unique(atoms['Z'])).difference(set(args['phaseshifts_f'].keys()))})
			
		parent.transfer_function_args.update(args)
	
		return cls(atoms, **args)

	# noinspection PyUnusedLocal
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
			self.phaseshifts_f = {i: self.atom_potential_generator.cis_phaseshift_f(i, self.energy, roi_y, roi_x) for i in numpy.unique(self.atoms['Z'])}
		
		tf = numpy.ones(kk.shape, dtype=self.dtype)
		itf = numpy.empty(roi_kk.shape, dtype=self.dtype)
		
		for a in self.atoms:
			py, px = a['zyx'][1], a['zyx'][2]
			rpy, ipy = numpy.modf((py-self.y[0])/dy)
			rpx, ipx = numpy.modf((px-self.x[0])/dx)

			select = numpy.s_[ipy+roi_yl if ipy+roi_yl>=0 else 0:ipy+roi_yu if ipy+roi_yu<=self.y.size else self.y.size,
							  ipx+roi_xl if ipx+roi_xl>=0 else 0:ipx+roi_xu if ipx+roi_xu<=self.x.size else self.x.size]

			iselect = numpy.s_[0 if ipy+roi_yl>=0 else -(ipy+roi_yl): roi_y.size if ipy+roi_yu<=self.y.size else self.y.size+self.roi_y.size-ipy-roi_yu,
							   0 if ipx+roi_xl>=0 else -(ipx+roi_xl): roi_x.size if ipx+roi_xu<=self.x.size else self.x.size+self.roi_x.size-ipx-roi_xu]

			itf = numexpr.evaluate('ps*exp(-1j*(xs*kx+ys*ky)-kk*B/8)',
								   local_dict={'ps':self.phaseshifts_f[a['Z']],
											   'ys':dy*rpy, 'xs':dx*rpx,
											   'ky':roi_ky[:,None], 'kx':roi_kx[None,:],
											   'kk':roi_kk, 'B':a['B']})

			tf[select] *= FT.ifft(itf)[iselect]
		self.transfer_function = tf

	def apply(self, wave):
		if self.transfer_function is None:
			self.generate_tf()
		res = numexpr.evaluate("tf*wave", local_dict=dict(tf=self.transfer_function, wave=wave))
		if self.forgetful:
			self.transfer_function = None
		return res
