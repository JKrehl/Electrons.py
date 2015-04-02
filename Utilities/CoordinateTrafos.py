from __future__ import division, generators

import numpy, re

def lazy_property(fn):
	decorated_name = '__'+fn.__name__
	@property
	def __lazy_property(self):
		if not hasattr(self, decorated_name):
			setattr(self, decorated_name, fn(self))
		return getattr(self, decorated_name)
	return __lazy_property

def normalise_quart(quart):
	quart = numpy.require(quart)
	
	if len(quart.shape)==1:
		return quart / numpy.linalg.norm(quart, axis=-1)
	else:
		return quart / numpy.linalg.norm(quart, axis=-1)[...,None]

def quart_to_rotmat(quart):
	quart = numpy.require(quart)

	r, i, j, k = quart[...,0], quart[...,1], quart[...,2], quart[...,3]
	rr, ri, rj, rk = r*r, 2*r*i, 2*r*j, 2*r*k
	ii, ij, ik = i*i, 2*i*j, 2*i*k
	jj, jk = j*j, 2*j*k
	kk = k*k

	return numpy.array(((rr+ii-jj-kk, ij+rk, ik-rj), (ij-rk, rr-ii+jj-kk, jk+ri), (ik+rj, jk-ri, rr-ii-jj+kk)))

def axisangle_to_rotmat(axisangle, unit='rad'):
	axisangle = numpy.require(axisangle)
	axi = axisangle[...,:3]
	angl = axisangle[...,3]
	if unit=='deg':
		angl = numpy.deg2rad(angl)
	
	res = numpy.empty(axisangle.shape, dtype=numpy.float)
	res[...,0] = numpy.cos(angl/2)
	res[...,1:4] = numpy.sin(angl/2)[...,None]*axi/numpy.linalg.norm(axi, axis=-1)[...,None]
	return quart_to_rotmat(res)

def affine_to_4mat(affine):
	res = numpy.eye(4,dtype=affine.dtype)
	res[:3,:3] = affine
	return res

class Trafo:
	def __init__(self):
		pass

	@property
	def dtype(self):
		return self.mat.dtype

	@property
	def inv(self):
		return self.__class__(mat=numpy.linalg.inv(self.mat))
	
	@lazy_property
	def n(self):
		return self.mat.shape[0]-1
	
	@property
	def affine(self):
		return self.mat[:self.n, :self.n]
	
	@property
	def offset(self):
		return self.mat[:self.n, self.n]
	
	@property
	def diag(self):
		return numpy.diag(self.mat)
	
	@property
	def is_diag(self):
		return numpy.all(self.mat==numpy.diag(self.diag))
	
	@property
	def affine_diag(self):
		return numpy.diag(self.mat[self.n, self.n])
	
	@property
	def affine_is_diag(self):
		return numpy.all(self.affine==numpy.diag(numpy.diag(self.affine))) 
	
	def apply_(self, a, axis=-1):
		a = numpy.require(a)
		if a.shape[axis] == self.n+1:
			return numpy.rollaxis(numpy.tensordot(a,self.mat, axes = ((axis,),(1,))), -1,axis)
		elif a.shape[axis] == self.n:
			return numpy.rollaxis(numpy.tensordot(a,self.affine, axes = ((axis,),(1,)))+self.offset[...,:], -1,axis)
	
	def apply_affine(self, a, axis=-1):
		a = numpy.require(a)
		if a.shape[axis] == self.n+1:
			return numpy.rollaxis(numpy.tensordot(a,self.mat, axes = ((axis,),(1,))), -1,axis)
		elif a.shape[axis] == self.n:
			return numpy.rollaxis(numpy.tensordot(a,self.affine, axes = ((axis,),(1,)))+self.offset[...,:], -1,axis)
		

	def apply_to_base(self, dimension, b):
		if numpy.isscalar(b):
			return (b*self.affine[:,dimension])
		elif isinstance(b, numpy.ndarray):
			return (b[...,None]*self.affine[...,:,dimension])
		
	def apply_to_bases(self, *bases):
		assert len(bases) == self.n, 'Dimensional Mismatch: {}\t{}'.format(len(bases), self.n)
		return tuple(self.apply_to_base(i, b) for i,b in enumerate(bases))+(self.offset,)
	
	def __mul__(self, other):
		if isinstance(other, self.__class__):
			return self.__class__(mat=numpy.dot(self.mat, other.mat))
		elif isinstance(other, numpy.ndarray):
			return self.apply_(other)
		
	def __rmul__(self, other):
		return self*other
	
	def __repr__(self):
		if hasattr(self.__class__, '__name__'):
			return '<{}> \n{}'.format(self.__class__.__name__, repr(self.mat))
		else:
			return '<{}> \n{}'.format('Some Trafo', repr(self.mat)) 
	
	def copy(self):
		return self.__class__(mat=self.mat.copy())

class Trafo3D(Trafo):
	__name__ = 'Trafo3D'
	
	def __init__(self, **kwargs):

		if kwargs.has_key('mat'): 
			self.mat = kwargs['mat']
		else:
			self.mat = numpy.eye(4,dtype=numpy.float)

			def keyfun(e):
				match = re.search('-?\d+', e[0])
				return 0 if match is None else int(match.group(0))
		
			sorted_kwargs = sorted(kwargs.items(), key=keyfun)
			
			for name, value in sorted_kwargs:
				if name.startswith('affine'):
					self.mat = numpy.dot(affine_to_4mat(value), self.mat)
				elif name.startswith('quar'):
					self.mat = numpy.dot(affine_to_4mat(quart_to_rotmat(normalise_quart(value))), self.mat)
				elif name.startswith('axisrad'):
					self.mat = numpy.dot(affine_to_4mat(axisangle_to_rotmat(value)), self.mat)
				elif name.startswith('axisdeg'):
					self.mat = numpy.dot(affine_to_4mat(axisangle_to_rotmat(value, 'deg')), self.mat)
				elif name.startswith('rotmat'): 
					self.mat = numpy.dot(affine_as_4mat(value), self.mat)
				elif name.startswith('scale'): 
					self.add_postscale(value)
				elif name.startswith('transpose'): 
					self.mat[:3,:] = self.mat[:3,:][list(value)]
				elif name.startswith('shift'): 
					self.add_postshift(value)
				elif name.startswith('inv'):
					if val: self.mat = numpy.linalg.inv(self.mat)
				elif name.startswith('mat_inv'):
					if val: self.mat[:3,:3] = numpy.linalg.inv(self.mat[:3,:3])
				elif name.startswith('shift_inv'):
					if val: self.mat[:3,3] = -self.mat[:3,3]
				else:
					raise TypeError('Invalid keyword')

	def add_postscale(self, value):
		mt = numpy.eye(4, dtype=self.mat.dtype)
		mt[((0,1,2),(0,1,2))] *= value
		self.mat = numpy.dot(mt, self.mat)
		
	def add_prescale(self, value):
		mt = numpy.eye(4, dtype=self.mat.dtype)
		mt[((0,1,2),(0,1,2))] *= value
		self.mat = numpy.dot(self.mat, mt)
		
	def add_preshift(self, value):
		mt = numpy.eye(4, dtype=self.mat.dtype)
		mt[:3,3] = value
		self.mat = numpy.dot(self.mat, mt)
		
	def add_postshift(self, value):
		mt = numpy.eye(4, dtype=self.mat.dtype)
		mt[:3,3] = value
		self.mat = numpy.dot(mt, self.mat)
		
	def coord_shift(self, value):
		value = numpy.require(value)
		self.add_preshift(value)
		self.add_postshift(-value)
		
	def coord_scale(self, value):
		value = numpy.require(value)
		self.add_prescale(value)
		self.add_postscale(1/value)

class Trafo2D(Trafo):
	__name__ = 'Trafo2D'
	
	def __init__(self, **kwargs):
		self.mat = numpy.eye(3)
		
		def keyfun(e):
			match = re.search('-?\d+', e[0])
			return 0 if match is None else int(match.group(0))
		
		sorted_kwargs = sorted(kwargs.items(), key=keyfun)
		
		for name, value in sorted_kwargs:
			if name.startswith('shift'): 
				self.add_shift(value)
			elif name.startswith('rad'):
				self.add_rotation(value)
			elif name.startswith('deg'):
				self.add_rotation(numpy.deg2rad(value))
			elif name.startswith('mat'):
				self.mat = numpy.dot(value, self.mat)
			elif name.startswith('scale'):
				self.add_postscale(value)
			else:
				raise TypeError('Invalid keyword')

	def add_shift(self, shift):
		mt = numpy.eye(3, dtype=self.mat.dtype)
		mt[:2,2] = shift
		self.mat = numpy.dot(mt, self.mat)
		
	def add_rotation(self, angle):        
		mt = numpy.eye(3, dtype=self.mat.dtype)
		mt[:2,:2] = ((numpy.cos(angle), -numpy.sin(angle)),(numpy.sin(angle), numpy.cos(angle)))
		self.mat = numpy.dot(mt, self.mat)

	def add_postscale(self, value):
		mt = numpy.eye(3, dtype=self.mat.dtype)
		mt[((0,1),(0,1))] *= value
		self.mat = numpy.dot(mt, self.mat)
		
	def add_prescale(self, value):
		mt = numpy.eye(3, dtype=self.mat.dtype)
		mt[((0,1),(0,1))] *= value
		self.mat = numpy.dot(self.mat, mt)
		
	def add_preshift(self, value):
		mt = numpy.eye(3, dtype=self.mat.dtype)
		mt[:2,2] = value
		self.mat = numpy.dot(self.mat, mt)
		
	def add_postshift(self, value):
		mt = numpy.eye(3, dtype=self.mat.dtype)
		mt[:2,2] = value
		self.mat = numpy.dot(mt, self.mat)		
