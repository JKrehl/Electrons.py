from __future__ import division, generators, unicode_literals

import numpy, re

from ..Utilities.Magic import lazy_property

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
			return numpy.multiply.outer(self.affine[:,dimension],b)
		
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
	
	def __init__(self, mat=None):

		if mat is not None:
			self.mat = numpy.require(mat).copy()
		else:
			self.mat = numpy.eye(4,dtype=numpy.float)


	def affine_transform(self, value):
		self.mat = numpy.dot(affine_to_4mat(numpy.require(value)), self.mat)
		return self

	def quarternion(self, value):
		self.mat = numpy.dot(affine_to_4mat(quart_to_rotmat(normalise_quart(value))), self.mat)
		return self

	def axisrad(self, value):
		self.mat = numpy.dot(affine_to_4mat(axisangle_to_rotmat(value)), self.mat)
		return self

	def axisdeg(self, value):
		self.mat = numpy.dot(affine_to_4mat(axisangle_to_rotmat(value, 'deg')), self.mat)
		return self

	def rotmat(self, value):
		self.mat = numpy.dot(affine_as_4mat(value), self.mat)
		return self

	def transpose(self, value):
		self.mat[:3,:] = self.mat[:3,:][list(value)]
		return self

	def permute(self, value):
		self.mat = self.mat[:,list(value)+[3,]][list(value)+[3,],:]
		return self
	
	def postscale(self, value):
		mt = numpy.eye(4, dtype=self.mat.dtype)
		mt[((0,1,2),(0,1,2))] *= value
		self.mat = numpy.dot(mt, self.mat)

		return self

	def scale(self, value):
		return self.postscale(self, value)
		
	def prescale(self, value):
		mt = numpy.eye(4, dtype=self.mat.dtype)
		mt[((0,1,2),(0,1,2))] *= value
		self.mat = numpy.dot(self.mat, mt)

		return self
		
	def preshift(self, value):
		mt = numpy.eye(4, dtype=self.mat.dtype)
		mt[:3,3] = value
		self.mat = numpy.dot(self.mat, mt)

		return self
		
	def postshift(self, value):
		mt = numpy.eye(4, dtype=self.mat.dtype)
		mt[:3,3] = value
		self.mat = numpy.dot(mt, self.mat)

		return self
		
	def coord_shift(self, value):
		value = numpy.require(value)
		self.add_preshift(value)
		self.add_postshift(-value)

		return self
		
	def coord_scale(self, value):
		value = numpy.require(value)
		self.add_prescale(value)
		self.add_postscale(1/value)

		return self

class Trafo2D(Trafo):
	__name__ = 'Trafo2D'
	
	def __init__(self, mat=None):
		if mat is None:
			self.mat = numpy.eye(3)
		else:
			self.mat = mat.copy()
	
	def shift(self, shift):
		mt = numpy.eye(3, dtype=self.mat.dtype)
		mt[:2,2] = shift
		self.mat = numpy.dot(mt, self.mat)

		return self
		
	def rotation(self, angle):        
		mt = numpy.eye(3, dtype=self.mat.dtype)
		mt[:2,:2] = ((numpy.cos(angle), -numpy.sin(angle)),(numpy.sin(angle), numpy.cos(angle)))
		self.mat = numpy.dot(mt, self.mat)
		return self

	def rad(self, angle):
		return self.rotation(angle)
	
	def deg(self, angle):
		return self.rotation(numpy.deg2rad(angle))

	def postscale(self, value):
		mt = numpy.eye(3, dtype=self.mat.dtype)
		mt[((0,1),(0,1))] *= value
		self.mat = numpy.dot(mt, self.mat)

		return self

	def scale(self, value):
		return self.postscale(value)
	
	def prescale(self, value):
		mt = numpy.eye(3, dtype=self.mat.dtype)
		mt[((0,1),(0,1))] *= value
		self.mat = numpy.dot(self.mat, mt)

		return self
		
	def preshift(self, value):
		mt = numpy.eye(3, dtype=self.mat.dtype)
		mt[:2,2] = value
		self.mat = numpy.dot(self.mat, mt)

		return self
		
	def postshift(self, value):
		mt = numpy.eye(3, dtype=self.mat.dtype)
		mt[:2,2] = value
		self.mat = numpy.dot(mt, self.mat)

		return self
