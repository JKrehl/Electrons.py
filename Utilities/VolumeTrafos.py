from __future__ import division, generators

import numpy
import numexpr

from scipy import ndimage

from . import CoordinateTrafos as CT

class VTrafo3D:

	def __init__(self, trafo, shape, origin='center', units='equal'):
		assert isinstance(trafo, CT.Trafo3D)
		self.trafo = trafo.inv

		self.shape = shape
		
		self.origin = origin

		if units == 'equal':
			self.units = numpy.array((1,1,1))
		elif numpy.isscalar(units):
			self.units = numpy.array((units,units,units))
		elif hasattr(units, '__getitem__'):
			self.units = numpy.require(units)
		else:
			raise AttributeError

		z = numpy.arange(self.shape[0])
		y = numpy.arange(self.shape[1])
		x = numpy.arange(self.shape[2])
		
		if self.origin == 'center':
			z -= shp[0]//2
			y -= shp[1]//2
			x -= shp[2]//2
		elif self.origin == 'zero':
			pass
		else:
			raise AttributeError

		nz, ny, nx, off = self.trafo.apply_to_bases(z,y,x)

		self.to_coords = 
		

	def apply_to(self, V):


		
