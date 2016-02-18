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

from scipy import ndimage

from . import CoordinateTrafos as CT

import pyximport
pyximport.install()
from . import VolumeTrafos_cy as cy

class VolumeTrafo3D:

	def __init__(self, trafo, shape, tshape, origin='center'):
		assert isinstance(trafo, CT.Trafo3D)
		self.trafo = trafo.inv

		self.shape = shape
		self.tshape = tshape
		
		self.origin = origin

		self.idxtrafo = self.trafo.copy()
		
		if self.origin == 'center':
			self.idxtrafo.preshift([-i//2 for i in self.shape])
			self.idxtrafo.postshift([i//2 for i in self.tshape])
		elif self.origin == 'zero':
			pass
		else:
			raise AttributeError

		tzyxo = self.idxtrafo.inv.apply_to_bases(*tuple(numpy.arange(i) for i in self.shape))
		self.proj_coords = numexpr.evaluate('z+y+x+o', local_dict=dict(z=tzyxo[0][:,:,None,None], y=tzyxo[1][:,None,:,None], x=tzyxo[2][:,None,None,:], o=tzyxo[3][:,None,None,None]))

		zyxo = self.idxtrafo.apply_to_bases(*tuple(numpy.arange(i) for i in self.tshape))
		self.bproj_coords = numexpr.evaluate('z+y+x+o', local_dict=dict(z=zyxo[0][:,:,None,None], y=zyxo[1][:,None,:,None], x=zyxo[2][:,None,None,:], o=zyxo[3][:,None,None,None]))

	def apply_to(self, a):
		#return ndimage.interpolation.map_coordinates(a, self.proj_coords, order=1)
		return cy.map_coordinates3(a, self.proj_coords)
    
	def apply_fro(self, a):
		#return ndimage.interpolation.map_coordinates(a, self.bproj_coords, order=1)
		return cy.map_coordinates3(a, self.bproj_coords)
