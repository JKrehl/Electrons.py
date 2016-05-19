#!/usr/bin/env python

import numpy

import numexpr

import pyximport
pyximport.install()
from . import Interpolator2D_cy as cy

class Interpolator2D:
	def __init__(self, ybase, xbase, data):
		self.ybase = ybase
		self.xbase = xbase
		self.data = data
		
		self.yscale = self.ybase[1]-self.ybase[0]
		self.ylength = self.ybase.size
		self.yoffset = self.ybase[0]
		
		self.xscale = self.xbase[1]-self.xbase[0]
		self.xlength = self.xbase.size
		self.xoffset = self.xbase[0]
		
	def __call__(self, ycoords, xcoords):
		if numpy.isscalar(ycoords): ycoords = (ycoords,)
		if numpy.isscalar(xcoords): xcoords = (xcoords,)
		
		yrem, yind = numpy.modf(numexpr.evaluate("(ycoords-yoffset)/yscale", local_dict=dict(ycoords=numpy.require(ycoords), yoffset=self.yoffset, yscale=self.yscale)))
		xrem, xind = numpy.modf(numexpr.evaluate("(xcoords-xoffset)/xscale", local_dict=dict(xcoords=numpy.require(xcoords), xoffset=self.xoffset, xscale=self.xscale)))
		
		yind = yind.astype(numpy.int)
		xind = xind.astype(numpy.int)
		
		res = cy.interpolator2d(yind, xind, yrem, xrem, self.data, self.ylength, self.xlength)
		
		return res
