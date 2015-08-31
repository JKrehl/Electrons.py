from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import register_cmap
import numpy

class SymNorm(Normalize):    
	def __init__(self, vmin=None, vmax=None, clip=False):
		super().__init__(vmin, vmax, clip)
	
	def autoscale(self, A):
		mx = numpy.amax(A)
		mn = numpy.amin(A)
		self.vmin = min(mn,-mx)
		self.vmax = max(-mn,mx)
	
	def autoscale_None(self, A):
		' autoscale only None-valued vmin or vmax'
		if self.vmin is None and numpy.size(A) > 0:
			self.vmin = min(numpy.amin(A), -numpy.amax(A))
		if self.vmax is None and numpy.size(A) > 0:
			self.vmax = max(-numpy.amin(A), numpy.amax(A))
			
cdict = {}
cdict["blue"] = ((0,1,1),(.5,1,0),(1,0,0))
cdict["red"] = ((0,0,0),(.5,0,1),(1,1,1))
cdict["green"] = ((0,0,0),(1,0,0))
cdict["alpha"] = ((0,1,1),(.5,0,0),(1,1,1))

register_cmap('btr', cmap=LinearSegmentedColormap("btr", cdict))
