import numpy

import scipy.misc
from matplotlib import cm

from . import SymNorm

import math
import decimal

def human_readable_limits(lower, upper):
	diff = int(math.floor(math.log10((upper-lower)/2)))
	nlower = math.floor(decimal.Decimal(lower*10**-diff))*10**decimal.Decimal(diff)
	nupper = math.ceil(decimal.Decimal(upper*10**-diff))*10**decimal.Decimal(diff)
	return (nlower, nupper)

def imexport(path, a, y, x, norm=SymNorm(), cmap='viridis', clim=None):
	sm = cm.ScalarMappable(norm=norm, cmap=cmap)
	sm.set_array(a)

	if clim is None:
		sm.autoscale()
		sm.set_clim(*human_readable_limits(*sm.get_clim()))
	else:
		sm.set_clim(*clim)

	scipy.misc.imsave(path, sm.to_rgba(a))

	with open(path+".pars", "w") as file:
		file.writelines("{}\n".format(x[0]))
		file.writelines("{}\n".format((x[-1]+x[1]-x[0])))
		file.writelines("{}\n".format(y[0]))
		file.writelines("{}\n".format((y[-1]+y[1]-y[0])))
		file.writelines("{}\n".format(sm.get_clim()[0]))
		file.writelines("{}\n".format(sm.get_clim()[1]))
		file.writelines("{}\n".format(sm.get_cmap().name))