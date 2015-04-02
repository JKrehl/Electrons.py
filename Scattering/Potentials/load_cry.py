from __future__ import print_function

import numpy
import re

import os.path
__dir__ = os.path.dirname(os.path.abspath(__file__))+'/'

elements = numpy.load(__dir__+'elements.npy')

def load_cry(filename):
	fle = open(filename, 'r')
	fle.seek(0)

	renum = re.compile("([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)")

	lst = []
	i = 0

	while len(lst)<3:
		line = re.match('^.*(?=$|#)',fle.readline())
		i += 1

		if line!=None:
			line = renum.findall(line.group())
			if len(line)>0:
				lst.append(line)
				
	fle.close()
	
	dat = numpy.genfromtxt(filename, skiprows=i, dtype=['O','f','f','f','f','f'])

	size = numpy.array(lst[1], dtype=numpy.float)

	res = dict()
	res['sgroup'] = float(lst[0][0])
	res['extent'] = size
	res['angles'] = numpy.array(lst[2], dtype=numpy.float)
	res['atoms'] = numpy.empty(dat.shape[0], dtype=[('Z','i'), ('xyz','3f'), ('B','f'), ('occ','f')])

	xyz = numpy.vstack((dat['f1'],dat['f2'],dat['f3'])).T

	Z = [numpy.argwhere(i==elements['symbol']) for i in dat['f0']]
	
	res['atoms']['Z'] = Z
	res['atoms']['xyz'] = 1e-9*(xyz-.5)*size[None,:]
	res['atoms']['B'] = 1e-18*dat['f4']
	res['atoms']['occ'] = dat['f5']

	return res
