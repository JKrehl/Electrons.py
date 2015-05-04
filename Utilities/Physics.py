from __future__ import division, absolute_import

import numpy

emass = 5.11e5 # eV
hbarc = 1.9732697181209571e-7 # eVm
bohrr = 5.29177e-11 # m
echarge = 1.44e-9 # Vm

def wavenumber(energy):
	return ((energy+emass)**2-emass**2)**.5/hbarc

def interaction_const(energy):
	return (energy+emass)/(hbarc*(energy*(energy+2*emass))**.5)
