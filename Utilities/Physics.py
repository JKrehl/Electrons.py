from __future__ import division, absolute_import

import numpy

emass = 5.11e5 # eV
hbarc = 1.9732697181209571e-7 # eVm
bohrr = 5.29177e-11 # m
echarge = 1.44e-9 # Vm
sol = 299792458 # m/s
hbar = 6.5821192815e-16 # eVs

def wavenumber(energy):
	return ((energy+emass)**2-emass**2)**.5/hbarc

def interaction_const(energy):
	return (energy+emass)/(hbarc*(energy*(energy+2*emass))**.5)

def momentum(energy):
	return numpy.sqrt(2*energy*emass+energy**2)/sol

def speed(energy):
	return momentum(energy)*sol**2/(emass*lorentz(energy))

def lorentz(energy):
	return energy/emass+1
