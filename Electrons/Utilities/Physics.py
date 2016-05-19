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
