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

class Operator:
	def apply(self, wave):
		raise NotImplemented

class PlaneOperator(Operator):
	def __init__(self, z):
		self.z = z

class IntervalOperator:
	def __init__(self, zi, zf):
		self.zi = zi
		self.zf = zf
	
	def split(self, z):
		raise NotImplemented

class SliceStacker():
	def __init__(self, callback=None):
		super().__init__(None)

		self.callback = callback



class Slice(PlaneOperator):
	def __init__(self, z, callback=None):
		super().__init__(z)
		self.callback = callback

		self.slice = None

	def apply(self,wave):
		if self.callback is not None:
			self.callback(wave)
		else:
			self.slice = wave.copy()
		return wave

class SliceStacker():
	def __init__(self, callback=None):
		self.callback = callback
		self.stack = []

	def draw(self, z):
		self.stack.append(Slice(z, self.callback))
		return self.stack[-1]

	def get(self):
		return numpy.concatenate(tuple(i.slice.reshape(1, *i.slice.shape) for i in self.stack), axis=0)