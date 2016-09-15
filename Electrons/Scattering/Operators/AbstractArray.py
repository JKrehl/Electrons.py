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

import reikna.cluda.cuda
import reikna.cluda.ocl

class AbstractArray:
	_modes = {}

	def __new__(cls, array, mode = "numpy", *args, **kwargs):
		assert mode in cls._modes
		if mode in cls._modes:
			if isinstance(array, AbstractArray):
				return array._as(mode, *args, **kwargs)
			else:
				return cls._modes[mode].from_ndarray(array, *args, **kwargs)

	@classmethod
	def from_ndarray(cls, array, *args, **kwargs):
		raise NotImplemented

	def to_ndarray(self):
		raise NotImplemented

	def to_same_mode(self, *args, **kwargs):
		return self

	def _as(self, mode, *args, **kwargs):
		assert mode in self._modes
		if mode == self.mode:
			return self.to_same_mode(*args, **kwargs)
		else:
			return self._modes[mode].from_ndarray(self.to_ndarray(), *args, **kwargs)

class AbstractArray_Numpy(numpy.ndarray, AbstractArray):
	mode = "numpy"

	@classmethod
	def from_ndarray(cls, array):
		self = array.view(__class__)
		return self

	def to_ndarray(self):
		return self.view(numpy.ndarray)

AbstractArray._modes['numpy'] = AbstractArray_Numpy

class AbstractArray_CUDA(reikna.cluda.cuda.Array, AbstractArray):
	mode = "cuda"

	def __init__(self, *args, **kwargs):
		pass

	@staticmethod
	def get_thread(thread=None):
		if isinstance(thread, reikna.cluda.cuda.Thread):
			return thread
		elif isinstance(thread, reikna.cluda.ocl.Thread):
			raise TypeError("Thread of wrong CLUDA Backend given")
		else:
			return reikna.cluda.cuda.Thread.create()

	@classmethod
	def from_ndarray(cls, array, thread = None):
		if isinstance(thread, reikna.cluda.cuda.Thread):
			pass
		elif isinstance(thread, reikna.cluda.ocl.Thread):
			raise TypeError("Thread of wrong CLUDA Backend given")
		else:
			thread = reikna.cluda.cuda.Thread.create()

		self = __class__.get_thread(thread).to_device(array)
		self.__class__ = __class__

		return self

	def to_ndarray(self):
		return self.get()

	def to_same_mode(self, thread=None):
		if self.thread == thread or thread == None:
			return self
		else:
			return self.from_ndarray(self.to_ndarray(), thread)

AbstractArray._modes['cuda'] = AbstractArray_CUDA

class AbstractArray_OpenCL(reikna.cluda.ocl.Array, AbstractArray):
	mode = "opencl"

	def __init__(self, *args, **kwargs):
		pass

	@staticmethod
	def get_thread(thread=None):
		if isinstance(thread, reikna.cluda.ocl.Thread):
			return thread
		elif isinstance(thread, reikna.cluda.cuda.Thread):
			raise TypeError("Thread of wrong CLUDA Backend given")
		else:
			return reikna.cluda.ocl.Thread.create()

	@classmethod
	def from_ndarray(cls, array, thread=None):

		self = __class__.get_thread(thread).to_device(array)
		self.__class__ = __class__

		return self

	def to_ndarray(self):
		return self.get()

	def to_same_mode(self, thread = None):
		if self.thread == thread or thread == None:
			return self
		else:
			return self.from_ndarray(self.to_ndarray(), thread)

AbstractArray._modes['opencl'] = AbstractArray_OpenCL
