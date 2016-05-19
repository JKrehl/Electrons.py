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

class GPUArraySwitcher:
	def __init__(self, array, thread=None):
		self.memory = array

		if thread is not None:
			self.gpu = thread.to_device(array)
			self._mode = "memory"
		else:
			self.gpu = None
			self._mode = "memory_locked"

	@property
	def on_gpu(self):
		if self._mode == "memory_locked":
			raise BufferError("requested gpu array without enabling gpu operations")
		if self._mode == "memory":
			self.gpu.thread.to_device(self.memory, self.gpu)
			self._mode = "gpu"

		return self.gpu

	@property
	def in_mem(self):
		if self._mode == "gpu":
			self.gpu.thread.from_device(self.gpu, self.memory)
			self.mode = "memory"

		return self.memory