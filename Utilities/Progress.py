from __future__ import print_function

try:
	from IPython.html.widgets import HBox, IntProgress, Latex, FloatProgress
	from IPython.display import display, HTML
	ipython = True
except ImportError:
	ipython = False

class Progress():
	_value = -1
	_iter = None
	__iter__ = lambda self: self
	
	def __init__(self, iterable, length=None, keep=False):
		self._iter = iterable.__iter__()
		self.keep = keep

		if ipython:
			if length is None:
				length = len(iterable)

			self.box = HBox(width="100%", align="center")
			#HTML("""<style> .widget-progress {flex: 2;}</style>""")
			self.bar = IntProgress(width="auto", min=0, max=length, margin='auto')
			self.text = Latex(width="auto", margin='auto')
			l = len(str(length))
			self.textfmt = "[{}/{}]".format("{:%dd}"%l, length)
			self.text.value = self.textfmt.format(self.bar.value)

			self.box.children = (self.bar,self.text)

			display(self.box)
			
	@property
	def value(self):
		return self._value
	
	@value.setter
	def value(self, value):
		self._value = value
		if ipython:
			self.bar.value = self._value
			self.text.value = self.textfmt.format(self.bar.value)
			
	def __next__(self):
		if ipython:
			self.value += 1
			try:
				return self._iter.__next__()
			except StopIteration:
				if not self.keep:
					self.box.close()
				raise StopIteration

class FlProgress():
	_value = -1

	def __init__(self, mn, mx):
		self.box = HBox(width="100%", align="center")
		#HTML("""<style> .widget-progress {flex: 2;}</style>""")
		self.bar = FloatProgress(width="auto", min=mn, max=mx, margin='auto')
		self.text = Latex(width="auto", margin='auto')
		l = max(len(str(mn)),len(str(mx)))
		self.textfmt = "[{} < {} > {}]".format(mn, "{:%dg}"%l, mx)
		self.text.value = self.textfmt.format(self.bar.value)

		self.box.children = (self.bar, self.text)

		display(self.box)

	@property
	def value(self):
		return self._value
	
	@value.setter
	def value(self, value):
		self._value = value
		if ipython:
			self.bar.value = self._value
			self.text.value = self.textfmt.format(self.bar.value)

	def finished(self):
		self.box.close()
