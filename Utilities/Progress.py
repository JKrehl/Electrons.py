from IPython.display import display
import ipywidgets

try:
	get_ipython()
	ipy = True
except NameError:
	ipy = False

class Progress():
	__iter__ = lambda self: self
	__len__ = lambda self: self.length

	def __init__(self, iterable, length=None, keep=False, style=''):
		self._iter = iterable.__iter__()
		self.keep = keep
		self._value = -1

		if length is None:
			self.length = len(iterable)
		else:
			self.length = length

		if ipy:
			self.box = ipywidgets.HBox(width="100%", align="center")
			self.bar = ipywidgets.IntProgress(min=0, max=self.length, margin="auto", width="auto")
			self.label = ipywidgets.Latex(margin="auto", width="auto")
			self.labelproto = "[{}/{}]".format("{:%dd}"%len(str(self.length)), self.length)
			self.label.value = self.labelproto.format(self.bar.value)

			self.box.children = (self.bar, self.label)

			display(self.box)

	@property
	def value(self):
		return self._value

	@value.setter
	def value(self, value):
		self._value = value
		if ipy:
			self.bar.value = self._value
			self.label.value = self.labelproto.format(self.bar.value)

	def __next__(self):
		self.value += 1
		try:
			return self._iter.__next__()
		except StopIteration:
			if ipy and not self.keep:
				self.box.close()
			raise StopIteration
