from __future__ import print_function

try:
    from IPython.html.widgets import HBox, IntProgress, Latex
    from IPython.display import display, HTML
    ipython = True
except ImportError:
    ipython = False

import collections

class Progress(collections.Iterator):
    _value = -1
    
    def __init__(self, iterable, length=None, keep=False):
        self.iter = iterable.__iter__()
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
    
    def next(self):
        if ipython:
            self.value += 1
            try:
                return self.iter.next()
            except StopIteration:
                if not self.keep:
                    self.box.close()
                raise StopIteration
