import numpy
import numexpr


from ...Utilities import Progress
from ...Utilities import CompressedSparse as CS
from ...Utilities.Magic import apply_if

from .Kernel import Kernel

class RayKernel(Kernel):
	def __init__(self, path=None, memory_strategy=0):
		if not hasattr(self, '_arrays'):
			self._arrays = {}

		self._arrays.update(y=0, x=0, t=0, d=0, mask=2, dtype=2, itype=2, dat=1, row=1, col=1)

		if memory_strategy == 0:
			self._arrays = {key:0 if val==1 else val for key,val in self._arrays.items()}
		elif memory_strategy == 1:
			pass
		elif memory_strategy == 2:
			self._arrays = {key:1 if val==0 else val for key,val in self._arrays.items()}
		else:
			raise ValueError

		super().__init__(path, memory_strategy)

	def init(self, y, x, t, d, mask=None, dtype=numpy.float64, itype=numpy.int32):
		self.init_empty_arrays()

		self.y = y
		self.x = x
		self.t = t
		self.d = d

		self.mask = mask

		self.dtype = dtype
		self.itype = itype

		return self

	@property
	def shape(self):
		assert self.t is not None and self.d is not None and self.y is not None and self.x is not None
		return self.t.shape+self.d.shape+self.y.shape+self.x.shape

	@property
	def fshape(self):
		assert self.t is not None and self.d is not None and self.y is not None and self.x is not None
		return (self.t.size*self.d.size, self.y.size*self.x.size)

	def prep(self):
		xd = abs(self.x[1] - self.x[0])
		yd = abs(self.y[1] - self.y[0])
		dd = abs(self.d[1] - self.d[0])

		if self.mask:
			mask = numpy.add.outer((numpy.arange(self.y.size)-self.y.size//2)**2,(numpy.arange(self.x.size)-self.x.size//2)**2).flatten()<(.25*min(self.y.size**2, self.x.size**2))
		else:
			mask = None

		return (self.y/yd, self.x/xd, self.d/dd, yd*xd/dd, mask)

	def calc_one_angle(self, ti, y, x, d, unit_area, mask):
		al = abs((-ti+numpy.pi/4)%(numpy.pi/2) - numpy.pi/4)
		a = .5*(numpy.cos(al)-numpy.sin(al))
		b = .5*(numpy.cos(al)+numpy.sin(al))
		h = 1/numpy.cos(al)

		if b==a: f = 0
		else: f = h/(b-a)

		e = numexpr.evaluate("x*cos(t)+y*sin(t) -d", local_dict=dict(x=x[None,None,:], y=y[None,:,None], d=d[:,None,None], t=ti)).reshape(d.size, y.size*x.size)
		if mask is not None:
			sel = numexpr.evaluate("mask&(abs(e)<(b+.5))", local_dict=dict(mask=mask[None,:], e=e, b=b))
		else:
			sel = numexpr.evaluate("(abs(e)<(b+.5))", local_dict=dict(e=e, b=b))

		e = e[sel]
		dsel, yxsel = numpy.where(sel)
		dsel = dsel.astype(self.itype, copy=False)
		yxsel = yxsel.astype(self.itype, copy=False)

		calcs = 'where({0}<-b, 0, where({0}<-a, .5*f*({0}+b)**2, where({0}<a, .5*f*(a-b)**2+h*({0}+a), where({0}<b, 1-.5*f*(b-{0})**2, 1))))'
		ker = numexpr.evaluate('area*('+calcs.format('(dif+.5)')+'-'+calcs.format('(dif-.5)')+')', local_dict=dict(dif=e, a=a, b=b, f=f, h=h, area=unit_area))
		csel = ker>=0
		dsel = dsel[csel]
		yxsel = yxsel[csel]

		return (ker[csel], dsel, yxsel)


	def calc(self, progress=False):
		y, x, d, unit_area, mask = self.prep()

		dat_concatenator = self.arrays.dat.concatenator(self.dtype)
		row_concatenator = self.arrays.row.concatenator(self.itype)
		col_concatenator = self.arrays.col.concatenator(self.itype)

		for it,ti in apply_if(enumerate(self.t), Progress, progress, length=self.t.size):
			idat, irow, icol = self.calc_one_angle(ti, y, x, d, unit_area, mask)

			dat_concatenator.append(idat)
			row_concatenator.append(irow+self.d.size*it)
			col_concatenator.append(icol)

			del idat, irow, icol

		dat_concatenator.finalize()
		del dat_concatenator
		row_concatenator.finalize()
		del row_concatenator
		col_concatenator.finalize()
		del col_concatenator

		return self

class RayKernelCS(RayKernel):
	def __init__(self, y, x, t, d, mask=None, dtype=numpy.float64, itype=numpy.int32, memory_strategy=0, path=None):
		self._arrays = dict(Tdat=1, Trow=1, Tcol=1)
		super().__init__(y, x, t, d, mask=mask, dtype=dtype, itype=itype, memory_strategy=memory_strategy, path=path)

	def calc(self, track_progress=False):
		super().calc(track_progress)

		with self.open():
			dat = self.dat[...]
			row = self.row[...]
			col = self.col[...]

		self.row, self.dat, self.col = CS.compress_sparse(row, self.fshape[0], dat, col)
		self.Trow, self.Tdat, self.Tcol = CS.compress_sparse(col, self.fshape[1], dat, row)

		del dat, row, col

		return self