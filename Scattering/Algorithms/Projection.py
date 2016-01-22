import numpy
from ..Operators import OperatorChain
from ..Operators.TransferFunctions import FlatAtomDW
from ..Potentials.AtomPotentials import WeickenmeierKohl
from ...Mathematics import FourierTransforms as FT
from ...Utilities import Physics
from ...Utilities import Progress

class Projection:
	def __init__(self, y, x, potential, energy, zi=None, zf=None, trafo=None, forgetful=False,
	             atom_potential_generator=WeickenmeierKohl, transfer_function=FlatAtomDW, transfer_function_args=None):

		if transfer_function_args is None: transfer_function_args = {}

		self.__dict__.update(dict(y=y, x=x, energy=energy,
								  zi=zi, zf=zf, trafo=trafo,
								  forgetful = forgetful,
								  atom_potential_generator=atom_potential_generator, transfer_function=transfer_function, transfer_function_args=transfer_function_args))
		self.prepared = False
		self.opchain = None
		self.k = None
		self.kx = None
		self.ky = None
		self.kk = None
		
		if self.trafo is not None:
			self.potential = potential.transform(self.trafo)
		else:
			self.potential = potential.copy()
		self.potential.zsort()

		if self.zf is None:
			self.zf = self.potential.zmax()
		if self.zi is None:
			self.zi = self.potential.zmin()
		
	def prepare(self):
		self.potential.zsort()

		self.opchain = OperatorChain(zi=self.zi, zf=self.zf)

		self.k = Physics.wavenumber(self.energy)

		self.kx, self.ky = FT.reciprocal_coords(self.x, self.y)
		self.kk =  numpy.add.outer(self.kx**2, self.ky**2)
		
		#for i in range(self.potential.atoms.size):
		self.opchain.append(self.transfer_function.inherit(self, self.potential.atoms))#[i:i+1]))

		self.opchain.impose_zorder()
		
		self.prepared = True
		
	def run(self, wave=None, progress=False):
		if wave is None:
			wave = numpy.ones(self.x.shape+self.y.shape, dtype=numpy.complex)
	
		if not self.prepared:
			self.prepare()

		if progress:
			for op in Progress(self.opchain['operator'], self.opchain.size):
				wave = op.apply(wave)
		else:
			for op in self.opchain['operator']:
				wave = op.apply(wave)

		if hasattr(wave, 'thread'):
			thread = wave.thread
			wave = wave.get()
			thread.synchronize()
			thread.release()
			del thread

		return wave
