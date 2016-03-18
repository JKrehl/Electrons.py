## Scripts and programs for TEM scattering simulation and tomographic reconstruction.

The code is oriented towards maximum extensivity and reusability. A lot of concepts are implemented as relatively powerful meta-concepts with explicit implementations (which inherit a lot of the meta-concepts functionalities). 

There is still room for more abstraction, but without a concrete need and a forseeable set of use-cases I do consider it fruitless.

The scripts are largely undocumented and might carry crippling bugs. Explanation, documentation and usage examples will be added upon request (and at the discretion of the author).

## Content

#### Scattering
- a general framework for algrorithms, where a chain of operators is applied to the wave is supplied
- the multislice algortihms, a simple projection and a single scattering algorithm are implemented
- the following parametrisations for the electron scattering potential are supplied (Weickenmeier & Kohl (both), Kirkland and Peng-Dudarev)
- the transfer-function is constructed by applying a Debye-Waller factor, a ROI-patching method is implemented (for large fields of view)
- the propagator is implemented paraxially both in real and Fourier space
- some operators have been ported via Reikna on GPU

#### Tomography
- a general framework for iterative reconstruction using general linear least squares solvers is supplied
- the linear operator is split in a projector which constructs the tensor virtually from the transversal repetitions of a kernel and the kernel
- the kernel (the minimal representation of the tensor) can be computed as either a ray (as in the Radon transform) or a Fresnel propagation convolution kernel
- the projectors can construct the tensor for the 2d problem from a 2d kernel (as in the in-plane ray transform) or the 3d problem from a 2d kernel (as in the ray transform with an aligned tilt axis) or semi-3d kernel (as in the Fresnel kernel with an aligned tilt axis)
- the operations are based on COO-formatted sparse arrays
- a powerful system for the hybrid existence of objects on disk and in memory is supplied for on-demand loading and persistence of arrays in memory (in preparation for a CSR/CSC-model for the arrays)

#### Mathematics
- coordinate transformations in 2d and 3d, chaining of transformations
- Fourier transformation related functions
- a 2d and a 3d interpolation algorithm with COO-sparse in- and output (only linear interpolation)
- calculation of Laplace kernels, optimized of azimuthal uniform gain

#### Utilites
- an atomic add for use in Cython even for complex data types)
- a Jupyter progress bar for tracking Python iterator objects
- a slice player for Matplotlib
- an atom viewer for VisPy using the array of atoms type from /Scattering/Potentials/AtomicPotential
- a norm for matplotlib for colourmaps symmetric around 0 (or some other value)
- ...

## Dependencies
- Python3 <https://www.python.org/>
- NumPy <http://www.numpy.org/>
- SciPy <https://www.scipy.org/>
- NumExpr <https://github.com/pydata/numexpr>
- h5py <http://www.h5py.org/>
- Cython <http://cython.org/>
- Reikna <https://reikna.readthedocs.org/en/latest/>
- Matplotlib <http://matplotlib.org/>
- VisPy <http://vispy.org/>
- pyFFTW <https://hgomersall.github.io/pyFFTW/index.html> (optional)
