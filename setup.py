from setuptools import setup, find_packages

import versioneer

__version__ = versioneer.get_version()

setup_requires = ['versioneer']

setup(
	name='Electrons',
	version= __version__,
	cmdclass = versioneer.get_cmdclass(),
    packages = find_packages(),
	include_package_data = True,
	package_data = {'': ['**/*.pxd', '**/*.pyx', '**/*.pyxbld',
	                     '**/*.npy', '**/*.hpp', '**/*.cpp', '**/*.dat']},
)