from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from . import Mathematics
from . import Scattering
from . import Tomography
from . import Utilities

__all__ = [s for s in dir() if not s.startswith('_')]
