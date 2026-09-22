# MTEX: the crystallographic texture analysis toolbox
from . import geometry as _geometry
from .geometry import *  # noqa: F401,F403
from .geometry import __all__ as _geometry_all
from .maps import *  # noqa: F401,F403
from .maps import __all__ as _maps_all
from .functions import *  # noqa: F401,F403
from .functions import __all__ as _functions_all
from .tensors import *  # noqa: F401,F403
from .tensors import __all__ as _tensors_all
from .analysis import *  # noqa: F401,F403
from .analysis import __all__ as _analysis_all
from .io import *  # noqa: F401,F403
from .io import __all__ as _io_all
from .plotting import *  # noqa: F401,F403
from .plotting import __all__ as _plotting_all
from .wizard import importWizard  # noqa: F401

__all__ = list(_geometry_all) + list(_maps_all) + list(_functions_all) + list(_tensors_all) + list(_analysis_all) + list(_io_all) + list(_plotting_all) + ["importWizard", "xvector", "yvector", "zvector"]


# xvector, yvector, zvector are fresh on every access
def __getattr__(name):
  if name in ("xvector", "yvector", "zvector"):
    return getattr(_geometry, name)
  raise AttributeError(name)
