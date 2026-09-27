# MTEX: the crystallographic texture analysis toolbox
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError, version as _version

# the version of the installed distribution
try:
  __version__ = _version("mtex")
except _PackageNotFoundError:
  __version__ = "unknown"

from typing import TYPE_CHECKING

from . import geometry as _geometry
from . import maps as _maps
from . import maps3d as _maps3d
from . import functions as _functions
from . import tensors as _tensors
from . import analysis as _analysis
from . import io as _io
from . import plotting as _plotting
from .geometry import *  # noqa: F401,F403
from .maps import *  # noqa: F401,F403
from .maps3d import *  # noqa: F401,F403
from .functions import *  # noqa: F401,F403
from .tensors import *  # noqa: F401,F403
from .analysis import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403
from .plotting import *  # noqa: F401,F403
from .wizard import importWizard  # noqa: F401

# the one flat namespace, the layers in the order they import each other
__all__ = []
__all__ += _geometry.__all__
__all__ += _maps.__all__
__all__ += _maps3d.__all__
__all__ += _functions.__all__
__all__ += _tensors.__all__
__all__ += _analysis.__all__
__all__ += _io.__all__
__all__ += _plotting.__all__
__all__ += ["importWizard", "xvector", "yvector", "zvector"]


# the displays of a notebook and of the IPython terminal, when mtex is imported in one
from .plotting.richdisplay import registerDisplays as _registerDisplays
_registerDisplays()

# the user's saved settings, what MATLAB keeps in mtex_settings.m
from .geometry.settings import loadSettings as _loadSettings
_loadSettings()


# xvector, yvector, zvector are fresh on every access
def __getattr__(name):
  if name in ("xvector", "yvector", "zvector"):
    return getattr(_geometry, name)
  raise AttributeError(name)


if TYPE_CHECKING:
  from .geometry.vector3d import xvector, yvector, zvector
