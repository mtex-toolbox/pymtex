# the geometry layer: vectors, rotations, orientations, symmetry, frames, grids
import importlib as _importlib

__all__ = []
for _name in ("units", "session", "vector3d", "quaternion", "homochoric", "plottingconvention", "gridlayout", "latticetype", "symmetry", "crystalaxes", "frames", "miller", "orientation", "products", "symreduce", "symmetryops", "sphericalregion", "s2triangulation", "orientationregion", "frametransform", "standardorientations", "tangent", "fibre", "hull", "slipsystem", "crystalshape", "plane", "parent2child", "grids.s1grid", "grids.s2grid", "grids.fibonacci", "grids.healpix", "grids.so3grid", "grids.superfibonacci", "grids.localgrids", "spatialtransform", "funcs"):
  _module = _importlib.import_module(f"{__name__}.{_name}")
  for _public in _module.__all__:
    globals()[_public] = getattr(_module, _public)
  __all__ += _module.__all__

_lazy = {name: mod for name in ("xvector", "yvector", "zvector") for mod in [_importlib.import_module(f"{__name__}.vector3d")]}


# the vector constants are minted on every access, see vector3d.py
def __getattr__(name):
  if name in _lazy:
    return getattr(_lazy[name], name)
  raise AttributeError(name)
