# the analysis layer: the algorithms that iterate or optimise over the cores, from the
# pole figure inversion to the denoising of a map
import importlib as _importlib

__all__ = []
for _name in ("elasticity", "mechanics", "optics", "taylor", "texture", "pf2odf", "embedding", "optimalsample", "cluster", "variational", "dislocations", "denoising", "gbnd"):
  _module = _importlib.import_module(f"{__name__}.{_name}")
  for _public in _module.__all__:
    globals()[_public] = getattr(_module, _public)
  __all__ += [n for n in _module.__all__ if n not in __all__]
