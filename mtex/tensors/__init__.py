# the tensor layer: the tensor container over the geometry, its algebra and the physical tensors
import importlib as _importlib

__all__ = []
for _name in ("tensor",):
  _module = _importlib.import_module(f"{__name__}.{_name}")
  for _public in _module.__all__:
    globals()[_public] = getattr(_module, _public)
  __all__ += _module.__all__
