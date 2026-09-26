# the per-element properties of a list (MATLAB @dynProp): a dict of name to one value per
# element, displayed as MATLAB displays the struct, one line per name with its size and kind
from __future__ import annotations

import numpy as np

from ..geometry._array import MtexArray


class dynProp(dict):

  def __repr__(self):
    if not self:
      return "no properties"
    w = max(len(k) for k in self)
    return "\n".join(f"{k.rjust(w)}: {_describe(v)}" for k, v in self.items())


# the size and the kind of one property value, as MATLAB's [n×1 double]
def _describe(v):
  if isinstance(v, MtexArray):
    return f"[{'×'.join(map(str, v.shape))} {type(v).__name__}]"
  if isinstance(v, np.ndarray):
    return f"[{'×'.join(map(str, v.shape))} {v.dtype}]"
  if isinstance(v, (list, tuple)):
    return f"[{len(v)} {type(v).__name__}]"
  return repr(v)


__all__ = ["dynProp"]
