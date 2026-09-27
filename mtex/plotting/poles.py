# the poles of a list of orientations, without graphics (MATLAB @orientation/plotPDF,
# plotIPDF; docs/plans/plotting.md S6)
#
# The pole figure of a list of orientations is the list of its poles: an orientation takes
# the crystal direction h to the specimen direction o * h, and its symmetric equivalents
# take the equivalents of h there, so the pole figure of n orientations is the n x m list
# over the m distinct equivalents of h. The poles are axes where the pole figure cannot
# tell h from -h, and an axis and its antipode count once, since a marker stands for an
# orbit and the budget of a plot is in markers. The inverse pole figure is the mirror
# image: the crystal directions along the equivalents of a specimen direction.
from __future__ import annotations

import numpy as np

from ..geometry.frames import crystalFrame
from ..geometry.kernels import qrot
from ..geometry.vector3d import vector3d
from ..functions.funops import _antipodalPole


def orbit(xyz, fr, antipodal=False, tolerance=1e-6):
  """the distinct images of one direction under the group of a frame, an axis and its
  antipode counted once, as an (m, 3) array"""
  x = np.asarray(xyz, dtype=float).reshape(3)
  x = x / np.linalg.norm(x)
  if fr is None or fr.numSym() <= 1:
    return x[None]
  els = fr.sym.rot.flatten().data
  imgs = qrot(els[:, :4], x[None]) * els[:, 4:5]
  d = imgs @ imgs.T
  same = (np.abs(d) if antipodal else d) > 1 - tolerance
  keep = np.ones(len(imgs), dtype=bool)
  for k in range(len(imgs)):
    if keep[k]:
      keep[k + 1:] &= ~same[k, k + 1:]
  return imgs[keep]


def _oneDirection(h, fr, what):
  v = h if isinstance(h, vector3d) else vector3d(h)
  if v.size != 1:
    raise ValueError(f"one {what} at a time")
  # the frame of the orientations, a crystal frame compatible with it, or none at all
  if v.frame is not None and fr is not None and v.frame is not fr:
    if not (isinstance(v.frame, crystalFrame) and isinstance(fr, crystalFrame) and v.frame.isCompatible(fr)):
      raise ValueError(f"the {what} lives in another frame than the orientations")
  return v


def _conj(q):
  return q * np.array([1.0, -1.0, -1.0, -1.0])


def poles(o, h, antipodal=False):
  """the poles of orientations: every orientation applied to every distinct equivalent of the
  crystal direction h, and every image of those under the specimen group, shaped (..., m)
  in the specimen frame (MATLAB ori.SS * (ori * symmetrise(h)))"""
  h = _oneDirection(h, o.CS, "crystal direction")
  ax = antipodal or _antipodalPole(vector3d(h.data, frame=o.CS, antipodal=h.antipodal), o.CS)
  eq = orbit(h.data, o.CS, ax)
  q = o.data.reshape(-1, 4)
  ps = qrot(q[:, None, :], eq[None, :, :])
  ss = o.SS.sym.rot.flatten().data if o.SS is not None and o.SS.numSym() > 1 else None
  if ss is not None:
    ps = (qrot(ss[None, :, None, :4], ps[:, None, :, :]) * ss[None, :, None, 4:5]).reshape(len(q), -1, 3)
  return vector3d(ps.reshape(o.shape + (ps.shape[-2], 3)), frame=o.SS, antipodal=ax, isNormalized=True)


def inversePoles(o, r, antipodal=False):
  """the inverse poles: the crystal directions that lie along the specimen direction r, over
  the distinct equivalents of r under the specimen group, shaped (..., m) in the crystal frame"""
  r = _oneDirection(r, o.SS, "specimen direction")
  ax = antipodal or _antipodalPole(vector3d(r.data, frame=o.SS, antipodal=r.antipodal), o.SS)
  eq = orbit(r.data, o.SS, ax)
  q = _conj(o.data.reshape(-1, 4))
  hs = qrot(q[:, None, :], eq[None, :, :])
  return vector3d(hs.reshape(o.shape + (len(eq), 3)), frame=o.CS, antipodal=ax, isNormalized=True)


def dynamicMarkerSize(n, default=10):
  """MTEX's dynamic marker size as one rule: fifty times the mean spacing of n markers on a
  hemisphere, never larger than the default size"""
  return default if n == 0 else float(np.clip(50 * np.sqrt(2 * np.pi / n), 1.5, default))


__all__ = ["poles", "inversePoles", "dynamicMarkerSize"]
