# the poles of a list of orientations, without graphics (MATLAB @orientation/plotPDF,
# plotIPF; docs/plans/plotting.md S6)
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
from ..geometry.kernels import qconj, qrot
from ..geometry.vector3d import vector3d, asVector3d
from ..functions.funops import _antipodalPole


def orbit(xyz, fr, antipodal=False, tol=1e-6):
  """the distinct images of one direction under the group of a frame, an axis and its
  antipode counted once, as an (m, 3) array"""
  x = np.asarray(xyz, dtype=float).reshape(3)
  x = x / np.linalg.norm(x)
  if fr is None or fr.numSym() <= 1:
    return x[None]
  els = fr.sym.rot.flatten().data
  imgs = qrot(els[:, :4], x[None]) * els[:, 4:5]
  d = imgs @ imgs.T
  same = (np.abs(d) if antipodal else d) > 1 - tol
  keep = np.ones(len(imgs), dtype=bool)
  for k in range(len(imgs)):
    if keep[k]:
      keep[k + 1:] &= ~same[k, k + 1:]
  return imgs[keep]


def _oneDirection(h, fr, what):
  v = asVector3d(h)
  if v.size != 1:
    raise ValueError(f"one {what} at a time")
  # the frame of the orientations, a crystal frame compatible with it, or none at all
  if v.frame is not None and fr is not None and v.frame is not fr:
    if not (isinstance(v.frame, crystalFrame) and isinstance(fr, crystalFrame) and v.frame.isCompatible(fr)[0]):
      raise ValueError(f"the {what} lives in another frame than the orientations")
  return v


def poles(o, h, antipodal=False):
  """the poles of orientations: every orientation applied to every distinct equivalent of the
  crystal direction h, and every image of those under the specimen group; MATLAB's
  ori.SS * (ori * symmetrise(h))

    p = poles(ori, Miller(1, 0, 0, cs))            # the poles of the {100} directions
    p = poles(ori, Miller(1, 1, 1, cs), antipodal=True)  # an axis and its antipode counted once

  The poles are what plotPF draws for a list of orientations. An equivalent counts once, and
  an axis and its antipode count once where the pole figure cannot tell h from -h: when
  antipodal is asked for, h is an axis, or -h is an equivalent of h. The result is antipodal
  then.

  Input
    ori - orientation, shaped (...)
    h - Miller or vector3d, one crystal direction
  Output
    p - vector3d in the specimen frame, shaped (..., m) over the m distinct equivalents
  Keywords
    antipodal - count an axis and its antipode once
  See also
    inversePoles, plotPF, symmetrise"""
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
  the distinct equivalents of r under the specimen group; MATLAB's inv(ori) * symmetrise(r)

    h = inversePoles(ori, zvector)                  # the crystal directions along z
    h = inversePoles(ori, zvector, antipodal=True)  # as axes

  They are what plotIPF draws for a list of orientations, which folds them into the
  fundamental sector of the crystal; here they are not folded. An axis and its antipode
  count once when antipodal is asked for, r is an axis, or -r is an equivalent of r.

  Input
    ori - orientation, shaped (...)
    r - vector3d, one specimen direction
  Output
    h - vector3d in the crystal frame, shaped (..., m) over the m distinct equivalents of r
  Keywords
    antipodal - count an axis and its antipode once
  See also
    poles, plotIPF, ipfColorKey"""
  r = _oneDirection(r, o.SS, "specimen direction")
  ax = antipodal or _antipodalPole(vector3d(r.data, frame=o.SS, antipodal=r.antipodal), o.SS)
  eq = orbit(r.data, o.SS, ax)
  q = qconj(o.data.reshape(-1, 4))
  hs = qrot(q[:, None, :], eq[None, :, :])
  return vector3d(hs.reshape(o.shape + (len(eq), 3)), frame=o.CS, antipodal=ax, isNormalized=True)


def dynamicMarkerSize(n, default=10):
  """the size of the markers of a plot of n points; MTEX's dynamic marker size as one rule

    s = dynamicMarkerSize(1000)                    # the size for 1000 markers
    s = dynamicMarkerSize(1000, default=5)         # never larger than 5

  Fifty times the mean spacing of n markers on a hemisphere, sqrt(2 pi / n), never smaller
  than 1.5 and never larger than the default size.

  Input
    n - the number of markers
  Output
    s - the marker size in points
  Keywords
    default - the largest size, that of few markers
  See also
    plotPF, scatter"""
  return default if n == 0 else float(np.clip(50 * np.sqrt(2 * np.pi / n), 1.5, default))


__all__ = ["poles", "inversePoles", "dynamicMarkerSize"]
