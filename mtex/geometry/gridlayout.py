# the layout of an array of grid cells: along which direction its first and its second index
# run (MATLAB gridLayout)
#
#   gridLayout(row, col)            the first index along row, the second along col, perpendicular
#   gridLayout.columnMajor()        the first index along y, the second along x: MATLAB's images
#   gridLayout.rowMajor()           the first index along x, the second along y
#   gridLayout.assumedFor(fr)       the layout an array drawn in fr's convention is stored in
#   lin = layoutIndex(gL, (d1, d2), shape, nearest=False)
#
# A layout says which way round an array is stored, which is a matter of the array and not of
# how it is plotted. An array is taken from one layout to another by a transpose and flips
# alone, and layoutIndex gives the index that does it; nearest=True takes the layout a
# permutation reaches closest where the two differ by more.
from __future__ import annotations

import numpy as np

from .vector3d import vector3d

X, Y = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])


def _unit(v):
  v = np.asarray(v.data if isinstance(v, vector3d) else v, dtype=float).reshape(3)
  return v / np.linalg.norm(v)


class gridLayout:

  def __init__(self, row=None, col=None, name=""):
    if row is None and col is None:
      row, col = Y, X
    r, c = _unit(row), _unit(col)
    if abs(r @ c) > 1e-6:
      raise ValueError(f"the row and the column direction of a grid are perpendicular, but the ones given are "
                       f"{np.degrees(np.arccos(np.clip(r @ c, -1, 1))):.1f}° apart")
    self.row, self.col = vector3d(r, isNormalized=True), vector3d(c, isNormalized=True)
    self.name = name

  @staticmethod
  def columnMajor():
    return gridLayout(Y, X, name="columnMajor")

  @staticmethod
  def rowMajor():
    return gridLayout(X, Y, name="rowMajor")

  # the layout of an array drawn with row 0 at the top and the columns to the right, read
  # against the convention of the frame: what the relation was before layouts were stated,
  # and named so that the assumption shows wherever the layout is displayed
  @staticmethod
  def assumedFor(fr):
    from .plottingconvention import plottingConvention
    pC = getattr(fr, "how2plot", None)
    pC = plottingConvention.default if pC is None else pC
    rot = pC.rot * plottingConvention.ij.rot.inv()
    return gridLayout(rot * vector3d(Y), rot * vector3d(X), name="assumed from plot")

  @property
  def basis(self):
    return vector3d(np.stack([self.row.data, self.col.data, np.cross(self.row.data, self.col.data)]), isNormalized=True)

  # two layouts are one when their directions are
  def __eq__(self, other):
    return isinstance(other, gridLayout) and np.allclose(self.row.data, other.row.data) and np.allclose(self.col.data, other.col.data)

  def __repr__(self):
    fmt = lambda v: "(" + ", ".join(f"{x:g}" for x in np.round(v.data.reshape(3), 4) + 0.0) + ")"
    return f"gridLayout {self.name}\n  row: {fmt(self.row)}\n  col: {fmt(self.col)}".replace("gridLayout \n", "gridLayout\n")


# the index that lays out an array whose first and second index run along d = (d1, d2) as gL
# asks: new[r, c] = old.flat[lin[r, c]]. It transposes where the axes are crossed and flips an
# axis that runs backwards; a tie, a grid turned by 45 degrees, keeps the order unless the
# column of gL is nearer y than x
def layoutIndex(gL, d, shape, nearest=False, return_transpose=False):
  d1, d2 = (_unit(v) for v in d)
  t1, t2 = _unit(gL.row), _unit(gL.col)
  Q = np.array([[t1 @ d1, t1 @ d2], [t2 @ d1, t2 @ d2]])
  off, dia = abs(Q[0, 1]) + abs(Q[1, 0]), abs(Q[0, 0]) + abs(Q[1, 1])
  transpose = off > dia if abs(off - dia) > 1e-12 else abs(t2 @ Y) > abs(t2 @ X)
  if not nearest:
    onAxis = np.abs(Q) > 1 - 1e-6
    if onAxis.sum() != 2 or not onAxis.any(axis=0).all() or not onAxis.any(axis=1).all():
      raise ValueError("an array is laid out differently only by a transpose and flips, and the layout it is in and the "
                       "one asked for differ by more; resample it, or take the nearest layout with nearest=True")
  lin = np.arange(int(np.prod(shape))).reshape(shape)
  if transpose:
    Q = Q[:, ::-1]
    lin = lin.T
  if Q[0, 0] < 0:
    lin = lin[::-1, :]
  if Q[1, 1] < 0:
    lin = lin[:, ::-1]
  lin = np.ascontiguousarray(lin)
  return (lin, bool(transpose)) if return_transpose else lin


__all__ = ["gridLayout", "layoutIndex"]
