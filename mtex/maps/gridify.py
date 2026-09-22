# the grid of a map: how its lattice is scanned, and the map with its arrays in map shape
# (MATLAB gridify, Julia gridify.jl)
#
#   gridify(ebsd)     the same map with its arrays in map shape, one entry per cell of the scan
#
# How a lattice is scanned: the rows run along the lattice vector nearest the plane's first
# axis, follow each other along the one nearest its second, and a row is shifted against the
# one below it by that step's share along the row, half a cell on a hexagonal lattice and
# nothing on a square one. The array of cells is then the rectangle the scan covered, which
# is what a file calls its grid, and not the parallelogram the lattice indices span. The
# array is (rows, columns) in NumPy's order, so ebsd[i, j] is the pixel in row i and column j.
from __future__ import annotations

import numpy as np

from .lattice import lattice


class scanLayout:

  def __init__(self, l):
    A = l.A
    a1, a2 = A[:, 0], A[:, 1]
    cands = [(np.array([1, 0]), a1), (np.array([0, 1]), a2)]
    if l.shape == "hex":
      cands.append((np.array([1, -1]), a1 - a2))

    # the candidate reaching farthest along an axis of the plane, turned to run forward
    def pick(axis, taken):
      k = max((k for k in range(len(cands)) if k != taken), key=lambda k: abs(cands[k][1][axis]))
      c, v = cands[k]
      return (k, -c, -v) if v[axis] < 0 else (k, c, v)

    ku, cu, u = pick(0, None)
    _, cv, v = pick(1, ku)
    self.B = np.stack([cu, cv], axis=1).astype(np.int64)          # ij = B @ (m, n)
    self.Binv = np.rint(np.linalg.inv(self.B)).astype(np.int64)
    self.s = float(u @ v / (u @ u))
    mn = l.ij @ self.Binv.T
    n0 = int(mn[:, 1].min()) if len(mn) else 0
    self.lo = np.array([0, n0])
    cells = self.scanCell(l.ij)
    c0 = int(cells[:, 0].min()) if len(cells) else 0
    self.lo = np.array([c0, n0])
    self.dims = (int(cells[:, 1].max()) - n0 + 1, int(cells[:, 0].max()) - c0 + 1) if len(cells) else (0, 0)

  # the column and the row of lattice indices (n, 2); a quarter of a cell decides which column
  # a half shifted row starts in, whatever the fit made of the half
  def scanCell(self, ij):
    mn = np.asarray(ij) @ self.Binv.T
    col = mn[:, 0] + np.floor((mn[:, 1] - self.lo[1]) * self.s + 0.25).astype(np.int64)
    return np.stack([col, mn[:, 1]], axis=1)

  # the lattice index of columns and rows
  def scanSite(self, c, r):
    m = c - np.floor((r - self.lo[1]) * self.s + 0.25).astype(np.int64)
    return np.stack([m, r], axis=-1) @ self.B.T

  # the linear cell of every measurement, rows first
  def scanCells(self, ij):
    cells = self.scanCell(ij)
    return (cells[:, 1] - self.lo[1]) * self.dims[1] + (cells[:, 0] - self.lo[0])

  # the lattice index of every cell of the array, in the array's order
  def scanIndex(self):
    r, c = np.meshgrid(np.arange(self.dims[0]) + self.lo[1], np.arange(self.dims[1]) + self.lo[0], indexing="ij")
    return self.scanSite(c.reshape(-1), r.reshape(-1))


# the same map with its arrays in map shape
#
# A complete scan is already in map order, so for one the whole function is a reshape and
# the gridded map shares its arrays with the scattered one. A cell with no measurement is
# filled in at its ideal position, not indexed, with a NaN rotation and zero properties.
def gridify(ebsd):
  from .ebsd import EBSD, _rotations, _vectors
  if ebsd.isGridded:
    return ebsd
  l = ebsd.lattice
  if l is None:
    raise ValueError("the positions are not on a lattice, so the map cannot be gridded")
  lay = scanLayout(l)
  dims = lay.dims
  cells = lay.scanCells(l.ij)
  ij = lay.scanIndex()
  lo = ij.min(axis=0)
  gl = lattice._build(l.plane, l.origin + l.A @ lo, l.A, ij - lo, l.shape)
  n = dims[0] * dims[1]
  if len(cells) == n and np.array_equal(cells, np.arange(n)):
    pos = _vectors(ebsd.pos.data.reshape(dims + (3,)), ebsd.frame)
    rot = _rotations(ebsd.rotations.data.reshape(dims + (4,)))
    prop = {k: v.reshape(dims + v.shape[1:]) for k, v in ebsd.prop.items()}
    return EBSD(pos, rot, ebsd.phaseId.reshape(dims), ebsd.CSList, prop, scanUnit=ebsd.scanUnit, frame=ebsd.frame,
                lattice=gl, meta=ebsd.meta, phaseMap=ebsd.phaseMap, id=ebsd.id.reshape(dims))
  lin = np.full(n, -1, dtype=np.int64)
  # a count per cell, one pass; np.unique would hash and sort the cells
  counts = np.bincount(cells, minlength=n)
  if counts.max() > 1:
    a = np.flatnonzero(counts > 1)[0]
    dup = np.flatnonzero(cells == a)
    raise ValueError(f"measurements {dup[0]} and {dup[1]} sit in one cell of the grid, so gridding would drop one of them")
  lin[cells] = np.arange(len(cells))
  missing = lin < 0
  k = np.where(missing, 0, lin)
  pos = ebsd.pos.data.reshape(-1, 3)[k]
  pos[missing] = gl.position(gl.ij[missing])
  rot = ebsd.rotations.data.reshape(-1, 4)[k]
  rot[missing] = np.nan
  ph = ebsd.phaseId.reshape(-1)[k]
  ph[missing] = 0
  ids = ebsd.id.reshape(-1)[k]
  ids[missing] = ebsd.id.max() + 1 + np.arange(missing.sum())
  prop = {}
  for key, v in ebsd.prop.items():
    src = v.reshape((-1,) + v.shape[ebsd.ndim:])
    out = src[k].copy()
    out[missing] = 0
    prop[key] = out.reshape(dims + src.shape[1:])
  return EBSD(_vectors(pos.reshape(dims + (3,)), ebsd.frame), _rotations(rot.reshape(dims + (4,))), ph.reshape(dims),
              ebsd.CSList, prop, scanUnit=ebsd.scanUnit, frame=ebsd.frame, lattice=gl, meta=ebsd.meta, phaseMap=ebsd.phaseMap, id=ids.reshape(dims))


__all__ = ["gridify"]
