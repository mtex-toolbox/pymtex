# the grid of a map: how its lattice is scanned, and the map with its arrays in map shape
# (MATLAB gridify, Julia gridify.jl)
#
#   gridify(ebsd)                       the same map with its arrays in map shape, one entry per cell of the scan
#   gridify(ebsd, gL)                   stored in the layout gL, a gridLayout or 'rowMajor', 'columnMajor'
#   gridify(ebsd, unitCell=uC)          the map resampled on the square lattice of another unit cell
#   grid, newId = gridify(ebsd, return_index=True)   and the flat index of every measurement in the grid
#
# How a lattice is scanned: the rows run along the lattice vector nearest the plane's first
# axis, follow each other along the one nearest its second, and a row is shifted against the
# one below it by that step's share along the row, half a cell on a hexagonal lattice and
# nothing on a square one. The array of cells is then the rectangle the scan covered, which
# is what a file calls its grid, and not the parallelogram the lattice indices span. The
# array is (rows, columns) in NumPy's order, so ebsd[i, j] is the pixel in row i and column j.
from __future__ import annotations

import warnings

import numpy as np

from ..geometry.gridlayout import gridLayout, layoutIndex
from .lattice import lattice, latticeBasis, mapPlane


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
# padding: filled in at its ideal position, not indexed, with a NaN rotation and zero
# properties, and marked in isPadding. With return_index the flat index of every measurement
# in the grid comes too, MATLAB's second output: grid.pos.flatten()[newId] is ebsd.pos. A
# layout reorders the arrays by a transpose and flips; a hexagonal grid is stored column major
# whatever is asked, as MATLAB stores it.
def gridify(ebsd, layout=None, unitCell=None, return_index=False):
  layout = _layoutOf(layout)
  if unitCell is not None:
    return resample(ebsd, unitCell, layout, return_index)
  grid, newId = _gridify(ebsd)
  if layout is not None and grid.size > 1:
    if grid.lattice is not None and grid.lattice.shape == "hex":
      if layout != gridLayout.columnMajor():
        warnings.warn("a hexagonal grid is always stored column major - ignoring the layout requested", stacklevel=2)
    else:
      d = arrayDirections(grid)
      lin = layoutIndex(layout, d, grid.shape)
      grid = _relayout(grid, lin)
      inv = np.empty(lin.size, dtype=np.int64)
      inv[lin.reshape(-1)] = np.arange(lin.size)
      newId = inv[newId]
  return (grid, newId) if return_index else grid


def _layoutOf(layout):
  if layout is None or isinstance(layout, gridLayout):
    return layout
  if layout in ("rowMajor", "columnMajor"):
    return getattr(gridLayout, layout)()
  raise ValueError(f"a layout is a gridLayout, 'rowMajor' or 'columnMajor', not {layout!r}")


# the directions the first and the second index of a gridded map run along, the first taken
# perpendicular to the second, as a hexagonal row shifts against the one below it
def arrayDirections(grid):
  P = grid.pos.data
  n1, n2 = P.shape[0], P.shape[1]
  d2 = P[0, 1] - P[0, 0] if n2 > 1 else None
  d1 = P[1, 0] - P[0, 0] if n1 > 1 else None
  N = grid.N.data.reshape(3)
  if d2 is None:
    d2 = np.cross(d1, N)
  d2 = d2 / np.linalg.norm(d2)
  if d1 is None:
    d1 = np.cross(N, d2)
  d1 = d1 - (d1 @ d2) * d2
  return d1 / np.linalg.norm(d1), d2


# a gridded map with its arrays reordered: new[r, c] = old.flat[lin[r, c]]
def _relayout(grid, lin):
  from .ebsd import EBSD, _rotations, _vectors
  k, shape = lin.reshape(-1), lin.shape
  flat = lambda a, m: a.reshape((-1,) + a.shape[grid.ndim:])[k].reshape(shape + a.shape[grid.ndim:])
  l = grid.lattice
  gl = None if l is None else lattice._build(l.plane, l.origin, l.A, l.ij[k], l.shape)
  prop = {key: flat(v, grid.ndim) for key, v in grid.prop.items()}
  return EBSD(_vectors(flat(grid.pos.data, 2), grid.frame), _rotations(flat(grid.rotations.data, 2)), flat(grid.phaseId, 2),
              grid.CSList, prop, scanUnit=grid.scanUnit, frame=grid.frame, lattice=gl, meta=grid.meta, phaseMap=grid.phaseMap,
              id=flat(grid.id, 2), isPadding=flat(grid.isPadding, 2))


def _gridify(ebsd):
  from .ebsd import EBSD, _rotations, _vectors
  if ebsd.isGridded:
    return ebsd, np.arange(ebsd.size)
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
                lattice=gl, meta=ebsd.meta, phaseMap=ebsd.phaseMap, id=ebsd.id.reshape(dims), isPadding=ebsd.isPadding.reshape(dims)), cells
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
              ebsd.CSList, prop, scanUnit=ebsd.scanUnit, frame=ebsd.frame, lattice=gl, meta=ebsd.meta, phaseMap=ebsd.phaseMap, id=ids.reshape(dims),
              isPadding=missing.reshape(dims)), cells


# the translations of a square cell as MATLAB's latticeBasis orders them: twice the vectors to
# the edge midpoints, turned into the upper half plane, the first of least angle to x and the
# second the first one across it
def _cellBasis(C):
  V = C - C.mean(axis=0)
  trans = V + np.roll(V, -1, axis=0)
  tol = 1e-12 * np.linalg.norm(trans, axis=1).max()
  flip = (trans[:, 1] < -tol) | ((np.abs(trans[:, 1]) <= tol) & (trans[:, 0] < 0))
  trans[flip] = -trans[flip]
  a1 = trans[np.argmin(np.arctan2(trans[:, 1], trans[:, 0]))]
  across = np.abs(trans @ a1) / (np.linalg.norm(trans, axis=1) * np.linalg.norm(a1) + np.finfo(float).eps) < 0.5
  return np.stack([a1, trans[np.flatnonzero(across)[0]]], axis=1)


# the map read on the square lattice of another unit cell (MATLAB gridify(ebsd, 'unitCell',
# uC)): the lattice is laid in the plane of the map over its extent, from the corner of least
# x and y, and every site takes the nearest measurement - nothing is interpolated - where one
# lies within the farthest corner of the map's own cell, and is not indexed beyond. The array
# runs along the cell's translations as MATLAB orders them and is laid out as near the layout
# as a transpose and flips reach, column major by default. newId is the site every
# measurement falls in
def resample(ebsd, unitCell, layout=None, return_index=False):
  from scipy.spatial import cKDTree
  from .ebsd import EBSD, _rotations, _vectors
  _, stencil, _ = latticeBasis(unitCell)
  if len(stencil) != 4:
    raise ValueError("a map is resampled on a square cell; a hexagonal one is not ported")
  keep = ~ebsd.isPadding.reshape(-1)
  src = np.flatnonzero(keep)
  xyz = ebsd.pos.data.reshape(-1, 3)[src]
  pl = ebsd.lattice.plane if ebsd.lattice is not None else mapPlane(xyz)
  P = pl.to2d(xyz)
  T = P.dtype
  corners = np.asarray(unitCell.data if hasattr(unitCell, "data") else unitCell, dtype=float).reshape(-1, 3)
  A = _cellBasis(pl.to2d(corners + pl.o)).astype(T)
  x0, x1, y0, y1 = P[:, 0].min(), P[:, 0].max(), P[:, 1].min(), P[:, 1].max()
  c0 = np.array([x0, y0], dtype=T)
  Ainv = np.linalg.inv(A)
  IJ = (np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]], dtype=T) - c0) @ Ainv.T
  # the sites that cover the extent; one within a thousandth of a cell of it adds nothing
  lo, hi = np.floor(IJ.min(axis=0) + 1e-3).astype(np.int64), np.ceil(IJ.max(axis=0) - 1e-3).astype(np.int64)
  dims = (int(hi[0] - lo[0] + 1), int(hi[1] - lo[1] + 1))
  ii, jj = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1), indexing="ij")
  ij = np.stack([ii.reshape(-1), jj.reshape(-1)], axis=1)
  Q = c0 + ij @ A.T
  # the nearest measurement of every site, within the reach of the map's own cell; of equally
  # near measurements the first of the list, as MATLAB's knnsearch takes it
  cell = ebsd.lattice.unitCell if ebsd.lattice is not None else None
  reach = np.inf if cell is None else float(np.linalg.norm(cell, axis=1).max()) * (1 + 1e-10)
  d, near = cKDTree(P).query(Q, k=min(4, len(P)))
  d, near = d.reshape(len(Q), -1), near.reshape(len(Q), -1)
  tie = d <= d[:, :1] * (1 + 1e-9) + np.finfo(T).tiny
  near = np.where(tie, near, len(P)).min(axis=1)
  hit = d[:, 0] <= reach
  k = src[near[hit]]
  n = len(Q)
  rot = np.full((n, 4), np.nan, dtype=ebsd.rotations.data.dtype)
  rot[hit] = ebsd.rotations.data.reshape(-1, 4)[k]
  ph = np.zeros(n, dtype=ebsd.phaseId.dtype)
  ph[hit] = ebsd.phaseId.reshape(-1)[k]
  prop = {}
  for key, v in ebsd.prop.items():
    flat = v.reshape((-1,) + v.shape[ebsd.ndim:])
    out = np.full((n,) + flat.shape[1:], np.nan, dtype=np.result_type(flat.dtype, np.float32))
    out[hit] = flat[k]
    prop[key] = out.reshape(dims + flat.shape[1:])
  gl = lattice._build(pl, c0 + A @ lo, A, ij - lo, "square")
  grid = EBSD(_vectors(pl.from2d(Q).astype(T).reshape(dims + (3,)), ebsd.frame), _rotations(rot.reshape(dims + (4,))),
              ph.reshape(dims), ebsd.CSList, prop, scanUnit=ebsd.scanUnit, frame=ebsd.frame, lattice=gl, meta=ebsd.meta,
              phaseMap=ebsd.phaseMap)
  lin = layoutIndex(layout or gridLayout.columnMajor(), (pl.from2d(A[:, 0]) - pl.o, pl.from2d(A[:, 1]) - pl.o), dims, nearest=True)
  grid = _relayout(grid, lin)
  if not return_index:
    return grid
  # the site of every measurement, rounded on the new lattice, and where the layout put it
  mine = np.clip(np.rint((P - c0) @ Ainv.T).astype(np.int64), lo, hi) - lo
  inv = np.empty(lin.size, dtype=np.int64)
  inv[lin.reshape(-1)] = np.arange(lin.size)
  newId = np.full(ebsd.size, -1, dtype=np.int64)
  newId[src] = inv[mine[:, 0] * dims[1] + mine[:, 1]]
  return grid, newId


__all__ = ["gridify"]
