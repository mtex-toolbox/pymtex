# the quantities of the orientations within a volume (MATLAB EBSD3square gradientX/Y/Z, ../master)
#
#   g = ebsd3.gradientX(), gradientY(), gradientZ()   the left tangent vectors per unit length
#   kappa = ebsd3.curvature()                          the full curvature tensor of every voxel
#   gnd, rho = ebsd3.calcGND(dS)                       the dislocation densities that fit it
#   kam = ebsd3.KAM(order=1, threshold=10*degree, max=False)
#   ebsd3 = ebsd3.fill(extrapolate=False)              the empty voxels from the nearest measured one
#   ebsd, d = ebsd3.spatialProfile(points)             the voxels along a polyline and their distance
#
# The gradient along an axis is MATLAB's gradientDim: the logarithm of the next voxel's
# orientation at this one's, in specimen coordinates, over the step, backwards in the last
# layer, NaN across a grain boundary. A volume measures all three gradients, so its curvature
# is complete: kappa = g1 (x) u1 + g2 (x) u2 + g3 (x) u3 with the unit steps u. MATLAB's
# curvature and calcGND of a volume stop at its two dimensional unit cell, its KAM, fill and
# reduce work on x and y alone, its spatialProfile is not written: these are this port's.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import qconj
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .ebsd3 import EBSD3
from ..geometry.install import install


# the rotations of the gridded volume and a one phase check
def _grid(ebsd):
  g = ebsd.gridify()
  ids = [p for p in np.unique(g.phaseId[~g.isPadding]) if g.CSList[int(p)].isIndexed]
  if len(ids) != 1:
    raise ValueError("the gradients are of one indexed phase: select it first")
  return g, g.CSList[int(ids[0])]


def gradientDim(ebsd, dim):
  """the derivative of the orientation along the array axis dim of a gridded volume per unit length,
  the forward difference, backwards at the last voxel, NaN across a grain boundary"""
  g, cs = _grid(ebsd)
  q = g.rotations.data.astype(float)
  q[g.phaseId != g.CSList.phaseId(cs)] = np.nan
  n = g.shape[dim]
  fwd = np.concatenate([np.arange(1, n), [n - 2]])
  qf = np.take(q, fwd, axis=dim)
  frame = g.frame
  ori, orif = orientation(rotation._fromData(q), cs, frame.stripSym()), orientation(rotation._fromData(qf), cs, frame.stripSym())
  t = np.asarray(orif.log(ori, "left").data, dtype=float).reshape(g.shape + (3,)) / g.lattice.spacing[dim]
  last = [slice(None)] * 3
  last[dim] = n - 1
  t[tuple(last)] *= -1
  if "grainId" in g.prop:
    gid = np.asarray(g.prop["grainId"])
    t[gid != np.take(gid, fwd, axis=dim)] = np.nan
  return vector3d(t, frame=frame)


def gradientX(ebsd, stencil=None):
  """the derivative of the orientation along the first axis of the volume per unit length,
  gradientDim(ebsd, 0)"""
  return gradientDim(ebsd, 0)


def gradientY(ebsd, stencil=None):
  """the derivative of the orientation along the second axis of the volume per unit length,
  gradientDim(ebsd, 1)"""
  return gradientDim(ebsd, 1)


def gradientZ(ebsd, stencil=None):
  """the derivative of the orientation along the third axis of the volume per unit length,
  gradientDim(ebsd, 2)"""
  return gradientDim(ebsd, 2)


def curvature(ebsd, stencil=None):
  """the curvature tensor of every voxel, the gradients along the three lattice directions, in the
  specimen frame and in the inverse scan unit"""
  from ..tensors.tensor import curvatureTensor
  g = ebsd.gridify()
  D = g.lattice.D
  U = D / np.linalg.norm(D, axis=1, keepdims=True)
  M = sum(gradientDim(ebsd, k).data[..., :, None] * U[k][None, :] for k in range(3))
  return curvatureTensor(M, g.frame, unit=f"1/{g.scanUnit}")


def KAM(ebsd, order=1, threshold=None, max=False):
  """the mean misorientation angle to the voxels within order steps along the axes, among those
  of the same grain, or below the threshold where the volume has no grains"""
  from itertools import product
  g, cs = _grid(ebsd)
  p = g.CSList.phaseId(cs)
  q = g.rotations.data.astype(float)
  ok = (g.phaseId == p) & ~g.isPadding & np.all(np.isfinite(q), axis=-1)
  gid = np.asarray(g.prop["grainId"]) if "grainId" in g.prop and threshold is None else None
  thr = 10 * degree if threshold is None else threshold
  G = cs.sym.properGroup().rotations().data.reshape(-1, 4)
  total = np.zeros(g.shape)
  count = np.zeros(g.shape)
  best = np.zeros(g.shape)
  offs = [o for o in product(range(-order, order + 1), repeat=3) if 0 < sum(abs(x) for x in o) <= order]
  from ..geometry.kernels import qmul
  for o in offs:
    src = tuple(slice(max_(0, -k), s - max_(0, k)) for k, s in zip(o, g.shape))
    dst = tuple(slice(max_(0, k), s - max_(0, -k)) for k, s in zip(o, g.shape))
    a, b = q[src], q[dst]
    m = qmul(qconj(a), b)
    c = np.max(np.abs(m.reshape(-1, 4) @ G.T), axis=1).reshape(m.shape[:-1])
    w = 2 * np.arccos(np.clip(c, -1, 1))
    use = ok[src] & ok[dst]
    use &= (gid[src] == gid[dst]) if gid is not None else (w < thr)
    total[src] += np.where(use, w, 0)
    count[src] += use
    best[src] = np.where(use, np.maximum(best[src], w), best[src])
  out = best if max else np.where(count > 0, total / np.maximum(count, 1), np.nan)
  out[~ok] = np.nan
  return _back(ebsd, g, out)


def max_(a, b):
  """the larger of two numbers, the builtin max being shadowed by the keyword of KAM"""
  return a if a > b else b


# a gridded result read back in the order of the volume that was asked
def _back(ebsd, g, a):
  if ebsd.isGridded:
    return a
  _, newId = ebsd.gridify(return_index=True)
  return a.reshape(-1)[newId].reshape(ebsd.shape)


def fill(ebsd, extrapolate=False):
  """the empty voxels, padding and not indexed, given the phase, orientation and properties of
  the nearest measured voxel; without extrapolate only those enclosed by measured voxels, the
  ones the outside of the volume reaches over empty voxels staying empty"""
  from scipy.ndimage import distance_transform_edt, label
  g = ebsd.gridify().copy()
  empty = ~g.isIndexed | g.isPadding
  if not empty.any():
    return g
  _, (i, j, k) = distance_transform_edt(empty, sampling=g.lattice.spacing, return_indices=True)
  sel = empty.copy()
  if not extrapolate:
    lab, _ = label(empty)
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                       lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    sel &= ~np.isin(lab, border[border > 0])
  src = (i[sel], j[sel], k[sel])
  g.phaseId[sel] = g.phaseId[src]
  g.rotations.data[sel] = g.rotations.data[src]
  for v in g.prop.values():
    v[sel] = v[src]
  if g._pad is not None:
    g._pad[sel] = False
  return g


def spatialProfile(ebsd, points):
  """the voxels along a polyline, sampled at the voxel spacing, and their distance along it"""
  P = np.asarray(points.data if isinstance(points, vector3d) else points, dtype=float).reshape(-1, 3)
  step = ebsd.dPos
  pts, dist, s0 = [], [], 0.0
  for a, b in zip(P[:-1], P[1:]):
    L = np.linalg.norm(b - a)
    t = np.arange(0, L, step)
    pts.append(a + t[:, None] * (b - a) / max(L, 1e-300))
    dist.append(s0 + t)
    s0 += L
  pts.append(P[-1:])
  dist.append([s0])
  return ebsd.interp(np.concatenate(pts)), np.concatenate(dist)


for _f in (gradientX, gradientY, gradientZ, curvature, KAM, fill, spatialProfile):
  install(EBSD3, _f)
install(EBSD3, gradientDim)

__all__ = []
