# the fit of a sum of radial kernels on SO(3) to values at orientations (MATLAB
# SO3FunRBF.interpolate with its private spatialMethod, createSummationMatrix and mlsq)
#
#   SO3F = SO3FunRBF.interpolate(ori, values)                        kernels on a 5 degree grid, LSQR
#   SO3F = SO3FunRBF.interpolate(ori, values, density=True)          nonnegative weights of mean one
#   SO3F = SO3FunRBF.interpolate(ori, values, halfwidth=10 * degree, resolution=5 * degree)
#   SO3F = SO3FunRBF.interpolate(ori, values, kernel=psi, SO3Grid=S3G)
#   SO3F = SO3FunRBF.interpolate(ori, values, exact=True)            a kernel at every node
#   SO3F, iterations = SO3FunRBF.interpolate(ori, values, withIterations=True)
#
# The matrix holds the value of every centre's kernel at every node, the mean over the
# symmetric equivalents as SO3FunRBF evaluates it; within 3.5 halfwidths only the nearest
# equivalent is near enough to count, so a grid of centres gives a sparse matrix through
# the radius search. Unconstrained weights come from LSQR, weights of one sign from MATLAB's
# mlsq, a projected gradient that keeps them positive.
from __future__ import annotations

import numpy as np

from ..geometry.arguments import refuseFlags
from ..geometry.orientation import orientation
from ..geometry.symreduce import reductionGroups
from ..geometry.units import degree
from .kernelfunctions import SO3DeLaValleePoussinKernel


def summationMatrix(psi, centres, nodes, exact=False, epsilon=None):
  """the values of the kernels at the centres at the nodes, rows the nodes, columns the centres"""
  from scipy.sparse import csr_matrix
  L, R = reductionGroups(nodes)
  G = len(L) * len(R)
  eps = np.pi if exact else min(np.pi, 3.5 * psi.halfwidth() if epsilon is None else epsilon)
  if not exact and isinstance(centres, orientation) and eps < np.pi:
    from ..geometry.grids.orientationtree import gridRadius
    _, C = gridRadius(centres, nodes.flatten(), eps)
    return csr_matrix((np.real(psi.eval(C.data)) / G, C.indices, C.indptr), shape=C.shape)
  # every pair: the kernel at all equivalents of the centres, their mean
  from ..geometry.kernels import _qmulNumpy
  q = nodes.data.reshape(-1, 4)
  c = centres.data.reshape(-1, 4)
  a, b = np.asarray(L).reshape(-1, 4), np.asarray(R).reshape(-1, 4)
  eq = _qmulNumpy(_qmulNumpy(a[:, None, None, :], c[None, :, None, :]), b[None, None, :, :]).reshape(-1, 4)
  t = np.minimum(np.abs(q @ eq.T), 1.0)
  return np.real(psi.eval(t)).reshape(len(q), len(a), len(c), len(b)).mean(axis=(1, 3))


def mlsq(A, y, c0, maxIter=100, tol=1e-3):
  """MATLAB's mlsq: least squares over weights of the sign of the start, the steepest descent
  direction scaled by the weights so that none crosses zero"""
  chat = np.real(np.asarray(c0, dtype=float))
  r = A @ chat - y
  oldErr = np.inf
  k = 0
  for k in range(1, maxIter + 1):
    gradient = A.T @ r
    lam = np.real(gradient - gradient @ chat)
    ctilde = lam * chat
    taumax = 1 / np.max(np.abs(lam))
    rtilde = A @ ctilde
    taub = np.real((rtilde @ r) / (rtilde @ rtilde))
    taub = np.sign(taub) * max(min(abs(taub), taumax), 0)
    r = r - taub * rtilde
    newErr = np.linalg.norm(r)
    if oldErr - newErr < tol:
      break
    chat = np.real(chat - taub * ctilde)
    oldErr = newErr
  return chat, k


def voronoiVolumes(nodes, resolution=2.5 * degree):
  """the volumes of the Voronoi cells of orientations, estimated by the share of a fine grid
  nearest to each of them, normalised to one"""
  from ..geometry.grids.orientationtree import OrientationTree
  from ..geometry.grids.so3grid import equispacedSO3Grid
  fine = equispacedSO3Grid(nodes.CS, nodes.SS, resolution=resolution).flatten()
  ind, _ = OrientationTree(nodes.flatten()).nearest(fine)
  w = np.bincount(np.asarray(ind).ravel(), minlength=nodes.size).astype(float)
  return w / w.sum()


# the weights of one column of values
def _spatial(M, y, density, mean, maxIter, tol, nodes):
  eps = np.max(y) * 1e-3
  signed = np.all(y > -eps) or np.all(y < eps)
  method = "lsqr"
  if density:
    method = "mlsq"
  elif mean is not None and signed:
    method = "mlsq"
  elif M.shape[0] < 1e4 and signed:
    w = voronoiVolumes(nodes)
    mean = float(np.sum(w * y))
    method = "mlsq"
  if method == "lsqr":
    from .lsqr import lsqr
    x, _, _, it = lsqr(lambda c: M @ c, lambda r: M.T @ r, y, M.shape[1], tol=tol, maxIter=30 if maxIter is None else maxIter)
    return x, it, mean
  c0 = M.T @ y
  if density:
    c0 = np.where(c0 < eps, eps, c0)
  else:
    small = np.abs(c0) <= eps
    c0 = np.where(small, np.sign(c0) * eps, c0)
  c0 = (1.0 if mean is None else mean) * c0 / np.sum(c0)
  chat, it = mlsq(M, y, c0, 100 if maxIter is None else maxIter, tol)
  return chat, it, mean


def interpolateRBF(nodes, values, *, kernel=None, halfwidth=None, resolution=None, SO3Grid=None, density=False,
                   mean=None, exact=False, tol=1e-3, maxIter=None, withIterations=False):
  """the sum of radial kernels through values at orientations, MATLAB's SO3FunRBF.interpolate"""
  from ..geometry.grids.so3grid import equispacedSO3Grid
  from .rbf import SO3FunRBF, unimodalODF
  nodes = (nodes if isinstance(nodes, orientation) else orientation(nodes)).flatten()
  hw = 5 * degree if resolution is None else resolution
  hw = hw if halfwidth is None else halfwidth
  psi = SO3DeLaValleePoussinKernel(halfwidth=hw) if kernel is None else kernel
  res = max(0.75 * degree, psi.halfwidth()) if resolution is None else resolution
  grid = nodes if exact else (equispacedSO3Grid(nodes.CS, nodes.SS, resolution=res) if SO3Grid is None else SO3Grid)
  centres = grid.flatten()
  M = summationMatrix(psi, grid, nodes, exact=exact)
  y = np.real(np.asarray(values, dtype=float)).reshape(nodes.size, -1)
  cols, its = [], []
  for k in range(y.shape[1]):
    chat, it, _ = _spatial(M, y[:, k], density, mean, maxIter, tol, nodes)
    cols.append(chat)
    its.append(it)
  w = np.stack(cols, axis=-1).reshape((centres.size,) + np.asarray(values).shape[1:])
  centres = orientation(centres.data, centres.CS, centres.SS)
  # a density of one column is MATLAB's unimodalODF of the weights, which drops the centres
  # of negligible weight and scales to mean one
  if density and w.ndim == 1:
    f = unimodalODF(centres, psi, weights=w, thinning=not (exact))
  else:
    f = SO3FunRBF(centres, psi, w if w.ndim > 1 else w.ravel())
    # the mean asked for, one for a density, by scaling the weights
    target = 1.0 if density else mean
    if target is not None:
      f.weights = target * f.weights / (np.sum(f.weights, axis=0) * psi.mean())
  if withIterations:
    return f, (its[0] if len(its) == 1 else np.array(its))
  return f


def interpSO3(nodes, values=None, *flags, regularisation=None, SobolevIndex=None, **kwargs):
  """MATLAB's rotation/interp: the harmonic fit ('harmonic'), the Bingham distribution of the
  weighted orientations ('bingham'), or the sum of kernels, density=True and exact=True as there"""
  refuseFlags(flags, "interp", names=("harmonic", "bingham"))
  if "harmonic" in flags:
    from .density import interpolate
    if regularisation is not None:
      kwargs["lambda_"] = regularisation
    if SobolevIndex is not None:
      kwargs["s"] = SobolevIndex
    return interpolate(nodes, values, **kwargs)
  if "bingham" in flags:
    from .bingham import calcBinghamODF
    return calcBinghamODF(nodes, weights=values)
  return interpolateRBF(nodes, values, density=kwargs.pop("density", False), exact=kwargs.pop("exact", False), **kwargs)


__all__ = ["interpolateRBF", "summationMatrix", "mlsq", "interpSO3"]
