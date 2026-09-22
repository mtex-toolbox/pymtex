# dislocations: the densities fitted to the curvature and the geometrically necessary
# dislocation content (MATLAB @EBSD/calcGND, @curvatureTensor/fitDislocationSystems; Julia
# dislocations.jl)
#
#   rho = fitDislocationSystems(kappa, dS)
#   gnd, rho = ebsd.calcGND(dS), ebsd.calcGND(dS, stencil='full')
#
# The fit of the densities is a linear program, the least total line energy that
# reproduces the curvature, solved by the dense two phase simplex of the maps kernels over
# all pixels in parallel, with SciPy's HiGHS per pixel as the fallback and the oracle; MTEX
# needs the Optimization Toolbox for the same step. The equations are the components of
# the curvature the map actually
# knows, read off the tensor's own NaNs. The densities come out in the units the map and
# the crystal are given in: a density per length of the scan unit times a length of the
# Burgers vector; to read them as 1/m^2, divide by the product of the two lengths in metres.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import backend, qrot
from ..maps.ebsd import EBSD
from ..maps.intragranular import curvature


def _unit(a):
  n = np.linalg.norm(a, axis=-1, keepdims=True)
  return a / np.where(n == 0, 1, n)


# the curvature tensor of one dislocation system: the dislocation density tensor of a
# system is b (x) l^ and Nye's relation turns it into the curvature
def systemCurvature(b, l):
  b, l = np.asarray(b, dtype=float), _unit(np.asarray(l, dtype=float))
  return l[..., :, None] * b[..., None, :] - (np.sum(b * l, axis=-1) / 2)[..., None, None] * np.eye(3)


# the densities of least total line energy reproducing the known components of the
# curvature: min sum u |rho| under sum rho_j kappa_j = kappa, over the systems and their
# negatives; None where no combination of them reproduces the curvature
def fitDensities(M, known, b, l, u):
  from scipy.optimize import linprog
  ns = len(b)
  K = systemCurvature(b, l).reshape(ns, 9)[:, known].T
  A = np.concatenate([K, -K], axis=1)
  res = linprog(np.concatenate([u, u]), A_eq=A, b_eq=np.asarray(M, dtype=float).reshape(9)[known], bounds=(0, None), method="highs")
  if not res.success:
    return None
  return res.x[:ns] - res.x[ns:]


# the dislocation densities that account for a curvature tensor with the least total line
# energy, one signed density per system; the tensor and the systems are in one frame
def fitDislocationSystems(kappa, dS):
  M = np.asarray(kappa, dtype=float).reshape(9)
  known = np.flatnonzero(~np.isnan(M))
  return fitDensities(M, known, dS.b.data, dS.l.data, dS.u)


# the geometrically necessary dislocation content of every pixel, after Pantleon (2008):
# the curvature of the map is explained by the dislocation systems with the least total
# line energy, and the energy of that solution is the GND. Returns the energy per pixel and
# the signed density of every system
def calcGND(ebsd, dS, stencil="oneSided"):
  kappa = np.ascontiguousarray(curvature(ebsd, stencil=stencil).reshape(-1, 9))
  n, ns = len(kappa), len(dS)
  b0, l0, u = np.ascontiguousarray(dS.b.data, dtype=float), np.ascontiguousarray(_unit(dS.l.data)), np.ascontiguousarray(dS.u, dtype=float)
  rot = np.ascontiguousarray(ebsd.rotations.data, dtype=float).reshape(-1, 4)
  ind = np.ascontiguousarray(ebsd.isIndexed.reshape(-1))
  known = np.flatnonzero(~np.isnan(kappa[0])) if n else np.zeros(0, dtype=int)
  gnd, rho = np.full(n, np.nan), np.full((n, ns), np.nan)
  if backend() == "numba":
    from ..maps.kernels import pixelKernel
    loop = pixelKernel("gnd")
    if loop is not None:
      loop(rot, ind, kappa, np.ascontiguousarray(known, dtype=np.int64), b0, l0, u, 1e-9, 10000, gnd, rho)
      return gnd.reshape(ebsd.shape), rho
  for k in np.flatnonzero(ind):
    if np.any(np.isnan(kappa[k, known])):
      continue
    # the systems in crystal coordinates, rotated into the specimen at this pixel
    b, l = qrot(rot[k], b0), qrot(rot[k], l0)
    x = fitDensities(kappa[k], known, b, l, u)
    if x is None:
      continue
    rho[k] = x
    gnd[k] = np.sum(np.abs(x) * u)
  return gnd.reshape(ebsd.shape), rho


EBSD.calcGND = calcGND

__all__ = ["fitDislocationSystems"]
