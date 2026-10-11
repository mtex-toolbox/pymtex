# dislocations: the densities fitted to the curvature and the geometrically necessary
# dislocation content (MATLAB @EBSD/calcGND, @curvatureTensor/fitDislocationSystems; Julia
# dislocations.jl)
#
#   rho, factor = fitDislocationSystems(kappa, dS), fitDislocationSystems(ebsd.curvature(), ebsd.orientations.reshape(-1, 1) * dS)
#   gnd, rho = ebsd.calcGND(dS), ebsd.calcGND(dS, stencil='full')
#
# The fit of the densities is a linear program, the least total line energy that
# reproduces the curvature, solved by the dense two phase simplex of the maps kernels over
# all pixels in parallel, with SciPy's HiGHS per pixel as the fallback and the oracle; MTEX
# needs the Optimization Toolbox for the same step. The equations are the components of
# the curvature the map actually
# knows, read off the tensor's own NaNs. The densities come out in the units the map and
# the crystal are given in, a density per length of the scan unit times a length of the
# Burgers vector, and the factor MATLAB returns beside them turns them into 1/m^2.
from __future__ import annotations

import numpy as np

from ..geometry.representations import unitRows
from ..geometry.kernels import backend
from ..maps.ebsd import EBSD
from ..geometry.install import install


def systemCurvature(b, l):
  """the curvature tensor of one dislocation system: the dislocation density tensor of a
  system is b (x) l^ and Nye's relation turns it into the curvature"""
  b, l = np.asarray(b, dtype=float), unitRows(np.asarray(l, dtype=float))
  return l[..., :, None] * b[..., None, :] - (np.sum(b * l, axis=-1) / 2)[..., None, None] * np.eye(3)


def fitDensities(M, known, b, l, u):
  """the densities of least total line energy reproducing the known components of the
  curvature: min sum u |rho| under sum rho_j kappa_j = kappa, over the systems and their
  negatives; None where no combination of them reproduces the curvature"""
  from scipy.optimize import linprog
  ns = len(b)
  K = systemCurvature(b, l).reshape(ns, 9)[:, known].T
  A = np.concatenate([K, -K], axis=1)
  res = linprog(np.concatenate([u, u]), A_eq=A, b_eq=np.asarray(M, dtype=float).reshape(9)[known], bounds=(0, None), method="highs")
  if not res.success:
    return None
  return res.x[:ns] - res.x[ns:]


# the length of a unit in metres: the scan units and 'au', the Angstrom of the axes
UNIT_SCALE = {"m": 1.0, "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "nm": 1e-9, "au": 1e-10, "A": 1e-10}


def fitDislocationSystems(kappa, dS):
  """rho, factor = fitDislocationSystems(kappa, dS): the dislocation densities that account for
  the curvature with the least total line energy, one signed density per system, NaN where no
  combination of them reproduces it. The curvature is one tensor or a map of them, the systems
  one list for all or, rotated by the orientations of the map, a (pixels, systems) array; both
  in one frame. The factor turns the densities into 1/m^2, from the units of the curvature
  and of the Burgers vectors"""
  from ..tensors.tensor import tensor
  M = kappa.M if isinstance(kappa, tensor) else np.asarray(kappa, dtype=float)
  single = M.shape == (3, 3)
  K = np.ascontiguousarray(M.reshape(-1, 9))
  ns = dS.shape[-1]
  b = np.ascontiguousarray(dS.b.data.reshape(-1, ns, 3), dtype=float)
  l = np.ascontiguousarray(unitRows(dS.l.data).reshape(-1, ns, 3))
  u = np.ascontiguousarray(np.broadcast_to(dS.u, dS.shape).reshape(-1, ns), dtype=float)
  known = np.flatnonzero(~np.isnan(K[0])) if len(K) else np.zeros(0, dtype=int)
  gnd, rho = np.full(len(K), np.nan), np.full((len(K), ns), np.nan)
  loop = None
  if backend() == "numba":
    from ..maps.kernels import pixelKernel
    loop = pixelKernel("gnd")
  if loop is not None:
    loop(K, np.ascontiguousarray(known, dtype=np.int64), b, l, u, 1e-9, 10000, gnd, rho)
  else:
    for k in range(len(K)):
      kb = k if len(b) > 1 else 0
      if np.any(np.isnan(K[k, known])) or np.any(np.isnan(b[kb])) or np.any(np.isnan(l[kb])):
        continue
      x = fitDensities(K[k], known, b[kb], l[kb], u[kb])
      if x is not None:
        rho[k] = x
  unit = getattr(kappa, "unit", None)
  factor = 1 / UNIT_SCALE["au"] / UNIT_SCALE[unit.replace("1/", "")] if unit else 1.0
  return (rho[0] if single else rho), factor


def calcGND(ebsd, dS, stencil="oneSided"):
  """gnd, rho = calcGND(ebsd, dS): the geometrically necessary dislocation content of every
  pixel after Pantleon (2008), MATLAB's route: the curvature of the map, the systems rotated
  into every pixel, the densities of least total line energy reproducing the curvature. The
  content is the sum of the line energies times the absolute densities, in 1/m^2 for line
  energies without a unit, and the densities one row per pixel"""
  # the curvature of the map, or of the volume, whose curvature is complete
  kappa = ebsd.curvature(stencil=stencil)
  dSRot = ebsd.orientations.reshape(-1, 1) * dS
  rho, factor = fitDislocationSystems(kappa, dSRot)
  gnd = factor * np.sum(np.abs(rho * dSRot.u), axis=1)
  return gnd.reshape(ebsd.shape), rho


install(EBSD, calcGND)

__all__ = ["fitDislocationSystems"]
