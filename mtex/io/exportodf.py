# an ODF written to a file (MATLAB @SO3Fun/export, export_generic, export_mtex, export_VPSC)
#
#   export(odf, 'odf.txt', 'Bunge', resolution=10*degree)   its values on a regular Euler grid
#   export(odf, 'odf.txt', S3G, 'Bunge', 'generic')          its values on the given grid
#   export(odf, 'odf.mtex', 'Bunge', interface='mtex')       its components, human readable
#   export(odf, 'odf-vpsc.txt', 'VPSC', points=5000)         a sample of orientations for VPSC
#
# The generic file has three header lines naming the symmetries, one naming the columns,
# and one row per grid node: the Euler angles in degrees and the value. The MTEX file
# lists the uniform, fibre and radial components with their kernels and weights; a
# harmonic component is noted as not written. The VPSC file is the one of export_VPSC
# over a sample drawn from the function.
from __future__ import annotations

import numpy as np

from ..geometry.orientation import orientation
from ..geometry.orientationregion import eulerRegion
from ..geometry.units import degree
from .exportorientation import EULER_NAMES, export_VPSC

INTERFACES = ("generic", "mtex", "VPSC")


def _convention(args):
  for a in args:
    if isinstance(a, str) and a in ("Bunge", "ZXZ"):
      return "Bunge"
    if isinstance(a, str) and a in ("ABG", "ZYZ", "Matthies"):
      return "ZYZ"
  return "Bunge"


# a regular grid over the Euler box of the frames, phi1 running fastest, the box's open ends
# left out, as MATLAB's regularSO3Grid does
def _regularEuler(CS, SS, resolution):
  m1, m2, m3 = eulerRegion(CS, SS)
  a = np.arange(0, m1 - 1e-8, resolution)
  b = np.arange(0, m2 + 1e-8, resolution)
  c = np.arange(0, m3 - 1e-8, resolution)
  C, B, A = np.meshgrid(c, b, a, indexing="ij")
  return np.stack([A.ravel(), B.ravel(), C.ravel()], axis=1)


def _header(f, odf):
  f.write("% MTEX ODF\n")
  f.write(f'% crystal symmetry: "{odf.CS.char()}"\n')
  f.write(f'% specimen symmetry: "{odf.SS.char()}"\n')


def exportODF_generic(odf, fname, *args, resolution=5 * degree, radians=False, **kwargs):
  from ..geometry.grids.so3grid import SO3Grid
  convention = _convention(args)
  grid = next((a for a in args if isinstance(a, SO3Grid)), None)
  if grid is not None:
    ori = orientation(grid) if not isinstance(grid, orientation) else grid
    e = np.stack([np.asarray(x, dtype=float).ravel() for x in ori.Euler(convention)], axis=1)
  else:
    e = _regularEuler(odf.CS, odf.SS, resolution)
    ori = orientation.byEuler(e[:, 0], e[:, 1], e[:, 2], convention, odf.CS, odf.SS)
  v = np.real(np.asarray(odf.eval(ori))).ravel()
  e = np.mod(e, 2 * np.pi)
  if not radians:
    e = e / degree
  names = EULER_NAMES[convention]
  with open(fname, "w", encoding="utf-8") as f:
    _header(f, odf)
    f.write(f"% {names[0]:<7s} {names[1]:<7s} {names[2]:<7s} value\n")
    for row, val in zip(e, v):
      f.write(f"{row[0]:.5f} {row[1]:.5f} {row[2]:.5f} {val:.5f}\n")


def exportODF_mtex(odf, fname, *args, radians=False, **kwargs):
  from ..functions.handles import SO3FunComposition
  from ..functions.rbf import SO3FunCBF, SO3FunRBF
  from ..functions.so3fun import SO3FunHarmonic
  convention = _convention(args)
  parts = odf.components if isinstance(odf, SO3FunComposition) else [odf]
  names = EULER_NAMES[convention]
  with open(fname, "w", encoding="utf-8") as f:
    _header(f, odf)
    for c in parts:
      if isinstance(c, SO3FunRBF) and c.c0 != 0:
        f.write(f"%\n%% uniform component\n% weight: {c.c0:.5f}\n")
      if isinstance(c, SO3FunHarmonic):
        f.write("%\n%% harmonic component: currently not supported! \n")
      elif isinstance(c, SO3FunCBF):
        for i in range(c.h.size):
          f.write(f"%\n%% fibre component\n% weight: {c.weights[i] * c.psi.mean():.5f}\n% kernel: {c.psi}\n")
          from ..geometry.miller import Miller
          f.write(f"% fibre h: {Miller(c.h[i], c.CS).char()}\n")
          f.write("% fibre r: " + ",".join(f"{x:g}" for x in c.r[i].data.reshape(3)) + "\n")
      elif isinstance(c, SO3FunRBF) and c.center.size:
        f.write(f"%\n%% radial component:\n% kernel: {c.psi}\n")
        e = np.stack([np.asarray(x, dtype=float).ravel() for x in c.center.Euler(convention)], axis=1)
        if not radians:
          e = e / degree
        f.write(f"% {names[0]:<7s} {names[1]:<7s} {names[2]:<7s} weight\n")
        for row, w in zip(e, c.weights * c.psi.mean()):
          f.write(f"{row[0]:.5f} {row[1]:.5f} {row[2]:.5f} {w:.5f}\n")


def exportODF_VPSC(odf, fname, *args, points=10000, **kwargs):
  from ..functions.characteristics import discreteSample
  ori = discreteSample(odf, points)
  export_VPSC(ori, fname, *[a for a in args if isinstance(a, str) and a in EULER_NAMES])


def exportODF(odf, fname, *args, interface=None, **kwargs):
  """export(odf, fname, ...): the function on a grid, as its components, or as a VPSC sample"""
  if interface is None:
    interface = next((a for a in args if isinstance(a, str) and a in INTERFACES), "generic")
  if interface not in INTERFACES:
    raise ValueError(f"unknown export interface {interface!r}; available are " + ", ".join(INTERFACES))
  args = tuple(a for a in args if not (isinstance(a, str) and a in INTERFACES))
  return {"generic": exportODF_generic, "mtex": exportODF_mtex, "VPSC": exportODF_VPSC}[interface](odf, fname, *args, **kwargs)


__all__ = ["exportODF", "exportODF_generic", "exportODF_mtex", "exportODF_VPSC"]
