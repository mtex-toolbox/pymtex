# measured pole figures written as text tables (MATLAB @PoleFigure/export): one file per
# pole figure, named by the base name and the crystal directions of the entry, a row per
# specimen direction of its polar angle, its azimuth and the intensity, the angles in
# radians unless asked for degrees; the columns loadPoleFigure reads back
#
#   export(pf, 'dubna')                 dubna_{101̅0}.txt, ...
#   export(pf, 'dubna', 'degree')       the angles in degrees, export(pf, 'dubna', degree=True) too
from __future__ import annotations

import os

import numpy as np

from ..functions.polefigure import PoleFigure
from ..geometry.units import degree as _degree
from ..geometry.install import install


def exportPoleFigure(pf, fname, *args, degree=False):
  """write every pole figure of a data set to its own three column table; returns the file names"""
  fname = os.fspath(fname)
  if any(isinstance(a, str) and a.lower() == "degree" for a in args):
    degree = True
  s = 1 / _degree if degree else 1.0
  names = []
  for h, r, I in zip(pf.allH, pf.allR, pf.allI):
    theta, rho, _ = r.flatten().polar()
    name = f"{fname}_{h.char()}.txt"
    np.savetxt(name, np.column_stack([theta * s, rho * s, I.ravel()]), fmt="%.8e")
    names.append(name)
  return names


install(PoleFigure, exportPoleFigure, "export")

__all__ = ["exportPoleFigure"]
