# measured pole figures written as text tables (MATLAB @PoleFigure/export): one file per
# pole figure, named by the base name and the crystal directions of the entry, a row per
# specimen direction of its polar angle, its azimuth and the intensity, the angles in
# radians unless asked for degrees, as MATLAB writes them; the columns loadPoleFigure reads back,
# radians=True for a table in radians
#
#   export(pf, 'dubna')                 dubna_{101̅0}.txt, ...
#   export(pf, 'dubna', degree=True)    the angles in degrees
from __future__ import annotations

import os

import numpy as np

from ..geometry.arguments import refuseFlags
from ..functions.polefigure import PoleFigure
from ..geometry.units import degree as _degree
from ..geometry.install import install


def exportPoleFigure(pf, fname, *args, degree=False):
  """write every pole figure of a data set to its own three column table; returns the file names;
  MATLAB's PoleFigure/export

    names = export(pf, 'dubna')                # dubna_{101̅0}.txt, ..., the angles in radians
    names = export(pf, 'dubna', degree=True)   # the angles in degrees

  Every crystal direction (or superposition of directions) gets the file
  <fname>_<directions>.txt with three columns, the polar angle and the azimuth of the specimen
  directions and the intensity, the table loadPoleFigure reads back, with radians=True unless
  written with degree=True. The angles are in radians by default, as MATLAB writes them.

  Input
    pf - a PoleFigure
    fname - the base name of the files
  Output
    names - the names of the files written
  Keywords
    degree - the polar angle and the azimuth in degrees instead of radians
  See also
    loadPoleFigure, PoleFigure.load"""
  fname = os.fspath(fname)
  refuseFlags(args, "export")
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
