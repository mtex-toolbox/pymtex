# an ODF read from a file (MATLAB SO3Fun.load, loadODF_generic, loadODF_VPSC)
#
#   odf = SO3Fun.load('odf.txt', CS=cs, SS=ss, 'Bunge', 'interp', columnNames=['Euler 1', 'Euler 2', 'Euler 3', 'weights'])
#   odf = SO3Fun.load('odf.txt', CS=cs, 'density', halfwidth=10*degree, ...)
#   odf = SO3Fun.load('TEX_PH1.OUT', cs)                                  a VPSC texture file
#
# A generic table holds Euler angles and one more column. 'interp' reads that column as
# the values of the function at the orientations and fits a function through them;
# 'density' reads it as the masses of kernels centred there; without either flag varying
# values mean 'interp', equal ones 'density'. The angles are degrees when one exceeds 15,
# as MATLAB's reader decides, unless 'radians' says so. A VPSC file holds blocks of
# orientations with weights, one density per block.
from __future__ import annotations

import os
import re
from types import SimpleNamespace

import numpy as np

from ..geometry.frames import crystalFrame, specimenFrame
from ..geometry.orientation import orientation
from ..geometry.units import degree
from .exportorientation import EULER_NAMES

WEIGHT_NAMES = ("weights", "weight", "intensity", "value", "values")
ANGLE_NAMES = {"euler 1": 0, "euler 2": 1, "euler 3": 2, "phi1": 0, "phi": 1, "phi2": 2, "alpha": 0, "beta": 1, "gamma": 2}


def _numericBlock(fname):
  rows = []
  with open(fname, encoding="utf-8", errors="replace") as f:
    for line in f:
      s = line.strip()
      if not s or s[0] in "%#":
        continue
      try:
        rows.append([float(x) for x in re.split(r"[\s,;]+", s)])
      except ValueError:
        continue
  if not rows:
    raise ValueError(f"no numeric data in {fname}")
  n = max(len(r) for r in rows)
  return np.array([r for r in rows if len(r) == n], dtype=float)


def _frames(args, CS, SS):
  for a in args:
    if isinstance(a, crystalFrame) and CS is None:
      CS = a
    elif isinstance(a, specimenFrame) and SS is None:
      SS = a
  return CS, specimenFrame.default if SS is None else SS


def loadODF_generic(fname, *args, CS=None, SS=None, columnNames=None, columns=None, radians=False, halfwidth=None, resolution=None,
                    **kwargs):
  CS, SS = _frames(args, CS, SS)
  if CS is None:
    raise ValueError("no crystal symmetry specified: SO3Fun.load(fname, CS=cs, ...)")
  d = _numericBlock(fname)
  if columnNames is None:
    names = ["Euler 1", "Euler 2", "Euler 3", "weights"][:d.shape[1]]
  else:
    names = list(columnNames)
  cols = list(range(len(names))) if columns is None else [int(c) - 1 for c in columns]
  angleCols, weightCol = [None] * 3, None
  for name, c in zip(names, cols):
    key = name.strip().lower()
    if key in ANGLE_NAMES:
      angleCols[ANGLE_NAMES[key]] = c
    elif key in WEIGHT_NAMES:
      weightCol = c
  if any(c is None for c in angleCols):
    raise ValueError("the column names must name the three Euler angles")
  e = d[:, angleCols]
  if not radians and np.max(np.abs(e)) > 15:
    e = e * degree
  flags = [a for a in args if isinstance(a, str)]
  convention = next((a for a in flags if a in EULER_NAMES), "Bunge")
  ori = orientation.byEuler(e[:, 0], e[:, 1], e[:, 2], convention, CS, SS)
  w = np.ones(len(e)) if weightCol is None else d[:, weightCol]
  method = "interp" if "interp" in flags else "density" if "density" in flags else "interp" if len(np.unique(w)) > 1 else "density"
  if method == "density":
    from ..functions.density import calcDensity
    return calcDensity(ori, weights=w, halfwidth=halfwidth, **({} if resolution is None else {"resolution": resolution}))
  print("  Interpolating the ODF. This might take some time...")
  from ..functions.density import interpolate
  return interpolate(ori, w, **{k: v for k, v in kwargs.items() if k in ("bandwidth", "lambda_", "maxit", "tol")})


VPSC_CONVENTIONS = {"B": "Bunge", "K": "Kocks", "R": "Roe"}


def loadODF_VPSC(fname, *args, CS=None, **kwargs):
  from ..functions.density import calcDensity
  CS, SS = _frames(args, CS, None)
  CS = crystalFrame("432") if CS is None else CS
  with open(fname, encoding="utf-8", errors="replace") as f:
    lines = f.read().split("\n")
  heads = [(i, m) for i, m in ((i, re.match(r"^\s*([BbKkRr])\s+(\d+)\s*$", l)) for i, l in enumerate(lines)) if m]
  if not heads or heads[0][0] != 3:
    raise ValueError("the file is not a VPSC texture file: its fourth line names the convention and the count")
  # the leading numbers of a header line, NaN for what a weight file leaves out
  def numbers(line, k):
    v = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line.split("<--")[0])][:k]
    return np.array(v + [np.nan] * (k - len(v)))

  odfs = []
  for i, m in heads:
    n = int(m.group(2))
    convention = VPSC_CONVENTIONS[m.group(1).upper()]
    block = np.array([[float(x) for x in l.split()] for l in lines[i + 1:i + 1 + n]], dtype=float)
    ori = orientation.byEuler(block[:, 0] * degree, block[:, 1] * degree, block[:, 2] * degree, convention, CS, SS)
    odf = calcDensity(ori, weights=block[:, 3], **kwargs)
    # the strain, the phase ellipsoid, the orientations and the further columns, MATLAB's odf.opt
    strain = re.search(r"STRAIN\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", lines[i - 3])
    odf.opt = SimpleNamespace(strain=float(strain.group(1)) if strain else np.nan,
                              strainEllipsoid=numbers(lines[i - 2], 3), strainEllipsoidAngles=numbers(lines[i - 1], 3),
                              orientations=ori, data=block[:, 4:])
    odfs.append(odf)
  return odfs[0] if len(odfs) == 1 else odfs


def loadODF(fname, *args, interface=None, **kwargs):
  """the ODF a file holds, the interface told by its content: a VPSC texture file or a generic table"""
  fname = os.fspath(fname)
  if interface is None:
    with open(fname, encoding="utf-8", errors="replace") as f:
      head = [next(f, "") for _ in range(4)]
    interface = "VPSC" if re.match(r"^\s*[BbKkRr]\s+\d+\s*$", head[3]) else "generic"
  if interface == "VPSC":
    return loadODF_VPSC(fname, *args, **kwargs)
  return loadODF_generic(fname, *args, **kwargs)


__all__ = ["loadODF", "loadODF_generic", "loadODF_VPSC"]
