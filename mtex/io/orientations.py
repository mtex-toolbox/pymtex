# a table of orientations: read by the names of its columns (MATLAB orientation/load.m,
# loadOrientation_generic.m)
#
#   ori, S = loadOrientation('ori.txt', cs, columnNames=['phi1', 'Phi', 'phi2'])
#   ori, S = loadOrientation('ori.txt', cs, ss, columnNames=['Quat real', 'Quat i', 'Quat j', 'Quat k'])
#   orientation.load('ori.txt', cs, columnNames=[...])    the orientations alone
#
# The Euler angles are Bunge's under phi1, Phi, phi2 and Matthies' under alpha, beta, gamma,
# in degrees unless radians=True; four quaternion columns need no unit. passive=True reads
# the rows as the inverse map. The columns that are not coordinates come back in S, a dict
# of one array per name, lower case and without whitespace, as MATLAB names its fields.
from __future__ import annotations

import re

import numpy as np

from ..geometry.frames import specimenFrame
from ..geometry.orientation import orientation
from ..geometry.units import degree
from .table import isNumericRow, leadingLines, linesFrom, readFile, readTable, text

# the Euler conventions by the names of their three columns
EULER_COLUMNS = {("phi1", "phi", "phi2"): "Bunge", ("alpha", "beta", "gamma"): "Matthies"}
QUATERNION_COLUMNS = ("quat real", "quat i", "quat j", "quat k")


def _canonical(name):
  return re.sub(r"\s+", " ", str(name).strip().lower())


def loadOrientation(fname, cs, ss=None, columnNames=None, columns=None, radians=False, delimiter=None, header=None,
                    passive=False):
  """the orientations a text table holds, and the other columns it names

    loadOrientation('ori.txt', cs, columnNames=['phi1', 'Phi', 'phi2'])
    loadOrientation('ori.txt', cs, columnNames=['phi1', 'Phi', 'phi2', 'weight'])   S['weight'] alongside
    loadOrientation('ori.txt', cs, columnNames=[...], columns=[3, 1, 6])          the columns the names sit in, from zero
    loadOrientation('ori.txt', cs, columnNames=['Quat real', 'Quat i', 'Quat j', 'Quat k'])
  """
  ss = specimenFrame.default if ss is None else ss
  data = readFile(fname)
  if header is None:
    sep = delimiter.encode() if delimiter is not None and delimiter.strip() else None
    head, offset = leadingLines(data, lambda l: isNumericRow(l if sep is None else l.replace(sep, b" ")))
    if offset >= len(data):
      raise ValueError(f"{fname} holds no row of numbers")
  else:
    head, offset = linesFrom(data, 0, int(header))
  names = list(columnNames) if columnNames is not None else (text(head[-1]).split() if head else None)
  if names is None:
    raise ValueError(f"{fname} does not name its columns; pass them as columnNames=['phi1', 'Phi', 'phi2']")
  M = readTable(data, offset, delimiter=delimiter)
  if columns is not None:
    M = M[:, list(columns)]
  if M.shape[1] < len(names):
    raise ValueError(f"{fname} has {M.shape[1]} columns, the names given are {len(names)}")
  at = {_canonical(n): M[:, k] for k, n in enumerate(names)}
  used = None
  for cols, convention in EULER_COLUMNS.items():
    if all(c in at for c in cols):
      s = 1.0 if radians else degree
      ori = orientation.byEuler(at[cols[0]] * s, at[cols[1]] * s, at[cols[2]] * s, convention, cs, ss)
      used = cols
      break
  else:
    if all(c in at for c in QUATERNION_COLUMNS):
      ori = orientation(np.stack([at[c] for c in QUATERNION_COLUMNS], axis=-1), cs, ss)
      used = QUATERNION_COLUMNS
    else:
      raise ValueError("the columns name no orientation: phi1, Phi, phi2; alpha, beta, gamma; or Quat real, Quat i, Quat j, Quat k")
  if passive:
    ori = orientation(ori.inv(), cs, ss)
  S = {re.sub(r"\s+", "", _canonical(n)): at[_canonical(n)] for n in names if _canonical(n) not in used}
  return ori, S


__all__ = ["loadOrientation"]
