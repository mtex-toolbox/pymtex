# a table of orientations: read by the names of its columns (MATLAB orientation/load.m,
# loadOrientation_generic.m)
#
#   ori, S = loadOrientation('ori.txt', cs, columnNames=['phi1', 'Phi', 'phi2'])
#   ori, S = loadOrientation('ori.txt', cs, ss, columnNames=['Quat real', 'Quat i', 'Quat j', 'Quat k'])
#   orientation.load('ori.txt', cs, columnNames=[...])    the orientations alone
#
# The Euler angles are Bunge's under phi1, Phi, phi2 or Euler 1, 2, 3 and Matthies' under alpha, beta, gamma,
# in degrees unless radians=True; four quaternion columns need no unit. passive=True reads
# the rows as the inverse map. The columns that are not coordinates come back in S, a dict
# of one array per name, lower case and without whitespace, as MATLAB names its fields.
from __future__ import annotations

import re

import numpy as np

from ..geometry.frames import specimenFrame
from ..geometry.orientation import orientation
from ..geometry.units import degree
from .table import _canonicalName, _tableStart, readFile, readTable, text

# the Euler conventions by the names of their three columns
EULER_COLUMNS = {("phi1", "phi", "phi2"): "Bunge", ("euler 1", "euler 2", "euler 3"): "Bunge",
                 ("alpha", "beta", "gamma"): "Matthies"}
QUATERNION_COLUMNS = ("quat real", "quat i", "quat j", "quat k")


def loadOrientation(fname, CS=None, SS=None, columnNames=None, columns=None, radians=False, delimiter=None, header=None,
                    passive=False):
  """the orientations a text table holds, and the other columns it names

    ori, S = loadOrientation('ori.txt', cs)                                  # the columns named in the header
    ori, S = loadOrientation('ori.txt', cs, columnNames=['phi1', 'Phi', 'phi2', 'weight'])  # the names given
    ori, S = loadOrientation('ori.txt', cs, columnNames=['phi1', 'Phi', 'phi2'], columns=[3, 1, 6])  # where they sit
    ori, S = loadOrientation('ori.txt', cs, columnNames=['Quat real', 'Quat i', 'Quat j', 'Quat k'])  # quaternions

  The columns are named by columnNames or by the last header line; three of them have to be
  Euler angles, phi1, Phi, phi2 or Euler 1, Euler 2, Euler 3 (Bunge) or alpha, beta, gamma
  (Matthies), or the four quaternion components. The other named columns come back in the
  dictionary S, under their names without blanks. A .csv file is separated by commas.

  Input
    fname - the name of the file
    CS - crystalFrame, by default the triclinic one
    SS - specimenFrame, by default the default specimen frame
  Output
    ori - orientation
    S - dict of the other columns, one array each
  Keywords
    columnNames - the names of the columns, instead of the header line
    columns - the zero based columns the names refer to
    radians - the Euler angles are in radians instead of degree
    delimiter - the separator of the columns, by default blank space
    header - the number of header lines, by default the lines before the first row of numbers
    passive - the angles describe the passive rotation, which is inverted
  See also
    exportOrientation, loadEBSD, loadODF, orientation.byEuler, OrientationImport"""
  # no crystal frame named: the triclinic one, as MATLAB's; a .csv file separates by commas
  from ..geometry.frames import crystalFrame
  CS = crystalFrame("1") if CS is None else CS
  SS = specimenFrame.default if SS is None else SS
  if delimiter is None and str(fname).lower().endswith(".csv"):
    delimiter = ","
  data = readFile(fname)
  head, offset = _tableStart(data, header, delimiter)
  if offset is None:
    raise ValueError(f"{fname} holds no row of numbers")
  names = list(columnNames) if columnNames is not None else (text(head[-1]).split() if head else None)
  if names is None:
    raise ValueError(f"{fname} does not name its columns; pass them as columnNames=['phi1', 'Phi', 'phi2']")
  M = readTable(data, offset, delimiter=delimiter)
  if columns is not None:
    M = M[:, list(columns)]
  if M.shape[1] < len(names):
    raise ValueError(f"{fname} has {M.shape[1]} columns, the names given are {len(names)}")
  at = {_canonicalName(n): M[:, k] for k, n in enumerate(names)}
  used = None
  for cols, convention in EULER_COLUMNS.items():
    if all(c in at for c in cols):
      s = 1.0 if radians else degree
      ori = orientation.byEuler(at[cols[0]] * s, at[cols[1]] * s, at[cols[2]] * s, convention, CS, SS)
      used = cols
      break
  else:
    if all(c in at for c in QUATERNION_COLUMNS):
      ori = orientation(np.stack([at[c] for c in QUATERNION_COLUMNS], axis=-1), CS, SS)
      used = QUATERNION_COLUMNS
    else:
      raise ValueError("the columns name no orientation: phi1, Phi, phi2; alpha, beta, gamma; or Quat real, Quat i, Quat j, Quat k")
  if passive:
    ori = orientation(ori.inv(), CS, SS)
  S = {re.sub(r"\s+", "", n): v for n, v in at.items() if n not in used}
  return ori, S


__all__ = ["loadOrientation"]
