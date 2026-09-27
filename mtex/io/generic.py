# a table whose columns the caller names, or the file itself (MATLAB loadEBSD_generic, Julia loadEBSDTable)
#
#   ebsd = loadEBSD_generic('map.txt', CS, columnNames=['x', 'y', 'Euler1', 'Euler2', 'Euler3', 'phase'])
#   ebsd = loadEBSD_generic('map.txt', CS)                 the columns named by the line above the numbers
#   ebsd = loadEBSD_generic('map.txt', CS, 'ABG', radians=True, columns=[1, 2, 4, 5, 6, 3])
#
# The names x, y, phase and the three Euler angles are the map; every other name is a
# property under that name, lowercased. MTEX opens the import wizard where no names are
# given and reads the map as one phase where the phase column holds more than forty values;
# here a file that names nothing asks for columnNames and a column the caller named is the
# column they named.
from __future__ import annotations

import re

import numpy as np

from ..geometry.frames import referenceFrame
from ..geometry.units import degree
from ..maps.phaselist import phaseList
from .load import ebsdFromColumns
from .table import isNumericRow, leadingLines, linesFrom, readFile, readTable, text

# the name a header gives a column, in the names this reader knows
COLUMN_NAMES = {"x": "x", "y": "y", "phase": "phase", "phi1": "phi1", "phi": "Phi", "phi2": "phi2",
                "euler1": "phi1", "euler2": "Phi", "euler3": "phi2", "euler 1": "phi1", "euler 2": "Phi", "euler 3": "phi2",
                "alpha": "phi1", "beta": "Phi", "gamma": "phi2"}


def _canonical(name):
  n = re.sub(r"\s+", " ", str(name).strip().lower())
  return COLUMN_NAMES.get(n, re.sub(r"\W+", "_", n))


def tableColumns(head):
  """the columns a file names itself, on the last line before its numeric block: a line of
  words above a block of numbers need not be a header at all, so it has to name the positions"""
  if not head:
    return None
  names = [s for s in re.split(r"[\t,;]| ", text(head[-1])) if s.strip()]
  names = [_canonical(n) for n in names]
  return names if "x" in names and "y" in names else None


def loadEBSD_generic(fname, CS, *args, columnNames=None, columns=None, radians=False, delimiter=None, header=None,
                     eulerCorrection=None, frame=None, scanUnit="um", replaceExpr=None):
  """the map a column oriented text file holds

    loadEBSD_generic('map.txt', CS, columnNames=['x', 'y', 'Euler1', 'Euler2', 'Euler3', 'phase'])
    loadEBSD_generic('map.txt', CS)                           the names read off the line above the block
    loadEBSD_generic('map.txt', CS, 'ABG', radians=True)      the Euler convention, the unit
    loadEBSD_generic('map.txt', CS, columns=[0, 1, 4, 5, 6])  the columns named, counted from zero

  CS is a crystalFrame or the list a phase number selects, the not indexed one first; a file
  with no phase column holds one phase. header is the number of lines above the block,
  delimiter what separates the numbers where it is not blank space, replaceExpr a list of
  (text, replacement) pairs applied to the file first, for a file that spells a missing
  value in words.
  """
  if CS is None:
    raise ValueError("a table states no crystal frame; pass the phases as loadEBSD(path, cs) or loadEBSD(path, [cs1, cs2])")
  phases = CS if isinstance(CS, phaseList) else phaseList([CS] if isinstance(CS, referenceFrame) else list(CS))
  convention = "Bunge"
  for a in args:
    if isinstance(a, str):
      convention = a
    else:
      raise TypeError(f"loadEBSD_generic does not understand the argument {a!r}")
  data = readFile(fname)
  for old, new in (replaceExpr or []):
    data = data.replace(old.encode(), new.encode())
  if header is None:
    sep = delimiter.encode() if delimiter is not None and delimiter.strip() else None
    head, offset = leadingLines(data, lambda l: isNumericRow(l if sep is None else l.replace(sep, b" ")))
    if offset >= len(data):
      raise ValueError(f"{fname} holds no row of numbers")
  else:
    head, offset = linesFrom(data, 0, int(header))
  names = list(columnNames) if columnNames is not None else tableColumns(head)
  if names is None:
    raise ValueError(f"{fname} does not name its columns; pass them as columnNames=['x', 'y', 'Euler1', 'Euler2', 'Euler3', 'phase']")
  names = [_canonical(n) for n in names]
  M = readTable(data, offset, delimiter=delimiter)
  if columns is not None:
    M = M[:, list(columns)]
  if M.shape[1] < len(names):
    raise ValueError(f"{fname} has {M.shape[1]} columns, the names given are {len(names)}")
  at = {n: M[:, k] for k, n in enumerate(names)}
  for key in ("x", "y", "phi1", "Phi", "phi2"):
    if key not in at:
      raise ValueError("a map is named by x, y and the three Euler angles phi1, Phi and phi2 or Euler1, Euler2, Euler3")
  s = 1.0 if radians else degree
  # a file with no phase column holds one phase, the first indexed one of the list
  phase = at.get("phase")
  if phase is None:
    phase = np.full(M.shape[0], int(phases.indexedPhaseId[0]) if len(phases.indexedPhaseId) else 0)
  props = {n: at[n] for n in names if n not in ("x", "y", "phi1", "Phi", "phi2", "phase")}
  return ebsdFromColumns(at["x"], at["y"], at["phi1"] * s, at["Phi"] * s, at["phi2"] * s, phase, phases, props,
                         convention=convention, eulerCorrection=eulerCorrection, scanUnit=scanUnit, frame=frame,
                         shift=0 if "phase" not in at else None)


__all__ = ["loadEBSD_generic"]
