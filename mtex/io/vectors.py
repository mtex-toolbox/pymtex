# a table of directions: read by the names of its coordinate columns, written as one
# (MATLAB vector3d/load.m and vector3d/export.m)
#
#   v, S = loadVector3d('vectors.txt', columnNames=['polar angle', 'azimuth angle'])
#   exportVector3d(v, 'vectors.txt')                 x y z
#   exportVector3d(v, 'vectors.txt', 'polar')        polar angle and azimuth in degrees
#   exportVector3d(v, 'vectors.txt', {'weight': w})  a column per named value
#
# The columns that are not coordinates come back in S, a dict of one array per name.
from __future__ import annotations

import os
import re

import numpy as np

from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .table import isNumericRow, leadingLines, linesFrom, readFile, readTable, text

# the words a coordinate column may be named by, and what they read as
COORDINATES = {"x": "x", "y": "y", "z": "z",
               "polar": "polar", "polar angle": "polar", "colatitude": "polar",
               "azimuth": "azimuth", "azimuth angle": "azimuth", "longitude": "azimuth",
               "latitude": "latitude"}


def _canonical(name):
  return re.sub(r"\s+", " ", str(name).strip().lower())


def loadVector3d(fname, columnNames=None, columns=None, radians=False, delimiter=None, header=None):
  """the directions a text table holds, and the other columns it names

    loadVector3d('vectors.txt', columnNames=['polar angle', 'azimuth angle'])
    loadVector3d('vectors.txt', columnNames=['x', 'y', 'z', 'weight'])   S['weight'] alongside
    loadVector3d('vectors.txt', columnNames=[...], columns=[1, 0])       the columns the names sit in, from zero
    loadVector3d('vectors.txt')                                          the names on the line above the block

  The coordinates are x, y, z, kept at their length; polar and azimuth, colatitude and
  longitude, or latitude and longitude, angles in degrees unless radians=True, read as unit
  directions.
  """
  data = readFile(fname)
  if header is None:
    sep = delimiter.encode() if delimiter is not None and delimiter.strip() else None
    head, offset = leadingLines(data, lambda l: isNumericRow(l if sep is None else l.replace(sep, b" ")))
    if offset >= len(data):
      raise ValueError(f"{fname} holds no row of numbers")
  else:
    head, offset = linesFrom(data, 0, int(header))
  names = list(columnNames) if columnNames is not None else _headerNames(head)
  if names is None:
    raise ValueError(f"{fname} does not name its columns; pass them as columnNames=['x', 'y', 'z']")
  M = readTable(data, offset, delimiter=delimiter)
  if columns is not None:
    M = M[:, list(columns)]
  if M.shape[1] < len(names):
    raise ValueError(f"{fname} has {M.shape[1]} columns, the names given are {len(names)}")
  at = {_canonical(n): M[:, k] for k, n in enumerate(names)}
  kind = {COORDINATES[n]: at[n] for n in at if n in COORDINATES}
  s = 1.0 if radians else degree
  if all(k in kind for k in ("x", "y", "z")):
    v = vector3d(kind["x"], kind["y"], kind["z"])
  elif "polar" in kind and "azimuth" in kind:
    v = vector3d.byPolar(kind["polar"] * s, kind["azimuth"] * s)
  elif "latitude" in kind and "azimuth" in kind:
    v = vector3d.byPolar(np.pi / 2 - kind["latitude"] * s, kind["azimuth"] * s)
  else:
    raise ValueError("the columns name no coordinates: x, y, z; polar and azimuth; or latitude and longitude")
  S = {n: at[_canonical(n)] for n in names if _canonical(n) not in COORDINATES}
  return v, S


# the names on the last header line, when it names coordinates at all
def _headerNames(head):
  if not head:
    return None
  line = text(head[-1]).strip()
  # a two word name, "polar angle", is one column; try the pairs first
  words = [w for w in re.split(r"[\t,;]| ", line) if w.strip()]
  names, k = [], 0
  while k < len(words):
    pair = " ".join(words[k:k + 2]).lower()
    if pair in COORDINATES:
      names.append(pair)
      k += 2
    else:
      names.append(words[k])
      k += 1
  return names if any(_canonical(n) in COORDINATES for n in names) else None


def exportVector3d(v, fname, *args, radians=False):
  """write directions as a text table with a line of column names

    exportVector3d(v, 'vectors.txt')                    the Cartesian components, at their length
    exportVector3d(v, 'vectors.txt', 'polar')           polar angle and azimuth, in degrees unless radians=True
    exportVector3d(v, 'vectors.txt', {'weight': w})     one more column per named value
  """
  polar, extra = False, {}
  for a in args:
    if isinstance(a, str) and a.lower() == "polar":
      polar = True
    elif isinstance(a, dict):
      extra.update(a)
    else:
      raise TypeError(f"export does not understand the argument {a!r}")
  d = v.data.reshape(-1, 3)
  if polar:
    s = 1.0 if radians else 1 / degree
    names = ["polar angle", "azimuth angle"]
    cols = [np.asarray(v.theta).reshape(-1) * s, np.asarray(v.rho).reshape(-1) * s]
  else:
    names = ["x", "y", "z"]
    cols = [d[:, 0], d[:, 1], d[:, 2]]
  for key, val in extra.items():
    val = np.asarray(val, dtype=float).reshape(-1)
    if val.size != d.shape[0]:
      raise ValueError(f"the column {key!r} has {val.size} values for {d.shape[0]} vectors")
    names.append(str(key))
    cols.append(val)
  # six significant digits, every column as wide as its widest entry, as MATLAB writes it
  cells = [[f"{x:g}" for x in c] for c in cols]
  widths = [max(len(n), *(len(s) for s in c)) for n, c in zip(names, cells)]
  lines = [" ".join(n.rjust(w) for n, w in zip(names, widths))]
  for row in zip(*cells):
    lines.append(" ".join(s.rjust(w) for s, w in zip(row, widths)))
  with open(os.fspath(fname), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  return os.fspath(fname)


__all__ = ["loadVector3d", "exportVector3d"]
