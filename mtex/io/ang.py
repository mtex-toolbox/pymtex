# the EDAX/TSL .ang text format (MATLAB loadEBSD_ang, Julia ang.jl)
#
#   ebsd = loadEBSD('map.ang')                 the Euler angles aligned by setting 2
#   ebsd = loadEBSD('map.ang', setting=3)      one of the four EDAX alignments, 0 none
#   ebsd = loadEBSD('map.ang', EMSphInx=True)  the phase column counted from zero
#   ebsd = loadEBSD('acom.ang')                a TEM ACOM map: setting 4, positions in nm
#
# A header of lines beginning with '#', one block per phase, then a numeric table whose
# first eight columns are fixed by the format, phi1 PHI phi2 x y IQ CI phase, and whose
# remaining columns moved between the vendor's versions. The eight are read by the
# convention and the rest kept as extra1, extra2, ...; columnNames names them all.
from __future__ import annotations

import re

import numpy as np

from ..geometry.frames import crystalFrame
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .load import ebsdFromColumns, placedPhases, regularPoly, tslPointGroup, unmeasuredRows, scanGrid
from .table import headerDict, leadingLines, readFile, readTable, text

# the keys of a phase block, which the header dict leaves out
PHASE_KEYS = ("materialname", "formula", "info", "symmetry", "pointgroupid", "latticeconstants", "numberfamilies",
              "hklfamilies", "elasticconstants", "categories", "phase")
_KEY = re.compile(r"^#\s*([^\s:]+):?\s*(.*)$")


def eulerCorrectionRotation(setting):
  """the Euler to map alignment behind an EDAX setting, MATLAB's eulerCorrectionRotation"""
  if setting not in (0, 1, 2, 3, 4):
    raise ValueError("the setting has to be 0 to 4")
  if setting == 0:
    return rotation.id
  axis = [vector3d(1, 1, 0), vector3d(1, -1, 0), vector3d(1, 0, 0), vector3d(0, 1, 0)][setting - 1]
  return rotation.byAxisAngle(axis, np.pi)


# the header lines that open a numbered phase block, "# Phase 1"
def _phaseStarts(header):
  return [i for i, l in enumerate(header) if re.match(r"^#\s*Phase\s+\d", l)]


# the value after a token in a block of header lines
def _token(block, token):
  for l in block:
    m = _KEY.match(l)
    if m and m.group(1).lower() == token:
      return m.group(2).strip()
  return None


def angPhases(header):
  """the phases the header declares, placed by the number they declare: olivineopticalmap.ang
  declares its four phases as 4, 3, 2, 1, so reading them in file order pairs every name
  with the wrong lattice"""
  starts = _phaseStarts(header)
  numbered = bool(starts)
  if not numbered:
    starts = [i for i, l in enumerate(header) if l.startswith("# MaterialName")]
  found, names, families = {}, [], []
  for k, (s, stop) in enumerate(zip(starts, starts[1:] + [len(header)])):
    block = header[s:stop]
    num = int(re.match(r"^#\s*Phase\s+(\d+)", header[s]).group(1)) if numbered else k + 1
    name = angMineral(_token(block, "materialname"))
    names.append(name or "")
    families.append(_token(block, "numberfamilies"))
    name = re.sub(r"\s+", " ", name).strip() if name else f"phase {num}"
    pg = tslPointGroup(_token(block, "symmetry") or "1", _token(block, "pointgroupid"))
    abc = _token(block, "latticeconstants")
    cell = [float(v) for v in abc.split()] if abc else [1, 1, 1, 90, 90, 90]
    found[num] = crystalFrame(pg, cell[:3], np.array(cell[3:6]) * degree, "EDAX", mineral=name)
  return placedPhases(found), names, families


def angMineral(value):
  """the mineral a MaterialName line names: the first quoted name where the line quotes one,
  as the ACOM export writes a database number and the name twice, 5000216 'Copper' 'Copper'"""
  if value is None:
    return None
  m = re.search(r"'([^']+)'", value)
  return m.group(1).strip() if m else value


def angIsACOM(header):
  """whether the header is the ACOM export of a TEM orientation map, NanoMEGAS's ASTAR, which
  writes the positions in nm and the Euler angles in the frame of EDAX's setting 4"""
  return any(re.match(r"^#.*\b(ACOM|TEM data)\b", l) for l in header)


def angIsEMSphInx(header, names, families):
  """whether the header is EMSphInx's rather than OIM's: it names no material, states zero
  reflector families for every phase and carries no version line. The two write the same
  layout and number the phase column differently, EDAX from one with zero for not indexed,
  EMSphInx from zero with 255 for not indexed."""
  return bool(names) and all(not n for n in names) and all(f == "0" for f in families) and \
    not any(l.startswith("# VERSION") for l in header)


def angHeader(header):
  """the header lines outside the phase blocks as a flat dict, MATLAB's angHeaderStruct"""
  starts = _phaseStarts(header)
  exclude =np.zeros(len(header), dtype=bool)
  for k, s in enumerate(starts):
    if k + 1 < len(starts):
      stop = starts[k + 1]
    else:
      stop = s + 1
      while stop < len(header):
        m = _KEY.match(header[stop])
        if m is None or any(m.group(1).lower().startswith(pk) for pk in PHASE_KEYS):
          stop += 1
        else:
          break
    exclude[s:stop] = True
  pairs = []
  for l, ex in zip(header, exclude):
    m = _KEY.match(l)
    if not ex and m and m.group(1):
      pairs.append((m.group(1), m.group(2)))
  return headerDict(pairs)


# the EDAX/TSL map of an .ang file
#
#   loadEBSD('map.ang')                        setting 2, a half turn about (1, -1, 0)
#   loadEBSD('map.ang', setting=0)             the Euler angles as the file states them
#   loadEBSD('map.ang', eulerCorrection=rot)   the alignment stated outright
#   loadEBSD('map.ang', scanUnit='nm')         the unit of the positions, um by default
#
# EDAX and EMSphInx write the same layout and number the phase column differently; the header
# is fingerprinted, and EDAX=True or EMSphInx=True overrules the guess. The ACOM export of a
# TEM map is fingerprinted too and read with setting 4 and the positions in nm, unless the
# call states them.
def _loadEBSD_ang(fname, *, eulerCorrection=None, setting=None, columnNames=None, frame=None, scanUnit=None,
                  EDAX=False, EMSphInx=False):
  data = readFile(fname)
  head, offset = leadingLines(data, lambda l: not l.startswith(b"#"))
  header = [text(l) for l in head]
  phases, names, families = angPhases(header)
  emsphinx = EMSphInx or (not EDAX and angIsEMSphInx(header, names, families))
  M = readTable(data, offset, comments="#")
  if M.shape[1] < 8:
    raise ValueError(f"an .ang file needs at least eight columns, this has {M.shape[1]}")
  cols = list(columnNames) if columnNames is not None else _angColumnNames(M.shape[1])
  if len(cols) > M.shape[1]:
    raise ValueError(f"{len(cols)} column names for {M.shape[1]} columns")
  at = {n.lower(): M[:, k] for k, n in enumerate(cols)}
  for key in ("phi1", "phi", "phi2", "x", "y", "phase"):
    if key not in at:
      raise ValueError(f"the columns of an .ang file name {key}")
  props = {n: M[:, k] for k, n in enumerate(cols) if n.lower() not in ("phi1", "phi", "phi2", "x", "y", "phase")}
  # EDAX marks a pattern it could not index by a negative confidence index; EMSphInx writes
  # the pixels it never measured as a phase at the zero orientation
  if emsphinx:
    unindexed = unmeasuredRows(at["phi1"], at["phi"], at["phi2"], props)
  else:
    unindexed = at["ci"] < 0 if "ci" in at else None
  meta = {"header": angHeader(header)}
  acom = angIsACOM(header)
  if eulerCorrection is None:
    if setting is None:
      setting = 4 if acom else 2
    eulerCorrection = eulerCorrectionRotation(setting)
  if scanUnit is None:
    scanUnit = "nm" if acom else "um"
  unitCell, grid = _headerGrid(meta["header"])
  return ebsdFromColumns(at["x"], at["y"], at["phi1"], at["phi"], at["phi2"], at["phase"], phases, props,
                         eulerCorrection=eulerCorrection, scanUnit=scanUnit, frame=frame, meta=meta,
                         shift=1 if emsphinx else None, unindexed=unindexed, unitCell=unitCell, scanGrid=grid)


# the names of the m columns of an .ang file: the ten the vendor's versions wrote, the rest
# numbered extra1, extra2, ...
def _angColumnNames(m):
  return ["phi1", "Phi", "phi2", "x", "y", "iq", "ci", "phase", "sem_signal", "fit"][:m] + [f"extra{k}" for k in range(1, m - 9)]


# the cell the header states, preferred over the estimate from the positions, and the scan: its
# rows and columns, the steps, the shape of the cell; None for what it does not state
def _headerGrid(header):
  xs, ys, nOdd, nEven, nRows = (header.get(k) for k in ("XSTEP", "YSTEP", "NCOLS_ODD", "NCOLS_EVEN", "NROWS"))
  hex = str(header.get("GRID", "")).lower() == "hexgrid"
  steps = isinstance(xs, float) and isinstance(ys, float)
  unitCell = None
  if steps and xs > 0 and ys > 0:
    unitCell = regularPoly(6, xs) if hex else regularPoly(4, [xs, ys])
  grid = None
  if steps and all(isinstance(v, float) for v in (nOdd, nEven, nRows)):
    grid = scanGrid(nOdd, nRows, xs, ys, hex, nEven)
  return unitCell, grid


__all__ = []
