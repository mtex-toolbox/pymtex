# the EDAX/TSL .ang text format (MATLAB loadEBSD_ang, Julia ang.jl)
#
#   ebsd = loadEBSD_ang('map.ang')                 the Euler angles aligned by setting 2
#   ebsd = loadEBSD_ang('map.ang', setting=3)      one of the four EDAX alignments, 0 none
#   ebsd = loadEBSD_ang('map.ang', EMSphInx=True)  the phase column counted from zero
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
from .load import ebsdFromColumns, placedPhases, regularPoly, tslPointGroup, unmeasuredRows
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
  starts = [i for i, l in enumerate(header) if re.match(r"^#\s*Phase\s+\d", l)]
  numbered = bool(starts)
  if not numbered:
    starts = [i for i, l in enumerate(header) if l.startswith("# MaterialName")]
  found, names, families = {}, [], []
  for k, s in enumerate(starts):
    stop = starts[k + 1] if k + 1 < len(starts) else len(header)
    block = header[s:stop]
    num = int(re.match(r"^#\s*Phase\s+(\d+)", header[s]).group(1)) if numbered else k + 1
    name = _token(block, "materialname")
    names.append(name or "")
    families.append(_token(block, "numberfamilies"))
    name = re.sub(r"\s+", " ", name).strip() if name else f"phase {num}"
    pg = tslPointGroup(_token(block, "symmetry") or "1", _token(block, "pointgroupid"))
    abc = _token(block, "latticeconstants")
    cell = [float(v) for v in abc.split()] if abc else [1, 1, 1, 90, 90, 90]
    found[num] = crystalFrame(pg, cell[:3], np.array(cell[3:6]) * degree, "EDAX", mineral=name)
  return placedPhases(found), names, families


def angIsEMSphInx(header, names, families):
  """whether the header is EMSphInx's rather than OIM's: it names no material, states zero
  reflector families for every phase and carries no version line. The two write the same
  layout and number the phase column differently, EDAX from one with zero for not indexed,
  EMSphInx from zero with 255 for not indexed."""
  return bool(names) and all(not n for n in names) and all(f == "0" for f in families) and \
    not any(l.startswith("# VERSION") for l in header)


def angHeader(header):
  """the header lines outside the phase blocks as a flat dict, MATLAB's angHeaderStruct"""
  starts = [i for i, l in enumerate(header) if re.match(r"^#\s*Phase\s+\d", l)]
  exclude = np.zeros(len(header), dtype=bool)
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


def loadEBSD_ang(fname, *, eulerCorrection=None, setting=2, columnNames=None, frame=None, scanUnit="um",
                 EDAX=False, EMSphInx=False):
  """the EDAX/TSL map of an .ang file

    loadEBSD_ang('map.ang')                        setting 2, a half turn about (1, -1, 0)
    loadEBSD_ang('map.ang', setting=0)             the Euler angles as the file states them
    loadEBSD_ang('map.ang', eulerCorrection=rot)   the alignment stated outright

  EDAX and EMSphInx write the same layout and number the phase column differently; the header
  is fingerprinted, and EDAX=True or EMSphInx=True overrules the guess.
  """
  data = readFile(fname)
  head, offset = leadingLines(data, lambda l: not l.startswith(b"#"))
  header = [text(l) for l in head]
  phases, names, families = angPhases(header)
  emsphinx = True if EMSphInx else (False if EDAX else angIsEMSphInx(header, names, families))
  M = readTable(data, offset, comments="#")
  if M.shape[1] < 8:
    raise ValueError(f"an .ang file needs at least eight columns, this has {M.shape[1]}")
  cols = list(columnNames) if columnNames is not None else \
    ["phi1", "Phi", "phi2", "x", "y", "iq", "ci", "phase", "sem_signal", "fit"][:M.shape[1]] + [f"extra{k}" for k in range(1, M.shape[1] - 9)]
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
  if eulerCorrection is None:
    eulerCorrection = eulerCorrectionRotation(setting)
  return ebsdFromColumns(at["x"], at["y"], at["phi1"], at["phi"], at["phi2"], at["phase"], phases, props,
                         eulerCorrection=eulerCorrection, scanUnit=scanUnit, frame=frame, meta=meta,
                         shift=1 if emsphinx else None, unindexed=unindexed, unitCell=_cellHint(meta["header"]),
                         scanGrid=_scanGrid(meta["header"]))


# the scan the header states: its rows and columns, the steps, the shape of the cell
def _scanGrid(header):
  xs, ys, nOdd, nEven, nRows = (header.get(k) for k in ("XSTEP", "YSTEP", "NCOLS_ODD", "NCOLS_EVEN", "NROWS"))
  if not all(isinstance(v, float) for v in (xs, ys, nOdd, nEven, nRows)):
    return None
  hex = str(header.get("GRID", "")).lower() == "hexgrid"
  return (nOdd, nRows, xs, ys, hex, nEven)


# the cell the header states, preferred over the estimate from the positions
def _cellHint(header):
  xs, ys = header.get("XSTEP"), header.get("YSTEP")
  if not isinstance(xs, float) or not isinstance(ys, float) or xs <= 0 or ys <= 0:
    return None
  if str(header.get("GRID", "")).lower() == "hexgrid":
    return regularPoly(6, xs)
  return regularPoly(4, [xs, ys])


__all__ = ["loadEBSD_ang"]
