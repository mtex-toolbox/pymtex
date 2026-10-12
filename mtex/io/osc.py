# the EDAX/TSL binary map (MATLAB loadEBSD_osc, Julia osc.jl)
#
#   ebsd = loadEBSD('map.osc')                the Euler angles aligned by setting 2
#   ebsd = loadEBSD('map.osc', setting=0)     the angles as the file states them
#
# Eight 32 bit words open the file, of which the fifth to the seventh are the two cell
# counts and the number of measurements. Two markers cut the header out of the rest,
# B9 0B EF FF 01 00 00 00 opening it and ...02 00 00 00 closing it, and the measurements
# follow the closing one as one block of 32 bit floats, a fixed number of columns per pixel;
# a scan stopped early keeps the count and fills the rest with placeholders, which are dropped.
# The header holds one record per phase: 256 bytes of name, the TSL symmetry code, the six
# cell parameters, then its reflections, so a record is not a fixed length. MTEX finds the
# records by the 485 phase names it keeps in osc_phases.txt; here a record is recognised by
# being one, a name that is a name, a symmetry code and six numbers that can be a cell.
from __future__ import annotations

import numpy as np

from ..geometry.frames import crystalFrame
from ..geometry.units import degree
from ..maps.phaselist import phaseList
from .ang import eulerCorrectionRotation
from .load import EDAX_POINT_GROUPS, TSL_LAUE, _edaxPointGroup, _properties, ebsdFromColumns, regularPoly, scanGrid
from .table import readFile

OSC_HEADER_OPEN = b"\xb9\x0b\xef\xff\x01\x00\x00\x00"
OSC_HEADER_CLOSE = b"\xb9\x0b\xef\xff\x02\x00\x00\x00"
OSC_PHASE_NAME = 256      # bytes of name
OSC_PHASE_LENGTH = 288    # the fixed part of the record
# the ten columns the format fixes; what a file writes after them keeps its number
OSC_COLUMNS = ["phi1", "phi", "phi2", "x", "y", "imagequality", "confidenceindex", "phase", "semsignal", "fit"]


def _f32(b, o):
  return float(np.frombuffer(b, dtype="<f4", count=1, offset=o)[0])


def _u32(b, o):
  return int(np.frombuffer(b, dtype="<u4", count=1, offset=o)[0])


def oscPhaseAt(b, o):
  """a phase record at offset o, or None where the bytes are not one: a name that is a name,
  a code and a cell of positive lengths and angles each smaller than the sum of the other two"""
  if o + OSC_PHASE_LENGTH > len(b) or o < 0:
    return None
  field = b[o:o + OSC_PHASE_NAME]
  stop = field.find(b"\x00")
  name = (field if stop < 0 else field[:stop]).decode("ascii", errors="replace").strip()
  if not name or not all(" " <= c <= "~" for c in name) or not any(c.isalpha() for c in name):
    return None
  cell = [_f32(b, o + 260 + 4 * k) for k in range(3)]
  angles = [_f32(b, o + 272 + 4 * k) for k in range(3)]
  if not all(0 < v < 1000 for v in cell) or not all(0 < v < 180 for v in angles) or sum(angles) >= 360:
    return None
  if not all(angles[k] < sum(angles) - angles[k] for k in range(3)):
    return None
  # EDAX writes the symmetry as a TSL Laue code, 1 to 43, or as a point group id, 100 to 131;
  # a phase whose code is in neither table keeps its cell and loses its symmetry rather than
  # being given one it does not have
  code = _u32(b, o + OSC_PHASE_NAME)
  symbol = TSL_LAUE.get(code) or _edaxPointGroup(code) or "1"
  try:
    return crystalFrame(symbol, cell, np.array(angles) * degree, "EDAX", mineral=name)
  except ValueError:
    return None


def oscPhases(b, first, last):
  """the phases the header block describes

  The block is scanned for the places a symmetry code can sit, each tried as the code of a
  record beginning 256 bytes before it; after a hit the scan steps past the record's fixed
  part, without which a record matches again a few bytes along where its name begins with
  blanks. What keeps a code from being found in the bytes of a reflection is oscPhaseAt."""
  frames = []
  codes = sorted(TSL_LAUE) + list(range(100, 100 + len(EDAX_POINT_GROUPS)))
  hits = []
  for code in codes:
    pat = int(code).to_bytes(4, "little")
    p = b.find(pat, first, last)
    while p >= 0:
      hits.append(p - OSC_PHASE_NAME)
      p = b.find(pat, p + 1, last)
  after = first
  for o in sorted(hits):
    if o < after:
      continue
    fr = oscPhaseAt(b, o)
    if fr is not None:
      frames.append(fr)
      after = o + OSC_PHASE_LENGTH
  return phaseList(frames)


def oscDataStart(b, close, n):
  """where the block of measurements begins, and the two step sizes it was measured on: a
  length word may or may not sit between the closing marker and the data, and it is one
  when it counts the floats that follow"""
  o = close + 8
  words = _u32(b, o)
  if round((words / 4 - 2) / 10 / n) == 1:
    o += 4
  xstep = _f32(b, o)
  # a file that writes a zero first states both steps after it
  if xstep == 0:
    o += 4
    xstep = _f32(b, o)
  ystep = _f32(b, o + 4)
  return o + 8, xstep, ystep


def oscColumnCount(b, start, xstep):
  """how many columns a record holds: nowhere in the file, so read off the map, the second pixel
  of the first row sitting one step along x and none along y"""
  for m in range(5, 31):
    x, y = _f32(b, start + 4 * (m + 3)), _f32(b, start + 4 * (m + 4))
    if abs(x - xstep) < 1e-4 and abs(y) < 1e-6:
      return m
  raise ValueError("no column count between 5 and 30 puts the second pixel one step along x")


def oscTrim(M):
  """the records without the placeholders a scan stopped early leaves at its end: the file keeps
  the count it announced and fills the pixels never measured with the three Euler angles 4 pi at
  the position (0, 0), all of which would sit on the first pixel's site"""
  unmeasured = (M[:, 3] == 0) & (M[:, 4] == 0) & np.all(M[:, :3] == np.float32(4 * np.pi), axis=1)
  kept = np.flatnonzero(~unmeasured)
  return M[:kept[-1] + 1] if len(kept) else M[:0]


def oscHeader(b, nx, ny, n, xstep, ystep):
  """what MATLAB reads out of the fixed places of the header"""
  def chars(pos):
    stop = b.find(b"\x00", pos)
    return b[pos:stop if stop >= 0 else pos].decode("ascii", errors="replace").strip()
  header = {"info": chars(72), "operator": chars(1095), "sampleId": chars(1350), "scanId": chars(1605)}
  cal = np.frombuffer(b, dtype="<f4", count=4, offset=1860)
  header.update(x_star=float(cal[0]), y_star=float(cal[1]), z_star=float(cal[2]), workingDistance=float(cal[3]))
  header.update(nColumns=nx, nRows=ny, nPoints=n, xstep=xstep, ystep=ystep)
  return header


# the layout of an .osc: the cell counts and the number of measurements of its first words, the
# phases between the two markers, where the measurements begin, the two steps and the columns
def _oscLayout(b, fname):
  if len(b) <= 32:
    raise ValueError(f"{fname} is too short to be an .osc")
  nx, ny, n = _u32(b, 16), _u32(b, 20), _u32(b, 24)
  if n <= 0:
    raise ValueError(f"{fname} announces {n} measurements")
  # the header sits between the two markers, and the data behind the closing one
  opening = b.find(OSC_HEADER_OPEN)
  close = b.find(OSC_HEADER_CLOSE, max(opening, 0))
  if close < 0:
    raise ValueError(f"{fname} carries no header marker")
  phases = phaseList() if opening < 0 else oscPhases(b, opening + 8, close)
  start, xstep, ystep = oscDataStart(b, close, n)
  return nx, ny, n, phases, start, xstep, ystep, oscColumnCount(b, start, xstep)


# the names of m columns: the ten the format fixes, the rest numbered
def _oscColumnNames(m):
  return OSC_COLUMNS[:m] + [f"extra{j}" for j in range(1, m - len(OSC_COLUMNS) + 1)]


# the EDAX/TSL binary map of an .osc file
#
# loadEBSD('map.osc')                        setting 2, a half turn about (1, -1, 0)
# loadEBSD('map.osc', eulerCorrection=rot)   the alignment stated outright
def _loadEBSD_osc(fname, *, eulerCorrection=None, setting=2, frame=None, scanUnit="um"):
  b = readFile(fname)
  nx, ny, n, phases, start, xstep, ystep, m = _oscLayout(b, fname)
  if start + 4 * n * m > len(b):
    raise ValueError(f"{fname} ends before its {n} records of {m} columns do")
  M = oscTrim(np.frombuffer(b, dtype="<f4", count=n * m, offset=start).reshape(n, m))
  cols = dict(zip(_oscColumnNames(m), M.T))
  props = _properties(cols, ("x", "y", "phi1", "phi", "phi2", "phase"), f"{fname} holds {m} columns, too few for a map")
  # a pattern EDAX could not index carries a negative confidence index
  unindexed = cols["confidenceindex"] < 0 if "confidenceindex" in cols else None
  hex = abs(xstep - ystep) > 1e-6 * max(xstep, ystep)
  unitCell = regularPoly(6, xstep) if hex else regularPoly(4, [xstep, ystep])
  meta = {"header": oscHeader(b, nx, ny, n, xstep, ystep)}
  if eulerCorrection is None:
    eulerCorrection = eulerCorrectionRotation(setting)
  return ebsdFromColumns(cols["x"], cols["y"], cols["phi1"], cols["phi"], cols["phi2"], cols["phase"], phases, props,
                         eulerCorrection=eulerCorrection, scanUnit=scanUnit, frame=frame, meta=meta, unindexed=unindexed,
                         unitCell=unitCell if xstep > 0 and ystep > 0 else None,
                         scanGrid=_scanGrid(nx, ny, len(M), xstep, ystep, hex), propType=np.float64)


# the grid the header words state; a hexagonal scan alternates nx and a shorter row
def _scanGrid(nx, ny, n, xstep, ystep, hex):
  if nx < 1 or ny < 1 or xstep <= 0 or ystep <= 0:
    return None
  if not hex:
    return scanGrid(nx, ny, xstep, ystep)
  odd, even = (ny + 1) // 2, ny // 2
  if even == 0:
    return scanGrid(nx, ny, xstep, ystep, True, nx) if nx * odd == n else None
  rest = n - odd * nx
  return scanGrid(nx, ny, xstep, ystep, True, rest // even) if rest % even == 0 and rest // even > 0 else None


__all__ = []
