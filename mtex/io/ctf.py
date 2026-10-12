# the Oxford/Channel .ctf text format (MATLAB loadEBSD_ctf, Julia ctf.jl)
#
#   ebsd = loadEBSD('map.ctf')                      a half turn about z, the tilt of the surface undone
#   ebsd = loadEBSD('map.ctf', eulerCorrection=r)   the alignment stated outright
#
# A header of tab separated key and value lines, a "Phases n" line followed by one line per
# phase, then a line naming the columns and the numeric table. Unlike an .ang the columns
# are named by the file, so nothing about them is guessed.
from __future__ import annotations

import warnings

import numpy as np

from ..geometry.frames import crystalFrame, notIndexedFrame
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..maps.phaselist import phaseList
from .load import CHANNEL_LAUE, _properties, ebsdFromColumns, regularPoly, scanGrid
from .table import headerDict, leadingLines, linesFrom, readFile, readTable, text


def _number(s):
  return float(s.strip().replace(",", "."))


def ctfPhases(phaseLines):
  """the phases the lines after a Channel header's Phases line declare: lattice, angles, name,
  Laue class, space group"""
  frames = [notIndexedFrame()]
  for k, line in enumerate(phaseLines, 1):
    f = line.split("\t")
    if len(f) < 3:
      continue
    abc = [_number(s) for s in f[0].split(";")]
    abg = [_number(s) for s in f[1].split(";")]
    name = f[2].strip() or f"phase {k}"
    laue, spaceId = None, None
    try:
      laue = CHANNEL_LAUE[int(f[3]) - 1] if 1 <= int(f[3]) <= len(CHANNEL_LAUE) else None
    except (ValueError, IndexError):
      pass
    if laue is None:
      try:
        spaceId = int(f[4]) or None
      except (ValueError, IndexError):
        pass
    if laue is not None:
      frames.append(crystalFrame(laue, abc, np.array(abg) * degree, mineral=name))
    elif spaceId is not None:
      frames.append(crystalFrame(abc, np.array(abg) * degree, spaceId=spaceId, mineral=name))
    else:
      frames.append(crystalFrame("-1", abc, np.array(abg) * degree, mineral=name))
  return phaseList(frames)


def ctfHeader(lines):
  """the general header lines as a flat dict, MATLAB's ctfHeaderStruct: a line of an even number
  of fields is key value pairs, an odd one carries a leading note that is dropped"""
  pairs = []
  for l in lines:
    parts = l.split("\t")
    if len(parts) % 2 == 1:
      parts = parts[1:]
    pairs += list(zip(parts[0::2], parts[1::2]))
  return headerDict(pairs)


def oxfordCorrection(acq):
  """a .ctf refers its angles to the sample primary frame CS0 and the map lies in the acquisition
  surface CS1; the tilt of the surface, AcqE2 and AcqE3, is undone before the half turn
  about z, and a rotation about the surface normal is not, as far as two exports can tell"""
  half = rotation.byEuler(np.pi, 0.0, 0.0, "Bunge")
  tilt = _surfaceTilt(acq)
  return half if tilt is None else half * tilt


# the rotation that undoes the tilt of an acquisition surface, its second and third Euler angle in
# degree; None for a surface without a tilt or none stated
def _surfaceTilt(acq):
  if acq is None or (abs(acq[1]) <= 1e-6 and abs(acq[2]) <= 1e-6):
    return None
  return rotation.byEuler(0.0, acq[1] * degree, acq[2] * degree, "Bunge").inv()


# the header of a .ctf up to its column names: the general header lines as a dict, the phases,
# the acquisition surface where all three of its angles are stated, and the offset of the line
# naming the columns
def _ctfHead(data, fname):
  head, offset = leadingLines(data, lambda l: l.startswith(b"Phases"))
  if offset >= len(data):
    raise ValueError(f"{fname} declares no phases")
  (phasesLine,), offset = linesFrom(data, offset, 1)
  n = int(text(phasesLine).split("\t")[-1])
  phaseLines, offset = linesFrom(data, offset, n)
  header = ctfHeader([text(l) for l in head])
  acq = None
  if all(isinstance(header.get(k), float) for k in ("AcqE1", "AcqE2", "AcqE3")):
    acq = (header["AcqE1"], header["AcqE2"], header["AcqE3"])
  return header, ctfPhases([text(l) for l in phaseLines]), acq, offset


# the Oxford/Channel map of a .ctf file
#
#   loadEBSD('map.ctf')                        the Euler angles half a turn about z from the map
#   loadEBSD('map.ctf', eulerCorrection=rot)   the alignment stated outright
#
# A stated acquisition surface with a tilt is undone before the half turn, with a warning;
# the surface is kept in ebsd.meta['surface'].
def _loadEBSD_ctf(fname, *, eulerCorrection=None, frame=None, scanUnit="um"):
  data = readFile(fname)
  if b"Channel Text File" not in data[:200].split(b"\n")[0]:
    raise ValueError(f"{fname} does not begin with the Channel Text File marker")
  header, phases, acq, offset = _ctfHead(data, fname)
  if eulerCorrection is None:
    eulerCorrection = oxfordCorrection(acq)
    if _surfaceTilt(acq) is not None:
      warnings.warn(f"{fname} states an acquisition surface of {acq} degree; its tilt is undone before the half turn "
                    "about z, since the file refers its angles to CS0 and the map lies in CS1. Pass eulerCorrection "
                    "to state the alignment yourself.", stacklevel=2)
  # the line after the phases names the columns; names pair with columns from the left
  (nameLine,), offset = linesFrom(data, offset, 1)
  names = [s.strip().lower() for s in text(nameLine).split("\t")]
  M = readTable(data, offset)
  at = dict(zip(names, M.T))
  props = _properties(at, ("phase", "x", "y", "euler1", "euler2", "euler3"), f"{fname} names no column {{}}")
  xs, ys, nx, ny = (header.get(k) for k in ("XStep", "YStep", "XCells", "YCells"))
  steps = isinstance(xs, float) and isinstance(ys, float) and xs > 0 and ys > 0
  unitCell = regularPoly(4, [xs, ys]) if steps else None
  grid = scanGrid(nx, ny, xs, ys) if steps and isinstance(nx, float) and isinstance(ny, float) else None
  meta = {"header": header, "surface": acq}
  # Channel always numbers from one with zero for a pixel that was not indexed
  return ebsdFromColumns(at["x"], at["y"], at["euler1"] * degree, at["euler2"] * degree, at["euler3"] * degree,
                         at["phase"], phases, props, eulerCorrection=eulerCorrection, scanUnit=scanUnit, frame=frame,
                         meta=meta, shift=0, unitCell=unitCell, scanGrid=grid)


__all__ = []
