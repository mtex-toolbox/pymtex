# the Oxford/Channel binary pair (MATLAB loadEBSD_crc, Julia crc.jl)
#
#   ebsd = loadEBSD_crc('map.cpr')     either name of the pair reads it
#   ebsd = loadEBSD_crc('map.crc')
#
# A .crc is one fixed size record per pixel and its .cpr sibling is the INI file that says
# what a record holds: [Fields] names the columns by the number Channel gives them, the
# number says the width, and [Job] says how many pixels there are and where they sit. So
# the record layout is data: a NumPy structured dtype built from [Fields], and the file is
# one np.fromfile, a column a strided view of it.
from __future__ import annotations

import os
import warnings

import numpy as np

from ..geometry.frames import crystalFrame, notIndexedFrame
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..maps.phaselist import phaseList
from .load import CHANNEL_LAUE, ebsdFromColumns, regularPoly
from .table import coerceNumeric, readFile, text

# the columns Channel numbers, and the width of each in bytes
CRC_COLUMNS = [("x", 4), ("y", 4), ("phi1", 4), ("phi", 4), ("phi2", 4), ("mad", 4), ("bc", 1), ("bs", 1),
               ("unknown", 1), ("bands", 1), ("error", 1), ("reliabilityindex", 4)]


def readIni(path):
  """an INI file as a section to key to value dict, names lowercased without blanks, the values
  numbers where they are, which is the cpr struct MATLAB keeps in ebsd.opt.header"""
  out = {}
  section = out.setdefault("", {})
  for line in readFile(path).split(b"\n"):
    s = text(line).strip()
    if not s:
      continue
    if s.startswith("["):
      section = out.setdefault(s.strip("[]").replace(" ", "").lower(), {})
    elif "=" in s:
      key, value = s.split("=", 1)
      section[key.replace(" ", "").lower()] = coerceNumeric(value)
  out.pop("", None)
  return out


def _number(ini, section, key, default=None):
  v = ini.get(section, {}).get(key)
  return v if isinstance(v, float) else default


def cprPhases(ini, path):
  """the phases the .cpr declares; a section that does not state its crystallography leaves its
  pixels not indexed rather than guessing a lattice for them"""
  frames = [notIndexedFrame()]
  n = int(_number(ini, "phases", "count", 0.0))
  missing = []
  for p in range(1, n + 1):
    sec = ini.get(f"phase{p}", {})
    abc = [_number(ini, f"phase{p}", k) for k in ("a", "b", "c")]
    abg = [_number(ini, f"phase{p}", k) for k in ("alpha", "beta", "gamma")]
    name = sec.get("structurename", "")
    if any(v is None for v in abc + abg) or not isinstance(name, str) or not name:
      frames.append(notIndexedFrame())
      missing.append(str(p))
      continue
    # the space group number is the sharper statement, the Laue code the fallback
    space = _number(ini, f"phase{p}", "spacegroup", 0.0)
    laue = sec.get("lauegroup")
    if space > 0:
      frames.append(crystalFrame(abc, np.array(abg) * degree, spaceId=int(space), mineral=name))
    elif isinstance(laue, float) and 1 <= laue <= len(CHANNEL_LAUE):
      frames.append(crystalFrame(CHANNEL_LAUE[int(laue) - 1], abc, np.array(abg) * degree, mineral=name))
    else:
      frames.append(crystalFrame(str(laue) if laue else "-1", abc, np.array(abg) * degree, mineral=name))
  if missing:
    warnings.warn(f"{path} announces {n} phases but does not describe phase {', '.join(missing)}; "
                  "pixels of that phase are imported as not indexed", stacklevel=3)
  return phaseList(frames)


def crcNames(ini, codes):
  """the names of the columns the codes stand for: the table above for the twelve Channel
  writes, the [EDX Windows] section for the windows above a hundred"""
  names = []
  for c in codes:
    if 1 <= c <= len(CRC_COLUMNS):
      names.append(CRC_COLUMNS[c - 1][0])
    else:
      w = ini.get("edxwindows", {}).get(f"window{c - 100}", "") if c > 100 else ""
      names.append(str(w).replace(" ", "").lower() if w else f"unknown{c}")
  return names


def crcAcquisition(ini):
  """the surface the map was acquired on, [Acquisition Surface] Euler1 to Euler3 in degree;
  a .crc keeps its angles in that surface's frame, CS1, so it is kept and not applied"""
  e = [_number(ini, "acquisitionsurface", f"euler{k}") for k in (1, 2, 3)]
  return None if any(v is None for v in e) else tuple(e)


def loadEBSD_crc(fname, *, eulerCorrection=None, frame=None, scanUnit="um"):
  """the Oxford/Channel map of a .cpr and .crc pair, by either name

    loadEBSD_crc('map.cpr')                        the Euler angles half a turn about z from the map
    loadEBSD_crc('map.crc', eulerCorrection=rot)   the alignment stated outright

  The header is kept in ebsd.meta['header'] as the sections of the .cpr, the acquisition
  surface in ebsd.meta['surface'].
  """
  base = os.path.splitext(os.fspath(fname))[0]
  cprPath, crcPath = base + ".cpr", base + ".crc"
  if not os.path.isfile(cprPath):
    raise ValueError(f"{crcPath} has no .cpr beside it to describe its columns")
  if not os.path.isfile(crcPath):
    raise ValueError(f"{cprPath} has no .crc beside it to hold its data")
  ini = readIni(cprPath)
  phases = cprPhases(ini, cprPath)
  # the columns the [Fields] section lists, in the order it lists them, as a record
  count = int(_number(ini, "fields", "count", 0.0))
  if count <= 0:
    raise ValueError(f"{cprPath} names no fields")
  codes = [int(_number(ini, "fields", f"field{k}", 0.0)) for k in range(1, count + 1)]
  names = crcNames(ini, codes)
  widths = [CRC_COLUMNS[c - 1][1] if 1 <= c <= len(CRC_COLUMNS) else 4 for c in codes]
  record = np.dtype([("phase", "i1")] + [(f"f{k}", "<f4" if w == 4 else "u1") for k, w in enumerate(widths)])
  # how many pixels, and where they sit when the file leaves the positions implicit
  nx, ny = _number(ini, "job", "xcells"), _number(ini, "job", "ycells")
  if nx is not None and ny is not None:
    n = int(nx) * int(ny)
  else:
    n = _number(ini, "job", "noofpoints")
    if n is None:
      raise ValueError(f"{cprPath} states neither its cell counts nor its number of points")
    n = int(n)
  size = os.path.getsize(crcPath)
  if size < n * record.itemsize:
    raise ValueError(f"{crcPath} holds {size} bytes, {n} records of {record.itemsize} bytes need {n * record.itemsize}")
  rec = np.fromfile(crcPath, dtype=record, count=n)
  cols = {name: rec[f"f{k}"] for k, name in enumerate(names)}
  dx = _number(ini, "job", "griddistx", 1.0)
  dy = _number(ini, "job", "griddisty", dx)
  if "x" in cols and "y" in cols:
    x, y = cols["x"], cols["y"]
  else:
    if nx is None:
      raise ValueError(f"{cprPath} states no positions and no grid to place them on")
    k = np.arange(n)
    x, y = dx * (k % int(nx)), dy * (k // int(nx))
  for key in ("phi1", "phi", "phi2"):
    if key not in cols:
      raise ValueError(f"{cprPath} names no Euler angles among its fields")
  props = {name: cols[name] for name in names if name not in ("x", "y", "phi1", "phi", "phi2")}
  meta = {"header": ini, "surface": crcAcquisition(ini)}
  if eulerCorrection is None:
    eulerCorrection = rotation.byEuler(np.pi, 0.0, 0.0, "Bunge")
  # Channel numbers its phases from one and writes zero where nothing was indexed
  return ebsdFromColumns(x, y, cols["phi1"], cols["phi"], cols["phi2"], rec["phase"], phases, props, shift=0,
                         eulerCorrection=eulerCorrection, scanUnit=scanUnit, frame=frame, meta=meta,
                         unitCell=regularPoly(4, [dx, dy]) if dx > 0 and dy > 0 else None,
                         scanGrid=(nx, ny, dx, dy) if nx is not None and ny is not None else None, propType=np.float64)


__all__ = ["loadEBSD_crc"]
