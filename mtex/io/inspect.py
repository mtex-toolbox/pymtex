# what an EBSD file says about itself before its data is read (Julia inspect.jl; the import
# wizard's preview, docs/plans/wizard.md W1)
#
#   info = inspectEBSD('map.ctf')      the phases, the properties, the grid, the Euler correction
#   info.describe()                    the rows of the file information panel
#   ebsdFiles(folder)                  the maps a folder holds
#
# A header is not a map: a text file is read in chunks until its header ends, a binary one
# until its header marker, an HDF5 file through its metadata, so describing a 200 MB file
# costs the same as describing a small one.
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..maps.phaselist import phaseList
from ..geometry.display import aligned
from .load import VENDOR_FORMATS

EBSD_EXTENSIONS = tuple("." + ext for ext in VENDOR_FORMATS)
CHUNK = 1 << 18
HEAD_LIMIT = 1 << 26


@dataclass
class EBSDFileInfo:
  """what an EBSD file says about itself from its header alone: format, vendor, phases,
  properties, data sets, grid, step and Euler correction, as inspectEBSD returns it"""
  path: str
  format: str
  vendor: str
  size: int
  date: datetime | None
  phases: phaseList
  properties: list = field(default_factory=list)
  dataSets: list = field(default_factory=list)
  points: int | None = None
  grid: tuple | None = None
  hex: bool = False
  step: tuple | None = None
  eulerCorrection: rotation = None
  header: dict = field(default_factory=dict)

  @property
  def name(self):
    """the file name without its folder"""
    return os.path.basename(self.path)

  @property
  def extent(self):
    """the width and height the grid spans, in the scan unit"""
    if self.grid is None or self.step is None:
      return None
    nx, ny = self.grid
    dx, dy = self.step
    return ((nx - 1) * dx, (ny - 1) * dy)

  @property
  def eulerAxesNames(self):
    """the axis names the vendor writes its Euler angles in"""
    labels = _find(self.header, ("sample surface direction labels", "direction labels"))
    if isinstance(labels, (list, tuple)) and len(labels) == 3:
      return tuple(str(l) for l in labels)
    if self.vendor.startswith("EDAX"):
      return ("A1", "A2", "A3")
    if self.vendor == "Oxford":
      return ("X1", "Y1", "Z1")
    return ("x", "y", "z")

  def describe(self):
    """the rows of the file information panel, name and value"""
    rows = [("file", self.name), ("format", self.format), ("vendor", self.vendor), ("size", _sizeLabel(self.size))]
    if self.date is not None:
      rows.append(("date", self.date.strftime("%Y-%m-%d %H:%M")))
    if self.points is not None:
      rows.append(("points", f"{self.points:,}".replace(",", " ")))
    if self.grid is not None:
      nx, ny = self.grid
      kind = "hex grid" if self.hex else "square grid"
      stepText = ""
      if self.step is not None:
        dx, dy = self.step
        stepText = f", d = {dx:.4g} µm" if self.hex or abs(dx - dy) < 1e-9 * max(dx, dy, 1e-12) else f", dx = {dx:.4g}, dy = {dy:.4g} µm"
      rows.append(("grid", f"{kind} {nx} x {ny}{stepText}"))
    if self.extent is not None:
      w, h = self.extent
      rows.append(("extent", f"{_length(w)} x {_length(h)}"))
    if len(self.phases) and any(f.isIndexed for f in self.phases.frames):
      rows.append(("phases", ", ".join(f.name for f in self.phases.frames if f.isIndexed)))
    if self.properties:
      rows.append(("properties", ", ".join(self.properties)))
    if len(self.dataSets) > 1:
      rows.append(("data sets", str(len(self.dataSets))))
    if self.eulerCorrection is not None:
      rows.append(("Euler correction", _rotationLabel(self.eulerCorrection)))
    return rows

  def __repr__(self):
    w = max(len(k) for k, _ in self.describe())
    return aligned("EBSD file\n" + "\n".join(f"  {k.ljust(w)}  {v}" for k, v in self.describe()))


# ------------------------------------------------------------------ the files of a folder

def isEBSDFile(name):
  """whether a name is an EBSD file by its extension; a .crc beside its .cpr is the same map"""
  return os.path.splitext(name)[1].lower() in EBSD_EXTENSIONS


def ebsdFiles(folder):
  """the EBSD files of a folder by their extension, sorted, a .crc beside its .cpr left out"""
  names = sorted(n for n in os.listdir(folder) if isEBSDFile(n) and os.path.isfile(os.path.join(folder, n)))
  stems = {os.path.splitext(n)[0] for n in names if n.lower().endswith(".cpr")}
  return [n for n in names if not (n.lower().endswith(".crc") and os.path.splitext(n)[0] in stems)]


# ------------------------------------------------------------------ the description

def inspectEBSD(path):
  """what an EBSD file says about itself, read from its header alone

    inspectEBSD('map.ang')     the phases, the properties, the grid and the step, the Euler correction
    inspectEBSD('map.h5oina')  and the data sets a project file holds
  """
  path = os.fspath(path)
  inspect = _INSPECTORS.get(VENDOR_FORMATS.get(os.path.splitext(path)[1].lower().lstrip(".")))
  if inspect is None:
    raise ValueError(f"{path} has no extension of an EBSD file; they are {', '.join(EBSD_EXTENSIONS)}")
  info = inspect(path)
  st = os.stat(path)
  info.size = st.st_size
  if info.date is None:
    info.date = datetime.fromtimestamp(st.st_mtime)
  return info


# the head of a file up to the first line the predicate accepts, read in chunks, with the
# offset of that line and the bytes read so far
def _headLines(path, stop):
  from .table import leadingLines
  buf = b""
  with open(path, "rb") as f:
    while len(buf) < HEAD_LIMIT:
      chunk = f.read(CHUNK)
      buf += chunk
      lines, pos = leadingLines(buf, stop)
      if pos < len(buf) or not chunk:
        return lines, pos, buf
  raise ValueError(f"the header of {path} does not end within {HEAD_LIMIT >> 20} MB")


def _inspectAng(path):
  from .ang import _angColumnNames, angHeader, angIsACOM, angIsEMSphInx, angPhases
  from .table import linesFrom, text
  head, offset, buf = _headLines(path, lambda l: not l.startswith(b"#"))
  header = [text(l) for l in head]
  phases, names, families = angPhases(header)
  acom = angIsACOM(header)
  if acom:
    vendor = "ACOM"
  else:
    vendor = "EMSphInx" if angIsEMSphInx(header, names, families) else "EDAX"
  h = angHeader(header)
  (first,), _ = linesFrom(buf, offset, 1)
  cols = [c for c in _angColumnNames(len(text(first).split())) if c not in ("phi1", "Phi", "phi2", "x", "y", "phase")]
  xs, ys, nOdd, nEven, nRows = (h.get(k) for k in ("XSTEP", "YSTEP", "NCOLS_ODD", "NCOLS_EVEN", "NROWS"))
  hexGrid = str(h.get("GRID", "")).lower() == "hexgrid"
  info = EBSDFileInfo(path, "ang", vendor, 0, _headerDate(h), phases, cols, header=h)
  if all(isinstance(v, float) for v in (nOdd, nRows)):
    info.grid = (int(nOdd), int(nRows))
    info.hex = hexGrid
    if hexGrid and isinstance(nEven, float):
      odd, even = (int(nRows) + 1) // 2, int(nRows) // 2
      info.points = odd * int(nOdd) + even * int(nEven)
    else:
      info.points = int(nOdd) * int(nRows)
  if isinstance(xs, float) and isinstance(ys, float):
    info.step = (xs, ys)
  from .ang import eulerCorrectionRotation
  info.eulerCorrection = eulerCorrectionRotation(4 if acom else 2)
  return info


def _inspectCtf(path):
  from .ctf import _ctfHead, oxfordCorrection
  from .table import linesFrom, text
  _, _, buf = _headLines(path, lambda l: l.startswith(b"Phases"))
  h, phases, acq, offset = _ctfHead(buf, path)
  (nameLine,), _ = linesFrom(buf, offset, 1)
  names = [s.strip().lower() for s in text(nameLine).split("\t")]
  cols = [c for c in names if c not in ("phase", "x", "y", "euler1", "euler2", "euler3")]
  info = EBSDFileInfo(path, "ctf", "Oxford", 0, _headerDate(h), phases, cols, header=h, eulerCorrection=oxfordCorrection(acq))
  xs, ys, nx, ny = (h.get(k) for k in ("XStep", "YStep", "XCells", "YCells"))
  if isinstance(nx, float) and isinstance(ny, float):
    info.grid, info.points = (int(nx), int(ny)), int(nx) * int(ny)
  if isinstance(xs, float) and isinstance(ys, float) and xs > 0 and ys > 0:
    info.step = (xs, ys)
  return info


def _inspectCpr(path):
  from .crc import _number, cprPhases, crcNames, readIni
  from .ctf import oxfordCorrection
  base = os.path.splitext(path)[0]
  cprPath = base + ".cpr"
  if not os.path.isfile(cprPath):
    raise ValueError(f"{path} has no .cpr beside it to describe its columns")
  ini = readIni(cprPath)
  phases = cprPhases(ini, cprPath)
  count = int(_number(ini, "fields", "count", 0.0))
  codes = [int(_number(ini, "fields", f"field{k}", 0.0)) for k in range(1, count + 1)]
  cols = [c for c in crcNames(ini, codes) if c not in ("x", "y", "phi1", "phi", "phi2")]
  job = ini.get("job", {})
  info = EBSDFileInfo(path, "cpr", "Oxford", 0, _headerDate(ini), phases, cols, header=ini,
                      eulerCorrection=oxfordCorrection(None))
  nx, ny = job.get("xcells"), job.get("ycells")
  if isinstance(nx, float) and isinstance(ny, float):
    info.grid, info.points = (int(nx), int(ny)), int(nx) * int(ny)
  elif isinstance(job.get("noofpoints"), float):
    info.points = int(job["noofpoints"])
  dx = job.get("griddistx")
  if isinstance(dx, float) and dx > 0:
    dy = job.get("griddisty")
    info.step = (dx, dy if isinstance(dy, float) and dy > 0 else dx)
  return info


def _inspectOsc(path):
  from .ang import eulerCorrectionRotation
  from .osc import OSC_HEADER_CLOSE, _oscColumnNames, _oscLayout, oscHeader
  buf = b""
  with open(path, "rb") as f:
    while len(buf) < HEAD_LIMIT:
      chunk = f.read(CHUNK)
      buf += chunk
      close = buf.find(OSC_HEADER_CLOSE)
      if close >= 0 and len(buf) >= close + 4096 or not chunk:
        break
  nx, ny, n, phases, start, xstep, ystep, m = _oscLayout(buf, path)
  cols = [c for c in _oscColumnNames(m) if c not in ("x", "y", "phi1", "phi", "phi2", "phase")]
  h = oscHeader(buf, nx, ny, n, xstep, ystep)
  info = EBSDFileInfo(path, "osc", "EDAX", 0, None, phases, cols, points=int(n), header=h,
                      eulerCorrection=eulerCorrectionRotation(2))
  if nx >= 1 and ny >= 1:
    info.grid = (int(nx), int(ny))
    info.hex = abs(xstep - ystep) > 1e-6 * max(xstep, ystep)
  if xstep > 0 and ystep > 0:
    info.step = (float(xstep), float(ystep))
  return info


def _inspectH5(path):
  import h5py
  from .h5 import _eulerEntry, _floats, h5Reader, loadedField
  with h5py.File(path, "r") as f:
    r = h5Reader(f, path)
    r.choose()
    ebsd, groups = r.ebsd, r.groups
    phases, _ = r.phases()
    header = r.header()
    info = EBSDFileInfo(path, os.path.splitext(path)[1].lstrip("."), r.name, 0, _headerDate(header), phases, header=header)
    info.dataSets = [p.strip("/") for p in r.sets]
    # the count off the shape of a data set, the grid off the header, the step off the config;
    # what the loader reads as the map is not a property
    pos = ebsd["position"]
    rot = ebsd["rotation"]
    eu = _eulerEntry(rot)
    direct = pos.get("type", "direct") == "direct"
    entries = [ebsd["phase"], *([pos["direct"]["x"], pos["direct"]["y"]] if direct else []),
               *(eu[k] for k in ("phi", "phi1", "Phi", "phi2") if k in eu)]
    # every entry is remembered, so that none of them is taken for a property
    for entry in entries:
      p = r.remember(entry)
      if p is None or p in groups or info.points is not None:
        continue
      d = f[p]
      # a data set of the three angles as rows of three counts its rows
      stacked = eu.get("phi") is entry and d.ndim > 1 and d.shape[-1] == 3
      info.points = int(d.shape[0]) if stacked else int(np.prod(d.shape))
    if not direct:
      ind = pos["indirect"]
      nx, ny = r.value(ind["grid_size_x"]), r.value(ind.get("grid_size_y"))
      if nx is not None and ny is not None:
        info.grid = (int(_floats(nx)[0]), int(_floats(ny)[0]))
        if info.points is None:
          info.points = info.grid[0] * info.grid[1]
      dx = r.number(ind["step_size_x"])
      if dx is not None:
        info.step = (dx, r.number(ind["step_size_y"], dx))
    if info.step is None:
      sx, sy = r.cellSteps()
      if sx is not None and sy is not None:
        info.step = (sx, sy)
    if info.grid is None:
      nx, ny = _find(header, ("ncols", "ncolumns", "x cells", "xcells")), _find(header, ("nrows", "y cells", "ycells"))
      if _isNumber(nx) and _isNumber(ny):
        info.grid = (int(_floats(nx)[0]), int(_floats(ny)[0]))
        if info.points is None:
          info.points = info.grid[0] * info.grid[1]
    # the properties: every per pixel data set of the additions groups and the fields of a table
    if info.points:
      info.properties += [k for k in r.additions(info.points) if k not in info.properties]
    table = ebsd.get("ebsd_data")
    p = None if table is None else r.one(table)
    if p is not None and f[p].dtype.names:
      taken = (loadedField(rot), loadedField(ebsd["phase"]))
      info.properties += [k.replace(" ", "").lower() for k in f[p].dtype.names if k not in taken and f[p].dtype[k].ndim == 0]
      if info.points is None:
        info.points = int(np.prod(f[p].shape))
    info.eulerCorrection = r.correction()
  return info


# the description of a file by the module that reads its format
_INSPECTORS = {"ang": _inspectAng, "ctf": _inspectCtf, "crc": _inspectCpr, "osc": _inspectOsc, "h5": _inspectH5}


# ------------------------------------------------------------------ the labels

# a value of a nested header found by the tail of its key, case and blanks ignored
def _find(h, keys):
  if not isinstance(h, dict):
    return None
  wanted = {key.replace(" ", "") for key in keys}
  for k, v in h.items():
    if str(k).replace(" ", "").lower() in wanted:
      return v
  for v in h.values():
    if isinstance(v, dict):
      hit = _find(v, keys)
      if hit is not None:
        return hit
  return None


def _isNumber(v):
  try:
    return v is not None and np.asarray(v, dtype=float).size >= 1
  except (TypeError, ValueError):
    return False


# the date a header states, where it states one
def _headerDate(h):
  v = _find(h, ("date", "acquisition date", "acquisitiondate", "creationdate", "created"))
  if isinstance(v, (bytes, np.bytes_)):
    v = v.decode("ascii", errors="replace")
  if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 1:
    v = v[0]
  if not isinstance(v, str):
    return None
  for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
              "%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y"):
    try:
      return datetime.strptime(v.strip()[:19], fmt)
    except ValueError:
      continue
  return None


def _sizeLabel(n):
  for unit, div in (("GB", 1 << 30), ("MB", 1 << 20), ("kB", 1 << 10)):
    if n >= div:
      return f"{n / div:.1f} {unit}"
  return f"{n} B"


# a length in µm with its unit chosen
def _length(v):
  if v >= 1e3:
    return f"{v / 1e3:.4g} mm"
  return f"{v:.4g} µm"


# '180° about z' or the Euler angles of a general correction
def _rotationLabel(r):
  a = float(r.angle())
  if a < 1e-9:
    return "none"
  ax = r.axis().data.reshape(3)
  named = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
  for n, e in named.items():
    if abs(abs(float(np.dot(ax, e))) - 1) < 1e-6:
      return f"{a / degree:.4g}° about {'-' if np.dot(ax, e) < 0 else ''}{n}"
  # the EDAX axis (1, -1, 0) and its like: the axis scaled to its largest component
  v = ax / np.max(np.abs(ax))
  if np.all(np.abs(v - np.round(v)) < 1e-6):
    return f"{a / degree:.4g}° about ({','.join(str(int(round(x))) for x in v)})"
  e = r.Euler("Bunge")
  return "Bunge (" + ", ".join(f"{float(x) / degree:.4g}°" for x in e) + ")"


__all__ = ["EBSDFileInfo", "inspectEBSD", "ebsdFiles", "isEBSDFile"]
