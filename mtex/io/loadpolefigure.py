# reading pole figures from the files the diffractometers write (MATLAB PoleFigure.load,
# loadPoleFigure_generic, interfaces/loadPoleFigure_*.m; docs/plans/io.md I6)
#
#   pf = loadPoleFigure('Q(10-10)_amp.cnv', cs)                 the format read off the file
#   pf = loadPoleFigure([f1, f2], [h1, h23], cs, superposition=[1, [0.52, 1.23]])
#   pf = loadPoleFigure('scan.txt', h, cs, columnNames=['polar angle', 'azimuth angle', 'intensity'], header=21)
#   pf = loadPoleFigure('scan.dat', cs, interface='geesthacht')
#
# A file names its reflection in its name or its header; the crystal direction is read off
# it when none is given, the crystal frame being the one passed. The format is chosen, not
# searched for: the extension names the candidates and the marker of the header decides
# between them (chooseFormat below); the readers of the formats this file does not hold
# are polefigureformats.py. Every reader returns the pole figures of one file;
# loadPoleFigure joins the files and puts the directions and the structure coefficients
# given on the result.
from __future__ import annotations

import os
import re

import numpy as np

from ..functions.polefigure import PoleFigure
from ..geometry.frames import crystalFrame, referenceFrame
from ..geometry.grids.s2grid import regularS2Grid
from ..geometry.miller import Miller
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from . import polefigureformats as F
from .table import _tableStart, isNumericRow, readFile, readTable, text
from ..geometry.install import install

# ------------------------------------------------------------------ the reflection in a name

def string2Miller(s, CS=None):
  """the Miller indices a file name or a header line spells: four digits with optional signs,
  else three; every such group of the name is one direction, several a superposition;
  MATLAB's string2Miller

    h = string2Miller('Q(10-11)_amp.cnv', cs)          # {10-11}
    h = string2Miller('Q(10-11)(01-11)_amp.cnv', cs)   # two directions, a superposition
    h = string2Miller('PF_002_R.UXD')                  # {002} of the cubic frame

  Only the base name of a path is read. Four digits from 0 to 6 with optional minus signs are
  taken when the first three sum to zero, else three digits without a sign, else three with
  signs; a name with none is (100). MATLAB's second output, whether the name spelled one, is
  not returned.

  Input
    s - a file name or a line of text
  Output
    h - Miller, one direction or several
  Keywords
    CS - the crystal frame of the directions, m-3m when not given
  See also
    loadPoleFigure, Miller"""
  CS = crystalFrame("m-3m") if CS is None else CS
  s = os.path.basename(str(s))
  # four indices with signs first, then three without a sign, then three with
  for n, signed in ((4, True), (3, False), (3, True)):
    pattern = r"(?:(?:-?\s?[0-6]\s?){%d})" if signed else r"(?:(?:[0-6]\s?){%d})"
    groups = re.findall(pattern % n, s)
    found = []
    for g in groups:
      digits = [int(d) for d in re.findall(r"-?[0-6]" if signed else r"[0-6]", g)]
      if len(digits) == n and (n == 3 or sum(digits[:3]) == 0):
        found.append(digits)
    if found:
      return vector3d.cat(*[Miller(*d, CS) for d in found]) if len(found) > 1 else Miller(*found[0], CS)
  return Miller(1, 0, 0, CS)


# the frames among those given that are not None
def _given(*frames):
  return [f for f in frames if f is not None]


def _frames(args):
  cs = next((a for a in args if isinstance(a, crystalFrame)), None)
  ss = next((a for a in args if isinstance(a, referenceFrame) and not isinstance(a, crystalFrame)), None)
  return cs, ss


# the lines of a file, whichever of the three line ends it writes
def _lines(fname):
  return [text(l) for l in re.split(rb"\r\n|\r|\n", readFile(fname))]


# ------------------------------------------------------------------ the formats

def loadPoleFigure_dubna(fname, *args, **kwargs):
  """the .cnv and .cns files of the Dubna neutron goniometer: 72 azimuths per polar ring,
  the rings of a .cnv starting at the goniometer's own azimuths"""
  cs, ss = _frames(args)
  ext = os.path.splitext(fname)[1].lower()
  if ext not in (".cnv", ".cns"):
    raise ValueError(f"{fname} is not a Dubna file")
  values = np.array([float(x) for x in text(readFile(fname)).split()])
  if values.size % 72 or not 3 * 72 <= values.size <= 40 * 72:
    raise ValueError(f"{fname} does not hold rings of 72 intensities")
  d = values.reshape(-1, 72).T
  h = string2Miller(fname, cs)
  if ext == ".cnv":
    r = DubnaGrid(d.shape[1])
  else:
    r = regularS2Grid(points=d.shape, maxTheta=5 * degree * (d.shape[1] - 1), antipodal=True)
  return PoleFigure(h, r, d, *_given(cs, ss), **kwargs)


def DubnaGrid(ntheta):
  """the specimen grid of the Dubna goniometer: 19 rings five degrees apart, each ring of 72
  azimuths five degrees apart, the last at 355 degree, starting where the instrument starts it"""
  theta = np.linspace(0, np.pi / 2, 19)[:ntheta]
  rho = np.arange(72) * 5 * degree
  start = np.array([360.00, 336.40, 327.05, 320.11, 314.44, 309.57, 305.26, 301.37, 297.80, 294.47, 291.34, 288.38,
                    285.54, 282.81, 280.16, 277.57, 275.02, 272.50, 270.00])[::-1][:ntheta] * degree
  T, R = np.meshgrid(theta, rho, indexing="xy")
  return vector3d.byPolar(T, R + start[None, :], antipodal=True)


def loadPoleFigure_uxd(fname, *args, **kwargs):
  """the Bruker .uxd text: one block per tilt with its _KEY=value header and the azimuths
  and counts below, a new pole figure where the Bragg angle changes"""
  cs, ss = _frames(args)
  blocks, header, data = [], {}, []
  for line in _lines(fname):
    s = line.strip()
    if not s or s.startswith(";"):
      continue
    if s.startswith("_"):
      if data:
        blocks.append((header, np.array(data)))
        header, data = {}, []
      key, _, value = s[1:].partition("=")
      header[key.strip()] = value.strip().strip("'")
    elif isNumericRow(s.encode()):
      data.extend(float(x) for x in s.split())
  if data:
    blocks.append((header, np.array(data)))
  if not blocks:
    raise ValueError(f"{fname} holds no _KHI block")
  allH, allR, allI = [], [], []
  lastTheta, r, d = None, [], []
  for header, values in blocks:
    th = header.get("THETA")
    if th is None or not 0 < float(th) < 90:
      raise ValueError(f"{fname} has a block without a Bragg angle")
    if th != lastTheta:
      if lastTheta is not None:
        allH.append(string2Miller(fname, cs))
        allR.append(vector3d.cat(*r))
        allI.append(np.concatenate(d))
      lastTheta, r, d = th, [], []
    theta = float(header["KHI"]) * degree
    if not 0 <= theta <= np.pi / 2:
      raise ValueError(f"{fname} has a tilt out of range")
    # azimuths given in the first column when they step evenly, else equispaced
    if values.size % 2 == 0 and values.size >= 4 and np.allclose(np.diff(values[0::2]), values[2] - values[0]):
      rho, counts = values[0::2] * degree, values[1::2]
    else:
      rho, counts = np.arange(values.size) * 2 * np.pi / values.size, values
    r.append(vector3d.byPolar(np.full(rho.size, theta), rho))
    d.append(counts)
  allH.append(string2Miller(fname, cs))
  allR.append(vector3d.cat(*r))
  allI.append(np.concatenate(d))
  return PoleFigure(allH, allR, allI, *_given(cs, ss), **kwargs)


def loadPoleFigure_aachen_exp(fname, *args, **kwargs):
  """the AUTEX experimental pole figure file of Aachen: the grid on line four, then one
  HKL block per pole figure, one row of azimuths per tilt"""
  cs, ss = _frames(args)
  lines = _lines(fname)
  if not lines or not lines[0].startswith("AUTEX"):
    raise ValueError(f"{fname} is not an AUTEX file")
  m = re.match(r"\s*ALPHASTART\s+([\d.]+)\s+END\s+([\d.]+)\s+DELTA\s+([\d.]+)\s+BETASTART\s+([\d.]+)\s+DELTA\s+([\d.]+)", lines[3])
  if m is None:
    raise ValueError(f"{fname} does not state its grid")
  a0, a1, da, b0, db = (float(x) for x in m.groups())
  theta = np.arange(a0, a1 + da / 2, da) * degree
  rho = np.arange(b0, 360 - db + db / 2, db) * degree
  r = regularS2Grid(theta=theta, rho=rho, antipodal=True)
  allH, allI = [], []
  k = 4
  while k < len(lines):
    if lines[k].startswith("HKL"):
      hkl = re.search(r"KL\s*([0-9_-]*)\.?\s*THETA", lines[k])
      allH.append(string2Miller(hkl.group(1), cs))
      # every ring closes with its tilt marker, -05.00, which is no intensity
      tokens = []
      k += 1
      while k < len(lines) and not lines[k].startswith("HKL"):
        tokens.extend(lines[k].split())
        k += 1
      values, pos = [], 0
      for _ in range(theta.size):
        values.extend(float(x) for x in tokens[pos:pos + rho.size])
        pos += rho.size
        if pos < len(tokens) and tokens[pos].startswith("-"):
          pos += 1
      allI.append(np.array(values).reshape(theta.size, rho.size).T)
    else:
      k += 1
  if not allH:
    raise ValueError(f"{fname} holds no HKL block")
  return PoleFigure(allH, [r] * len(allH), allI, *_given(cs, ss), **kwargs)


def loadPoleFigure_rw1(fname, *args, **kwargs):
  """the binary .rw1 export of a Philips X'Pert: a text description naming the reflection,
  then 16 bit counts on a five degree grid"""
  cs, ss = _frames(args)
  if os.path.splitext(fname)[1].lower() != ".rw1":
    raise ValueError(f"{fname} is not a .rw1 file")
  raw = readFile(fname)
  description = raw[:54].decode("latin-1")
  data = np.frombuffer(raw[54 + 352:], dtype="<i2").astype(float)
  m = re.search(r"hkl:(\w*) (\w*) (\w*)", description)
  h = string2Miller("".join(m.groups()), cs) if m else Miller(1, 0, 0, F._frame(cs))
  if data.size % 72 == 0:
    d = data.reshape(-1, 72).T[:, :17]
    r = regularS2Grid(theta=np.arange(0, 81, 5) * degree, rho=np.arange(2.5, 360, 5) * degree)
  elif data.size % 73 == 0:
    d = data.reshape(-1, 73).T
    r = regularS2Grid(theta=np.arange(d.shape[1]) * 5 * degree, rho=np.arange(0, 361, 5) * degree, antipodal=True)
  else:
    raise ValueError(f"{fname} holds {data.size} counts, no five degree grid")
  return PoleFigure(h, r, d ** 2 / 500, *_given(cs, ss), **kwargs)


# the names a column may carry, MATLAB's loadHelper
_COLUMNS = {"polar angle": "theta", "polarangle": "theta", "theta": "theta", "colatitude": "theta",
            "azimuth angle": "rho", "azimuthangle": "rho", "azimuth": "rho", "rho": "rho", "phi": "rho",
            "intensity": "intensity", "intensities": "intensity", "counts": "intensity", "x": "x", "y": "y", "z": "z"}


# a text file of one pole figure: rows of polar angle, azimuth and intensity, or of x, y,
# z and intensity, the columns named by columnNames and picked by columns, counted from
# zero; a block of more than fifteen columns is read as intensities on a regular grid
#
#   loadPoleFigure('seifert-111.nja', cs, header=21, columnNames=['polar angle', 'azimuth angle', 'intensity'])
def _loadPoleFigure_generic(fname, *args, columnNames=None, columns=None, radians=False, delimiter=None, header=None,
                            superposition=None, complete=False):
  cs, ss = _frames(args)
  h = next((a for a in args if isinstance(a, vector3d)), None)
  data = readFile(fname)
  _, offset = _tableStart(data, header, delimiter)
  if offset is None or offset >= len(data):
    raise ValueError(f"{fname} holds no row of numbers")
  M = readTable(data, offset, delimiter=delimiter)
  frames = _given(cs, ss)
  if M.shape[1] > 15:
    if M.shape[0] % 2 and np.array_equal(M[0], M[-1]):
      M = M[:-1]
    r = regularS2Grid(points=M.shape, antipodal=not complete)
    return PoleFigure(string2Miller(fname, cs) if h is None else h, r, M, *frames, superposition=superposition, complete=complete)
  if columnNames is None:
    raise ValueError(f"{fname} needs its columns named, columnNames=['polar angle', 'azimuth angle', 'intensity']")
  names = [_COLUMNS.get(str(n).strip().lower(), None) for n in columnNames]
  if columns is not None:
    M = M[:, list(columns)]
  if M.shape[1] < len(names):
    raise ValueError(f"{fname} has {M.shape[1]} columns, the names given are {len(names)}")
  at = {n: M[:, k] for k, n in enumerate(names) if n is not None}
  if "intensity" not in at:
    raise ValueError("a pole figure file names an intensity column")
  if "theta" in at and "rho" in at:
    s = 1.0 if radians else degree
    r = vector3d.byPolar(at["theta"] * s, at["rho"] * s)
  elif all(k in at for k in ("x", "y", "z")):
    r = vector3d(at["x"], at["y"], at["z"])
  else:
    raise ValueError("a pole figure file names its specimen directions by polar angle and azimuth angle, or by x, y, z")
  if r.size < 5:
    raise ValueError(f"{fname} holds too few measurements")
  return PoleFigure(string2Miller(fname, cs) if h is None else h, r, at["intensity"], *frames, superposition=superposition, complete=complete)


# ------------------------------------------------------------------ which reader a file gets
#
# The extension narrows and a signature decides: `.epf` is written by Popla and by LaboTEX,
# `.txt` by Philips, Aachen and two Rigaku programs, `.dat` by Siemens, Geesthacht and a
# Rigaku raster, so every format states the marker its own header carries. MATLAB instead
# tries its readers in turn and keeps the first that does not throw, which reads a broken
# file of one format as some other format. A file of a known extension that no candidate
# claims, or that several claim, is an error naming them, and interface= settles it; an
# extension no format writes is a table whose columns the caller names, as for EBSD.load.


def isPopla(lines):
  """whether the lines are those of a Popla file: Miller indices and four numbers in five character
  fields on the second line"""
  s = lines[1] if len(lines) > 1 else ""
  return len(s) >= 45 and F.millerIndices(s[:5]) is not None and all(F._float(s[5 * k:5 * k + 5]) is not None for k in range(1, 5))


SIGNATURES = {
  "popla": isPopla,
  "labotex": lambda lines: any(l.strip().upper().startswith("STRUCTURE CODE") for l in lines[:6]),
  "aachen": lambda lines: any(F.isAachenHeader(l) for l in lines[:4]),
  "philips": lambda lines: any(l.strip().startswith("Psi:") for l in lines),
  "siemens": lambda lines: any("*Pole figure:" in l for l in lines),
  "uxd": lambda lines: any(l.strip().startswith("_KHI") for l in lines),
  "juelich": lambda lines: len(lines) > 1 and re.search(r"alpha\s+beta\s+intensity", lines[1], re.I) is not None,
  "geesthacht": lambda lines: bool(lines) and "CREATOR" in lines[0],
  "nja": lambda lines: any(l.strip().startswith("#Values") for l in lines),
  "rigaku": lambda lines: bool(lines) and lines[0].strip().startswith("#") and any("axis type" in l or "number of data" in l for l in lines),
  "rigaku_dat": lambda lines: bool(lines) and [f.lower() for f in lines[0].split()] == ["x", "y", "z", "data"],
  "rigaku_txt": lambda lines: any(l.startswith("ScanningMode") for l in lines) and
                any(F.firstToken(l.replace(",", " ")).lower() == "start" for l in lines),
}


# a reader of the vendor module as one of the table: the pole figures of one file
def _vendor(read, binary=False):
  def load(fname, lines, CS, SS, h=None, **kwargs):
    extra = {"raw": readFile(fname)} if binary else {}
    hs, rs, Is, fr = read(lines, fname, CS, h=h, **extra, **kwargs)
    return PoleFigure(hs, rs, Is, *_given(fr, SS))
  return load


# a reader of this module as one of the table
def _own(read):
  def load(fname, lines, CS, SS, h=None, **kwargs):
    return read(fname, *_given(CS, SS), **kwargs)
  return load


# the name MATLAB gives the interface, the extensions it is written with and its reader
FORMATS = {
  "dubna": ((".cnv", ".cns"), _own(loadPoleFigure_dubna)),
  "ana": ((".ana", ".cor"), _vendor(F.readAna)),
  "popla": ((".epf", ".gpf"), _vendor(F.readPopla)),
  "labotex": ((".epf", ".pow", ".ppf"), _vendor(F.readLabotex)),
  "aachen": ((".txt", ".cpf", ""), _vendor(F.readAachen)),
  "aachen_exp": ((".exp",), _own(loadPoleFigure_aachen_exp)),
  "philips": ((".txt",), _vendor(F.readPhilips)),
  "plf": ((".plf",), _vendor(F.readPlf)),
  "ibm": ((".ibm",), _vendor(F.readIbm)),
  "out1": ((".out",), _vendor(F.readOut1)),
  "slc": ((".slc",), _vendor(F.readSlc)),
  "beartex": ((".xpa", ".xpe"), _vendor(F.readBeartex)),
  "xrd": ((".asc", ".ras"), _vendor(F.readXrd)),
  "siemens": ((".dat", ".ptx", ".rpf"), _vendor(F.readSiemens)),
  "uxd": ((".uxd",), _own(loadPoleFigure_uxd)),
  "juelich": ((".jul", ""), _vendor(F.readJuelich)),
  "geesthacht": ((".dat",), _vendor(F.readGeesthacht)),
  "nja": ((".nja",), _vendor(F.readNja)),
  "pwd": ((".pwd",), _vendor(F.readPwd)),
  "rigaku": ((".txt",), _vendor(F.readRigaku)),
  "rigaku_dat": ((".dat",), _vendor(F.readRigakuDat)),
  "rigaku_txt": ((".txt",), _vendor(F.readRigakuTxt)),
  "gpol": ((".gpol",), _vendor(F.readGpol, binary=True)),
  "rw1": ((".rw1",), _own(loadPoleFigure_rw1)),
  "xrdml": ((".xrdml", ".xml"), _vendor(F.readXrdml, binary=True)),
}
INTERFACES = tuple(FORMATS) + ("generic",)
BINARY = {".gpol", ".rw1", ".xrdml", ".xml"}
# the options of a table, which name the generic reader whatever the extension
TABLE_KEYS = {"columnNames", "columns", "header", "delimiter", "radians"}


def chooseFormat(fname, interface=None):
  """the format of a file: the one named, else the one candidate of its extension whose
  signature the file carries; None for an extension no format writes"""
  if interface is not None:
    name = str(interface).lower()
    if name not in INTERFACES:
      raise ValueError(f"no pole figure interface {interface!r}; the interfaces are {', '.join(INTERFACES)}")
    return name
  ext = os.path.splitext(fname)[1].lower()
  candidates = [n for n, (exts, _) in FORMATS.items() if ext in exts]
  if not candidates:
    return None
  if len(candidates) == 1 and candidates[0] not in SIGNATURES:
    return candidates[0]
  lines = [] if ext in BINARY else _lines(fname)
  claimed = [n for n in candidates if SIGNATURES.get(n, lambda _: True)(lines)]
  if len(claimed) == 1:
    return claimed[0]
  # a file no format claims that is a plain block of numbers is a table
  if not claimed and F.isNumericRow(next((l for l in lines if l.strip()), "")):
    return None
  if not claimed:
    raise ValueError(f"{fname} carries the marker of none of the {ext!r} formats {', '.join(candidates)}; "
                     f"pass interface=..., or columnNames=[...] for a table")
  raise ValueError(f"{fname} carries the marker of {' and '.join(claimed)}; pass interface={claimed[0]!r}")


def loadPoleFigure(fname, *args, interface=None, superposition=None, **kwargs):
  """the pole figures a file holds, or the files of a list, read by their format; MATLAB's
  PoleFigure.load and loadPoleFigure

    pf = loadPoleFigure('Q(10-10)_amp.cnv', cs)              # the format and the direction found from the file
    pf = PoleFigure.load([f1, f2, f3], cs)                   # several files, one pole figure or more each
    pf = loadPoleFigure([f1, f2], [h1, [h2, h3]], cs, superposition=[1, [0.52, 1.23]])  # given directions, a superposed pair weighted
    pf = loadPoleFigure('scan.dat', cs, interface='geesthacht')  # a format named
    pf = loadPoleFigure('scan.txt', h, cs, columnNames=['polar angle', 'azimuth angle', 'intensity'], radians=True)  # a table

  The format follows the extension and the marker the file carries; interface= names it when
  the extension leaves it open, one of 'dubna', 'ana', 'popla', 'labotex', 'aachen',
  'aachen_exp', 'philips', 'plf', 'ibm', 'out1', 'slc', 'beartex', 'xrd', 'siemens', 'uxd',
  'juelich', 'geesthacht', 'nja', 'pwd', 'rigaku', 'rigaku_dat', 'rigaku_txt', 'gpol', 'rw1',
  'xrdml' or 'generic'.

  The crystal directions are read off the file names (string2Miller) unless given. Give one
  per file or, for a file of several pole figures, a list of them; a list inside it is a
  superposition of directions, weighted equally unless superposition gives the weights. The
  frames are given positionally.

  A table option (columnNames, columns, header, delimiter, radians) reads a table whose columns
  the caller names: the polar angle and the azimuth, in degrees unless radians=True, or x, y, z,
  and the intensity.

  Input
    fname - a file name or a list of them
    h - Miller, the crystal directions, a list of them, a list for a superposition (optional)
    cs, ss - the crystal and the specimen frame (optional)
  Output
    pf - a PoleFigure
  Keywords
    interface - the format: 'dubna', 'uxd', 'geesthacht', 'xrdml', 'generic', ...; by
        default told by the extension and the marker the file carries
    superposition - the structure coefficients of the directions, one number or a list of
        numbers per pole figure
  See also
    PoleFigure, string2Miller, exportPoleFigure, mtexdata"""
  fnames = [os.fspath(f) for f in fname] if isinstance(fname, (list, tuple)) else [os.fspath(fname)]
  hs = [a for a in args if isinstance(a, vector3d) or isinstance(a, (list, tuple))]
  frames = [a for a in args if isinstance(a, referenceFrame)]
  parts = []
  cs, ss = _frames(frames)
  for k, f in enumerate(fnames):
    name = "generic" if interface is None and TABLE_KEYS & kwargs.keys() else chooseFormat(f, interface)
    if name in (None, "generic"):
      parts.append(_loadPoleFigure_generic(f, *frames, **kwargs))
      continue
    # the directions the caller names go to a reader whose file names none
    if not hs:
      hk = None
    elif len(fnames) == 1:
      hk = hs[0]
    else:
      hk = list(hs[0])[k]
    lines = [] if os.path.splitext(f)[1].lower() in BINARY else _lines(f)
    parts.append(FORMATS[name][1](f, lines, cs, ss, h=hk, **kwargs))
  pf = PoleFigure.cat(*parts)
  if hs:
    entries = F._entries(hs[0])
    if len(entries) != pf.numPF:
      raise ValueError(f"{len(entries)} crystal directions for {pf.numPF} pole figures")
    pf = PoleFigure(entries, pf.allR, pf.allI, *frames, superposition=[None] * pf.numPF, complete=True)
  if superposition is not None:
    pf = PoleFigure(pf.allH, pf.allR, pf.allI, superposition=superposition, complete=True)
  return pf


install(PoleFigure, staticmethod(loadPoleFigure), "load")

__all__ = ["loadPoleFigure", "string2Miller", "DubnaGrid"]
