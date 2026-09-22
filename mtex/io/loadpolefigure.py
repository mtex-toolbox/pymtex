# reading pole figures from the files the diffractometers write (MATLAB PoleFigure.load,
# interfaces/loadPoleFigure_*.m; docs/plans/io.md I6)
#
#   pf = loadPoleFigure('Q(10-10)_amp.cnv', cs)                 the format read off the file
#   pf = loadPoleFigure([f1, f2], [h1, h23], cs, superposition=[1, [0.52, 1.23]])
#   pf = loadPoleFigure('scan.txt', h, cs, columnNames=['polar angle', 'azimuth angle', 'intensity'], header=21)
#   pf = loadPoleFigure('scan.dat', cs, interface='geesthacht')
#
# A file names its reflection in its name or its header; the crystal direction is read off
# it when none is given, the crystal frame being the one passed. Every loader returns the
# pole figures of one file; loadPoleFigure joins the files and puts the directions and the
# structure coefficients given on the result.
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
from .table import isNumericRow, leadingLines, linesFrom, readFile, readTable, text

INTERFACES = ("dubna", "siemens", "geesthacht", "uxd", "aachen_exp", "rw1", "generic")


# ------------------------------------------------------------------ the reflection in a name

# the Miller indices a file name or a header line spells: four digits with optional signs,
# else three; every such group of the name is one direction, several a superposition
def string2Miller(s, cs=None):
  cs = crystalFrame("m-3m") if cs is None else cs
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
      return vector3d.cat(*[Miller(*d, cs) for d in found]) if len(found) > 1 else Miller(*found[0], cs)
  return Miller(1, 0, 0, cs)


def _frames(args):
  cs = next((a for a in args if isinstance(a, crystalFrame)), None)
  ss = next((a for a in args if isinstance(a, referenceFrame) and not isinstance(a, crystalFrame)), None)
  return cs, ss


def _lines(fname):
  return [text(l).rstrip("\r") for l in readFile(fname).split(b"\n")]


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
  return PoleFigure(h, r, d, *(f for f in (cs, ss) if f is not None), **kwargs)


# the specimen grid of the Dubna goniometer: 19 rings five degrees apart, each ring of 72
# azimuths starting where the instrument starts it
def DubnaGrid(ntheta):
  theta = np.linspace(0, np.pi / 2, 19)[:ntheta]
  rho = np.linspace(0, 2 * np.pi, 72)
  start = np.array([360.00, 336.40, 327.05, 320.11, 314.44, 309.57, 305.26, 301.37, 297.80, 294.47, 291.34, 288.38,
                    285.54, 282.81, 280.16, 277.57, 275.02, 272.50, 270.00])[::-1][:ntheta] * degree
  T, R = np.meshgrid(theta, rho, indexing="xy")
  return vector3d.byPolar(T, R + start[None, :], antipodal=True)


def loadPoleFigure_siemens(fname, *args, **kwargs):
  """the .ptx dump of a Siemens D5000: one section per pole figure, one block of azimuths
  per tilt, a background line subtracted where the file states one"""
  cs, ss = _frames(args)
  allH, allD, allTheta = [], [], []
  bg = []
  for line in _lines(fname):
    if "*Pole figure:" in line:
      allH.append(string2Miller(line[line.index(":") + 1:], cs))
      allD.append([])
      allTheta.append([])
    elif "*Khi" in line:
      allTheta[-1].append(float(line.split("=")[1]))
      bg = []
    elif "background" in line:
      bg.append(float(line[-9:]))
    elif line.startswith("*") or not line.strip():
      continue
    elif allD:
      values = np.array([float(x) for x in line.split()])
      allD[-1].extend(values - (np.mean(bg) if bg else 0.0))
  if not allH:
    raise ValueError(f"{fname} holds no '*Pole figure:' section")
  allR, allI = [], []
  for d, th in zip(allD, allTheta):
    n = len(d) // len(th)
    rho = np.arange(n) * 2 * np.pi / n
    allR.append(regularS2Grid(theta=np.asarray(th) * degree, rho=rho))
    allI.append(np.asarray(d).reshape(len(th), n).T)
  return PoleFigure(allH, allR, allI, *(f for f in (cs, ss) if f is not None), **kwargs)


def loadPoleFigure_geesthacht(fname, *args, **kwargs):
  """the TEX EINZEL table of the Geesthacht neutron diffractometer: one row per count with
  the diffraction angle that names the pole figure, the elevation and the azimuth"""
  cs, ss = _frames(args)
  rows = []
  lines = _lines(fname)
  if not lines or not lines[0].startswith("CREATOR"):
    raise ValueError(f"{fname} is not a Geesthacht file")
  for line in lines[1:]:
    s = line.strip()
    if not s or s.startswith("$"):
      continue
    parts = s.split()
    if len(parts) < 6:
      raise ValueError(f"{fname} has a row of {len(parts)} entries")
    rows.append([float(x) for x in parts[:6]])
  d = np.array(rows)
  if not np.array_equal(d[:, 0], np.arange(1, len(d) + 1)):
    raise ValueError(f"{fname} does not count its rows")
  if np.any(d[:, 3] < 0) or np.any(d[:, 3] > 90) or np.any(np.abs(d[:, 4]) > 370):
    raise ValueError(f"{fname} has angles out of range")
  allH, allR, allI = [], [], []
  pos = 0
  while pos < len(d):
    npos = np.flatnonzero(d[:, 1] == d[pos, 1])[-1] + 1
    allR.append(vector3d.byPolar(np.pi / 2 - d[pos:npos, 3] * degree, d[pos:npos, 4] * degree))
    allI.append(d[pos:npos, 5])
    allH.append(Miller(1, 0, 0, crystalFrame("m-3m") if cs is None else cs))
    pos = npos
  return PoleFigure(allH, allR, allI, *(f for f in (cs, ss) if f is not None), **kwargs)


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
  return PoleFigure(allH, allR, allI, *(f for f in (cs, ss) if f is not None), **kwargs)


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
  return PoleFigure(allH, [r] * len(allH), allI, *(f for f in (cs, ss) if f is not None), **kwargs)


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
  h = string2Miller("".join(m.groups()), cs) if m else Miller(1, 0, 0, crystalFrame("m-3m") if cs is None else cs)
  if data.size % 72 == 0:
    d = data.reshape(-1, 72).T[:, :17]
    r = regularS2Grid(theta=np.arange(0, 81, 5) * degree, rho=np.arange(2.5, 360, 5) * degree)
  elif data.size % 73 == 0:
    d = data.reshape(-1, 73).T
    r = regularS2Grid(theta=np.arange(d.shape[1]) * 5 * degree, rho=np.arange(0, 361, 5) * degree, antipodal=True)
  else:
    raise ValueError(f"{fname} holds {data.size} counts, no five degree grid")
  return PoleFigure(h, r, d ** 2 / 500, *(f for f in (cs, ss) if f is not None), **kwargs)


# the names a column may carry, MATLAB's loadHelper
_COLUMNS = {"polar angle": "theta", "polarangle": "theta", "theta": "theta", "colatitude": "theta",
            "azimuth angle": "rho", "azimuthangle": "rho", "azimuth": "rho", "rho": "rho", "phi": "rho",
            "intensity": "intensity", "intensities": "intensity", "counts": "intensity", "x": "x", "y": "y", "z": "z"}


def loadPoleFigure_generic(fname, *args, columnNames=None, columns=None, radians=False, delimiter=None, header=None,
                           superposition=None, complete=False):
  """a text file of one pole figure: rows of polar angle, azimuth and intensity, or of x, y,
  z and intensity, the columns named by columnNames and picked by columns, counted from
  zero; a block of more than fifteen columns is read as intensities on a regular grid

    loadPoleFigure_generic('seifert-111.nja', cs, header=21, columnNames=['polar angle', 'azimuth angle', 'intensity'])
  """
  cs, ss = _frames(args)
  h = next((a for a in args if isinstance(a, vector3d)), None)
  data = readFile(fname)
  if header is None:
    sep = delimiter.encode() if delimiter is not None and delimiter.strip() else None
    head, offset = leadingLines(data, lambda l: isNumericRow(l if sep is None else l.replace(sep, b" ")))
  else:
    head, offset = linesFrom(data, 0, int(header))
  if offset >= len(data):
    raise ValueError(f"{fname} holds no row of numbers")
  M = readTable(data, offset, delimiter=delimiter)
  frames = [f for f in (cs, ss) if f is not None]
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


# ------------------------------------------------------------------ the dispatcher

LOADERS = {"dubna": loadPoleFigure_dubna, "siemens": loadPoleFigure_siemens, "geesthacht": loadPoleFigure_geesthacht,
           "uxd": loadPoleFigure_uxd, "aachen_exp": loadPoleFigure_aachen_exp, "rw1": loadPoleFigure_rw1,
           "generic": loadPoleFigure_generic}
EXTENSIONS = {"cnv": "dubna", "cns": "dubna", "ptx": "siemens", "uxd": "uxd", "exp": "aachen_exp", "rw1": "rw1"}


# the format read off the extension, else off the first bytes
def _detect(fname):
  ext = os.path.splitext(fname)[1].lower().lstrip(".")
  if ext in EXTENSIONS:
    return EXTENSIONS[ext]
  head = readFile(fname)[:4096]
  if head.startswith(b"CREATOR"):
    return "geesthacht"
  if head.startswith(b"AUTEX"):
    return "aachen_exp"
  if b"*Pole figure:" in head:
    return "siemens"
  if b"_FILEVERSION" in head or b"_KHI" in head:
    return "uxd"
  return "generic"


def loadPoleFigure(fname, *args, interface=None, superposition=None, **kwargs):
  """the pole figures a file holds, or the files of a list, read by their format

    loadPoleFigure('Q(10-10)_amp.cnv', cs)
    loadPoleFigure([f1, f2], [h1, h23], cs, superposition=[1, [0.52, 1.23]])
    loadPoleFigure('scan.dat', h, cs, interface='geesthacht')
    loadPoleFigure('scan.txt', cs, columnNames=['polar angle', 'azimuth angle', 'intensity'], header=21)

  The crystal directions are read off the file names unless given, one per file or, for
  a file of several pole figures, a list of them; the frames are given positionally.
  """
  fnames = [os.fspath(f) for f in fname] if isinstance(fname, (list, tuple)) else [os.fspath(fname)]
  hs = [a for a in args if isinstance(a, vector3d) or isinstance(a, (list, tuple))]
  frames = [a for a in args if isinstance(a, referenceFrame)]
  parts = []
  for f in fnames:
    name = _detect(f) if interface is None else str(interface).lower()
    if name not in LOADERS:
      raise ValueError(f"no pole figure interface {interface!r}; the interfaces are {', '.join(INTERFACES)}")
    parts.append(LOADERS[name](f, *frames, **kwargs))
  pf = PoleFigure.cat(*parts)
  if hs:
    h = hs[0]
    entries = [h[i] for i in range(h.size)] if isinstance(h, vector3d) and h.size > 1 else [h] if isinstance(h, vector3d) else list(h)
    if len(entries) != pf.numPF:
      raise ValueError(f"{len(entries)} crystal directions for {pf.numPF} pole figures")
    pf = PoleFigure(entries, pf.allR, pf.allI, *frames, superposition=[None] * pf.numPF, complete=True)
  if superposition is not None:
    pf = PoleFigure(pf.allH, pf.allR, pf.allI, superposition=superposition, complete=True)
  return pf


PoleFigure.load = staticmethod(lambda fname, *args, **kwargs: loadPoleFigure(fname, *args, **kwargs))

__all__ = ["loadPoleFigure", "loadPoleFigure_generic", "string2Miller", "DubnaGrid"]
