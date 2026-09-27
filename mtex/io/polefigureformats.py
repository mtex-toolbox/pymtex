# the vendor pole figure formats (MATLAB interfaces/loadPoleFigure_*.m, Julia polefigureformats.jl)
#
# A diffractometer writes what it scanned: a polar ladder and an azimuth ladder, each a
# start, a step and a stop, then the counts ring by ring. What a format needs is a way to
# find its ladders, its crystal directions and the layout of its number block; the grid,
# the block and the repetition over several pole figures in one file are shared here.
#
# Every reader takes the lines of the file, its path, the crystal frame and the specimen
# frame the caller passed (None where not) and the crystal directions the caller named,
# and returns the directions, the specimen directions and the intensities of every pole
# figure it holds, with the crystal frame they are written in. A format that states its
# own crystal frame uses it unless the caller passes one. Where Julia and MATLAB part, the
# reader follows Julia and says why.
from __future__ import annotations

import os
import re
import warnings

import numpy as np

from ..geometry.frames import crystalFrame
from ..geometry.grids.s2grid import regularS2Grid
from ..geometry.miller import Miller
from ..geometry.units import degree
from ..geometry.vector3d import vector3d

NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# ------------------------------------------------------------------ the shared pieces

def _float(s):
  try:
    return float(s)
  except ValueError:
    return None


# the frame the directions are written in: the caller's, else the file's, else cubic
def _frame(cs, fromFile=None):
  return cs if cs is not None else fromFile if fromFile is not None else crystalFrame("m-3m")


def millerIndices(s):
  """the signed indices of a group of digits, three or four of them, a minus binding to the
  digit after it; None where the group holds another count"""
  idx, neg = [], False
  for c in s:
    if c == "-":
      neg = True
    elif c.isdigit():
      idx.append(-int(c) if neg else int(c))
      neg = False
  return idx if len(idx) in (3, 4) else None


def millerFromGroup(s, cs):
  idx = millerIndices(s)
  if idx is None:
    raise ValueError(f"cannot read {s.strip()!r} as Miller indices")
  return Miller(*idx, cs)


# the crystal direction a file name carries, the caller's where given
def _fromName(path, cs, h=None):
  from .loadpolefigure import string2Miller
  return h if h is not None else string2Miller(os.path.basename(path), cs)


# the k-th of the directions the caller named for a file of npf pole figures
def _directionOf(h, k, npf, path, cs):
  if h is None:
    return _fromName(path, cs)
  hs = _entries(h)
  if npf == 1 and len(hs) == 1:
    return hs[0]
  if len(hs) != npf:
    raise ValueError(f"{path} holds {npf} pole figures and h names {len(hs)}")
  return hs[k]


def _entries(h):
  if isinstance(h, (list, tuple)):
    return list(h)
  return [h[i] for i in range(h.size)] if h.size > 1 else [h]


def firstToken(line):
  """the first token of a line, empty where the line is"""
  t = line.split()
  return t[0] if t else ""


def fixedFields(line, width, count):
  """the numbers of a row cut into fields of a fixed width, as Fortran writes them: the row
  holds count of them after a leading remainder that belongs to no field"""
  out = []
  for a in range(len(line) % width, len(line) - width + 1, width):
    if len(out) >= count:
      break
    f = line[a:a + width].strip()
    if not f:
      continue
    v = _float(f)
    if v is None:
      raise ValueError(f"{f!r} is not a number of a {width} character field")
    out.append(v)
  return out


def fixedBlock(lines, i, n, width, count, toBlank=False):
  """a block of n numbers in rows of count fields of a fixed width, from line i on, and the
  line after it; toBlank reads to the blank line that ends the block and keeps the first n"""
  out = []
  while i < len(lines) and (toBlank or len(out) < n):
    if not lines[i].strip():
      if toBlank:
        i += 1
        break
    else:
      out.extend(fixedFields(lines[i], width, count))
    i += 1
  if len(out) < n:
    raise ValueError(f"the block holds {len(out)} of the {n} values the header announces")
  return np.array(out[:n]), i


def numberBlock(lines, i, n):
  """a block of n numbers written free format, from line i on, and the line after it"""
  out = []
  while len(out) < n and i < len(lines):
    for f in lines[i].split():
      v = _float(f)
      if v is None:
        break
      out.append(v)
    i += 1
  if len(out) < n:
    raise ValueError(f"the block holds {len(out)} of the {n} values the header announces")
  return np.array(out[:n]), i


def numbersOf(lines):
  """the numbers a run of lines holds, whatever separates them"""
  out = []
  for l in lines:
    for f in re.split(r"[ \t,;]+", l.strip()):
      v = _float(f) if f else None
      if v is not None:
        out.append(v)
  return out


def isNumericRow(line):
  """a data row starts with a number, a header line does not"""
  t = line.split()
  return bool(t) and _float(t[0].rstrip(",;")) is not None


def readColumns(lines):
  """the columns of the rows that hold numbers, whatever separates the fields"""
  rows = [[float(f) for f in re.split(r"[ \t,;]+", l.strip()) if f] for l in lines if isNumericRow(l)]
  if not rows:
    raise ValueError("the file holds no data rows")
  n = len(rows[0])
  for k, r in enumerate(rows):
    if len(r) != n:
      raise ValueError(f"row {k + 1} has {len(r)} columns, expected {n}")
  return [np.array(c) for c in zip(*rows)]


def ladderOf(start, step, stop):
  """the ladder of a start, a step and a stop in degrees, its count taken from the three of
  them rather than accumulated, since 26 steps of 0.45 land above 11.7"""
  n = int(np.floor((stop - start) / step + 1e-8)) + 1
  if n <= 0:
    raise ValueError(f"a ladder from {start} to {stop} in steps of {step} holds no point")
  return (start + step * np.arange(n)) * degree


def regularGrid(theta, rho):
  """the grid of a polar and an azimuth ladder, nRho x nTheta, as axes: a diffraction
  intensity does not tell a direction from its opposite"""
  return regularS2Grid(theta=np.asarray(theta), rho=np.asarray(rho), antipodal=True)


def onGrid(I, r):
  """counts written ring by ring laid onto a grid of nRho x nTheta"""
  return np.asarray(I, dtype=float).reshape(r.shape[1], r.shape[0]).T


def ringOrder(r):
  """the points of a grid in the order the files write them, ring by ring"""
  return vector3d(np.swapaxes(r.data, 0, 1).reshape(-1, 3), antipodal=True)


def polarList(theta, rho):
  """specimen directions from two lists of degrees, as the measurement scanned them"""
  return vector3d.byPolar(np.asarray(theta, dtype=float) * degree, np.asarray(rho, dtype=float) * degree, antipodal=True)


def markerSections(lines, isMarker):
  """the lines a marker opens, as the index ranges from one marker to the next"""
  starts = [i for i, l in enumerate(lines) if isMarker(l)]
  return [range(s, starts[k + 1] if k + 1 < len(starts) else len(lines)) for k, s in enumerate(starts)]


# the point groups LaboTEX and BearTex number from one to eleven
STRUCTURE_CODES = ("C1", "C2", "D2", "C4", "D4", "T", "O", "C3", "D3", "C6", "D6")


def structureFrame(code, abc, angles):
  if not 1 <= code <= len(STRUCTURE_CODES):
    raise ValueError(f"{code} is not one of the eleven structure codes")
  return crystalFrame(STRUCTURE_CODES[code - 1], list(abc), [a * degree for a in angles])


def latticeFromSymbol(s):
  """the point group a five character field names, as much of it as names one"""
  t = s.strip()
  for n in range(len(t), 0, -1):
    try:
      return crystalFrame(t[:n])
    except Exception:
      continue
  return None


# ------------------------------------------------------------------ ana

def readAna(lines, path, cs, h=None):
  """a comment and twenty numbers that give the two ladders"""
  head, i = numberBlock(lines, 1, 20)
  n = int(round(head[0]))
  r = regularGrid(ladderOf(head[5] + head[6], head[8], head[5] + head[7]), ladderOf(head[9], head[11], head[10] - head[11]))
  if r.size != n:
    raise ValueError(f"the header announces {n} values on a grid of {r.size}")
  I, _ = numberBlock(lines, i, n)
  cs = _frame(cs)
  return [_fromName(path, cs, h)], [r], [onGrid(I, r)], cs


# ------------------------------------------------------------------ popla

def readPopla(lines, path, cs, h=None):
  """a comment, a header of five character fields, then counts in four character ones; two
  flags of the header say whether a ladder starts at a cell centre or an edge, and the
  scale is a divisor. Its rings close, 18, 18, 18 and 19 values, so a block is read to its
  blank line and the header's count kept"""
  cs = _frame(cs)
  hs, rs, Is = [], [], []
  i = 0
  while i + 1 < len(lines) and lines[i + 1].strip():
    s = lines[i + 1]
    field = lambda k: float(s[5 * k:5 * k + 5])
    flag = lambda k: int(s[25 + 2 * k:27 + 2 * k])
    dt, tmax, dr, rmax, scale = field(1), field(2), field(3), field(4), field(7)
    theta = ladderOf(dt / 2 if flag(0) == 0 else 0.0, dt, tmax)
    rho = ladderOf(dr / 2 if flag(1) == 0 else 0.0, dr, rmax - dr / (2 if flag(1) == 0 else 1))
    r = regularGrid(theta, rho)
    I, i = fixedBlock(lines, i + 2, r.size, 4, 18, toBlank=True)
    hs.append(millerFromGroup(s[:5], cs))
    rs.append(r)
    Is.append(onGrid(I / scale, r))
  if not hs:
    raise ValueError(f"{path} holds no Popla header")
  return hs, rs, Is, cs


# ------------------------------------------------------------------ LaboTEX

def readLabotex(lines, path, cs, h=None):
  """a structure code with the cell, one header line per pole figure with its own ladders,
  then the blocks in the same order"""
  c, _ = numberBlock(lines, 3, 7)
  cs = _frame(cs, structureFrame(int(round(c[0])), c[1:4], c[4:7]))
  npf = int(round(numberBlock(lines, 4, 1)[0][0]))
  heads = []
  for n in range(npf):
    hr, _ = numberBlock(lines, 6 + n, 11)
    heads.append((Miller(hr[8], hr[9], hr[10], cs), regularGrid(ladderOf(hr[1], hr[3], hr[2]), ladderOf(hr[4], hr[6], hr[5]))))
  i = 6 + npf
  hs, rs, Is = [], [], []
  for hkl, r in heads:
    I, i = numberBlock(lines, i, r.size)
    hs.append(hkl)
    rs.append(r)
    Is.append(onGrid(I, r))
  return hs, rs, Is, cs


# ------------------------------------------------------------------ Aachen

# a header of five character fields: the indices, the point group, the polar and azimuth
# steps and the polar reach with decimals, the window and symmetry flags as integers, and
# the Fortran format of its block, how wide its fields are and how many sit on a line,
# eighteen of four where it states none. MATLAB reads one line of each block and returns
# NaN, testing the count of pole figures for the length of the block; here a block runs to
# the next header, and the factor the header states is the divisor it names
AACHEN_FORMAT = re.compile(r"\(\s*(\d+)\s*[fF]\s*(\d+)")


def isAachenHeader(l):
  if len(l) < 35 or millerIndices(l[:5]) is None:
    return False
  steps = [l[10:15], l[15:20], l[20:25]]
  flags = [l[25:30], l[30:35]]
  if not all("." in f and _float(f) is not None for f in steps) or not all(f.strip().isdigit() for f in flags):
    return False
  dt, dr, tmax = (float(f) for f in steps)
  return 0.5 < dt < 20 and 0.5 < dr < 20 and 10 <= tmax <= 90


def readAachen(lines, path, cs, h=None):
  heads = [i for i, l in enumerate(lines) if isAachenHeader(l)]
  if not heads:
    raise ValueError(f"{path} holds no Aachen header")
  hs, rs, Is = [], [], []
  fr = None
  for k, i in enumerate(heads):
    s = lines[i]
    m = AACHEN_FORMAT.search(s[35:])
    count, width = (int(m.group(1)), int(m.group(2))) if m else (18, 4)
    fr = _frame(cs, latticeFromSymbol(s[5:10]))
    dt, dr, tmax = float(s[10:15]), float(s[15:20]), float(s[20:25])
    iwin, isim = int(s[25:30]), int(s[30:35])
    rmax = {0: 360.0, 2: 180.0, 4: 90.0}.get(isim)
    if rmax is None:
      raise ValueError(f"the symmetry flag of an Aachen header is 0, 2 or 4, not {isim}")
    scale = float(lines[i + 1][:6])
    r = regularGrid(ladderOf(dt / 2 if iwin == 0 else 0.0, dt, tmax), ladderOf(0, dr, rmax - dr))
    stop = heads[k + 1] if k + 1 < len(heads) else len(lines)
    I, _ = fixedBlock(lines[:stop], i + 2, r.size, width, count)
    hs.append(millerFromGroup(s[:5], fr))
    rs.append(r)
    Is.append(onGrid(I / scale, r))
  return hs, rs, Is, fr


# ------------------------------------------------------------------ Philips

def readPhilips(lines, path, cs, h=None):
  """the two ladders on their own named lines, the block a table whose first column is the
  azimuth it stands for; the azimuth ladder closes the turn, so its last point is the first"""
  def ladder(name):
    line = next((l for l in lines if l.strip().startswith(name)), None)
    if line is None:
      raise ValueError(f"{path} states no {name} line")
    v = [x for x in (_float(f) for f in line.replace(name, "").split()) if x is not None]
    if len(v) != 3:
      raise ValueError(f"the {name} line states {len(v)} of its start, end and step")
    return v
  psi, phi = ladder("Psi:"), ladder("Phi:")
  r = regularGrid(ladderOf(psi[0], psi[2], psi[1]), ladderOf(phi[0], phi[2], phi[1] - phi[2]))
  cols = readColumns(lines)
  if len(cols) != r.shape[1] + 1:
    raise ValueError(f"the table holds {len(cols)} columns, the polar ladder {r.shape[1]} of them plus the azimuth")
  I = np.stack(cols[1:], axis=1)
  if I.shape != r.shape:
    raise ValueError(f"the table is {I.shape} where the grid is {r.shape}")
  cs = _frame(cs)
  return [_fromName(path, cs, h)], [r], [I], cs


# ------------------------------------------------------------------ the grid a constant or a count of the header

def readPlf(lines, path, cs, h=None):
  """Queens plf: a fixed 90 by 17 grid to 80°, one hkl line and one block per pole figure"""
  r = regularS2Grid(points=(90, 17), maxTheta=80 * degree, antipodal=True)
  cs = _frame(cs)
  hs, Is = [], []
  i = 0
  while i < len(lines):
    idx = millerIndices(firstToken(lines[i]))
    if idx is None:
      i += 1
      continue
    I, i = numberBlock(lines, i + 1, r.size)
    hs.append(Miller(*idx, cs))
    Is.append(onGrid(I, r))
  if not hs:
    raise ValueError(f"{path} holds no hkl line")
  return hs, [r] * len(hs), Is, cs


def readIbm(lines, path, cs, h=None):
  """IBM: a header line per pole figure with its indices and ring counts, and the polar angle
  the scan reached on the line after it, where MATLAB spreads the rings over the sphere"""
  cs = _frame(cs)
  hs, rs, Is = [], [], []
  i = 0
  while i < len(lines):
    head = [x for x in (_float(f) for f in lines[i].split()) if x is not None]
    if len(head) < 4 or millerIndices(firstToken(lines[i])) is None:
      i += 1
      continue
    steps, nr = int(round(head[1])), int(round(head[2]))
    m = re.search(r"WKO\s*=\s*([0-9.]+)", lines[i + 1])
    if m is None:
      raise ValueError(f"{path} states no WKO line for the pole figure of line {i + 1}")
    r = regularS2Grid(points=(nr, steps + 1), maxTheta=float(m.group(1)) * degree, antipodal=True)
    hkl = millerFromGroup(firstToken(lines[i]), cs)
    I, i = numberBlock(lines, i + 2, r.size)
    hs.append(hkl)
    rs.append(r)
    Is.append(onGrid(I, r))
  if not hs:
    raise ValueError(f"{path} holds no IBM header")
  return hs, rs, Is, cs


def readOut1(lines, path, cs, h=None):
  """GADDS out1: a two column table of several pole figures on one fixed grid, a point not
  measured written with a negative intensity; the file names no reflections, so the caller
  has to, where MATLAB fills in a list of eight it keeps in the reader"""
  r = ringOrder(regularS2Grid(points=(72, 18), maxTheta=85 * degree, antipodal=True))
  n = r.size
  cols = readColumns([l for l in lines if not l.strip().startswith("!@!")])
  if len(cols) != 2:
    raise ValueError(f"a GADDS table holds two columns, this one {len(cols)}")
  I = cols[1]
  if I.size % n:
    raise ValueError(f"{I.size} intensities are not a whole number of pole figures of {n} points")
  npf = I.size // n
  if h is None:
    raise ValueError(f"{path} names no crystal directions; pass one for each of its {npf} pole figures")
  hs = _entries(h)
  if len(hs) != npf:
    raise ValueError(f"{path} holds {npf} pole figures and h names {len(hs)}")
  rs, Is = [], []
  for k in range(npf):
    v = I[k * n:(k + 1) * n]
    keep = v >= 0
    rs.append(r[keep])
    Is.append(v[keep])
  return hs, rs, Is, _frame(cs)


def readSlc(lines, path, cs, h=None):
  """slc: a table whose first column is the azimuth in tens of degrees and whose others are
  the polar ladder, from the pole down to the other pole"""
  cols = readColumns([l.replace("FRAME", " ") for l in lines])
  if len(cols) <= 2:
    raise ValueError(f"{path} holds {len(cols)} columns")
  r = regularS2Grid(theta=np.linspace(np.pi / 2, -np.pi / 2, len(cols) - 1), rho=cols[0] * 10 * degree, antipodal=True)
  cs = _frame(cs)
  return [_fromName(path, cs, h)], [r], [np.stack(cols[1:], axis=1)], cs


def readBeartex(lines, path, cs, h=None):
  """BearTex: seven header lines per pole figure, the cell and the two structure codes on the
  sixth and the indices with the polar range of the scan on the seventh; the block holds
  the whole 5° grid and the header says which rings were measured"""
  hs, rs, Is = [], [], []
  fr = cs
  grid = ringOrder(regularS2Grid(points=(72, 19), antipodal=True))
  i = 0
  while i + 6 < len(lines):
    cell = [x for x in (_float(f) for f in lines[i + 5].split()) if x is not None]
    if len(cell) < 8:
      break
    if fr is None:
      fr = structureFrame(int(round(cell[6])), cell[0:3], cell[3:6])
    s = lines[i + 6]
    hkl = [float(x) for x in s[:10].split()]
    info = fixedFields(s[10:40], 5, 6)
    I, i = fixedBlock(lines, i + 7, grid.size, 4, 18)
    keep = (grid.theta >= info[0] * degree - 1e-9) & (grid.theta <= info[1] * degree + 1e-9)
    hs.append(Miller(hkl[0], hkl[1], hkl[2], fr))
    rs.append(grid[keep])
    Is.append(I[keep])
    i += 1
  if not hs:
    raise ValueError(f"{path} holds no BearTex header")
  return hs, rs, Is, fr


# xrd: the azimuth ladder from the scan tokens of the header, one polar angle per ring from
# the tilt token, in either of the two spellings the vendors write them
XRD_TOKENS = ((r"\*MEAS_SCAN_START\s*\"(\S*)\"", r"\*MEAS_SCAN_STEP\s*\"(\S*)\"", r"\*MEAS_SCAN_STOP\s*\"(\S*)\"",
               r"\*MEAS_3DE_ALPHA_ANGLE\s*\"(\S*)\""),
              (r"\*START\s*=\s*(\S*)", r"\*STEP\s*=\s*(\S*)", r"\*STOP\s*=\s*(\S*)", r"\*PF_AANGLE\s*=\s*(\S*)"))


def readXrd(lines, path, cs, h=None):
  def values(token):
    return [float(m.group(1)) for m in (re.search(token, l) for l in lines) if m is not None]
  for start, step, stop, tilt in XRD_TOKENS:
    a, s, b, t = values(start), values(step), values(stop), values(tilt)
    if not (a and s and b and t):
      continue
    r = regularS2Grid(theta=np.pi / 2 - np.array(t) * degree, rho=ladderOf(a[0], s[0], b[0]), antipodal=True)
    d = numbersOf([l for l in lines if not l.strip().startswith("*")])
    # a file that writes the angle and the weight beside the count holds three of them
    if len(d) >= 3 * r.size:
      d = d[1::3]
    if len(d) < r.size:
      raise ValueError(f"{path} holds {len(d)} values on a grid of {r.size}")
    cs = _frame(cs)
    return [_fromName(path, cs, h)], [r], [onGrid(d[:r.size], r)], cs
  raise ValueError(f"{path} carries neither spelling of the scan tokens")


# ------------------------------------------------------------------ the sections a marker opens

def readSiemens(lines, path, cs, h=None):
  """Siemens D5000: `*Pole figure:` opens one and names its indices, `*Khi` opens a ring at its
  polar angle, and the background lines after it are subtracted from that ring. The counts
  sit in eight character fields, and a five digit one fills its field, so a line is cut
  into its fields rather than split at the blanks"""
  cs = _frame(cs)
  hs, rs, Is = [], [], []
  for sec in markerSections(lines, lambda l: "*Pole figure:" in l):
    first = lines[sec[0]]
    hkl = millerFromGroup(first[first.index("*Pole figure:") + 13:].strip(), cs)
    theta, I, bg = [], [], []
    for k in sec[1:]:
      l = lines[k]
      if "*Khi" in l:
        v = NUMBER.findall(l)
        if v:
          theta.append(float(v[-1]))
        bg = []
      elif "background" in l:
        v = NUMBER.findall(l)
        if v:
          bg.append(float(v[-1]))
      elif l.strip().startswith("*"):
        continue
      else:
        v = np.array(fixedFields(l, 8, len(l) // 8))
        I.extend(v - (np.mean(bg) if bg else 0.0))
    if not theta:
      continue
    n = len(I) // len(theta)
    if n <= 0:
      raise ValueError(f"{path} holds {len(theta)} rings and no intensity")
    r = regularGrid(np.array(theta) * degree, ladderOf(0, 360 / n, 360 - 360 / n))
    hs.append(hkl)
    rs.append(r)
    Is.append(onGrid(I, r))
  if not hs:
    raise ValueError(f"{path} holds no '*Pole figure:' line")
  return hs, rs, Is, cs


def readJuelich(lines, path, cs, h=None):
  """Jülich: two header lines, then the polar angle, the azimuth and the count"""
  cols = readColumns(lines)
  if len(cols) < 3:
    raise ValueError(f"a Jülich file holds three columns, this one {len(cols)}")
  cs = _frame(cs)
  return [_directionOf(h, 0, 1, path, cs)], [polarList(cols[0], cols[1])], [cols[2]], cs


def readGeesthacht(lines, path, cs, h=None):
  """Geesthacht: a `$` commented header and one row per measured direction, its second column
  the scan the row belongs to and its fourth the elevation above the equator; the last field
  is the time the point was counted, written 00:07.76. The rows of one scan are one pole figure"""
  cols = readColumns([l.replace(":", " ") for l in lines if "$" not in l and not l.startswith("CREATOR")])
  if len(cols) < 6:
    raise ValueError(f"a Geesthacht file holds eleven columns, this one {len(cols)}")
  scan, elevation, rho, I = cols[1], cols[3], cols[4], cols[5]
  bounds = np.concatenate([[0], np.flatnonzero(np.diff(scan) != 0) + 1, [scan.size]])
  npf = len(bounds) - 1
  cs = _frame(cs)
  sl = [slice(bounds[k], bounds[k + 1]) for k in range(npf)]
  return ([_directionOf(h, k, npf, path, cs) for k in range(npf)], [polarList(90 - elevation[s], rho[s]) for s in sl],
          [I[s] for s in sl], cs)


def _keyValue(s):
  k, _, v = s.partition("=")
  return k.strip(), v.strip()


def readNja(lines, path, cs, h=None):
  """Seifert nja: an `&KEY=value` header that names the reflection and the ladders, and the
  block the `#Values` line opens, the two angles, the count and its background. MATLAB reads
  the indices from token positions and the block from a fixed line, which drops its first
  measurement; both are named in the file and read by their names here"""
  keys = {}
  for l in lines:
    for f in l.split("&"):
      k, v = _keyValue(f)
      if v:
        keys[k.upper()] = v
  i = next((k for k, l in enumerate(lines) if l.strip().startswith("#Values")), None)
  if i is None:
    raise ValueError(f"{path} holds no #Values line")
  cols = readColumns(lines[i:])
  if len(cols) < 3:
    raise ValueError(f"an nja block holds four columns, this one {len(cols)}")
  if "NOVALUES" in keys and cols[0].size != int(keys["NOVALUES"]):
    raise ValueError(f"{path} announces {keys['NOVALUES']} values and holds {cols[0].size}")
  cs = _frame(cs)
  if h is not None:
    hkl = h
  elif all(k in keys for k in ("H", "K", "L")):
    hkl = Miller(*(float(keys[k]) for k in ("H", "K", "L")), cs)
  else:
    hkl = _fromName(path, cs)
  return [hkl], [polarList(cols[0], cols[1])], [cols[2]], cs


def readPwd(lines, path, cs, h=None):
  """a Siemens defocussing curve: `R111` opens the curve of that reflection, whose rows hold
  the polar angle, the intensity and the background under it, one azimuth"""
  cs = _frame(cs)
  hs, rs, Is = [], [], []
  for sec in markerSections(lines, lambda l: re.match(r"^\s*R\s*[0-9\-]+\s*$", l) is not None):
    hkl = millerFromGroup(lines[sec[0]].strip()[1:].strip(), cs)
    rows = [v for v in (numbersOf([lines[k]]) for k in sec[1:] if isNumericRow(lines[k])) if len(v) >= 3]
    if not rows:
      continue
    theta = [v[0] for v in rows]
    hs.append(hkl)
    rs.append(polarList(theta, np.zeros(len(theta))))
    Is.append(np.array([v[1] - v[2] for v in rows]))
  if not hs:
    raise ValueError(f"{path} holds no reflection line")
  return hs, rs, Is, cs


def readRigaku(lines, path, cs, h=None):
  """Rigaku 3DExplore: a header of `#` lines and the elevation above the equator, the azimuth
  and the count"""
  cols = readColumns(lines)
  if len(cols) < 3:
    raise ValueError(f"a Rigaku table holds three columns, this one {len(cols)}")
  cs = _frame(cs)
  return [_directionOf(h, 0, 1, path, cs)], [polarList(90 - cols[0], cols[1])], [cols[2]], cs


def readRigakuDat(lines, path, cs, h=None):
  """Rigaku raster: a direction written as its two components in the plane, in degrees off the
  pole, so the polar angle is their distance from it and the azimuth their angle"""
  cols = readColumns(lines)
  if len(cols) < 4:
    raise ValueError(f"a Rigaku raster holds four columns, this one {len(cols)}")
  x, y = cols[0], cols[1]
  cs = _frame(cs)
  r = vector3d.byPolar(np.hypot(x, y) * degree, np.arctan2(y, x), antipodal=True)
  return [_directionOf(h, 0, 1, path, cs)], [r], [cols[3]], cs


def readRigakuTxt(lines, path, cs, h=None, theta=None):
  """Rigaku text: one section per ring, each stating the azimuth ladder it was scanned on. The
  polar ladder is nowhere in the file: MATLAB takes the azimuth step for it and counts the
  rings down from the last, reaching past the pole; the same count without that step off
  the end is the default here, and theta states the ladder the file does not"""
  secs = markerSections(lines, lambda l: firstToken(l.replace(",", " ")).lower() == "start")
  if not secs:
    raise ValueError(f"{path} holds no Start line")
  th, rho, I = [], [], []
  step = 1.0
  for k, sec in enumerate(secs):
    keys, rows = {}, []
    for i in sec:
      l = lines[i].replace(",", " ")
      if isNumericRow(l):
        v = numbersOf([lines[i]])
        if len(v) >= 2:
          rows.append(v)
      else:
        f = l.split()
        if len(f) >= 2 and _float(f[1]) is not None:
          keys[f[0].lower()] = float(f[1])
    if not rows:
      continue
    step = keys.get("step", step)
    ring = (len(secs) - 1 - k) * step if theta is None else np.asarray(theta)[k] / degree
    th.extend([ring] * len(rows))
    rho.extend(v[0] for v in rows)
    I.extend(v[1] for v in rows)
  if not I:
    raise ValueError(f"{path} holds no ring under its Start lines")
  if theta is None and max(th) < 45:
    warnings.warn(f"{path} states no polar ladder, and the azimuth step puts its {len(secs)} rings {step}° apart, reaching "
                  f"{max(th)}°; pass theta=... where the scan covered another range", stacklevel=3)
  cs = _frame(cs)
  return [_directionOf(h, 0, 1, path, cs)], [polarList(th, rho)], [np.array(I)], cs


# ------------------------------------------------------------------ the binary ones and the XML one

def gaddsRecord(raw, record):
  """the value of one of the eighty character records a GADDS header is written in"""
  a = (record - 1) * 80
  if a + 80 > len(raw):
    raise ValueError(f"the header holds fewer than {record} records")
  return raw[a + 8:a + 80].decode("latin-1").strip()


def readGpol(lines, path, cs, h=None, raw=None):
  """GADDS gpol: an image of the pole figure in a stereographic projection, one pixel per
  measured direction and zero where nothing was, the counts that do not fit a pixel listed
  after the image as sixteen character records of nine digits of count and seven of position"""
  header = 96 * 80
  if len(raw) <= header:
    raise ValueError(f"{path} is shorter than its own header")
  kind = gaddsRecord(raw, 4)
  if "stereo" not in kind.lower():
    raise ValueError(f"{path} is written in the {kind} projection, and only the stereographic one states its directions")
  width = int(round(float(gaddsRecord(raw, 40))))
  rows, ncols = int(round(float(gaddsRecord(raw, 41)))), int(round(float(gaddsRecord(raw, 42))))
  T = {1: "<u1", 2: "<u2", 4: "<u4"}.get(width)
  if T is None:
    raise ValueError(f"a pixel of {width} bytes is none of the widths GADDS writes")
  n = rows * ncols
  A = np.frombuffer(raw[header:header + n * width], dtype=T).astype(float)
  full = float(np.iinfo(np.dtype(T)).max)
  tail = raw[header + n * width:].decode("latin-1")
  for k in range(len(tail) // 16):
    v, p = _float(tail[16 * k:16 * k + 9].strip()), tail[16 * k + 9:16 * k + 16].strip()
    if v is None or not p.isdigit() or not 0 <= int(p) < n:
      break
    if A[int(p)] == full:
      A[int(p)] = v
  # the image as the file stores it, column after column; a pixel names the direction it
  # stands for about the centre of the projection, and a pixel of zero was not measured
  MT = A.reshape(ncols, rows)
  c, rr = np.nonzero(MT)
  i1, i2 = rr + 1.0, c + 1.0
  theta = 2 * np.arctan(np.hypot(i1 - ncols / 2 - 0.5, i2 - rows / 2 - 0.5) / (rows / 2 - 1))
  rho = -np.arctan2(i2 - rows / 2 - 0.5, i1 - ncols / 2 - 0.5)
  cs = _frame(cs)
  return [_directionOf(h, 0, 1, path, cs)], [vector3d.byPolar(theta, rho, antipodal=True)], [MT[c, rr]], cs


# the elements of one tag, as the attributes of the opening tag and the text between the two
def _xmlElements(s, tag):
  return [(m.group(1), m.group(2)) for m in re.finditer(rf"<{tag}\b([^>]*)>(.*?)</{tag}>", s, re.S)]


def _xmlAttribute(attrs, name):
  m = re.search(rf'{name}\s*=\s*"([^"]*)"', attrs)
  return None if m is None else m.group(1)


def _xmlText(s, tag):
  e = _xmlElements(s, tag)
  return e[0][1].strip() if e else None


# the angles an axis of the goniometer took, its first and last value in radians
def _xmlPositions(scan):
  out = {}
  for attrs, body in _xmlElements(scan, "positions"):
    axis = _xmlAttribute(attrs, "axis")
    if axis is None:
      continue
    scale = degree if (_xmlAttribute(attrs, "unit") or "").lower() == "deg" else 1.0
    common = _xmlText(body, "commonPosition")
    if common is not None:
      a = b = float(common)
    else:
      s, e = _xmlText(body, "startPosition"), _xmlText(body, "endPosition")
      if s is None or e is None:
        continue
      a, b = float(s), float(e)
    out[axis] = (a * scale, b * scale)
  return out


def readXrdml(lines, path, cs, h=None, raw=None):
  """the PANalytical XML: one scan per ring, the direction of a point the goniometer put
  together out of the four angles the scan states, Rz(phi) Ry(psi) Rx(omega + (pi - 2theta) / 2)
  applied to y; read with a tag scanner, since the file is machine written and regular"""
  txt = raw.decode("utf-8", errors="replace")
  measurements = _xmlElements(txt, "xrdMeasurement")
  if not measurements:
    raise ValueError(f"{path} holds no xrdMeasurement")
  cs = _frame(cs)
  hs, rs, Is = [], [], []
  for k, (_, measurement) in enumerate(measurements):
    xyz, I, hkl = [], [], None
    for _, scan in _xmlElements(measurement, "scan"):
      counts = _xmlText(scan, "intensities") or _xmlText(scan, "counts") or ""
      if not counts:
        continue
      v = np.array([float(f) for f in counts.split()])
      t = _xmlText(scan, "commonCountingTime")
      if t is not None:
        v = v / float(t)
      pos = _xmlPositions(scan)
      ladder = lambda name: np.linspace(pos[name][0], pos[name][1], v.size) if name in pos else np.zeros(v.size)
      t2, om, ph = ladder("2Theta"), ladder("Omega"), ladder("Phi")
      ps = ladder("Psi") if "Psi" in pos else ladder("Chi")
      a = om + (np.pi - t2) / 2
      x0, y0, z0 = np.sin(a) * np.sin(ps), np.cos(a), np.sin(a) * np.cos(ps)
      xyz.append(np.stack([x0 * np.cos(ph) - y0 * np.sin(ph), x0 * np.sin(ph) + y0 * np.cos(ph), z0], axis=1))
      I.append(v)
      refl = _xmlElements(scan, "reflection")
      if hkl is None and refl:
        idx = [_xmlText(refl[0][1], n) for n in ("h", "k", "l")]
        if all(x is not None for x in idx):
          hkl = Miller(*(float(x) for x in idx), cs)
    if not I:
      continue
    hs.append(_directionOf(h, k, len(measurements), path, cs) if h is not None else hkl if hkl is not None else _fromName(path, cs))
    rs.append(vector3d(np.concatenate(xyz), antipodal=True))
    Is.append(np.concatenate(I))
  if not hs:
    raise ValueError(f"{path} holds no scan with intensities")
  return hs, rs, Is, cs


__all__ = []
