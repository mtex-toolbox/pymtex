# the HDF5 maps of the vendors (MATLAB loadEBSD_h5 with interfaces/hdf5_config/*.json, Julia h5.jl)
#
#   ebsd = loadEBSD('map.h5oina')                 the vendor read off the file
#   ebsd = loadEBSD('map.oh5', dataSet=1)          one of the maps a file holds, by number or name
#
# The layout of a vendor's tree is a config and not code: hdf5_config/<vendor>.json says
# where the manufacturer key, the data sets, the positions, the phases, the Euler angles
# and the crystallography sit, and every entry says how its value is matched:
#
#   absolute      the full path, literally
#   search_root   below the key of the block it stands in, matched against the full path
#   search_free   below the group that contains the data set, against the full path
#   search_set    below the data set, against the path relative to it
#
# The configs are MTEX's own and are copied, not rewritten: they describe the vendors'
# layout, so a new vendor is one JSON file on both sides. The reader is the resolver.
from __future__ import annotations

import functools
import json
import os
import re
import warnings

import numpy as np

from ..geometry.frames import crystalFrame, notIndexedFrame
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from ..maps.phaselist import distinctMinerals, phaseList
from .ang import eulerCorrectionRotation
from .ctf import _surfaceTilt
from .load import _mapFromRotations, ebsdFromColumns, phaseColumn, phaseShift, regularPoly, scanGrid, tslPointGroup, unmeasuredRows

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "hdf5_config")
# what EDAX writes into the coordinates of a pixel it did not measure
EDAX_NO_POSITION = -1111111.0
H5_HEADER_LIMIT = 16_000_000    # elements; a 4k SEM image is 8M, a pattern stack is not


@functools.cache
def hdf5Configs():
  """the HDF5 configurations of the vendors, read once from io/hdf5_config: a dict from the name
  to the parsed JSON"""
  configs = {}
  for name in sorted(os.listdir(CONFIG_DIR)):
    if name.endswith(".json"):
      with open(os.path.join(CONFIG_DIR, name)) as f:
        configs[name[:-5]] = json.load(f)
  return configs


# ------------------------------------------------------------------ the resolver

class H5Scope:
  """where an entry is searched: the key of its block, the data set, and the group holding it"""
  def __init__(self, root, set=None, free=None):
    self.root, self.set, self.free = root, root if set is None else set, root if free is None else free


def relativeTo(path, scope):
  """the path relative to the group scope, the path itself where it lies outside"""
  return path if scope == "/" else (path[len(scope):] if path.startswith(scope) else path)


def h5Resolve(paths, entry, scope):
  """the paths of a file that an entry matches, in the order the file lists them, matched case
  insensitively as MTEX's regexpi does: Bruker writes `X SAMPLE` where the config says `X Sample`"""
  if not isinstance(entry, dict) or "value" not in entry:
    return []
  value = str(entry["value"])
  mode = entry.get("mode", "search_root")
  if mode == "absolute":
    return [p for p in paths if p == value]
  under, relative = {"search_free": (scope.free, False), "search_set": (scope.set, True)}.get(mode, (scope.root, False))
  rx = re.compile(value, re.IGNORECASE)
  prefix = "" if under == "/" else under
  return [p for p in paths if p.startswith(prefix) and rx.search(relativeTo(p, under) if relative else p)]


def h5Read(f, path):
  """a data set as the plain array, string or scalar it holds"""
  d = f[path]
  if not hasattr(d, "shape"):
    return None
  v = d[()]
  if isinstance(v, bytes):
    return _text(v)
  if isinstance(v, np.ndarray):
    if v.dtype.kind in "SO":
      s = [_text(x) for x in v.reshape(-1)]
      return s[0] if len(s) == 1 else s
    if v.size == 1:
      return v.reshape(-1)[0].item()
    return v
  return v.item() if hasattr(v, "item") else v


# the phase a master pattern file is of, for a phase the file leaves unnamed: the file's name
# without its folder, its extension and whatever follows a '[' or '{', 'Fe(alpha-iron)' of
# 'Fe(alpha-iron)[A2]{20kV}.sht', as MATLAB's EMSphInx reader
def _masterName(path):
  return re.sub(r"[\[{].*$", "", re.sub(r"\.[^.]*$", "", re.sub(r"^.*[\\/]", "", str(path)))).strip()


def _text(x):
  if isinstance(x, bytes):
    x = x.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
  return str(x).strip()


def h5Paths(f):
  """every group and data set of a file, depth first, and which of them are groups"""
  paths, groups = [], set()
  def visit(name, obj):
    p = "/" + name
    paths.append(p)
    if not hasattr(obj, "shape"):
      groups.add(p)
  f.visititems(visit)
  return paths, groups


def h5ReadGroup(f, paths, groups, group):
  """a group of the file read whole as nested dicts, data sets above the header size limit left out"""
  out = {}
  for p in paths:
    rel = relativeTo(p, group + "/")
    if not p.startswith(group + "/") or "/" in rel:
      continue
    if p in groups:
      out[rel] = h5ReadGroup(f, paths, groups, p)
    else:
      try:
        if f[p].size <= H5_HEADER_LIMIT:
          out[rel] = h5Read(f, p)
      except Exception:
        pass
  return out


# ------------------------------------------------------------------ the values

def angleScale(angles):
  """whether angles are written in degrees or in radians, which the vendors do not agree on and
  no config states: a lattice angle above a full turn is degrees, as MTEX decides it too"""
  return degree if np.any(np.asarray(angles, dtype=float) > 2 * np.pi) else 1.0


def _floats(v):
  return None if v is None else np.asarray(v, dtype=float).reshape(-1)


def spaceGroupNumber(v):
  """the space group number a header value holds, None where it is no number"""
  if v is None:
    return None
  x = v[0] if isinstance(v, (list, np.ndarray)) and len(v) else v
  if isinstance(x, (list, np.ndarray)) or isinstance(x, str) or isinstance(x, bool):
    return None
  try:
    return int(round(float(x)))
  except (TypeError, ValueError):
    return None


def alignmentList(s):
  """the alignments a vendor states, "X||a*, Z||c" or EDAX's "a || x, c* || z", as crystalFrame takes them"""
  out = []
  for t in str(s).split(","):
    t = t.strip()
    if not t:
      continue
    if "||" in t:
      a, b = (x.strip() for x in t.split("||", 1))
      if a.upper() in ("X", "Y", "Z"):
        t = f"{a.upper()}||{b}"
      elif b.upper() in ("X", "Y", "Z"):
        t = f"{b.upper()}||{a}"
    out.append(t)
  return out


def eulerColumns(v):
  """the three angles a data set of them holds, written as 3 x n, n x 3 or laid out on the map
  as ny x nx x 3, and whether it holds them component by component"""
  m = np.asarray(v)
  if m.ndim > 2:
    m = m.reshape(-1, m.shape[-1]) if m.shape[-1] == 3 else m.reshape(m.shape[0], -1)
  # a 3 x 3 data set is read as columns
  if m.ndim == 2 and m.shape[1] == 3:
    return False, (m[:, 0], m[:, 1], m[:, 2])
  if m.ndim == 2 and m.shape[0] == 3:
    return True, (m[0], m[1], m[2])
  raise ValueError(f"a data set of Euler angles has three columns or three rows, not the shape {m.shape}")


def h5PhaseShift(values, nPhases):
  """how a vendor numbers its phases: a negative marker says it counts from zero with 255 for
  not indexed, which is -1 in the byte column it writes it in; else the rule of the text formats"""
  v = np.asarray(values)
  return 1 if v.size and v.min() < 0 else phaseShift(v, nPhases)


def loadedField(entry):
  """the name of the field an entry of a config loads from a table, the empty string if it loads none"""
  return str(entry["load"]["value"]).split(".")[-1] if isinstance(entry, dict) and "load" in entry else ""


def h5Loaded(entry, table):
  """the column of a table an entry of a config loads, None if it loads none"""
  if not (isinstance(entry, dict) and "load" in entry and table is not None):
    return None
  return table.get(loadedField(entry))


def matrixRotations(m):
  """the rotations a table of 3 x 3 matrices holds, written column by column, and which of
  them are no orientation at all: EDAX writes a matrix of zeros where a pattern failed"""
  M = np.asarray(m, dtype=float).reshape(-1, 3, 3).transpose(0, 2, 1)
  none = ~M.reshape(len(M), -1).any(axis=1)
  M = M.copy()
  M[none] = np.eye(3)
  return rotation.byMatrix(M), none


def _unitScale(entry, values):
  stated = entry.get("format", {}).get("data") if isinstance(entry, dict) else None
  if stated is None:
    return angleScale(values)
  return degree if "deg" in str(stated).lower() else 1.0


# ------------------------------------------------------------------ the pictures

IMAGE_MIN = 16              # pixels on a side, below which an array is no picture
IMAGE_CELLS = (("x cells", "y cells"), ("xcells", "ycells"), ("ncols", "nrows"), ("width", "height"))


def h5ImageIndex(f, paths, groups, n=0):
  """the pictures a file holds, as an index and not as pixels: a vendor writes its electron
  images beside the map, and one Oxford file carries a dozen of three megapixels each, so
  what the map keeps is where they are and what shape they have. h5Image reads one.

    from mtex.io.h5 import h5Paths
    f = h5py.File(mtexdatafile('EMSphinx'))
    images = h5ImageIndex(f, *h5Paths(f))              # every picture of the file
    images = h5ImageIndex(f, *h5Paths(f), n=508*955)   # leaving out the columns of a map of n pixels

  A picture is a numeric data set at least 16 pixels on a side: a plane, a plane of three or
  four colour channels either way round, or a flat run whose size the cell counts of the
  nearest Header group state. A group is never one. Every picture is a dict of its name, its
  path, its shape (rows, columns[, channels]) and its dtype. The loaders of the vendors' HDF5
  files store this index as ebsd.meta['images'].

  Input
    f - an open h5py file
    paths - every group and data set of the file, as h5Paths lists them
    groups - the set of those paths that are groups
  Output
    images - a list of dicts with the keys name, path, shape and dtype
  Keywords
    n - the number of measurements of the map; a data set of n rows, or of n values, is a
        column of the map and no picture
  See also
    h5Image, loadEBSD"""
  out = []
  for p in paths:
    if p in groups:
      continue
    d = f[p]
    if getattr(d, "dtype", None) is None or d.dtype.kind not in "iuf" or d.size < IMAGE_MIN * IMAGE_MIN:
      continue
    # the map's own columns are runs of one value per measurement, and the vendors write
    # them exactly as they write a flat picture; the map is not a picture of itself
    if n and (d.shape[0] == n or d.size == n):
      continue
    shape = _pictureShape(f, paths, groups, p, d)
    if shape is not None:
      out.append({"name": relativeTo(p, "/"), "path": p, "shape": shape, "dtype": str(d.dtype)})
  return out


# the shape a data set is a picture in, or None: a plane, a plane of three or four channels
# either way round, or a flat run of the pixels its own header counts
def _pictureShape(f, paths, groups, path, d):
  shape = tuple(int(v) for v in d.shape)
  if len(shape) == 3:
    if shape[0] in (1, 3, 4) and min(shape[1:]) >= IMAGE_MIN:
      return (shape[1], shape[2]) if shape[0] == 1 else (shape[1], shape[2], shape[0])
    if shape[2] in (3, 4) and min(shape[:2]) >= IMAGE_MIN:
      return shape
    return None
  if len(shape) == 2 and shape[1] not in (3, 4):
    return shape if min(shape) >= IMAGE_MIN else None
  # a flat run: the header of the group it sits under says how wide the picture is
  cells = _headerCells(f, paths, groups, path)
  if cells is None:
    return None
  nx, ny = cells
  channels = shape[1] if len(shape) == 2 else 1
  if nx * ny != shape[0] or min(nx, ny) < IMAGE_MIN:
    return None
  return (ny, nx) if channels == 1 else (ny, nx, channels)


# the cell counts of the nearest enclosing Header group, which is where a vendor states the
# size of the pictures beside it
def _headerCells(f, paths, groups, path):
  parts = path.split("/")
  for k in range(len(parts) - 1, 0, -1):
    scope = "/".join(parts[:k])
    header = next((g for g in groups if g.startswith(scope + "/") and relativeTo(g, scope + "/").lower() == "header"), None)
    if header is None:
      continue
    found = {}
    for q in paths:
      if q.startswith(header + "/") and "/" not in relativeTo(q, header + "/"):
        found[relativeTo(q, header + "/").lower()] = q
    for xKey, yKey in IMAGE_CELLS:
      if xKey in found and yKey in found:
        try:
          nx, ny = int(_floats(f[found[xKey]][()])[0]), int(_floats(f[found[yKey]][()])[0])
        except (ValueError, IndexError, TypeError):
          continue
        if nx > 0 and ny > 0:
          return nx, ny
  return None


# one picture of a file, ready to draw: a plane for a grey one, (height, width, 3) for a
# colour one, and the channels moved behind the pixels where the vendor wrote them first
def h5Image(fname, entry):
  """the pixels of a picture an EBSD file holds, by an entry of ebsd.meta['images']"""
  import h5py
  with h5py.File(os.fspath(fname), "r") as f:
    a = f[entry["path"]][()]
  shape = tuple(entry["shape"])
  if a.ndim == 3 and a.shape[0] in (1, 3, 4) and a.shape[1:] != shape[:2]:
    a = a[0] if a.shape[0] == 1 else np.moveaxis(a, 0, -1)
  a = np.asarray(a).reshape(shape)
  if a.ndim == 3 and a.shape[2] == 4:
    a = a[..., :3]
  # a colour picture is drawn from bytes or from fractions
  if a.ndim == 3 and a.dtype != np.uint8:
    hi = float(np.nanmax(a)) or 1.0
    a = np.clip(a / hi, 0, 1).astype(np.float32)
  return a


# ------------------------------------------------------------------ the reader

# the letters an X-ray line is named with, which no property name holds
GREEK = {"α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "h", "θ": "th", "ι": "i", "κ": "k", "λ": "l",
         "μ": "mu", "ν": "n", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "s", "τ": "t", "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "w"}


# one part of a property name as MATLAB's reader cleans it: the case kept, blanks and
# punctuation dropped, whatever else a name cannot hold an underscore
def _namePart(text):
  for g, latin in GREEK.items():
    text = text.replace(g, latin)
  text = re.sub(r"[ ,\-:|%~#\[\]()]", "", text)
  return re.sub(r"_{2,}", "_", re.sub(r"[^0-9A-Za-z_]", "_", text)).rstrip("_")


def additionName(field):
  """the property a data set of an additions group becomes, by its path below the group: the
  name in lower case without blanks, as "Band Contrast" is bandcontrast. A data set sorted into
  a subgroup that starts with a chemical element, as the "Co Ka1" an Oxford EDS map holds under
  "Window Integral" and again under "Peak Area", puts the element first, then the subgroup, the
  case kept: Co_WindowIntegral_Ka1, C_PeakArea_Kseries, Ca_Composition_Wt, as MATLAB's reader"""
  group, _, name = field.rpartition("/")
  m = re.match(r"([A-Z][a-z]?)\s+(\S.*)$", name.strip()) if group else None
  if m is None:
    return field.replace(" ", "").replace("/", "").lower()
  return "_".join([m.group(1)] + [_namePart(g) for g in group.split("/")] + [_namePart(m.group(2))])


class h5Reader:
  """an open HDF5 file read by the config of its vendor: its paths and groups listed once, the
  data set chosen, the scope the config's entries are searched in, and the data sets read so far"""

  def __init__(self, f, fname):
    self.f, self.fname = f, os.fspath(fname)
    self.paths, self.groups = h5Paths(f)
    self.scope = H5Scope("/")
    self.used = set()
    vendor = self._vendor()
    if vendor is None:
      raise ValueError(f"{self.fname} names no manufacturer any config claims; the configs are {', '.join(sorted(hdf5Configs()))}")
    self.name, self.conf = vendor
    self.ebsd = self.conf["ebsd"]
    self.sets = self._dataSets()
    if not self.sets:
      raise ValueError(f"{self.fname} holds no data set the {self.name} config recognises")
    self.chosen = None

  # ---------------------------------------------------------------- the resolver

  def one(self, entry, scope=None):
    """the first path an entry of the config matches, None if none does"""
    m = h5Resolve(self.paths, entry, self.scope if scope is None else scope)
    return m[0] if m else None

  def remember(self, entry):
    """the path an entry matches, recorded as read by the config, so that it is no property"""
    p = self.one(entry)
    if p is not None:
      self.used.add(p)
    return p

  def value(self, entry, scope=None):
    """the value of an entry: what the file holds at the path it names, or the constant it states"""
    if not isinstance(entry, dict):
      return None
    if "data" in entry:
      return entry["data"]
    p = self.one(entry, scope)
    return None if p is None else h5Read(self.f, p)

  def number(self, entry, default=None):
    """the first number an entry holds, the default where it holds none"""
    v = self.value(entry)
    return default if v is None else float(_floats(v)[0])

  # the vendor whose config claims the file: by the manufacturer key it states, or by the
  # extension where a config names one instead
  def _vendor(self):
    ext = os.path.splitext(self.fname)[1].lower()
    for name, conf in hdf5Configs().items():
      settings = conf["settings"]
      key = self.value(settings["key_path"])
      wanted = settings["manufacturer_keys"]["data"]
      wanted = [wanted] if isinstance(wanted, str) else list(wanted)
      keyText = "" if key is None else (" ".join(key) if isinstance(key, list) else str(key)).lower()
      for k in wanted:
        if (k.lower() == ext) if k.startswith(".") else (bool(keyText) and k.lower() in keyText):
          return name, conf
    return None

  # the data sets the file offers, in the preference order the config's alternatives state; a
  # match lying under another match is part of that one, so only the outermost are data sets
  def _dataSets(self):
    key = self.ebsd["key"]
    out = []
    for alt in str(key["value"]).split("|"):
      for p in h5Resolve(self.paths, {"mode": key.get("mode", "search_root"), "value": alt}, H5Scope("/")):
        if p not in out:
          out.append(p)
    outer = [p for p in out if not any(p.startswith(q + "/") for q in out)]
    return outer or out

  def choose(self, dataSet=None, verbose=False):
    """the data set read, by its number counted from zero or by part of its name, the first by
    default; its entries are searched from then on below it"""
    sets, fname = self.sets, self.fname
    if dataSet is None:
      chosen = sets[0]
    elif isinstance(dataSet, (int, np.integer)):
      if not 0 <= dataSet < len(sets):
        raise ValueError(f"{fname} holds {len(sets)} data sets, numbered 0 to {len(sets) - 1}")
      chosen = sets[dataSet]
    else:
      chosen = next((p for p in sets if str(dataSet) in p), None)
      if chosen is None:
        raise ValueError(f"no data set of {fname} is called {dataSet!r}; they are {', '.join(sets)}")
    # a file of several maps says which it holds and which one is read
    if len(sets) > 1 and verbose:
      print(f"  {self.name} file with {len(sets)} data sets:")
      for k, p in enumerate(sets):
        print(f"  {'▸' if p == chosen else ' '} [{k}] {p}")
      print(f"  pick one by EBSD.load(fname, dataSet={len(sets) - 1}) or dataSet={sets[-1].split('/')[-2]!r}")
    self.chosen = chosen
    self.scope = H5Scope(chosen, chosen, chosen.rsplit("/", 1)[0] or "/")
    return chosen

  # ---------------------------------------------------------------- the phases

  # the numbers of a lattice entry, from one data set or from separate scalars, None where one
  # is missing
  def _latticeValues(self, entry, scope):
    if not isinstance(entry, dict):
      return None
    if entry.get("type", "") == "separate":
      out = []
      for e in _separate(entry):
        v = self.value(e, scope)
        if v is None:
          return None
        out.append(float(_floats(v)[0]))
      return np.array(out)
    v = self.value(entry, scope)
    return None if v is None else _floats(v)

  # the cell a lattice block names, whatever of the three shapes a vendor writes it in: six
  # named scalars, a triple of lengths beside a triple of angles, or one row of six
  def _latticeCell(self, lat, scope):
    if not isinstance(lat, dict):
      return None
    if lat.get("type", "") == "all_together":
      v = self._latticeValues(lat, scope)
      return None if v is None or len(v) < 6 else (v[:3], v[3:6])
    dim = self._latticeValues(lat.get("dim"), scope)
    ang = self._latticeValues(lat.get("angle"), scope)
    if dim is None or ang is None or len(dim) < 3 or len(ang) < 3:
      return None
    return dim[:3], ang[:3]

  # the data sets a lattice was read from, for the writer
  def _latticePaths(self, entry, scope):
    if not isinstance(entry, dict):
      return []
    if entry.get("type", "") != "separate":
      p = self.one(entry, scope)
      return [] if p is None else [p]
    out = []
    for e in _separate(entry):
      p = self.one(e, scope)
      if p is None:
        return []
      out.append(p)
    return out

  # the point group of a phase: the space group number where the file states one, the EDAX
  # codes where it states those, and the triclinic group where it states neither
  def _pointGroup(self, entry, scope, idEntry=None):
    if isinstance(entry, dict) and entry.get("type", "") == "EDAX":
      code = self.value(entry.get("laue_code"), scope)
      if code is None:
        return "1"
      pg = self.value(entry.get("point_group_id"), scope)
      if pg is None or pg == "":
        pg = self.value(entry.get("point_group_name"), scope)
      return tslPointGroup(code if isinstance(code, str) else _floats(code)[0], None if pg is None or pg == "" else pg)
    n = spaceGroupNumber(self.value(entry, scope))
    if n is None:
      n = spaceGroupNumber(self.value(idEntry, scope))
    return n if n is not None and 1 <= n <= 230 else None

  def phases(self):
    """the crystal frames of the phase groups the config's cs block names, and where each was read"""
    frames = [notIndexedFrame()]
    prov = []
    cs = self.ebsd.get("cs")
    root = None if cs is None else self.one(cs["key"])
    if root is None:
      return phaseList(distinctMinerals(frames)), prov
    depth = root.count("/") + 1
    d = cs["default"]
    masters = self.value(self.ebsd.get("master_files"))
    masters = [] if masters is None else [masters] if isinstance(masters, str) else list(masters)
    for i, g in enumerate(p for p in self.paths if p.startswith(root + "/") and p.count("/") == depth and p in self.groups):
      inner = H5Scope(g, self.scope.set, self.scope.free)
      name = self.value(d.get("name"), inner)
      lat = d.get("lattice", {})
      cell = self._latticeCell(lat, inner)
      if cell is None:
        frames.append(notIndexedFrame())
        continue
      dim, ang = cell
      symbol = self._pointGroup(d.get("space_group"), inner, d.get("space_group_id"))
      reference = d.get("reference_frame", {})
      setup = self.value(reference, inner)
      if setup is None:
        setup = reference.get("fallback", "")
      mineral = "" if name is None else (" ".join(name) if isinstance(name, list) else str(name)).strip()
      if not mineral and i < len(masters):
        mineral = _masterName(masters[i])
      if isinstance(symbol, str):
        args, kw = [symbol], {}
      elif symbol:
        args, kw = [], {"spaceId": symbol}
      else:
        args, kw = ["-1"], {}
      frames.append(crystalFrame(*args, dim, angleScale(ang) * ang, *alignmentList(setup), mineral=mineral, **kw))
      whole = self._latticePaths(lat, inner) if lat.get("type", "") == "all_together" else []
      prov.append({"name": self.one(d.get("name"), inner), "cell": whole, "dim": self._latticePaths(lat.get("dim"), inner),
                   "angle": self._latticePaths(lat.get("angle"), inner), "angleScale": angleScale(ang)})
    return phaseList(distinctMinerals(frames)), prov

  # ---------------------------------------------------------------- the columns

  def table(self):
    """the compound data set the config reads whole as ebsd_data, its columns by field name, and its path"""
    entry = self.ebsd.get("ebsd_data")
    p = None if entry is None else self.one(entry)
    if p is None:
      return None, None
    rows = self.f[p][()]
    if not isinstance(rows, np.ndarray) or rows.dtype.names is None or rows.size == 0:
      return None, p
    return {k: rows[k] for k in rows.dtype.names}, p

  def additions(self, n):
    """the data sets of n values the config's additions groups hold beyond what it reads itself,
    by the key of the property each becomes; a file holds patterns beside its map, so the shape
    decides what is one value per pixel"""
    out = {}
    add = self.conf.get("additions")
    if not isinstance(add, dict):
      return out
    excluded = [re.compile(str(e)) for e in add.get("exclude", {}).get("data", [])]
    for g in (add["group"] if isinstance(add["group"], list) else [add["group"]]):
      root = self.one(g)
      if root is None:
        continue
      for p in self.paths:
        if not p.startswith(root + "/") or p in self.used or p in self.groups:
          continue
        field = relativeTo(p, root + "/")
        if any(e.search(field) for e in excluded):
          continue
        d = self.f[p]
        if d.size != n or d.dtype.kind not in "iufb":
          continue
        out.setdefault(additionName(field), p)
    return out

  # ---------------------------------------------------------------- the correction

  def surface(self):
    """the surface the map was acquired on, where the config names it: three Euler angles from
    that surface to the sample primary frame, CS1 to CS0, in degree"""
    entry = self.ebsd.get("acquisition_surface")
    v = self.value(entry) if isinstance(entry, dict) else None
    if v is None:
      return None
    a = _floats(v)
    if len(a) != 3:
      return None
    return tuple(float(x) for x in a * _unitScale(entry, a) / degree)

  def mapCorrection(self):
    """the rotation the config says the map is turned by against its Euler angles: a constant,
    the scan rotation the file writes on top of the flip about x, or a row of a table its own id picks"""
    entry = self.ebsd.get("map_correction")
    if not isinstance(entry, dict):
      return rotation.id
    kind = entry.get("type", "default")
    if kind == "rotation":
      return _bunge(entry, _floats(entry["data"]))
    if kind == "scanRotation":
      a = self.number(entry, float(entry.get("fallback", 0.0)))
      a = 0.0 if np.isnan(a) else a
      # the map runs y down, its z into the surface, where Z1 of the Euler frame points out of it
      turn = rotation.byAxisAngle(vector3d(0, 0, 1), a * _unitScale(entry, [a]))
      return rotation.byAxisAngle(vector3d(1, 0, 0), np.pi) * turn
    if kind == "by_id":
      rows = entry.get("correct_data", {}).get("data")
      if rows is None:
        return rotation.id
      # the id names the row it stands for, counting from one as the table is written; a
      # vendor that states no setting writes one outside the table, EDAX -1
      k = int(round(self.number(entry.get("id"), 0.0)))
      if not 1 <= k <= len(rows):
        return rotation.id
      row = rows[k - 1]
      if not isinstance(row, (list, tuple)):
        # a table of one column holds the angle about y
        return rotation.byAxisAngle(vector3d(0, 1, 0), float(row) * _unitScale(entry, [row]))
      return _bunge(entry, _floats(row))
    v = self.value(entry)
    return rotation.id if v is None else _bunge(entry, _floats(v))

  def correction(self, warn=False):
    """the rotation the Euler angles are corrected by: the config's map correction, after the
    tilt of the acquisition surface where the file refers its angles to CS0, as a .ctf; an EDAX
    file that states no alignment is read as its .ang files are"""
    correction = self.mapCorrection()
    surface = self.surface()
    tilt = _surfaceTilt(surface)
    if tilt is not None:
      correction = correction * tilt
      if warn:
        warnings.warn(f"{self.fname} states an acquisition surface of {surface} degree; its tilt is undone, since the file "
                      "refers its angles to CS0 and the map lies in CS1. Pass eulerCorrection to state the alignment yourself.",
                      stacklevel=3)
    if self.name.startswith("EDAX") and correction.angle() < 1e-9:
      correction = eulerCorrectionRotation(2)
    return correction

  def cellSteps(self):
    """the two steps of the cell the config's unitCell block names by fromStep, None for each it
    does not state"""
    uc = self.ebsd.get("unitCell")
    if not (isinstance(uc, dict) and uc.get("type") == "fromStep"):
      return None, None
    return self.number(uc["step_size_x"]), self.number(uc["step_size_y"])

  def header(self):
    """the header group of the data set, read whole as nested dicts: every vendor calls it Header"""
    s = self.scope.set
    root = next((p for p in self.paths if p.startswith(s + "/") and relativeTo(p, s + "/").lower() == "header"), None)
    return {} if root is None else h5ReadGroup(self.f, [p for p in self.paths if p.startswith(root + "/")], self.groups, root)


# the entries of a lattice entry of type separate, in the order of their names
def _separate(entry):
  return [entry[k] for k in sorted(entry) if k != "type"]


# a rotation of three Euler angles in the unit the entry states or the values suggest; the
# identity for fewer than three
def _bunge(entry, a):
  if len(a) < 3:
    return rotation.id
  a = a * _unitScale(entry, a)
  return rotation.byEuler(a[0], a[1], a[2], "Bunge")


# ------------------------------------------------------------------ the map

# the map of an HDF5 file of one of the vendors hdf5_config describes
#
#   loadEBSD('map.h5oina')                       the vendor read off the file
#   loadEBSD('map.oh5', dataSet=1)               a map by its number, counted from zero
#   loadEBSD('map.oh5', dataSet='OIM Map 2')     a map by part of its name
#
# The maps a file holds are listed in ebsd.meta['dataSets']; the header group of the one read
# is ebsd.meta['header'], and ebsd.meta['h5'] records where every value came from.
def _loadEBSD_h5(fname, *, dataSet=None, eulerCorrection=None, frame=None, scanUnit="um", verbose=True):
  import h5py
  fname = os.fspath(fname)
  with h5py.File(fname, "r") as f:
    r = h5Reader(f, fname)
    r.choose(dataSet, verbose)
    ebsd = r.ebsd
    table, tablePath = r.table()
    if tablePath is not None:
      r.used.add(tablePath)
    rot = ebsd["rotation"]
    o = _orientations(r, rot, table)
    n = o["n"]
    x, y, grid = _positions(r, n)
    unitCell = _unitCell(r, grid)

    # the phases, numbered from one unless the vendor counts from zero, and a number outside
    # the list a pixel that was not indexed
    phases, csProvenance = r.phases()
    phasePath = r.remember(ebsd["phase"])
    phaseInTable = h5Loaded(ebsd["phase"], table) is not None
    raw = h5Loaded(ebsd["phase"], table) if phaseInTable else r.value(ebsd["phase"])
    raw = np.zeros(n, dtype=np.int64) if raw is None else np.asarray(raw).reshape(-1)
    zeroBased = ebsd["phase"].get("type", "default") == "zeroBased"
    shift = 1 if zeroBased else h5PhaseShift(raw, len(phases) - 1)
    ph = phaseColumn(raw, shift, len(phases))

    # every per pixel data set the config does not read itself becomes a property, and every
    # field of a table but the orientations and the phases
    propPaths = r.additions(n)
    props = {k: f[p][()].reshape(-1) for k, p in propPaths.items()}
    tableFields = {}
    if table is not None:
      taken = (loadedField(rot), loadedField(ebsd["phase"]))
      for k, v in table.items():
        if k not in taken and v.ndim == 1:
          key = k.replace(" ", "").lower()
          props[key], tableFields[key] = v, k

    # a pixel EDAX did not measure carries its marker instead of a position; a measurement
    # without a place is none, so those rows are left out rather than placed at the marker
    placed = np.isfinite(x) & np.isfinite(y) & (x != EDAX_NO_POSITION) & (y != EDAX_NO_POSITION)
    keep = None
    if not placed.all():
      keep = np.flatnonzero(placed)
      x, y, ph = x[keep], y[keep], ph[keep]
      for k in ("rots", "noRotation", "phi1", "Phi", "phi2"):
        if o[k] is not None:
          o[k] = o[k][keep]
      props = {k: v[keep] for k, v in props.items()}
      grid = None

    surface = r.surface()
    correction = r.correction(warn=True) if eulerCorrection is None else eulerCorrection
    byMatrix = o["rots"] is not None
    record = {"file": os.path.abspath(fname), "rows": keep, "fileLength": n,
              "euler": {"paths": o["paths"], "componentsFirst": o["componentsFirst"], "scale": o["scale"]},
              "phase": {"path": phasePath, "shift": shift, "zeroBased": zeroBased},
              "cs": csProvenance, "props": propPaths, "correction": correction,
              # a map packed into one record per measurement: the table, and the fields of it
              "table": None if tablePath is None else {"path": tablePath, "matrixField": loadedField(rot) if byMatrix else None,
                                                       "phaseField": loadedField(ebsd["phase"]) if phaseInTable else None,
                                                       "propFields": tableFields}}
    meta = {"vendor": r.name, "dataSet": r.chosen, "dataSets": r.sets, "unplaced": 0 if keep is None else n - len(keep),
            "header": r.header(), "surface": surface, "images": h5ImageIndex(f, r.paths, r.groups, n), "h5": record}
    # the matrices are rotations already: the map is built from their quaternions
    if byMatrix:
      ph[np.asarray(o["noRotation"], dtype=bool)] = 0
      return _mapFromRotations(x, y, rotation(np.array(o["rots"].data, dtype=float)), ph, phases, props, shift,
                               eulerCorrection=correction, scanUnit=scanUnit, frame=frame, meta=meta, unitCell=unitCell,
                               scanGrid=grid)
    scale = o["scale"]
    e1, e2, e3 = o["phi1"] * scale, o["Phi"] * scale, o["phi2"] * scale
    unindexed = unmeasuredRows(e1, e2, e3, props) if r.name == "EMSphInx" else None
    return ebsdFromColumns(x, y, e1, e2, e3, ph, phases, props, shift=0, fileShift=shift, unindexed=unindexed, eulerCorrection=correction,
                           scanUnit=scanUnit, frame=frame, meta=meta, unitCell=unitCell, scanGrid=grid)


# the block of a config's rotation entry that names the Euler angles, by the type it states;
# empty for a table of matrices
def _eulerEntry(rot):
  kind = rot.get("type", "euler")
  return {} if kind == "byMatrix" else rot.get(kind, rot.get("euler", {}))


# the orientations, from one data set of three angles, from three of one, or from a table of
# matrices; euler and euler_stack differ in the shape of the data set alone
def _orientations(r, rot, table):
  byMatrix = rot.get("type", "euler") == "byMatrix"
  eu = _eulerEntry(rot)
  out = dict(paths=[], componentsFirst=True, rots=None, noRotation=None, phi1=None, Phi=None, phi2=None,
             scale=degree if "deg" in str(eu.get("format", {}).get("data", "radian")).lower() else 1.0)
  missing = f"{r.fname} holds no Euler angles where the {r.name} config looks"
  if byMatrix:
    m = h5Loaded(rot, table)
    if m is None:
      raise ValueError(f"{r.fname} holds no orientation matrices where the {r.name} config looks")
    out["rots"], out["noRotation"] = matrixRotations(m)
    out["n"] = len(out["noRotation"])
    return out
  entries = [eu["phi"]] if "phi" in eu else [eu[k] for k in ("phi1", "Phi", "phi2")]
  values = []
  for e in entries:
    p = r.remember(e)
    if p is not None:
      out["paths"].append(p)
    v = r.value(e)
    if v is None:
      raise ValueError(missing)
    values.append(v)
  if "phi" in eu:
    out["componentsFirst"], (out["phi1"], out["Phi"], out["phi2"]) = eulerColumns(values[0])
  else:
    out["phi1"], out["Phi"], out["phi2"] = (np.asarray(v, dtype=float).reshape(-1) for v in values)
  out["n"] = len(out["phi1"])
  return out


# the positions, either measured or implied by the step sizes and the grid, and the grid a
# header states
def _positions(r, n):
  pos = r.ebsd["position"]
  if pos.get("type", "direct") == "direct":
    r.remember(pos["direct"]["x"])
    r.remember(pos["direct"]["y"])
    x, y = r.value(pos["direct"]["x"]), r.value(pos["direct"]["y"])
    if x is None or y is None:
      raise ValueError(f"{r.fname} holds no positions the {r.name} config finds")
    return np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1), None
  ind = pos["indirect"]
  dx = r.number(ind["step_size_x"], 1.0)
  dy = r.number(ind["step_size_y"], dx)
  nx = int(r.number(ind["grid_size_x"], 0))
  ny = r.number(ind.get("grid_size_y"))
  if nx <= 0:
    raise ValueError(f"{r.fname} states no grid to place its measurements on")
  k = np.arange(n)
  grid = scanGrid(nx, int(ny), dx, dy) if ny is not None and dx > 0 and dy > 0 else None
  return dx * (k % nx), dy * (k // nx), grid


# the cell of a measurement: from the steps the config names, else from the grid's
def _unitCell(r, grid):
  uc = r.ebsd.get("unitCell")
  if isinstance(uc, dict) and uc.get("type") == "fromStep":
    sx, sy = r.cellSteps()
    return regularPoly(4, [sx, sy]) if sx is not None and sy is not None and sx > 0 and sy > 0 else None
  return None if grid is None else regularPoly(4, [grid.dx, grid.dy])



__all__ = ["h5Image", "h5ImageIndex"]
