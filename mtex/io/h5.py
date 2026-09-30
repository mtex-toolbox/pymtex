# the HDF5 maps of the vendors (MATLAB loadEBSD_h5 with interfaces/hdf5_config/*.json, Julia h5.jl)
#
#   ebsd = loadEBSD_h5('map.h5oina')                 the vendor read off the file
#   ebsd = loadEBSD_h5('map.oh5', dataSet=1)          one of the maps a file holds, by number or name
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

import json
import os
import re

import numpy as np

from ..geometry.frames import crystalFrame, notIndexedFrame
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from ..maps.phaselist import phaseList
from .ang import eulerCorrectionRotation
from .load import TSL_LAUE, ebsdFromColumns, phaseColumn, phaseShift, regularPoly, tslPointGroup, unmeasuredRows

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "hdf5_config")
# what EDAX writes into the coordinates of a pixel it did not measure
EDAX_NO_POSITION = -1111111.0
H5_HEADER_LIMIT = 16_000_000    # elements; a 4k SEM image is 8M, a pattern stack is not
_CONFIGS = None


def hdf5Configs():
  """the HDF5 configurations of the vendors, read once from io/hdf5_config: a dict from the name
  to the parsed JSON"""
  global _CONFIGS
  if _CONFIGS is None:
    _CONFIGS = {}
    for name in sorted(os.listdir(CONFIG_DIR)):
      if name.endswith(".json"):
        with open(os.path.join(CONFIG_DIR, name)) as f:
          _CONFIGS[name[:-5]] = json.load(f)
  return _CONFIGS


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
  under, relative = (scope.free, False) if mode == "search_free" else (scope.set, True) if mode == "search_set" else (scope.root, False)
  rx = re.compile(value, re.IGNORECASE)
  prefix = "" if under == "/" else under
  return [p for p in paths if p.startswith(prefix) and rx.search(relativeTo(p, under) if relative else p)]


def h5One(paths, entry, scope):
  """the first path of the file an entry of a config matches, None if none does"""
  m = h5Resolve(paths, entry, scope)
  return m[0] if m else None


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


def _text(x):
  if isinstance(x, bytes):
    x = x.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
  return str(x).strip()


def h5Value(f, paths, entry, scope):
  """the value of an entry: what the file holds at the path it names, or the constant it states"""
  if not isinstance(entry, dict):
    return None
  if "data" in entry:
    return entry["data"]
  p = h5One(paths, entry, scope)
  return None if p is None else h5Read(f, p)


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


def h5Vendor(f, paths, fname):
  """the vendor whose config claims the file: by the manufacturer key it states, or by the
  extension where a config names one instead"""
  ext = os.path.splitext(fname)[1].lower()
  for name, conf in hdf5Configs().items():
    settings = conf["settings"]
    key = h5Value(f, paths, settings["key_path"], H5Scope("/"))
    wanted = settings["manufacturer_keys"]["data"]
    wanted = [wanted] if isinstance(wanted, str) else list(wanted)
    keyText = "" if key is None else (" ".join(key) if isinstance(key, list) else str(key)).lower()
    for k in wanted:
      if (k.lower() == ext) if k.startswith(".") else (bool(keyText) and k.lower() in keyText):
        return name, conf
  return None


def h5DataSets(paths, conf):
  """the data sets a file offers, in the preference order the config's alternatives state; a
  match lying under another match is part of that one, so only the outermost are data sets"""
  key = conf["ebsd"]["key"]
  out = []
  for alt in str(key["value"]).split("|"):
    for p in h5Resolve(paths, {"mode": key.get("mode", "search_root"), "value": alt}, H5Scope("/")):
      if p not in out:
        out.append(p)
  outer = [p for p in out if not any(p.startswith(q + "/") for q in out)]
  return outer or out


# ------------------------------------------------------------------ the phases

def angleScale(angles):
  """whether angles are written in degrees or in radians, which the vendors do not agree on and
  no config states: a lattice angle above a full turn is degrees, as MTEX decides it too"""
  return degree if np.any(np.asarray(angles, dtype=float) > 2 * np.pi) else 1.0


def _floats(v):
  if v is None:
    return None
  a = np.asarray(v, dtype=float).reshape(-1)
  return a


def h5LatticeValues(f, paths, entry, scope):
  """the numbers of a lattice entry of a config, from one data set or from separate scalars,
  None where one is missing"""
  if not isinstance(entry, dict):
    return None
  if entry.get("type", "") == "separate":
    out = []
    for k in sorted(entry):
      if k == "type":
        continue
      v = h5Value(f, paths, entry[k], scope)
      if v is None:
        return None
      out.append(float(_floats(v)[0]))
    return np.array(out)
  v = h5Value(f, paths, entry, scope)
  return None if v is None else _floats(v)


def h5LatticeCell(f, paths, lat, scope):
  """the cell a config's lattice block names, whatever of the three shapes a vendor writes it in:
  six named scalars, a triple of lengths beside a triple of angles, or one row of six"""
  if not isinstance(lat, dict):
    return None
  if lat.get("type", "") == "all_together":
    v = h5LatticeValues(f, paths, lat, scope)
    return None if v is None or len(v) < 6 else (v[:3], v[3:6])
  dim = h5LatticeValues(f, paths, lat.get("dim"), scope)
  ang = h5LatticeValues(f, paths, lat.get("angle"), scope)
  if dim is None or ang is None or len(dim) < 3 or len(ang) < 3:
    return None
  return dim[:3], ang[:3]


def h5LatticePaths(paths, entry, scope):
  """the data sets a lattice was read from, for the writer"""
  if not isinstance(entry, dict):
    return []
  if entry.get("type", "") != "separate":
    p = h5One(paths, entry, scope)
    return [] if p is None else [p]
  out = []
  for k in sorted(entry):
    if k == "type":
      continue
    p = h5One(paths, entry[k], scope)
    if p is None:
      return []
    out.append(p)
  return out


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


def h5PointGroup(f, paths, entry, scope, idEntry=None):
  """the point group of a phase: the space group number where the file states one, the EDAX
  codes where it states those, and the triclinic group where it states neither"""
  if isinstance(entry, dict) and entry.get("type", "") == "EDAX":
    code = h5Value(f, paths, entry.get("laue_code"), scope)
    if code is None:
      return "1"
    pg = h5Value(f, paths, entry.get("point_group_id"), scope) if "point_group_id" in entry else None
    if pg is None or pg == "":
      pg = h5Value(f, paths, entry.get("point_group_name"), scope) if "point_group_name" in entry else None
    return tslPointGroup(_floats(code)[0] if not isinstance(code, str) else code, pg if pg not in ("", None) else None)
  n = spaceGroupNumber(h5Value(f, paths, entry, scope))
  if n is None:
    n = spaceGroupNumber(h5Value(f, paths, idEntry, scope))
  return n if n is not None and 1 <= n <= 230 else None


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


def h5Phases(f, paths, groups, conf, scope):
  """the crystal frames of the phase groups a config's cs block names, and where each was read"""
  frames = [notIndexedFrame()]
  prov = []
  cs = conf["ebsd"].get("cs")
  if cs is None:
    return phaseList(frames), prov
  root = h5One(paths, cs["key"], scope)
  if root is None:
    return phaseList(frames), prov
  depth = root.count("/") + 1
  d = cs["default"]
  for g in [p for p in paths if p.startswith(root + "/") and p.count("/") == depth and p in groups]:
    inner = H5Scope(g, scope.set, scope.free)
    name = h5Value(f, paths, d.get("name"), inner)
    lat = d.get("lattice", {})
    cell = h5LatticeCell(f, paths, lat, inner)
    if cell is None:
      frames.append(notIndexedFrame())
      continue
    dim, ang = cell
    symbol = h5PointGroup(f, paths, d.get("space_group"), inner, d.get("space_group_id"))
    reference = d.get("reference_frame", {})
    setup = h5Value(f, paths, reference, inner)
    if setup is None:
      setup = reference.get("fallback", "")
    mineral = "" if name is None else (" ".join(name) if isinstance(name, list) else str(name)).strip()
    args = [symbol] if isinstance(symbol, str) else []
    kw = {} if isinstance(symbol, str) else ({"spaceId": symbol} if symbol else {})
    if not args and not kw:
      args = ["-1"]
    frames.append(crystalFrame(*args, dim, angleScale(ang) * ang, *alignmentList(setup), mineral=mineral, **kw))
    whole = h5LatticePaths(paths, lat, inner) if lat.get("type", "") == "all_together" else []
    prov.append({"name": h5One(paths, d.get("name"), inner), "cell": whole, "dim": h5LatticePaths(paths, lat.get("dim"), inner),
                 "angle": h5LatticePaths(paths, lat.get("angle"), inner), "angleScale": angleScale(ang)})
  return phaseList(frames), prov


# ------------------------------------------------------------------ the orientations and the phase column

def eulerColumns(v):
  """the three angles a data set of them holds, written as 3 x n, n x 3 or laid out on the map
  as ny x nx x 3, and whether it holds them component by component"""
  m = np.asarray(v)
  if m.ndim > 2:
    m = m.reshape(-1, m.shape[-1]) if m.shape[-1] == 3 else m.reshape(m.shape[0], -1)
  if m.ndim == 2 and m.shape[0] == 3 and m.shape[1] != 3:
    return True, (m[0], m[1], m[2])
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


def h5Table(f, paths, ebsd, scope):
  """the compound data set a config reads whole as ebsd_data, its columns by field name"""
  entry = ebsd.get("ebsd_data")
  if entry is None:
    return None, None
  p = h5One(paths, entry, scope)
  if p is None:
    return None, None
  rows = f[p][()]
  if not isinstance(rows, np.ndarray) or rows.dtype.names is None or rows.size == 0:
    return None, p
  return {k: rows[k] for k in rows.dtype.names}, p


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


# ------------------------------------------------------------------ the correction

def _unitScale(entry, values):
  stated = entry.get("format", {}).get("data") if isinstance(entry, dict) else None
  if stated is None:
    return angleScale(values)
  return degree if "deg" in str(stated).lower() else 1.0


def h5Surface(f, paths, conf, scope):
  """the surface the map was acquired on, where the config names it: three Euler angles from
  that surface to the sample primary frame, CS1 to CS0, in degree"""
  entry = conf["ebsd"].get("acquisition_surface")
  if not isinstance(entry, dict):
    return None
  v = h5Value(f, paths, entry, scope)
  if v is None:
    return None
  a = _floats(v)
  if len(a) != 3:
    return None
  scale = _unitScale(entry, a)
  return tuple(float(x) for x in a * scale / degree)


def h5MapCorrection(f, paths, conf, scope):
  """the rotation the config says the map is turned by against its Euler angles: a constant,
  the scan rotation the file writes, or a row of a table its own id picks"""
  entry = conf["ebsd"].get("map_correction")
  if not isinstance(entry, dict):
    return rotation.id
  kind = entry.get("type", "default")
  bunge = lambda a: rotation.byEuler(a[0], a[1], a[2], "Bunge")
  if kind == "rotation":
    a = _floats(entry["data"])
    return bunge(a * _unitScale(entry, a))
  if kind == "scanRotation":
    v = h5Value(f, paths, entry, scope)
    a = float(entry.get("fallback", 0.0)) if v is None else float(_floats(v)[0])
    if np.isnan(a):
      a = 0.0
    return rotation.byAxisAngle(vector3d(0, 0, 1), a * _unitScale(entry, [a]))
  if kind == "by_id":
    rows = entry.get("correct_data", {}).get("data")
    if rows is None:
      return rotation.id
    idv = h5Value(f, paths, entry.get("id"), scope)
    # the id names the row it stands for, counting from one as the table is written; a
    # vendor that states no setting writes one outside the table, EDAX -1
    k = 0 if idv is None else int(round(float(_floats(idv)[0])))
    if not 1 <= k <= len(rows):
      return rotation.id
    row = rows[k - 1]
    if not isinstance(row, (list, tuple)):
      # a table of one column holds the angle about y
      return rotation.byAxisAngle(vector3d(0, 1, 0), float(row) * _unitScale(entry, [row]))
    a = _floats(row)
    return bunge(a * _unitScale(entry, a)) if len(a) >= 3 else rotation.id
  v = h5Value(f, paths, entry, scope)
  if v is None:
    return rotation.id
  a = _floats(v)
  return bunge(a * _unitScale(entry, a)) if len(a) >= 3 else rotation.id


# ------------------------------------------------------------------ the header

def h5Header(f, paths, groups, scope):
  """the header group of a data set, read whole as nested dicts: every vendor calls it Header"""
  root = next((p for p in paths if p.startswith(scope.set + "/") and relativeTo(p, scope.set + "/").lower() == "header"), None)
  return {} if root is None else h5ReadGroup(f, [p for p in paths if p.startswith(root + "/")], groups, root)


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


# ------------------------------------------------------------------ the pictures

IMAGE_MIN = 16              # pixels on a side, below which an array is no picture
IMAGE_CELLS = (("x cells", "y cells"), ("xcells", "ycells"), ("ncols", "nrows"), ("width", "height"))


def h5ImageIndex(f, paths, groups, n=0):
  """the pictures a file holds, as an index and not as pixels: a vendor writes its electron
  images beside the map, and one Oxford file carries a dozen of three megapixels each, so
  what the map keeps is where they are and what shape they have. h5Image reads one."""
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


# ------------------------------------------------------------------ the map

def loadEBSD_h5(fname, *, dataSet=None, eulerCorrection=None, frame=None, scanUnit="um", verbose=True):
  """the map of an HDF5 file of one of the vendors hdf5_config describes

    loadEBSD_h5('map.h5oina')                       the vendor read off the file
    loadEBSD_h5('map.oh5', dataSet=1)               a map by its number, counted from zero
    loadEBSD_h5('map.oh5', dataSet='OIM Map 2')     a map by part of its name

  The maps a file holds are listed in ebsd.meta['dataSets']; the header group of the one read
  is ebsd.meta['header'], and ebsd.meta['h5'] records where every value came from.
  """
  import h5py
  fname = os.fspath(fname)
  with h5py.File(fname, "r") as f:
    paths, groups = h5Paths(f)
    vendor = h5Vendor(f, paths, fname)
    if vendor is None:
      raise ValueError(f"{fname} names no manufacturer any config claims; the configs are {', '.join(sorted(hdf5Configs()))}")
    name, conf = vendor
    sets = h5DataSets(paths, conf)
    if not sets:
      raise ValueError(f"{fname} holds no data set the {name} config recognises")
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
      print(f"  {name} file with {len(sets)} data sets:")
      for k, p in enumerate(sets):
        print(f"  {'▸' if p == chosen else ' '} [{k}] {p}")
      print(f"  pick one by EBSD.load(fname, dataSet={len(sets) - 1}) or dataSet={sets[-1].split('/')[-2]!r}")
    parent = chosen.rsplit("/", 1)[0] or "/"
    scope = H5Scope(chosen, chosen, parent)
    ebsd = conf["ebsd"]
    used = set()

    def remember(entry):
      p = h5One(paths, entry, scope)
      if p is not None:
        used.add(p)
      return p

    # the orientations, from one data set of three angles, from three of one, or from a
    # table of matrices; euler and euler_stack differ in the shape of the data set alone
    table, tablePath = h5Table(f, paths, ebsd, scope)
    if tablePath is not None:
      used.add(tablePath)
    rot = ebsd["rotation"]
    byMatrix = rot.get("type", "euler") == "byMatrix"
    eu = {} if byMatrix else rot.get(rot.get("type", "euler"), rot.get("euler", {}))
    scale = degree if "deg" in str(eu.get("format", {}).get("data", "radian")).lower() else 1.0
    eulerPaths, componentsFirst, rots, noRotation = [], True, None, None
    if byMatrix:
      m = h5Loaded(rot, table)
      if m is None:
        raise ValueError(f"{fname} holds no orientation matrices where the {name} config looks")
      rots, noRotation = matrixRotations(m)
      phi1 = Phi = phi2 = None
      n = len(noRotation)
    elif "phi" in eu:
      p = remember(eu["phi"])
      if p is not None:
        eulerPaths.append(p)
      v = h5Value(f, paths, eu["phi"], scope)
      if v is None:
        raise ValueError(f"{fname} holds no Euler angles where the {name} config looks")
      componentsFirst, (phi1, Phi, phi2) = eulerColumns(v)
      n = len(phi1)
    else:
      cols = []
      for k in ("phi1", "Phi", "phi2"):
        p = remember(eu[k])
        if p is not None:
          eulerPaths.append(p)
        v = h5Value(f, paths, eu[k], scope)
        if v is None:
          raise ValueError(f"{fname} holds no Euler angles where the {name} config looks")
        cols.append(np.asarray(v, dtype=float).reshape(-1))
      phi1, Phi, phi2 = cols
      n = len(phi1)

    # the positions, either measured or implied by the step sizes and the grid
    pos = ebsd["position"]
    scanGrid = None
    if pos.get("type", "direct") == "direct":
      remember(pos["direct"]["x"])
      remember(pos["direct"]["y"])
      x, y = h5Value(f, paths, pos["direct"]["x"], scope), h5Value(f, paths, pos["direct"]["y"], scope)
      if x is None or y is None:
        raise ValueError(f"{fname} holds no positions the {name} config finds")
      x, y = np.asarray(x, dtype=float).reshape(-1), np.asarray(y, dtype=float).reshape(-1)
    else:
      ind = pos["indirect"]
      dx = h5Value(f, paths, ind["step_size_x"], scope)
      dx = 1.0 if dx is None else float(_floats(dx)[0])
      dy = h5Value(f, paths, ind["step_size_y"], scope)
      dy = dx if dy is None else float(_floats(dy)[0])
      nx = h5Value(f, paths, ind["grid_size_x"], scope)
      nx = 0 if nx is None else int(_floats(nx)[0])
      ny = h5Value(f, paths, ind.get("grid_size_y"), scope)
      if nx <= 0:
        raise ValueError(f"{fname} states no grid to place its measurements on")
      k = np.arange(n)
      x, y = dx * (k % nx), dy * (k // nx)
      if ny is not None and dx > 0 and dy > 0:
        scanGrid = (nx, int(_floats(ny)[0]), dx, dy)
    unitCell = None
    uc = ebsd.get("unitCell")
    if isinstance(uc, dict) and uc.get("type") == "fromStep":
      sx, sy = h5Value(f, paths, uc["step_size_x"], scope), h5Value(f, paths, uc["step_size_y"], scope)
      if sx is not None and sy is not None:
        sx, sy = float(_floats(sx)[0]), float(_floats(sy)[0])
        if sx > 0 and sy > 0:
          unitCell = regularPoly(4, [sx, sy])
    elif scanGrid is not None:
      unitCell = regularPoly(4, [scanGrid[2], scanGrid[3]])

    # the phases, numbered from one unless the vendor counts from zero, and a number outside
    # the list a pixel that was not indexed
    phases, csProvenance = h5Phases(f, paths, groups, conf, scope)
    phasePath = remember(ebsd["phase"])
    raw = h5Loaded(ebsd["phase"], table)
    if raw is None:
      raw = h5Value(f, paths, ebsd["phase"], scope)
    raw = np.zeros(n, dtype=np.int64) if raw is None else np.asarray(raw).reshape(-1)
    zeroBased = ebsd["phase"].get("type", "default") == "zeroBased"
    shift = 1 if zeroBased else h5PhaseShift(raw, len(phases) - 1)
    ph = phaseColumn(raw, shift, len(phases))

    # every per pixel data set the config does not read itself becomes a property; the shape
    # decides what is one value per pixel, since a file holds patterns beside its map
    props, propPaths = {}, {}
    add = conf.get("additions")
    if isinstance(add, dict):
      excluded = [re.compile(str(e)) for e in add.get("exclude", {}).get("data", [])]
      for g in (add["group"] if isinstance(add["group"], list) else [add["group"]]):
        root = h5One(paths, g, scope)
        if root is None:
          continue
        for p in paths:
          if not p.startswith(root + "/") or p in used or p in groups:
            continue
          field = relativeTo(p, root + "/")
          if any(e.search(field) for e in excluded):
            continue
          d = f[p]
          if d.size != n or d.dtype.kind not in "iufb":
            continue
          key = field.replace(" ", "").replace("/", "").lower()
          if key in props:
            continue
          props[key] = d[()].reshape(-1)
          propPaths[key] = p
    tableFields = {}
    if table is not None:
      taken = (loadedField(rot), loadedField(ebsd["phase"]))
      for k, v in table.items():
        if k not in taken and v.ndim == 1:
          props[k.replace(" ", "").lower()] = v
          tableFields[k.replace(" ", "").lower()] = k

    # a pixel EDAX did not measure carries its marker instead of a position; a measurement
    # without a place is none, so those rows are left out rather than placed at the marker
    placed = np.isfinite(x) & np.isfinite(y) & (x != EDAX_NO_POSITION) & (y != EDAX_NO_POSITION)
    keep = None
    if not placed.all():
      keep = np.flatnonzero(placed)
      x, y, ph = x[keep], y[keep], ph[keep]
      if byMatrix:
        rots, noRotation = rots[keep], noRotation[keep]
      else:
        phi1, Phi, phi2 = phi1[keep], Phi[keep], phi2[keep]
      props = {k: v[keep] for k, v in props.items()}
      scanGrid = None

    surface = h5Surface(f, paths, conf, scope)
    if eulerCorrection is None:
      correction = h5MapCorrection(f, paths, conf, scope)
      # a file that refers its angles to CS0 has the tilt of its surface undone first, as a .ctf
      if surface is not None and (abs(surface[1]) > 1e-6 or abs(surface[2]) > 1e-6):
        correction = correction * rotation.byEuler(0.0, surface[1] * degree, surface[2] * degree, "Bunge").inv()
        import warnings
        warnings.warn(f"{fname} states an acquisition surface of {surface} degree; its tilt is undone, since the file "
                      "refers its angles to CS0 and the map lies in CS1. Pass eulerCorrection to state the alignment yourself.",
                      stacklevel=2)
      # an EDAX file that states no alignment is read as its .ang files are
      if name.startswith("EDAX") and correction.angle() < 1e-9:
        correction = eulerCorrectionRotation(2)
    else:
      correction = eulerCorrection
    record = {"file": os.path.abspath(fname), "rows": keep, "fileLength": n,
              "euler": {"paths": eulerPaths, "componentsFirst": componentsFirst, "scale": scale},
              "phase": {"path": phasePath, "shift": shift, "zeroBased": zeroBased},
              "cs": csProvenance, "props": propPaths, "correction": correction,
              # a map packed into one record per measurement: the table, and the fields of it
              "table": None if tablePath is None else {"path": tablePath, "matrixField": loadedField(rot) if byMatrix else None,
                                                       "phaseField": loadedField(ebsd["phase"]) if h5Loaded(ebsd["phase"], table) is not None else None,
                                                       "propFields": tableFields}}
    meta = {"vendor": name, "dataSet": chosen, "dataSets": sets, "unplaced": 0 if keep is None else n - len(keep),
            "header": h5Header(f, paths, groups, scope), "surface": surface, "images": h5ImageIndex(f, paths, groups, n),
            "h5": record}
    if byMatrix:
      q = np.array(rots.data, dtype=float)
      # the matrices are rotations already; ebsdFromColumns takes Euler angles, so they go in as Matthies angles of the identity
      # is not possible: the map is built here from the quaternions instead
      return _fromRotations(x, y, q, ph, phases, props, noRotation, correction, scanUnit, frame, meta, unitCell, scanGrid)
    e1, e2, e3 = phi1 * scale, Phi * scale, phi2 * scale
    unindexed = unmeasuredRows(e1, e2, e3, props) if name == "EMSphInx" else None
    return ebsdFromColumns(x, y, e1, e2, e3, ph, phases, props, shift=0, unindexed=unindexed, eulerCorrection=correction,
                           scanUnit=scanUnit, frame=frame, meta=meta, unitCell=unitCell, scanGrid=scanGrid)


# the same map from the rotations a file states directly, as quaternions rather than angles
def _fromRotations(x, y, q, ph, phases, props, unindexed, correction, scanUnit, frame, meta, unitCell, scanGrid):
  from .load import mapFromColumns, rowOrderLattice
  ph = ph.copy()
  if unindexed is not None:
    ph[np.asarray(unindexed, dtype=bool)] = 0
  r = rotation(q)
  if correction is not None:
    r = correction * r
    meta = dict(meta or {}, EulerCorrection=correction)
  # the rotations were just made, so their array is changed in place
  q = np.asarray(r.data, dtype=float)
  indexed = np.array([fr.isIndexed for fr in phases.frames])[ph]
  q[~indexed] = np.nan
  x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
  lat = rowOrderLattice(x, y, scanGrid)
  return mapFromColumns(x, y, q, ph, phases, props, lat, unitCell=unitCell, scanUnit=scanUnit, frame=frame, meta=meta,
                        phaseMap=np.arange(len(phases)))


__all__ = ["loadEBSD_h5", "h5Image", "h5ImageIndex"]
