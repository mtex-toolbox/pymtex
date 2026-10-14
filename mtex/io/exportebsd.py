# writing a map to the vendor formats (MATLAB @EBSD/export, exportEBSD_ang, exportEBSD_ctf, exportEBSD_h5; Julia export.jl)
#
#   export(ebsd, 'map.ang')        TSL / EDAX
#   export(ebsd, 'map.ctf')        Oxford / Channel
#   export(ebsd, 'map.h5oina')     HDF5, written into a copy of the file the map came from
#   export(ebsd, 'map.txt')        a plain table, x y phi1 Phi phi2 phase and the properties
#
# An exporter undoes what the loader of its format applies (branch ADR 0001): the Euler
# correction comes back out, so a map written and read again is the map it came from. What
# the header states is what the port knows, the phases with their lattices, the step of the
# grid, the columns, and what the file the map came from stated is written back where the
# format has a line for it, since a header of zeros makes a file no vendor tool reads.
from __future__ import annotations

import os
import shutil
import warnings

import numpy as np

from ..functions.polefigure import PoleFigure
from ..geometry.frames import crystalFrame
from ..geometry.quaternion import rotation
from ..geometry.representations import EULER_LABELS, eulerConvention
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .ang import eulerCorrectionRotation
from .ctf import oxfordCorrection
from .kernels import formatTable
from .load import CHANNEL_LAUE, TSL_LAUE

# the TSL and Channel codes of the eleven Laue groups, the tables of load.py read backwards
TSL_CODES = {v: k for k, v in TSL_LAUE.items() if k != 2}


# the Laue class of a frame in the spelling the vendor tables use: a monoclinic or trigonal
# setting, 12/m1 or -3m1, is its class without the ones
def _laueName(fr):
  name = fr.sym.LaueName()
  if name in CHANNEL_LAUE:
    return name
  short = name.replace("1", "") if name.replace("1", "") in CHANNEL_LAUE else name
  return short


def tslCode(fr):
  """the TSL symmetry code of the Laue class of a frame, as an .ang file writes it; 1 when there is none"""
  return TSL_CODES.get(_laueName(fr), 1)


def channelCode(fr):
  """the Channel 5 Laue group number of a frame, as a .ctf file writes it; 1 when there is none"""
  name = _laueName(fr)
  return CHANNEL_LAUE.index(name) + 1 if name in CHANNEL_LAUE else 1


def headerValue(ebsd, names, default):
  """a value the header of the file the map came from states, under any of the names given"""
  h = ebsd.meta.get("header")
  if not isinstance(h, dict):
    return default
  flat = {}
  for k, v in h.items():
    if isinstance(v, dict):
      for k2, v2 in v.items():
        flat.setdefault(str(k2).lower(), v2)
    else:
      flat.setdefault(str(k).lower(), v)
  for name in names:
    if name.lower() in flat:
      return flat[name.lower()]
  return default


def _headerNumber(ebsd, names, default):
  v = headerValue(ebsd, names, None)
  try:
    return float(v) if v is not None and v != "" else default
  except (TypeError, ValueError):
    return default


def propertyColumn(ebsd, names, default):
  """the first property the format names, under any of the names the vendors give it"""
  for name in names:
    if name in ebsd.prop:
      return np.asarray(ebsd.prop[name], dtype=float).reshape(-1)
  return np.full(ebsd.size, float(default))


def gridStep(values):
  """the step of a grid, the smallest step actually between two of its coordinates"""
  u = np.unique(np.round(values, 6))
  return float(np.diff(u).min()) if len(u) > 1 else 1.0


def scanOf(ebsd):
  """the scan a header states: the two steps, the points of the first two rows and the rows;
  from the lattice where the map has one, since a hexagonal scan's x coordinates are half
  a step apart between its rows, and from the coordinates otherwise"""
  pos = ebsd.pos.data.reshape(-1, 3)
  xs, ys = pos[:, 0], pos[:, 1]
  l = ebsd.lattice
  hex = l is not None and l.shape == "hex"
  if l is not None:
    stepX, stepY = float(abs(l.A[0, 0])), float(abs(l.A[1, 1]))
  else:
    stepX, stepY = gridStep(xs), gridStep(ys)
  rows = np.unique(np.round(ys, 6))
  if ebsd.ndim == 2:
    nRows, nOdd = ebsd.shape[0], ebsd.shape[1]
    nEven = nOdd - 1 if hex else nOdd
  else:
    counts = np.bincount(np.searchsorted(rows, np.round(ys, 6)))
    nRows, nOdd, nEven = len(rows), int(counts[0]), int(counts[1] if len(counts) > 1 else counts[0])
  return stepX, stepY, nOdd, nEven, nRows, hex


def exportPhaseNumbers(ebsd):
  """the phase numbers a file writes: the phases the map measures, from one, and zero where
  nothing was indexed. Only the phases that occur are declared: a file that declares a phase
  it never uses cannot be told from one that counts its phases from zero, and the rule that
  decides it is the reader's, so the writer does not leave it the choice."""
  ids = ebsd.phaseId.reshape(-1)
  present = np.unique(ids)
  indexed = [int(j) for j in present if ebsd.CSList.isIndexed(int(j))]
  numbers = np.zeros(len(ebsd.CSList), dtype=np.int64)
  for k, j in enumerate(indexed):
    numbers[j] = k + 1
  return numbers[ids], indexed


# the rotations in the crystal frames the reader builds: a file states a phase's point group and
# lattice, not the alignment of its Euclidean axes, which the reader fixes (EDAX's for .ang, the
# default for .ctf)
def _inFileFrames(ebsd, r, alignment):
  from ..geometry.orientation import orientation
  ids = ebsd.phaseId.reshape(-1)
  q = r.data.copy()
  for j in np.unique(ids):
    if not ebsd.CSList.isIndexed(int(j)):
      continue
    fr = ebsd.CSList.frames[int(j)]
    fileFrame = crystalFrame(fr.pointGroup, fr.abc, fr.abg, *alignment, mineral=fr.mineral)
    if np.allclose(fr.transformationMatrix(fileFrame), np.eye(3), atol=1e-10):
      continue
    sel = ids == j
    o = orientation(q[sel], fr)
    q[sel] = o.transformReferenceFrame(fileFrame).data.reshape(-1, 4)
  return rotation(q)


# the Bunge angles the file gets: the correction taken out, zeros where a pixel has none
def _eulerColumns(ebsd, eulerCorrection, scale, alignment=()):
  r = rotation(ebsd.rotations.data.reshape(-1, 4))
  r = _inFileFrames(ebsd, r, alignment)
  if eulerCorrection is not None:
    r = eulerCorrection.inv() * r
  a, b, c = (np.asarray(v, dtype=float) for v in r.Euler("Bunge"))
  none = ~np.isfinite(a)
  return tuple(np.where(none, 0.0, v) * scale for v in (a, b, c))


def export(ebsd, fname, *args, **kwargs):
  """write a map to the file the extension names

    export(ebsd, 'map.ang')  # TSL / EDAX, the correction of setting 2 undone
    export(ebsd, 'map.ctf')  # Oxford / Channel, the half turn about y undone
    export(ebsd, 'map.h5oina')  # into a copy of the file the map came from
    export(ebsd, 'map.txt', 'ABG', radians=True)  # a table of x y, the Euler angles, the phase and the properties
    export(v, 'vectors.txt')  # a list of directions as a table, see exportVector3d
    export(pf, 'dubna', degree=True)  # a pole figure data set, one table per pole figure"""
  # an orientation list is a table of its angles
  from ..geometry.quaternion import quaternion
  if isinstance(ebsd, quaternion):
    from .exportorientation import exportOrientation
    return exportOrientation(ebsd, fname, *args, **kwargs)
  # a function on SO(3) goes to its own writers
  from ..functions.so3funclass import SO3Fun
  if isinstance(ebsd, SO3Fun):
    from .exportodf import exportODF
    return exportODF(ebsd, fname, *args, **kwargs)
  # a pole figure data set goes to one table per pole figure
  if isinstance(ebsd, PoleFigure):
    from .exportpolefigure import exportPoleFigure
    return exportPoleFigure(ebsd, fname, *args, **kwargs)
  fname = os.fspath(fname)
  # a list of directions goes to its own table writer
  if isinstance(ebsd, vector3d):
    from .vectors import exportVector3d
    return exportVector3d(ebsd, fname, *args, **kwargs)
  ext = os.path.splitext(fname)[1].lower()
  if ext == ".ang":
    return _exportEBSD_ang(ebsd, fname, *args, **kwargs)
  if ext == ".ctf":
    return _exportEBSD_ctf(ebsd, fname, *args, **kwargs)
  if ext in (".h5", ".hdf5", ".h5oina", ".oh5", ".edaxh5", ".dream3d"):
    return _exportEBSD_h5(ebsd, fname, *args, **kwargs)
  if ext in (".cpr", ".crc"):
    raise NotImplementedError("the Oxford binary pair is not written; write a .ctf")
  return _exportEBSD_txt(ebsd, fname, *args, **kwargs)


# ------------------------------------------------------------------ the text formats

# write a map as an EDAX/TSL .ang file, the correction of the setting undone
def _exportEBSD_ang(ebsd, fname, *, eulerCorrection=None, setting=2):
  if eulerCorrection is None:
    eulerCorrection = eulerCorrectionRotation(setting)
  phi1, Phi, phi2 = _eulerColumns(ebsd, eulerCorrection, 1.0, ('EDAX',))
  pos = ebsd.pos.data.reshape(-1, 3)
  numbers, indexed = exportPhaseNumbers(ebsd)
  iq = propertyColumn(ebsd, ("iq", "imagequality", "bc", "bandcontrast", "radonquality"), 0)
  ci = propertyColumn(ebsd, ("ci", "confidenceindex", "bs", "bandslope"), 0)
  sem = propertyColumn(ebsd, ("sem", "semsignal"), 1)
  fit = propertyColumn(ebsd, ("fit", "mad", "meanangulardeviation"), 0)
  lines = []
  for name, key, default in (("TEM_PIXperUM", "tem_pixperum", 1.0), ("x-star", "x_star", 0.0), ("y-star", "y_star", 0.0),
                             ("z-star", "z_star", 0.0), ("WorkingDistance", "workingdistance", 0.0)):
    lines.append(f"# {name:<22}{_headerNumber(ebsd, (key,), default):.6f}")
  lines.append("#")
  for k, j in enumerate(indexed):
    fr = ebsd.CSList.frames[j]
    a, b, c = fr.abc
    al, be, ga = fr.abg / degree
    lines += [f"# Phase {k + 1}", f"# MaterialName  \t{fr.mineral}", "# Formula     \t", "# Info \t\t",
              f"# {'Symmetry':<22}{tslCode(fr)}",
              f"# {'LatticeConstants':<22} {a:4.3f} {b:5.3f} {c:5.3f} {al:7.3f} {be:7.3f} {ga:7.3f}",
              f"# {'NumberFamilies':<22}0", "#"]
  xs, ys = pos[:, 0], pos[:, 1]
  stepX, stepY, nOdd, nEven, nRows, hex = scanOf(ebsd)
  lines += [f"# GRID: {'HexGrid' if hex else 'SqrGrid'}", f"# XSTEP: {stepX:.6f}", f"# YSTEP: {stepY:.6f}",
            f"# NCOLS_ODD: {nOdd}", f"# NCOLS_EVEN: {nEven}",
            f"# NROWS: {nRows}", "#", f"# OPERATOR: \t{headerValue(ebsd, ('operator',), '')}", "#",
            f"# SAMPLEID: \t{headerValue(ebsd, ('sampleid',), '')}", "#", f"# SCANID: \t{headerValue(ebsd, ('scanid',), '')}", "#",
            "# VERSION 5", "# COLUMN_NOTES: phi1, PHI, phi2, x, y, iq, ci, phase, sem, fit", "#"]
  # the fields are separated rather than laid out to a width, since an angle of three digits
  # fills the width TSL writes and the columns then run into one another
  table = np.column_stack([phi1, Phi, phi2, xs, ys, iq, ci, numbers, np.rint(sem), fit])
  with open(fname, "wb") as f:
    f.write(("\n".join(lines) + "\n").encode())
    f.write(formatTable(table, [5, 5, 5, 5, 5, 1, 3, -1, -1, 3]))
  return fname


# write a map as an Oxford/Channel .ctf file, the half turn about y and the tilt of a stated surface undone
def _exportEBSD_ctf(ebsd, fname, *, eulerCorrection=None):
  acq = tuple(_headerNumber(ebsd, (k,), 0.0) for k in ("acqe1", "acqe2", "acqe3"))
  if eulerCorrection is None:
    eulerCorrection = oxfordCorrection(acq)
  phi1, Phi, phi2 = _eulerColumns(ebsd, eulerCorrection, 1 / degree)
  pos = ebsd.pos.data.reshape(-1, 3)
  numbers, indexed = exportPhaseNumbers(ebsd)
  bands = propertyColumn(ebsd, ("bands",), 0)
  err = propertyColumn(ebsd, ("error",), 0)
  mad = propertyColumn(ebsd, ("mad", "meanangulardeviation", "fit"), 0)
  bc = propertyColumn(ebsd, ("bc", "bandcontrast", "iq", "imagequality"), 0)
  bs = propertyColumn(ebsd, ("bs", "bandslope", "ci", "confidenceindex"), 0)
  xs, ys = pos[:, 0], pos[:, 1]
  stepX, stepY, nOdd, nEven, nRows, hex = scanOf(ebsd)
  lines = ["Channel Text File", f"Prj\t{headerValue(ebsd, ('prj',), os.path.basename(fname))}",
           f"Author\t{headerValue(ebsd, ('author',), '')}", f"JobMode\t{headerValue(ebsd, ('jobmode',), 'Grid')}",
           f"XCells\t{nOdd}", f"YCells\t{nRows}", f"XStep\t{stepX:.4f}", f"YStep\t{stepY:.4f}"]
  lines += [f"AcqE{k + 1}\t{acq[k]:.4f}" for k in range(3)]
  lines.append("Euler angles refer to Sample Coordinate system (CS0)!\tMag\t0\tCoverage\t0\tDevice\t0\tKV\t0\tTiltAngle\t0\tTiltAxis\t0")
  lines.append(f"Phases\t{len(indexed)}")
  for j in indexed:
    fr = ebsd.CSList.frames[j]
    a, b, c = fr.abc
    al, be, ga = fr.abg / degree
    lines.append(f"{a:.3f};{b:.3f};{c:.3f}\t{al:.3f};{be:.3f};{ga:.3f}\t{fr.mineral}\t{channelCode(fr)}\t0\t\t\t")
  lines.append("Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS")
  table = np.column_stack([numbers, xs, ys, bands, err, phi1, Phi, phi2, mad, bc, bs])
  with open(fname, "wb") as f:
    f.write(("\r\n".join(lines) + "\r\n").encode())
    f.write(formatTable(table, [-1, 4, 4, -1, -1, 4, 4, 4, 4, -1, -1], b"\t", crlf=True))
  return fname


# write a map as a table with a line of column names: x y, the Euler angles, the phase, the properties
#
# The phase column counts the phases present from one, zero for not indexed, so the table
# reads back with those phases as its CS list.
def _exportEBSD_txt(ebsd, fname, *args, radians=False, eulerCorrection=None):
  convention = "Bunge"
  for a in args:
    if isinstance(a, str):
      convention = a
    else:
      raise TypeError(f"export does not understand the argument {a!r}")
  r = rotation(ebsd.rotations.data.reshape(-1, 4))
  if eulerCorrection is not None:
    r = eulerCorrection.inv() * r
  angles = [np.asarray(v, dtype=float) for v in r.Euler(convention)]
  scale = 1.0 if radians else 1 / degree
  angles = [np.where(np.isfinite(a), a, 0.0) * scale for a in angles]
  pos = ebsd.pos.data.reshape(-1, 3)
  # the phases present, counted from one, as the vendor formats write them
  numbers, _ = exportPhaseNumbers(ebsd)
  names = ["x", "y", *EULER_LABELS[eulerConvention(convention)], "phase"]
  cols = [pos[:, 0], pos[:, 1], *angles, numbers.astype(float)]
  decimals = [6, 6, 6, 6, 6, -1]
  for key, v in ebsd.prop.items():
    v = np.asarray(v)
    if v.ndim == ebsd.ndim and np.issubdtype(v.dtype, np.number):
      names.append(key)
      cols.append(v.reshape(-1).astype(float))
      decimals.append(-1 if np.issubdtype(v.dtype, np.integer) else 6)
  with open(fname, "wb") as f:
    f.write(("\t".join(names) + "\n").encode())
    f.write(formatTable(np.column_stack(cols), decimals, b"\t"))
  return fname


# ------------------------------------------------------------------ the HDF5 write-back

# There is no "the" HDF5 EBSD format: every vendor nests its data differently and stores far
# more than an importer models, so a file written from scratch loses all of it and no tool
# reads it back. What is written is the file the map came from: it is copied, and only the
# data sets the import resolved are written into the copy, so the result is still a file of
# the vendor's own format and everything else in it passes through untouched. The record
# the loader left in meta['h5'] says which data set holds what, in which unit, which rows of
# the file these measurements are, and what the import turned the numbers into on the way.

# the rows of a data set a map occupies, written and the rest left alone; a data set laid
# out on the map is the same list folded, so it is written flat and folded back
def _writeRows(f, path, rows, values):
  d = f[path]
  if (rows is not None and d.size <= rows.max()) or (rows is None and d.size != len(values)):
    return False
  raw = d[()].reshape(-1)
  if rows is None:
    raw[:] = np.asarray(values).astype(d.dtype)
  else:
    raw[rows] = np.asarray(values).astype(d.dtype)
  d[()] = raw.reshape(d.shape)
  return True


def _writeStacked(f, path, rows, components, componentsFirst):
  d = f[path]
  raw = d[()]
  flat = raw.reshape(3, -1) if componentsFirst else raw.reshape(-1, 3)
  n = flat.shape[1] if componentsFirst else flat.shape[0]
  idx = np.arange(n) if rows is None else rows
  if idx.max() >= n:
    return False
  for c in range(3):
    if componentsFirst:
      flat[c, idx] = components[c]
    else:
      flat[idx, c] = components[c]
  d[()] = flat.reshape(raw.shape)
  return True


def _writeValue(f, path, value):
  d = f[path]
  if d.dtype.kind in "SO":
    if not isinstance(value, str):
      return False
    raw = d[()]
    if d.dtype.kind == "S" and len(value.encode()) > d.dtype.itemsize:
      warnings.warn(f"{path} holds {d.dtype.itemsize} characters, so {value!r} is cut short", stacklevel=3)
    v = np.array(value.encode() if d.dtype.kind == "S" else value, dtype=d.dtype)
    d[()] = v if np.ndim(raw) == 0 else np.full(np.shape(raw), v, dtype=d.dtype)
    return True
  v = np.asarray(value, dtype=float).reshape(-1)
  if d.size != v.size:
    return False
  d[()] = v.reshape(d.shape).astype(d.dtype)
  return True


# the rows idx of one field of a table of records written, the rest of the field left as it is
def _writeField(d, name, idx, values):
  field = d[name]
  field[idx] = np.asarray(values).astype(field.dtype)
  d[..., name] = field


def h5PhaseColumn(ids, dtype, zeroBased, shift):
  """the phase numbers the file writes: the shift the import added taken out again, and the
  marker the vendor writes where nothing was indexed, -1 in a signed column, the largest value else"""
  ids = np.asarray(ids, dtype=np.int64)
  marker = -1 if dtype.kind == "i" else (np.iinfo(dtype).max if dtype.kind == "u" else 255)
  out = ids - shift
  if zeroBased:
    out[ids == 0] = marker
  return out


# write the map into a copy of the file it was read from, or of reference
#
#   export(ebsd, 'denoised.h5oina')                            the file the map came from is copied
#   export(ebsd, 'denoised.h5oina', reference='map.h5oina')    another file of the same layout
#
# Only the data sets the import resolved are written; a map that lost or gained measurements
# since it was read is written to .ang or .ctf instead.
def _exportEBSD_h5(ebsd, fname, *, reference=None, properties=True):
  import h5py
  rec = ebsd.meta.get("h5")
  if rec is None:
    raise ValueError("this map carries no record of which data set of an HDF5 file holds what, so no file can be "
                     "written into; read it with loadEBSD first")
  ref = os.fspath(rec["file"] if reference is None else reference)
  if not os.path.isfile(ref):
    raise ValueError(f"the file to copy, {ref}, is not there; name one with reference='map.h5'")
  fname = os.fspath(fname)
  if os.path.abspath(ref) == os.path.abspath(fname):
    raise ValueError(f"{fname} is the file to copy itself, and writing it would destroy it")
  rows = rec["rows"]
  n = rec["fileLength"] if rows is None else len(rows)
  # the rows of the file the measurements are, None for all of them in order, and as indices
  rowIdx = None if rows is None else np.asarray(rows)
  idx = np.arange(n) if rows is None else rowIdx
  if n != ebsd.size:
    raise ValueError(f"the map holds {ebsd.size} measurements and the record {n} rows of {ref}; a map that lost or "
                     "gained measurements is written to .ang or .ctf")
  # an old file is removed rather than truncated: ext4 writes a truncated and rewritten file
  # through to the disk when it is closed, 0.9 s for 160 MB
  if os.path.lexists(fname):
    os.remove(fname)
  shutil.copyfile(ref, fname)
  os.chmod(fname, 0o644)
  written = []
  with h5py.File(fname, "r+") as f:
    # the orientations, the correction the import applied taken out again and in the unit the
    # file writes; a pixel that was not indexed has none, so its row is left as it is
    eu = rec["euler"]
    r = rotation(ebsd.rotations.data.reshape(-1, 4))
    if rec["correction"] is not None:
      r = rec["correction"].inv() * r
    angles = [np.asarray(v, dtype=float) / eu["scale"] for v in r.Euler("Bunge")]
    ok = np.flatnonzero(np.isfinite(angles[0]))
    okRows = idx[ok]
    if len(eu["paths"]) == 1:
      if _writeStacked(f, eu["paths"][0], okRows, [a[ok] for a in angles], eu["componentsFirst"]):
        written.append(eu["paths"][0])
    else:
      for c, p in enumerate(eu["paths"]):
        if _writeRows(f, p, okRows, angles[c][ok]):
          written.append(p)
    # a map packed into one record per measurement goes back field by field: the matrices
    # written column by column, zeros where a pixel has no orientation, the phase, the
    # properties the table held; a property the map gained since has no room there
    table = rec.get("table")
    if table is not None:
      d = f[table["path"]]
      if table["matrixField"]:
        M = np.array(r.matrix(), dtype=float).reshape(-1, 3, 3)
        M[~np.isfinite(angles[0])] = 0.0
        _writeField(d, table["matrixField"], idx, M.transpose(0, 2, 1).reshape(-1, 9))
        written.append(table["path"] + "." + table["matrixField"])
      if table["phaseField"]:
        dtype = d.dtype[table["phaseField"]]
        _writeField(d, table["phaseField"], idx,
                    h5PhaseColumn(ebsd.phaseId.reshape(-1), dtype, rec["phase"]["zeroBased"], rec["phase"]["shift"]))
      if properties:
        for key, name in table["propFields"].items():
          if key in ebsd.prop:
            _writeField(d, name, idx, np.asarray(ebsd.prop[key]).reshape(-1))
    # the phase column, in the numbering the file uses
    ph = rec["phase"]
    if ph["path"] is not None:
      column = h5PhaseColumn(ebsd.phaseId.reshape(-1), f[ph["path"]].dtype, ph["zeroBased"], ph["shift"])
      if _writeRows(f, ph["path"], rowIdx, column):
        written.append(ph["path"])
    # the phase list: a renamed mineral and a corrected lattice are changes to the map like
    # any other, and the file states both. The symmetry is not written: every vendor codes it
    # differently and the port tells apart more groups than those codes do.
    ids = ebsd.CSList.indexedPhaseId
    for k, entry in enumerate(rec["cs"]):
      if k >= len(ids):
        break
      fr = ebsd.CSList.frames[int(ids[k])]
      if not isinstance(fr, crystalFrame):
        continue
      if entry["name"] is not None and _writeValue(f, entry["name"], fr.mineral):
        written.append(entry["name"])
      abc, abg = fr.abc, fr.abg / entry["angleScale"]
      whole = entry.get("cell", [])
      if len(whole) == 1:
        if _writeValue(f, whole[0], np.concatenate([abc, abg])):
          written.append(whole[0])
        continue
      for name, values in (("dim", abc), ("angle", abg)):
        ps = entry[name]
        if len(ps) == 1 and _writeValue(f, ps[0], values):
          written.append(ps[0])
        elif len(ps) == 3:
          for i in range(3):
            if _writeValue(f, ps[i], values[i]):
              written.append(ps[i])
    # the properties, back into the data sets they came from; one the map gained since goes
    # next to them, which a file that packs its map into one record per measurement has no room for
    if properties:
      if rec["props"]:
        group = os.path.dirname(next(iter(rec["props"].values())))
      elif ph["path"]:
        group = os.path.dirname(ph["path"])
      else:
        group = None
      for key, v in ebsd.prop.items():
        w = np.asarray(v).reshape(-1)
        if not np.issubdtype(w.dtype, np.number):
          continue
        if key in rec["props"]:
          if _writeRows(f, rec["props"][key], rowIdx, w):
            written.append(rec["props"][key])
        elif table is not None and key in table["propFields"]:
          continue
        elif group is not None:
          p = group + "/" + key
          if p in f:
            _writeRows(f, p, rowIdx, w)
          else:
            full = np.full(rec["fileLength"], np.nan)
            full[idx] = w.astype(float)
            f.create_dataset(p, data=full)
          written.append(p)
  return fname


__all__ = ["export"]
