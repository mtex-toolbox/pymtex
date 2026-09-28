# reading EBSD maps from the files the vendors write (MATLAB EBSD.load, Julia load.jl)
#
#   ebsd = loadEBSD('map.ang')                      the format from the extension
#   ebsd = loadEBSD('map.ctf', eulerCorrection=r)   a convention the vendor states, overridden
#   ebsd = loadEBSD('map.txt', CS=[cs], columnNames=['x', 'y', 'Euler1', 'Euler2', 'Euler3'])
#   ebsd = loadEBSD('map.ang', noGrid=True)         the list as the file holds it
#
# The format is the extension: MTEX tries every installed loader in turn and keeps the
# first that does not throw, which turns a malformed file into a wrong format rather
# than an error. Every format ends in ebsdFromColumns, which is where the phase numbering
# of the file is turned into ids of the phase list, once, and where the Euler correction
# is applied, once. A complete scan comes back on its grid, as MATLAB returns it.
from __future__ import annotations

import os
import re
import warnings

import numpy as np

from ..geometry.frames import notIndexedFrame, referenceFrame
from ..geometry.quaternion import rotation
from ..geometry.session import getMTEXpref
from ..geometry.symmetry import symmetry
from ..maps.ebsd import EBSD
from ..maps.gridify import gridify
from ..maps.phaselist import phaseList

# the Laue group a TSL symmetry code names; an older file writes 2 for 2/m
TSL_LAUE = {1: "-1", 2: "2/m", 20: "2/m", 22: "mmm", 4: "4/m", 42: "4/mmm", 3: "-3", 32: "-3m",
            6: "6/m", 62: "6/mmm", 23: "m-3", 43: "m-3m"}
# the 11 Laue classes in the order Channel numbers them
CHANNEL_LAUE = ["-1", "2/m", "mmm", "4/m", "4/mmm", "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m"]
# the 32 point groups in the order EDAX numbers them, PointGroupID = 100 + position
EDAX_POINT_GROUPS = ["1", "-1", "2", "m", "2/m", "222", "mm2", "mmm", "4", "-4", "4/m", "422", "4mm", "-42m",
                     "4/mmm", "3", "-3", "32", "3m", "-3m", "6", "-6", "6/m", "622", "6mm", "-62m", "6/mmm",
                     "23", "m-3", "432", "-43m", "m-3m"]


def tslPointGroup(code, pointGroup=None):
  """the point group an EDAX file states: the Laue group of its symmetry code, refined by the
  point group id or name where one is given and belongs to that Laue class"""
  try:
    laue = TSL_LAUE[int(float(str(code).split()[0]))]
  except (KeyError, ValueError, IndexError):
    laue = "-1"
  name = _edaxPointGroupName(pointGroup)
  if name is None:
    return laue
  try:
    if symmetry(name).LaueName() == symmetry(laue).LaueName():
      return name
  except Exception:
    pass
  return laue


def _edaxPointGroupName(pointGroup):
  if pointGroup is None:
    return None
  s = str(pointGroup).strip()
  if not s:
    return None
  m = re.search(r"\[([^\]]+)\]", s)
  if m:
    return m.group(1).strip()
  try:
    i = int(float(s))
  except ValueError:
    return s
  return EDAX_POINT_GROUPS[i - 100] if 100 <= i < 100 + len(EDAX_POINT_GROUPS) else None


# ------------------------------------------------------------------ the phase numbering

def phaseShift(values, nPhases):
  """how the phase numbers of a file map onto the phase list: 0 when the file counts its phases
  from one and writes zero for not indexed, 1 when it counts from zero

  Which one a file does is nowhere stated in it, so it is read off the values: a set that
  runs from zero and stops one short of the number of declared phases is zero based. A
  negative value means not indexed and takes no part in the decision."""
  v = np.asarray(values)
  v = v if v.dtype.kind in "iu" else np.rint(v.astype(float))
  v = v[v >= 0]
  if v.size == 0 or nPhases == 0:
    return 0
  return 1 if (v.min() == 0 and v.max() == nPhases - 1) else 0


def phaseColumn(raw, shift, nPhases):
  """the phase id of every measurement: the file's number shifted onto the list, and a number
  outside it a measurement of no phase"""
  raw = np.asarray(raw)
  v = (raw.astype(np.int64) if raw.dtype.kind in "iu" else np.rint(raw.astype(float)).astype(np.int64)) + shift
  v[(v < 1) | (v >= nPhases)] = 0
  return v.astype(np.int8)


# the property names that say how well a pattern was indexed
QUALITY_MEASURES = re.compile(r"^(iq|imagequality|image_quality|ci|confidenceindex|confidence_index|confidence|metric|xc|fit|mad|error|bc|bs|bandcontrast|bandslope|quality)$")


def unmeasuredRows(phi1, Phi, phi2, props):
  """the measurements an indexing program wrote without ever having measured them: a phase at
  the zero orientation with every quality measure the file holds at zero"""
  quality = [np.asarray(v) for k, v in props.items() if QUALITY_MEASURES.match(str(k).lower()) and np.size(v) == np.size(phi1)]
  if not quality:
    return None
  mask = (np.asarray(phi1) == 0) & (np.asarray(Phi) == 0) & (np.asarray(phi2) == 0)
  for q in quality:
    mask &= q == 0
  return mask if mask.any() else None


# ------------------------------------------------------------------ the map

def ebsdFromColumns(x, y, phi1, Phi, phi2, phase, phases, props, convention="Bunge", eulerCorrection=None,
                    scanUnit="um", frame=None, meta=None, shift=None, unindexed=None, unitCell=None, scanGrid=None):
  """the map from the columns every format has: a position, three Euler angles and a phase,
  with whatever else the file carries as properties

  phase is the file's own numbering; shift states it where the format fixes it and is read
  off the values otherwise. unindexed marks what the file says failed by other means, a
  negative confidence index or a pixel never measured. The rotation of a measurement of no
  phase is nan."""
  phases = phases if isinstance(phases, phaseList) else phaseList(phases)
  n = len(phases)
  shift = phaseShift(phase, n - 1) if shift is None else shift
  ids = phaseColumn(phase, shift, n)
  if unindexed is not None:
    ids[np.asarray(unindexed, dtype=bool)] = 0
  phi1, Phi, phi2 = (np.asarray(a, dtype=float) for a in (phi1, Phi, phi2))
  # an angle that is not a number is no orientation, and such a measurement is of no phase
  none = ~(np.isfinite(phi1) & np.isfinite(Phi) & np.isfinite(phi2))
  if none.any():
    ids[none] = 0
    phi1, Phi, phi2 = np.where(none, 0.0, phi1), np.where(none, 0.0, Phi), np.where(none, 0.0, phi2)
  rot = rotation.byEuler(phi1, Phi, phi2, convention)
  if eulerCorrection is not None:
    rot = eulerCorrection * rot
    meta = dict(meta or {}, EulerCorrection=eulerCorrection)
  q = np.array(rot.data, dtype=float)
  indexed = np.array([f.isIndexed for f in phases.frames])[ids]
  q[~indexed] = np.nan
  pos = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.zeros(len(ids))])
  # the file's own numbering of the phases, with what it writes for not indexed
  phaseMap = np.arange(n) - shift
  phaseMap[0] = -1 if shift else 0
  if scanGrid is None:
    scanGrid = scanGridFromPositions(pos[:, 0], pos[:, 1])
  lat = gridLattice(pos, *scanGrid) if scanGrid is not None else None
  # the rotations of Euler angles are unit already; another normalisation is a pass for nothing
  return EBSD(pos, rotation._fromData(q), ids, phases, dict(props), unitCell=unitCell, scanUnit=scanUnit, frame=frame,
              meta=meta, phaseMap=phaseMap, lattice=lat)


def scanGridFromPositions(x, y, dx=None, dy=None):
  """the scan the row order of the positions states: a file written row by row with x running
  fastest has its first row where y first changes, a square scan rows of one length, and a
  hexagonal scan rows of two lengths alternating with the second row half a step along;
  None where the positions are not written that way, which leaves the lattice to be
  inferred. A scan stopped early ends in a shorter row. What comes out is only a statement to
  be verified against every position."""
  n = len(y)
  if n < 4:
    return None
  x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
  # a row ends where x jumps back by many steps; along a row it moves by one step, the
  # step being what most neighbours are apart
  d = np.diff(x)
  step = np.median(d)
  if not (abs(step) > 0) or not np.isfinite(step):
    return None
  breaks = np.flatnonzero(np.abs(d - step) > 1.5 * abs(step))
  if breaks.size == 0 or breaks[0] < 1:
    return None
  starts = np.concatenate([[0], breaks + 1])
  lengths = np.diff(np.append(starts, n))
  ny = len(starts)
  if ny < 2:
    return None
  nx = int(lengths[0])
  if dx is None:
    dx = (x[nx - 1] - x[0]) / (nx - 1) if nx > 1 else step
  if dy is None:
    dy = (y[starts[-1]] - y[0]) / (ny - 1)
  if not np.isfinite(dx) or dx == 0 or not np.isfinite(dy) or dy == 0:
    return None
  dx, dy = float(dx), float(dy)
  shift = abs(x[starts[1]] - x[0])
  # a scan stopped early ends in a row shorter than the others
  full, last = lengths[:-1], lengths[-1]
  if np.all(full == nx) and last <= nx:
    if shift < 0.25 * abs(dx):
      return (nx, ny, dx, dy)
    return (nx, ny, dx, dy, True, nx) if abs(shift - abs(dx) / 2) < 0.25 * abs(dx) else None
  second = int(lengths[1])
  if np.all(full[0::2] == nx) and np.all(full[1::2] == second) and last <= (nx, second)[(ny - 1) % 2] \
      and abs(shift - abs(dx) / 2) < 0.25 * abs(dx):
    return (nx, ny, dx, dy, True, second)
  return None


def gridLattice(pos, nx, ny, xs, ys, hex=False, nxEven=None):
  """the lattice of the scan a header states, nx by ny cells of steps xs and ys written row by row
  with x running fastest, a hexagonal scan alternating nx and nxEven points per row with the
  odd rows shifted by half a step, the last row shorter where the scan stopped early; verified against the positions, and None where they
  disagree, which leaves the lattice to be inferred from the positions as for any map

  What a header says about the scan is exact where the positions carry it rounded, and
  reading the index off the row order costs one pass where inferring it costs a fit."""
  from ..maps.lattice import lattice, mapPlane
  P3 = np.asarray(pos, dtype=float).reshape(-1, 3)
  n = len(P3)
  nx, ny = int(nx), int(ny)
  if n < 4 or nx < 1 or ny < 1 or xs == 0 or ys == 0 or not np.isfinite(P3).all():
    return None
  # a map in the xy plane is its own plane coordinates; any other is projected
  plane = mapPlane(P3)
  P = P3[:, :2] if not P3[:, 2].any() else plane.to2d(P3)
  # a hexagonal scan alternates two row lengths, and which one the first row has is what
  # the positions say, so both arrangements are tried
  arrangements = [(nx, nx if nxEven is None else int(nxEven))] if hex else [(nx, nx)]
  if hex and arrangements[0][0] != arrangements[0][1]:
    arrangements.append(arrangements[0][::-1])
  for first, second in arrangements:
    got = _gridIndex(P, n, ny, first, second, xs, ys, hex)
    if got is not None:
      A, ij, lo = got
      return lattice._build(plane, P[0] + A @ lo, A, ij, "hex" if hex else "square")
  return None


# the index of every position in file order on the stated grid, counted from the least one, the
# basis and the least index, or None; a scan written against an axis has a negative step, and the
# index runs the other way
def _gridIndex(P, n, ny, nx, nxEven, xs, ys, hex):
  sx, sy = (-1 if xs < 0 else 1), (-1 if ys < 0 else 1)
  xs, ys = abs(xs), abs(ys)
  counts = np.where(np.arange(ny) % 2 == 0, nx, nxEven) if hex else np.full(ny, nx)
  # a scan stopped early ends in a shorter last row
  counts[-1] -= counts.sum() - n
  if not 0 < counts[-1] <= (nx, nxEven if hex else nx)[(ny - 1) % 2]:
    return None
  A = np.array([[xs, xs / 2], [0.0, ys]]) if hex else np.array([[xs, 0.0], [0.0, ys]])
  # the header's steps where the positions are on them, else the basis fitted to the
  # positions where that agrees with the header to the rounding of a header, since the
  # positions are written rounded too, else the fit itself
  step = min(xs, ys)
  ij, lo, resid = _rowOrder(P, counts, hex, sx, sy, A)
  # a file writing its positions as 32 bit floats rounds them to 2^-24 of their size, which far
  # from the origin is more than a ten thousandth of a step
  size = float(np.abs(P[0]).max()) + max(nx, nxEven if hex else nx) * xs + ny * ys
  if not resid < 1e-4 * step + 2.0 ** -20 * size:
    M = (ij + lo).astype(float)
    G = M.T @ M
    if np.linalg.cond(G) > 1e12:
      return None
    F = np.linalg.solve(G, M.T @ (P - P[0])).T
    if not np.allclose(F, A, rtol=1e-4, atol=1e-4 * step):
      A = F
    resid = np.abs(P - P[0] - M @ A.T).max()
    if not resid < 0.1 * step:
      return None
  return A, ij, lo


# the lattice index of the positions written row by row, counts[r] in row r, counted from the
# least one, the least index, and the largest distance of a position from its site under the
# basis A from the first one
def _rowOrder(P, counts, hex, sx, sy, A):
  from ..maps.kernels import gridKernel
  n = int(counts.sum())
  loop = gridKernel("rowOrder")
  if loop is not None:
    ij = np.empty((n, 2), dtype=np.int64)
    resid, i0, j0 = loop(np.ascontiguousarray(P, dtype=np.float64), counts.astype(np.int64), bool(hex), sx, sy, A[0, 0], A[0, 1], A[1, 1], ij)
    return ij, np.array([i0, j0]), resid
  k = np.arange(n)
  r = np.repeat(np.arange(len(counts)), counts)
  c = k - np.repeat(np.cumsum(counts) - counts, counts)
  i, j = (c - r // 2, r) if hex else (c, r)
  if sx < 0:
    i = -i - (j if hex else 0)
  if sy < 0:
    j = -j
  x0, y0 = P[0]
  fi, fj = i.astype(float), j.astype(float)
  resid = max(np.abs(P[:, 0] - (x0 + A[0, 0] * fi + A[0, 1] * fj)).max(), np.abs(P[:, 1] - (y0 + A[1, 1] * fj)).max())
  ij = np.stack([i, j], axis=1)
  lo = ij.min(axis=0)
  return ij - lo, lo, resid


def regularPoly(s, d, rot=0.0):
  """the corners of a regular cell of s sides and diameter d, MATLAB's regularPoly"""
  a = np.pi / s + np.arange(s) * 2 * np.pi / s + rot
  d = np.broadcast_to(np.asarray(d, dtype=float), (2,))
  return np.column_stack([np.cos(a) * d[0], np.sin(a) * d[1], np.zeros(s)]) / np.sqrt(s / 2)


def placedPhases(found):
  """the placed phase list a header declares: the frames by the number the file gives them,
  a number the header skips a not indexed phase of that number"""
  if not found:
    return phaseList()
  frames = [found.get(k, notIndexedFrame(f"phase {k}")) for k in range(1, max(found) + 1)]
  return phaseList([notIndexedFrame()] + frames)


# ------------------------------------------------------------------ the grid

def tryGridify(ebsd):
  """the map on its grid where no measurement is lost by putting it there, MATLAB's tryGridify:
  a stray position asking for a raster of nothing, two measurements in one cell and a
  degenerate cell each keep the list with a warning that says why"""
  if ebsd.isGridded or ebsd.isempty:
    return ebsd
  l = ebsd.lattice
  if l is None:
    warnings.warn("the measurements are not on a lattice; keeping the data as a list", stacklevel=3)
    return ebsd
  ij = l.ij
  nCells = int(np.prod(ij.max(axis=0) - ij.min(axis=0) + 1))
  if nCells > 10 * max(1, len(ebsd)):
    warnings.warn(f"the measurements span {nCells} lattice cells but there are only {len(ebsd)} of them, so they do "
                  "not form a grid; keeping the data as a list, call gridify explicitly for the raster anyway", stacklevel=3)
    return ebsd
  try:
    return gridify(ebsd)
  except (ValueError, np.linalg.LinAlgError) as e:
    warnings.warn(f"the measurements could not be put on a grid ({e}); keeping the data as a list", stacklevel=3)
    return ebsd


# ------------------------------------------------------------------ the entry point

VENDOR_EXTENSIONS = {"ang", "ctf", "cpr", "crc", "osc", "h5", "hdf5", "h5oina", "oh5", "edaxh5", "dream3d"}


def loadEBSD(fname, CS=None, *args, interface=None, noGrid=False, **kwargs):
  """the map a file holds, read by its extension

    loadEBSD('map.ang'), loadEBSD('map.ctf')          the vendor formats declare their phases
    loadEBSD('map.txt', CS=cs, columnNames=[...])     a table whose columns the caller names
    loadEBSD('map.dat', interface='ang')              the format named rather than read off the name
    loadEBSD(['a.ang', 'b.ang'])                      one map per file

  Every keyword of the format's loader passes through: eulerCorrection, setting for an EDAX
  file, columnNames, columns, radians, delimiter, header for a table. A complete scan comes
  back on its grid; noGrid=True keeps the list, and so does a scan whose measurements do not
  form a grid, which says why.
  """
  if isinstance(fname, (list, tuple)):
    return [loadEBSD(f, CS, *args, interface=interface, noGrid=noGrid, **kwargs) for f in fname]
  fname = os.fspath(fname)
  ext = (os.path.splitext(fname)[1] if interface is None else interface).lower().lstrip(".")
  if ext == "ang":
    from .ang import loadEBSD_ang
    ebsd = loadEBSD_ang(fname, *args, **kwargs)
  elif ext == "ctf":
    from .ctf import loadEBSD_ctf
    ebsd = loadEBSD_ctf(fname, *args, **kwargs)
  elif ext in ("cpr", "crc"):
    from .crc import loadEBSD_crc
    ebsd = loadEBSD_crc(fname, *args, **kwargs)
  elif ext == "osc":
    from .osc import loadEBSD_osc
    ebsd = loadEBSD_osc(fname, *args, **kwargs)
  elif ext in ("h5", "hdf5", "h5oina", "oh5", "edaxh5", "dream3d"):
    from .h5 import loadEBSD_h5
    ebsd = loadEBSD_h5(fname, *args, **kwargs)
  else:
    from .generic import loadEBSD_generic
    ebsd = loadEBSD_generic(fname, CS, *args, **kwargs)
  # the phases a vendor file declares, replaced by the list the caller states, as an import
  # script does: one frame per phase of the file in its order, the not indexed one first
  if CS is not None and ext in VENDOR_EXTENSIONS:
    ebsd = withPhases(ebsd, CS)
  if getMTEXpref("gridifyOnImport", True) and not noGrid:
    ebsd = tryGridify(ebsd)
  return ebsd


def withPhases(ebsd, CS):
  """the map with the phases of a vendor file replaced by the frames given, one per phase of the
  file in its order; a list of indexed phases alone keeps the file's not indexed one in front"""
  frames = [CS] if isinstance(CS, referenceFrame) else list(CS)
  have = ebsd.CSList.frames
  # a list of indexed phases alone keeps the file's not indexed one in front
  if frames[0].isIndexed:
    frames = [have[0]] + frames
  if len(frames) != len(have):
    raise ValueError(f"the file declares {len(have) - 1} phases and the list names {len(frames)}, the not indexed one first")
  return EBSD(ebsd.pos, ebsd.rotations, ebsd.phaseId, phaseList(frames), ebsd.prop, scanUnit=ebsd.scanUnit, meta=ebsd.meta,
              phaseMap=ebsd.phaseMap, id=ebsd.id, lattice=ebsd.lattice, isPadding=ebsd.isPadding)


__all__ = ["loadEBSD"]
