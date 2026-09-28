# the geometry of an EBSD map plot, without graphics (MATLAB @mapPlot, scaleBar,
# refFrameGeometry; docs/plans/plotting.md S4)
#
# A map on a rigid lattice is a picture: mapRaster puts it on a rectangle of pixels, one
# lookup per pixel, which the drawing layer shows as one image. A map whose lattice the
# measurements have drifted from is drawn as its cells, mapCells, one polygon per
# measurement over a cell whose corners follow the neighbouring centres, so a square and a
# hexagonal grid are the same picture with a different cell. A map on no lattice is its
# points. Either is drawn under a layout, the 2 x 2 matrix mapLayout(pC) that takes the
# in-plane coordinates of a measurement to the screen, east to the right and north up; a
# convention that turns or flips the map is a signed permutation, so the picture is turned
# rather than resampled.
from __future__ import annotations

import numpy as np

from ..geometry.plottingconvention import plottingConvention
from ..geometry.vector3d import vector3d, xvector, yvector, zvector
from ..maps.lattice import cellCorners, mapPlane
from ..maps.phaselist import phaseColorNames

# a lattice is rigid when no measurement sits further than this from its site, in cells
RIGID_TOL = 0.01
# a map with fewer cells than this is drawn as its cells unless asked otherwise
CELLS_MAX = 100_000


def identityLayout():
  """the identity layout, the 2 x 2 matrix that draws the plane's x and y as east and north"""
  return np.eye(2)


def mapLayout(pC=None):
  """the layout a convention asks for: the components of east and north along the plane's x
  and y, snapped to the signed permutation an axis aligned convention is"""
  pC = plottingConvention.default if pC is None else plottingConvention(pC)
  e, n = pC.east.data.reshape(3), pC.north.data.reshape(3)
  M = np.array([[e[0], e[1]], [n[0], n[1]]])
  snapped = np.where(np.abs(M) < 1e-12, 0.0, np.where(np.abs(np.abs(M) - 1) < 1e-12, np.sign(M), M))
  return snapped


# ------------------------------------------------------------------ the picture

def latticeTable(ebsd):
  """the measurements in the plane of the map, and the site of every index of the lattice:
  the measurement that sits there, -1 where none does"""
  l = ebsd.lattice
  ij = l.ij
  dims = (int(ij[:, 0].max()) + 1, int(ij[:, 1].max()) + 1)
  lin = np.full(dims, -1, dtype=np.int32)
  lin[ij[:, 0], ij[:, 1]] = np.arange(len(ij), dtype=np.int32)
  return lin


def mapBox(x):
  """the corners of the box round the measurements, in the plane of the map: the box of a map
  in a coordinate plane is exactly its extent, and a map of forty million points is not
  projected point by point to learn where it ends. The padding of a grid is left out."""
  P = x.pos.data.reshape(-1, 3)
  pad = getattr(x, "_pad", None)
  if pad is not None:
    P = P[~pad.reshape(-1)]
  if len(P) == 0:
    return np.zeros((0, 2))
  lo, hi = _columnBounds(P)
  corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
  return planeOf(x).to2d(corners)


# the least and the greatest of every column of a tall (n, 3) array: NumPy reduces a short
# axis strided and slowly, a block of rows at a time is ten times faster
def _columnBounds(P, block=64):
  n = len(P) - len(P) % block
  if n == 0:
    return P.min(axis=0), P.max(axis=0)
  head = P[:n].reshape(-1, block, P.shape[1])
  lo, hi = head.min(axis=0).min(axis=0), head.max(axis=0).max(axis=0)
  if n < len(P):
    tail = P[n:]
    lo, hi = np.minimum(lo, tail.min(axis=0)), np.maximum(hi, tail.max(axis=0))
  return lo, hi


def latticeResidual(ebsd):
  """how far the measurements sit from the sites of their lattice, the worst of them in cells;
  a pass over the whole map, so the lattice remembers it"""
  l = ebsd.lattice
  r = getattr(l, "_residual", None)
  if r is None:
    P = l.plane.to2d(ebsd.pos.data.reshape(-1, 3))
    d = P - l.origin - l.ij @ l.A.T
    r = float(np.sqrt(np.max(np.sum(d * d, axis=1)))) / l.dxy if len(P) else 0.0
    l._residual = r
  return r


def cappedSize(nx, ny, maxPixels):
  """a picture of at most maxPixels, scaled down with its shape kept and never below a pixel"""
  nx, ny = max(1, nx), max(1, ny)
  f = np.sqrt(maxPixels / (nx * ny))
  return (max(1, int(nx * f)), max(1, int(ny * f))) if f < 1 else (nx, ny)


def mapRaster(ebsd, maxPixels=16_000_000, layout=None):
  """the picture a map on a rigid lattice is drawn as: the rectangle its cells span and, for
  every pixel, the measurement whose cell covers it or -1 where none does, as an array of
  shape (ny, nx) with the first row at the bottom; None where the map is no picture

  A pixel belongs to the cell of the nearest site: by rounding the lattice coordinates where
  the lattice is aligned with the screen and the parallelogram is the cell, and otherwise
  among the four corners of the lattice parallelogram it lies in, since on a hexagonal
  lattice the parallelogram is a rhombus and not the cell. One pixel per cell where the
  lattice is aligned with the screen, a quarter of the spacing elsewhere. A lattice the measurements
  have drifted from has no picture, judged at the scale of the picture with a hundredth of
  a cell as the floor."""
  l = ebsd.lattice
  if l is None or len(l.ij) != ebsd.size or ebsd.size == 0:
    return None
  M = identityLayout() if layout is None else np.asarray(layout, dtype=float)
  lin = latticeTable(ebsd)
  A = M @ l.A
  h = l.dxy
  Q = mapBox(ebsd) @ M.T
  xlo, xhi, ylo, yhi = Q[:, 0].min(), Q[:, 0].max(), Q[:, 1].min(), Q[:, 1].max()
  if not (h > 0 and np.all(np.isfinite([xlo, xhi, ylo, yhi]))):
    return None
  # the picture spans the cells and not their centres
  cell = cellCorners(l.A, l.shape) @ M.T
  hx, hy = np.max(np.abs(cell[:, 0])), np.max(np.abs(cell[:, 1]))
  xs, ys = (xlo - hx, xhi + hx), (ylo - hy, yhi + hy)
  # the lattice fit is good to a few 1e-8 of the spacing, so a basis is axis-aligned at 1e-6 of it
  aligned = min(abs(A[0, 0]), abs(A[0, 1])) < 1e-6 * h and min(abs(A[1, 0]), abs(A[1, 1])) < 1e-6 * h
  px = max(abs(A[0, 0]), abs(A[0, 1])) if aligned else h / 4
  py = max(abs(A[1, 0]), abs(A[1, 1])) if aligned else h / 4
  nx, ny = cappedSize(int(round((xs[1] - xs[0]) / px)), int(round((ys[1] - ys[0]) / py)), maxPixels)
  dx, dy = (xs[1] - xs[0]) / nx, (ys[1] - ys[0]) / ny
  if latticeResidual(ebsd) * h > max(RIGID_TOL * h, min(dx, dy) / 4):
    return None
  Ainv = np.linalg.inv(A)
  o = M @ l.origin
  x = xs[0] + (np.arange(nx) + 0.5) * dx
  y = ys[0] + (np.arange(ny) + 0.5) * dy
  pad = getattr(ebsd, "_pad", None)
  from .kernels import mapRasterIndex
  index = mapRasterIndex(x, y, o, Ainv, A, aligned, lin, pad)
  if index is not None:
    return xs, ys, index
  if aligned:
    # the lattice coordinates of the pixel centres, one axis at a time, rounded
    ia = np.rint((x[None, :] - o[0]) * Ainv[0, 0] + (y[:, None] - o[1]) * Ainv[0, 1]).astype(np.int64)
    ib = np.rint((x[None, :] - o[0]) * Ainv[1, 0] + (y[:, None] - o[1]) * Ainv[1, 1]).astype(np.int64)
  else:
    qx, qy = np.meshgrid(x, y)                          # (ny, nx)
    q = np.stack([qx, qy], axis=-1) - o
    u = q @ Ainv.T
    a, b = np.round(u[..., 0]), np.round(u[..., 1])
    sa, sb = np.where(u[..., 0] < a, -1, 1), np.where(u[..., 1] < b, -1, 1)
    best = np.full(qx.shape, np.inf)
    ia, ib = np.zeros(qx.shape, dtype=np.int64), np.zeros(qx.shape, dtype=np.int64)
    for db in (0, 1):
      for da in (0, 1):
        ca, cb = a + da * sa, b + db * sb
        d = np.stack([ca, cb], axis=-1) @ A.T - q
        d2 = np.sum(d * d, axis=-1)
        closer = d2 < best
        best = np.where(closer, d2, best)
        ia = np.where(closer, ca, ia).astype(np.int64)
        ib = np.where(closer, cb, ib).astype(np.int64)
  inside = (ia >= 0) & (ia < lin.shape[0]) & (ib >= 0) & (ib < lin.shape[1])
  index = np.full((ny, nx), -1, dtype=np.int32)
  index[inside] = lin[ia[inside], ib[inside]]
  # the padding of a grid is no measurement and shows nothing
  if pad is not None:
    index[index >= 0] = np.where(pad.reshape(-1)[index[index >= 0]], -1, index[index >= 0])
  return xs, ys, index


# ------------------------------------------------------------------ the cells

def cornerSharing(l, cell):
  """the lattice offsets of the cells that share each corner of the cell: the sites as far
  from the corner as the cell's own site is, four round a corner of a square lattice and
  three round one of a hexagonal lattice"""
  tol = 0.15 * l.dxy
  offs = np.array([(i, j) for j in (-1, 0, 1) for i in (-1, 0, 1)])
  return [[o for o in offs if abs(np.linalg.norm(v - l.A @ o) - np.linalg.norm(v)) < tol] for v in cell]


def mapCells(ebsd, layout=None):
  """the corners of every cell of a map, shape (n, m, 2), on the screen: a corner is shared by
  the cells around it and sits at the mean of where each of them puts it, the measured
  centre plus the ideal offset to the corner. On a rigid grid that is the ideal corner;
  under a stage drift the corners follow the centres, so the cells tile without gaps"""
  l = ebsd.lattice
  if l is None:
    raise ValueError("the map is not on a lattice, so it has no cells to draw")
  M = identityLayout() if layout is None else np.asarray(layout, dtype=float)
  P, lin = l.plane.to2d(ebsd.pos.data.reshape(-1, 3)), latticeTable(ebsd)
  cell = cellCorners(l.A, l.shape)
  share = cornerSharing(l, cell)
  ij = l.ij
  n, m = len(P), len(cell)
  corners = np.zeros((n, m, 2))
  for j in range(m):
    acc = np.zeros((n, 2))
    cnt = np.zeros(n)
    for o in share[j]:
      ia, ib = ij[:, 0] + o[0], ij[:, 1] + o[1]
      ok = (ia >= 0) & (ia < lin.shape[0]) & (ib >= 0) & (ib < lin.shape[1])
      s = np.full(n, -1, dtype=np.int64)
      s[ok] = lin[ia[ok], ib[ok]]
      has = s >= 0
      acc[has] += P[s[has]] + (cell[j] - l.A @ o)
      cnt[has] += 1
    corners[:, j] = acc / np.maximum(cnt, 1)[:, None]
  return corners @ M.T


def mapPoints(ebsd, layout=None):
  """the measurements as points on the screen"""
  M = identityLayout() if layout is None else np.asarray(layout, dtype=float)
  return planeOf(ebsd).to2d(ebsd.pos.data.reshape(-1, 3)) @ M.T


def planeOf(x):
  """the plane a map, grains or a boundary lie in: the lattice's for a map, else one fitted
  to its positions"""
  if hasattr(x, "lattice"):
    return x.lattice.plane if x.lattice is not None else mapPlane(x.pos.data.reshape(-1, 3))
  return x.plane


# ------------------------------------------------------------------ boundaries and grains

def boundaryPolyline(gB):
  """a boundary as one polyline, the chains in walk order separated by a NaN row"""
  iv = gB.chainV
  out = np.full((len(iv), 2), np.nan)
  ok = iv >= 0
  out[ok] = gB.V[iv[ok]]
  return out


def boundarySegments(gB):
  """the two vertices of every segment, shape (n, 2, 2)"""
  return gB.V[gB.F]


def grainPolygons(g, k):
  """the loops of a grain as a list of rings, the outer one first, each (m, 2)"""
  return [g.V[np.asarray(loop)] for loop in g.loops[k]]


def mapExtent(x, layout=None):
  """the extent of what a map plot has to show, on the screen: (xlo, xhi, ylo, yhi)"""
  M = identityLayout() if layout is None else np.asarray(layout, dtype=float)
  if hasattr(x, "lattice"):
    P = mapBox(x)
    r = x.lattice.dxy if x.lattice is not None else 0.0
  elif hasattr(x, "loops"):
    gB = x.boundary
    P, r = gB.V[np.unique(gB.F)] if len(gB.F) else np.zeros((0, 2)), 0.0
  elif hasattr(x, "boundaryId"):
    P, r = (np.asarray(x.V).reshape(-1, 2) if len(x.grainId) else np.zeros((0, 2))), 0.0
  else:
    P, r = x.V[np.unique(x.F)] if len(x.F) else np.zeros((0, 2)), 0.0
  if len(P) == 0:
    return (0.0, 0.0, 0.0, 0.0)
  Q = P @ M.T
  return (float(Q[:, 0].min() - r), float(Q[:, 0].max() + r), float(Q[:, 1].min() - r), float(Q[:, 1].max() + r))


def phaseColor(pl, i):
  """the colour a phase is drawn in: the one its frame carries, else the next of MATLAB's order
  for an indexed phase, white for a not indexed one without a colour"""
  c = phaseColorNames(pl.frames)[i]
  return "white" if c is None else c


def phasePalette(pl):
  """the colour of every phase of a list"""
  return [phaseColor(pl, i) for i in range(len(pl))]


# ------------------------------------------------------------------ the scale bar

LENGTH_UNITS = ("nm", "um", "mm", "m")
SCALE_BAR_STEPS = (1, 2, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200, 500, 750)


def unitLabel(u):
  """the label of a length unit on a plot, µm for 'um'"""
  return "µm" if u == "um" else str(u)


def switchUnit(x, unit):
  """the same length in the unit that keeps its number between one and a thousand: the
  number, its unit and the factor it was multiplied by"""
  if unit not in LENGTH_UNITS or x <= 0:
    return float(x), unit, 1.0
  i = LENGTH_UNITS.index(unit)
  y, f = float(x), 1.0
  while y >= 1000 and i < len(LENGTH_UNITS) - 1:
    y, f, i = y / 1000, f / 1000, i + 1
  while y < 1 and i > 0:
    y, f, i = y * 1000, f * 1000, i - 1
  return y, LENGTH_UNITS[i], f


def scaleBar(x, unit="um", fraction=0.1):
  """how long a scale bar of a map should be, in the coordinates the map is drawn in, and how
  it reads: a round number near a tenth of the width, in the unit that keeps it readable"""
  if not isinstance(x, (int, float, np.floating, np.integer)):
    e = mapExtent(x)
    return scaleBar(e[1] - e[0], getattr(x, "scanUnit", "um"), fraction)
  y, u, f = switchUnit(fraction * x, unit)
  n = SCALE_BAR_STEPS[int(np.argmin([abs(s - y) for s in SCALE_BAR_STEPS]))]
  return n / f, f"{n:g} {unitLabel(u)}"


# ------------------------------------------------------------------ the frame triad

def frameOnScreen(pl, layout=None):
  """the (right, up, towards the viewer) components of the axes of the frame in the picture of
  a map: the plane's coordinates taken to the screen by the layout, and the normal towards
  the viewer, who looks from the other side of the plane when the layout flips it"""
  M = identityLayout() if layout is None else np.asarray(layout, dtype=float)
  side = np.sign(np.linalg.det(M))
  n = pl.normal
  out = []
  for a in (xvector, yvector, zvector):
    d = a.data.reshape(3)
    s = M @ np.array([d @ pl.e1, d @ pl.e2])
    out.append((float(s[0]), float(s[1]), float(side * (d @ n))))
  return out


def triadGeometry(screen, labels, t=1.0):
  """the reference frame indicator MTEX draws in the scale bar box, with the text height t as
  its unit: an arrow per axis that has a direction on screen, shortened by its tilt out of
  the plane, a circle for an axis leaving the plane with its label inside, crossed by four
  spokes when it points away from the viewer, and a label beside every arrow. Returns the
  arrows as polygons, the symbol as a polyline with NaN breaks, the labels with their
  positions, and the bounding box (umin, umax, vmin, vmax)"""
  arm, shaftHalf = 1.2 * t, t / 8
  headHalf, headLen = 3 * shaftHalf, 0.36 * t
  symR, pad = 0.46 * t, 0.45 * t
  maxLen = max((len(s) for s in labels), default=1)
  labHalf = np.array([(0.15 + 0.25 * maxLen) * t, 0.4 * t])
  brk = np.array([np.nan, np.nan])
  arrows, symbol, texts = [], [], []
  inPlane = [np.hypot(s[0], s[1]) >= 0.15 for s in screen]
  kSym = next((k for k, p in enumerate(inPlane) if not p), None)
  r0 = 0.0
  if kSym is not None:
    r0 = 1.2 * symR
    phi = np.linspace(0, 2 * np.pi, 49)
    symbol += [np.array([symR * np.cos(p), symR * np.sin(p)]) for p in phi]
    if screen[kSym][2] < 0:
      r, ri = symR / np.sqrt(2), 0.62 * symR / np.sqrt(2)
      for sx, sy in ((-1, -1), (1, 1), (-1, 1), (1, -1)):
        symbol += [brk, np.array([sx * r, sy * r]), np.array([sx * ri, sy * ri])]
    if kSym < len(labels):
      texts.append((np.zeros(2), labels[kSym]))
  for k, p in enumerate(inPlane):
    if not p:
      continue
    d = np.array([screen[k][0], screen[k][1]])
    length = np.linalg.norm(d) * arm
    d = d / np.linalg.norm(d)
    q = np.array([-d[1], d[0]])
    hl = min(headLen, 0.6 * length)
    arrows.append(np.array([r0 * d + shaftHalf * q, (r0 + length - hl) * d + shaftHalf * q, (r0 + length - hl) * d + headHalf * q,
                            (r0 + length) * d, (r0 + length - hl) * d - headHalf * q, (r0 + length - hl) * d - shaftHalf * q,
                            r0 * d - shaftHalf * q]))
    if k < len(labels):
      texts.append(((r0 + length + pad) * d, labels[k]))
  pts = [p for a in arrows for p in a] + [q for q in symbol if not np.isnan(q[0])] + [np.zeros(2)]
  pts += [q - labHalf for q, _ in texts if np.any(q != 0)] + [q + labHalf for q, _ in texts if np.any(q != 0)]
  P = np.array(pts)
  bbox = (float(P[:, 0].min()), float(P[:, 0].max()), float(P[:, 1].min()), float(P[:, 1].max()))
  return arrows, np.array(symbol).reshape(-1, 2), texts, bbox


__all__ = ["mapLayout", "mapRaster", "mapCells", "mapPoints", "mapExtent", "boundaryPolyline", "boundarySegments",
           "grainPolygons", "scaleBar", "switchUnit", "triadGeometry", "frameOnScreen", "phaseColor", "phasePalette"]
