# the geometry of an EBSD map plot, without graphics (MATLAB @mapPlot, scaleBar,
# refFrameGeometry; docs/plans/plotting.md S4)
#
# A map on a rigid lattice is a picture: mapRaster puts it on a rectangle of pixels, one
# lookup per pixel, which the drawing layer shows as one image. A map whose lattice the
# measurements have drifted from is drawn as its cells, mapCells, one polygon per
# measurement over a cell whose corners follow the neighbouring centres, so a square and a
# hexagonal grid are the same picture with a different cell. A map on no lattice is its
# points. Either is drawn under a layout, the 2 x 2 matrix mapLayout(pC, plane) that takes the
# in-plane coordinates of a measurement to the screen, east to the right and north up; a
# convention that turns or flips the map is a signed permutation, so the picture is turned
# rather than resampled.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import columnMax, columnMin
from ..geometry.plottingconvention import plottingConvention
from ..geometry.vector3d import vector3d, xvector, yvector, zvector
from ..maps.boundary import triplePointList
from ..maps.ebsd import EBSD
from ..maps.grains import grain2d
from ..maps.lattice import cellCorners, mapPlane
from ..maps.phaselist import phaseColorNames

# a lattice is rigid when no measurement sits further than this from its site, in cells
RIGID_TOL = 0.01
# a map with fewer cells than this is drawn as its cells unless asked otherwise
CELLS_MAX = 100_000


def identityLayout():
  """the identity layout, the 2 x 2 matrix that draws the plane's x and y as east and north"""
  return np.eye(2)


# the layout given as a float matrix, the identity for none
def _layoutOf(layout):
  return identityLayout() if layout is None else np.asarray(layout, dtype=float)


def mapLayout(pC=None, plane=None):
  """the layout a convention asks for: the 2 x 2 matrix taking the in-plane coordinates of a map
  to the screen, east to the right and north up

    M = mapLayout()                                  # the default plotting convention
    M = mapLayout('x↑→y')                            # x up, y to the right
    M = mapLayout(pC, ebsd.lattice.plane)            # a map in another plane

  The rows are the components of east and north along the coordinates of the map's plane (x
  and y without a plane), snapped to the signed permutation an axis aligned convention is,
  so that the picture is turned rather than resampled. A convention whose screen is not the
  plane, as the xy convention of a map in the yz plane, draws the plane's own coordinates
  east and north.

  Output
    M - a 2 x 2 array
  Keywords
    pC - plottingConvention, or the string of one; by default plottingConvention.default
    plane - the plane of the map, whose e1 and e2 are its coordinates
  See also
    plottingConvention, mapRaster, mapCells, mapPoints"""
  pC = plottingConvention.default if pC is None else plottingConvention(pC)
  e, n = pC.east.data.reshape(3), pC.north.data.reshape(3)
  e1, e2 = (np.eye(3)[0], np.eye(3)[1]) if plane is None else (plane.e1, plane.e2)
  M = np.array([[e @ e1, e @ e2], [n @ e1, n @ e2]], dtype=float)
  if abs(np.linalg.det(M)) < 1e-6:
    return identityLayout()
  return np.where(np.abs(M) < 1e-9, 0.0, np.where(np.abs(np.abs(M) - 1) < 1e-9, np.sign(M), M))


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
  if x._pad is not None:
    P = P[~x._pad.reshape(-1)]
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
    return columnMin(P), columnMax(P)
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

    xs, ys, index = mapRaster(ebsd)                          # one pixel per cell
    xs, ys, index = mapRaster(ebsd, maxPixels=10000)         # a coarser picture
    xs, ys, index = mapRaster(ebsd, layout=mapLayout('x↑→y'))  # turned to another convention

  Every pixel of the picture shows the measurement whose site is nearest. Where the lattice
  is aligned with the screen the picture has one pixel per cell; elsewhere, as on a turned or
  a hexagonal lattice, a quarter of the spacing.

  A map whose positions have drifted from their lattice has no picture: the drift is judged
  at the scale of the picture, with a hundredth of a cell as the floor. The padding of a grid
  shows nothing.

  Input
    ebsd - EBSD
  Output
    xs, ys - the screen extent (lo, hi) of the picture in x and in y
    index - array of shape (ny, nx), the measurement of every pixel, -1 for none
  Keywords
    maxPixels - the largest number of pixels; a larger picture is scaled down, its shape kept
    layout - the 2 x 2 matrix taking the plane to the screen, mapLayout; by default the
        identity
  See also
    mapCells, mapPoints, mapLayout, plotEBSD"""
  l = ebsd.lattice
  if l is None or len(l.ij) != ebsd.size or ebsd.size == 0:
    return None
  M = _layoutOf(layout)
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
  pad = ebsd._pad
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
    shown = index >= 0
    index[shown] = np.where(pad.reshape(-1)[index[shown]], -1, index[shown])
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
  """the corners of every cell of a map, shape (n, m, 2), on the screen

    V = mapCells(ebsd)                               # four corners a cell on a square grid
    V = mapCells(ebsd, layout=mapLayout('x↑→y'))     # turned to another convention

  A corner is shared by the cells around it and sits at the mean of where each of them puts
  it, the measured centre plus the ideal offset to the corner. On a rigid grid that is the
  ideal corner; under a stage drift the corners follow the centres, so the cells tile without
  gaps. A map on no lattice is refused.

  Input
    ebsd - EBSD on a lattice
  Output
    V - array of shape (n, m, 2), the m corners of the cell of each of the n measurements
  Keywords
    layout - the 2 x 2 matrix taking the plane to the screen, mapLayout; by default the
        identity
  See also
    mapRaster, mapPoints, mapLayout"""
  l = ebsd.lattice
  if l is None:
    raise ValueError("the map is not on a lattice, so it has no cells to draw")
  M = _layoutOf(layout)
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
  """the measurements as points on the screen

    P = mapPoints(ebsd)                              # the positions in the map's plane
    P = mapPoints(ebsd, mapLayout('x↑→y'))           # turned to another convention

  Input
    ebsd - EBSD
  Output
    P - array of shape (n, 2)
  Keywords
    layout - the 2 x 2 matrix taking the plane to the screen, mapLayout; by default the
        identity
  See also
    mapRaster, mapCells"""
  return planeOf(ebsd).to2d(ebsd.pos.data.reshape(-1, 3)) @ _layoutOf(layout).T


def planeOf(x):
  """the plane a map, grains or a boundary lie in: the lattice's for a map, else one fitted
  to its positions"""
  if isinstance(x, EBSD):
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
  """the extent of what a map plot has to show, on the screen: (xlo, xhi, ylo, yhi)

    e = mapExtent(ebsd)                              # a map
    e = mapExtent(grains, layout=mapLayout('x↑→y'))  # grains, turned to another convention

  A map reaches the unit cell around its outer measurements, half a spacing on a square grid,
  as MATLAB's plot, the padding of a grid left out; grains end at the vertices of their boundary, a boundary or a list of triple points at
  its vertices. Nothing to show gives zeros.

  Input
    x - EBSD, grain2d, grainBoundary or triplePointList
  Output
    e - (xlo, xhi, ylo, yhi)
  Keywords
    layout - the 2 x 2 matrix taking the plane to the screen, mapLayout; by default the
        identity
  See also
    scaleBar, mapLayout"""
  M = _layoutOf(layout)
  # a map reaches the cell around its outer measurements, anything else ends at its vertices
  cell = np.zeros((1, 2))
  if isinstance(x, EBSD):
    P = mapBox(x)
    if x.lattice is not None:
      cell = cellCorners(x.lattice.A, x.lattice.shape)
  elif isinstance(x, triplePointList):
    P = np.asarray(x.V).reshape(-1, 2) if len(x.grainId) else np.zeros((0, 2))
  else:
    # grains end at their boundary, a boundary or an image outline at its segments
    gB = x.boundary if isinstance(x, grain2d) else x
    P = gB.V[np.unique(gB.F)] if len(gB.F) else np.zeros((0, 2))
  if len(P) == 0:
    return (0.0, 0.0, 0.0, 0.0)
  Q, C = P @ M.T, cell @ M.T
  lo, hi = Q.min(axis=0) + C.min(axis=0), Q.max(axis=0) + C.max(axis=0)
  return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))


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


def scaleBar(x, unit="um", fraction=0.1, atMost=None):
  """how long a scale bar of a map should be, in the coordinates the map is drawn in, and how
  it reads: a round number near a tenth of the width, in the unit that keeps it readable

    length, label = scaleBar(ebsd)                   # (5000.0, '5 mm') for a map 36.6 mm wide
    length, label = scaleBar(36550.0, 'um', fraction=0.2)  # from a width in a unit
    length, label = scaleBar(0.5, 'mm')              # (0.05, '50 µm')
    length, label = scaleBar(36550.0, 'um', atMost=2000.0)  # (2000.0, '2 mm'), no longer than 2 mm

  The fraction of the width is brought to the unit among nm, µm, mm and m that keeps its
  number between one and a thousand and rounded to the nearest of 1, 2, 5, 10, 15, 20, 25,
  50, 75, 100, 125, 150, 200, 500, 750. A map, grains or boundary give their width on the
  screen and their scanUnit.

  Input
    x - the width of the map, or the map, grains or boundary
  Output
    length - the length of the bar in the units of the map
    label - its text, as '5 mm'
  Keywords
    unit - the unit of a width given as a number
    fraction - the length aimed at, as a fraction of the width
    atMost - a length the bar must not exceed; a longer one becomes the largest round length
        below it
  See also
    mapExtent, plotEBSD"""
  if not isinstance(x, (int, float, np.floating, np.integer)):
    e = mapExtent(x)
    return scaleBar(e[1] - e[0], getattr(x, "scanUnit", "um"), fraction, atMost)
  steps = np.asarray(SCALE_BAR_STEPS)
  y, u, f = switchUnit(fraction * x, unit)
  n = steps[int(np.argmin(np.abs(steps - y)))]
  if atMost is not None and n / f > atMost:
    y, u, f = switchUnit(atMost, unit)
    n = steps[max(int(np.searchsorted(steps, y, side="right")) - 1, 0)]
  return float(n / f), f"{n:g} {unitLabel(u)}"


# ------------------------------------------------------------------ the frame triad

def frameOnScreen(pl, layout=None):
  """the (right, up, towards the viewer) components of the axes of the frame in the picture of
  a map: the plane's coordinates taken to the screen by the layout, and the normal towards
  the viewer, who looks from the other side of the plane when the layout flips it

    s = frameOnScreen(ebsd.lattice.plane)                     # x, y, z under the identity
    s = frameOnScreen(ebsd.lattice.plane, mapLayout('x↑→y'))  # z pointing away

  Input
    pl - the plane of the map
  Output
    s - a list of three (right, up, out) triples, for x, y and z
  Keywords
    layout - the 2 x 2 matrix taking the plane to the screen, mapLayout; by default the
        identity
  See also
    triadGeometry, mapLayout"""
  M = _layoutOf(layout)
  side = np.sign(np.linalg.det(M))
  n = pl.normal
  out = []
  for a in (xvector, yvector, zvector):
    d = a.data.reshape(3)
    s = M @ np.array([d @ pl.e1, d @ pl.e2])
    out.append((float(s[0]), float(s[1]), float(side * (d @ n))))
  return out


def triadGeometry(screen, labels, t=1.0, box=None):
  """the reference frame indicator MTEX draws in the scale bar box and the import wizard in its
  pictograms, with the text height t as its unit: an arrow per axis that has a direction on
  screen, shortened by its tilt out of the plane, with its name beyond the head, and the name
  of the axis along the view at the crossing the arrows leave from - a right handed frame
  needs no symbol to tell out of the screen from into it. Every name stands the same gap off
  its arrow, measured to its ink. Returns the arrows as polygons, the names with their
  positions, and the bounding box (umin, umax, vmin, vmax)

    arrows, texts, bbox = triadGeometry(frameOnScreen(pl), ['x', 'y', 'z'])  # in text heights
    arrows, texts, bbox = triadGeometry(screen, labels, t=2.0)  # twice as large

  Input
    screen - the (right, up, out) triples of the axes, frameOnScreen
    labels - the name of every axis; an axis without one reserves no room
  Output
    arrows - a list of polygons of shape (7, 2)
    texts - a list of (position, name), the position the point a name is centred at
    bbox - (umin, umax, vmin, vmax) of the arrows, the names' ink and the crossing
  Keywords
    t - the height of the text, the unit of the drawing; the names are 0.8 t in size
    box - the ink of a name about the point it is centred at, (left, right, bottom, top), as
        the drawing measures it; by default estimated from the number of letters
  See also
    frameOnScreen, scaleBar"""
  e = 0.8 * t
  arm, headLen, shaftHalf, headHalf, gap = 2.55 * e, 0.78 * e, 0.125 * e, 0.27 * e, 0.45 * e
  if box is None:
    box = lambda s: (-0.31 * max(1, len(s)) * e, 0.31 * max(1, len(s)) * e, -0.36 * e, 0.36 * e)
  names = [labels[k] if k < len(labels) and labels[k] else "" for k in range(len(screen))]

  # how far the ink of a name reaches from its centre along the direction d
  def reach(b, d):
    return max(x * d[0] + y * d[1] for x in b[:2] for y in b[2:])

  inPlane = [np.hypot(s[0], s[1]) >= 0.15 for s in screen]
  kOut = next((k for k, p in enumerate(inPlane) if not p), None)
  hub = box(names[kOut]) if kOut is not None and names[kOut] else (0.0, 0.0, 0.0, 0.0)
  arrows, texts, pts = [], [], [np.zeros(2)]
  if kOut is not None and names[kOut]:
    texts.append((np.zeros(2), names[kOut]))
  for k, p in enumerate(inPlane):
    if not p:
      continue
    # the arrow starts clear of the name in the crossing, and is shortened by its tilt
    d = np.array([screen[k][0], screen[k][1]], dtype=float)
    u = d / np.linalg.norm(d)
    s0 = gap + reach(hub, u)
    tip = s0 + max(np.linalg.norm(d) * (arm - gap), headLen)
    b, q = tip - headLen, np.array([-u[1], u[0]])
    arrows.append(np.array([s0 * u + shaftHalf * q, b * u + shaftHalf * q, b * u + headHalf * q, tip * u,
                            b * u - headHalf * q, b * u - shaftHalf * q, s0 * u - shaftHalf * q]))
    if names[k]:
      texts.append(((tip + gap + reach(box(names[k]), -u)) * u, names[k]))
  pts += [p for a in arrows for p in a]
  for c, s in texts:
    l, r, lo, hi = box(s)
    pts += [c + (l, lo), c + (r, hi)]
  P = np.array(pts)
  bbox = (float(P[:, 0].min()), float(P[:, 0].max()), float(P[:, 1].min()), float(P[:, 1].max()))
  return arrows, texts, bbox


__all__ = ["mapLayout", "mapRaster", "mapCells", "mapPoints", "mapExtent", "boundaryPolyline", "boundarySegments",
           "grainPolygons", "scaleBar", "switchUnit", "triadGeometry", "frameOnScreen", "phaseColor", "phasePalette"]
