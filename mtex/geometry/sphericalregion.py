# fundamental sectors and spherical regions (MATLAB @sphericalRegion,
# @referenceFrame/fundamentalSector)
#
# The fundamental sector of a group is the conventional one, the sector MTEX draws and an
# inverse pole figure shows, wherever it is a fundamental domain of the group, which it is
# for every crystal frame. Its rules read the group in the axes it is written in, so for a
# group turned against them, as the rotations two frames share, the sector is the Dirichlet
# domain instead: the directions closer to a centre c than to any of its images s * c,
# bounded by the great circles with normals c - s * c, a fundamental domain for every
# group. A direction is projected into either by folding it with the group.
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING
import numpy as np

from .representations import unitRows
from .latticetype import latticeType
from .quaternion import orthogonalTransform, qrot, rotation
from .symmetry import symmetry
from .units import degree
from .symreduce import _distinctUnitRows
from .vector3d import _xyz, vector3d, xvector, yvector, zvector
from .display import display


class sphericalRegion:
  """a region on the sphere bounded by small circles: all directions v with dot(v, N_i) >= alpha_i
  for every normal N_i

    sR = sphericalRegion()                           # the whole sphere
    sR = sphericalRegion(N)                          # the intersection of the hemispheres about N
    sR = sphericalRegion(N, alpha)                   # the intersection of the caps dot(v, N_i) >= alpha_i
    sR = sphericalRegion(N, antipodal=True)          # a region of axes, v or -v inside
    sR = sphericalRegion('upper')                    # the hemisphere about z, 'lower' about -z
    sR = sphericalRegion(N, frame=cs)                # a region given in a reference frame
    sR = sphericalRegion.byVertices(V)               # the polygon through the vertices

  An alpha of 0 makes the boundary a great circle, a positive alpha a cap smaller than a
  hemisphere. The fundamental sectors of the point groups are spherical regions:
  fundamentalSector, dirichletSector. sR.checkInside(v) tests directions, sR.volume() gives the
  fraction of the sphere the region covers.

  Input
    N - vector3d, the normals of the bounding circles, or 'upper' or 'lower'
    alpha - array, the cosines of the bounding circles, one for all or one per normal; by
        default 0
  Output
    sR - sphericalRegion
  Keywords
    frame - the referenceFrame the region is given in
    antipodal - a region of axes: a direction is inside when v or -v is
  See also
    fundamentalSector, dirichletSector, sphericalRegion.checkInside, VectorGrids,
    FundamentalSector"""

  def __init__(self, N=None, alpha=None, frame=None, antipodal=False):
    if isinstance(N, sphericalRegion):
      self.N, self.alpha, self.frame, self.antipodal = N.N.copy(), N.alpha.copy(), N.frame, N.antipodal
      return
    if isinstance(N, str):
      N = vector3d(0, 0, 1 if N == "upper" else -1)
    if N is None:
      self.N = np.zeros((0, 3))
    else:
      self.N = unitRows(np.asarray(N.data if isinstance(N, vector3d) else N, dtype=float).reshape(-1, 3))
    self.alpha = np.zeros(len(self.N)) if alpha is None else np.broadcast_to(np.asarray(alpha, dtype=float), (len(self.N),)).copy()
    self.frame = frame
    self.antipodal = antipodal

  @classmethod
  def byVertices(cls, V, frame=None):
    """the region bounded by the great circles through consecutive vertices V, taken counterclockwise

      sR = sphericalRegion.byVertices(vector3d([[0, 0, 1], [1, 0, 1], [1, 1, 1]]))  # a triangle
      sR = sphericalRegion.byVertices(V, frame=cs)  # in a crystal frame

    The normal of every side is the cross product of two consecutive vertices, the last
    joined to the first, so the region lies to the left when walking along the vertices.

    Input
      V - vector3d, the vertices in counterclockwise order
    Output
      sR - sphericalRegion
    Keywords
      frame - the frame the region is written in
    See also
      sphericalRegion, sphericalRegion.vertices"""
    V = np.asarray(V.data if isinstance(V, vector3d) else V, dtype=float).reshape(-1, 3)
    N = np.cross(V, np.roll(V, -1, axis=0))
    return cls(N, frame=frame)

  # ------------------------------------------------------------------ membership

  def checkInside(self, v, tol=1e-4):
    """whether directions lie in the region, up to a small tolerance at the boundary

      inside = sR.checkInside(v)                    # a boolean per direction
      inside = sR.checkInside(v, tol=1e-2)          # a wider margin at the boundary

    A direction is inside when <v, N> >= alpha - tol for every boundary circle, the length
    of v not mattering. For an antipodal region or antipodal directions, -v inside counts too.

    Input
      sR - sphericalRegion
      v - vector3d, or an (..., 3) array
    Output
      inside - boolean array in the shape of v
    Keywords
      tol - the margin in the cosines
    See also
      projectIntoRegion, sphericalRegion.volume"""
    xyz = _xyz(v)
    if len(self.N) == 0:
      return np.ones(xyz.shape[:-1], dtype=bool)
    n = np.linalg.norm(xyz, axis=-1)
    inside = np.all(xyz @ self.N.T >= (self.alpha - tol) * n[..., None], axis=-1)
    if self.antipodal or (isinstance(v, vector3d) and v.antipodal):
      inside |= np.all(-xyz @ self.N.T >= (self.alpha - tol) * n[..., None], axis=-1)
    return inside

  def volume(self, resolution=1 * degree):
    """the fraction of the sphere the region covers, by a quadrature on a fine grid

      a = sR.volume()                               # on a grid of 1 degree
      a = sR.volume(resolution=0.25*degree)         # finer, more accurate

    The share of the points of a regular grid in polar coordinates inside the region,
    weighted by sin(theta); 1 is the whole sphere, so 4 pi times it is the area. Antipodal
    regions count -v as well.

    Input
      sR - sphericalRegion
    Output
      a - the fraction of the sphere, between 0 and 1
    Keywords
      resolution - the spacing of the quadrature grid
    See also
      sphericalRegion.checkInside, sphericalRegion.center"""
    theta = np.arange(resolution / 2, np.pi, resolution)
    rho = np.arange(0, 2 * np.pi - resolution / 2, resolution)
    th, rh = np.meshgrid(theta, rho, indexing="ij")
    w = np.sin(th)
    v = vector3d.byPolar(th, rh)
    return np.sum(w * self.checkInside(v)) / np.sum(w)

  def center(self, resolution=1 * degree):
    """the centre (MATLAB @sphericalRegion/center): the pole of a single circle, the bisector of
    the two boundaries furthest apart for a lune, the barycentre of the vertices of a
    triangle, and the area centroid of anything larger

      c = sR.center()                               # the centre of the region
      c = sR.center(resolution=0.5*degree)          # the centroid on a finer grid

    The whole sphere has its centre at z. The area centroid is the normalised sum of the
    points of an equispaced grid inside the region; resolution is its spacing and matters
    only there.

    Input
      sR - sphericalRegion
    Output
      c - vector3d, a unit vector in the frame of the region
    Keywords
      resolution - the spacing of the grid of the area centroid
    See also
      sphericalRegion.vertices, sectorCenter"""
    N = _distinctUnitRows(self.N, 1e-9)
    if len(N) < 2:
      return vector3d(np.array([0.0, 0.0, 1.0]) if len(N) == 0 else unitRows(N[0]), isNormalized=True, frame=self.frame)
    V = self.vertices().data.reshape(-1, 3)
    if len(V) < 3:
      pairs = [(i, j) for i in range(len(N) - 1) for j in range(i + 1, len(N))]
      i, j = max(pairs, key=lambda ij: np.arccos(np.clip(N[ij[0]] @ N[ij[1]], -1, 1)))
      return vector3d(unitRows(N[i] + N[j]), isNormalized=True, frame=self.frame)
    if len(V) < 4:
      return vector3d(unitRows(np.sum(V, axis=0)), isNormalized=True, frame=self.frame)
    from .grids.s2grid import equispacedS2Grid
    g = equispacedS2Grid(resolution=resolution)
    inside = g.data[self.checkInside(g)]
    c = np.sum(inside, axis=0) if len(inside) else np.zeros(3)
    return vector3d(np.array([0.0, 0.0, 1.0]) if np.linalg.norm(c) < 1e-4 else unitRows(c), isNormalized=True, frame=self.frame)

  def __eq__(self, other):
    return (isinstance(other, sphericalRegion) and self.N.shape == other.N.shape
            and np.allclose(self.N, other.N) and np.allclose(self.alpha, other.alpha))

  __hash__ = None

  def restrictToUpper(self, up=None):
    """the hemisphere seen from a direction added to the region (MATLAB's sphericalRegion.restrict2Upper)

      sR = sR.restrictToUpper()                     # the part with z >= 0
      sR = sR.restrictToUpper(xvector)              # the part towards x
      sR = sR.restrictToUpper(plottingConvention('y↑→x'))  # towards the viewer of a convention

    The region is returned as it is when it already lies in that hemisphere.

    Input
      sR - sphericalRegion
      up - vector3d, or a plottingConvention whose outOfScreen direction is taken; z by
          default
    Output
      sR - sphericalRegion
    See also
      sphericalRegion.restrictToLower"""
    from .plottingconvention import plottingConvention
    up = zvector if up is None else up.outOfScreen if isinstance(up, plottingConvention) else up
    u = unitRows(_xyz(up))
    if np.any((self.N @ u > 1 - 1e-10) & (self.alpha >= 0)):
      return self
    return sphericalRegion(np.concatenate([u[None], self.N]), np.concatenate([[0.0], self.alpha]), self.frame, self.antipodal)

  def restrictToLower(self, up=None):
    """the hemisphere opposite to a direction, z by default, added to the region
    (MATLAB's sphericalRegion.restrict2Lower)

      sR = sR.restrictToLower()                     # the part with z <= 0
      sR = sR.restrictToLower(xvector)              # the part away from x

    Input
      sR - sphericalRegion
      up - vector3d, z by default
    Output
      sR - sphericalRegion
    See also
      sphericalRegion.restrictToUpper"""
    up = zvector if up is None else up
    return self.restrictToUpper(-up if isinstance(up, vector3d) else -np.asarray(up, dtype=float))

  # ------------------------------------------------------------------ the shape

  def thetaIntervals(self, rho):
    """the polar intervals of the region along the meridian rho: every circle
    dot(v, n) >= alpha cuts the meridian in R cos(theta - phi) >= alpha, whose solutions
    are arcs; a cap taken away can leave the meridian in two pieces, so a list"""
    segs = [(0.0, float(np.pi))]
    for n, alpha in zip(self.N, self.alpha):
      a = n[0] * np.cos(rho) + n[1] * np.sin(rho)
      r = np.hypot(a, n[2])
      if r < 1e-12:
        if alpha > 0:
          return []
        continue
      c = alpha / r
      if c > 1:
        return []
      if c < -1:
        continue
      w, phi = np.arccos(c), np.arctan2(a, n[2])
      allowed = []
      for k in (-1, 0, 1):
        lo, hi = max(phi - w + 2 * np.pi * k, 0.0), min(phi + w + 2 * np.pi * k, float(np.pi))
        if lo < hi:
          allowed.append((lo, hi))
      segs = sorted((max(l1, l2), min(h1, h2)) for l1, h1 in segs for l2, h2 in allowed if max(l1, l2) < min(h1, h2))
      if not segs:
        return segs
    return segs

  def thetaInterval(self, rho):
    """the polar interval (min, max) of the region along the meridian rho, None where it misses it"""
    segs = self.thetaIntervals(rho)
    return None if not segs else (segs[0][0], segs[-1][1])

  def vertices(self, tol=1e-8):
    """the corners of the region: the intersections of its circles that lie inside

      V = sR.vertices()                             # the corners, each once

    Input
      sR - sphericalRegion
    Output
      V - vector3d of unit vectors in the frame of the region
    Keywords
      tol - the margin by which an intersection may lie outside the other circles
    See also
      sphericalRegion.boundary, sphericalRegion.byVertices"""
    cand = [v for i, j in itertools.combinations(range(len(self.N)), 2)
            for v in _circleIntersection(self.N[i], self.alpha[i], self.N[j], self.alpha[j])]
    cand = np.array(cand).reshape(-1, 3)
    V = _distinctUnitRows(cand[self.checkInside(cand, tol)], 1e-9)
    return vector3d(V, isNormalized=True, frame=self.frame)

  def boundary(self, resolution=1 * degree):
    """the boundary as arcs, one vector3d per arc: each circle is cut at the corners on it and
    the pieces whose midpoint lies inside are kept

      arcs = sR.boundary()                          # points every degree along each arc
      arcs = sR.boundary(resolution=5*degree)       # coarser

    Input
      sR - sphericalRegion
    Output
      arcs - list of vector3d, the points along every arc of the boundary
    Keywords
      resolution - the largest spacing of the points along an arc
    See also
      sphericalRegion.vertices, plot"""
    V = self.vertices().data
    arcs = []
    for n, alpha in zip(self.N, self.alpha):
      r = np.sqrt(max(1 - alpha ** 2, 0.0))
      if r < 1e-10:
        continue
      u = vector3d(n).orth().data
      w = np.cross(n, u)
      def point(t):
        return unitRows(alpha * n + r * (np.cos(t)[..., None] * u + np.sin(t)[..., None] * w))
      ts = sorted(np.arctan2(v @ w, v @ u) for v in V if abs(v @ n - alpha) < 1e-8)
      pieces = [(0.0, 2 * np.pi)] if not ts else [(ts[k], ts[k + 1] if k + 1 < len(ts) else ts[0] + 2 * np.pi) for k in range(len(ts))]
      for t0, t1 in pieces:
        if t1 - t0 < 1e-10 or not self.checkInside(point(np.array((t0 + t1) / 2)), 1e-8):
          continue
        m = max(int(np.ceil(r * (t1 - t0) / resolution)), 1)
        arcs.append(vector3d(point(np.linspace(t0, t1, m + 1)), isNormalized=True, frame=self.frame))
    return arcs

  def __repr__(self):
    return display("sphericalRegion", [("faces", len(self.N)), ("vertices", 0 if len(self.N) == 0 else self.vertices().size),
                                       ("antipodal", "true" if self.antipodal else None)])

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..plotting.sphericalplot import plotRegion as plot


# the two points where the circles dot(v, n1) = a1 and dot(v, n2) = a2 meet
def _circleIntersection(n1, a1, n2, a2):
  m = n1 @ n2
  if 1 - m ** 2 < 1e-12:
    return []
  a = (a1 - m * a2) / (1 - m ** 2)
  b = (a2 - m * a1) / (1 - m ** 2)
  c2 = (1 - a ** 2 - b ** 2 - 2 * a * b * m) / (1 - m ** 2)
  if c2 < -1e-12:
    return []
  w = np.cross(n1, n2)
  c = np.sqrt(max(c2, 0.0))
  base = a * n1 + b * n2
  return [unitRows(base)] if c < 1e-10 else [unitRows(base + c * w), unitRows(base - c * w)]


# ------------------------------------------------------------------ the fundamental sector

def _elements(pg):
  return pg.rot.data.reshape(-1, 5)


def _apply(els, v):
  return qrot(els[:, :4], v[None]) * els[:, 4:5]


# the symmetry directions of a lattice system, as symmetry.py reads them off a symbol
def _symmetryDirections(pg):
  from .symmetry import _symmetryDirections
  a, b, c = (vector3d(pg.axes[k]) for k in range(3))
  return _symmetryDirections(pg.lattice, a, b, c)


def sectorCenter(pg):
  """the centre of the Dirichlet domain: the barycentre of the conventional sector - the
  triangle c, a, tertiary for the tetragonal and hexagonal systems, [001], [101], [111]
  for the cubic one - in the axes the group is written in; the sum of the axes otherwise.
  A centre some element other than the identity fixes has a cell too large, as the axes of
  a group turned against its own axes can give, so it is moved off every such element"""
  from .frames import referenceFrame
  if isinstance(pg, referenceFrame):
    fr = pg
    if fr.lattice in (latticeType.triclinic, latticeType.monoclinic):
      N = conventionalSector(fr).N
      if len(N):
        return _offFixed(fr.sym, unitRows(np.sum(N, axis=0)))
    pg = fr.sym
  a, b, c = (np.asarray(pg.axes[k], dtype=float) for k in range(3))
  if pg.lattice is latticeType.cubic:
    return _offFixed(pg, unitRows(unitRows(c) + unitRows(a + c) + unitRows(a + b + c)))
  if pg.lattice is latticeType.icosahedral and pg.numProper() == 60:
    # the centre of the Moebius triangle between neighbouring five-, three- and twofold
    # axes, the kite of two such triangles for the proper group, whatever the alignment
    R = pg.rot[~pg.rot.isImproper]
    omega, axes = R.angle(), R.axis().data
    # a group generated from rounded axes closes only approximately, so the orders are read
    # with a tolerance far below the gap between them
    def fold(order):
      return axes[np.abs(omega - 2 * np.pi / order) < 1e-3]
    f, threes = fold(5)[0], fold(3)
    t = threes[np.argmax(np.abs(threes @ f))]
    t = t if t @ f > 0 else -t
    w = max(fold(2), key=lambda x: abs(x @ f) + abs(x @ t))
    w = w if w @ (f + t) > 0 else -w
    return vector3d(unitRows(f + t + w if pg.isLaue else f + t), isNormalized=True)
  dirs = _symmetryDirections(pg)
  return _offFixed(pg, unitRows(np.sum([d.normalize().data for d in dirs], axis=0)))


# a centre no element but the identity fixes: the direction itself, or the first of a fixed
# sequence of nearby directions that is
def _offFixed(pg, c):
  els = _elements(pg)
  for k in range(12):
    d = c if k == 0 else unitRows(c + 0.1 * k * unitRows(np.array([1.0, 2.0 ** 0.5, 3.0 ** 0.5])) * (-1) ** k)
    if np.linalg.norm(d) > 0 and np.sum(np.linalg.norm(_apply(els, d) - d, axis=1) < 1e-8) <= 1:
      return vector3d(d, isNormalized=True)
  return vector3d(c, isNormalized=True)


# the bisectors that bound the domain: a great circle is a face when it carries an edge,
# two distinct vertices inside the domain
def _essentialNormals(N, tol=1e-4):
  if len(N) <= 1:
    return N
  # a group generated from rounded axes meets at its corners only to about 1e-5, so a corner
  # is taken to that tolerance, far below the distance between the corners of any sector
  i, j = np.triu_indices(len(N), 1)
  w = np.cross(N[i], N[j])
  w = unitRows(w[np.linalg.norm(w, axis=1) >= 1e-10])
  # both ends of every edge line, in the order of the pairs, those inside the domain
  cand = np.stack([w, -w], axis=1).reshape(-1, 3)
  verts = _distinctUnitRows(cand[np.all(cand @ N.T >= -tol, axis=1)], tol ** 2)
  if not len(verts):
    return N
  return np.array([n for n in N if sum(abs(n @ v) < tol for v in verts) >= 2])


def dirichletSector(pg, center=None, frame=None, antipodal=False):
  """the Dirichlet domain of a point group about a centre, a fundamental domain for every group,
  whatever axes it is written in

    sR = dirichletSector(cs)                             # about the centre of the conventional sector
    sR = dirichletSector(cs, center=vector3d(1, 2, 3))   # about another centre
    sR = dirichletSector(cs, antipodal=True)             # of the Laue group, for axes
    sR = dirichletSector(pg, frame=cs)                   # of a group, the region given in a frame

  The region is bounded by the great circles bisecting the centre and each of its images
  under the group: the directions nearer the centre than any of its other images. Where the
  centre is not given, it is the centre of the conventional sector, or, when the inversion
  cancels it, the one of the proper subgroup. fundamentalSector takes the Dirichlet domain only
  for a group the conventional sector does not fit, such as the rotations two frames share.

  Input
    pg - a point group, or a reference frame, whose group is taken and which the region is
        given in
  Output
    sR - sphericalRegion
  Keywords
    center - vector3d, the centre of the domain; by default the centre of the conventional
        sector
    frame - the referenceFrame the region is given in
    antipodal - the domain of the Laue group, for axes
  See also
    fundamentalSector, sphericalRegion, tilesSphere, FundamentalSector"""
  from .frames import referenceFrame
  if isinstance(pg, referenceFrame):
    frame = pg
    if center is None:
      center = sectorCenter(pg)
    pg = pg.sym
  if antipodal:
    pg = pg.Laue()
  c = sectorCenter(pg).data if center is None else _xyz(center)
  # a centre the inversion cancels, as for the centrosymmetric icosahedral group: the
  # centre of the proper subgroup's sector serves, its Dirichlet cell under the whole group
  if center is None and (not np.all(np.isfinite(c)) or np.linalg.norm(c) < 1e-8):
    c = sectorCenter(pg.properSubGroup()).data
  c = unitRows(c.reshape(3))
  els = _elements(pg)
  d = c[None] - _apply(els, c)
  keep = np.linalg.norm(d, axis=1) >= 1e-10
  N = _essentialNormals(_distinctUnitRows(unitRows(d[keep]), 1e-10))
  return sphericalRegion(N, np.zeros(len(N)), frame, antipodal)


def fundamentalSector(pg, center=None, frame=None, antipodal=False, angle=None):
  """the fundamental sector: the conventional one where it is a fundamental domain of the
  group, which it is for every crystal frame; the Dirichlet domain for a group the rules of
  the conventional sector do not fit, as the rotations two frames share, turned against
  the axes they are written in; with a centre given, that centre's Dirichlet domain"""
  from .frames import referenceFrame
  if angle is not None:
    from .orientationregion import fundamentalRegion
    return fundamentalRegion(pg if frame is None else frame, antipodal=antipodal).axisSector(angle)
  if center is not None:
    return dirichletSector(pg, center, frame, antipodal)
  up = east = None
  if isinstance(pg, referenceFrame):
    frame = pg
    pC = pg.how2plot
    up, east = pC.outOfScreen, pC.east
    pg = pg.sym
  grp = pg.Laue() if antipodal else pg
  sR = conventionalSector(grp, up=up, east=east, frame=frame)
  if not tilesSphere(sR, grp):
    sR = dirichletSector(grp, frame=frame)
  sR.antipodal = antipodal
  return sR


# whether a region is a fundamental domain of a group: six thousand evenly spread
# directions are folded into it by the group, every one must land inside, and the region
# must cover the share of the sphere one element's image does; the answer is kept per
# group and region
_TILES = {}


def tilesSphere(sR, pg, n=6000):
  """whether the region is a fundamental domain of the point group: n evenly spread directions
  folded into it all land inside, and it covers the share of one element; cached

    tilesSphere(cs.fundamentalSector(), cs.sym)   # True for every crystal frame
    tilesSphere(sR, cs.sym, n=20000)              # more test directions

  The region covers the share of one element when the fraction of the test directions
  inside it times the order of the group is 1 up to 0.15. The answer is kept per group and
  region, so n matters only on the first call.

  Input
    sR - sphericalRegion
    pg - the point group, or a frame holding it
  Output
    b - True or False
  Keywords
    n - the number of test directions, a Fibonacci lattice
  See also
    fundamentalSector, dirichletSector, projectIntoRegion"""
  from .kernels import projectIntoSector
  els = _elements(pg)
  key = (np.round(els, 9).tobytes(), np.round(sR.N, 9).tobytes(), np.round(sR.alpha, 9).tobytes())
  hit = _TILES.get(key)
  if hit is None:
    plain = sphericalRegion(sR.N, sR.alpha)
    P = _fibonacciSphere(n)
    share = plain.checkInside(P).mean() * len(els)
    folded = projectIntoSector(P, els, plain.N, plain.alpha, regionAnti=False, vAnti=False)
    hit = bool(abs(share - 1) < 0.15 and np.all(plain.checkInside(folded)))
    _TILES[key] = hit
  return hit


# n directions spread evenly over the sphere, the Fibonacci lattice
def _fibonacciSphere(n):
  k = np.arange(n) + 0.5
  z = 1 - 2 * k / n
  rho = np.pi * (1 + 5 ** 0.5) * k
  r = np.sqrt(1 - z ** 2)
  return np.stack([r * np.cos(rho), r * np.sin(rho), z], axis=1)


def conventionalSector(pg, up=None, east=None, frame=None):
  """the conventional sector, the one MTEX draws: the hemisphere towards up when some element
  sends up below it, a wedge about the c axis of the angle the order leaves, starting on a
  mirror plane through the axis, else on a two-fold axis in the plane, else at the a axis,
  and the standard triangles of the cubic groups"""
  from .frames import referenceFrame
  if isinstance(pg, referenceFrame):
    frame = pg
    pC = pg.how2plot
    up = pC.outOfScreen if up is None else up
    east = pC.east if east is None else east
    pg = pg.sym
  n = pg.numSym()
  u = unitRows(zvector.data if up is None else _xyz(up))
  east = np.asarray(pg.axes[1], dtype=float) if east is None else _xyz(east)
  lat = pg.lattice
  if lat is latticeType.triclinic:
    return sphericalRegion(None if n == 1 else u[None], frame=frame)
  if lat is latticeType.monoclinic:
    return _monoclinicSector(pg, u, east, frame)
  if lat is latticeType.icosahedral and pg.numProper() == 60:
    return dirichletSector(pg, frame=frame)
  # the sector is built about c* with a at azimuth zero, the conventional alignment, and
  # turned to wherever the lattice points (MATLAB a3e47891a)
  R = _latticeTurn(pg)
  if lat is latticeType.cubic:
    if n == 48:
      planes = [(1, -1, 0), (-1, 0, 1), (0, 1, 0)]
    elif n == 24 and (pg.isProper or pg.isLaue):
      planes = [(0, -1, 1), (-1, 0, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    elif n == 24:
      planes = [(1, -1, 0), (1, 1, 0), (-1, 0, 1)]
    else:
      planes = [(1, 1, 0), (1, -1, 0), (0, 1, 1), (0, -1, 1)]
    return sphericalRegion(_snap(np.array(planes, dtype=float) @ R.T), frame=frame)
  N = []
  els = _elements(pg)
  # an element that turns the out-of-screen axis around sends everything to its hemisphere
  if np.any(_apply(els, u) @ u < -1 + 1e-7):
    N.append(u)
  if n > 1 + len(N):
    drho = 2 * np.pi * (1 + len(N)) / n
    rho0 = _wedgeStart(pg, R)
    wedge = np.array([[np.cos(rho0 + np.pi / 2), np.sin(rho0 + np.pi / 2), 0.0],
                      [np.cos(rho0 + drho - np.pi / 2), np.sin(rho0 + drho - np.pi / 2), 0.0]])
    N += list(wedge @ R.T)
  N = _snap(np.array(N).reshape(-1, 3))
  # a plane given twice bounds once, as the hemisphere of 222 and -4 is also a wedge plane
  N = N[[k for k in range(len(N)) if not any(np.allclose(N[k], N[j], atol=1e-9) for j in range(k))]]
  # an orthorhombic sector with a mirror opens east on the chart, so the plotting
  # convention turns it, as the monoclinic one is turned
  if lat is latticeType.orthorhombic and not pg.isProper:
    N = _openEast(N, u, unitRows(east))
  return sphericalRegion(N, frame=frame)


# the round-off of the turn to the lattice set to zero, so that a normal along an axis has
# exact zeros
def _snap(N):
  return np.where(np.abs(N) < 1e-12, 0.0, N)


# the normals lying in the plane of the chart, flipped so that the sector they bound opens
# east of the centre (MATLAB fundamentalSector, its mmm and mm2 cases)
def _openEast(N, up, east):
  north = np.cross(up, east)
  out = N.copy()
  for i, n in enumerate(out):
    if abs(n @ up) > 1e-8:
      continue
    if np.mod(np.arctan2(n @ north, n @ east) + 5 * degree, 2 * np.pi) > np.pi:
      out[i] = -n
  return out


# a monoclinic domain is bounded by the unique symmetry axis and/or a plane containing it
def _monoclinicSector(pg, up, east, frame):
  o = pg.rot
  two = o[np.abs(o.angle() - np.pi) < 1e-8][0]
  a = two.axis().data
  k = int(np.argmax([abs(a @ unitRows(np.asarray(pg.axes[j], dtype=float))) for j in range(3)]))
  if a @ pg.axes[k] < 0:
    a = -a
  if pg.numSym() == 2 and not pg.isProper:
    if pg.id == 4:
      a = -a
    if a @ up < -1e-8:
      a = -a
    return sphericalRegion(a[None], frame=frame)
  p = up - (up @ a) * a
  if np.linalg.norm(p) < 1e-8:
    for v in (east, *[np.asarray(pg.axes[j], dtype=float) for j in range(3)]):
      if np.linalg.norm(np.cross(a, v)) > 1e-8:
        p = np.cross(a, v)
        break
  normal = unitRows(p)
  if pg.isProper:
    return sphericalRegion(normal[None], frame=frame)
  if abs(a @ up) > 1e-8:
    if a @ up < 0:
      a = -a
  else:
    azimuth = np.arctan2(a @ np.cross(up, east), a @ east)
    if np.mod(azimuth + 5 * degree, 2 * np.pi) > np.pi:
      a = -a
  return sphericalRegion(np.stack([normal, a]), frame=frame)


# the turn taking z to c* and x to a, so that the conventional sector built about z with a
# at azimuth zero sits on the lattice, MATLAB's rotation.map(z, c*, x, a); the identity for a
# group on the canonical axes
def _latticeTurn(pg):
  a = unitRows(np.asarray(pg.axes[0], dtype=float))
  c = unitRows(np.cross(a, np.asarray(pg.axes[1], dtype=float)))
  x = unitRows(a - (a @ c) * c)
  return np.stack([x, np.cross(c, x), c], axis=1)


# the azimuth the wedge starts at, in the turned position R gives: the mirror plane through
# the axis, else the two-fold axis in the plane, at or just below the a axis at azimuth
# zero; the a axis itself without either
def _wedgeStart(pg, R):
  o = pg.rot
  sel = (np.abs(o.angle() - np.pi) < 1e-6)
  ax = o[sel].axis().data.reshape(-1, 3) @ R
  inPlane = np.abs(ax[:, 2]) < 1e-6
  improper = o[sel].isImproper
  rho = np.arctan2(ax[:, 1], ax[:, 0])
  planes = rho[inPlane & improper] + np.pi / 2
  axes = rho[inPlane & ~improper]
  cand = planes if len(planes) else axes
  if len(cand) == 0:
    return 0.0
  return -np.min(np.mod(-cand + 1e-9, np.pi) - 1e-9)


def projectIntoRegion(v, sR, pg=None):
  """the image of a direction that lies in a region, under the group; the antipode of an axis
  counts as an image; the direction itself when no image lies inside

    w = projectIntoRegion(v, sR)                  # under the group of v's frame
    w = projectIntoRegion(v, sR, cs.sym)          # under another group

  The antipode counts when v or the region is antipodal.

  Input
    v - vector3d or Miller
    sR - sphericalRegion
    pg - the point group, by default that of the frame of v
  Output
    w - the images, of the class and shape of v
  See also
    sphericalRegion.checkInside, projectIntoFundamentalRegion, tilesSphere"""
  from .kernels import projectIntoSector
  pg = v.frame.sym if pg is None else pg
  xyz = v.data.reshape(-1, 3)
  if len(xyz) == 0:
    return v
  out = projectIntoSector(xyz, _elements(pg), sR.N, sR.alpha, regionAnti=bool(sR.antipodal), vAnti=bool(v.antipodal))
  return v._new(out.reshape(v.data.shape))


__all__ = ["sphericalRegion", "fundamentalSector", "conventionalSector", "dirichletSector", "tilesSphere", "sectorCenter",
           "projectIntoRegion"]
