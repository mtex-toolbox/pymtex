# the geometry of spherical plots, without graphics (MATLAB @sphericalProjection,
# makeSphericalProjection; docs/plans/plotting.md S1)
#
# A spherical plot draws a region of the sphere on a plane. The plotting convention says
# which direction points out of the screen and which east; in its frame a direction has
# the polar angle theta and the azimuth rho, and a projection is one radial function
# r(theta), the point drawn at r (cos rho, sin rho). A region reaching into both
# hemispheres is drawn as two charts, the lower one with theta -> pi - theta, so that it
# is seen through the sphere from above as in MTEX; an antipodal chart draws the antipode
# of a direction that falls into the other hemisphere. The chart also knows the grid a
# function is evaluated on, the boundary of its region, the grid lines and the labels,
# all as plane coordinates, which is what the drawing layer draws.
from __future__ import annotations

import numpy as np

from ..geometry.frames import referenceFrame
from ..geometry.frames import crystalFrame
from ..geometry.miller import brackets, indexString
from ..geometry.plottingconvention import plottingConvention
from ..geometry.quaternion import rotation
from ..geometry.session import getMTEXpref
from ..geometry.sphericalregion import conventionalSector, sphericalRegion
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, xvector, yvector, zvector, asVector3d
from ..geometry.display import aligned
from ..geometry.arguments import refuseFlags


class sphericalProjection:
  """a projection of the sphere to the plane, restricted to a region and a hemisphere

    sphericalProjection(sR, pC, antipodal=False, upper=True)
    proj.project(v)        the plane coordinates, NaN outside the region
    proj.iproject(x, y)    the direction drawn at a point of the plane

  Fields: sR, the region shown; pC, the plotting convention; antipodal, whether v and -v
  are one point; upper, whether the chart shows the hemisphere out of the screen;
  symmetric, whether the data carry the symmetry of the frame, as a function does.
  """

  name = "earea"
  symmetric = False

  def __init__(self, sR=None, pC=None, antipodal=False, upper=True):
    self.sR = sphericalRegion(frame=None) if sR is None else sR
    self.pC = plottingConvention(pC) if pC is not None else _conventionOf(self.sR.frame)
    self.antipodal = bool(antipodal)
    self.upper = bool(upper)

  # ------------------------------------------------------------------ the radial function

  @staticmethod
  def radius(theta):
    """the plane radius r(theta) a polar angle is drawn at, 2 sin(theta / 2) for the equal area chart"""
    return 2 * np.sin(theta / 2)

  @staticmethod
  def polarAngle(r):
    """the polar angle drawn at the plane radius r, the inverse of radius"""
    return 2 * np.arcsin(np.clip(r / 2, 0, 1))

  @classmethod
  def extent(cls):
    """the radius of the whole hemisphere, the natural extent of a chart"""
    return cls.radius(np.pi / 2)

  @property
  def frame(self):
    """the frame of the region shown"""
    return self.sR.frame

  @property
  def isPlain(self):
    """whether the chart draws the polar angles as plain coordinates, False here"""
    return False

  # ------------------------------------------------------------------ projecting

  def polarIn(self, v):
    """the polar coordinates of directions in the frame of the convention, folded into the
    hemisphere of the chart; NaN outside the region"""
    v = asVector3d(v).normalize()
    w = self.pC.rot.inv() * v
    if self.antipodal:
      flip = w.z < -1e-10 if self.upper else w.z > 1e-10
      w = w._new(np.where(flip[..., None], -w.data, w.data))
    inside = self.sR.checkInside(self.pC.rot * w)
    theta, rho = w.theta, w.rho
    theta = theta if self.upper else np.pi - theta
    nan = np.where(inside, 0.0, np.nan)
    return theta + nan, rho + nan

  def project(self, v):
    """the plane coordinates (x, y) of directions, NaN outside the region"""
    theta, rho = self.polarIn(v)
    r = self.radius(theta)
    return r * np.cos(rho), r * np.sin(rho)

  def projectUnchecked(self, v):
    """the plane coordinates of directions known to lie in the chart"""
    w = self.pC.rot.inv() * asVector3d(v)
    theta, rho = w.theta, w.rho
    theta = theta if self.upper else np.pi - theta
    r = self.radius(theta)
    return r * np.cos(rho), r * np.sin(rho)

  def iproject(self, x, y):
    """the direction drawn at the plane point (x, y)"""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    theta = self.polarAngle(np.hypot(x, y))
    theta = theta if self.upper else np.pi - theta
    return self.pC.rot * vector3d.byPolar(theta, np.arctan2(y, x))

  # ------------------------------------------------------------------ the shape of the chart

  def withRegion(self, sR):
    """the same chart showing another region, for an outline drawn behind the data"""
    sR = sR.restrictToUpper(self.pC) if self.upper else sR.restrictToLower(self.pC.outOfScreen)
    return type(self)(sR, self.pC, self.antipodal, self.upper)

  def chartRegion(self):
    """the region as seen in the frame of the convention, folded into the upper hemisphere"""
    N = (self.pC.rot.inv() * vector3d(self.sR.N)).data.reshape(-1, 3)
    if not self.upper:
      N = N * np.array([1.0, 1.0, -1.0])
    return sphericalRegion(N, self.sR.alpha)

  def grid(self, resolution=1 * degree):
    """a grid regular in the polar angles of the chart, one strip of polar angles per azimuth
    so that the cells follow the boundary: the directions, in the frame of the region, and
    their plane coordinates, all nTheta x nRho; a column of NaN separates the strips of a
    region whose polar set has two branches, where the pieces do not touch"""
    sR = self.chartRegion()
    rhoLo, rhoHi = rhoRange(sR)
    nRho = max(2, int(round((rhoHi - rhoLo) / resolution)) + 1)
    rhos = np.linspace(rhoLo, rhoHi, nRho)
    ivs = [sR.thetaIntervals(rho) for rho in rhos]
    nb = max((len(iv) for iv in ivs), default=0)
    if nb == 0:
      return _emptyGrid(self.frame)
    # where a branch has run out its strip collapses onto the end of the one before it, so
    # that the surface closes where the two pieces meet instead of leaving a seam
    cols = []
    for b in range(nb):
      if b > 0:
        cols.append(None)
      for rho, iv in zip(rhos, ivs):
        if len(iv) > b:
          cols.append((rho, iv[b][0], iv[b][1]))
        elif iv:
          cols.append((rho, iv[-1][1], iv[-1][1]))
        else:
          cols.append(None)
    while cols and cols[0] is None:
      cols.pop(0)
    while cols and cols[-1] is None:
      cols.pop()
    if not cols:
      return _emptyGrid(self.frame)
    width = max(hi - lo for col in cols if col is not None for (_, lo, hi) in (col,))
    nTheta = max(3, 2 * int(round(width / resolution / 2)) + 1)
    t = np.linspace(0, 1, nTheta)
    theta = np.empty((nTheta, len(cols)))
    rho = np.empty((nTheta, len(cols)))
    blank = np.zeros(len(cols), dtype=bool)
    last = (rhos[0], 0.0, np.pi)
    for j, col in enumerate(cols):
      blank[j] = col is None
      r, lo, hi = last if col is None else col
      last = last if col is None else col
      theta[:, j] = lo + t * (hi - lo)
      rho[:, j] = r
    return self._gridFrom(theta, rho, blank)

  def _gridFrom(self, theta, rho, blank):
    th = theta if self.upper else np.pi - theta
    V = self.pC.rot * vector3d.byPolar(th, rho)
    V = vector3d(V.data, frame=self.frame, isNormalized=True)
    X, Y = self.projectUnchecked(V)
    X[:, blank], Y[:, blank] = np.nan, np.nan
    return V, X, Y

  def boundary(self, resolution=0.5 * degree):
    """the boundary of the region as plane curves, one (x, y) pair per arc; a whole
    hemisphere has no boundary of its own and gets the horizon"""
    arcs = self.sR.boundary(resolution)
    if not arcs:
      arcs = [self._horizon()]
    return [self.projectUnchecked(arc) for arc in arcs]

  def _horizon(self):
    t = np.linspace(0, 2 * np.pi, 361)
    u, w = self.pC.east.data, self.pC.north.data
    return vector3d(np.cos(t)[:, None] * u + np.sin(t)[:, None] * w, isNormalized=True)

  def gridLines(self, step=30 * degree):
    """the meridians and small circles of the convention at a spacing, clipped to the region:
    plane curves with NaN where they leave it"""
    step = np.pi / max(1, int(round(np.pi / step)))
    curves = []
    rhos = np.linspace(0, 2 * np.pi, 361)
    for theta in np.arange(step, np.pi - step / 2, step):
      curves.append(self.project(self.pC.rot * vector3d.byPolar(theta, rhos)))
    thetas = np.linspace(-np.pi, np.pi, 361)
    for rho in np.arange(0, np.pi - step / 2, step):
      curves.append(self.project(self.pC.rot * vector3d.byPolar(thetas, rho)))
    return curves

  def bounds(self, margin=0.02):
    """the bounding box of the drawn region, from its boundary, as (xmin, xmax, ymin, ymax)"""
    b = self.boundary(2 * degree)
    xs = np.concatenate([x for x, _ in b])
    ys = np.concatenate([y for _, y in b])
    return _withMargin(xs, ys, margin)

  def labels(self):
    """the labels: the Miller indices at the corners of the sector of a crystal frame, or at
    its axes when it has no sector; the axes names of any other frame where they lie in
    the chart; a list of (direction, text)"""
    fr = self.frame
    if isinstance(fr, crystalFrame):
      V = self.sR.vertices()
      # a symmetric function over the whole sphere is labelled at the equivalent corners of
      # the sector, anything else at the crystal axes, the positive end where it lies inside
      if V.size == 0 and self.symmetric:
        V = _equivalentVertices(fr, self.sR)
      if V.size == 0:
        V = [next(s * a for s in (1, -1) if self.sR.checkInside(s * a)) for a in (fr.aAxis, fr.bAxis, fr.cAxis)
             if self.sR.checkInside(a) or self.sR.checkInside(-a)]
        V = vector3d(np.array([v.data.reshape(3) for v in V]).reshape(-1, 3)).normalize() if V else vector3d()
      return [(v, s) for v in V.flatten() for s in (millerLabel(v, fr),) if s is not None]
    # a frame-free direction is named by the axes of the session's specimen frame
    from ..geometry.frames import specimenFrame
    names = tuple((specimenFrame.default if fr is None else fr).axesNames)
    labels = []
    # an antipodal chart shows an axis of the other hemisphere at its antipode; the axis
    # at the pole of the chart is left unlabelled, since a label there would hide the data
    pole = self.pC.outOfScreen.data.reshape(3)
    for v, name in zip((xvector, yvector, zvector), names):
      if abs(v.data.reshape(3) @ pole) > 1 - 1e-9 and not self.isPlain:
        continue
      if self.sR.checkInside(v):
        labels.append((vector3d(v), name))
      elif self.antipodal and self.sR.checkInside(-v):
        labels.append((-vector3d(v), name))
    return labels

  def __repr__(self):
    hemi = "upper" if self.upper else "lower"
    return (aligned(f"{type(self).__name__} ({hemi}{', antipodal' if self.antipodal else ''})\n"
            f"  convention: {self.pC.char()}\n  region: {len(self.sR.N)} faces"))


class eareaProjection(sphericalProjection):
  """the equal area, Schmidt, projection: r = 2 sin(theta / 2)"""
  name = "earea"


class eangleProjection(sphericalProjection):
  """the equal angle, stereographic, projection: r = tan(theta / 2)"""
  name = "eangle"

  @staticmethod
  def radius(theta):
    """the plane radius tan(theta / 2) of the polar angle theta"""
    return np.tan(theta / 2)

  @staticmethod
  def polarAngle(r):
    """the polar angle 2 arctan(r) drawn at the plane radius r"""
    return 2 * np.arctan(r)


class edistProjection(sphericalProjection):
  """the equidistant projection: r = theta"""
  name = "edist"

  @staticmethod
  def radius(theta):
    """the plane radius of the polar angle theta, theta itself"""
    return np.asarray(theta, dtype=float)

  @staticmethod
  def polarAngle(r):
    """the polar angle drawn at the plane radius r, r itself"""
    return np.asarray(r, dtype=float)


class orthographicProjection(sphericalProjection):
  """the orthographic projection: r = sin(theta)"""
  name = "orthographic"

  @staticmethod
  def radius(theta):
    """the plane radius sin(theta) of the polar angle theta"""
    return np.sin(theta)

  @staticmethod
  def polarAngle(r):
    """the polar angle arcsin(r) drawn at the plane radius r"""
    return np.arcsin(np.clip(r, 0, 1))


class gnomonicProjection(sphericalProjection):
  """the gnomonic projection: r = tan(theta), great circles drawn as straight lines"""
  name = "gnomonic"

  @staticmethod
  def radius(theta):
    """the plane radius tan(theta) of the polar angle theta"""
    return np.tan(theta)

  @staticmethod
  def polarAngle(r):
    """the polar angle arctan(r) drawn at the plane radius r"""
    return np.arctan(r)

  @classmethod
  def extent(cls):
    """the radius of the polar angle 80 degrees, the extent of a gnomonic chart"""
    return cls.radius(80 * degree)


class plainProjection(sphericalProjection):
  """rho over theta in degrees: the polar angles of the frame itself, no convention"""
  name = "plain"

  @property
  def isPlain(self):
    """whether the chart draws the polar angles as plain coordinates, True here"""
    return True

  def polarIn(self, v):
    """the polar coordinates of directions in their own frame, NaN outside the region"""
    v = asVector3d(v).normalize()
    inside = self.sR.checkInside(v)
    nan = np.where(inside, 0.0, np.nan)
    return v.theta + nan, v.rho + nan

  def project(self, v):
    """the plane coordinates (rho, theta) of directions in degrees, NaN outside the region"""
    theta, rho = self.polarIn(v)
    return np.mod(rho, 2 * np.pi) / degree, theta / degree

  def projectUnchecked(self, v):
    """the plane coordinates (rho, theta) in degrees of directions known to lie in the chart"""
    v = asVector3d(v)
    return np.mod(v.rho, 2 * np.pi) / degree, v.theta / degree

  def iproject(self, x, y):
    """the direction of the azimuth x and the polar angle y, both in degrees"""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return vector3d.byPolar(y * degree, x * degree)

  def withRegion(self, sR):
    """the same chart showing another region"""
    return type(self)(sR, self.pC, self.antipodal, self.upper)

  def chartRegion(self):
    """the region shown, without its frame"""
    return sphericalRegion(self.sR.N, self.sR.alpha)

  def _gridFrom(self, theta, rho, blank):
    V = vector3d(vector3d.byPolar(theta, rho).data, frame=self.frame, isNormalized=True)
    X, Y = rho / degree, theta / degree
    X, Y = X.copy(), Y.copy()
    X[:, blank], Y[:, blank] = np.nan, np.nan
    return V, X, Y

  def boundary(self, resolution=0.5 * degree):
    """the boundary of the region as plane coordinates, one pair of arrays per arc"""
    return [self.projectUnchecked(arc) for arc in self.sR.boundary(resolution)]

  def bounds(self, margin=0.0):
    """the azimuth and polar ranges of the region in degrees, widened by the margin"""
    sR = self.chartRegion()
    rhoLo, rhoHi = rhoRange(sR)
    ivs = [iv for iv in (sR.thetaInterval(rho) for rho in np.linspace(rhoLo, rhoHi, 91)) if iv is not None]
    ys = np.array([min(lo for lo, _ in ivs), max(hi for _, hi in ivs)]) if ivs else np.array([0.0, np.pi])
    return _withMargin(np.array([rhoLo, rhoHi]) / degree, ys / degree, margin)


PROJECTIONS = {"earea": eareaProjection, "schmidt": eareaProjection, "eangle": eangleProjection, "stereo": eangleProjection,
               "stereographic": eangleProjection, "edist": edistProjection, "orthographic": orthographicProjection,
               "gnomonic": gnomonicProjection, "plain": plainProjection}


def projectionClass(name):
  """the projection class of a name ('earea', 'schmidt', 'eangle', 'stereo', 'edist',
  'orthographic', 'gnomonic', 'plain'), of a projection or of a class"""
  if isinstance(name, type) and issubclass(name, sphericalProjection):
    return name
  if isinstance(name, sphericalProjection):
    return type(name)
  key = str(name).lower()
  if key not in PROJECTIONS:
    raise ValueError(f"unknown projection '{name}'; one of {', '.join(sorted(PROJECTIONS))}")
  return PROJECTIONS[key]


# ------------------------------------------------------------------ which charts a plot needs

def withinHemisphere(sR, up, tol=1e-6):
  """does the region lie within the hemisphere towards up, the boundary allowed on the equator"""
  u = asVector3d(up)
  if sR.checkInside(-u, tol=-tol) or len(sR.N) == 0:
    return False
  return all(np.all(arc.data @ u.data.reshape(3) >= -tol) for arc in sR.boundary(2 * degree))


def isUpper(sR, pC):
  """whether the region lies within the hemisphere out of the screen of the convention pC"""
  return withinHemisphere(sR, pC.outOfScreen)


def isLower(sR, pC):
  """whether the region lies within the hemisphere into the screen of the convention pC"""
  return withinHemisphere(sR, pC.intoScreen)


def makeSphericalProjection(v=None, projection="earea", upper=False, lower=False, complete=False, antipodal=None,
                            how2plot=None, fundamentalRegion=False, symmetric=None):
  """the charts a region is drawn on: one, or an upper and a lower one when it reaches into
  both hemispheres; an antipodal region needs the upper one only; the plain projection
  shows everything in one chart (MATLAB makeSphericalProjection)"""
  sR, how2plot, ax = _regionFor(v, antipodal, how2plot, fundamentalRegion)
  antipodal = ax if antipodal is None else bool(antipodal)
  cls = projectionClass(projection)
  pC = _conventionOf(sR.frame) if how2plot is None else plottingConvention(how2plot)
  # complete drops the symmetry sector; with upper or lower one hemisphere of it is drawn,
  # with both of them, as MATLAB reads 'upper', 'lower', the two
  if upper and lower:
    upper, lower, complete = False, False, True
  # a function of a frame with symmetry carries that symmetry, a set of directions not; a
  # function drawn over the whole sphere passes its region and says so
  if symmetric is None:
    symmetric = v is not None and not isinstance(v, (vector3d, sphericalRegion)) and hasSymmetry(sR.frame)
  charts = _charts(cls, sR, pC, antipodal, upper, lower, complete)
  for c in charts:
    c.symmetric = symmetric
  return charts


def _charts(cls, sR, pC, antipodal, upper, lower, complete):
  if cls is plainProjection:
    return [cls(sR, pC, antipodal, True)]
  # an antipodal chart shows a region of the far hemisphere through its antipode
  if antipodal and not upper and not lower and not complete and isLower(sR, pC):
    sR = sphericalRegion(-sR.N, sR.alpha, sR.frame, sR.antipodal)
  upperOnly = upper or (not complete and not lower and (isUpper(sR, pC) or antipodal))
  lowerOnly = lower or (not complete and not upperOnly and isLower(sR, pC))
  if upperOnly:
    return [cls(sR.restrictToUpper(pC), pC, antipodal, True)]
  if lowerOnly:
    return [cls(sR.restrictToLower(pC.outOfScreen), pC, antipodal, False)]
  return [cls(sR.restrictToUpper(pC), pC, antipodal, True), cls(sR.restrictToLower(pC.outOfScreen), pC, antipodal, False)]


# the region the data of a plot covers, and whether the data are axes: the whole sphere of
# the frame, or the sector of the frame's group when asked for
def _regionFor(v, antipodal, how2plot, fundamentalRegion):
  if isinstance(v, sphericalRegion):
    return v, how2plot, bool(v.antipodal)
  if v is None:
    return sphericalRegion(frame=None), how2plot, False
  if isinstance(v, vector3d):
    fr, ax = v.frame, v.antipodal
  elif isinstance(v, referenceFrame):
    fr, ax, fundamentalRegion = v, False, True
  else:
    fr, ax, fundamentalRegion = getattr(v, "frame", None), bool(getattr(v, "antipodal", False)), hasSymmetry(getattr(v, "frame", None)) if not fundamentalRegion else True
  ax = ax if antipodal is None else bool(antipodal)
  return plotRegion(fr, fundamentalRegion, ax, how2plot), how2plot, ax


def hasSymmetry(fr):
  """whether a frame carries a group that folds the sphere: the specimen sector of a pole
  figure is as real as the crystal sector of an inverse one"""
  return fr is not None and fr.numSym() > 1


def plotRegion(fr, fundamentalRegion=False, antipodal=False, how2plot=None):
  """the whole sphere, or the conventional sector of the frame's group, that of the Laue group
  for axes (MATLAB plot 'fundamentalRegion')"""
  if fundamentalRegion and hasSymmetry(fr):
    return plotSector(fr, antipodal, how2plot)
  return sphericalRegion(frame=fr)


def plotSector(fr, antipodal=False, how2plot=None):
  """the sector of the frame's group an inverse pole figure shows, that of the Laue group for axes,
  turned as MATLAB draws it for 321"""
  pC = _conventionOf(fr) if how2plot is None else plottingConvention(how2plot)
  pg = fr.sym
  if antipodal and not pg.isLaue:
    pg = pg.Laue()
  sR = conventionalSector(pg, up=pC.outOfScreen, east=pC.east, frame=fr)
  # MATLAB draws the wedge of 321 from an m direction, half way between the two-folds the
  # conventional sector starts at
  if pg.pointGroup == "321":
    turn = rotation.byAxisAngle(vector3d(pC.outOfScreen.data), -np.pi / 6)
    sR = sphericalRegion((turn * vector3d(sR.N)).data.reshape(-1, 3), sR.alpha, frame=fr)
  return sR


def rhoRange(sR):
  """the azimuth interval of a region: the whole circle when a pole lies inside, otherwise
  the complement of the widest gap between the azimuths of its boundary"""
  if len(sR.N) == 0:
    return 0.0, 2 * np.pi
  if sR.checkInside(zvector, tol=-1e-8) or sR.checkInside(-zvector, tol=-1e-8):
    return 0.0, 2 * np.pi
  pts = [arc.data for arc in sR.boundary(0.5 * degree)]
  if not pts:
    return 0.0, 2 * np.pi
  xyz = np.concatenate(pts)
  xyz = xyz[np.abs(xyz[:, 2]) < 1 - 1e-8]
  if len(xyz) == 0:
    return 0.0, 2 * np.pi
  rhos = np.sort(np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2 * np.pi))
  gaps = np.append(np.diff(rhos), rhos[0] + 2 * np.pi - rhos[-1])
  k = int(np.argmax(gaps))
  if gaps[k] < 2 * degree:
    return 0.0, 2 * np.pi
  return (rhos[0], rhos[-1]) if k == len(rhos) - 1 else (rhos[k + 1] - 2 * np.pi, rhos[k])


# ------------------------------------------------------------------ labels

def millerLabel(v, fr, utf8=None):
  """a direction of a crystal frame as its rounded indices, the direct basis first and the
  reciprocal one if that does not round, in the brackets of one direction; None if neither"""
  utf8 = getMTEXpref("UTF8Output") if utf8 is None else utf8
  u = v.data.reshape(3) / np.linalg.norm(v.data.reshape(3))
  hex_ = fr.lattice.isTriHex
  for style in (("UVTW" if hex_ else "uvw"), ("hkil" if hex_ else "hkl")):
    m = vector3d(u, frame=fr, dispStyle=style).round(style)
    w = m.data.reshape(3)
    if np.linalg.norm(w) > 0 and np.arccos(np.clip(w @ u / np.linalg.norm(w), -1, 1)) < 1e-4:
      left, right = brackets(style, False)
      return left + indexString(getattr(m, style).reshape(-1), utf8) + right
  return None


# a region without vertices is labelled by the vertices of the fundamental sector in all
# their equivalent positions inside it, those at a pole or on the equator, as MATLAB does
def _equivalentVertices(fr, sR):
  V = fr.fundamentalSector().vertices()
  if V.size == 0:
    return vector3d()
  X = np.einsum("nij,mj->nmi", fr.rot.matrix(), V.data.reshape(-1, 3)).reshape(-1, 3)
  X = X / np.linalg.norm(X, axis=-1, keepdims=True)
  X = X[np.unique(np.round(X, 6), axis=0, return_index=True)[1]]
  X = X[sR.checkInside(X)]
  theta = np.arccos(np.clip(X[:, 2], -1, 1))
  X = X[np.min(np.abs(theta[:, None] - np.array([0, np.pi / 2, np.pi])), axis=1) < degree]
  return vector3d(X, isNormalized=True)


# ------------------------------------------------------------------ helpers

def _conventionOf(fr):
  pC = None if fr is None else fr.how2plot
  return plottingConvention(pC) if pC is not None else plottingConvention(plottingConvention.default)


def _emptyGrid(frame):
  return vector3d(np.zeros((0, 0, 3)), frame=frame), np.zeros((0, 0)), np.zeros((0, 0))


def _withMargin(xs, ys, margin):
  xlo, xhi, ylo, yhi = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
  d = margin * min(xhi - xlo, yhi - ylo)
  return xlo - d, xhi + d, ylo - d, yhi + d


def plotS2Grid(*args, resolution=1 * degree, upper=False, lower=False, complete=False, antipodal=None):
  """plotS2Grid(resolution=0.5 * degree, 'upper', cs): the directions a chart samples its surface
  at, a (theta, rho) array in the frame given, so that values computed on it are drawn as they
  lie, contourf(rGrid, values), without interpolation; MATLAB's plotS2Grid"""
  from ..geometry.frames import referenceFrame
  refuseFlags(args, "plotS2Grid")
  sR = next((a for a in args if isinstance(a, sphericalRegion)), None)
  fr = next((a for a in args if isinstance(a, referenceFrame)), None)
  sR = sR if sR is not None else sphericalRegion(frame=fr)
  charts = makeSphericalProjection(sR, upper=upper, lower=lower, complete=complete, antipodal=antipodal)
  grids = [c.grid(resolution)[0].data for c in charts]
  # the two hemispheres side by side, a blank column between them
  data = grids[0] if len(grids) == 1 else np.concatenate([grids[0], np.full((grids[0].shape[0], 1, 3), np.nan), grids[1]], axis=1)
  V = vector3d(data, frame=sR.frame if fr is None else fr)
  V._plotGrid = dict(upper=upper or (len(charts) == 1 and charts[0].upper), lower=lower or (len(charts) == 1 and not charts[0].upper),
                     antipodal=charts[0].antipodal)
  return V


__all__ = ["sphericalProjection", "eareaProjection", "eangleProjection", "edistProjection", "orthographicProjection",
           "gnomonicProjection", "plainProjection", "makeSphericalProjection", "plotRegion", "plotSector", "rhoRange",
           "isUpper", "isLower", "millerLabel", "plotS2Grid"]
