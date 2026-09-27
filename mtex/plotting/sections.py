# ODF sections: the slicing of orientation space, without graphics (MATLAB
# plotting/ODFSections; docs/plans/plotting.md S2)
#
# Orientation space is three dimensional, so a section plot is a change of coordinates
# SO(3) -> S2 x R: one scalar picks the slice, the two degrees of freedom left are a
# direction on the sphere. Three such changes cover MTEX's section classes:
#
#   the Euler sections   fix the first or the last angle of an Euler convention; the middle
#                        angle is the polar angle of the picture, the other outer one its
#                        azimuth (phi1Sections, phi2Sections, alphaSections, gammaSections)
#   the pole figure      the direction is ori * h1, the coordinate the angle by which
#   sections             ori * h2 is twisted about it against a reference field
#                        (pfSections, sigmaSections)
#   axisAngleSections    the direction is the rotation axis of the representative in the
#                        fundamental region, the coordinate its rotation angle
#
# Every section is drawn on the charts of spherical.py, so the drawing layer draws
# sections with the code it draws pole figures with, and nothing here knows about figures.
from __future__ import annotations

import numpy as np

from ..geometry.frames import crystalFrame, specimenFrame
from ..geometry.grids.s2grid import equispacedS2Grid
from ..geometry.kernels import qrot
from ..geometry.orientation import orientation
from ..geometry.orientationregion import eulerRegion, fundamentalRegion
from ..geometry.plottingconvention import plottingConvention
from ..geometry.quaternion import rotation
from ..geometry.representations import EULER_LABELS, eulerConvention, quat2euler, quatAngle, quatAxis
from ..geometry.session import getMTEXpref
from ..geometry.sphericalregion import conventionalSector, projectIntoRegion, sphericalRegion
from ..geometry.symmetry import symmetry
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, xvector, zvector
from .spherical import makeSphericalProjection
from ..geometry.display import aligned

GREEK = {"phi1": "φ1", "Phi": "Φ", "phi2": "φ2", "alpha": "α", "beta": "β", "gamma": "γ", "sigma": "σ", "omega": "ω"}


def _name(key):
  return GREEK.get(key, key) if getMTEXpref("UTF8Output") else key


def sectionIndex(x, values, tol, period=np.inf):
  """the section a coordinate falls in, or -1 when it falls beside every one: the distance to
  a section value taken over the period of the coordinate"""
  x = np.asarray(x, dtype=float)
  d = x[..., None] - np.asarray(values, dtype=float)
  if np.isfinite(period):
    d = np.mod(d + period / 2, period) - period / 2
  d = np.abs(d)
  k = np.argmin(d, axis=-1)
  return np.where(np.take_along_axis(d, k[..., None], axis=-1)[..., 0] <= tol, k, -1)


def sectionTolerance(values, tol):
  """the tolerance a section reaches: never more than half the gap to its neighbour"""
  values = np.sort(np.asarray(values, dtype=float))
  return float(tol) if len(values) < 2 else min(float(tol), float(np.min(np.diff(values))) / 2.1)


def sectionList(values, period, n):
  """the section values, either given or a turn of the period cut into equal parts"""
  return np.linspace(0, period, n + 1)[:-1] if values is None else np.asarray(values, dtype=float).reshape(-1)


# the orientation every symmetric equivalent came from: symmetrise shapes them (m, n)
def _equivalents(ori):
  es = ori.flatten().symmetrise()
  n = ori.size
  return es.data.reshape(-1, 4), np.arange(es.size) % n


class ODFSections:
  """sections through orientation space

    oS.CS1, oS.CS2       the two frames
    oS.tol               the tolerance an orientation may deviate from a section
    oS.numSections       how many
    oS.sectionCharts(k)  the charts section k is drawn on
    oS.sectionData(ori)  where the equivalents of a list fall: directions, section, source
    oS.orientationsAt(k, v)   the orientations at points of a chart of section k
    oS.plot(x, ...)      plotSection(x, oS, ...)
  """

  axisLabels = None
  outerSector = None

  def __init__(self, CS1, CS2, values, tol):
    self.CS1, self.CS2 = CS1, CS2
    self.values = np.asarray(values, dtype=float).reshape(-1)
    self.tol = sectionTolerance(self.values, tol)

  @property
  def CS(self):
    """the first frame of the sections, CS1"""
    return self.CS1

  @property
  def SS(self):
    """the second frame of the sections, CS2"""
    return self.CS2

  @property
  def numSections(self):
    """the number of sections"""
    return len(self.values)

  def sectionScale(self, k):
    """the scale of the axes of section k, one unless the section shrinks with its value"""
    return 1.0

  def sectionField(self, k, resolution=15 * degree):
    """the directions and vectors of a field drawn over section k, None for sections without one"""
    return None

  def makeGrid(self, resolution=2.5 * degree):
    """ori = oS.makeGrid(): the orientations of the nodes of every section chart, section by
    section, at which values are computed for oS.plot(values), MATLAB's makeGrid"""
    oris, counts = [], []
    for k in range(self.numSections):
      for proj in self.sectionCharts(k):
        V = proj.grid(resolution)[0]
        counts.append(V.size)
        if V.size:
          oris.append(self.orientationsAt(k, V.flatten()))
    self.plotGrid = (resolution, counts)
    return orientation.cat(*oris)

  def plot(self, x, *args, **kwargs):
    """oS.plot(odf), oS.plot(ori), and oS.plot(values, 'smooth') of values on makeGrid's nodes"""
    from .sectionplot import _plotSectionFun, plotSection
    if isinstance(x, np.ndarray) or isinstance(x, list):
      if getattr(self, "plotGrid", None) is None:
        raise ValueError("values are drawn on the nodes of makeGrid: call oS.makeGrid() first")
      resolution, counts = self.plotGrid
      values = iter(np.split(np.asarray(x, dtype=float).reshape(-1), np.cumsum([c for c in counts if c])[:-1]))
      kwargs.update({a: True for a in args if isinstance(a, str)})
      return _plotSectionFun(lambda ori: next(values), self, resolution=resolution, **kwargs)
    return plotSection(x, self, *args, **kwargs)

  def __repr__(self):
    vals = ", ".join(f"{v / degree:g}" for v in self.values)
    return aligned(f"{type(self).__name__} ({self.CS1.char()} → {self.CS2.char()})\n  sections: {vals}°")


# ------------------------------------------------------------------ Euler angle sections

def eulerBox(maxTheta, maxRho):
  """the box as a spherical region: a cap theta <= maxTheta cut by the two meridians of the
  azimuth"""
  N, alpha = [], []
  if maxTheta < np.pi - 1e-9:
    N.append((0.0, 0.0, 1.0))
    alpha.append(np.cos(maxTheta))
  if maxRho < 2 * np.pi - 1e-9:
    N += [(0.0, 1.0, 0.0), (np.sin(maxRho), -np.cos(maxRho), 0.0)]
    alpha += [0.0, 0.0]
  return sphericalRegion(np.array(N).reshape(-1, 3), np.array(alpha))


class eulerSections(ODFSections):
  """sections of constant Euler angle, the first or the third of a convention

    phi2Sections(cs, ss, sections=6), phi2Sections(cs, ss, phi2=(0:15:90)*degree)
    phi1Sections, alphaSections, gammaSections
  """

  convention = "Bunge"
  first = False

  def __init__(self, CS1, CS2=None, sections=6, values=None, tolerance=5 * degree, maxPhi=None, maxAzimuth=None, **kwargs):
    CS2 = specimenFrame.default if CS2 is None else CS2
    values = kwargs.pop(self.angleName, values)
    if kwargs:
      raise TypeError(f"unknown option {next(iter(kwargs))}")
    p1, pPhi, p2 = eulerRegion(CS1, CS2)
    self.period, azimuth = (p1, p2) if self.first else (p2, p1)
    self._tolerance = tolerance
    super().__init__(CS1, CS2, sectionList(values, self.period, sections), tolerance)
    self.maxPhi = pPhi if maxPhi is None else maxPhi
    self.maxAzimuth = azimuth if maxAzimuth is None else maxAzimuth
    self.sR = eulerBox(self.maxPhi, self.maxAzimuth)

  # the section angles set afterwards, sP.phi1 = (0:10:90)*degree
  def _setValues(self, values):
    self.values = np.asarray(values, dtype=float).reshape(-1)
    self.tol = sectionTolerance(self.values, self._tolerance)

  @property
  def names(self):
    """the names of the three Euler angles of the convention of the sections"""
    return EULER_LABELS[eulerConvention(self.convention)]

  @property
  def angleName(self):
    """the name of the Euler angle that is constant on each section"""
    return self.names[0 if self.first else 2]

  @property
  def axisLabels(self):
    """the labels of the two chart axes, the other outer Euler angle and the middle one"""
    return (_name(self.names[2 if self.first else 0]), _name(self.names[1]))

  def sectionLabel(self, k):
    """the title of section k, the constant angle in degrees"""
    return f"{_name(self.angleName)} = {round(self.values[k] / degree)}°"

  def sectionCharts(self, k):
    """the chart section k is drawn on, the Euler angle box as a plain projection"""
    return makeSphericalProjection(self.sR, projection="plain", how2plot=plottingConvention.default)

  # the angles of the equivalents ordered as the picture reads them: fixed, polar, azimuth
  def _triples(self, q):
    a, b, c = quat2euler(q, self.convention)
    return (a, b, c) if self.first else (c, b, a)

  def sectionData(self, ori):
    """where the symmetric equivalents of orientations fall: the chart directions, the section of
    each and the index of the orientation it comes from; a degenerate one falls into every section"""
    q, src = _equivalents(ori)
    fixed, theta, azim = self._triples(q)
    degenerate = (theta <= 1e-5) | (theta >= np.pi - 1e-5)
    k = sectionIndex(fixed, self.values, self.tol, self.period)
    keep = ~degenerate & (k >= 0)
    th, az, sec, frm = [theta[keep]], [np.mod(azim[keep], 2 * np.pi)], [k[keep]], [src[keep]]
    # where the middle angle is 0 or pi only the sum or the difference of the outer two is
    # determined, so the orientation belongs to every section, at the azimuth that leaves
    for i in np.flatnonzero(degenerate):
      for j, s in enumerate(self.values):
        a = azim[i] + fixed[i] - s if theta[i] <= 1e-5 else azim[i] - fixed[i] + s
        th.append([np.clip(theta[i], 1e-5, np.pi - 1e-5)])
        az.append([np.mod(a, 2 * np.pi)])
        sec.append([j])
        frm.append([src[i]])
    th, az = np.concatenate(th), np.concatenate(az)
    return vector3d.byPolar(th, az), np.concatenate(sec).astype(int), np.concatenate(frm).astype(int)

  def orientationsAt(self, k, vs):
    """the orientations at points vs of the chart of section k"""
    s = np.full(vs.shape, self.values[k])
    theta, azim = vs.theta, vs.rho
    a, b, c = (s, theta, azim) if self.first else (azim, theta, s)
    return orientation.byEuler(a, b, c, self.convention, self.CS1, self.CS2)


class phi2Sections(eulerSections):
  """sections of constant phi2 in the Bunge convention, MTEX's default ODF sections

    oS = phi2Sections(cs, ss)
    oS = phi2Sections(cs, ss, sections=6)
    oS = phi2Sections(cs, ss, phi2=np.arange(0, 91, 15)*degree)"""
  convention, first = "Bunge", False

  @property
  def phi2(self):
    """the phi2 values of the sections"""
    return self.values

  @phi2.setter
  def phi2(self, values):
    """sets the phi2 values of the sections"""
    self._setValues(values)


class phi1Sections(eulerSections):
  """sections of constant phi1 in the Bunge convention

    oS = phi1Sections(cs, ss, sections=6)
    oS = phi1Sections(cs, ss, phi1=np.arange(0, 91, 15)*degree)"""
  convention, first = "Bunge", True

  @property
  def phi1(self):
    """the phi1 values of the sections"""
    return self.values

  @phi1.setter
  def phi1(self, values):
    """sets the phi1 values of the sections"""
    self._setValues(values)


class gammaSections(eulerSections):
  """sections of constant gamma in the Matthies (ZYZ) convention

    oS = gammaSections(cs, ss, sections=6)
    oS = gammaSections(cs, ss, gamma=np.arange(0, 91, 15)*degree)"""
  convention, first = "Matthies", False

  @property
  def gamma(self):
    """the gamma values of the sections"""
    return self.values

  @gamma.setter
  def gamma(self, values):
    """sets the gamma values of the sections"""
    self._setValues(values)


class alphaSections(eulerSections):
  """sections of constant alpha in the Matthies (ZYZ) convention

    oS = alphaSections(cs, ss, sections=6)
    oS = alphaSections(cs, ss, alpha=np.arange(0, 91, 15)*degree)"""
  convention, first = "Matthies", True

  @property
  def alpha(self):
    """the alpha values of the sections"""
    return self.values

  @alpha.setter
  def alpha(self, values):
    """sets the alpha values of the sections"""
    self._setValues(values)


# ------------------------------------------------------------------ pole figure sections
#
# A section is the pole figure of h1 split by the angle the tangential part of ori * h2
# makes with a reference field. The field is the smallest rotation taking the pole of the
# plotting convention to the direction, applied to its east, so that its one singularity
# sits where the viewer does not look.

class pfSections(ODFSections):
  """pole figure sections: the pole figure of h1, split by the twist of h2 about it

    pfSections(cs, ss, h1=..., h2=..., sections=6, omega=values)
    sigmaSections(cs, ss, sections=6, sigma=values)   with h1 = c*, h2 = a
  """

  angleName = "omega"

  def __init__(self, CS1, CS2=None, h1=None, h2=None, sections=6, values=None, tolerance=5 * degree, maxOmega=None, **kwargs):
    CS2 = specimenFrame.default if CS2 is None else CS2
    values = kwargs.pop(self.angleName, values)
    if kwargs:
      raise TypeError(f"unknown option {next(iter(kwargs))}")
    crystal = isinstance(CS1, crystalFrame)
    self._CS1, self._maxOmega, self._sections, self._tolerance = CS1, maxOmega, sections, tolerance
    self._d1 = (CS1.cAxisRec if crystal else zvector) if h1 is None else h1
    self._d2 = (CS1.aAxis if crystal else xvector) if h2 is None else h2
    self._values = values
    self._setAxes()
    super().__init__(CS1, CS2, self.values, self.tol)
    self.pC = plottingConvention(CS2.how2plot) if CS2.how2plot is not None else plottingConvention.default
    self.sR = conventionalSector(CS2)

  # h1 and h2 as unit vectors, the period of the twist from the fold of h1, and the section
  # values cut from it unless given; only the part of h2 perpendicular to h1 enters, since
  # the angle is measured in the tangent plane, which makes the way back from a point of
  # the chart exact for a crystal whose axes are not perpendicular
  def _setAxes(self):
    d1 = np.asarray(self._d1.data if isinstance(self._d1, vector3d) else self._d1, dtype=float).reshape(3)
    d2 = np.asarray(self._d2.data if isinstance(self._d2, vector3d) else self._d2, dtype=float).reshape(3)
    d1 = d1 / np.linalg.norm(d1)
    t = d2 - (d2 @ d1) * d1
    if np.linalg.norm(t) < 1e-10:
      raise ValueError("h1 and h2 must not be parallel")
    self._h1 = vector3d(d1, frame=self._CS1, isNormalized=True)
    self._h2 = vector3d(t / np.linalg.norm(t), frame=self._CS1, isNormalized=True)
    self.period = 2 * np.pi / int(np.asarray(self._CS1.sym.nfold(vector3d(d1)))) if self._maxOmega is None else float(self._maxOmega)
    self.values = sectionList(self._values, self.period, self._sections)
    self.tol = sectionTolerance(self.values, self._tolerance)

  @property
  def h1(self):
    """the crystal direction whose pole figure every section shows, a unit vector"""
    return self._h1

  @h1.setter
  def h1(self, h):
    """sets the crystal direction whose pole figure the sections show"""
    self._d1 = h
    self._setAxes()

  @property
  def h2(self):
    """the part of the second crystal direction perpendicular to h1, whose twist about h1 splits
    the sections"""
    return self._h2

  @h2.setter
  def h2(self, h):
    """sets the second crystal direction, whose twist about h1 splits the sections"""
    self._d2 = h
    self._setAxes()

  @property
  def omega(self):
    """the twist angles of the sections"""
    return self.values

  @omega.setter
  def omega(self, values):
    """sets the twist angles of the sections"""
    self._values = values
    self._setAxes()

  def sectionLabel(self, k):
    """the title of section k, the twist angle in degrees"""
    return f"{_name(self.angleName)} = {round(self.values[k] / degree)}°"

  def referenceField(self, r):
    """the field the twist is measured against, at directions of the specimen frame"""
    r = r if isinstance(r, vector3d) else vector3d(r)
    xyz = r.normalize().data.reshape(-1, 3)
    up, east = self.pC.outOfScreen.data, self.pC.east.data
    q = rotation.map(np.tile(up, (len(xyz), 1)), xyz).data
    return vector3d(qrot(q, np.tile(east, (len(xyz), 1))).reshape(r.shape + (3,)), isNormalized=True)

  def twisted(self, r, omega):
    """the field turned about the direction by the angle of the section, where ori * h2 has to
    point for the orientation to belong there"""
    r = r if isinstance(r, vector3d) else vector3d(r)
    q = rotation.byAxisAngle(vector3d(r.normalize().data), omega).data
    return vector3d(qrot(q, self.referenceField(r).data), isNormalized=True)

  def sectionCharts(self, k):
    """the chart section k is drawn on, the conventional sector of the specimen frame, the upper
    hemisphere when h1 has a symmetric equivalent opposite it"""
    h = self.h1.data
    equivalentPoles = bool(np.any(qrot(self.CS1.sym.rot.flatten().data[:, :4], h[None]) @ h < -1 + 1e-10))
    return makeSphericalProjection(self.sR, how2plot=self.pC, upper=equivalentPoles)

  def sectionData(self, ori, withRotation=False):
    """where the symmetric equivalents of orientations fall: the direction of ori * h1, the section
    of the twist of ori * h2 and the index of the orientation; withRotation=True adds the rotations"""
    q, src = _equivalents(ori)
    v = vector3d(qrot(q, np.tile(self.h1.data, (len(q), 1))), isNormalized=True)
    w = vector3d(qrot(q, np.tile(self.h2.data, (len(q), 1))), isNormalized=True)
    omega = self.referenceField(v).angle(w, N=v)
    k = sectionIndex(omega, self.values, self.tol, self.period)
    keep = k >= 0
    out = (vector3d(v.data[keep], frame=self.CS2, isNormalized=True), k[keep], src[keep])
    return out + (q[keep],) if withRotation else out

  def sectionField(self, k, resolution=15 * degree):
    """the reference field turned into the section, at an equispaced grid of its region"""
    g = equispacedS2Grid(resolution=resolution)
    vs = vector3d(g.data[self.sR.checkInside(g)], isNormalized=True)
    return vs, self.twisted(vs, self.values[k])

  def orientationsAt(self, k, vs):
    """the orientations at points vs of the chart of section k: h1 on the point, h2 on the twisted
    reference field there"""
    vs = vector3d(vs.normalize().data)
    n = vs.size
    tw = self.twisted(vs, self.values[k])
    q = rotation.map(np.tile(self.h1.data, (n, 1)), vs.data.reshape(n, 3),
                     np.tile(self.h2.data, (n, 1)), tw.data.reshape(n, 3)).data
    return orientation._fromRotation(q.reshape(vs.shape + (4,)), (self.CS1, self.CS2))


class sigmaSections(pfSections):
  """pole figure sections of the c axis split by the twist sigma of the a axis about it, the
  sections MATLAB's sigma sections draw

    oS = sigmaSections(cs, ss, sections=6)
    oS = sigmaSections(cs, ss, sigma=np.arange(0, 60, 10)*degree)"""
  angleName = "sigma"

  @property
  def sigma(self):
    """the sigma values of the sections"""
    return self.values

  @sigma.setter
  def sigma(self, values):
    """sets the sigma values of the sections"""
    self.omega = values


# ------------------------------------------------------------------ axis angle sections

class axisAngleSections(ODFSections):
  """sections of constant rotation angle, each on the axis sector the region allows there

    axisAngleSections(cs1, cs2, angles=(10:20:180)*degree)
    axisAngleSections(cs1, cs2, sections=5, volumeScaling=False, antipodal=True)
  """

  def __init__(self, CS1, CS2=None, sections=None, angles=None, values=None, tolerance=5 * degree, antipodal=False,
               volumeScaling=True, axisAngle=None):
    CS2 = specimenFrame.default if CS2 is None else CS2
    angles = axisAngle if axisAngle is not None else values if values is not None else angles
    self.oR = fundamentalRegion(CS1, CS2, antipodal=antipodal)
    maxAngle = self.oR.maxAngle()
    if sections is not None:
      edges = np.linspace(0, maxAngle, sections + 1)
      vals = (edges[:-1] + edges[1:]) / 2
    else:
      vals = np.arange(5, 181, 10) * degree if angles is None else np.asarray(angles, dtype=float).reshape(-1)
      vals = vals[vals <= maxAngle]
    if len(vals) == 0:
      raise ValueError(f"no section angle below the largest angle of the region, {maxAngle / degree:.1f}°")
    super().__init__(CS1, CS2, vals, tolerance)
    # the axis is only defined modulo the rotations both groups share; the sections are drawn
    # on the conventional sector of those, in the convention of the frame the rotation maps
    # into, so that the sections sit the same way up as the pole figures
    self.pC = plottingConvention(CS2.how2plot) if CS2.how2plot is not None else plottingConvention.default
    # with grain exchange an axis and its opposite are one, the inverse misorientation
    common = self.oR.common
    if len(common) > 1 or antipodal:
      self.jointCS = symmetry(rotation._fromData(common if len(common) else np.array([[1.0, 0, 0, 0]])))
      if antipodal:
        self.jointCS = self.jointCS.Laue()
      self.jointSector = conventionalSector(self.jointCS, up=self.pC.outOfScreen, east=self.pC.east, frame=CS1)
    else:
      self.jointCS = None
      self.jointSector = sphericalRegion(frame=CS1)
    self.axesSectors = [self.oR.axisSector(w, sector=self.jointSector) for w in self.values]
    self.volumeScaling = bool(volumeScaling)

  @property
  def angles(self):
    """the rotation angles of the sections"""
    return self.values

  @property
  def outerSector(self):
    """the sector the axes are drawn on, the conventional sector of the rotations both frames share"""
    return self.jointSector

  def sectionLabel(self, k):
    """the title of section k, the rotation angle in degrees"""
    # three decimals at most, as MATLAB's xnum2str writes the angle
    return f"{_name('omega')} = {round(float(self.values[k] / degree), 3):g}°"

  def sectionCharts(self, k):
    """the chart section k is drawn on, the axis sector the region allows at its angle"""
    return makeSphericalProjection(self.axesSectors[k], how2plot=self.pC)

  def sectionScale(self, k):
    """the sector shrinks with the angle, so the axes grow with 1/sin(w/2) to keep areas comparable"""
    return float(np.sin(np.max(self.values) / 2) / np.sin(self.values[k] / 2)) if self.volumeScaling else 1.0

  def sectionData(self, ori):
    """where orientations fall: the rotation axes of their projections into the fundamental region,
    the section of their angle and the index of the orientation"""
    p = ori.flatten().project2FundamentalRegion()
    q = p.data.reshape(-1, 4)
    k = sectionIndex(quatAngle(q), self.values, self.tol, np.inf)
    keep = k >= 0
    axes = vector3d(quatAxis(q[keep]), frame=self.CS1, isNormalized=True)
    if self.jointCS is not None and axes.size:
      axes = projectIntoRegion(axes, self.jointSector, self.jointCS)
    return axes, k[keep], np.flatnonzero(keep)

  def orientationsAt(self, k, vs):
    """the orientations at axes vs rotated by the angle of section k"""
    return orientation.byAxisAngle(vector3d(vs.normalize().data), self.values[k], self.CS1, self.CS2)


# ------------------------------------------------------------------ the chooser

SECTION_KINDS = {"phi2": phi2Sections, "phi1": phi1Sections, "gamma": gammaSections, "alpha": alphaSections,
                 "sigma": sigmaSections, "omega": sigmaSections, "pf": pfSections, "axisangle": axisAngleSections}
SECTION_OPTIONS = ("sections", "tolerance", "phi1", "phi2", "alpha", "gamma", "sigma", "omega", "angles", "axisAngle",
                   "volumeScaling", "maxPhi", "maxAzimuth", "maxOmega", "h1", "h2", "antipodal")


def newODFSectionPlot(CS1, CS2=None, kind=None, **kwargs):
  """the sections a plot uses: phi2 sections against a specimen, axis angle sections between
  two crystals, or the kind named (MATLAB newODFSectionPlot)"""
  CS2 = specimenFrame.default if CS2 is None else CS2
  if isinstance(kind, ODFSections):
    return kind
  if kind is None:
    kind = "phi2" if isinstance(CS1, specimenFrame) or isinstance(CS2, specimenFrame) else "axisAngle"
  cls = SECTION_KINDS.get(str(kind).lower())
  if cls is None:
    raise ValueError(f"unknown section type '{kind}': phi1, phi2, alpha, gamma, sigma, pf or axisAngle")
  opts = {k: v for k, v in kwargs.items() if k in SECTION_OPTIONS}
  if cls is not axisAngleSections:
    opts.pop("antipodal", None)
    opts.pop("volumeScaling", None)
  return cls(CS1, CS2, **opts)


__all__ = ["ODFSections", "eulerSections", "phi1Sections", "phi2Sections", "alphaSections", "gammaSections", "pfSections",
           "sigmaSections", "axisAngleSections", "newODFSectionPlot", "sectionIndex", "sectionTolerance"]
