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
from ..geometry.vector3d import vector3d, xvector, zvector, asVector3d
from .spherical import _conventionOf, makeSphericalProjection
from ..geometry.display import aligned
from ..geometry.arguments import refuseFlags

GREEK = {"phi1": "φ1", "Phi": "Φ", "phi2": "φ2", "alpha": "α", "beta": "β", "gamma": "γ", "sigma": "σ", "omega": "ω"}


def _name(key):
  return GREEK.get(key, key) if getMTEXpref("UTF8Output") else key


def sectionIndex(x, values, tol, period=np.inf):
  """the section a coordinate falls in, or -1 when it falls beside every one: the distance to
  a section value taken over the period of the coordinate

    k = sectionIndex(x, values, tol)                      # the nearest section within tol, -1 beside every one
    k = sectionIndex(x, values, tol, period=360*degree)   # a coordinate that wraps, 359 degree next to 0

  Every coordinate is compared with every section value; the nearest one is taken when it
  lies within tol, else -1. With a finite period the distance is taken modulo the period,
  so the sections of an angle that turns are found across its seam.

  Input
    x - array of coordinates, the angles that pick the section
    values - the section values
    tol - the largest distance from a section value that still falls into it
  Output
    k - int array of the shape of x, the index of the section or -1
  Keywords
    period - the period of the coordinate, inf for one that does not wrap
  See also
    sectionTolerance, ODFSections, newODFSectionPlot"""
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

    oS.frameA, oS.frameB       the two frames
    oS.tol               the tolerance an orientation may deviate from a section
    oS.numSections       how many
    oS.sectionCharts(k)  the charts section k is drawn on
    oS.sectionData(ori)  where the equivalents of a list fall: directions, section, source
    oS.orientationsAt(k, v)   the orientations at points of a chart of section k
    oS.plot(x, ...)      plotSection(x, oS, ...)
  """

  axisLabels = None
  outerSector = None
  # the resolution and the node counts of the grid makeGrid made, None before it ran
  plotGrid = None

  def __init__(self, frameA, frameB, values, tol):
    self.frameA, self.frameB = frameA, frameB
    self.values = np.asarray(values, dtype=float).reshape(-1)
    self.tol = sectionTolerance(self.values, tol)

  @property
  def CS(self):
    """the first frame of the sections, frameA"""
    return self.frameA

  @property
  def SS(self):
    """the second frame of the sections, frameB"""
    return self.frameB

  @property
  def numSections(self):
    """the number of sections"""
    return len(self.values)

  def sectionScale(self, k):
    """the scale of the axes of section k, one unless the section shrinks with its value"""
    return 1.0

  def sectionField(self, k, resolution=15 * degree):
    """the directions and vectors of a field drawn over section k, None for sections without one

      oS.sectionField(k)   # None: Euler and axis angle sections carry no field

    Only the pole figure sections draw a field over their charts, pfSections.sectionField;
    every other kind of section returns None.

    Input
      k - int, the index of the section
    Output
      None
    Keywords
      resolution - the spacing of the grid the field would be given on
    See also
      pfSections.sectionField, quiverSectionField"""
    return None

  def makeGrid(self, resolution=2.5 * degree):
    """the orientations of the nodes of every section chart, section by section, at which values
    are computed for oS.plot(values); MATLAB's makeGrid

      ori = oS.makeGrid()                      # the nodes at 2.5 degree
      ori = oS.makeGrid(resolution=5*degree)   # a coarser grid
      oS.plot(odf.eval(ori))                   # values on the nodes drawn as sections

    The nodes are the grids of the section charts, chart by chart, turned into orientations;
    the sections remember the resolution and the node counts, so oS.plot(values) splits the
    values onto the charts again.

    Output
      ori - orientation, a list of the nodes of every chart in turn
    Keywords
      resolution - the spacing of the grid on every chart
    See also
      plotSection, sphericalProjection.grid"""
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
    if isinstance(x, (np.ndarray, list)):
      if self.plotGrid is None:
        raise ValueError("values are drawn on the nodes of makeGrid: call oS.makeGrid() first")
      resolution, counts = self.plotGrid
      values = iter(np.split(np.asarray(x, dtype=float).reshape(-1), np.cumsum([c for c in counts if c])[:-1]))
      refuseFlags(args, "plot")
      return _plotSectionFun(lambda ori: next(values), self, resolution=resolution, **kwargs)
    return plotSection(x, self, *args, **kwargs)

  def __repr__(self):
    vals = ", ".join(f"{v / degree:g}" for v in self.values)
    return aligned(f"{type(self).__name__} ({self.frameA.char()} → {self.frameB.char()})\n  sections: {vals}°")


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

  def __init__(self, frameA, frameB=None, sections=6, values=None, tol=5 * degree, maxPhi=None, maxAzimuth=None, **kwargs):
    frameB = specimenFrame.default if frameB is None else frameB
    values = kwargs.pop(self.angleName, values)
    if kwargs:
      raise TypeError(f"unknown option {next(iter(kwargs))}")
    p1, pPhi, p2 = eulerRegion(frameA, frameB)
    self.period, azimuth = (p1, p2) if self.first else (p2, p1)
    self._tolerance = tol
    super().__init__(frameA, frameB, sectionList(values, self.period, sections), tol)
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
    # where the middle angle is 0 or pi only the sum or the difference of the outer two is
    # determined, so the orientation belongs to every section, at the azimuth that leaves
    i, m = np.flatnonzero(degenerate), len(self.values)
    s = self.values[None, :]
    a = np.where((theta[i] <= 1e-5)[:, None], azim[i, None] + fixed[i, None] - s, azim[i, None] - fixed[i, None] + s)
    th = np.concatenate([theta[keep], np.repeat(np.clip(theta[i], 1e-5, np.pi - 1e-5), m)])
    az = np.concatenate([np.mod(azim[keep], 2 * np.pi), np.mod(a, 2 * np.pi).reshape(-1)])
    sec = np.concatenate([k[keep], np.tile(np.arange(m), len(i))]).astype(int)
    frm = np.concatenate([src[keep], np.repeat(src[i], m)]).astype(int)
    return vector3d.byPolar(th, az), sec, frm

  def orientationsAt(self, k, vs):
    """the orientations at points vs of the chart of section k"""
    s = np.full(vs.shape, self.values[k])
    theta, azim = vs.theta, vs.rho
    a, b, c = (s, theta, azim) if self.first else (azim, theta, s)
    return orientation.byEuler(a, b, c, self.convention, self.frameA, self.frameB)


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

  def __init__(self, frameA, frameB=None, h1=None, h2=None, sections=6, values=None, tol=5 * degree, maxOmega=None, **kwargs):
    frameB = specimenFrame.default if frameB is None else frameB
    values = kwargs.pop(self.angleName, values)
    if kwargs:
      raise TypeError(f"unknown option {next(iter(kwargs))}")
    crystal = isinstance(frameA, crystalFrame)
    self._CS1, self._maxOmega, self._sections, self._tolerance = frameA, maxOmega, sections, tol
    self._d1 = (frameA.cAxisRec if crystal else zvector) if h1 is None else h1
    self._d2 = (frameA.aAxis if crystal else xvector) if h2 is None else h2
    self._values = values
    self._setAxes()
    super().__init__(frameA, frameB, self.values, self.tol)
    self.pC = _conventionOf(frameB)
    self.sR = conventionalSector(frameB)

  # h1 and h2 as unit vectors, the period of the twist from the fold of h1, and the section
  # values cut from it unless given; only the part of h2 perpendicular to h1 enters, since
  # the angle is measured in the tangent plane, which makes the way back from a point of
  # the chart exact for a crystal whose axes are not perpendicular
  def _setAxes(self):
    d1, d2 = (asVector3d(d).data.reshape(3) for d in (self._d1, self._d2))
    d1 = d1 / np.linalg.norm(d1)
    t = d2 - (d2 @ d1) * d1
    if np.linalg.norm(t) < 1e-10:
      raise ValueError("h1 and h2 must not be parallel")
    self._h1 = vector3d(d1, frame=self._CS1, isNormalized=True)
    self._h2 = vector3d(t / np.linalg.norm(t), frame=self._CS1, isNormalized=True)
    if self._maxOmega is None:
      self.period = 2 * np.pi / int(np.asarray(self._CS1.sym.nfold(vector3d(d1))))
    else:
      self.period = float(self._maxOmega)
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
    r = asVector3d(r)
    xyz = r.normalize().data.reshape(-1, 3)
    up, east = self.pC.outOfScreen.data, self.pC.east.data
    q = rotation.map(np.tile(up, (len(xyz), 1)), xyz).data
    return vector3d(qrot(q, np.tile(east, (len(xyz), 1))).reshape(r.shape + (3,)), isNormalized=True)

  def twisted(self, r, omega):
    """the field turned about the direction by the angle of the section, where ori * h2 has to
    point for the orientation to belong there"""
    r = asVector3d(r)
    q = rotation.byAxisAngle(vector3d(r.normalize().data), omega).data
    return vector3d(qrot(q, self.referenceField(r).data), isNormalized=True)

  def sectionCharts(self, k):
    """the chart section k is drawn on, the conventional sector of the specimen frame, the upper
    hemisphere when h1 has a symmetric equivalent opposite it"""
    h = self.h1.data
    equivalentPoles = bool(np.any(qrot(self.frameA.sym.rot.flatten().data[:, :4], h[None]) @ h < -1 + 1e-10))
    return makeSphericalProjection(self.sR, how2plot=self.pC, upper=equivalentPoles)

  def sectionData(self, ori, withRotation=False):
    """where the symmetric equivalents of orientations fall: the direction of ori * h1, the section
    of the twist of ori * h2 and the index of the orientation; withRotation=True adds the rotations

      v, k, src = oS.sectionData(ori)                       # the points of every equivalent in its section
      v, k, src, q = oS.sectionData(ori, withRotation=True)  # with the quaternions of the equivalents

    Every orientation is symmetrised; an equivalent falls into the section whose twist angle
    of ori * h2 about ori * h1, measured against the reference field, lies within oS.tol of
    the section's angle, and is drawn at ori * h1. Equivalents beside every section are left
    out.

    Input
      ori - orientation between oS.frameA and oS.frameB
    Output
      v - vector3d of the specimen frame, the direction ori * h1 of every equivalent kept
      k - int array, the section of every equivalent kept
      src - int array, the index into ori.flatten() every equivalent came from
      q - (n, 4) array, the quaternions of the equivalents kept (withRotation=True only)
    Keywords
      withRotation - also return the quaternions of the equivalents
    See also
      pfSections, sigmaSections, plotSection"""
    q, src = _equivalents(ori)
    v = vector3d(qrot(q, np.tile(self.h1.data, (len(q), 1))), isNormalized=True)
    w = vector3d(qrot(q, np.tile(self.h2.data, (len(q), 1))), isNormalized=True)
    omega = self.referenceField(v).angle(w, N=v)
    k = sectionIndex(omega, self.values, self.tol, self.period)
    keep = k >= 0
    out = (vector3d(v.data[keep], frame=self.frameB, isNormalized=True), k[keep], src[keep])
    return out + (q[keep],) if withRotation else out

  def sectionField(self, k, resolution=15 * degree):
    """the reference field turned into the section, at an equispaced grid of its region

      v, d = oS.sectionField(0)                         # the field of the first section
      v, d = oS.sectionField(1, resolution=10*degree)   # on a coarser grid

    The field shows where ori * h2 points for an orientation to belong to section k: the
    reference field of the plotting convention turned about every direction by the section's
    twist angle. quiverSectionField draws it.

    Input
      k - int, the index of the section
    Output
      v - vector3d, the grid directions inside the region of the chart
      d - vector3d, the field at them
    Keywords
      resolution - the spacing of the equispaced grid
    See also
      sigmaSections, quiverSectionField"""
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
    return orientation._fromRotation(q.reshape(vs.shape + (4,)), (self.frameA, self.frameB))


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

  def __init__(self, frameA, frameB=None, sections=None, angles=None, values=None, tol=5 * degree, antipodal=False,
               volumeScaling=True, axisAngle=None):
    frameB = specimenFrame.default if frameB is None else frameB
    # the angles under any of their names, axisAngle first
    angles = next((a for a in (axisAngle, values) if a is not None), angles)
    self.oR = fundamentalRegion(frameA, frameB, antipodal=antipodal)
    maxAngle = self.oR.maxAngle()
    if sections is not None:
      edges = np.linspace(0, maxAngle, sections + 1)
      vals = (edges[:-1] + edges[1:]) / 2
    else:
      vals = np.arange(5, 181, 10) * degree if angles is None else np.asarray(angles, dtype=float).reshape(-1)
      vals = vals[vals <= maxAngle]
    if len(vals) == 0:
      raise ValueError(f"no section angle below the largest angle of the region, {maxAngle / degree:.1f}°")
    super().__init__(frameA, frameB, vals, tol)
    # the axis is only defined modulo the rotations both groups share; the sections are drawn
    # on the conventional sector of those, in the convention of the frame the rotation maps
    # into, so that the sections sit the same way up as the pole figures
    self.pC = _conventionOf(frameB)
    # with grain exchange an axis and its opposite are one, the inverse misorientation
    common = self.oR.common
    if len(common) > 1 or antipodal:
      self.jointCS = symmetry(rotation._fromData(common if len(common) else np.array([[1.0, 0, 0, 0]])))
      if antipodal:
        self.jointCS = self.jointCS.Laue()
      self.jointSector = conventionalSector(self.jointCS, up=self.pC.outOfScreen, east=self.pC.east, frame=frameA)
    else:
      self.jointCS = None
      self.jointSector = sphericalRegion(frame=frameA)
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
    p = ori.flatten().projectIntoFundamentalRegion()
    q = p.data.reshape(-1, 4)
    k = sectionIndex(quatAngle(q), self.values, self.tol, np.inf)
    keep = k >= 0
    axes = vector3d(quatAxis(q[keep]), frame=self.frameA, isNormalized=True)
    if self.jointCS is not None and axes.size:
      axes = projectIntoRegion(axes, self.jointSector, self.jointCS)
    return axes, k[keep], np.flatnonzero(keep)

  def orientationsAt(self, k, vs):
    """the orientations at axes vs rotated by the angle of section k"""
    return orientation.byAxisAngle(vector3d(vs.normalize().data), self.values[k], self.frameA, self.frameB)


# ------------------------------------------------------------------ the chooser

SECTION_KINDS = {"phi2": phi2Sections, "phi1": phi1Sections, "gamma": gammaSections, "alpha": alphaSections,
                 "sigma": sigmaSections, "omega": sigmaSections, "pf": pfSections, "axisangle": axisAngleSections}
SECTION_OPTIONS = ("sections", "tol", "phi1", "phi2", "alpha", "gamma", "sigma", "omega", "angles", "axisAngle",
                   "volumeScaling", "maxPhi", "maxAzimuth", "maxOmega", "h1", "h2", "antipodal")


def newODFSectionPlot(frameA, frameB=None, kind=None, **kwargs):
  """the sections a plot uses: phi2 sections against a specimen, axis angle sections between
  two crystals, or the kind named (MATLAB newODFSectionPlot)

    oS = newODFSectionPlot(cs, ss)                              # phi2 sections
    oS = newODFSectionPlot(cs, cs)                              # axis angle sections of misorientations
    oS = newODFSectionPlot(cs, ss, 'phi1', sections=4)          # four phi1 sections
    oS = newODFSectionPlot(cs, ss, 'sigma', sigma=[0, 30*degree])  # sigma sections at given angles
    oS = newODFSectionPlot(cs, ss, 'pf', h1=Miller(1,1,1,cs), h2=Miller(1,-1,0,cs), sections=3)  # pole figure sections

  The kind is one of 'phi1', 'phi2', 'alpha', 'gamma' (Euler sections), 'sigma' or 'omega'
  (sigmaSections), 'pf' (pfSections) and 'axisAngle' (axisAngleSections), in any case, or
  sections already made, which are returned as they are. The keywords the kind takes are
  passed on: sections, tol, the section angles by the name of the kind (phi2=..., sigma=...,
  angles=...), maxPhi, maxAzimuth, maxOmega, h1, h2, and antipodal and volumeScaling for axis
  angle sections; any other keyword is dropped, so the options of a plot can be passed whole.

  Input
    frameA - the first frame, the crystal frame
    frameB - the second frame, a specimen frame or a second crystal frame; by default the
        session's specimen frame
  Output
    oS - ODFSections, phi2Sections, sigmaSections, axisAngleSections, ...
  Keywords
    kind - the name of the section kind, or ODFSections; by default phi2 against a specimen
        and axisAngle between two crystals
  See also
    phi2Sections, sigmaSections, pfSections, axisAngleSections, plotSection"""
  frameB = specimenFrame.default if frameB is None else frameB
  if isinstance(kind, ODFSections):
    return kind
  if kind is None:
    kind = "phi2" if isinstance(frameA, specimenFrame) or isinstance(frameB, specimenFrame) else "axisAngle"
  cls = SECTION_KINDS.get(str(kind).lower())
  if cls is None:
    raise ValueError(f"unknown section type '{kind}': phi1, phi2, alpha, gamma, sigma, pf or axisAngle")
  opts = {k: v for k, v in kwargs.items() if k in SECTION_OPTIONS}
  if cls is not axisAngleSections:
    opts.pop("antipodal", None)
    opts.pop("volumeScaling", None)
  return cls(frameA, frameB, **opts)


__all__ = ["ODFSections", "eulerSections", "phi1Sections", "phi2Sections", "alphaSections", "gammaSections", "pfSections",
           "sigmaSections", "axisAngleSections", "newODFSectionPlot", "sectionIndex", "sectionTolerance"]
