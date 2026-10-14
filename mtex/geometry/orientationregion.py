# fundamental regions of orientation space as Dirichlet domains (MATLAB @orientationRegion,
# @referenceFrame/fundamentalRegion)
#
# The fundamental region of orientation space is the same idea in the quaternion sphere:
# the domain of the identity under the two-sided action, bounded by the half-spaces
# dot(q, N) <= 0 with N the quaternion of half the minimal rotation about every symmetry
# axis. When the two groups share rotations those fix the identity by conjugation, which
# turns the axis and keeps the angle, so the angle condition alone holds every conjugate and
# an axis sector picks one.
from __future__ import annotations

from typing import TYPE_CHECKING
import itertools

import numpy as np

from ._array import MtexArray
from .kernels import regionMembership

from .quaternion import qmul, rotation
from .representations import quatAngle, quatAxis
from .sphericalregion import fundamentalSector, sphericalRegion
from .symmetry import symmetry
from .symreduce import _isAboutZ, reductionGroups, uniqueRotations
from .vector3d import vector3d
from .display import display, frameLabel, header


def commonRotations(L, R):
  """the rotations both groups share"""
  return L[np.any(np.abs(L @ R.T) > 1 - 1e-8, axis=1)]


# the sectors built so far, by the frames and the exchange; every projection asks again
_axisSectors = {}


def commonAxisSector(o):
  """the sector of the axes of orientations between two frames, its centre and the rotations
  that turn an axis into an equivalent one: the fundamental sector of the frame the axes are
  expressed in (axisFrame), which holds the elements both frames share, and their rotation
  parts, since an improper element on both sides conjugates like its proper part; with grain
  exchange the Laue group, a misorientation then being its inverse, the same angle about the
  opposite axis. It is the sector a plot of the axes shows, so the region, the projection and
  the plot never disagree; None where nothing is shared"""
  from .symmetryops import axisFrame
  key = (o.frameA, o.frameB, bool(o.antipodal))
  if key in _axisSectors:
    return _axisSectors[key]
  frame = axisFrame(o)
  shared = frame.sym.properGroup().rotations().data.reshape(-1, 4)
  # the axes are signed: the sector holds the axis, not the axis or its opposite
  sector = sphericalRegion(fundamentalSector(frame))
  sector.antipodal = False
  found = (sector, sector.center().data.reshape(3), shared) if len(sector.N) else (None, np.array([0.0, 0.0, 1.0]), shared)
  # a frame made for one call is no key to keep
  if o.frameA._sealed and o.frameB._sealed:
    _axisSectors[key] = found
  return found


class orientationRegion:
  """a region of orientation space bounded by planes in the quaternions, mostly the fundamental
  region of a pair of frames

    oR = fundamentalRegion(cs)                         # the fundamental region of a crystal frame
    oR = fundamentalRegion(cs, cs, antipodal=True)     # of misorientations, with grain exchange
    oR = orientationRegion(N, cs, specimenFrame())     # the region bounded by the quaternions N

  The region holds the quaternions q with dot(q, N) <= 0 for every bounding quaternion N; q
  and -q are the same rotation.

  The fundamental region of two frames, which fundamentalRegion builds, is the set of
  rotations closer to the identity than to any of their symmetric equivalents;
  projectIntoFundamentalRegion maps orientations into it. Where the two frames share
  rotations, about a common axis for instance, the planes alone do not separate the
  equivalents, and the region also restricts the rotation axes to a sector.

  Input
    N - quaternion, or an array (n, 4), the normals of the bounding planes
    frameA - the first frame, the crystal frame
    frameB - the second frame, the specimen or a second crystal frame
  Output
    oR - orientationRegion, with oR.CS and oR.SS its two frames, MATLAB's CS1 and CS2
  Keywords
    antipodal - identify a misorientation with its inverse, the grain exchange symmetry
    common - the rotations, as an array (n, 4), the two frames share
    sector - sphericalRegion, the sector of rotation axes where the frames share rotations
    axisCenter - the centre of that sector, an array of three
  See also
    fundamentalRegion, orientation.projectIntoFundamentalRegion, orientationRegion.checkInside,
    orientationRegion.plot, fundamentalSector"""

  def __init__(self, N, frameA, frameB, antipodal=False, common=None, sector=None, axisCenter=None):
    self.N = np.asarray(N, dtype=float).reshape(-1, 4)
    self.frameA, self.frameB, self.antipodal = frameA, frameB, antipodal
    self.common = np.zeros((0, 4)) if common is None else common
    self.sector = sector
    self.axisCenter = np.array([0.0, 0.0, 1.0]) if axisCenter is None else axisCenter

  @property
  def CS(self):
    """the first frame of the region, frameA: the crystal frame (MATLAB's orientationRegion.CS1)"""
    return self.frameA

  @property
  def SS(self):
    """the second frame of the region, frameB: the specimen or second crystal frame
    (MATLAB's orientationRegion.CS2)"""
    return self.frameB

  def checkInside(self, o, tol=2e-3):
    """whether orientations lie inside the region, up to the tolerance, as a boolean array

      inside = oR.checkInside(ori)                  # the orientations inside the region
      inside = oR.checkInside(ori, tol=1*degree)    # a wider margin at the boundary

    A point counts as inside when it violates no bound of the region by more than half the
    tolerance, an angle; the sector of axes, where the region has one, is checked with the
    same tolerance.

    Input
      oR - orientationRegion
      o - orientation, rotation or quaternions as an (..., 4) array
    Output
      inside - boolean array in the shape of o
    Keywords
      tol - the tolerance at the boundary of the region, MATLAB's 'tolerance'
    See also
      orientationRegion, fundamentalRegion, projectIntoFundamentalRegion"""
    q = o.data if isinstance(o, MtexArray) else np.asarray(o, dtype=float)
    t = tol / 2
    inside = regionMembership(q, self.N, 1, t, 0.0)[0].reshape(q.shape[:-1])
    if self.sector is not None:
      inside &= (quatAngle(q) < 1e-8) | self.sector.checkInside(quatAxis(q), tol)
    return inside

  def axisSector(self, omega, sector="own"):
    """the sector of rotation axes the region allows at a fixed rotation angle: a bound with
    N = (-sin(w_i/4), cos(w_i/4) a_i) reads for q = (cos(w/2), sin(w/2) v) as
    dot(v, a_i) <= cot(w/2) tan(w_i/4), a small circle about the axis

      sR = oR.axisSector(40*degree)                 # the axes of a 40 degree rotation in the region
      sR = oR.axisSector(40*degree, sector=None)    # without the region's own sector of axes

    The bounds whose small circle covers every axis at this angle drop out; the remaining
    circles are joined with a sector of axes, by default the region's own, the sector its
    symmetry leaves for the axes.

    Input
      oR - orientationRegion
      omega - the rotation angle
    Output
      sR - sphericalRegion of the axes, in the first frame of the region
    Keywords
      sector - the sphericalRegion the axes are restricted to as well, 'own' the region's own
          sector, None no restriction
    See also
      orientationRegion.maxAngle, sphericalRegion"""
    sector = self.sector if sector == "own" else sector
    t = np.inf if omega <= 0 else 1 / np.tan(omega / 2)
    v = self.N[:, 1:]
    nv = np.linalg.norm(v, axis=1)
    c = t * np.tan(np.arctan2(-self.N[:, 0], nv))
    # a bound whose circle covers every axis at this angle drops out
    keep = ~(c >= 1)
    N, alpha = -v[keep] / nv[keep, None], -c[keep]
    if sector is not None:
      N, alpha = np.concatenate([N, sector.N]), np.concatenate([alpha, sector.alpha])
    return sphericalRegion(N.reshape(-1, 3), alpha, frame=self.frameA)

  def maxAngle(self, v=None, hits=2000):
    """the largest rotation angle inside the region, at a vertex of it in Rodrigues space, or
    about a given axis

      omega = oR.maxAngle()                         # the largest angle in the region
      omega = oR.maxAngle(vector3d(1, 1, 1))        # the largest angle about an axis

    Without an axis the angle is found exactly at the vertices of the region in Rodrigues
    space; only a region whose sector of axes has caps is sampled, by random rotations until
    hits of them fall inside. About an axis it is the least angle at which a bound is met, 0
    for an axis outside the region's sector of axes, where MATLAB first projects the axis into
    the sector.

    Input
      oR - orientationRegion
      v - vector3d, a single rotation axis
    Output
      omega - the angle in radians
    Keywords
      hits - the number of random rotations inside the region the sampling waits for
    See also
      maxAngle, orientationRegion.axisSector"""
    if v is not None:
      d = v.normalize().data if isinstance(v, vector3d) else np.asarray(v, dtype=float) / np.linalg.norm(v)
      if self.sector is not None and not self.sector.checkInside(d):
        return 0.0
      # every bound the axis points towards limits the angle about it
      a = self.N[:, 1:]
      s = np.linalg.norm(a, axis=1)
      c = a @ d.reshape(3) / s
      bound = c > 1e-9
      return float(np.min(2 * np.arctan(-self.N[bound, 0] / s[bound] / c[bound]), initial=np.pi))
    if len(self.N) == 0:
      return float(np.pi)
    exact = self._maxAngleVertices()
    if exact is not None:
      return exact
    rng = np.random.default_rng(0)
    best, n, found = 0.0, 0, 0
    while found < hits and n < 100 * hits:
      q = rotation.rand(hits, rng=rng).data
      n += hits
      d = q @ self.N.T
      inside = np.all(d <= 0, axis=1) | np.all(d >= 0, axis=1)
      found += int(np.sum(inside))
      if np.any(inside):
        best = max(best, float(np.max(quatAngle(q[inside]))))
    return best

  def rodriguesPlanes(self):
    """the region in Rodrigues space as A rho <= b: a bound dot(q, N) <= 0 reads n . rho <= -N_0
    and a sector plane through the origin -m . rho <= 0; None when the sector has caps"""
    A = [n[1:] for n in self.N]
    b = [-n[0] for n in self.N]
    if self.sector is not None:
      if np.any(np.abs(self.sector.alpha) > 1e-12):
        return None
      A += [-m for m in self.sector.N]
      b += [0.0] * len(self.sector.N)
    return np.array(A).reshape(-1, 3), np.array(b)

  def edges(self, points=40, tol=1e-9):
    """the edges of the region, each a chain of points in axis angle coordinates in radians:
    every pair of planes meets in a line of Rodrigues space, kept where the other planes
    allow it; an open region is cut at a turn of nearly 180 degrees

      E = oR.edges()                                # 40 points along every edge
      E = oR.edges(points=5)                        # coarser chains

    Input
      oR - orientationRegion
    Output
      E - list of (points, 3) arrays, the axis times the angle; empty when the sector of axes
          has caps
    Keywords
      points - the number of points on each edge
      tol - the slack by which a parallel plane may miss the edge before it removes it
    See also
      plot, orientationRegion.maxAngle"""
    planes = self.rodriguesPlanes()
    if planes is None:
      return []
    A, b = planes
    far = np.tan(0.4999 * np.pi)
    out = []
    for i, j in itertools.combinations(range(len(A)), 2):
      d = np.cross(A[i], A[j])
      if np.linalg.norm(d) < 1e-10:
        continue
      d /= np.linalg.norm(d)
      p0 = np.linalg.lstsq(A[[i, j]], b[[i, j]], rcond=None)[0]
      lo, hi = -far, far
      for k in range(len(A)):
        if k in (i, j):
          continue
        a, r = A[k] @ d, b[k] - A[k] @ p0
        if abs(a) < 1e-12:
          if r < -tol:
            lo, hi = 1.0, 0.0
            break
          continue
        if a > 0:
          hi = min(hi, r / a)
        else:
          lo = max(lo, r / a)
      if hi - lo < 1e-9:
        continue
      rho = p0 + np.linspace(lo, hi, points)[:, None] * d
      n = np.linalg.norm(rho, axis=1, keepdims=True)
      out.append(np.where(n > 1e-12, 2 * np.arctan(n) * rho / np.maximum(n, 1e-12), 0.0))
    return out

  # in Rodrigues space every bound is a plane, dot(q, N) <= 0 reading n . rho <= -N_0 and a
  # sector through the origin -m . rho <= 0, so the region is a polyhedron and its largest
  # angle is at a vertex: every triple of planes solved, the points inside kept; a half turn
  # when the polyhedron is open, None when the sector has caps that are not planes
  def _maxAngleVertices(self, tol=1e-9):
    planes = self.rodriguesPlanes()
    if planes is None:
      return None
    A, b = planes
    # the region is open, holding the half turns of some axis, when a direction u has
    # A u <= 0; such cones are spanned by the edges of pairs of planes, or by what no plane
    # sees when they span less than space
    i, j = np.triu_indices(len(A), 1)
    _, s, Vt = np.linalg.svd(A)
    cand = np.concatenate([np.cross(A[i], A[j]), Vt[np.sum(s > 1e-10):]])
    nu = np.linalg.norm(cand, axis=1)
    Au = cand @ A.T
    tolU = 1e-9 * nu[:, None]
    if np.any((nu > 1e-10) & (np.all(Au <= tolU, axis=1) | np.all(-Au <= tolU, axis=1))):
      return float(np.pi)
    # every triple of planes at once, the regular ones solved and the vertices inside kept
    T = np.array(list(itertools.combinations(range(len(A)), 3)), dtype=np.intp).reshape(-1, 3)
    T = T[np.abs(np.linalg.det(A[T])) >= 1e-10]
    if not len(T):
      return None
    rho = np.linalg.solve(A[T], b[T][..., None])[..., 0]
    inside = np.all(rho @ A.T <= b + tol * (1 + np.abs(b)), axis=1)
    best = float(np.max(np.linalg.norm(rho[inside], axis=1))) if np.any(inside) else None
    return None if best is None else float(2 * np.arctan(best))

  def __repr__(self):
    return display(header("orientationRegion", f"{frameLabel(self.frameA)} → {frameLabel(self.frameB)}"),
                   [("faces", len(self.N)), ("antipodal", "true" if self.antipodal else None),
                    ("max angle", f"{self.maxAngle() * 180 / np.pi:.4g}°")])

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..plotting.spaceplot import plotRegion3d as plot, plotRegion3d as plot3d


def fundamentalRegion(frA, frB=None, *, antipodal=False):
  """the fundamental region of a pair of groups: the Dirichlet domain of the identity under
  l * q * r, |<q, 1>| >= |<q, s>| for every product s = l r. For s of angle w about the
  axis a this is the pair of half-spaces (q_v . a) cos(w/4) <= q_0 sin(w/4) and the same
  for (-a, 2pi - w), each dot(q, N) <= 0 with N = (-sin(w/4), a cos(w/4)); per signed axis
  the smallest angle binds"""
  from .frames import crystalFrame, referenceFrame, specimenFrame
  from .orientation import orientation
  if isinstance(frA, orientation):
    o = frA
    frA, frB, antipodal = o.frameA, o.frameB, o.antipodal
  if isinstance(frA, symmetry):
    frA = crystalFrame(frA.rot, frA.axes, intern=False)
  if frB is None:
    frB = specimenFrame.default
  if isinstance(frB, symmetry):
    frB = crystalFrame(frB.rot, frB.axes, intern=False)
  o = orientation(rotation(), frA, frB)
  L, R = reductionGroups(o)
  ids = uniqueRotations(qmul(L[:, None, :], R[None, :, :]).reshape(-1, 4))
  ids = ids[quatAngle(ids) > 1e-3]
  axes, angles = [], []
  for q in ids:
    for d, w in ((quatAxis(q), quatAngle(q)), (-quatAxis(q), 2 * np.pi - quatAngle(q))):
      hit = [k for k, x in enumerate(axes) if x @ d > 1 - 1e-10]
      if hit:
        angles[hit[0]] = min(angles[hit[0]], w)
      else:
        axes.append(d)
        angles.append(w)
  N = np.array([[-np.sin(w / 4), *(np.cos(w / 4) * d)] for d, w in zip(axes, angles)]).reshape(-1, 4)
  common = commonRotations(L, R)
  o.antipodal = antipodal
  sector, center, _ = commonAxisSector(o)
  return orientationRegion(N, frA, frB, antipodal, common, sector, center)


def fundamentalRegionEuler(frA, frB=None):
  """the box in Euler angles that carries every orientation at least once
  (fundamentalRegionEuler): the two-folds perpendicular to z halve Phi, and one of the
  reflections they make on phi1 or phi2 halves that angle when it folds its turn about zero
  fundamentalRegionEuler(cs, ss): MATLAB's name of the Euler plotting box

    maxPhi1, maxPhi, maxPhi2 = fundamentalRegionEuler(cs)       # with the default specimen frame
    maxPhi1, maxPhi, maxPhi2 = fundamentalRegionEuler(cs, ss)   # with a specimen symmetry

  The same as eulerRegion. MATLAB's flag 'complete', the full box, is not taken.

  Input
    frA - crystalFrame
    frB - the second frame, by default the default specimen frame
  Output
    maxPhi1, maxPhi, maxPhi2 - the upper bounds of the three Bunge Euler angles in radians
  See also
    eulerRegion, fundamentalRegion, fundamentalSector"""
  return eulerRegion(frA, frB)


def projectIntoEulerRegion(o):
  """the Bunge Euler angles of the representative of every orientation inside the Euler box
  of its frames, as MATLAB's project2EulerFR: three arrays in the shape of the list"""
  from .representations import quat2euler
  bounds = eulerRegion(o.CS, o.SS)
  # the representative given comes first, so that it stays when it is inside, and the first
  # and third angles are folded by the rotations about z, as MATLAB does it
  es = o.flatten().symmetrise()
  q = np.concatenate([o.flatten().data.reshape(1, -1, 4), es.data.reshape(es.shape[0], -1, 4)])
  a, b, c = quat2euler(q, "Bunge")
  fold1, fold2, _, _ = _zFolds(o.CS, o.SS)
  a, c = np.mod(a, fold1), np.mod(c, fold2)
  inside = (a <= bounds[0] + 1e-8) & (b <= bounds[1] + 1e-8) & (c <= bounds[2] + 1e-8)
  if not np.all(np.any(inside, axis=0)):
    raise ValueError("no representative fits the Euler box for these frames")
  idx = (np.argmax(inside, axis=0), np.arange(a.shape[1]))
  return a[idx].reshape(o.shape), b[idx].reshape(o.shape), c[idx].reshape(o.shape)


# the periods of the first and third Euler angle under the rotations about z of the two groups
def _zFolds(frA, frB):
  from .orientation import orientation
  L, R = reductionGroups(orientation(rotation(), frA, frB))
  return 2 * np.pi / np.sum(_isAboutZ(L)), 2 * np.pi / np.sum(_isAboutZ(R)), L, R


def eulerRegion(frA, frB=None):
  """the ranges (phi1, Phi, phi2) of the Bunge Euler angles covering the fundamental region of
  two frames, the specimen frame the default second

    maxPhi1, maxPhi, maxPhi2 = eulerRegion(cs)     # with the default specimen frame
    maxPhi1, maxPhi, maxPhi2 = eulerRegion(cs, ss) # with a specimen symmetry, or a second crystal

  The rotations about z of the two groups divide phi1 and phi2, the two-folds perpendicular
  to z halve Phi, and one of the reflections they make on phi1 or phi2 halves that angle.

  Input
    frA - crystalFrame
    frB - the second frame, by default the default specimen frame
  Output
    maxPhi1, maxPhi, maxPhi2 - the upper bounds of the three Bunge Euler angles in radians
  See also
    fundamentalRegionEuler, fundamentalRegion, projectIntoEulerRegion"""
  from .frames import specimenFrame
  frB = specimenFrame.default if frB is None else frB
  maxPhi1, maxPhi2, L, R = _zFolds(frA, frB)

  def perp(q):
    sel = np.abs(quatAngle(q) - np.pi) < 1e-8
    a = quatAxis(q[sel])
    a = a[np.abs(a[:, 2]) < 1e-8]
    return np.mod(np.arctan2(a[:, 1], a[:, 0]), np.pi)

  psi1, psi2 = perp(R), perp(L)
  if len(psi1) == 0 and len(psi2) == 0:
    return (maxPhi1, float(np.pi), maxPhi2)
  def folds(ps, Lp):
    return np.any(np.abs(np.remainder(2 * ps + Lp / 2, Lp) - Lp / 2) < 1e-8)
  if len(psi1) and len(psi2):
    if folds(psi2, maxPhi1):
      maxPhi1 /= 2
    elif folds(psi1, maxPhi2):
      maxPhi2 /= 2
  return (maxPhi1, np.pi / 2, maxPhi2)


def maxAngle(frA, frB=None, antipodal=False):
  """maxAngle(cs), maxAngle(cs1, cs2): the largest rotation angle of the fundamental region of
  the frames, MATLAB's maxAngle of a symmetry

    omega = maxAngle(cs)                          # the largest angle of an orientation
    omega = maxAngle(cs, cs)                      # the largest misorientation angle
    omega = maxAngle(cs, cs, antipodal=True)      # with the misorientations' inverses identified

  MATLAB's maxAngle(cs, axis), the largest angle about an axis, is
  fundamentalRegion(cs).maxAngle(axis) here.

  Input
    frA - crystalFrame
    frB - the second frame, by default the default specimen frame
  Output
    omega - the angle in radians
  Keywords
    antipodal - identify every misorientation with its inverse
  See also
    orientationRegion.maxAngle, fundamentalRegion"""
  return fundamentalRegion(frA, frB, antipodal=antipodal).maxAngle()


__all__ = ["orientationRegion", "fundamentalRegion", "eulerRegion", "fundamentalRegionEuler", "projectIntoEulerRegion", "maxAngle"]
