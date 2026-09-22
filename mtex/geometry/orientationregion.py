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

import numpy as np

from ._array import MtexArray

from .quaternion import qmul, rotation
from .representations import quatAngle, quatAxis
from .sphericalregion import fundamentalSector, sectorCenter, sphericalRegion
from .symmetry import symmetry
from .symreduce import reductionGroups, uniqueRotations
from .vector3d import vector3d


# the rotations both groups share
def commonRotations(L, R):
  return L[np.any(np.abs(L @ R.T) > 1 - 1e-8, axis=1)]


# the sector of the axes: the Dirichlet sector of the shared rotations, and its centre
def commonAxisSector(common):
  if len(common) <= 1:
    return None, np.array([0.0, 0.0, 1.0])
  pg = symmetry(rotation._fromData(common))
  return fundamentalSector(pg), sectorCenter(pg).data


def commonAxisCenter(common):
  return commonAxisSector(common)[1]


class orientationRegion:

  def __init__(self, N, frameA, frameB, antipodal=False, common=None, sector=None, axisCenter=None):
    self.N = np.asarray(N, dtype=float).reshape(-1, 4)
    self.frameA, self.frameB, self.antipodal = frameA, frameB, antipodal
    self.common = np.zeros((0, 4)) if common is None else common
    self.sector = sector
    self.axisCenter = np.array([0.0, 0.0, 1.0]) if axisCenter is None else axisCenter

  @property
  def CS1(self):
    return self.frameA

  @property
  def CS2(self):
    return self.frameB

  def checkInside(self, o, tolerance=2e-3):
    q = o.data if isinstance(o, MtexArray) else np.asarray(o, dtype=float)
    if len(self.N) == 0:
      return np.ones(q.shape[:-1], dtype=bool)
    t = tolerance / 2
    d = q @ self.N.T
    inside = np.all(d <= t, axis=-1) | np.all(d >= -t, axis=-1)
    if self.sector is not None:
      ax = quatAxis(q)
      inside &= (quatAngle(q) < 1e-8) | self.sector.checkInside(ax, tolerance)
    return inside

  # the sector of rotation axes the region allows at a fixed rotation angle: a bound with
  # N = (-sin(w_i/4), cos(w_i/4) a_i) reads for q = (cos(w/2), sin(w/2) v) as
  # dot(v, a_i) <= cot(w/2) tan(w_i/4), a small circle about the axis
  def axisSector(self, omega, sector="own"):
    sector = self.sector if sector == "own" else sector
    N, alpha = [], []
    t = np.inf if omega <= 0 else 1 / np.tan(omega / 2)
    for n in self.N:
      v = n[1:]
      c = t * np.tan(np.arctan2(-n[0], np.linalg.norm(v)))
      if c >= 1:
        continue
      N.append(-v / np.linalg.norm(v))
      alpha.append(-min(c, 1.0))
    if sector is not None:
      N += list(sector.N)
      alpha += list(sector.alpha)
    return sphericalRegion(np.array(N).reshape(-1, 3), np.array(alpha), frame=self.frameA)

  # the largest rotation angle inside the region, by sampling, or about a given axis
  def maxAngle(self, v=None, hits=2000):
    if v is not None:
      d = v.normalize().data if isinstance(v, vector3d) else np.asarray(v, dtype=float) / np.linalg.norm(v)
      if self.sector is not None and not self.sector.checkInside(d):
        return 0.0
      omega = float(np.pi)
      for n in self.N:
        a = n[1:]
        s = np.linalg.norm(a)
        c = a @ d / s
        if c > 1e-9:
          omega = min(omega, 2 * np.arctan(-n[0] / s / c))
      return omega
    if len(self.N) == 0:
      return float(np.pi)
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

  def __repr__(self):
    return f"orientationRegion ({self.frameA.char()} → {self.frameB.char()})\n  faces: {len(self.N)}\n  max angle: {self.maxAngle() * 180 / np.pi:.4g}°"


# the fundamental region of a pair of groups: the Dirichlet domain of the identity under
# l * q * r, |<q, 1>| >= |<q, s>| for every product s = l r. For s of angle w about the
# axis a this is the pair of half-spaces (q_v . a) cos(w/4) <= q_0 sin(w/4) and the same
# for (-a, 2pi - w), each dot(q, N) <= 0 with N = (-sin(w/4), a cos(w/4)); per signed axis
# the smallest angle binds
def fundamentalRegion(frA, frB=None, antipodal=False):
  from .frames import crystalFrame, referenceFrame, specimenFrame
  from .orientation import orientation
  if isinstance(frA, orientation):
    o = frA
    frA, frB, antipodal = o.frameA, o.frameB, o.antipodal
  if isinstance(frA, symmetry):
    frA = crystalFrame(frA.rot, noIntern=True)
  if frB is None:
    frB = specimenFrame.default
  if isinstance(frB, symmetry):
    frB = crystalFrame(frB.rot, noIntern=True)
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
  sector, center = commonAxisSector(common)
  return orientationRegion(N, frA, frB, antipodal, common, sector, center)


# the box in Euler angles that carries every orientation at least once
# (fundamentalRegionEuler): the two-folds perpendicular to z halve Phi, and one of the
# reflections they make on phi1 or phi2 halves that angle when it folds its turn about zero
# fundamentalRegionEuler(cs, ss): MATLAB's name of the Euler plotting box
def fundamentalRegionEuler(frA, frB=None):
  return eulerRegion(frA, frB)


# the Bunge Euler angles of the representative of every orientation inside the Euler box
# of its frames, as MATLAB's project2EulerFR: three arrays in the shape of the list
def project2EulerFR(o):
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
  aboutZ = lambda q: np.sum((quatAngle(q) < 1e-8) | (np.abs(quatAxis(q)[:, 2]) > 1 - 1e-8))
  return 2 * np.pi / aboutZ(L), 2 * np.pi / aboutZ(R), L, R


def eulerRegion(frA, frB=None):
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
  folds = lambda ps, Lp: np.any(np.abs(np.remainder(2 * ps + Lp / 2, Lp) - Lp / 2) < 1e-8)
  if len(psi1) and len(psi2):
    if folds(psi2, maxPhi1):
      maxPhi1 /= 2
    elif folds(psi1, maxPhi2):
      maxPhi2 /= 2
  return (maxPhi1, np.pi / 2, maxPhi2)


__all__ = ["orientationRegion", "fundamentalRegion", "eulerRegion", "fundamentalRegionEuler", "project2EulerFR"]
