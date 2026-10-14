# local grids (MATLAB localOrientationGrid, localS2Grid): a ball of orientations around a
# centre and a cap of directions around a direction, geometrically correct at the boundary
# of the fundamental region. Where the ball crosses that boundary, its far part is
# equivalent to a part inside, so a point is kept only when the centre itself is its
# closest symmetric copy of the centre; the copies of the kept set then reassemble the
# whole ball without repetition. When the centre has a stabiliser the ball is folded onto
# itself as well, and a second Dirichlet rule with a generic reference point cuts one wedge.
from __future__ import annotations

import numpy as np

from ..frames import referenceFrame
from ..orientation import orientation
from ..quaternion import qmul, qrot, rotation
from ..representations import axis2quat
from ..symreduce import _distinctUnitRows, chordAngle, reductionGroups, uniqueRotations
from ..units import degree
from ..vector3d import vector3d, zvector
from .s2grid import equispacedS2Grid


# the generic reference: the centre turned by a fixed small rotation
def _genericNear(q):
  return qmul(q, axis2quat(np.array([3.0, 2.0, 1.0]), 20 * degree))


def localOrientationGrid(*args, maxAngle=None, resolution=5 * degree, center=None, respectSymmetry=True):
  """an equispaced grid of orientations in a ball about a centre

    ori = localOrientationGrid(CS, SS, maxAngle=20*degree, center=ori0)   # the frames and the ball
    ori = localOrientationGrid(CS, SS, 20*degree)                        # about the identity
    ori = localOrientationGrid(ori0, 10*degree, resolution=2*degree)     # about an orientation
    ori = localOrientationGrid(ori0, 10*degree, respectSymmetry=False)   # every point of the ball

  The orientations lie on shells about the centre, resolution apart in angle, and on every
  shell the axes are spaced so that neighbouring orientations are resolution apart. With
  respectSymmetry the grid keeps one representative of every orientation the ball holds up to
  the symmetries of the frames, those nearer the centre than to any of its symmetric copies,
  so that a ball larger than the fundamental region holds no orientation twice.

  Input
    CS - crystalFrame
    SS - specimenFrame, by default the default specimen frame
    ori0 - orientation, the centre, whose frames are taken
  Output
    ori - orientation
  Keywords
    maxAngle - the radius of the ball, also given positionally
    resolution - the spacing of the grid
    center - orientation, the centre of the ball, by default the identity
    respectSymmetry - keep one representative of the orientations symmetrically equivalent
  See also
    equispacedSO3Grid, localS2Grid, regularSO3Grid"""
  from ..frames import specimenFrame
  args = list(args)
  if args and isinstance(args[0], orientation):
    center = args.pop(0)
  frames = [a for a in args if isinstance(a, referenceFrame)]
  nums = [a for a in args if not isinstance(a, referenceFrame)]
  if nums:
    maxAngle = nums[0]
  if maxAngle is None:
    raise ValueError("localOrientationGrid needs a maxAngle")
  if center is None:
    CS = frames[0]
    SS = frames[1] if len(frames) > 1 else specimenFrame.default
    center = orientation(rotation.id, CS, SS)
  res = float(resolution)
  rots = [np.array([[1.0, 0, 0, 0]])]
  for w in np.arange(res / 2, maxAngle + 1e-12, res):
    # the axis spacing delta giving an orientation spacing res on the shell
    delta = np.arccos(np.clip((np.cos(res / 2) - np.cos(w / 2) ** 2) / np.sin(w / 2) ** 2, -1, 1))
    ax = equispacedS2Grid(resolution=delta)
    rots.append(axis2quat(ax.data, np.full(ax.size, w)))
  q = qmul(np.concatenate(rots), center.data[None])
  ori = orientation(rotation._fromData(q), center.frameA, center.frameB)
  if not respectSymmetry:
    return ori
  L, R = reductionGroups(ori)
  if len(L) * len(R) == 1:
    return ori
  c = center.data
  allCopies = qmul(qmul(L[:, None, :], c[None, None, :]), R[None, :, :]).reshape(-1, 4)
  copies = uniqueRotations(allCopies, 1e-8)
  copies = copies[chordAngle(copies, c) < 2 * maxAngle]
  c2 = _genericNear(c)
  stab = chordAngle(allCopies, c) < 1e-8
  stabCopies = qmul(qmul(L[:, None, :], c2[None, None, :]), R[None, :, :]).reshape(-1, 4)[stab]
  keep = _keepInBall(chordAngle, q, c, copies, c2, stabCopies)
  return ori[keep].unique(tol=res / 100, symmetry=False, stable=True)


# a point x is kept when c is its closest copy among copies, and c2 its closest among the
# stabiliser images of c2
def _keepInBall(dist, x, c, copies, c2, stabCopies, tol=1e-10):
  keep = np.all(dist(x, c)[:, None] <= dist(x[:, None], copies[None]) + tol, axis=1)
  if len(stabCopies):
    keep &= np.all(dist(x, c2)[:, None] <= dist(x[:, None], stabCopies[None]) + tol, axis=1)
  return keep


def _vecAngle(a, b):
  return np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, axis=-1))


def localS2Grid(center, maxTheta=None, resolution=5 * degree, respectSymmetry=True):
  """an equispaced grid of directions in a spherical cap about a centre

    v = localS2Grid(center, maxTheta=10*degree)                          # a cap of 10 degree
    v = localS2Grid(center, 10*degree, resolution=1*degree)              # finer
    v = localS2Grid(center, 60*degree, respectSymmetry=False)            # every point of the cap

  The directions lie on rings about the centre, resolution apart. For a crystal direction in
  a frame with symmetry the grid keeps one representative of every direction the cap holds up
  to the symmetry, those nearer the centre than to any of its symmetric copies.

  Input
    center - vector3d or Miller, the centre of the cap
    maxTheta - the radius of the cap
  Output
    v - vector3d in the frame of the centre
  Keywords
    resolution - the spacing of the grid
    respectSymmetry - keep one representative of the directions symmetrically equivalent
  See also
    equispacedS2Grid, localOrientationGrid"""
  from ..sphericalregion import sectorCenter
  if maxTheta is None:
    raise ValueError("localS2Grid needs a maxTheta")
  c = center.normalize().data
  cap = equispacedS2Grid(resolution=resolution, maxTheta=maxTheta)
  pts = (rotation.map(zvector, vector3d(c)) * cap).data
  fr = center.frame
  if not (center.isCrystalDirection and respectSymmetry and fr.numSym() > 1):
    return vector3d(pts, frame=fr)
  els = fr.rot.data.reshape(-1, 5)
  images = qrot(els[:, :4], c[None]) * els[:, 4:5]
  copies = _distinctUnitRows(images, 1e-12)
  sc =sectorCenter(fr.sym).data
  c2 = c + 0.3 * vector3d(c).orth().data if abs(sc @ c) > 1 - 1e-8 else c + 0.3 * (sc - (sc @ c) * c)
  c2 = c2 / np.linalg.norm(c2)
  stab = images @ c > 1 - 1e-8
  stabCopies = (qrot(els[:, :4], c2[None]) * els[:, 4:5])[stab]
  keep = _keepInBall(_vecAngle, pts, c, copies, c2, stabCopies)
  return vector3d(pts[keep], frame=fr).unique(tol=resolution / 100, symmetry=False)


__all__ = ["localOrientationGrid", "localS2Grid"]
