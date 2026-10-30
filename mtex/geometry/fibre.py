# fibres in orientation space (MATLAB @fibre, Julia fibre.jl)
#
#   fibre(h, r), fibre(h, r, ss)      the orientations mapping the crystal direction h onto the specimen direction r
#   fibre(o1, o2)                     the arc from o1 to o2 about the axis of the rotation between them
#   fibre(o1, h)                      the whole fibre through o1 about h
#   fibre.alpha(cs, ss), ..., fibre.theta(cs, ss), fibre.rand(n, cs, ss), fibre.fit(ori)
#   f.orientation(points=1000)        orientations along the fibre, o1 * rot(h, omega)
#   angle(f, ori)                     the angle by which ori * h misses r, over the orbits of both frames,
#                                     with return_nearest=True also the nearest orientation on the fibre
#   symmetrise(f), multiplicity(f)    the equivalent fibres, one for every pair of images of h and r
#
# The equivalents of h and r are their images under the point groups, the improper elements
# included, as MATLAB's h.symmetrise: in a centrosymmetric crystal, or a specimen frame holding
# the inversion, the fibre of h || r is that of -h || r too, which no rotation of the symmetry
# reaches.
#
# A fibre holds two orientations between the same frames and the crystal direction h, a
# Miller in the frame of o1; r = o1 * h, the frames and the antipodality are read off them.
# The ends are equal for the full fibre. The three parts share a shape, so a fibre is
# also a list of fibres.
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from .kernels import qconj
from .arguments import refuseFlags
from .display import display, header, sizeString
from .frames import crystalFrame, specimenFrame
from .kernels import qmul, qrot
from .miller import Miller
from .orientation import _defaultFrames, _splitFrames, orientation
from .quaternion import rotation
from .representations import axis2quat, map1Kernel, quatAngle, quatAxis, quat2euler
from .session import SESSION
from .symmetryops import _images
from .symreduce import _firstOfClusters
from .units import degree
from .vector3d import vector3d
from .install import install
from .framepair import installFramePair

TOLERANCE = 1e-8


class fibre:
  """a fibre in orientation space: the orientations mapping the crystal direction h onto the
  specimen direction r, or the arc of them between two orientations; a list of fibres by shape

    f = fibre(h, r)  # the orientations with ori * h = r
    f = fibre(h, r, ss)
    f = fibre(o1, o2)  # the arc from o1 to o2 about the axis between them
    f = fibre(o1, h)  # the whole fibre through o1 about h
    f = fibre(o1, o2, full=True)  # the full fibre through the arc

  Input
    a - crystal direction h, or first orientation o1
    b - specimen direction r, second orientation o2, or a crystal direction for a fibre through o1
    frames - optional crystal and specimen frames
    full - include the entire circular fibre rather than only the arc between two orientations; default False"""

  def __init__(self, a, b, *frames, full=False):
    if isinstance(a, orientation):
      cs, ss = a.frameA, a.frameB
      o1 = a
      if isinstance(b, orientation):
        if b.frameA is not cs or b.frameB is not ss:
          raise ValueError("both ends of a fibre are between the same frames")
        o2 = b
        h = _axisBetween(o1, o2)
      elif isinstance(b, vector3d):
        o2 = o1
        h = Miller(b, cs) if isinstance(cs, crystalFrame) else vector3d(b, frame=cs)
        if b.frame is not None:
          SESSION.joinFrames(b.frame, cs)
      else:
        raise TypeError("a fibre through an orientation needs a second orientation or a crystal direction")
    elif isinstance(a, vector3d) and isinstance(b, vector3d):
      frames = _splitFrames(frames)[1]
      cs = a.frame if a.frame is not None else crystalFrame("1")
      ss = frames[0] if frames else (b.frame if b.frame is not None else specimenFrame.default)
      o1 = orientation._fromRotation(orientation.map(vector3d(a.data), vector3d(b.data)).data, (cs, ss))
      o2 = o1
      h = Miller(a, cs) if isinstance(cs, crystalFrame) else vector3d(a, frame=cs)
    else:
      raise TypeError("a fibre is built from two directions, two orientations, or an orientation and a direction")
    if o1.shape != o2.shape or o1.shape != h.shape:
      o1d, o2d, hd = np.broadcast_arrays(o1.data, o2.data, h.data[..., :1])
      o1, o2, h = o1._new(o1d.copy()), o2._new(o2d.copy()), h._new(np.broadcast_to(h.data, hd.shape[:-1] + (3,)).copy())
    self.o1, self.o2, self.h = o1, (o1 if full else o2), h

  @classmethod
  def _build(cls, o1, o2, h):
    f = object.__new__(cls)
    f.o1, f.o2, f.h = o1, o2, h
    return f

  # ------------------------------------------------------------------ derived properties

  @property
  def r(self):
    """the specimen direction of the fibre, o1 * h"""
    return self.o1 * self.h

  @property
  def CS(self):
    """the crystal frame of the fibre, MATLAB's f.CS"""
    return self.o1.frameA

  @property
  def SS(self):
    """the specimen frame of the fibre, MATLAB's f.SS"""
    return self.o1.frameB

  @property
  def antipodal(self):
    """whether the fibre is antipodal, read off its crystal direction"""
    return self.h.antipodal

  @property
  def isFull(self):
    """the fibre is full when both ends coincide"""
    return rotation.angle(self.o1, self.o2) < 1e-10

  def arcAngle(self):
    """the angle turned about h from o1 to o2, the whole circle when full"""
    return np.where(self.isFull, 2 * np.pi, rotation.angle(self.o1, self.o2))

  # ------------------------------------------------------------------ the list

  @property
  def shape(self):
    """the shape of the list of fibres"""
    return self.h.shape

  @property
  def ndim(self):
    """the number of dimensions of the list of fibres"""
    return self.h.ndim

  @property
  def size(self):
    """the number of fibres in the list"""
    return self.h.size

  def __len__(self):
    return self.size

  def __getitem__(self, idx):
    return self._build(self.o1[idx], self.o2[idx], self.h[idx])

  def __iter__(self):
    return (self[i] for i in range(self.shape[0]))

  def flatten(self):
    """the fibres as a flat list"""
    return self._build(self.o1.flatten(), self.o2.flatten(), self.h.flatten())

  def reshape(self, *shape):
    """the fibres in another shape

      f = f.reshape(m, n)"""
    return self._build(self.o1.reshape(*shape), self.o2.reshape(*shape), self.h.reshape(*shape))

  @classmethod
  def cat(cls, *fibres, axis=0):
    """the fibres joined into one list along an axis, MATLAB's [f1, f2]

      f = cat(f1, f2)            # one after the other
      f = cat(f1, f2, axis=1)    # side by side

    Input
      fibres - fibre, between the same frames
    Output
      f - fibre
    Keywords
      axis - the axis of the shape along which the fibres are joined
    See also
      fibre.reshape, fibre.flatten"""
    return cls._build(orientation.cat(*[f.o1 for f in fibres], axis=axis), orientation.cat(*[f.o2 for f in fibres], axis=axis),
                      vector3d.cat(*[f.h for f in fibres], axis=axis))

  # ------------------------------------------------------------------ sampling

  def rotation(self, points=1000, return_angles=False):
    """the rotations along the fibre, points x shape, turning about h from o1 to o2, MATLAB's
    fibre.rotation

      rot = f.rotation()                            # 1000 rotations along the fibre
      rot = f.rotation(points=100)                  # 100 of them
      rot, omega = f.rotation(return_angles=True)   # and the angles from o1

    The points are equispaced in the angle about h, from o1 (omega = 0) to o2; a full fibre runs
    once around.

    Input
      f - fibre
    Output
      rot - rotation, of shape (points,) + f.shape
      omega - the angles the rotations are turned from o1 by, of the same shape (return_angles)
    Keywords
      points - the number of rotations per fibre
      return_angles - also return omega, MATLAB's second output
    See also
      fibre.orientation, fibre.arcAngle"""
    omega = np.linspace(0, 1, points)[(slice(None),) + (None,) * self.ndim] * self.arcAngle()
    axis = np.broadcast_to(self.h.normalize().data, omega.shape + (3,))
    rot = rotation._fromData(qmul(self.o1.data, axis2quat(axis, omega)))
    return (rot, omega) if return_angles else rot

  def orientation(self, points=1000):
    """the orientations along the fibre, points x shape, turning about h from o1 to o2, MATLAB's
    fibre.orientation

      ori = f.orientation()             # 1000 orientations along the fibre
      ori = f.orientation(points=10)    # 10 of them

    Input
      f - fibre
    Output
      ori - orientation between the fibre's frames, of shape (points,) + f.shape
    Keywords
      points - the number of orientations per fibre
    See also
      fibre.rotation"""
    return orientation._fromRotation(self.rotation(points).data, (self.CS, self.SS))

  # ------------------------------------------------------------------ distance

  # the images of h under the crystal point group and of r under the specimen point group, as unit vectors
  def _orbits(self):
    return directionOrbits(self.CS, self.SS, self.h.normalize().data, self.r.normalize().data)

  def angle(self, ori, symmetry=True, return_nearest=False):
    """the angle by which ori * h misses r, the least over both orbits, from the chord of the
    closest pair; MATLAB's fibre.angle

      omega = angle(f, ori)                               # the angle of orientations to the fibre
      omega = angle(f, ori, symmetry=False)               # ignore the symmetric equivalents
      omega, oriF = angle(f, ori, return_nearest=True)    # and the orientations on the fibre
      omega = angle(f1, f2)                               # the angle between two fibres

    The angle of an orientation to the fibre h || r is the angle between ori * h and r, the least
    over the images of h under the crystal point group and of r under the specimen point group,
    the improper elements included: in a centrosymmetric crystal -h is an equivalent of h. With
    another fibre it is the largest angle of 1000 of its orientations to this one. The fibre is
    taken as the full fibre, its ends o1 and o2 do not count.

    Input
      f - fibre, a single one
      ori - orientation, or a fibre
    Output
      omega - the angles, of the shape of ori
      oriF - the orientations on the fibre nearest ori, which turn the closest image of h onto r
          (return_nearest)
    Keywords
      symmetry - take the least over the symmetric equivalents; False compares h and r only
      return_nearest - also return oriF, MATLAB's second output; not with a fibre
    See also
      orientation.angle, fibre.isclose"""
    if isinstance(ori, fibre):
      if return_nearest:
        raise ValueError("the nearest orientation is taken to an orientation, not to a fibre")
      return np.max(self.angle(ori.orientation(1000)))
    if self.size != 1:
      raise ValueError("the angle to a list of fibres is taken one fibre at a time")
    if isinstance(ori, orientation):
      SESSION.joinFrames(ori.frameA, self.CS)
      SESSION.joinFrames(ori.frameB, self.SS)
    if not symmetry:
      hs, rs = self.h.normalize().data.reshape(1, 3), self.r.normalize().data.reshape(1, 3)
    else:
      hs, rs = self._orbits()
    y = qrot(ori.data[..., None, :], hs)                        # (..., nH, 3)
    c = y @ rs.T                                                # (..., nH, nR)
    flat = c.reshape(c.shape[:-2] + (-1,))
    k = np.argmax(flat, axis=-1)
    kh, kr = np.unravel_index(k, c.shape[-2:])
    yBest = np.take_along_axis(y, kh[..., None, None], axis=-2)[..., 0, :]
    omega = 2 * np.arcsin(np.minimum(np.linalg.norm(yBest - rs[kr], axis=-1) / 2, 1.0))
    if not return_nearest:
      return omega
    # the smallest rotation turning the closest image ori * h onto its r, applied to ori
    q = qmul(map1Kernel(yBest, rs[kr]), ori.data)
    frames = (ori.frameA, ori.frameB) if isinstance(ori, orientation) else (self.CS, self.SS)
    return omega, orientation._fromRotation(q, frames)

  # ------------------------------------------------------------------ symmetry

  def symmetrise(self, unique=False):
    """the symmetrically equivalent fibres, MATLAB's fibre.symmetrise

      fS = f.symmetrise()               # every equivalent, numSS * numCS x shape
      fS = f.symmetrise(unique=True)    # the distinct ones of a single fibre

    There is one equivalent fibre for every element g of the crystal point group and s of the
    specimen point group, about g * h through s * r: for rotations it runs from s * o1 * inv(g),
    for an improper element, which maps h onto -(-g) h, from the rotation -g turned by a half
    turn about an axis normal to h (on the specimen side normal to s * r), so that an arc keeps
    its length. MATLAB's second output, the number of distinct ones, is f.multiplicity().

    Input
      f - fibre
    Output
      fS - fibre
    Keywords
      unique - keep only the distinct fibres, those whose h and r differ (a single fibre only)
    See also
      fibre.multiplicity, orientation.symmetrise"""
    T, hs = _crystalSide(self.CS, self.h.normalize().data)
    U = _specimenSide(self.SS, self.r.normalize().data)
    nL, nR = U.shape[-2], T.shape[-2]
    o1 = qmul(qmul(np.moveaxis(U, -2, 0)[:, None], self.o1.data[None, None]), np.moveaxis(T, -2, 0)[None, :])
    o2 = qmul(qmul(np.moveaxis(U, -2, 0)[:, None], self.o2.data[None, None]), np.moveaxis(T, -2, 0)[None, :])
    h = np.broadcast_to(np.moveaxis(hs, -2, 0)[None] * np.linalg.norm(self.h.data, axis=-1)[..., None], o1.shape[:-1] + (3,))
    out = self._build(self.o1._new(o1.reshape((nL * nR,) + self.shape + (4,))), self.o2._new(o2.reshape((nL * nR,) + self.shape + (4,))),
                      self.h._new(h.reshape((nL * nR,) + self.shape + (3,))))
    if not unique:
      return out
    if self.size != 1:
      raise ValueError("unique symmetrisation gives orbits of different sizes; symmetrise the fibres one by one")
    out = out.flatten()
    keep = _distinct(out.h.normalize().data, out.r.normalize().data)
    return out[keep]

  def multiplicity(self):
    """the number of distinct symmetrically equivalent fibres"""
    return self.symmetrise(unique=True).size

  # two fibres are equal when their frames are and h and r are; isclose compares up to symmetry
  def __eq__(self, other):
    if not isinstance(other, fibre):
      return NotImplemented
    if self.CS is not other.CS or self.SS is not other.SS:
      return np.zeros(np.broadcast_shapes(self.shape, other.shape), dtype=bool)
    return (self.h == other.h) & (self.r == other.r) & (self.isFull == other.isFull)

  def isclose(self, other, tol=1e-6):
    """equal when the frames are the same and h and r agree up to the symmetry of the frames
    within tol, both flipped counting as well; MATLAB's eq

      b = f1.isclose(f2)              # equal up to symmetry
      b = f1.isclose(f2, tol=1e-3)    # within a wider tolerance

    A segment is not the full fibre, and its ends are not compared. f1 == f2 compares h and r
    as they are, without the symmetry.

    Input
      f1, f2 - fibre
    Output
      b - bool, of the broadcast shape
    Keywords
      tol - the tolerance on the dot products of h and of r, 1 - cos of the angle
    See also
      fibre.angle"""
    if not isinstance(other, fibre):
      raise TypeError("a fibre is compared with a fibre")
    if self.CS is not other.CS or self.SS is not other.SS:
      return np.zeros(np.broadcast_shapes(self.shape, other.shape), dtype=bool)
    hd = np.sum(_images(self.CS, self.h.normalize().data) * other.h.normalize().data[..., None, :], axis=-1)
    rd = np.sum(_images(self.SS, self.r.normalize().data) * other.r.normalize().data[..., None, :], axis=-1)
    same = np.any(hd > 1 - tol, axis=-1) & np.any(rd > 1 - tol, axis=-1)
    flipped = np.any(hd < tol - 1, axis=-1) & np.any(rd < tol - 1, axis=-1)
    return (same | flipped) & (self.isFull == other.isFull)

  def __ne__(self, other):
    eq = self.__eq__(other)
    return NotImplemented if eq is NotImplemented else ~eq

  # ------------------------------------------------------------------ constructors

  @classmethod
  def rand(cls, *args, rng=None):
    """random fibres between random crystal and specimen directions, of the shape and between the
    frames given; MATLAB's fibre.rand

      f = fibre.rand(10, cs)                 # ten full fibres in the crystal frame cs
      f = fibre.rand(10, cs, ss, rng=rng)    # between cs and ss, from a generator

    Input
      args - the shape, then the frames, crystal and specimen
    Output
      f - fibre, full fibres h || r with h and r uniform on the sphere
    Keywords
      rng - a numpy random generator
    See also
      orientation.rand, vector3d.rand"""
    shape, frames = _splitFrames(args)
    cs, ss = _defaultFrames(frames)
    h = vector3d.rand(*shape, rng=rng)
    r = vector3d.rand(*shape, rng=rng)
    return cls(Miller(h, cs) if isinstance(cs, crystalFrame) else vector3d(h, frame=cs), vector3d(r, frame=ss))

  @classmethod
  def fit(cls, ori, *args, diagnostics=False, symmetry=True):
    """the fibre that fits a list of orientations best, MATLAB's fibre.fit

      f = fibre.fit(ori)                              # the fibre through the orientations
      f, lam, delta = fibre.fit(ori, diagnostics=True)  # with the eigenvalues and the mean angle
      f = fibre.fit(ori, symmetry=False)              # the orientations as stored

    The fibre is the plane of the two leading eigenvectors of the quaternion tensor of the
    orientations, after they are moved to the symmetric equivalents nearest their mean. This
    is MATLAB's 'local' fit of a concentrated cloud, the one algorithm here; MATLAB's fit to an
    ODF is not taken. The result is a full fibre, with r in the upper hemisphere.

    Input
      ori - orientation
      args - 'local', MATLAB's flag, accepted
    Output
      f - fibre, full
      lam - the eigenvalues of the quaternion tensor, ascending and summing to one (diagnostics)
      delta - the mean angle of the orientations to the fibre (diagnostics)
    Keywords
      diagnostics - also return lam and delta, MATLAB's second and third outputs
      symmetry - move the orientations to the equivalents nearest their mean first; False
          takes them as stored
    See also
      fibre.angle, orientation.mean"""
    # 'local' names the one algorithm there is, the local quaternion tensor fit
    refuseFlags(args, "fibre.fit", names=("local",))
    for a in args:
      if not isinstance(a, str):
        raise ValueError(f"fibre.fit does not understand {a!r}")
    o = ori.flatten()
    o = o[~np.isnan(o.data).any(axis=-1)]
    if symmetry:
      o = o.projectIntoFundamentalRegion(o.mean())
    q = o.data
    vals, vecs = np.linalg.eigh(q.T @ q / len(q))
    frames = (ori.frameA, ori.frameB)
    o1 = orientation._fromRotation(vecs[:, -1].copy(), frames)
    o2 = orientation._fromRotation(vecs[:, -2].copy(), frames)
    f = cls(o1, o2, full=True)
    # the eigenvectors fix the fibre up to (h, r) against (-h, -r); take r in the upper hemisphere
    r = f.r.data.reshape(3)
    k = next((j for j in (2, 1, 0) if abs(r[j]) > 1e-9), 2)
    if r[k] < 0:
      f.h = f.h._new(-f.h.data)
    if not diagnostics:
      return f
    return f, vals, float(np.nanmean(f.angle(ori)))

  # ------------------------------------------------------------------ display

  def __repr__(self):
    pairs = [("size", sizeString(self.shape) if self.size != 1 else None), ("antipodal", "true" if self.antipodal else None)]
    if self.size == 1:
      h = self.h.round()
      hChar = Miller(h, self.CS.stripSym()).char() if isinstance(self.CS, crystalFrame) else str(h)
      pairs.append(("h || r", f"{hChar} || {str(vector3d(self.r.data).round())}"))
      if not np.all(self.isFull):
        pairs.append(("o1 → o2", f"{_eulerString(self.o1)} → {_eulerString(self.o2)}"))
    return display(header("fibre", self.o1._frameSummary()), pairs)

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    @classmethod
    def alpha(cls, *frames, full=False) -> Any:
      """the alpha fibre of rolling textures, between the orientations MTEX puts at its ends; MATLAB's fibre.alpha

        f = fibre.alpha(cs)  # the segment in the crystal frame cs
        f = fibre.alpha(cs, ss, full=True)  # the whole fibre between cs and ss

      The segment runs from the Euler angles (45, 0, 0) to (0, 54.7356, 45) degrees
      (Bunge); the frames default to the cubic group 432 and the specimen frame.

      Input
        frames - the crystal frame, then the specimen frame
      Output
        f - fibre
      Keywords
        full - the whole fibre instead of the segment between the ends
      See also
        fibre, fibre.theta, fibre.orientation"""
    @classmethod
    def beta(cls, *frames, full=False) -> Any:
      """the beta fibre of rolling textures, between the orientations MTEX puts at its ends; MATLAB's fibre.beta

        f = fibre.beta(cs)  # the segment in the crystal frame cs
        f = fibre.beta(cs, ss, full=True)  # the whole fibre between cs and ss

      The segment runs from the Euler angles (180, 35.2644, 45) to (270, 62.7851, 45) degrees
      (Bunge); the frames default to the cubic group 432 and the specimen frame.

      Input
        frames - the crystal frame, then the specimen frame
      Output
        f - fibre
      Keywords
        full - the whole fibre instead of the segment between the ends
      See also
        fibre, fibre.theta, fibre.orientation"""
    @classmethod
    def gamma(cls, *frames, full=False) -> Any:
      """the gamma fibre of rolling textures, between the orientations MTEX puts at its ends; MATLAB's fibre.gamma

        f = fibre.gamma(cs)  # the segment in the crystal frame cs
        f = fibre.gamma(cs, ss, full=True)  # the whole fibre between cs and ss

      The segment runs from the Euler angles (180, 54.7356, 45) to (270, 54.7356, 45) degrees
      (Bunge); the frames default to the cubic group 432 and the specimen frame.

      Input
        frames - the crystal frame, then the specimen frame
      Output
        f - fibre
      Keywords
        full - the whole fibre instead of the segment between the ends
      See also
        fibre, fibre.theta, fibre.orientation"""
    @classmethod
    def eta(cls, *frames, full=False) -> Any:
      """the eta fibre of rolling textures, between the orientations MTEX puts at its ends; MATLAB's fibre.eta

        f = fibre.eta(cs)  # the segment in the crystal frame cs
        f = fibre.eta(cs, ss, full=True)  # the whole fibre between cs and ss

      The segment runs from the Euler angles (0, 0, 0) to (0, 45, 0) degrees
      (Bunge); the frames default to the cubic group 432 and the specimen frame.

      Input
        frames - the crystal frame, then the specimen frame
      Output
        f - fibre
      Keywords
        full - the whole fibre instead of the segment between the ends
      See also
        fibre, fibre.theta, fibre.orientation"""
    @classmethod
    def tau(cls, *frames, full=False) -> Any:
      """the tau fibre of rolling textures, between the orientations MTEX puts at its ends; MATLAB's fibre.tau

        f = fibre.tau(cs)  # the segment in the crystal frame cs
        f = fibre.tau(cs, ss, full=True)  # the whole fibre between cs and ss

      The segment runs from the Euler angles (315, 0, 0) to (270, 90, 45) degrees
      (Bunge); the frames default to the cubic group 432 and the specimen frame.

      Input
        frames - the crystal frame, then the specimen frame
      Output
        f - fibre
      Keywords
        full - the whole fibre instead of the segment between the ends
      See also
        fibre, fibre.theta, fibre.orientation"""
    @classmethod
    def epsilon(cls, *frames, full=False) -> Any:
      """the epsilon fibre of rolling textures, between the orientations MTEX puts at its ends; MATLAB's fibre.epsilon

        f = fibre.epsilon(cs)  # the segment in the crystal frame cs
        f = fibre.epsilon(cs, ss, full=True)  # the whole fibre between cs and ss

      The segment runs from the Euler angles (315, 0, 0) to (270, 54.7356, 45) degrees
      (Bunge); the frames default to the cubic group 432 and the specimen frame.

      Input
        frames - the crystal frame, then the specimen frame
      Output
        f - fibre
      Keywords
        full - the whole fibre instead of the segment between the ends
      See also
        fibre, fibre.theta, fibre.orientation"""
    @classmethod
    def theta(cls, *frames) -> Any:
      """the theta fibre of rolling textures, the orientations with <100> along the normal direction"""
    @property
    def frameA(self) -> Any:
      ...
    @frameA.setter
    def frameA(self, value) -> None:
      ...
    @property
    def frameB(self) -> Any:
      ...
    @frameB.setter
    def frameB(self, value) -> None:
      ...


# ------------------------------------------------------------------ helpers

def directionOrbits(cs, ss, h, r):
  """the distinct images of the unit vectors h under the crystal point group and r under the
  specimen point group, the improper elements included, as rows; -h and -r are among them where
  either group holds the inversion, and a tube about -h || r is that about h || -r"""
  return _uniqueDirections(_images(cs, h.reshape(3))), _uniqueDirections(_images(ss, r.reshape(3)))


# the rotations t with t (g h) = h for every element g of the crystal point group, (..., n, 4), and
# the images g h: inv(g) for a rotation, the half turn about an axis normal to h after inv(-g) for
# an improper element
def _crystalSide(frame, h):
  o = frame.rot.data.reshape(-1, 5)
  t = np.broadcast_to(qconj(o[:, :4]), h.shape[:-1] + (len(o), 4)).copy()
  flip = o[:, 4] < 0
  t[..., flip, :] = qmul(_halfTurn(h)[..., None, :], t[..., flip, :])
  return t, _images(frame, h)


# the rotations u with u r = s r for every element s of the specimen point group, (..., n, 4): s for
# a rotation, the half turn about an axis normal to s r after -s for an improper element
def _specimenSide(frame, r):
  o = frame.rot.data.reshape(-1, 5)
  u = np.broadcast_to(o[:, :4], r.shape[:-1] + (len(o), 4)).copy()
  flip = o[:, 4] < 0
  u[..., flip, :] = qmul(_halfTurn(_images(frame, r)[..., flip, :]), u[..., flip, :])
  return u


# the half turn about an axis normal to the unit vectors v, as quaternions
def _halfTurn(v):
  a = np.cross(v, [1.0, 0, 0])
  a = np.where((np.linalg.norm(a, axis=-1) < 0.5)[..., None], np.cross(v, [0, 1.0, 0]), a)
  a = a / np.linalg.norm(a, axis=-1, keepdims=True)
  return np.concatenate([np.zeros(a.shape[:-1] + (1,)), a], axis=-1)


# the axis of the rotation from o1 to o2 in the crystal frame, the c axis where they coincide
def _axisBetween(o1, o2):
  m = qmul(qconj(o1.data), o2.data)
  xyz = np.where((quatAngle(m) < 1e-10)[..., None], [0.0, 0.0, 1.0], quatAxis(m))
  cs = o1.frameA
  return Miller(vector3d(xyz), cs, "uvw") if isinstance(cs, crystalFrame) else vector3d(xyz, frame=cs)


# the distinct unit vectors of an orbit, shape (n, 3)
def _uniqueDirections(xyz, tol=TOLERANCE):
  xyz = xyz.reshape(-1, 3)
  return xyz[_firstOfClusters(xyz @ xyz.T >= 1 - tol)]


# the first of every set of equal fibres: h and r agree, or both flip sign
def _distinct(h, r, tol=TOLERANCE):
  hd, rd = h @ h.T, r @ r.T
  return _firstOfClusters((hd > 1 - tol) & (rd > 1 - tol) | (hd < tol - 1) & (rd < tol - 1))


def _eulerString(o):
  e = quat2euler(o.data.reshape(4), None)
  return "(" + ",".join(f"{x / degree:.4g}°" for x in e) + ")"


# the named fibres of rolling textures between the orientations MTEX puts at their ends, by
# their Bunge Euler angles in degrees; the ends share the fibre's crystal direction, alpha
# (001)[1-10] to (111)[1-10] about [1-10] || RD, tau (001)[110] to (110)[00-1] about
# [-110] || TD
NAMED_FIBRES = {"alpha": ((45, 0, 0), (0, 54.735610, 45)), "beta": ((180, 35.264390, 45), (270, 62.785077, 45)),
                "gamma": ((180, 54.735610, 45), (270, 54.735610, 45)), "eta": ((0, 0, 0), (0, 45, 0)),
                "tau": ((315, 0, 0), (270, 90, 45)), "epsilon": ((315, 0, 0), (270, 54.735610, 45))}


def _named(name):
  def build(cls, *frames, full=False):
    frames = _splitFrames(frames)[1]
    cs, ss = _defaultFrames(frames, "432")
    (a1, b1, c1), (a2, b2, c2) = NAMED_FIBRES[name]
    o1 = orientation.byEuler(a1 * degree, b1 * degree, c1 * degree, cs, ss)
    o2 = orientation.byEuler(a2 * degree, b2 * degree, c2 * degree, cs, ss)
    return cls(o1, o2, full=full)
  (a1, b1, c1), (a2, b2, c2) = NAMED_FIBRES[name]
  build.__doc__ = "\n".join([
    f"the {name} fibre of rolling textures, between the orientations MTEX puts at its ends; MATLAB's fibre.{name}",
    "",
    f"  f = fibre.{name}(cs)  # the segment in the crystal frame cs",
    f"  f = fibre.{name}(cs, ss, full=True)  # the whole fibre between cs and ss",
    "",
    f"The segment runs from the Euler angles ({a1:g}, {b1:g}, {c1:g}) to ({a2:g}, {b2:g}, {c2:g}) degrees",
    "(Bunge); the frames default to the cubic group 432 and the specimen frame.",
    "",
    "Input",
    "  frames - the crystal frame, then the specimen frame",
    "Output",
    "  f - fibre",
    "Keywords",
    "  full - the whole fibre instead of the segment between the ends",
    "See also",
    "  fibre, fibre.theta, fibre.orientation"])
  return classmethod(build)


for _name in NAMED_FIBRES:
  install(fibre, _named(_name), _name)
del _name


def _theta(cls, *frames):
  """the theta fibre of rolling textures, the orientations with <100> along the normal direction"""
  frames = _splitFrames(frames)[1]
  cs, ss = _defaultFrames(frames, "432")
  return cls(Miller(1, 0, 0, cs, "uvw"), vector3d(0, 0, 1, frame=ss))


install(fibre, classmethod(_theta), "theta")


# the two frames under their names of the Julia port, frameA and frameB
installFramePair(fibre)

__all__ = ["fibre"]
