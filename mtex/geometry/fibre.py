# fibres in orientation space (MATLAB @fibre, Julia fibre.jl)
#
#   fibre(h, r), fibre(h, r, ss)      the orientations mapping the crystal direction h onto the specimen direction r
#   fibre(o1, o2)                     the arc from o1 to o2 about the axis of the rotation between them
#   fibre(o1, h)                      the whole fibre through o1 about h
#   fibre.alpha(cs, ss), ..., fibre.theta(cs, ss), fibre.rand(n, cs, ss), fibre.fit(ori)
#   f.orientation(points=1000)        orientations along the fibre, o1 * rot(h, omega)
#   angle(f, ori)                     the angle by which ori * h misses r, over the orbits of both frames
#   symmetrise(f), multiplicity(f)    the equivalent fibres s * o1 * inv(g) about g * h
#
# A fibre holds two orientations between the same frames and the crystal direction h, a
# Miller in the frame of o1; r = o1 * h, the frames and the antipodality are read off them.
# The ends are equal for the full fibre. The three parts share a shape, so a fibre is
# also a list of fibres.
from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from .display import display, header, sizeString
from .frames import crystalFrame, referenceFrame, specimenFrame
from .kernels import qmul, qrot
from .miller import Miller
from .orientation import orientation
from .quaternion import rotation
from .representations import axis2quat, quatAngle, quatAxis, quat2euler
from .session import SESSION
from .symreduce import reductionGroups
from .units import degree
from .vector3d import vector3d
from .install import install
from .framepair import installFramePair

TOLERANCE = 1e-8


def _conj(q):
  return q * np.array([1.0, -1.0, -1.0, -1.0])


def _splitFrames(args):
  return [a for a in args if not isinstance(a, referenceFrame)], [a for a in args if isinstance(a, referenceFrame)]


class fibre:
  """a fibre in orientation space: the orientations mapping the crystal direction h onto the
  specimen direction r, or the arc of them between two orientations; a list of fibres by shape

    f = fibre(h, r)                    the orientations with ori * h = r
    f = fibre(h, r, ss)
    f = fibre(o1, o2)                  the arc from o1 to o2 about the axis between them
    f = fibre(o1, h)                   the whole fibre through o1 about h
    f = fibre(o1, o2, full=True)       the full fibre through the arc"""

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
    """the fibres in another shape"""
    return self._build(self.o1.reshape(*shape), self.o2.reshape(*shape), self.h.reshape(*shape))

  @classmethod
  def cat(cls, *fibres, axis=0):
    """the fibres joined into one list along an axis"""
    return cls._build(orientation.cat(*[f.o1 for f in fibres], axis=axis), orientation.cat(*[f.o2 for f in fibres], axis=axis),
                      vector3d.cat(*[f.h for f in fibres], axis=axis))

  # ------------------------------------------------------------------ sampling

  def rotation(self, points=1000):
    """the rotations along the fibre, points x shape, turning about h from o1 to o2"""
    omega = np.linspace(0, 1, points)[(slice(None),) + (None,) * self.ndim] * self.arcAngle()
    axis = np.broadcast_to(self.h.normalize().data, omega.shape + (3,))
    return rotation._fromData(qmul(self.o1.data, axis2quat(axis, omega)))

  def orientation(self, points=1000):
    """the orientations along the fibre, points x shape, turning about h from o1 to o2"""
    return orientation._fromRotation(self.rotation(points).data, (self.CS, self.SS))

  # ------------------------------------------------------------------ distance

  # the orbit of h under the crystal group and of r under the specimen group, as unit vectors
  def _orbits(self):
    L, R = reductionGroups(self.o1)
    hs = _uniqueDirections(qrot(R, self.h.normalize().data[..., None, :]))
    rs = _uniqueDirections(qrot(L, self.r.normalize().data[..., None, :]))
    return hs, rs

  def angle(self, ori, symmetry=True):
    """the angle by which ori * h misses r, the least over both orbits, from the chord of the
    closest pair; with another fibre the largest distance of its points to this one"""
    if isinstance(ori, fibre):
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
    best = np.take_along_axis(y, kh[..., None, None], axis=-2)[..., 0, :] - rs[kr]
    return 2 * np.arcsin(np.minimum(np.linalg.norm(best, axis=-1) / 2, 1.0))

  # ------------------------------------------------------------------ symmetry

  def symmetrise(self, unique=False):
    """the equivalent fibres s * o1 * inv(g) about g * h over both groups, or the distinct ones"""
    L, R = reductionGroups(self.o1)
    nL, nR = len(L), len(R)
    o1 = qmul(qmul(L[:, None, None, :], self.o1.data[None, None]), _conj(R)[None, :, None, :])
    o2 = qmul(qmul(L[:, None, None, :], self.o2.data[None, None]), _conj(R)[None, :, None, :])
    h = np.broadcast_to(qrot(R[None, :, None, :], self.h.data[None, None]), o1.shape[:-1] + (3,))
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
    within tol, both flipped counting as well; a segment is not the full fibre (MATLAB's eq)"""
    if not isinstance(other, fibre):
      raise TypeError("a fibre is compared with a fibre")
    if self.CS is not other.CS or self.SS is not other.SS:
      return np.zeros(np.broadcast_shapes(self.shape, other.shape), dtype=bool)
    L, R = reductionGroups(self.o1)
    hd = np.sum(qrot(R, self.h.normalize().data[..., None, :]) * other.h.normalize().data[..., None, :], axis=-1)
    rd = np.sum(qrot(L, self.r.normalize().data[..., None, :]) * other.r.normalize().data[..., None, :], axis=-1)
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
    frames given

      f = fibre.rand(10, cs)
      f = fibre.rand(10, cs, ss, rng=rng)"""
    shape, frames = _splitFrames(args)
    cs = frames[0] if frames else crystalFrame("1")
    ss = frames[1] if len(frames) > 1 else specimenFrame.default
    h = vector3d.rand(*shape, rng=rng)
    r = vector3d.rand(*shape, rng=rng)
    return cls(Miller(h, cs) if isinstance(cs, crystalFrame) else vector3d(h, frame=cs), vector3d(r, frame=ss))

  @classmethod
  def fit(cls, ori, *args, diagnostics=False, symmetry=True):
    """the fibre through a list of orientations: the plane of the two leading eigenvectors
    of their quaternion tensor, the orientations first moved to the representatives
    nearest their mean, or taken as stored with symmetry=False; with diagnostics the
    eigenvalues, ascending and summing to one, and the mean angle of the orientations to
    the fibre come too. This is the local fit of a concentrated cloud; the search through
    the pole figures of an ODF is the business of P2"""
    # 'local' names the one algorithm there is, the local quaternion tensor fit
    for a in args:
      if isinstance(a, str) and a == "noSymmetry":
        symmetry = False
      elif not (isinstance(a, str) and a == "local"):
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
      ...
    @classmethod
    def beta(cls, *frames, full=False) -> Any:
      ...
    @classmethod
    def gamma(cls, *frames, full=False) -> Any:
      ...
    @classmethod
    def eta(cls, *frames, full=False) -> Any:
      ...
    @classmethod
    def tau(cls, *frames, full=False) -> Any:
      ...
    @classmethod
    def epsilon(cls, *frames, full=False) -> Any:
      ...
    @classmethod
    def theta(cls, *frames) -> Any:
      ...
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

# the axis of the rotation from o1 to o2 in the crystal frame, the c axis where they coincide
def _axisBetween(o1, o2):
  m = qmul(_conj(o1.data), o2.data)
  xyz = np.where((quatAngle(m) < 1e-10)[..., None], [0.0, 0.0, 1.0], quatAxis(m))
  cs = o1.frameA
  return Miller(vector3d(xyz), cs, "uvw") if isinstance(cs, crystalFrame) else vector3d(xyz, frame=cs)


# the distinct unit vectors of an orbit, shape (n, 3)
def _uniqueDirections(xyz, tol=TOLERANCE):
  xyz = xyz.reshape(-1, 3)
  keep = np.ones(len(xyz), dtype=bool)
  for k in range(len(xyz)):
    if keep[k]:
      keep[k + 1:] &= xyz[k + 1:] @ xyz[k] < 1 - tol
  return xyz[keep]


# the first of every set of equal fibres: h and r agree, or both flip sign
def _distinct(h, r, tol=TOLERANCE):
  keep = np.ones(len(h), dtype=bool)
  for k in range(len(h)):
    if keep[k]:
      hd, rd = h[k + 1:] @ h[k], r[k + 1:] @ r[k]
      keep[k + 1:] &= ~((hd > 1 - tol) & (rd > 1 - tol) | (hd < tol - 1) & (rd < tol - 1))
  return keep


def _eulerString(o):
  e = quat2euler(o.data.reshape(4), None)
  return "(" + ",".join(f"{x / degree:.4g}°" for x in e) + ")"


# the named fibres of rolling textures between the orientations MTEX puts at their ends,
# by their Bunge Euler angles in degrees, so that the geodesic is MATLAB's whatever
# representative byMiller would pick
NAMED_FIBRES = {"alpha": ((315, 0, 0), (180, 54.735610, 45)), "beta": ((180, 35.264390, 45), (270, 62.785077, 45)),
                "gamma": ((180, 54.735610, 45), (270, 54.735610, 45)), "eta": ((0, 0, 0), (0, 45, 0)),
                "tau": ((315, 0, 0), (0, 45, 0)), "epsilon": ((315, 0, 0), (270, 54.735610, 45))}


def _named(name):
  def build(cls, *frames, full=False):
    frames = _splitFrames(frames)[1]
    cs = frames[0] if frames else crystalFrame("432")
    ss = frames[1] if len(frames) > 1 else specimenFrame.default
    (a1, b1, c1), (a2, b2, c2) = NAMED_FIBRES[name]
    o1 = orientation.byEuler(a1 * degree, b1 * degree, c1 * degree, cs, ss)
    o2 = orientation.byEuler(a2 * degree, b2 * degree, c2 * degree, cs, ss)
    return cls(o1, o2, full=full)
  return classmethod(build)


for _name in NAMED_FIBRES:
  install(fibre, _named(_name), _name)
del _name


def _theta(cls, *frames):
  frames = _splitFrames(frames)[1]
  cs = frames[0] if frames else crystalFrame("432")
  ss = frames[1] if len(frames) > 1 else specimenFrame.default
  return cls(Miller(1, 0, 0, cs, "uvw"), vector3d(0, 0, 1, frame=ss))


install(fibre, classmethod(_theta), "theta")


# the two frames under their names of the Julia port, frameA and frameB
installFramePair(fibre)

__all__ = ["fibre"]
