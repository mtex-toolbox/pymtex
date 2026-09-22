# slip systems and dislocation systems (MATLAB @slipSystem, @dislocationSystem; Julia
# slipsystem.jl)
#
#   sS = slipSystem(b, n), slipSystem(b, n, CRSS), slipSystem.fcc(cs), slipSystem.basal(cs, CRSS=2)
#   sS.symmetrise(), sS.symmetrise(antipodal=True), sS.SchmidFactor(v), ori * sS, sS.trace()
#   sS.deformationTensor(), sS.SchmidTensor(), sS.mPrime(sS2), sS.residualBurgersVector(sS2)
#   dS = dislocationSystem(sS), dislocationSystem(b, l, u), dislocationSystem.fcc(cs)
#   dS.isEdge, dS.isScrew, dS.tensor(), ori * dS
#
# A Burgers direction with a plane normal, or with a line vector. They are in crystal
# coordinates exactly while Miller indexed, and in specimen coordinates once an
# orientation has rotated them; the frame decides, never the group. A list of systems is
# one object over arrays, as MATLAB has it.
from __future__ import annotations

import numpy as np

from .frames import crystalFrame
from .miller import Miller, indexChar
from .symmetryops import symmetrise as _symmetrise
from .vector3d import vector3d

# the families: slip directions and plane normals, three or four indices
SLIP_FAMILIES = {
  "primitiveCubic": [((1, 0, 0), (0, 1, 0))],
  "fcc": [((0, 1, -1), (1, 1, 1))],
  "bcc": [((1, -1, 1), (0, 1, 1)), ((-1, 1, 1), (2, 1, 1)), ((-1, 1, 1), (3, 2, 1))],
  "basal": [((1, 1, -2, 0), (0, 0, 0, 1))],
  "prismaticA": [((2, -1, -1, 0), (0, 1, -1, 0))],
  "prismatic2A": [((0, 1, -1, 0), (2, -1, -1, 0))],
  "pyramidalA": [((2, -1, -1, 0), (0, 1, -1, 1))],
  "pyramidalCA": [((2, -1, -1, 3), (-1, 1, 0, 1))],
  "pyramidal2CA": [((2, -1, -1, 3), (-2, 1, 1, 2))],
  "twinT1": [((1, -1, 0, 1), (-1, 1, 0, 2))],
  "twinT2": [((2, -1, -1, 6), (-2, 1, 1, 1))],
  "twinC1": [((-1, 1, 0, -2), (-1, 1, 0, 1))],
  "twinC2": [((2, -1, -1, -3), (2, -1, -1, 2))],
}


def _unit(a):
  n = np.linalg.norm(a, axis=-1, keepdims=True)
  return a / np.where(n == 0, 1, n)


def _asList(v):
  v = vector3d(v) if not isinstance(v, vector3d) else v
  return v.reshape(-1) if v.ndim != 1 else v


# a vector as its Miller indices where it has a crystal frame, as coordinates otherwise
def _char(v):
  if isinstance(v.frame, crystalFrame):
    return indexChar(v)
  return "(" + ", ".join(f"{x:.3g}" for x in v.data.reshape(-1)) + ")"


def _sameFrame(a, b, what):
  if a.frame is not b.frame and (a.frame is not None and b.frame is not None):
    raise ValueError(f"{what} are given in different frames")


def _familyOf(family, cs, CRSS=1):
  if family == "hcp":
    names = sorted(k for k in SLIP_FAMILIES if k not in ("primitiveCubic", "fcc", "bcc"))
    raise ValueError("there is no hcp default: choose among " + ", ".join(names) + " and give each its CRSS")
  if family not in SLIP_FAMILIES:
    raise ValueError(f"unknown slip family {family!r}")
  bs, ns = zip(*SLIP_FAMILIES[family])
  direct = "uvw" if len(bs[0]) == 3 else "UVTW"
  recip = "hkl" if len(ns[0]) == 3 else "hkil"
  return slipSystem(Miller(np.array(bs, dtype=float), cs, direct), Miller(np.array(ns, dtype=float), cs, recip), CRSS)


# ------------------------------------------------------------------ slip systems

class slipSystem:

  def __init__(self, b, n, CRSS=1):
    b, n = _asList(b), _asList(n)
    _sameFrame(b, n, "the slip direction and the plane normal")
    bd, nd = np.broadcast_arrays(b.data, n.data)
    if np.any(np.abs(np.sum(_unit(bd) * _unit(nd), axis=-1)) > 1e-5):
      raise ValueError("the slip direction and the plane normal are not orthogonal")
    self.b, self.n = vector3d(bd.copy(), frame=n.frame), vector3d(nd.copy(), frame=n.frame)
    self.b.dispStyle, self.n.dispStyle = b.dispStyle, n.dispStyle
    self.CRSS = np.broadcast_to(np.asarray(CRSS, dtype=float), (len(bd),)).copy()

  @classmethod
  def _new(cls, b, n, CRSS):
    s = object.__new__(cls)
    s.b, s.n, s.CRSS = b, n, np.asarray(CRSS, dtype=float)
    return s

  def __len__(self):
    return len(self.CRSS)

  @property
  def size(self):
    return len(self.CRSS)

  def __getitem__(self, key):
    sel = np.arange(len(self))[key].reshape(-1)
    return self._new(self.b[sel], self.n[sel], self.CRSS[sel])

  def __iter__(self):
    return (self[k] for k in range(len(self)))

  # the list of several systems of one crystal, MATLAB's [sS1, sS2]
  @classmethod
  def cat(cls, *systems, axis=0):
    systems = [s for s in systems if s is not None]
    for s in systems[1:]:
      _sameFrame(systems[0].b, s.b, "the slip systems")
    b = vector3d.cat(*[s.b for s in systems])
    n = vector3d.cat(*[s.n for s in systems])
    b.dispStyle, n.dispStyle = systems[0].b.dispStyle, systems[0].n.dispStyle
    return cls._new(b, n, np.concatenate([s.CRSS for s in systems]))

  @property
  def CS(self):
    return self.b.frame if isinstance(self.b.frame, crystalFrame) else None

  @property
  def isCrystal(self):
    return isinstance(self.b.frame, crystalFrame)

  @property
  def frame(self):
    return self.b.frame

  # the systems of every family: the pairs of the orbits that stay orthogonal, both signs of
  # the Burgers vector unless antipodal
  def symmetrise(self, antipodal=False):
    if not self.isCrystal:
      return self
    bs, ns, crss = [], [], []
    for k in range(len(self)):
      mm = _symmetrise(self.b[k], unique=True, antipodal=True).data.reshape(-1, 3)
      if not antipodal:
        mm = np.concatenate([mm, -mm])
      nn = _symmetrise(self.n[k], unique=True, antipodal=True).data.reshape(-1, 3)
      ok = np.abs(_unit(mm) @ _unit(nn).T) < 1e-7
      r, c = np.nonzero(ok)
      bs.append(mm[r])
      ns.append(nn[c])
      crss.append(np.full(len(r), self.CRSS[k]))
    b = vector3d(np.concatenate(bs), frame=self.b.frame)
    n = vector3d(np.concatenate(ns), frame=self.n.frame)
    b.dispStyle, n.dispStyle = self.b.dispStyle, self.n.dispStyle
    return self._new(b, n, np.concatenate(crss))

  # the Schmid factor cos(phi) cos(lambda) of tension directions, a (systems, directions)
  # matrix, in the coordinates the systems are written in
  def SchmidFactor(self, v, relative=False):
    v = _asList(v)
    _sameFrame(self.b, v, "the slip systems and the tension direction")
    r = _unit(v.data)
    m = (_unit(self.b.data) @ r.T) * (_unit(self.n.data) @ r.T)
    if relative:
      m = m / self.CRSS[:, None]
    return m[:, 0] if len(v) == 1 else m

  # the dyad b n^T of every system, and its symmetric part
  def deformationTensor(self):
    return _unit(self.b.data)[:, :, None] * _unit(self.n.data)[:, None, :]

  def SchmidTensor(self):
    M = self.deformationTensor()
    return (M + np.swapaxes(M, 1, 2)) / 2

  # the Luster-Morris parameter of two lists of systems and the residual Burgers vector
  def mPrime(self, other):
    return np.abs((_unit(self.b.data) @ _unit(other.b.data).T) * (_unit(self.n.data) @ _unit(other.n.data).T))

  def residualBurgersVector(self, other):
    return np.linalg.norm(self.b.data[:, None, :] - other.b.data[None, :, :], axis=-1)

  # the trace of the slip plane in the surface normal to a direction, the xy plane by default
  def trace(self, nrm=None):
    from . import zvector
    nrm = zvector if nrm is None else nrm
    t = self.n.cross(nrm).normalize()
    t.antipodal = True
    return t

  # an orientation or a rotation carries the systems into specimen coordinates
  def rotate(self, rot):
    return self._new(rot * self.b, rot * self.n, self.CRSS)

  def __rmul__(self, rot):
    return self.rotate(rot)

  def __neg__(self):
    return self._new(-self.b, self.n, self.CRSS)

  def __eq__(self, other):
    if not isinstance(other, slipSystem) or len(self) != len(other) or self.frame is not other.frame:
      return False
    same = np.all(np.isclose(self.b.data, other.b.data), axis=-1) & np.all(np.isclose(self.n.data, other.n.data), axis=-1)
    flipped = np.all(np.isclose(self.b.data, -other.b.data), axis=-1) & np.all(np.isclose(self.n.data, -other.n.data), axis=-1)
    return bool(np.all(same | flipped))

  def __repr__(self):
    head = f"slipSystem ({self.frame.char()})" if self.frame is not None else "slipSystem"
    lines = [head, f"  size: {len(self)}"]
    rows = min(len(self), 12)
    for k in range(rows):
      lines.append(f"  {_char(self.n[k])}{_char(self.b[k])}" + (f"  CRSS {self.CRSS[k]:g}" if self.CRSS[k] != 1 else ""))
    if len(self) > rows:
      lines.append(f"  ... {len(self) - rows} more")
    return "\n".join(lines)

  # ------------------------------------------------------------------ the families

  @staticmethod
  def primitiveCubic(cs, CRSS=1):
    return _familyOf("primitiveCubic", cs, CRSS)

  @staticmethod
  def fcc(cs, CRSS=1):
    return _familyOf("fcc", cs, CRSS)

  @staticmethod
  def bcc(cs, CRSS=1):
    return _familyOf("bcc", cs, CRSS)

  @staticmethod
  def hcp(cs=None, CRSS=1):
    return _familyOf("hcp", cs, CRSS)

  @staticmethod
  def basal(cs, CRSS=1):
    return _familyOf("basal", cs, CRSS)

  @staticmethod
  def prismaticA(cs, CRSS=1):
    return _familyOf("prismaticA", cs, CRSS)

  @staticmethod
  def prismatic2A(cs, CRSS=1):
    return _familyOf("prismatic2A", cs, CRSS)

  @staticmethod
  def pyramidalA(cs, CRSS=1):
    return _familyOf("pyramidalA", cs, CRSS)

  @staticmethod
  def pyramidalCA(cs, CRSS=1):
    return _familyOf("pyramidalCA", cs, CRSS)

  @staticmethod
  def pyramidal2CA(cs, CRSS=1):
    return _familyOf("pyramidal2CA", cs, CRSS)

  @staticmethod
  def twinT1(cs, CRSS=1):
    return _familyOf("twinT1", cs, CRSS)

  @staticmethod
  def twinT2(cs, CRSS=1):
    return _familyOf("twinT2", cs, CRSS)

  @staticmethod
  def twinC1(cs, CRSS=1):
    return _familyOf("twinC1", cs, CRSS)

  @staticmethod
  def twinC2(cs, CRSS=1):
    return _familyOf("twinC2", cs, CRSS)


# the Burgers vector of a slip system in lattice units
def burgersLength(cs):
  lat = cs.lattice
  if lat.isTriHex:
    return 1 / 3
  if lat.name == "cubic":
    return 1 / 2
  raise ValueError(f"the length of the Burgers vector of a {lat} lattice has to be given")


# ------------------------------------------------------------------ dislocation systems

# a Burgers vector with a line vector, edge when orthogonal and screw when parallel; from
# slip systems, the edge dislocations of every system with the energy 2 and the screw ones
# of every distinct Burgers vector with the energy 1
class dislocationSystem:

  def __init__(self, b, l=None, u=None):
    if isinstance(b, slipSystem):
      sS = b
      fr = sS.frame
      scale = burgersLength(fr) if isinstance(fr, crystalFrame) else 1.0
      bd = scale * sS.b.data
      ld = -np.cross(sS.n.data, bd)
      screws = []
      for v in bd:
        if not any(np.linalg.norm(v - w) < 1e-8 or np.linalg.norm(v + w) < 1e-8 for w in screws):
          screws.append(v)
      screws = np.array(screws).reshape(-1, 3)
      b = vector3d(np.concatenate([bd, screws]), frame=fr)
      l = vector3d(np.concatenate([ld, screws]), frame=fr, antipodal=True)
      u = np.concatenate([np.full(len(bd), 2.0), np.ones(len(screws))])
      b.dispStyle = l.dispStyle = sS.b.dispStyle
    else:
      b, l = _asList(b), _asList(l)
      _sameFrame(b, l, "the Burgers vector and the line vector")
      bd, ld = np.broadcast_arrays(b.data, l.data)
      cosang = np.abs(np.sum(_unit(bd) * _unit(ld), axis=-1))
      if np.any((cosang > 1e-5) & (cosang < 1 - 1e-5)):
        raise ValueError("the line vector is neither orthogonal nor parallel to the Burgers vector")
      b, l = vector3d(bd, frame=b.frame), vector3d(ld, frame=b.frame, antipodal=True)
      u = np.ones(len(bd)) if u is None else np.broadcast_to(np.asarray(u, dtype=float), (len(bd),)).copy()
    self.b, self.l, self.u = b, l, np.asarray(u, dtype=float)

  @classmethod
  def _new(cls, b, l, u):
    d = object.__new__(cls)
    d.b, d.l, d.u = b, l, np.asarray(u, dtype=float)
    return d

  def __len__(self):
    return len(self.u)

  @property
  def size(self):
    return len(self.u)

  def __getitem__(self, key):
    sel = np.arange(len(self))[key].reshape(-1)
    return self._new(self.b[sel], self.l[sel], self.u[sel])

  @property
  def CS(self):
    return self.b.frame if isinstance(self.b.frame, crystalFrame) else None

  @property
  def frame(self):
    return self.b.frame

  @property
  def isEdge(self):
    return np.abs(np.sum(_unit(self.b.data) * _unit(self.l.data), axis=-1)) < 1e-5

  @property
  def isScrew(self):
    return ~self.isEdge

  # the dislocation density tensor of every system, b (x) l^
  def tensor(self):
    return self.b.data[:, :, None] * _unit(self.l.data)[:, None, :]

  def rotate(self, rot):
    l = rot * self.l
    l.antipodal = True
    return self._new(rot * self.b, l, self.u)

  def __rmul__(self, rot):
    return self.rotate(rot)

  def __repr__(self):
    head = f"dislocationSystem ({self.frame.char()})" if self.frame is not None else "dislocationSystem"
    lines = [head, f"  size: {len(self)}, edge: {int(np.count_nonzero(self.isEdge))}, screw: {int(np.count_nonzero(self.isScrew))}"]
    return "\n".join(lines)

  @staticmethod
  def fcc(cs):
    return dislocationSystem(slipSystem.fcc(cs).symmetrise(antipodal=True))

  @staticmethod
  def bcc(cs):
    return dislocationSystem(slipSystem.bcc(cs).symmetrise(antipodal=True))

  @staticmethod
  def hcp(cs=None):
    return _familyOf("hcp", cs)


__all__ = ["slipSystem", "dislocationSystem"]
