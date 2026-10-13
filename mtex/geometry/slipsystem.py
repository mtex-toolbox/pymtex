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

from .representations import unitRows
from .frames import crystalFrame
from .display import display, header, numString, sizeString, table, takeRows
from .miller import Miller
from .symmetryops import symmetrise as _symmetrise
from .symreduce import _firstOfClusters
from .vector3d import vector3d, asVector3d
from .display import aligned

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


def _asList(v):
  v = vector3d(v) if not isinstance(v, vector3d) else v
  return v.reshape(-1) if v.ndim != 1 else v


def _sameFrame(a, b, what):
  if a.frame is not b.frame and (a.frame is not None and b.frame is not None):
    raise ValueError(f"{what} are given in different frames")


def _familyOf(family, CS, CRSS=1):
  if family == "hcp":
    names = sorted(k for k in SLIP_FAMILIES if k not in ("primitiveCubic", "fcc", "bcc"))
    raise ValueError("there is no hcp default: choose among " + ", ".join(names) + " and give each its CRSS")
  if family not in SLIP_FAMILIES:
    raise ValueError(f"unknown slip family {family!r}")
  bs, ns = zip(*SLIP_FAMILIES[family])
  direct = "uvw" if len(bs[0]) == 3 else "UVTW"
  recip = "hkl" if len(ns[0]) == 3 else "hkil"
  return slipSystem(Miller(np.array(bs, dtype=float), CS, direct), Miller(np.array(ns, dtype=float), CS, recip), CRSS)


# ------------------------------------------------------------------ slip systems

class slipSystem:
  """slip systems: slip directions b with plane normals n orthogonal to them and a critical resolved
  shear stress each, in crystal coordinates while Miller indexed, in specimen coordinates once
  an orientation has rotated them

    sS = slipSystem(b, n)
    sS = slipSystem(b, n, CRSS)
    sS = slipSystem.fcc(cs)"""

  def __init__(self, b, n, CRSS=1):
    b, n = _asList(b), _asList(n)
    _sameFrame(b, n, "the slip direction and the plane normal")
    bd, nd = np.broadcast_arrays(b.data, n.data)
    if np.any(np.abs(np.sum(unitRows(bd) * unitRows(nd), axis=-1)) > 1e-5):
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
    return self.CRSS.shape[0] if self.CRSS.ndim else 1

  @property
  def size(self):
    """the number of slip systems"""
    return self.CRSS.size

  @property
  def shape(self):
    """a list, or an array once a column of orientations has rotated a row of systems"""
    return self.CRSS.shape

  def __getitem__(self, key):
    if self.CRSS.ndim > 1:
      return self._new(self.b[key], self.n[key], self.CRSS[key])
    sel = np.arange(len(self))[key].reshape(-1)
    return self._new(self.b[sel], self.n[sel], self.CRSS[sel])

  def __iter__(self):
    return (self[k] for k in range(len(self)))

  @classmethod
  def cat(cls, *systems, axis=0):
    """the list of several systems of one crystal, MATLAB's [sS1, sS2]"""
    systems = [s for s in systems if s is not None]
    for s in systems[1:]:
      _sameFrame(systems[0].b, s.b, "the slip systems")
    b = vector3d.cat(*[s.b for s in systems])
    n = vector3d.cat(*[s.n for s in systems])
    b.dispStyle, n.dispStyle = systems[0].b.dispStyle, systems[0].n.dispStyle
    return cls._new(b, n, np.concatenate([s.CRSS for s in systems]))

  @property
  def CS(self):
    """the crystal frame of the systems, None once they are in specimen coordinates"""
    return self.b.frame if isinstance(self.b.frame, crystalFrame) else None

  @property
  def isCrystal(self):
    """whether the systems are in crystal coordinates"""
    return isinstance(self.b.frame, crystalFrame)

  @property
  def frame(self):
    """the frame the slip directions and plane normals are written in"""
    return self.b.frame

  def symmetrise(self, *, antipodal=False, return_index=False):
    """the systems of every family: the pairs of the orbits that stay orthogonal, both signs of
    the Burgers vector unless antipodal"""
    # return_index also gives the system every equivalent comes from, MATLAB's slipId
    if not self.isCrystal:
      return (self, np.arange(len(self))) if return_index else self
    bs, ns, crss = [], [], []
    for k in range(len(self)):
      mm = _symmetrise(self.b[k], unique=True, antipodal=True).data.reshape(-1, 3)
      if not antipodal:
        mm = np.concatenate([mm, -mm])
      nn = _symmetrise(self.n[k], unique=True, antipodal=True).data.reshape(-1, 3)
      ok = np.abs(unitRows(mm) @ unitRows(nn).T) < 1e-7
      r, c = np.nonzero(ok)
      bs.append(mm[r])
      ns.append(nn[c])
      crss.append(np.full(len(r), self.CRSS[k]))
    b = vector3d(np.concatenate(bs), frame=self.b.frame)
    n = vector3d(np.concatenate(ns), frame=self.n.frame)
    b.dispStyle, n.dispStyle = self.b.dispStyle, self.n.dispStyle
    out = self._new(b, n, np.concatenate(crss))
    return (out, np.repeat(np.arange(len(self)), [len(c) for c in crss])) if return_index else out

  def SchmidFactor(self, v, relative=False):
    """the Schmid factor cos(phi) cos(lambda) of tension directions in the coordinates the
    systems are written in, MATLAB's (directions, systems) matrix: the directions keep their
    shape, a single one gives one factor per system"""
    v = asVector3d(v)
    _sameFrame(self.b, v, "the slip systems and the tension direction")
    r = unitRows(v.data.reshape(-1, 3))
    m = (r @ unitRows(self.b.data).T) * (r @ unitRows(self.n.data).T)
    if relative:
      m = m / self.CRSS
    if v.size == 1:
      return float(m[0, 0]) if self.size == 1 else m[0]
    return m.reshape(v.shape + (len(self),))

  # the dyad b n^T of the normalised vectors, one (3, 3) matrix per system
  def _dyads(self):
    return unitRows(self.b.data)[..., :, None] * unitRows(self.n.data)[..., None, :]

  def deformationTensor(self):
    """the dyad as a tensor in the frame of the systems, a single one for a single system; the
    Schmid tensor is the same dyad, MATLAB's velocity gradient of a unit shear"""
    from ..tensors.tensor import tensor
    M = self._dyads()
    return tensor(M[0] if len(self) == 1 else M, self.frame, rank=2)

  def SchmidTensor(self):
    """the Schmid tensor b n^T of the normalised vectors as a velocity gradient tensor, one per system"""
    from ..tensors.tensor import velocityGradientTensor
    M = self._dyads()
    return velocityGradientTensor(M[0] if len(self) == 1 else M, self.frame)

  def mPrime(self, other):
    """the Luster-Morris parameter |(n1 . n2)(b1 . b2)| and the residual Burgers vector of paired
    systems, entry by entry as NumPy broadcasts, MATLAB's element by element"""
    return np.abs(np.sum(unitRows(self.b.data) * unitRows(other.b.data), axis=-1) * np.sum(unitRows(self.n.data) * unitRows(other.n.data), axis=-1))

  def residualBurgersVector(self, other):
    """the length of the difference of the Burgers vectors of paired systems, entry by entry"""
    return np.linalg.norm(self.b.data - other.b.data, axis=-1)

  def trace(self, nrm=None):
    """the trace of the slip plane in the surface normal to a direction, the xy plane by default"""
    from . import zvector
    nrm = zvector if nrm is None else nrm
    t = self.n.cross(nrm).normalize()
    t.antipodal = True
    return t

  def rotate(self, rot):
    """an orientation or a rotation carries the systems into specimen coordinates, broadcast as
    NumPy does: a column of orientations times a row of systems is MATLAB's outer ori * sS"""
    b, n = rot * self.b, rot * self.n
    return self._new(b, n, np.broadcast_to(self.CRSS, b.shape).copy())

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

  def char(self, style=None):
    """one plane and one direction in it, (111)[01-1], one string per system; char(sS, 'latex')
    as one piece of mathematics"""
    out = [(self.n.reshape(-1)[k].char(style, family=False) + self.b.reshape(-1)[k].char(style, family=False)).replace("$$", "")
           for k in range(self.size)]
    return out[0] if self.size == 1 else out

  # MATLAB's display: the Burgers vector and the plane normal in indices beside the CRSS, a
  # rotated system in specimen coordinates to two decimals
  def __repr__(self):
    pairs = [("size", sizeString(self.shape) if self.size != 1 else None)]
    if not self.size:
      return display(header("slipSystem", self.b._headerChar()), pairs)
    take, pos = takeRows(self.size)
    s = self[take] if len(self.shape) <= 1 else self[np.unravel_index(take, self.shape)]
    if self.isCrystal:
      if self.CS.lattice.isTriHex:
        labels, d = ["U", "V", "T", "W", "| H", "K", "I", "L", "CRSS"], [s.b.UVTW, s.n.hkil]
      else:
        labels, d = ["u", "v", "w", "| h", "k", "l", "CRSS"], [s.b.uvw, s.n.hkl]
      d = np.concatenate(d + [s.CRSS[..., None]], axis=-1).reshape(len(take), -1)
      body = table(labels, np.round(d, 4).T, decimals=4, tiny=1e-10, rows=pos)
    else:
      d = np.round(100 * np.concatenate([s.b.data, s.n.data], axis=-1).reshape(len(take), 6)) / 100
      body = table(["x", "y", "z", "|  x", "y", "z"], d.T, decimals=2, tiny=1e-10, rows=pos)
    return display(header("slipSystem", self.b._headerChar()), pairs, body)

  # ------------------------------------------------------------------ the families

  @staticmethod
  def primitiveCubic(CS, CRSS=1):
    """the slip system <100>{010} of a primitive cubic crystal"""
    return _familyOf("primitiveCubic", CS, CRSS)

  @staticmethod
  def fcc(CS, CRSS=1):
    """the slip system <01-1>{111} of a face-centred cubic crystal"""
    return _familyOf("fcc", CS, CRSS)

  @staticmethod
  def bcc(CS, CRSS=1):
    """the slip systems <1-11>{011}, <-111>{211} and <-111>{321} of a body-centred cubic crystal"""
    return _familyOf("bcc", CS, CRSS)

  @staticmethod
  def hcp(CS=None, CRSS=1):
    """there is no hcp default: raises an error naming the hexagonal families to choose from
    (basal, prismaticA, pyramidalCA, twinT1, ...) with their CRSS"""
    return _familyOf("hcp", CS, CRSS)

  @staticmethod
  def basal(CS, CRSS=1):
    """the basal slip system <11-20>(0001) of a hexagonal crystal"""
    return _familyOf("basal", CS, CRSS)

  @staticmethod
  def prismaticA(CS, CRSS=1):
    """the prismatic a slip system <2-1-10>{01-10} of a hexagonal crystal"""
    return _familyOf("prismaticA", CS, CRSS)

  @staticmethod
  def prismatic2A(CS, CRSS=1):
    """the second prismatic a slip system <01-10>{2-1-10} of a hexagonal crystal"""
    return _familyOf("prismatic2A", CS, CRSS)

  @staticmethod
  def pyramidalA(CS, CRSS=1):
    """the pyramidal a slip system <2-1-10>{01-11} of a hexagonal crystal"""
    return _familyOf("pyramidalA", CS, CRSS)

  @staticmethod
  def pyramidalCA(CS, CRSS=1):
    """the pyramidal c+a slip system <2-1-13>{-1101} of a hexagonal crystal"""
    return _familyOf("pyramidalCA", CS, CRSS)

  @staticmethod
  def pyramidal2CA(CS, CRSS=1):
    """the second order pyramidal c+a slip system <2-1-13>{-2112} of a hexagonal crystal"""
    return _familyOf("pyramidal2CA", CS, CRSS)

  @staticmethod
  def twinT1(CS, CRSS=1):
    """the tensile twin T1 <1-101>{-1102} of a hexagonal crystal"""
    return _familyOf("twinT1", CS, CRSS)

  @staticmethod
  def twinT2(CS, CRSS=1):
    """the tensile twin T2 <2-1-16>{-2111} of a hexagonal crystal"""
    return _familyOf("twinT2", CS, CRSS)

  @staticmethod
  def twinC1(CS, CRSS=1):
    """the compressive twin C1 <-110-2>{-1101} of a hexagonal crystal"""
    return _familyOf("twinC1", CS, CRSS)

  @staticmethod
  def twinC2(CS, CRSS=1):
    """the compressive twin C2 <2-1-1-3>{2-1-12} of a hexagonal crystal"""
    return _familyOf("twinC2", CS, CRSS)


def burgersLength(CS):
  """the Burgers vector of a slip system in units of its slip direction: a hexagonal direction
  [11-20] has the length a already, a cubic one is twice the Burgers vector"""
  lat = CS.lattice
  if lat.isTriHex:
    return 1.0
  if lat.name == "cubic":
    return 1 / 2
  raise ValueError(f"the length of the Burgers vector of a {lat} lattice has to be given")


# ------------------------------------------------------------------ dislocation systems

class dislocationSystem:
  """a Burgers vector with a line vector, edge when orthogonal and screw when parallel; from
  slip systems, the edge dislocations of every system and the screw ones of every distinct
  Burgers vector. The line energy u is |b|^2 for an edge and 0.7 |b|^2 for a screw
  dislocation unless given, the isotropic line energy with Poisson ratio 0.3"""

  def __init__(self, b, l=None, u=None):
    if isinstance(b, slipSystem):
      sS = b
      fr = sS.frame
      scale = burgersLength(fr) if isinstance(fr, crystalFrame) else 1.0
      bd = scale * sS.b.data
      ld = -np.cross(sS.n.data, bd)
      # one screw dislocation per Burgers vector up to sign
      flat = bd.reshape(-1, 3)
      same = ((np.linalg.norm(flat[:, None] - flat[None], axis=-1) < 1e-8)
              | (np.linalg.norm(flat[:, None] + flat[None], axis=-1) < 1e-8))
      screws = flat[_firstOfClusters(same)]
      b = vector3d(np.concatenate([bd, screws]), frame=fr)
      l = vector3d(np.concatenate([ld, screws]), frame=fr, antipodal=True)
      b.dispStyle = l.dispStyle = sS.b.dispStyle
    else:
      b, l = _asList(b), _asList(l)
      _sameFrame(b, l, "the Burgers vector and the line vector")
      bd, ld = np.broadcast_arrays(b.data, l.data)
      cosang = np.abs(np.sum(unitRows(bd) * unitRows(ld), axis=-1))
      if np.any((cosang > 1e-5) & (cosang < 1 - 1e-5)):
        raise ValueError("the line vector is neither orthogonal nor parallel to the Burgers vector")
      b, l = vector3d(bd, frame=b.frame), vector3d(ld, frame=b.frame, antipodal=True)
    self.b, self.l = b, l
    if u is None:
      u = b.norm() ** 2 * (1 - 0.3 * self.isScrew)
    self.u = np.broadcast_to(np.asarray(u, dtype=float), (len(b),)).copy()

  @classmethod
  def _new(cls, b, l, u):
    d = object.__new__(cls)
    d.b, d.l, d.u = b, l, np.asarray(u, dtype=float)
    return d

  def __len__(self):
    return self.u.shape[0] if self.u.ndim else 1

  @property
  def size(self):
    """the number of dislocation systems"""
    return self.u.size

  @property
  def shape(self):
    """a list, or an array once a column of orientations has rotated a row of systems"""
    return self.u.shape

  def __getitem__(self, key):
    if self.u.ndim > 1:
      return self._new(self.b[key], self.l[key], self.u[key])
    sel = np.arange(len(self))[key].reshape(-1)
    return self._new(self.b[sel], self.l[sel], self.u[sel])

  @property
  def CS(self):
    """the crystal frame of the systems, None once they are in specimen coordinates"""
    return self.b.frame if isinstance(self.b.frame, crystalFrame) else None

  @property
  def frame(self):
    """the frame the Burgers and line vectors are written in"""
    return self.b.frame

  @property
  def isEdge(self):
    """which systems are edge dislocations, the line vector orthogonal to the Burgers vector"""
    return np.abs(np.sum(unitRows(self.b.data) * unitRows(self.l.data), axis=-1)) < 1e-5

  @property
  def isScrew(self):
    """which systems are screw dislocations, the line vector parallel to the Burgers vector"""
    return ~self.isEdge

  def tensor(self):
    """the dislocation density tensor of every system, b (x) l^, in the length unit of the axes"""
    from ..tensors.tensor import dislocationDensityTensor
    M = self.b.data[..., :, None] * unitRows(self.l.data)[..., None, :]
    return dislocationDensityTensor(M[0] if self.size == 1 else M, self.frame, unit="au")

  def rotate(self, rot):
    """an orientation carries the systems into the specimen, broadcast as NumPy does: a column
    of orientations times a row of systems is MATLAB's outer ori * dS"""
    b, l = rot * self.b, rot * self.l
    l.antipodal = True
    return self._new(b, l, np.broadcast_to(self.u, b.shape).copy())

  def __rmul__(self, rot):
    return self.rotate(rot)

  # MATLAB's display: the edge and the screw dislocations each as a table of their Burgers
  # and line vectors, energies and lengths
  def __repr__(self):
    lines = ["dislocationSystem"]
    if self.CS is not None:
      lines.append(("  mineral: " if self.CS.mineral else "  symmetry: ") + self.CS.char())
    def names(v, sel):
      return ["[" + "  ".join(numString(x) for x in row) + "]" for row in np.round(v.coordinates[sel], 4)]
    for label, sel, withLine in (("edge dislocations : ", self.isEdge, True), ("screw dislocations: ", self.isScrew, False)):
      if not np.any(sel):
        continue
      lines.append(f"  {label}{int(np.count_nonzero(sel))}")
      if self.CS is None or self.u.ndim > 1:
        continue
      cols = [("Burgers vector", names(self.b, sel))] + ([("line vector", names(self.l, sel))] if withLine else [])
      cols += [("energy", [numString(x) for x in self.u[sel]]), ("length", [numString(x, decimals=2) for x in np.round(self.b[sel].norm(), 2)])]
      widths = [max(len(h), *(len(c) for c in col)) for h, col in cols]
      lines.append("  " + "  ".join(h.rjust(w) for (h, _), w in zip(cols, widths)))
      for row in zip(*(col for _, col in cols)):
        lines.append("  " + "  ".join(c.rjust(w) for c, w in zip(row, widths)))
    return aligned("\n".join(lines))

  @staticmethod
  def fcc(CS):
    """the edge and screw dislocations of the symmetrised face-centred cubic slip systems"""
    return dislocationSystem(slipSystem.fcc(CS).symmetrise(antipodal=True))

  @staticmethod
  def bcc(CS):
    """the edge and screw dislocations of the symmetrised body-centred cubic slip systems"""
    return dislocationSystem(slipSystem.bcc(CS).symmetrise(antipodal=True))

  @staticmethod
  def hcp(CS=None):
    """there is no hcp default: raises an error naming the hexagonal families to choose from"""
    return _familyOf("hcp", CS)


def mPrime(sS1, sS2):
  """mPrime(sS1, sS2) is sS1.mPrime(sS2): the Luster-Morris parameter |(n1 . n2)(b1 . b2)| of
  paired slip systems"""
  return sS1.mPrime(sS2)


__all__ = ["slipSystem", "dislocationSystem", "mPrime"]
