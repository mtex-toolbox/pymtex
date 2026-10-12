# crystal shapes (MATLAB @crystalShape and +crystalShape; Julia crystalshape.jl)
#
#   cS = crystalShape(N), crystalShape(N, habitus, extension)
#   cS = crystalShape.hex(cs), crystalShape.cube(cs), crystalShape.quartz(), crystalShape.olivine(cs)
#   cS.N, cS.V, cS.F, cS.E, cS.CS, cS.faceArea, cS.faceCenter, cS.volume, cS.diameter
#   ori * cS, 2 * cS, cS * s, cS + v, cS[Miller(1, 0, 0, cs)], cS[3], innerFace(cS, n)
#
# The polyhedron bounded by lattice planes: the intersection of the half spaces
# <v, n> <= 1 over the symmetrised face normals, every face at the reciprocal length of
# its normal, so that lengthening a normal moves its face inwards and shortening it can
# make it disappear (Enderlein 1997). The vertices are the triple plane intersections
# inside every half space, the faces are convex polygons ordered around their normals,
# the whole is scaled to a diameter of one. N keeps the forms as given, in the crystal
# frame, and V carries the frame the shape stands in: the crystal frame until an
# orientation takes it into the specimen. A list of shapes is one object over vertices of
# shape (n, nV) sharing the faces, which is what ori * cS, s * cS and cS + v make.
from __future__ import annotations

from typing import TYPE_CHECKING
from itertools import combinations

import numpy as np

from .representations import unitRows
from ._array import MtexArray
from .data.crystalshapes import CRYSTAL_SHAPES
from .frames import crystalFrame
from .miller import Miller
from .symmetryops import _images
from .vector3d import vector3d, asVector3d
from .install import install

# a face lies where dot(v, n) = 1, up to this
FACE_TOL = 1e-8
# a face is selected by a normal within this angle of its own
SELECT_ANGLE = np.pi / 180


# the first of every cluster of rows closer than tol
def _distinctRows(A, tol):
  keep = []
  for k in range(len(A)):
    if not keep or np.min(np.linalg.norm(A[keep] - A[k], axis=1)) > tol:
      keep.append(k)
  return np.array(keep, dtype=int)


def polyhedron(normals, tol=FACE_TOL):
  """the vertices and the faces of the intersection of the half spaces <v, n> <= 1: the
  faces as index lists ordered around their normals, and which normals bound a face"""
  normals = np.asarray(normals, dtype=float).reshape(-1, 3)
  n = len(normals)
  if n < 4:
    raise ValueError("a crystal shape needs at least four face normals to close")
  i, j, k = np.array(list(combinations(range(n), 3))).T
  A = np.stack([normals[i], normals[j], normals[k]], axis=1)
  lengths = np.linalg.norm(normals, axis=1)
  ok = np.abs(np.linalg.det(A)) > 1e-10 * lengths[i] * lengths[j] * lengths[k]
  V = np.linalg.solve(A[ok], np.ones((int(ok.sum()), 3, 1)))[:, :, 0]
  V = V[np.all(V @ normals.T <= 1 + tol, axis=1)]
  if len(V) == 0:
    raise ValueError("the half spaces of the face normals do not enclose a volume")
  V = V[_distinctRows(V, 1e-6 * np.max(np.linalg.norm(V, axis=1)))]
  faces, bounding = [], []
  for m in range(n):
    ids = np.nonzero(V @ normals[m] > 1 - tol)[0]
    if len(ids) < 3:
      continue
    d = normals[m] / lengths[m]
    u = vector3d(d).orth().data.reshape(3)
    w = np.cross(d, u)
    c = V[ids].mean(axis=0)
    ang = np.arctan2((V[ids] - c) @ w, (V[ids] - c) @ u)
    faces.append(ids[np.argsort(ang)])
    bounding.append(m)
  return V, faces, np.array(bounding, dtype=int)


# the faces as one int array padded with -1
def _padFaces(faces):
  width = max(len(f) for f in faces)
  F = np.full((len(faces), width), -1, dtype=int)
  for r, f in enumerate(faces):
    F[r, :len(f)] = f
  return F


class crystalShape:
  """a crystal shape, the polyhedron bounded by the lattice planes of the face normals N

    cS = crystalShape(N)                       faces at the reciprocal length of their normal
    cS = crystalShape(N, habitus, extension)   the distances from the indices alone
    cS = crystalShape.hex(cs), crystalShape.quartz(), crystalShape.olivine(cs, simple=True)
  """

  __array_ufunc__ = None

  def __init__(self, N, habitus=0, extension=(1, 1, 1)):
    N = asVector3d(N)
    if not N.isCrystalDirection:
      raise ValueError("the face normals of a crystal shape are Miller indices")
    self.N = N.flatten()
    self.habitus = float(habitus)
    self.extension = tuple(float(e) for e in np.asarray(extension, dtype=float).reshape(3))
    self._update()

  # the shape from its normals: the habitus scaling, the orbits as the distinct faces,
  # the polyhedron, the vertices scaled to a diameter of one
  def _update(self):
    xyz = self.N.data.reshape(-1, 3).copy()
    if self.habitus > 0:
      h = np.abs(self.N.hkl.reshape(-1, 3)) * np.asarray(self.extension)
      xyz = xyz / np.sum(h ** self.habitus, axis=1, keepdims=True) ** (1 / self.habitus)
    # the equivalents are the distinct faces from here on
    imgs = _images(self.N.frame, xyz)
    form = np.repeat(np.arange(len(xyz)), imgs.shape[1])
    imgs = imgs.reshape(-1, 3)
    keep = _distinctRows(imgs, 1e-8 * np.max(np.linalg.norm(imgs, axis=1)))
    normals, form = imgs[keep], form[keep]
    V, faces, bounding = polyhedron(normals)
    V = V / (2 * np.max(np.linalg.norm(V, axis=1)))
    self.V = vector3d(V, frame=self.N.frame, dispStyle="xyz")
    self.F = _padFaces(faces)
    self._faceN0 = unitRows(normals[bounding])
    self._form = form[bounding]

  # the same shape over other vertices, or other faces
  def _new(self, V, F=None, faceN0=None, form=None):
    s = object.__new__(crystalShape)
    s.N, s.habitus, s.extension = self.N, self.habitus, self.extension
    s.V = V
    s.F = self.F if F is None else F
    s._faceN0 = self._faceN0 if faceN0 is None else faceN0
    s._form = self._form if form is None else form
    return s

  # ------------------------------------------------------------------ the list

  # the number of shapes, one unless a list was made
  def __len__(self):
    return 1 if self.V.ndim == 1 else self.V.shape[0]

  @property
  def size(self):
    """the number of shapes, one unless the shape is a list of shapes"""
    return len(self)

  # the vertices as (n, nV, 3), a single shape as a list of one
  def _V(self):
    return self.V.data.reshape(-1, self.V.shape[-1], 3)

  # cS[N] selects the faces with those normals, cS[k] the k-th shape of a list
  def __getitem__(self, key):
    if isinstance(key, vector3d):
      return self.subSet(key)
    V = self.V.reshape(1, -1) if self.V.ndim == 1 else self.V
    return self._new(V[key])

  def __iter__(self):
    return (self[k] for k in range(len(self)))

  def repmat(self, n):
    """n copies of the shape as a list"""
    return self._new(vector3d(np.broadcast_to(self._V()[0], (int(n),) + self._V().shape[1:]).copy(), **self._attrs()))

  def _attrs(self):
    return dict(frame=self.V.frame, dispStyle="xyz")

  def subSet(self, NSelect):
    """the faces whose outward normal is within a degree of one of the given normals, in
    crystal coordinates as the shape was built"""
    NSelect = asVector3d(NSelect)
    sel = unitRows(NSelect.data.reshape(-1, 3))
    ind = np.any(self._faceN0 @ sel.T > np.cos(SELECT_ANGLE), axis=1)
    if not np.any(ind):
      raise ValueError("no face of the shape has one of the given normals")
    return self._new(self.V, self.F[ind], self._faceN0[ind], self._form[ind])

  # ------------------------------------------------------------------ the properties

  @property
  def CS(self):
    """the crystal frame of the face normals, the frame the shape was built in"""
    return self.N.CS

  @property
  def frame(self):
    """the frame the vertices stand in: the crystal frame, or the specimen frame once an
    orientation has turned the shape"""
    return self.V.frame

  @property
  def numFaces(self):
    """the number of faces of the shape"""
    return len(self.F)

  @property
  def numVertices(self):
    """the number of vertices of the shape"""
    return self.V.shape[-1]

  @property
  def E(self):
    """the edges as pairs of vertex indices, every edge once"""
    a, b = self.F, np.roll(self.F, -1, axis=1)
    last = np.sum(self.F >= 0, axis=1) - 1
    b[np.arange(len(self.F)), last] = self.F[:, 0]
    pairs = np.stack([a, b], axis=-1).reshape(-1, 2)
    pairs = np.sort(pairs[np.all(pairs >= 0, axis=1)], axis=1)
    return np.unique(pairs, axis=0)

  # the sum of the cross products around every face, twice its area along its normal
  def _faceVectors(self):
    V = self._V()
    out = np.zeros((V.shape[0], len(self.F), 3))
    for r, f in enumerate(self.F):
      f = f[f >= 0]
      P = V[:, f]
      out[:, r] = np.sum(np.cross(P, np.roll(P, -1, axis=1)), axis=1) / 2
    return out

  # the mean of the vertices of every face, per shape
  def _faceCenters(self):
    V = self._V()
    return np.stack([V[:, f[f >= 0]].mean(axis=1) for f in self.F], axis=1)

  def _single(self, a):
    return a[0] if self.V.ndim == 1 else a

  def faceNormal(self):
    """the outward unit normal of every face, where the shape stands"""
    return vector3d(self._single(unitRows(self._faceVectors())), **self._attrs())

  @property
  def faceArea(self):
    """the area of every face, per shape of a list"""
    return self._single(np.linalg.norm(self._faceVectors(), axis=-1))

  @property
  def faceCenter(self):
    """the centre of every face, the mean of its vertices, per shape of a list"""
    return vector3d(self._single(self._faceCenters()), **self._attrs())

  @property
  def volume(self):
    """a third of the face areas times their distance from the centre"""
    V = self._V()
    c = V.mean(axis=1)[:, None, :]
    fv = self._faceVectors()
    fc = self._faceCenters()
    return self._single(np.sum(np.sum((fc - c) * unitRows(fv), axis=-1) * np.linalg.norm(fv, axis=-1), axis=1) / 3)

  @property
  def diameter(self):
    """the largest distance between two vertices"""
    V = self._V()
    return self._single(np.max(np.linalg.norm(V[:, :, None] - V[:, None, :], axis=-1), axis=(1, 2)))

  @diameter.setter
  def diameter(self, d):
    """sets the diameter, the largest distance between two vertices, by scaling the shape"""
    self.V = self.V * (np.asarray(d, dtype=float) / self.diameter)

  # ------------------------------------------------------------------ moving the shape

  # the shift or the scale of every shape of a list, shaped to broadcast over the vertices
  def _perShape(self, s):
    s = np.asarray(s, dtype=float)
    return s.reshape(-1, 1) if s.size > 1 else s.reshape(())

  # s * cS scales, one factor for all or one per shape
  def __mul__(self, s):
    if isinstance(s, MtexArray):
      return NotImplemented
    return self._new(self.V * self._perShape(s))

  def __rmul__(self, s):
    if isinstance(s, MtexArray):
      return self.rotate(s)
    return self.__mul__(s)

  def __truediv__(self, s):
    return self.__mul__(1 / np.asarray(s, dtype=float))

  # cS + v moves the shape, one shift for all or one per shape; a two column array lies in
  # the plane z = 0
  def __add__(self, v):
    if not isinstance(v, vector3d):
      v = np.asarray(v, dtype=float)
      if v.ndim >= 1 and v.shape[-1] == 2:
        v = np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], axis=-1)
      v = vector3d(v)
    v = v.reshape(-1, 1) if v.size > 1 else v
    return self._new(self.V + v)

  __radd__ = __add__

  def __sub__(self, v):
    return self + (-v if isinstance(v, vector3d) else -np.asarray(v, dtype=float))

  def rotate(self, rot):
    """a rotation turns the shape where it is; an orientation takes it into its specimen; a
    list of them makes a list of shapes"""
    rot = rot.reshape(-1, 1) if rot.size > 1 else rot
    return self._new(rot * self.V)

  # ------------------------------------------------------------------ sections

  # a unit direction in the frame the shape stands in: a crystal direction is taken as
  # the shape was built, a specimen one once the shape stands in the specimen
  def _direction(self, v):
    if isinstance(v, vector3d):
      d = v.data.reshape(3)
      if v.frame is not None and self.V.frame is not None and v.frame is not self.V.frame:
        from .products import fitFrame
        t = fitFrame(v.frame, self.V.frame)
        if t is not None:
          d = (t * vector3d(d)).data.reshape(3)
    else:
      d = np.asarray(v, dtype=float).reshape(3)
    return d / np.linalg.norm(d)

  def exitDistance(self, d):
    """how far the ray from the centre along d runs before it leaves the shape"""
    if len(self) > 1:
      raise ValueError("the exit distance of a list of shapes: take one, cS[k]")
    d = self._direction(d)
    V = self.V.data.reshape(-1, 3)
    c = V.mean(axis=0)
    n = self.faceNormal().data.reshape(-1, 3)
    h = np.sum((self.faceCenter.data.reshape(-1, 3) - c) * n, axis=1)
    along = n @ d
    ok = along > 1e-9
    return float(np.min(h[ok] / along[ok]))

  def innerFace(self, n):
    """the section of the shape by the plane through its centre with the normal n: the
    vertices of the polygon in order, in the frame the shape stands in"""
    if len(self) > 1:
      raise ValueError("the inner face of a list of shapes: take one, cS[k]")
    d = self._direction(n)
    V = self.V.data.reshape(-1, 3)
    c = V.mean(axis=0)
    h = (V - c) @ d
    scale = np.max(np.abs(V - c))
    # a vertex on the plane is a point of the section, and no crossing of its edges
    h[np.abs(h) < 1e-9 * scale] = 0
    a, b = self.E.T
    a, b = a[h[a] * h[b] < 0], b[h[a] * h[b] < 0]
    crossings = V[a] + (V[b] - V[a]) * (h[a] / (h[a] - h[b]))[:, None]
    pts = np.concatenate([V[h == 0], crossings]).reshape(-1, 3)
    if len(pts) >= 3:
      pts = pts[_distinctRows(pts, 1e-9 * scale)]
      m = pts.mean(axis=0)
      u = vector3d(d).orth().data.reshape(3)
      w = np.cross(d, u)
      pts = pts[np.argsort(np.arctan2((pts - m) @ w, (pts - m) @ u))]
    return vector3d(pts, **self._attrs())

  # ------------------------------------------------------------------ display

  def __repr__(self):
    from .display import display
    cs = self.CS
    return display("crystalShape", [
      ("size", len(self) if len(self) > 1 else None),
      ("mineral", f"{cs.mineral} ({cs.pointGroup})" if cs.mineral else None), ("symmetry", None if cs.mineral else cs.pointGroup),
      ("frame", (self.V.frame.char() if self.V.frame is not None else "none") if self.V.frame is not cs else None),
      ("vertices", self.numVertices), ("faces", self.numFaces),
      ("habitus", f"{self.habitus:g}, extension: {', '.join(f'{e:g}' for e in self.extension)}" if self.habitus > 0 else None)])

  # ------------------------------------------------------------------ the simple shapes

  @staticmethod
  def cube(CS):
    """a cube, the faces {100}"""
    return crystalShape(Miller(1, 0, 0, CS))

  @staticmethod
  def hex(CS):
    """a hexagonal prism, the faces {10-10} and (0002)"""
    return crystalShape(Miller([[1, 0, 0], [0, 0, 2]], CS))

  @staticmethod
  def tetr(CS):
    """a tetragonal prism, the faces {100} and {001}"""
    return crystalShape(Miller([[1, 0, 0], [0, 0, 1]], CS))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    from ..plotting.shapeplot import plotShape as plot, plotInnerFace, plotInnerDirection, plotSlipSystem
    @staticmethod
    def albite(CS=None, simple=False) -> Any:
      """the albite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def amphibole(CS=None, simple=False) -> Any:
      """the amphibole crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def analcime(CS=None, simple=False) -> Any:
      """the analcime crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def andalusite(CS=None, simple=False) -> Any:
      """the andalusite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def anorthite(CS=None, simple=False) -> Any:
      """the anorthite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def antigorite(CS=None, simple=False) -> Any:
      """the antigorite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def apatite(CS=None, simple=False) -> Any:
      """the apatite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def aragonite(CS=None, simple=False) -> Any:
      """the aragonite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def augite(CS=None, simple=False) -> Any:
      """the augite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def biotite(CS=None, simple=False) -> Any:
      """the biotite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def brucite(CS=None, simple=False) -> Any:
      """the brucite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def calcite(CS=None, simple=False) -> Any:
      """the calcite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def chromite(CS=None, simple=False) -> Any:
      """the chromite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def clinochlore(CS=None, simple=False) -> Any:
      """the clinochlore crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def cordierite(CS=None, simple=False) -> Any:
      """the cordierite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def corundum(CS=None, simple=False) -> Any:
      """the corundum crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def diamond(CS=None, simple=False) -> Any:
      """the diamond crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def diopside(CS=None, simple=False) -> Any:
      """the diopside crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def dolomite(CS=None, simple=False) -> Any:
      """the dolomite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def enstatite(CS=None, simple=False) -> Any:
      """the enstatite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def epidote(CS=None, simple=False) -> Any:
      """the epidote crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def fluorite(CS=None, simple=False) -> Any:
      """the fluorite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def fluoriteComplex(CS=None, simple=False) -> Any:
      """the fluorite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def forsterite(CS=None, simple=False) -> Any:
      """the forsterite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def garnet(CS=None, simple=False) -> Any:
      """the garnet crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def gypsum(CS=None, simple=False) -> Any:
      """the gypsum crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def halite(CS=None, simple=False) -> Any:
      """the halite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def ice_VI(CS=None, simple=False) -> Any:
      """the ice VI crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def ilmenite(CS=None, simple=False) -> Any:
      """the ilmenite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def kyanite(CS=None, simple=False) -> Any:
      """the kyanite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def magnesite(CS=None, simple=False) -> Any:
      """the magnesite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def magnetite(CS=None, simple=False) -> Any:
      """the magnetite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def microcline(CS=None, simple=False) -> Any:
      """the microcline crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def monazite(CS=None, simple=False) -> Any:
      """the monazite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def muscovite(CS=None, simple=False) -> Any:
      """the muscovite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def olivine(CS=None, simple=False) -> Any:
      """the olivine crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def orthoclase(CS=None, simple=False) -> Any:
      """the orthoclase crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def plagioclase(CS=None, simple=False) -> Any:
      """the Plagioclase crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def quartz(CS=None, simple=False) -> Any:
      """the Quartz crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def rutile(CS=None, simple=False) -> Any:
      """the rutile crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def spinel(CS=None, simple=False) -> Any:
      """the spinel crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def staurolite(CS=None, simple=False) -> Any:
      """the staurolite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def titanite(CS=None, simple=False) -> Any:
      """the titanite crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def topaz(CS=None, simple=False) -> Any:
      """the topaz crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def tourmaline(CS=None, simple=False) -> Any:
      """the tourmaline crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def zircon(CS=None, simple=False) -> Any:
      """the zircon crystal shape of the catalogue, in cs when given"""
    @staticmethod
    def zoisite(CS=None, simple=False) -> Any:
      """the zoisite crystal shape of the catalogue, in cs when given"""


# the frame the catalogue writes a shape in
def _catalogueFrame(e):
  from .units import degree
  return crystalFrame(e["symbol"], list(e["abc"]), np.asarray(e["abg"], dtype=float) * degree, *e["alignment"], mineral=e["mineral"])


def catalogueShape(name, CS=None, simple=False):
  """a shape of the catalogue: its Miller indices in the given frame, which has to be of
  the catalogue's Laue class, or in the catalogue's own"""
  suffix = "_simple" if simple else ""
  key = name + suffix if name + suffix in CRYSTAL_SHAPES else name.lower() + suffix
  if key not in CRYSTAL_SHAPES:
    names = ", ".join(sorted(k for k in CRYSTAL_SHAPES if not k.endswith("_simple")))
    raise ValueError(f"no crystal shape named {name!r}" + (" with a simple variant" if simple else "") + f"; the catalogue has {names}")
  e = CRYSTAL_SHAPES[key]
  own = _catalogueFrame(e)
  if CS is None:
    CS = own
  elif CS.Laue().id != own.Laue().id:
    raise ValueError(f"the catalogue's {name} is of Laue class {own.LaueName()}, the frame given of class {CS.LaueName()}")
  N = Miller(np.array(e["normals"], dtype=float), CS)
  if "distances" in e:
    return crystalShape(N / np.asarray(e["distances"], dtype=float))
  return crystalShape(N, e["habitus"], e["extension"])


# crystalShape.quartz(), crystalShape.olivine(cs), crystalShape.garnet(simple=True): every
# mineral of the catalogue as a constructor
def _catalogueConstructor(name):
  def make(CS=None, simple=False):
    if isinstance(CS, str):
      CS, simple = None, CS.lower() == "simple"
    return catalogueShape(name, CS, simple)
  make.__name__ = name
  make.__doc__ = f"the {CRYSTAL_SHAPES[name]['mineral']} crystal shape of the catalogue, in cs when given"
  return staticmethod(make)


for _name in CRYSTAL_SHAPES:
  if not _name.endswith("_simple"):
    install(crystalShape, _catalogueConstructor(_name))


def innerFace(cS, n):
  """innerFace(cS, n): the section of the shape by the plane through its centre"""
  return cS.innerFace(n)


__all__ = ["crystalShape", "innerFace", "CRYSTAL_SHAPES"]
