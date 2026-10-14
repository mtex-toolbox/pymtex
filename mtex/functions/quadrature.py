# quadrature on the circle, the sphere and SO(3): the rules on the line and the product
# grids exact to a degree (MATLAB quadratureSO3Grid, quadratureS2Grid; docs/plans/functions.md F2)
#
#   g = quadratureSO3Grid(L, CS, SS), quadratureSO3Grid(L, 'GaussLegendre'), quadratureS2Grid(L), quadratureS1Grid(L)
#   g.nodes(), g.weights, g.bandwidth, g.folds, len(g)
#
# A rule for degree L integrates every product of two basis functions of degree up to L, so
# the adjoint transform on it is exact: equispaced in the periodic angles, Clenshaw-Curtis
# (default) or Gauss-Legendre in the polar one. An r fold axis along z on one side of SO(3)
# makes the function periodic in that Euler angle with period 2pi/r, so that angle is
# sampled on one period only with the weights scaled by r; the adjoint then aliases the
# coefficients the symmetry forbids and the projection of H5 removes them. Clenshaw-Curtis
# places its polar nodes equispaced as well, which makes the whole grid a lattice and the
# transform on it one FFT instead of an NUFFT; it pays with 2L+1 polar nodes where
# Gauss-Legendre needs L+1.
from __future__ import annotations

import math
from functools import cached_property

import numpy as np

from ..geometry.frames import referenceFrame, specimenFrame
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.vector3d import vector3d
from ..geometry.display import aligned

SCHEMES = ("ClenshawCurtis", "GaussLegendre")


def gaussLegendre(n):
  """nodes and weights of the n point Gauss-Legendre rule on [-1, 1]; scipy's Newton iteration
  on the asymptotic nodes, since NumPy's leggauss solves a dense eigenproblem, and scipy is
  loaded on first use"""
  from scipy.special import roots_legendre
  return roots_legendre(n)


def clenshawCurtis(n):
  """Clenshaw-Curtis nodes cos(pi b/(n-1)), b = 0..n-1, and weights exact to degree n-1"""
  if n == 1:
    return np.zeros(1), np.full(1, 2.0)
  N = n - 1
  b = np.arange(N + 1)
  k = np.arange(1, N // 2 + 1)
  bk = np.where(2 * k == N, 1.0, 2.0)
  acc = 1 - np.sum(bk[:, None] / (4 * k[:, None] ** 2 - 1) * np.cos(2 * np.outer(k, b) * np.pi / N), axis=0)
  w = np.where((b == 0) | (b == N), 1.0, 2.0) / N * acc
  return np.cos(np.pi * b / N), w


def legendreP(L, x):
  """the Legendre polynomials P_0(x) .. P_L(x) by their recurrence, in front of the shape of x"""
  x = np.asarray(x)
  x = x if x.dtype.kind == "f" else x.astype(float)
  P = np.empty((L + 1, *x.shape), dtype=x.dtype)
  P[0] = 1
  if L > 0:
    P[1] = x
  for n in range(1, L):
    P[n + 1] = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1)
  return P


def _polarRule(scheme, L):
  if scheme == "ClenshawCurtis":
    return clenshawCurtis(2 * L + 1)
  if scheme == "GaussLegendre":
    return gaussLegendre(L + 1)
  raise ValueError(f"the scheme is 'ClenshawCurtis' or 'GaussLegendre', not '{scheme}'")


def folds(CS, SS=None):
  """the z folds on the crystal and on the specimen side: the order of the rotation about z
  that leaves the frame's group invariant"""
  SS = specimenFrame.default if SS is None else SS
  return (CS.sym.multiplicityZ, SS.sym.multiplicityZ)


def bandwidthFor(L, rA, rB=1):
  """the smallest bandwidth at least L whose 2L+2 the folds divide"""
  if isinstance(rA, referenceFrame):
    rA, rB = folds(rA, rB if isinstance(rB, referenceFrame) else None)
  r = math.lcm(int(rA), int(rB))
  while (2 * L + 2) % r:
    L += 1
  return L


# the frames and the scheme out of the positional arguments, MATLAB style
def _readArgs(args):
  frames = [a for a in args if isinstance(a, referenceFrame)]
  schemes = [a for a in args if isinstance(a, str)]
  if len(schemes) > 1 or len(frames) > 2 or len(frames) + len(schemes) != len(args):
    raise TypeError("quadrature grids take up to two frames and one scheme name")
  return frames, schemes[0] if schemes else "ClenshawCurtis"


class quadratureSO3Grid:
  """the quadrature grid on SO(3) of a bandwidth: the nodes and weights that integrate every
  harmonic function up to that degree exactly

    G = quadratureSO3Grid(L)                                        # Clenshaw-Curtis in beta, no symmetry
    G = quadratureSO3Grid(bandwidthFor(L, cs), cs, ss, 'GaussLegendre')  # Gauss-Legendre, reduced by the frames
    G = quadratureSO3Grid(15, rA=4, rB=2)                           # reduced by fourfold and twofold folds about z
    ori, w = G.nodes(), G.weights                                   # the nodes and their weights

  The Euler angles alpha and gamma are equispaced, 2L + 2 of each, and beta follows a
  Clenshaw-Curtis rule, the default, or a Gauss-Legendre rule. The frames' rotations about z
  shorten the alpha range by the specimen fold rB and the gamma range by the crystal fold rA,
  which must divide 2L + 2: bandwidthFor(L, cs, ss) gives the next bandwidth that they divide.
  The weights sum to one, the Haar measure. Unlike MATLAB's, the grid is not itself an
  orientation: G.nodes() gives the orientations, beta fastest, and G.weights the weights.

  Input
    bandwidth - int, the degree up to which the rule is exact
    CS, SS - crystalFrame and specimenFrame (positional, optional)
    scheme - 'ClenshawCurtis' (default) or 'GaussLegendre' (positional, optional)
  Output
    G - quadratureSO3Grid
  Keywords
    rA - the fold about z on the crystal side, without frames
    rB - the fold about z on the specimen side, without frames
    dtype - the floating point type of the angles and weights
  See also
    bandwidthFor, SO3FunHarmonic.quadrature, regularSO3Grid, SO3FunQuadrature"""

  # the full rule, or the one for functions with an rA fold axis along z on the crystal side
  # (periodic in gamma) and an rB fold on the specimen side (periodic in alpha); 2L+2 must
  # be divisible by the folds, which bandwidthFor arranges
  def __init__(self, bandwidth, *args, rA=None, rB=None, dtype=np.float64):
    frames, self.scheme = _readArgs(args)
    self.CS = frames[0] if frames else None
    self.SS = frames[1] if len(frames) > 1 else None
    if self.CS is not None:
      rA, rB = folds(self.CS, self.SS)
    rA, rB = int(1 if rA is None else rA), int(1 if rB is None else rB)
    L = int(bandwidth)
    m = 2 * L + 2
    if m % rA or m % rB:
      raise ValueError(f"2L+2 = {m} is not divisible by the folds {rA} and {rB}; use bandwidthFor")
    x, w = _polarRule(self.scheme, L)
    self.bandwidth, self.folds = L, (rA, rB)
    self.alpha = (2 * np.pi * np.arange(m // rB) / m).astype(dtype)
    self.gamma = (2 * np.pi * np.arange(m // rA) / m).astype(dtype)
    self.beta = np.arccos(np.clip(x, -1, 1)).astype(dtype)
    self.wbeta = (w * (rA * rB) / 2 / m ** 2).astype(dtype)

  @property
  def shape(self):
    """the shape of the grid, the numbers of gamma, alpha and beta nodes"""
    return (len(self.gamma), len(self.alpha), len(self.beta))

  def __len__(self):
    return math.prod(self.shape)

  def nodes(self):
    """the nodes as orientations in the frames of the grid, beta fastest, then alpha, then gamma"""
    return self._nodes(self.beta)

  def _nodes(self, beta):
    from ..geometry.kernels import heapBlocks
    from .kernels import gridQuaternions
    # the array of every call the same size: from the heap, its pages mapped from the last one
    with heapBlocks():
      q = gridQuaternions(self.alpha, beta, self.gamma)
    if self.CS is None:
      return rotation._fromData(q)
    return orientation._fromRotation(q, (self.CS, self.SS if self.SS is not None else specimenFrame.default))

  @property
  def weights(self):
    """the quadrature weights of the nodes, in the order of nodes(); they sum to one, the Haar measure"""
    return np.broadcast_to(self.wbeta, self.shape).ravel()

  @property
  def fullGrid(self):
    """MATLAB's view of the grid: the nodes that differ under the symmetries, on which an
    evaluator need be called, and the full grid; two nodes are one when their symmetric
    equivalents coincide, the largest equivalent quaternion up to sign the key"""
    return self.nodes()

  # the first of the nodes equal under the symmetries and the index of every node's one
  @cached_property
  def _unique(self):
    q = self.nodes().data.reshape(-1, 4)
    if self.CS is None:
      return np.arange(len(q)), np.arange(len(q))
    a = self.CS.sym.properGroup().rotations().data.reshape(-1, 4)
    b = (self.SS if self.SS is not None else specimenFrame.default).sym.properGroup().rotations().data.reshape(-1, 4)
    from ..geometry.kernels import _qmulNumpy
    eq = _qmulNumpy(_qmulNumpy(b[None, :, None, :], q[:, None, None, :]), a[None, None, :, :]).reshape(len(q), -1, 4)
    # q and -q are one rotation: the first component that does not vanish is positive
    lead = np.take_along_axis(eq, np.argmax(np.abs(eq) > 1e-9, axis=-1)[..., None], axis=-1)
    key = np.round(eq * np.where(lead < 0, -1.0, 1.0), 8)
    # the largest equivalent in lexicographic order: the equivalents largest in the first
    # component, among them in the second, and so on
    top = np.ones(key.shape[:2], dtype=bool)
    for c in range(4):
      col = np.where(top, key[..., c], -np.inf)
      top &= col == col.max(axis=1, keepdims=True)
    canon = key[np.arange(len(q)), np.argmax(top, axis=1)]
    _, first, inverse = np.unique(canon, axis=0, return_index=True, return_inverse=True)
    index = np.sort(first)
    return index, np.searchsorted(index, first)[inverse.ravel()]

  @property
  def size(self):
    """the number of the distinct nodes, MATLAB's numel(grid)"""
    return len(self._unique[0])

  def flatten(self):
    """the distinct nodes, MATLAB's grid(:)"""
    return self.nodes().flatten()[self._unique[0]]

  def fullValues(self, values):
    """values at the distinct nodes spread onto the full grid, values at the full grid kept"""
    values = np.asarray(values)
    # values at every node need no distinct nodes, which take seconds to find
    if values.shape[0] == len(self):
      return values
    index, inverse = self._unique
    if values.shape[0] == len(index) and len(index) != len(self):
      return values[inverse]
    return values

  # a 2-fold axis perpendicular to z at the angle psi from x maps the grid onto itself: on the
  # crystal side (alpha, beta, gamma) -> (alpha + pi, pi - beta, -gamma - 2 psi), on the specimen
  # side -> (2 psi - alpha, pi - beta, gamma + pi), both nodes of the grid where 2 psi is a multiple
  # of its step. The image of the nodes beta < pi/2 is then the rest of the grid but the equator,
  # so a function invariant under the groups need only be evaluated on the half beta <= pi/2.
  # The index maps of the gamma and the alpha axis, or None
  @cached_property
  def _mirror(self):
    if self.CS is None:
      return None
    m = 2 * self.bandwidth + 2
    Ng, Na, _ = self.shape
    c, a = np.arange(Ng), np.arange(Na)
    for side, frame in (("crystal", self.CS), ("specimen", self.SS if self.SS is not None else specimenFrame.default)):
      q = frame.sym.properGroup().rotations().data.reshape(-1, 4)
      # the 2-fold axes perpendicular to z by 2 psi in steps of the grid, the first on the grid
      axes = q[(np.abs(q[:, 0]) < 1e-9) & (np.abs(q[:, 3]) < 1e-9)]
      steps = np.unique(np.round(np.arctan2(axes[:, 2], axes[:, 1]) % np.pi * m / np.pi, 6))
      s = next((int(x) for x in steps if x.is_integer()), None)
      if s is not None:
        return ((-c - s) % Ng, (a + m // 2) % Na) if side == "crystal" else ((c + m // 2) % Ng, (s - a) % Na)
    return None

  def halfNodes(self):
    """the nodes beta <= pi/2 as orientations, where a 2-fold axis perpendicular to z on either side
    maps them onto the rest of the grid; fromHalf spreads the values of an invariant function there
    onto all nodes. None where the groups have no such axis"""
    if self._mirror is None:
      return None
    nb = len(self.beta)
    return self._nodes(self.beta[:(nb + 1) // 2])

  def fromHalf(self, values):
    """values at halfNodes() spread onto all nodes, in the order of nodes(); the nodes beta > pi/2
    take the values of their images under the 2-fold axis"""
    cmap, amap = self._mirror
    Ng, Na, nb = self.shape
    h = (nb + 1) // 2
    values = np.asarray(values)
    V = values.reshape((Ng, Na, h) + values.shape[1:])
    full = np.empty((Ng, Na, nb) + values.shape[1:], dtype=values.dtype)
    full[:, :, :h] = V
    full[:, :, h:] = V[cmap][:, amap][:, :, nb - h - 1::-1]
    return full.reshape((len(self),) + values.shape[1:])

  def __repr__(self):
    return aligned(f"quadratureSO3Grid ({self.scheme})\n  bandwidth: {self.bandwidth}\n  nodes: {len(self)}\n  folds: {self.folds}")


class quadratureS2Grid(vector3d):
  """the nodes of a quadrature rule on the sphere, a list of directions as in MATLAB, rho
  slowest and theta fastest over the axes thetaGrid and rhoGrid, with its weights"""

  def __init__(self, bandwidth, *args, dtype=np.float64):
    frames, self.scheme = _readArgs(args)
    L = int(bandwidth)
    m = 2 * L + 2
    x, w = _polarRule(self.scheme, L)
    self.bandwidth = L
    self.thetaGrid = np.arccos(np.clip(x, -1, 1)).astype(dtype)
    self.rhoGrid = (2 * np.pi * np.arange(m) / m).astype(dtype)
    self.wtheta = (w * 2 * np.pi / m).astype(dtype)
    r, t = np.meshgrid(self.rhoGrid, self.thetaGrid, indexing="ij")
    xyz = np.stack([np.sin(t) * np.cos(r), np.sin(t) * np.sin(r), np.cos(t)], axis=-1).reshape(-1, 3)
    vector3d.__init__(self, xyz, isNormalized=True, frame=frames[0] if frames else None)

  # anything derived from the grid is a plain vector3d
  def _new(self, data, **attrs):
    obj = object.__new__(vector3d)
    obj.data = data
    for name in self._carried:
      setattr(obj, name, attrs.pop(name, getattr(self, name)))
    return obj

  @property
  def gridShape(self):
    """the shape of the nodes over the axes, rho by theta"""
    return (len(self.rhoGrid), len(self.thetaGrid))

  def nodes(self):
    """the nodes as a plain list of directions"""
    return self._new(self.data)

  @property
  def weights(self):
    """the weights integrate to 4 pi, the surface measure of the sphere"""
    return np.broadcast_to(self.wtheta, self.gridShape).ravel()

  def __repr__(self):
    return aligned(f"quadratureS2Grid ({self.scheme})\n  bandwidth: {self.bandwidth}\n  nodes: {len(self)}")


class quadratureS1Grid:
  """the circle: 2L+2 equispaced nodes with equal weights integrate degree 2L+1 exactly"""

  def __init__(self, bandwidth, dtype=np.float64):
    self.bandwidth = int(bandwidth)
    self.x = (2 * np.pi * np.arange(2 * self.bandwidth + 2) / (2 * self.bandwidth + 2)).astype(dtype)

  def __len__(self):
    return len(self.x)

  def nodes(self):
    """the nodes, 2L+2 equispaced angles in [0, 2 pi)"""
    return self.x

  @property
  def weights(self):
    """the quadrature weights, all equal and summing to one"""
    return np.full(len(self.x), 1 / len(self.x), dtype=self.x.dtype)

  def __repr__(self):
    return aligned(f"quadratureS1Grid\n  bandwidth: {self.bandwidth}\n  nodes: {len(self)}")


__all__ = ["gaussLegendre", "clenshawCurtis", "legendreP", "folds", "bandwidthFor", "quadratureSO3Grid",
           "quadratureS2Grid", "quadratureS1Grid"]
