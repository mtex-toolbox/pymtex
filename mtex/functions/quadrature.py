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

import numpy as np

from ..geometry.frames import referenceFrame, specimenFrame
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.vector3d import vector3d

SCHEMES = ("ClenshawCurtis", "GaussLegendre")


# nodes and weights of the n point Gauss-Legendre rule on [-1, 1]; scipy's Newton iteration
# on the asymptotic nodes, since NumPy's leggauss solves a dense eigenproblem, and scipy is
# loaded on first use
def gaussLegendre(n):
  from scipy.special import roots_legendre
  return roots_legendre(n)


# Clenshaw-Curtis nodes cos(pi b/(n-1)), b = 0..n-1, and weights exact to degree n-1
def clenshawCurtis(n):
  if n == 1:
    return np.zeros(1), np.full(1, 2.0)
  N = n - 1
  b = np.arange(N + 1)
  k = np.arange(1, N // 2 + 1)
  bk = np.where(2 * k == N, 1.0, 2.0)
  acc = 1 - np.sum(bk[:, None] / (4 * k[:, None] ** 2 - 1) * np.cos(2 * np.outer(k, b) * np.pi / N), axis=0)
  w = np.where((b == 0) | (b == N), 1.0, 2.0) / N * acc
  return np.cos(np.pi * b / N), w


# the Legendre polynomials P_0(x) .. P_L(x) by their recurrence, in front of the shape of x
def legendreP(L, x):
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


# the z folds on the crystal and on the specimen side: the order of the rotation about z
# that leaves the frame's group invariant
def folds(CS, SS=None):
  SS = specimenFrame.default if SS is None else SS
  return (CS.sym.multiplicityZ, SS.sym.multiplicityZ)


# the smallest bandwidth at least L whose 2L+2 the folds divide
def bandwidthFor(L, rA, rB=1):
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
    return (len(self.gamma), len(self.alpha), len(self.beta))

  def __len__(self):
    return math.prod(self.shape)

  # the nodes as orientations in the frames of the grid, beta fastest, then alpha, then gamma
  def nodes(self):
    g, a, b = np.meshgrid(self.gamma, self.alpha, self.beta, indexing="ij")
    r = rotation.byEuler(a.ravel(), b.ravel(), g.ravel(), "Matthies")
    return r if self.CS is None else orientation(r, self.CS, self.SS if self.SS is not None else specimenFrame.default)

  @property
  def weights(self):
    return np.broadcast_to(self.wbeta, self.shape).ravel()

  def __repr__(self):
    return f"quadratureSO3Grid ({self.scheme})\n  bandwidth: {self.bandwidth}\n  nodes: {len(self)}\n  folds: {self.folds}"


class quadratureS2Grid:

  def __init__(self, bandwidth, *args, dtype=np.float64):
    frames, self.scheme = _readArgs(args)
    self.CS = frames[0] if frames else None
    L = int(bandwidth)
    m = 2 * L + 2
    x, w = _polarRule(self.scheme, L)
    self.bandwidth = L
    self.theta = np.arccos(np.clip(x, -1, 1)).astype(dtype)
    self.rho = (2 * np.pi * np.arange(m) / m).astype(dtype)
    self.wtheta = (w * 2 * np.pi / m).astype(dtype)

  @property
  def shape(self):
    return (len(self.rho), len(self.theta))

  def __len__(self):
    return math.prod(self.shape)

  # the nodes as directions, theta fastest
  def nodes(self):
    r, t = np.meshgrid(self.rho, self.theta, indexing="ij")
    return vector3d(np.sin(t) * np.cos(r), np.sin(t) * np.sin(r), np.cos(t)).flatten()

  # the weights integrate to 4 pi, the surface measure of the sphere
  @property
  def weights(self):
    return np.broadcast_to(self.wtheta, self.shape).ravel()

  def __repr__(self):
    return f"quadratureS2Grid ({self.scheme})\n  bandwidth: {self.bandwidth}\n  nodes: {len(self)}"


# the circle: 2L+2 equispaced nodes with equal weights integrate degree 2L+1 exactly
class quadratureS1Grid:

  def __init__(self, bandwidth, dtype=np.float64):
    self.bandwidth = int(bandwidth)
    self.x = (2 * np.pi * np.arange(2 * self.bandwidth + 2) / (2 * self.bandwidth + 2)).astype(dtype)

  def __len__(self):
    return len(self.x)

  def nodes(self):
    return self.x

  @property
  def weights(self):
    return np.full(len(self.x), 1 / len(self.x), dtype=self.x.dtype)

  def __repr__(self):
    return f"quadratureS1Grid\n  bandwidth: {self.bandwidth}\n  nodes: {len(self)}"


__all__ = ["gaussLegendre", "clenshawCurtis", "legendreP", "folds", "bandwidthFor", "quadratureSO3Grid",
           "quadratureS2Grid", "quadratureS1Grid"]
