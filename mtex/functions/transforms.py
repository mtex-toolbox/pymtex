# the fast evaluation of harmonic series, docs/plans/functions.md F2: a Wigner series is a
# trivariate Fourier series in the Euler angles and a spherical harmonic series a bivariate
# one (H2, H4, H7), so the coefficient transform below followed by a NUFFT (finufft) evaluates
# both at arbitrary nodes and the adjoint transform recovers coefficients from weighted
# values; on a Clenshaw-Curtis quadrature grid the NUFFT is one FFT, and below a few dozen
# nodes the direct sums of F0 win
#
#   G = wignerTransform(fhat, L, real=, folds=)          the Fourier array, axes (l, k, j)
#   fhat = wignerTransformAdjoint(H, L, real=, folds=)
#   G = sphericalTransform(fhat, L, real=)               axes (m, j)
#   p = SO3FourierPlan(rots, L), p(fhat), p.adjoint(values, weights); S2FourierPlan, S1FourierPlan
#   p = SO3GridPlan(g, L), S2GridPlan(g, L)              the FFT on a Clenshaw-Curtis grid
#   wignerSeries(fhat, rots | grid), wignerSeriesAdjoint(rots | grid, values, L, weights=)
#   sphericalSeries, sphericalSeriesAdjoint, fourierSeries, fourierSeriesAdjoint
#
# The hot loops of the two Wigner transforms are numba loops with NumPy forms as fallback
# and oracle, selected by the preference 'kernels' as in the geometry.
from __future__ import annotations

import math

import numpy as np

from ..geometry._array import MtexArray
from ..geometry.kernels import backend, heapBlocks, passiveOpenMP, threads
from ..geometry.session import getMTEXpref
from ..geometry.quaternion import orthogonalTransform, quaternion
from ..geometry.representations import quat2euler
from ..geometry.vector3d import vector3d
from .kernels import (wignerSeriesDirect, halfPiQuadrants, realDefectSpherical, realDefectWigner, s2Nodes, so3Nodes, sphericalAdjoint,
                      sphericalForward, wignerAdjoint, wignerForward)
from .nfft import NFFTPlan, nfftLibrary
from .harmonics import _WignerD, deg2dim, degreeBlock, dim2deg, _sphericalY, wignerdDegrees
from .quadrature import bandwidthFor, quadratureS1Grid, quadratureS2Grid, quadratureSO3Grid
from ..geometry.display import display, sizeString

DIRECT_SUM_NODES = 50
# the compiled direct series on SO(3) beats a plan up to here, in nodes, at any bandwidth from 32
DIRECT_SERIES_NODES = 100
# on the sphere a plan costs more than a trigonometric sum on the Fourier array up to here,
# in nodes, or in nodes times the (2L+1)^2 terms of the sum; the sum is exact, the plan
# only to its tolerance
DIRECT_TRIG_NODES_S2 = 400
DIRECT_TRIG_WORK_S2 = 2e6
# on the circle the direct sum costs about 27 ns a term, nodes times 2L+1, and a plan 0.45 ms
DIRECT_SUM_WORK_S1 = 2e4
NUFFT_EPS = 1e-8          # finufft's tolerance in double precision
NUFFT_EPS_SINGLE = 1e-5   # and in single, where 1e-8 is out of reach


def _finufft():
  passiveOpenMP()
  import finufft
  return finufft


def _complexType(dtype):
  return np.result_type(np.dtype(dtype), 1j)


def _eps(eps, dtype):
  return eps if eps is not None else NUFFT_EPS if np.dtype(dtype).itemsize == 8 else NUFFT_EPS_SINGLE


# the options of a plan: one thread below a work of nodes times modes, where finufft's
# thread start-up costs more than the transform (30 ms against 2 ms on 100 nodes at
# bandwidth 32), else the threads of the preference; the oversampling factor finufft's own
# choice unless a caller names one: over nodes from 1e4 to 1e7 and lattices from 17 x 129^2 to
# 129 x 257^2 it is within the noise of the better of 1.25 and 2 but for one case 20 % off,
# where a rule by the density of the nodes, 1.25 below a quarter of the lattice, took the
# slower one up to three times over (1e6 nodes on the 4-fold lattice of m-3m: 296 ms, 356 at
# 1.25, 795 at 2)
NUFFT_SINGLE_THREAD_WORK = 2e8


def _options(nodes, dims, eps, upsampfac=None):
  size = int(np.prod(dims))
  opts = {"nthreads": 1 if nodes * size < NUFFT_SINGLE_THREAD_WORK else threads()}
  if upsampfac is not None:
    opts["upsampfac"] = float(upsampfac)
  return opts


# ------------------------------------------------------------------ the Wigner-d matrices at pi/2


# every coefficient transform walks the degrees 0..L in order and needs the matrix of one
# degree at a time. Holding all of them is 4L^3/3 numbers, 173 MB at L = 256, so the table is
# kept up to that bandwidth and streamed by the recursion above it; both yield (n, d_n)
HALF_PI_CACHE_MAX = 256
_HALF_PI_TABLES = {}


def halfPiDegrees(L, dtype=np.float64):
  """the Wigner-d matrices d_0(pi/2) .. d_L(pi/2) as pairs (n, d_n), held up to bandwidth 256 and
  streamed by the recursion above"""
  dtype = np.dtype(dtype)
  if L > HALF_PI_CACHE_MAX:
    return wignerdDegrees(dtype.type(np.pi / 2), L)
  table = _HALF_PI_TABLES.get(dtype.name)
  if table is None or len(table) < L + 1:
    table = [D for _, D in wignerdDegrees(dtype.type(np.pi / 2), max(L, min(64, HALF_PI_CACHE_MAX)))]
    _HALF_PI_TABLES[dtype.name] = table
  return enumerate(table[:L + 1])


# ------------------------------------------------------------------ the Wigner coefficient transform
# G[l, k, j] = sum_n sqrt(2n+1) i^{k-l} fhat_n^{kl} d_n^{jk}(pi/2) d_n^{jl}(pi/2), so that
# f(alpha, beta, gamma) = sum G e^{-i(k alpha + j beta + l gamma)}; for a real valued function
# the array is kept for l >= 0 only, with the plane l = 0 halved, and f = 2 Re of the half sum.
# A function with an rA fold about z on the crystal side and an rB fold on the specimen side
# has coefficients only at l = 0 mod rA and k = 0 mod rB, so the two periodic axes are indexed
# by l/rA and k/rB and shrink by the folds exactly


def foldedDims(L, real, folds):
  """the shape (l, k, j) of the Fourier array of bandwidth L, the two periodic axes shrunk by the
  z folds, the axis l halved for a real function"""
  rA, rB = folds
  return (L // rA + 1 if real else 2 * (L // rA) + 1, 2 * (L // rB) + 1, 2 * L + 1)


# the folds a NUFFT array can carry: a dimension too thin for the window costs accuracy
NUFFT_MIN_DIM = 4


def usableFolds(L, real, folds):
  """the z folds a NUFFT array of bandwidth L can carry, (1, 1) when a folded axis would be too thin"""
  d = foldedDims(L, real, folds)
  return (int(folds[0]), int(folds[1])) if min(d[0], d[1]) >= NUFFT_MIN_DIM else (1, 1)


# the phases sqrt(2n+1) i^{k-l} of one degree block, the plane l = 0 halved for a real function
def _blockPhase(n, real, dtype):
  k = np.arange(-n, n + 1)
  P = math.sqrt(2 * n + 1) * 1j ** ((k[:, None] - k[None, :]) % 4)
  if real:
    P[:, n] /= 2
  return P.astype(_complexType(dtype))


# one degree into the array, j >= 0 only, the NumPy form
def _wignerForwardNumpy(G, C, d, n, off, real, rA, rB, Kb, La):
  la, kb = n // rA, n // rB
  lp = np.arange(0 if real else -la, la + 1)
  kp = np.arange(-kb, kb + 1)
  Csub = C[np.ix_(kp * rB + n, lp * rA + n)]
  G[np.ix_(lp if real else lp + La, kp + Kb, np.arange(off + n, off + 2 * n + 1))] += \
    np.einsum("jk,jl,kl->lkj", d[n:, kp * rB + n], d[n:, lp * rA + n], Csub)


# one degree block out of the array, the sum over j of d d h[-j], h[j] at the index c + j
def _wignerAdjointNumpy(F, H, d, n, c, real, rA, rB, Kb, La):
  la, kb = n // rA, n // rB
  lp = np.arange(0 if real else -la, la + 1)
  kp = np.arange(-kb, kb + 1)
  Hsub = H[np.ix_(lp if real else lp + La, kp + Kb, np.arange(c + n, c - n - 1, -1))]
  F[np.ix_(kp * rB + n, lp * rA + n)] = np.einsum("jk,jl,lkj->kl", d[:, kp * rB + n], d[:, lp * rA + n], Hsub)


# the sign of the j mirror: G[l, k, -j] = (-1)^{k+l} G[l, k, j]
def _mirrorSign(L, real, folds):
  rA, rB = folds
  La, Kb = L // rA, L // rB
  l = (np.arange(La + 1) if real else np.arange(-La, La + 1)) * rA
  k = np.arange(-Kb, Kb + 1) * rB
  return np.where((l[:, None] + k[None, :]) % 2, -1.0, 1.0)


def wignerTransform(fhat, L, real=False, folds=(1, 1), degrees=None):
  """the compiled kernels of mtex/functions/kernels.py on the quadrant table, or the NumPy form
  degree by degree on the full matrices of halfPiDegrees or of the degrees given"""
  cdtype = _complexType(np.asarray(fhat).dtype)
  fhat = np.ascontiguousarray(fhat, dtype=cdtype)
  rdtype = np.dtype(cdtype).type(0).real.dtype
  if deg2dim(L + 1) > fhat.shape[-1]:
    raise ValueError(f"the coefficients end before degree {L}")
  if backend() == "numba" and degrees is None:
    G = np.empty(foldedDims(L, real, folds), dtype=cdtype)
    wignerForward(G, fhat, halfPiQuadrants(L, rdtype), L, real, folds)
    return G
  rA, rB = folds
  La, Kb = L // rA, L // rB
  G = np.zeros(foldedDims(L, real, folds), dtype=cdtype)
  for n, dn in (halfPiDegrees(L, rdtype) if degrees is None else degrees):
    F = degreeBlock(fhat, n)
    if not np.any(F):
      continue
    C = np.ascontiguousarray(F * _blockPhase(n, real, rdtype), dtype=cdtype)
    _wignerForwardNumpy(G, C, np.ascontiguousarray(dn[:2 * n + 1, :2 * n + 1]), n, L - n, real, rA, rB, Kb, La)
  G[:, :, :L] = _mirrorSign(L, real, folds)[:, :, None] * G[:, :, L + 1:][:, :, ::-1]
  return G


def wignerTransformAdjoint(H, L, real=False, folds=(1, 1), degrees=None, centre=None):
  """the adjoint: fhat_n^{kl} = sqrt(2n+1) i^{k-l} sum_j d_n^{jk} d_n^{jl} h[l, k, -j] for
  h = sum_m w_m v_m e^{i(k alpha + j beta + l gamma)}; for real values only l >= 0 is given and
  the rest follows from fhat_n^{-k,-l} = (-1)^{k-l} conj(fhat_n^{kl}). The folds step l by rA
  and k by rB, since the projection of H5 zeroes the rest anyway. The order j = 0 sits at the
  index centre of the last axis, L unless given, 0 for the output of an FFT, whose negative
  orders wrap around"""
  centre = L if centre is None else int(centre)
  cdtype = H.dtype
  rdtype = np.dtype(cdtype).type(0).real.dtype
  fhat = np.zeros(deg2dim(L + 1), dtype=cdtype)
  # the compiled loop reads a view in any axis order, the NFFT's lattice as it is
  if backend() == "numba" and degrees is None:
    wignerAdjoint(fhat, H, halfPiQuadrants(L, rdtype), L, real, folds, centre)
    return fhat
  H = np.ascontiguousarray(H)
  rA, rB = folds
  La, Kb = L // rA, L // rB
  for n, dn in (halfPiDegrees(L, rdtype) if degrees is None else degrees):
    F = np.zeros((2 * n + 1, 2 * n + 1), dtype=cdtype)
    _wignerAdjointNumpy(F, H, np.ascontiguousarray(dn[:2 * n + 1, :2 * n + 1]), n, centre, real, rA, rB, Kb, La)
    F *= _blockPhase(n, False, rdtype)
    if real:
      k = np.arange(-n, n + 1)
      sign = np.where((k[:, None] - k[None, :]) % 2, -1.0, 1.0)
      F[:, :n] = (sign * np.conj(F))[::-1, ::-1][:, :n]
    degreeBlock(fhat, n)[...] = F
  return fhat


# ------------------------------------------------------------------ the spherical coefficient transform
# G[m, j] = sum_l sqrt((2l+1)/4pi) i^m fhat_l^m d_l^{jm}(pi/2) d_l^{j0}(pi/2), so that
# f(theta, rho) = sum G[m, j] e^{i(m rho - j theta)}; a real valued function has
# fhat_l^{-m} = (-1)^m conj(fhat_l^m), so the array is kept for m >= 0 with the column m = 0
# halved, f = 2 Re of the half sum; d_l^{j0}(pi/2) vanishes for odd l + j


def sphericalTransform(fhat, L, real=False, degrees=None):
  """the Fourier array G[m, j] of spherical coefficients of bandwidth L, so that f(theta, rho) is
  the sum of G e^{i(m rho - j theta)}; for a real function only m >= 0, the column m = 0 halved"""
  fhat = np.asarray(fhat)
  cdtype = _complexType(fhat.dtype)
  rdtype = np.dtype(cdtype).type(0).real.dtype
  if backend() == "numba" and degrees is None:
    return sphericalForward(np.ascontiguousarray(fhat[:(L + 1) ** 2], dtype=cdtype), L, real, cdtype)
  G = np.zeros((L + 1 if real else 2 * L + 1, 2 * L + 1), dtype=cdtype)
  for l, dl in (halfPiDegrees(L, rdtype) if degrees is None else degrees):
    m = np.arange(0 if real else -l, l + 1)
    c = fhat[l * l + l + m] * (math.sqrt((2 * l + 1) / (4 * np.pi)) * 1j ** (m % 4))
    if real:
      c[0] /= 2
    if not np.any(c):
      continue
    rows = m if real else m + L
    G[rows, L:L + l + 1] += c[:, None] * (dl[l:2 * l + 1, m + l] * dl[l:2 * l + 1, l:l + 1]).T
  m = np.arange(L + 1) if real else np.arange(-L, L + 1)
  G[:, :L] = np.where(m % 2, -1.0, 1.0)[:, None] * G[:, L + 1:][:, ::-1]
  return G


def sphericalTransformAdjoint(H, L, real=False, degrees=None):
  """the adjoint: fhat_l^m = sqrt((2l+1)/4pi) i^m sum_j d_l^{jm} d_l^{j0} h[m, -j] for
  h[m, j] = sum_q w_q v_q e^{-i(m rho_q - j theta_q)}"""
  H = np.asarray(H)
  if backend() == "numba" and degrees is None:
    return sphericalAdjoint(H, L, real)
  cdtype = H.dtype
  rdtype = np.dtype(cdtype).type(0).real.dtype
  fhat = np.zeros((L + 1) ** 2, dtype=cdtype)
  for l, dl in (halfPiDegrees(L, rdtype) if degrees is None else degrees):
    m = np.arange(0 if real else -l, l + 1)
    rows = m if real else m + L
    h = H[np.ix_(rows, np.arange(L + l, L - l - 1, -1))]
    acc = np.einsum("jm,mj->m", dl[:2 * l + 1, m + l] * dl[:2 * l + 1, l:l + 1], h)
    fhat[l * l + l + m] = acc * (math.sqrt((2 * l + 1) / (4 * np.pi)) * 1j ** (m % 4))
    if real and l:
      mm = np.arange(1, l + 1)
      fhat[l * l + l - mm] = np.where(mm % 2, -1.0, 1.0) * np.conj(fhat[l * l + l + mm])
  return fhat


# ------------------------------------------------------------------ real coefficient vectors


def isRealWigner(fhat, tol=1e-10):
  """a coefficient vector is real valued when fhat_n^{-k,-l} = (-1)^{k-l} conj(fhat_n^{kl}) on
  SO(3), fhat_l^{-m} = (-1)^m conj(fhat_l^m) on S2, c_{-k} = conj(c_k) on S1"""
  return realDefectWigner(fhat, tol)


def isRealSpherical(fhat, tol=1e-10):
  """whether spherical coefficients are those of a real function, fhat_l^{-m} = (-1)^m conj(fhat_l^m)"""
  return realDefectSpherical(fhat, tol)


def isRealFourier(chat, tol=1e-10):
  """whether Fourier coefficients on the circle are those of a real function, c_{-k} = conj(c_k)"""
  chat = np.asarray(chat)
  return bool(np.all(np.abs(chat[::-1] - np.conj(chat)) <= tol * np.maximum(1, np.abs(chat))))


# ------------------------------------------------------------------ the plans on arbitrary nodes


# the quaternions of a rotation container, flat
def _quaternions(rots):
  if isinstance(rots, orthogonalTransform):
    rots = rots.rot
  if not isinstance(rots, quaternion):
    raise TypeError("the nodes are a container of rotations")
  return rots.data.reshape(-1, 4)


# the Matthies angles of a rotation container as flat arrays, and their float type
def _euler(rots):
  q = _quaternions(rots)
  return quat2euler(q, "Matthies"), q.dtype


# the NUFFT of one node set, reusable across coefficient vectors: type 2 evaluates, type 1 is
# the adjoint, each built on its first use since most calls take one direction. The nodes
# follow setNodes; a node that is not finite evaluates to NaN and adds nothing to an adjoint
class _FourierPlan:

  # the axes of the adjoint's finufft plan, a permutation of those of the array: finufft deals its
  # sorted nodes out to the threads in slabs of its slowest axis, the first, and every slab is
  # spread into a box of the full other axes and the slab widened by the kernel. A short first
  # axis makes the boxes of all threads cover the grid several times over
  _adjointAxes = None

  def _init(self, n, dtype, dims, eps, upsampfac):
    self.dims, self.dtype = dims, _complexType(dtype)
    self.eps = _eps(eps, dtype)
    self.opts, self._plans = _options(n, dims, self.eps, upsampfac), {}

  # the dimensions and the nodes of the finufft plan of a kind
  def _axes(self, kind, x):
    return x if kind == 2 or self._adjointAxes is None else tuple(x[i] for i in self._adjointAxes)

  def _plan(self, kind):
    p = self._plans.get(kind)
    if p is None:
      p = _finufft().Plan(kind, self._axes(kind, self.dims), eps=self.eps, isign=-1 if kind == 2 else 1, dtype=self.dtype.name, **self.opts)
      p.setpts(*self._axes(kind, self.nodes))
      self._plans[kind] = p
    return p

  @property
  def forward(self):
    """the finufft plan of type 2 on the nodes, which evaluates; built on first use"""
    return self._plan(2)

  @property
  def backward(self):
    """the finufft plan of type 1 on the nodes, the adjoint; built on first use"""
    return self._plan(1)

  def _setNodes(self, nodes, phase, bad):
    self.nodes, self.phase, self.bad = nodes, phase, bad
    # the finufft plans move their points, an NFFT plan is built again
    self._plans = {kind: p for kind, p in self._plans.items() if isinstance(kind, int)}
    for kind, p in self._plans.items():
      p.setpts(*self._axes(kind, nodes))

  # the shift of the half array of a real function onto the centred index set of finufft
  @property
  def _shift(self):
    return self.dims[0] // 2 if self.real else 0

  def __len__(self):
    return len(self.phase)

  def _evaluate(self, G):
    with heapBlocks():
      f = self.forward.execute(G)
    f = 2 * np.real(self.phase * f) if self.real else f
    if self.bad is not None:
      f[self.bad] = np.nan
    return f

  def _spread(self, values, weights):
    u = np.asarray(weights).ravel() * values * np.conj(self.phase)
    if self.bad is not None:
      u[self.bad] = 0
    with heapBlocks():
      H = self.backward.execute(np.ascontiguousarray(u, dtype=self.dtype))
    return H if self._adjointAxes is None else H.transpose(np.argsort(self._adjointAxes))

  # the NFFT of ../nfft (mtex/functions/nfft.py) where its library is there, at the default
  # tolerance its windows reach: every transform under the preference NUFFT = 'nfft', none
  # under 'finufft', and under 'auto' the SO(3) adjoint of fewer nodes than lattice points,
  # where its pruned FFT counts; with more the spreading does and finufft's is faster
  # (1e7 nodes at bandwidth 25: 3.8 s against 1.4)
  def _useNFFT(self, direction):
    choice = getMTEXpref("NUFFT", "auto")
    if choice == "finufft" or self.dtype != np.complex128 or self.eps < NUFFT_EPS or nfftLibrary() is None:
      return False
    return choice == "nfft" or self._nfftDefault(direction)

  # one NFFT plan per value type serves both directions; the preference NFFTPrecompute keeps
  # the window values at the nodes (PRE_PSI)
  def _nfftPlan(self, real):
    p = self._plans.get(("nfft", real))
    if p is None:
      N, X = self._nfftLattice(real)
      p = self._plans[("nfft", real)] = NFFTPlan(N, X, real, keep=bool(getMTEXpref("NFFTPrecompute", False)))
    return p

  # the nodes as the NFFT reads them, the angles over 2 pi in [-1/2, 1/2)
  @staticmethod
  def _nfftNodes(*angles):
    X = np.stack(angles, axis=1) / (2 * np.pi)
    X -= X >= 0.5
    return X

  def _nfftEvaluate(self, G):
    p = self._nfftPlan(self.real)
    p.fhat[...] = 0
    self._toNFFT(p.fhat, G)
    f = p.trafo()
    f = f.real.copy() if self.real else f.copy()
    if self.bad is not None:
      f[self.bad] = np.nan
    return f

  def _nfftSpread(self, values, weights, real):
    u = np.asarray(weights).ravel() * values
    if self.bad is not None:
      u[self.bad] = 0
    p = self._nfftPlan(real)
    p.f[:] = u
    return self._fromNFFT(p.adjoint(), real)


class SO3FourierPlan(_FourierPlan):
  """on SO(3): folds says that the function is invariant under those z folds on the two sides:
  in the forward its coefficients outside them are zero, in the adjoint the projection of H5
  is about to zero them. Both directions then run on the reduced index set, where the two
  periodic axes of the array shrink by the folds and the nodes are stretched"""

  # the adjoint on the axes (k, j, l): the halved or folded axis l is the short one
  _adjointAxes = (1, 2, 0)

  def __init__(self, rots, L, real=True, folds=(1, 1), eps=None, upsampfac=None):
    q = _quaternions(rots)
    self.L, self.real, self.folds = int(L), bool(real), usableFolds(L, real, folds)
    self._init(len(q), q.dtype, foldedDims(self.L, self.real, self.folds), eps, upsampfac)
    self.setNodes(rots)

  def setNodes(self, rots):
    """moves the plan to other rotations, the finufft plans kept"""
    x, y, z, phase, bad = so3Nodes(_quaternions(rots), self.folds, self._shift)
    self._setNodes((x, y, z), phase, bad)

  def __call__(self, fhat):
    G = wignerTransform(np.asarray(fhat, dtype=self.dtype), self.L, self.real, self.folds)
    return self._nfftEvaluate(G) if self._useNFFT("forward") else self._evaluate(G)

  def adjoint(self, values, weights):
    """coefficients from weighted values at the nodes; the NFFT reads real values on half the
    lattice"""
    values = np.asarray(values).ravel()
    if self._useNFFT("adjoint"):
      real = values.dtype.kind != "c"
      return wignerTransformAdjoint(self._nfftSpread(values, weights, real), self.L, real, self.folds)
    # a real plan holds half the lattice, complex values are the sum of two real adjoints
    if self.real and values.dtype.kind == "c":
      return self.adjoint(values.real, weights) + 1j * self.adjoint(values.imag, weights)
    return wignerTransformAdjoint(self._spread(values, weights), self.L, self.real, self.folds)

  def _nfftDefault(self, direction):
    return direction == "adjoint" and len(self) < math.prod(self.dims)

  # the NFFT lattice has the axes (k, j, l), l last so that a real plan halves it, and every
  # frequency at its index + N/2: the first index of each axis is the unused -N/2
  def _nfftLattice(self, real):
    La, Kb = self.L // self.folds[0], self.L // self.folds[1]
    x, y, z = self.nodes
    return (2 * Kb + 2, 2 * self.L + 2, 2 * La + 2), self._nfftNodes(y, z, x)

  def _toNFFT(self, fh, G):
    (fh[1:, 1:, :] if self.real else fh[1:, 1:, 1:])[...] = G.transpose(1, 2, 0)

  def _fromNFFT(self, fh, real):
    return (fh[1:, 1:, :] if real else fh[1:, 1:, 1:]).transpose(2, 0, 1)

  def __repr__(self):
    return display("SO3FourierPlan", [("bandwidth", self.L), ("nodes", len(self.nodes[0]) if hasattr(self, "nodes") else None),
                                      ("real", self.real), ("folds", f"{self.folds[0]}, {self.folds[1]}")])


class S2FourierPlan(_FourierPlan):
  """the NUFFT plan of a spherical harmonic series at arbitrary directions v up to bandwidth L,
  reusable across coefficient vectors: plan(fhat) evaluates, plan.adjoint(values, weights)"""

  def __init__(self, v, L, real=False, eps=None, upsampfac=None):
    v = v if isinstance(v, vector3d) else vector3d(v)
    dtype = v.data.dtype if v.data.dtype.kind == "f" else np.float64
    self.L, self.real = int(L), bool(real)
    self._init(v.size, dtype, (self.L + 1 if self.real else 2 * self.L + 1, 2 * self.L + 1), eps, upsampfac)
    self.setNodes(v)

  def setNodes(self, v):
    """moves the plan to other directions, the finufft plans kept"""
    v = v if isinstance(v, vector3d) else vector3d(v)
    x, y, phase, bad = s2Nodes(v.data.astype(self.dtype.type(0).real.dtype, copy=False), self._shift)
    self._setNodes((x, y), phase, bad)

  def __call__(self, fhat):
    G = sphericalTransform(np.asarray(fhat, dtype=self.dtype), self.L, self.real)
    return self._nfftEvaluate(G) if self._useNFFT("forward") else self._evaluate(G)

  def adjoint(self, values, weights):
    """the spherical coefficients from weighted values at the nodes, the adjoint of the evaluation"""
    values = np.asarray(values).ravel()
    if self._useNFFT("adjoint"):
      real = values.dtype.kind != "c"
      return sphericalTransformAdjoint(self._nfftSpread(values, weights, real), self.L, real)
    if self.real and values.dtype.kind == "c":
      return self.adjoint(values.real, weights) + 1j * self.adjoint(values.imag, weights)
    return sphericalTransformAdjoint(self._spread(values, weights), self.L, self.real)

  def _nfftDefault(self, direction):
    return False

  # the NFFT lattice has the axes (j, m), m last so that a real plan halves it
  def _nfftLattice(self, real):
    x, y = self.nodes
    return (2 * self.L + 2, 2 * self.L + 2), self._nfftNodes(y, x)

  def _toNFFT(self, fh, G):
    (fh[1:, :] if self.real else fh[1:, 1:])[...] = G.T

  def _fromNFFT(self, fh, real):
    return (fh[1:, :] if real else fh[1:, 1:]).T

  def __repr__(self):
    return display("S2FourierPlan", [("bandwidth", self.L), ("nodes", len(self.nodes[0]) if hasattr(self, "nodes") else None), ("real", self.real)])


class S1FourierPlan:
  """the circle: f(x) = sum_{k=-L}^{L} c_k e^{ikx}"""

  def __init__(self, x, L, eps=None):
    x = np.asarray(x).ravel()
    dtype = x.dtype if x.dtype.kind == "f" else np.float64
    self.L, self.dtype = int(L), _complexType(dtype)
    x = np.ascontiguousarray(np.mod(x, 2 * np.pi), dtype=dtype)
    f, eps = _finufft(), _eps(eps, dtype)
    opts = _options(len(x), (2 * self.L + 1,), eps)
    self.forward = f.Plan(2, (2 * self.L + 1,), eps=eps, isign=1, dtype=self.dtype.name, **opts)
    self.forward.setpts(x)
    self.backward = f.Plan(1, (2 * self.L + 1,), eps=eps, isign=-1, dtype=self.dtype.name, **opts)
    self.backward.setpts(x)
    self.n = len(x)

  def __len__(self):
    return self.n

  def __call__(self, chat):
    return self.forward.execute(np.ascontiguousarray(chat, dtype=self.dtype))

  def adjoint(self, values, weights):
    """the Fourier coefficients c_{-L} .. c_L from weighted values at the nodes, the adjoint of the
    evaluation"""
    return self.backward.execute(np.ascontiguousarray(np.asarray(weights).ravel() * np.asarray(values).ravel(), dtype=self.dtype))

  def __repr__(self):
    return display("S1FourierPlan", [("bandwidth", self.L), ("nodes", self.n)])


def adjointTransform(plan, values, weights):
  """adjointTransform(plan, values, weights) is plan.adjoint(values, weights): the coefficients from
  weighted values at the plan's nodes"""
  return plan.adjoint(values, weights)


# ------------------------------------------------------------------ the plans on a Clenshaw-Curtis grid
# every angle of the grid is equispaced, so the series on it is a plain multivariate DFT.
# On SO(3) f(alpha, beta, gamma) = sum G[l, k, j] e^{-i(k alpha + j beta + l gamma)} with
# beta_b = pi b/(2Lg), b = 0..2Lg, alpha_a = 2pi a/(2Lg+2), gamma_c = 2pi c/(2Lg+2), Lg the
# grid's bandwidth: beta is a transform of length 4Lg cut to the grid's rows, alpha and gamma
# transforms of the grid's own lengths, which an r fold divides by r, the order k/r at its
# residue. The transforms run one axis at a time, each on the data it has: beta on the
# occupied planes (l, k), alpha on the rows of beta, gamma last, where a real function is
# 2 Re of the half sum and one real valued inverse FFT. The adjoint takes the axes backwards,
# each inverse FFT keeping the orders the next one reads


def _fft():
  import scipy.fft
  return scipy.fft


# a plan's placement buffers: written at the same indices every call, zero elsewhere for good
class _GridBuffers:

  def __init__(self, dtype):
    self.dtype, self.held = dtype, {}

  def __call__(self, name, shape):
    B = self.held.get(name)
    if B is None or B.shape != shape:
      B = self.held[name] = np.zeros(shape, dtype=self.dtype)
    return B


# the orders -r..r at their residues modulo n
def _residues(r, n):
  return np.mod(np.arange(-r, r + 1), n)


# the Fourier array's j axis, j at L + j, wrapped onto the polar length and back
def _wrapPolar(B, G, L):
  B[..., :L + 1] = G[..., L:]
  B[..., B.shape[-1] - L:] = G[..., :L]


def _unwrapPolar(X, L):
  return np.concatenate([X[..., X.shape[-1] - L:], X[..., :L + 1]], axis=-1)


class SO3GridPlan:
  """the plan of a Wigner series on a Clenshaw-Curtis quadrature grid of SO(3) up to bandwidth L, a
  sequence of FFTs one axis at a time: plan(fhat) evaluates, plan.adjoint(values, weights)"""

  def __init__(self, g, L=None, real=True):
    L = g.bandwidth if L is None else int(L)
    if g.scheme != "ClenshawCurtis":
      raise ValueError("the FFT route needs a Clenshaw-Curtis grid")
    if not 0 < L <= g.bandwidth:
      raise ValueError(f"a bandwidth {L} does not fit the grid of bandwidth {g.bandwidth}")
    self.L, self.real, self.folds, self.grid = L, bool(real), g.folds, g
    self.dtype = _complexType(g.beta.dtype)
    self.shape = g.shape
    self.polar = 4 * g.bandwidth
    rA, rB = self.folds
    self._k = _residues(L // rB, self.shape[1])
    self._l = _residues(L // rA, self.shape[0])
    self._buffers = _GridBuffers(self.dtype)

  def __len__(self):
    return len(self.grid)

  def __call__(self, fhat):
    fft, w, L = _fft(), threads(), self.L
    Ng, Na, nb = self.shape
    G = wignerTransform(np.asarray(fhat, dtype=self.dtype), L, self.real, self.folds)
    B = self._buffers("beta", G.shape[:2] + (self.polar,))
    _wrapPolar(B, G, L)
    A = self._buffers("alpha", (G.shape[0], Na, nb))
    A[:, self._k] = fft.fft(B, axis=2, workers=w)[..., :nb]
    Z = fft.fft(A, axis=1, workers=w)
    if self.real:
      Z = np.conj(Z, out=Z)
      Z[0] *= 2
      return fft.irfft(Z, n=Ng, axis=0, norm="forward", workers=w).ravel()
    C = self._buffers("gamma", (Ng, Na, nb))
    C[self._l] = Z
    return fft.fft(C, axis=0, workers=w).ravel()

  def adjoint(self, values, weights=None):
    """the SO(3) coefficients from weighted values on the grid, the FFTs taken backwards; without
    weights those of the grid's rule"""
    values = np.asarray(values).ravel()
    real = values.dtype.kind != "c"
    fft, w, L = _fft(), threads(), self.L
    x = values.reshape(self.shape) * (self.grid.wbeta if weights is None else np.asarray(weights).reshape(self.shape))
    if real:
      Z = np.conj(fft.rfft(x, axis=0, workers=w)[:L // self.folds[0] + 1])
    else:
      Z = fft.ifft(x, axis=0, norm="forward", workers=w)[self._l]
    W = fft.ifft(Z, axis=1, norm="forward", workers=w)[:, self._k]
    X = fft.ifft(W, n=self.polar, axis=2, norm="forward", workers=w)
    return wignerTransformAdjoint(X, L, real, self.folds, centre=0)

  def __repr__(self):
    return display("SO3GridPlan", [("bandwidth", self.L), ("grid", sizeString(self.shape)), ("real", self.real)])


class S2GridPlan:
  """the sphere: f(theta, rho) = sum G[m, j] e^{i(m rho - j theta)}, theta a transform of length
  4Lg cut to the grid's rows, rho one of the grid's length, real valued for a real function"""

  def __init__(self, g, L=None, real=False):
    L = g.bandwidth if L is None else int(L)
    if g.scheme != "ClenshawCurtis":
      raise ValueError("the FFT route needs a Clenshaw-Curtis grid")
    if not 0 < L <= g.bandwidth:
      raise ValueError(f"a bandwidth {L} does not fit the grid of bandwidth {g.bandwidth}")
    self.L, self.real, self.grid = L, bool(real), g
    self.dtype = _complexType(g.thetaGrid.dtype)
    self.shape = g.gridShape
    self.polar = 4 * g.bandwidth
    self._m = _residues(L, self.shape[0])
    self._buffers = _GridBuffers(self.dtype)

  def __len__(self):
    return len(self.grid)

  def __call__(self, fhat):
    fft, w, L = _fft(), threads(), self.L
    Nr, nt = self.shape
    G = sphericalTransform(np.asarray(fhat, dtype=self.dtype), L, self.real)
    B = self._buffers("theta", (G.shape[0], self.polar))
    _wrapPolar(B, G, L)
    Y = fft.fft(B, axis=1, workers=w)[:, :nt]
    if self.real:
      Y[0] *= 2
      return fft.irfft(Y, n=Nr, axis=0, norm="forward", workers=w).ravel()
    C = self._buffers("rho", (Nr, nt))
    C[self._m] = Y
    return fft.ifft(C, axis=0, norm="forward", workers=w).ravel()

  def adjoint(self, values, weights):
    """the spherical coefficients from weighted values on the grid, the FFTs taken backwards"""
    values = np.asarray(values).ravel()
    real = values.dtype.kind != "c"
    fft, w, L = _fft(), threads(), self.L
    x = (np.asarray(weights).ravel() * values).reshape(self.shape)
    Z = fft.rfft(x, axis=0, workers=w)[:L + 1] if real else fft.fft(x, axis=0, workers=w)[self._m]
    X = fft.ifft(Z, n=self.polar, axis=1, norm="forward", workers=w)
    return sphericalTransformAdjoint(_unwrapPolar(X, L), L, real)

  def __repr__(self):
    return display("S2GridPlan", [("bandwidth", self.L), ("grid", sizeString(self.shape)), ("real", self.real)])


def wignerPlan(nodes, L, real=True, folds=(1, 1)):
  """the plan for a node set: the NUFFT at arbitrary nodes, the FFT on a Clenshaw-Curtis grid,
  the NUFFT on the nodes of a grid whose polar rule is not equispaced"""
  if isinstance(nodes, quadratureSO3Grid):
    return SO3GridPlan(nodes, L, real) if nodes.scheme == "ClenshawCurtis" else SO3FourierPlan(nodes.nodes(), L, real, nodes.folds)
  return SO3FourierPlan(nodes, L, real, folds)


def sphericalPlan(nodes, L, real=False):
  """the plan for a node set on the sphere: the FFT on a Clenshaw-Curtis grid, the NUFFT otherwise"""
  if isinstance(nodes, quadratureS2Grid):
    return S2GridPlan(nodes, L, real) if nodes.scheme == "ClenshawCurtis" else S2FourierPlan(nodes.nodes(), L, real)
  return S2FourierPlan(nodes, L, real)


# ------------------------------------------------------------------ evaluation and adjoint with the cost rule


def wignerSeries(fhat, rots, real=None, folds=(1, 1)):
  """the series at every rotation of a container or on a quadrature grid; the direct sum below
  DIRECT_SERIES_NODES nodes (DIRECT_SUM_NODES without numba), and a grid that cannot hold the bandwidth is evaluated at its nodes"""
  fhat = np.asarray(fhat)
  L = dim2deg(fhat.shape[-1])
  real = isRealWigner(fhat) if real is None else real
  if isinstance(rots, quadratureSO3Grid):
    if 0 < L <= rots.bandwidth:
      return wignerPlan(rots, L, real)(fhat)
    return wignerSeries(fhat, rots.nodes(), real, rots.folds)
  if rots.size < (DIRECT_SERIES_NODES if backend() == "numba" else DIRECT_SUM_NODES) or L == 0:
    f = wignerSeriesDirect(fhat, rots)
    f = np.real(f) if real else f
  else:
    f = SO3FourierPlan(rots, L, real, folds)(fhat).reshape(rots.shape)
  return f[()] if f.ndim == 0 else f


def wignerSeriesAdjoint(rots, values, L, weights=None, folds=(1, 1)):
  """coefficients from weighted values: the adjoint sum below DIRECT_SUM_NODES nodes; a
  quadrature grid brings the weights of its rule"""
  values = np.asarray(values).ravel()
  if isinstance(rots, quadratureSO3Grid):
    if 0 < L <= rots.bandwidth:
      plan = wignerPlan(rots, L, values.dtype.kind != "c")
      # the Clenshaw-Curtis plan weighs by its rule itself, without spreading the weights out
      return plan.adjoint(values, rots.weights if weights is None and not isinstance(plan, SO3GridPlan) else weights)
    return wignerSeriesAdjoint(rots.nodes(), values, L, rots.weights if weights is None else weights, rots.folds)
  n = rots.size
  weights = np.full(n, 1 / n) if weights is None else np.asarray(weights).ravel()
  if n < DIRECT_SUM_NODES or L == 0:
    (_, _, _), dtype = _euler(rots)
    fhat = np.zeros(deg2dim(L + 1), dtype=_complexType(np.result_type(dtype, values.dtype)))
    D = _WignerD(rots.flatten(), bandwidth=L)
    for n_ in range(L + 1):
      degreeBlock(fhat, n_)[...] = math.sqrt(2 * n_ + 1) * np.einsum("i,ikl->kl", weights * values, np.conj(D[n_]))
    return fhat
  return SO3FourierPlan(rots, L, values.dtype.kind != "c", folds).adjoint(values, weights)


# the adjoint of many rotations approximately: every rotation rounded to the nearest node of
# a Clenshaw-Curtis grid, the weighted values summed there, and the grid's exact FFT
# adjoint (MATLAB SO3FunHarmonic.adjoint(..., 'gridded')). The error is at most half the grid
# spacing, pi/(4 NG) in beta, which a kernel of bandwidth L cannot see when the grid has
# NG = min(2L, max(L, 128)); the cost no longer grows with the rotations. The folds act on the
# folded angles, which is all a folded coefficient sees, so the grid covers their region only.
# It is taken above 2 L^3 rotations, where it is faster than finufft by 2 to 12 times and the
# density moves by 0.4 to 0.9 percent in L2, 1.5 at a 2.4 degree kernel cut at L = 64
# (docs/research/benchmark-harmonics.md); MATLAB crosses at 10 L^3
GRIDDED_ROTATIONS_PER_L3 = 2


def wignerSeriesAdjointGridded(rots, values, L, weights=None, folds=(1, 1)):
  """the SO(3) coefficients up to bandwidth L from weighted values, every rotation rounded to the
  nearest node of a Clenshaw-Curtis grid and the grid's FFT adjoint taken, MATLAB's 'gridded';
  approximate, and its cost independent of the number of rotations"""
  values = np.asarray(values).ravel()
  q = _quaternions(rots)
  w = np.full(len(q), 1 / len(q)) if weights is None else np.asarray(weights).ravel()
  NG = bandwidthFor(min(2 * L, max(L, 128)), *folds)
  g = quadratureSO3Grid(NG, rA=folds[0], rB=folds[1])
  Ng, Na, nb = g.shape
  x, y, z, _, bad = so3Nodes(q, folds)
  # the folded gamma and alpha on the grid's own lengths, beta on the polar nodes
  c = np.rint(x * Ng / (2 * np.pi)).astype(np.int64) % Ng
  a = np.rint(y * Na / (2 * np.pi)).astype(np.int64) % Na
  b = np.clip(np.rint(z * (nb - 1) / np.pi).astype(np.int64), 0, nb - 1)
  u = w * values
  if bad is not None:
    u = np.where(bad, 0, u)
  idx = (c * Na + a) * nb + b
  if values.dtype.kind == "c":
    v = np.bincount(idx, u.real, minlength=len(g)) + 1j * np.bincount(idx, u.imag, minlength=len(g))
  else:
    v = np.bincount(idx, u, minlength=len(g))
  return SO3GridPlan(g, L, real=values.dtype.kind != "c").adjoint(v, np.ones(len(g)))


# the direct spherical sum, degree by degree through the harmonics of F0
def _sphericalSeriesDirect(fhat, v):
  L = math.isqrt(fhat.shape[-1]) - 1
  x = np.cos(v.theta).reshape(-1).astype(float)
  rho = np.asarray(v.rho, dtype=float).reshape(-1)
  st = np.sqrt(np.maximum(0, 1 - x * x))
  m = np.arange(L + 1)
  # the normalised associated Legendre functions degree by degree, all orders at once: the
  # diagonal P_m^m by its own recursion, the rest by the three term recursion in the degree
  diag = np.empty((L + 1, x.size))
  diag[0] = 1 / math.sqrt(4 * math.pi)
  for k in range(1, L + 1):
    diag[k] = diag[k - 1] * (-math.sqrt((2 * k + 1) / (2 * k)) * st)
  A = np.zeros((L + 1, x.size), dtype=_complexType(fhat.dtype))
  B = np.zeros((L + 1, x.size), dtype=_complexType(fhat.dtype))
  p2 = np.zeros((L + 1, x.size))
  p1 = np.zeros((L + 1, x.size))
  for n in range(L + 1):
    p = np.zeros((L + 1, x.size))
    p[n] = diag[n]
    if n > 0:
      mm = m[:n]
      a = np.sqrt((4 * n * n - 1) / (n * n - mm * mm))
      b = np.sqrt(np.maximum((n - 1) ** 2 - mm * mm, 0) / (4 * (n - 1) ** 2 - 1))
      p[:n] = a[:, None] * (x * p1[:n] - b[:, None] * p2[:n])
    # the coefficients of orders m and -m of this degree weight the Legendre values
    base = n * n + n
    A[:n + 1] += fhat[base + m[:n + 1]][:, None] * p[:n + 1]
    B[1:n + 1] += fhat[base - m[1:n + 1]][:, None] * p[1:n + 1]
    p2, p1 = p1, p
  E = np.exp(1j * np.outer(m, rho))
  # Y_n^-m = (-1)^m conj(Y_n^m)
  total = np.sum(A * E, axis=0) + np.sum(np.where(m % 2, -1.0, 1.0)[:, None] * B * np.conj(E), axis=0)
  return total.reshape(v.shape)


# the bivariate Fourier series of H7 summed directly at a few nodes, where a NUFFT plan
# costs more than the sum: f(theta, rho) = sum G[m, j] e^{i(m rho - j theta)}
# the transforms of the last few coefficient vectors, by the hash of their bytes: a search
# that evaluates one function at many small node sets pays for its transform once
_TRIG_CACHE = {}


def _cachedTransform(fhat, L, real):
  key = (hash(np.ascontiguousarray(fhat).tobytes()), fhat.dtype.str, L, bool(real))
  G = _TRIG_CACHE.get(key)
  if G is None:
    G = sphericalTransform(fhat, L, real)
    if len(_TRIG_CACHE) >= 4:
      _TRIG_CACHE.pop(next(iter(_TRIG_CACHE)))
    _TRIG_CACHE[key] = G
  return G


def _sphericalSeriesTrig(fhat, v, real):
  L = math.isqrt(fhat.shape[-1]) - 1
  G = _cachedTransform(fhat, L, real)
  theta, rho = v.flatten().theta, v.flatten().rho
  m = np.arange(L + 1) if real else np.arange(-L, L + 1)
  Et = np.exp(-1j * np.outer(theta, np.arange(-L, L + 1)))
  Er = np.exp(1j * np.outer(rho, m))
  f = np.einsum("nm,nm->n", Er, Et @ G.T)
  return (2 * np.real(f) if real else f).reshape(v.shape)


def sphericalSeries(fhat, v, real=None):
  """the spherical harmonic series fhat at directions or on a quadrature grid, the direct sum, the
  trigonometric sum or the NUFFT by the number of nodes"""
  fhat = np.asarray(fhat)
  L = math.isqrt(fhat.shape[-1]) - 1
  real = isRealSpherical(fhat) if real is None else real
  if isinstance(v, quadratureS2Grid):
    if 0 < L <= v.bandwidth:
      return sphericalPlan(v, L, real)(fhat)
    return sphericalSeries(fhat, v.nodes(), real)
  v = v if isinstance(v, vector3d) else vector3d(v)
  if v.size < DIRECT_SUM_NODES or L == 0:
    f = _sphericalSeriesDirect(fhat, v)
    f = np.real(f) if real else f
  elif v.size < DIRECT_TRIG_NODES_S2 or v.size * (2 * L + 1) ** 2 < DIRECT_TRIG_WORK_S2:
    f = _sphericalSeriesTrig(fhat, v, real)
  else:
    f = S2FourierPlan(v, L, real)(fhat).reshape(v.shape)
  return f[()] if f.ndim == 0 else f


def sphericalSeriesAdjoint(v, values, L, weights=None, real=False):
  """the spherical coefficients up to bandwidth L from weighted values at directions, the direct sum
  below 50 nodes; a quadrature grid brings the weights of its rule"""
  values = np.asarray(values).ravel()
  if isinstance(v, quadratureS2Grid):
    weights = v.weights if weights is None else weights
    if 0 < L <= v.bandwidth:
      return sphericalPlan(v, L, real).adjoint(values, weights)
    return sphericalSeriesAdjoint(v.nodes(), values, L, weights, real)
  v = (v if isinstance(v, vector3d) else vector3d(v)).flatten()
  n = v.size
  weights = np.full(n, 1 / n) if weights is None else np.asarray(weights).ravel()
  if n < DIRECT_SUM_NODES or L == 0:
    fhat = np.zeros((L + 1) ** 2, dtype=_complexType(values.dtype))
    for l in range(L + 1):
      fhat[l * l:(l + 1) ** 2] = np.conj(_sphericalY(l, v)) @ (weights * values)
    return fhat
  return S2FourierPlan(v, L, real).adjoint(values, weights)


def fourierSeries(chat, x, real=None):
  """the Fourier series sum of chat_k e^{ikx} at the angles x, directly or by NUFFT; real when the
  coefficients are those of a real function"""
  chat = np.asarray(chat)
  L = (chat.shape[-1] - 1) // 2
  real = isRealFourier(chat) if real is None else real
  if isinstance(x, quadratureS1Grid):
    x = x.nodes()
  x = np.asarray(x)
  # the direct sum is exact, the plan only to its tolerance; the sum is cheaper for few terms
  if x.size * (2 * L + 1) < DIRECT_SUM_WORK_S1:
    f = np.exp(1j * np.outer(x.ravel(), np.arange(-L, L + 1))) @ chat
  else:
    f = S1FourierPlan(x, L)(chat)
  f = f.reshape(x.shape)
  f = np.real(f) if real else f
  return f[()] if f.ndim == 0 else f


def fourierSeriesAdjoint(x, values, L, weights=None):
  """the Fourier coefficients c_{-L} .. c_L from weighted values at the angles x, the sum of
  w v e^{-ikx}; a quadrature grid brings the weights of its rule"""
  if isinstance(x, quadratureS1Grid):
    weights = x.weights if weights is None else weights
    x = x.nodes()
  x = np.asarray(x).ravel()
  values = np.asarray(values).ravel()
  weights = np.full(x.size, 1 / x.size) if weights is None else np.asarray(weights).ravel()
  if x.size * (2 * L + 1) < DIRECT_SUM_WORK_S1:
    return np.exp(-1j * np.outer(np.arange(-L, L + 1), x)) @ (weights * values)
  return S1FourierPlan(x, L).adjoint(values, weights)


__all__ = ["halfPiDegrees", "foldedDims", "usableFolds", "wignerTransform", "wignerTransformAdjoint",
           "sphericalTransform", "sphericalTransformAdjoint", "isRealWigner", "isRealSpherical", "isRealFourier",
           "SO3FourierPlan", "S2FourierPlan", "S1FourierPlan", "SO3GridPlan", "S2GridPlan", "adjointTransform",
           "wignerPlan", "sphericalPlan", "wignerSeries", "wignerSeriesAdjoint", "wignerSeriesAdjointGridded", "sphericalSeries",
           "sphericalSeriesAdjoint", "fourierSeries", "fourierSeriesAdjoint"]
