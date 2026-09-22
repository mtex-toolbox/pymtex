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
from ..geometry.kernels import _jit, _loop, backend
from ..geometry.quaternion import orthogonalTransform, quaternion
from ..geometry.representations import quat2euler
from ..geometry.vector3d import vector3d
from .harmonics import WignerD, deg2dim, degreeBlock, dim2deg, sphericalY, wignerdDegrees
from .harmonics import wignerSeries as _wignerSeriesDirect
from .quadrature import quadratureS1Grid, quadratureS2Grid, quadratureSO3Grid

DIRECT_SUM_NODES = 50
# on the sphere a plan costs more than a trigonometric sum on the Fourier array up to here
DIRECT_TRIG_NODES_S2 = 400
NUFFT_EPS = 1e-8          # finufft's tolerance in double precision
NUFFT_EPS_SINGLE = 1e-5   # and in single, where 1e-8 is out of reach


def _finufft():
  import finufft
  return finufft


def _complexType(dtype):
  return np.result_type(np.dtype(dtype), 1j)


def _eps(eps, dtype):
  return eps if eps is not None else NUFFT_EPS if np.dtype(dtype).itemsize == 8 else NUFFT_EPS_SINGLE


# the options of a plan: one thread below a work of nodes times modes, where finufft's
# thread start-up costs more than the transform (30 ms against 2 ms on 100 nodes at
# bandwidth 32); a caller that trades accuracy for the FFT of a smaller lattice, as the
# sampler does, asks for the low oversampling factor itself
NUFFT_SINGLE_THREAD_WORK = 2e8


def _threads(nodes, dims, upsampfac=None):
  opts = {"nthreads": 1} if nodes * int(np.prod(dims)) < NUFFT_SINGLE_THREAD_WORK else {}
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
  rA, rB = folds
  return (L // rA + 1 if real else 2 * (L // rA) + 1, 2 * (L // rB) + 1, 2 * L + 1)


# the folds a NUFFT array can carry: a dimension too thin for the window costs accuracy
NUFFT_MIN_DIM = 4


def usableFolds(L, real, folds):
  d = foldedDims(L, real, folds)
  return (int(folds[0]), int(folds[1])) if min(d[0], d[1]) >= NUFFT_MIN_DIM else (1, 1)


# the phases sqrt(2n+1) i^{k-l} of one degree block, the plane l = 0 halved for a real function
def _blockPhase(n, real, dtype):
  k = np.arange(-n, n + 1)
  P = math.sqrt(2 * n + 1) * 1j ** ((k[:, None] - k[None, :]) % 4)
  if real:
    P[:, n] /= 2
  return P.astype(_complexType(dtype))


# one degree into the array, j >= 0 only: the loop compiled, and the NumPy form
def _wignerForwardLoop(G, C, d, n, off, real, rA, rB, Kb, La):
  la, kb = n // rA, n // rB
  for lp in range(0 if real else -la, la + 1):
    l = lp * rA
    il = lp if real else lp + La
    for kp in range(-kb, kb + 1):
      k = kp * rB
      c = C[k + n, l + n]
      if c == 0:
        continue
      ic = kp + Kb
      for j in range(n, 2 * n + 1):
        G[il, ic, off + j] += c * d[j, k + n] * d[j, l + n]


def _wignerForwardNumpy(G, C, d, n, off, real, rA, rB, Kb, La):
  la, kb = n // rA, n // rB
  lp = np.arange(0 if real else -la, la + 1)
  kp = np.arange(-kb, kb + 1)
  Csub = C[np.ix_(kp * rB + n, lp * rA + n)]
  G[np.ix_(lp if real else lp + La, kp + Kb, np.arange(off + n, off + 2 * n + 1))] += \
    np.einsum("jk,jl,kl->lkj", d[n:, kp * rB + n], d[n:, lp * rA + n], Csub)


# one degree block out of the array, the sum over j of d d h[-j]
def _wignerAdjointLoop(F, H, d, n, off, real, rA, rB, Kb, La):
  la, kb = n // rA, n // rB
  for lp in range(0 if real else -la, la + 1):
    l = lp * rA
    il = lp if real else lp + La
    for kp in range(-kb, kb + 1):
      k = kp * rB
      ic = kp + Kb
      acc = 0j
      for jj in range(2 * n + 1):
        acc += d[jj, k + n] * d[jj, l + n] * H[il, ic, off + 2 * n - jj]
      F[k + n, l + n] = acc


def _wignerAdjointNumpy(F, H, d, n, off, real, rA, rB, Kb, La):
  la, kb = n // rA, n // rB
  lp = np.arange(0 if real else -la, la + 1)
  kp = np.arange(-kb, kb + 1)
  Hsub = H[np.ix_(lp if real else lp + La, kp + Kb, np.arange(off + 2 * n, off - 1, -1))]
  F[np.ix_(kp * rB + n, lp * rA + n)] = np.einsum("jk,jl,lkj->kl", d[:, kp * rB + n], d[:, lp * rA + n], Hsub)


def _forwardKernel():
  return _loop("wignerForward", _wignerForwardLoop) if backend() == "numba" else _wignerForwardNumpy


def _adjointKernel():
  return _loop("wignerAdjoint", _wignerAdjointLoop) if backend() == "numba" else _wignerAdjointNumpy


# the sign of the j mirror: G[l, k, -j] = (-1)^{k+l} G[l, k, j]
def _mirrorSign(L, real, folds):
  rA, rB = folds
  La, Kb = L // rA, L // rB
  l = (np.arange(La + 1) if real else np.arange(-La, La + 1)) * rA
  k = np.arange(-Kb, Kb + 1) * rB
  return np.where((l[:, None] + k[None, :]) % 2, -1.0, 1.0)


def wignerTransform(fhat, L, real=False, folds=(1, 1), degrees=None):
  fhat = np.asarray(fhat)
  cdtype = _complexType(fhat.dtype)
  rdtype = np.dtype(cdtype).type(0).real.dtype
  rA, rB = folds
  La, Kb = L // rA, L // rB
  G = np.zeros(foldedDims(L, real, folds), dtype=cdtype)
  kernel = _forwardKernel()
  for n, dn in (halfPiDegrees(L, rdtype) if degrees is None else degrees):
    F = degreeBlock(fhat, n)
    if not np.any(F):
      continue
    C = np.ascontiguousarray(F * _blockPhase(n, real, rdtype), dtype=cdtype)
    kernel(G, C, np.ascontiguousarray(dn[:2 * n + 1, :2 * n + 1]), n, L - n, real, rA, rB, Kb, La)
  G[:, :, :L] = _mirrorSign(L, real, folds)[:, :, None] * G[:, :, L + 1:][:, :, ::-1]
  return G


# the adjoint: fhat_n^{kl} = sqrt(2n+1) i^{k-l} sum_j d_n^{jk} d_n^{jl} h[l, k, -j] for
# h = sum_m w_m v_m e^{i(k alpha + j beta + l gamma)}; for real values only l >= 0 is given and
# the rest follows from fhat_n^{-k,-l} = (-1)^{k-l} conj(fhat_n^{kl}). The folds step l by rA
# and k by rB, since the projection of H5 zeroes the rest anyway
def wignerTransformAdjoint(H, L, real=False, folds=(1, 1), degrees=None):
  H = np.ascontiguousarray(H)
  cdtype = H.dtype
  rdtype = np.dtype(cdtype).type(0).real.dtype
  rA, rB = folds
  La, Kb = L // rA, L // rB
  fhat = np.zeros(deg2dim(L + 1), dtype=cdtype)
  kernel = _adjointKernel()
  for n, dn in (halfPiDegrees(L, rdtype) if degrees is None else degrees):
    F = np.zeros((2 * n + 1, 2 * n + 1), dtype=cdtype)
    kernel(F, H, np.ascontiguousarray(dn[:2 * n + 1, :2 * n + 1]), n, L - n, real, rA, rB, Kb, La)
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
  fhat = np.asarray(fhat)
  cdtype = _complexType(fhat.dtype)
  rdtype = np.dtype(cdtype).type(0).real.dtype
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


# the adjoint: fhat_l^m = sqrt((2l+1)/4pi) i^m sum_j d_l^{jm} d_l^{j0} h[m, -j] for
# h[m, j] = sum_q w_q v_q e^{-i(m rho_q - j theta_q)}
def sphericalTransformAdjoint(H, L, real=False, degrees=None):
  H = np.asarray(H)
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


# a coefficient vector is real valued when fhat_n^{-k,-l} = (-1)^{k-l} conj(fhat_n^{kl}) on
# SO(3), fhat_l^{-m} = (-1)^m conj(fhat_l^m) on S2, c_{-k} = conj(c_k) on S1
def isRealWigner(fhat, tol=1e-10):
  fhat = np.asarray(fhat)
  for n in range(dim2deg(fhat.shape[-1]) + 1):
    F = degreeBlock(fhat, n)
    k = np.arange(-n, n + 1)
    sign = np.where((k[:, None] - k[None, :]) % 2, -1.0, 1.0)
    if not np.all(np.abs(F[::-1, ::-1] - sign * np.conj(F)) <= tol * np.maximum(1, np.abs(F))):
      return False
  return True


def isRealSpherical(fhat, tol=1e-10):
  fhat = np.asarray(fhat)
  L = math.isqrt(fhat.shape[-1]) - 1
  for l in range(L + 1):
    c = fhat[l * l:(l + 1) ** 2]
    m = np.arange(-l, l + 1)
    if not np.all(np.abs(c[::-1] - np.where(m % 2, -1.0, 1.0) * np.conj(c)) <= tol * np.maximum(1, np.abs(c))):
      return False
  return True


def isRealFourier(chat, tol=1e-10):
  chat = np.asarray(chat)
  return bool(np.all(np.abs(chat[::-1] - np.conj(chat)) <= tol * np.maximum(1, np.abs(chat))))


# ------------------------------------------------------------------ the plans on arbitrary nodes


# the Matthies angles of a rotation container as flat arrays, and their float type
def _euler(rots):
  if isinstance(rots, orthogonalTransform):
    rots = rots.rot
  if not isinstance(rots, quaternion):
    raise TypeError("the nodes are a container of rotations")
  q = rots.data.reshape(-1, 4)
  return quat2euler(q, "Matthies"), q.dtype


# the NUFFT of one node set on SO(3), reusable across coefficient vectors. folds says that
# the function is invariant under those z folds on the two sides: in the forward its
# coefficients outside them are zero, in the adjoint the projection of H5 is about to zero
# them. Both directions then run on the reduced index set, where the two periodic axes of
# the array shrink by the folds and the nodes are rescaled
class SO3FourierPlan:

  def __init__(self, rots, L, real=True, folds=(1, 1), eps=None, upsampfac=None):
    (alpha, beta, gamma), dtype = _euler(rots)
    self.L, self.real, self.folds = int(L), bool(real), usableFolds(L, real, folds)
    rA, rB = self.folds
    self.dims = foldedDims(self.L, self.real, self.folds)
    self.dtype = _complexType(dtype)
    x = np.ascontiguousarray(np.mod(rA * gamma, 2 * np.pi), dtype=dtype)
    y = np.ascontiguousarray(np.mod(rB * alpha, 2 * np.pi), dtype=dtype)
    z = np.ascontiguousarray(beta, dtype=dtype)
    f, eps, threads = _finufft(), _eps(eps, dtype), _threads(len(x), self.dims, upsampfac)
    self.forward = f.Plan(2, self.dims, eps=eps, isign=-1, dtype=self.dtype.name, **threads)
    self.forward.setpts(x, y, z)
    self.backward = f.Plan(1, self.dims, eps=eps, isign=1, dtype=self.dtype.name, **threads)
    self.backward.setpts(x, y, z)
    # e^{-i s gamma} moves the half array onto the centred index set of the NUFFT
    s = self.dims[0] // 2 if self.real else 0
    self.phase = np.exp(-1j * s * x).astype(self.dtype)

  def __len__(self):
    return len(self.phase)

  def __call__(self, fhat):
    G = wignerTransform(np.asarray(fhat, dtype=self.dtype), self.L, self.real, self.folds)
    f = self.forward.execute(G)
    return 2 * np.real(self.phase * f) if self.real else f

  # coefficients from weighted values at the nodes
  def adjoint(self, values, weights):
    values = np.asarray(values).ravel()
    u = np.ascontiguousarray(np.asarray(weights).ravel() * values * np.conj(self.phase), dtype=self.dtype)
    H = self.backward.execute(u)
    return wignerTransformAdjoint(H, self.L, self.real and values.dtype.kind != "c", self.folds)


class S2FourierPlan:

  def __init__(self, v, L, real=False, eps=None, upsampfac=None):
    v = v if isinstance(v, vector3d) else vector3d(v)
    theta, rho = v.flatten().theta, v.flatten().rho
    dtype = v.data.dtype if v.data.dtype.kind == "f" else np.float64
    self.L, self.real = int(L), bool(real)
    self.dims = (self.L + 1 if self.real else 2 * self.L + 1, 2 * self.L + 1)
    self.dtype = _complexType(dtype)
    x = np.ascontiguousarray(np.mod(-rho, 2 * np.pi), dtype=dtype)
    y = np.ascontiguousarray(theta, dtype=dtype)
    f, eps, threads = _finufft(), _eps(eps, dtype), _threads(len(x), self.dims, upsampfac)
    self.forward = f.Plan(2, self.dims, eps=eps, isign=-1, dtype=self.dtype.name, **threads)
    self.forward.setpts(x, y)
    self.backward = f.Plan(1, self.dims, eps=eps, isign=1, dtype=self.dtype.name, **threads)
    self.backward.setpts(x, y)
    s = self.dims[0] // 2 if self.real else 0
    self.phase = np.exp(-1j * s * x).astype(self.dtype)

  def __len__(self):
    return len(self.phase)

  def __call__(self, fhat):
    G = sphericalTransform(np.asarray(fhat, dtype=self.dtype), self.L, self.real)
    f = self.forward.execute(G)
    return 2 * np.real(self.phase * f) if self.real else f

  def adjoint(self, values, weights):
    values = np.asarray(values).ravel()
    u = np.ascontiguousarray(np.asarray(weights).ravel() * values * np.conj(self.phase), dtype=self.dtype)
    H = self.backward.execute(u)
    return sphericalTransformAdjoint(H, self.L, self.real and values.dtype.kind != "c")


# the circle: f(x) = sum_{k=-L}^{L} c_k e^{ikx}
class S1FourierPlan:

  def __init__(self, x, L, eps=None):
    x = np.asarray(x).ravel()
    dtype = x.dtype if x.dtype.kind == "f" else np.float64
    self.L, self.dtype = int(L), _complexType(dtype)
    x = np.ascontiguousarray(np.mod(x, 2 * np.pi), dtype=dtype)
    f, eps, threads = _finufft(), _eps(eps, dtype), _threads(len(x), (2 * self.L + 1,))
    self.forward = f.Plan(2, (2 * self.L + 1,), eps=eps, isign=1, dtype=self.dtype.name, **threads)
    self.forward.setpts(x)
    self.backward = f.Plan(1, (2 * self.L + 1,), eps=eps, isign=-1, dtype=self.dtype.name, **threads)
    self.backward.setpts(x)
    self.n = len(x)

  def __len__(self):
    return self.n

  def __call__(self, chat):
    return self.forward.execute(np.ascontiguousarray(chat, dtype=self.dtype))

  def adjoint(self, values, weights):
    return self.backward.execute(np.ascontiguousarray(np.asarray(weights).ravel() * np.asarray(values).ravel(), dtype=self.dtype))


def adjointTransform(plan, values, weights):
  return plan.adjoint(values, weights)


# ------------------------------------------------------------------ the plans on a Clenshaw-Curtis grid
# every angle of the grid is equispaced, so the series on it is a plain multivariate DFT.
# On SO(3) f(alpha, beta, gamma) = sum G[l, k, j] e^{-i(k alpha + j beta + l gamma)} with
# beta_b = pi b/(2L), alpha_a = 2pi a/(2L+2), gamma_c = 2pi c/(2L+2) is a forward transform
# of length 4L in beta and 2L+2 in alpha and gamma, the array wrapped into that shape and
# the first 2L+1 beta rows kept; the adjoint is the same transform backwards. An r fold
# leaves k = 0 mod r and k/r indexes a transform of length (2L+2)/r, so the folded ring
# shrinks the transform by the fold exactly and leaves the forbidden coefficients at zero


def _fft():
  import scipy.fft
  return scipy.fft


# the coefficient array and the lattice hold the same numbers in different places: the
# periodic axes are already indexed by l/rA and k/rB, only j wraps modulo the polar length
def _wignerLattice(A, G, L, real, folds, out):
  rA, rB = folds
  La, Kb = L // rA, L // rB
  nl, nk = G.shape[0], G.shape[1]
  ic = np.mod(np.arange(nl) - (0 if real else La), A.shape[0])
  ib = np.mod(np.arange(nk) - Kb, A.shape[1])
  Nb = A.shape[2]
  head, tail = np.ix_(ic, ib, np.arange(L + 1)), np.ix_(ic, ib, np.arange(Nb - L, Nb))
  if out:
    G[:, :, L:] = A[head]
    G[:, :, :L] = A[tail]
  else:
    A[head] = G[:, :, L:]
    A[tail] = G[:, :, :L]


class SO3GridPlan:

  def __init__(self, g, L=None, real=True):
    L = g.bandwidth if L is None else int(L)
    if g.scheme != "ClenshawCurtis":
      raise ValueError("the FFT route needs a Clenshaw-Curtis grid")
    if not 0 < L <= g.bandwidth:
      raise ValueError(f"a bandwidth {L} does not fit the grid of bandwidth {g.bandwidth}")
    self.L, self.real, self.folds, self.grid = L, bool(real), g.folds, g
    self.dtype = _complexType(g.beta.dtype)
    self.shape = g.shape
    self.lattice = np.zeros((len(g.gamma), len(g.alpha), 4 * g.bandwidth), dtype=self.dtype)

  def __len__(self):
    return len(self.grid)

  def __call__(self, fhat):
    G = wignerTransform(np.asarray(fhat, dtype=self.dtype), self.L, self.real, self.folds)
    fft, A = _fft(), self.lattice
    A[...] = 0
    _wignerLattice(A, G, self.L, self.real, self.folds, False)
    # beta and alpha on the occupied gamma planes, then gamma at the polar nodes only
    ng, nb = (self.L // self.folds[0] + 1) if self.real else A.shape[0], self.shape[2]
    A[:ng] = fft.fftn(A[:ng], axes=(1, 2))
    v = fft.fft(A[:, :, :nb], axis=0)
    return (2 * np.real(v) if self.real else v).ravel()

  def adjoint(self, values, weights):
    values = np.asarray(values).ravel()
    real = values.dtype.kind != "c"
    fft, A = _fft(), self.lattice
    A[...] = 0
    nb = self.shape[2]
    A[:, :, :nb] = (np.asarray(weights).ravel() * values).reshape(self.shape)
    # alpha and gamma at the polar nodes, then beta on the gamma planes the transform reads
    A[:, :, :nb] = fft.ifftn(A[:, :, :nb], axes=(0, 1), norm="forward")
    ng = (self.L // self.folds[0] + 1) if real else A.shape[0]
    A[:ng] = fft.ifft(A[:ng], axis=2, norm="forward")
    H = np.empty(foldedDims(self.L, real, self.folds), dtype=self.dtype)
    _wignerLattice(A, H, self.L, real, self.folds, True)
    return wignerTransformAdjoint(H, self.L, real, self.folds)


# the sphere: f(theta, rho) = sum G[m, j] e^{i(m rho - j theta)}, a transform of length 4L in
# theta and 2L+2 in rho, the order held at -m so that both directions are forward transforms
def _sphericalLattice(A, G, L, real, out):
  nm = G.shape[0]
  m = np.arange(nm) - (0 if real else L)
  ic = np.mod(-m, A.shape[0])
  Nt = A.shape[1]
  head, tail = np.ix_(ic, np.arange(L + 1)), np.ix_(ic, np.arange(Nt - L, Nt))
  if out:
    G[:, L:] = A[head]
    G[:, :L] = A[tail]
  else:
    A[head] = G[:, L:]
    A[tail] = G[:, :L]


class S2GridPlan:

  def __init__(self, g, L=None, real=False):
    L = g.bandwidth if L is None else int(L)
    if g.scheme != "ClenshawCurtis":
      raise ValueError("the FFT route needs a Clenshaw-Curtis grid")
    if not 0 < L <= g.bandwidth:
      raise ValueError(f"a bandwidth {L} does not fit the grid of bandwidth {g.bandwidth}")
    self.L, self.real, self.grid = L, bool(real), g
    self.dtype = _complexType(g.theta.dtype)
    self.shape = g.shape
    self.lattice = np.zeros((len(g.rho), 4 * g.bandwidth), dtype=self.dtype)

  def __len__(self):
    return len(self.grid)

  def _rows(self, real):
    return np.mod(-np.arange(self.L + 1 if real else 2 * self.L + 1) + (0 if real else self.L), self.lattice.shape[0])

  def __call__(self, fhat):
    G = sphericalTransform(np.asarray(fhat, dtype=self.dtype), self.L, self.real)
    fft, A = _fft(), self.lattice
    A[...] = 0
    _sphericalLattice(A, G, self.L, self.real, False)
    rows, nt = self._rows(self.real), self.shape[1]
    A[rows] = fft.fft(A[rows], axis=1)
    v = fft.fft(A[:, :nt], axis=0)
    return (2 * np.real(v) if self.real else v).ravel()

  def adjoint(self, values, weights):
    values = np.asarray(values).ravel()
    real = values.dtype.kind != "c"
    fft, A = _fft(), self.lattice
    A[...] = 0
    nt = self.shape[1]
    A[:, :nt] = (np.asarray(weights).ravel() * values).reshape(self.shape)
    A[:, :nt] = fft.ifft(A[:, :nt], axis=0, norm="forward")
    rows = self._rows(real)
    A[rows] = fft.ifft(A[rows], axis=1, norm="forward")
    H = np.empty((self.L + 1 if real else 2 * self.L + 1, 2 * self.L + 1), dtype=self.dtype)
    _sphericalLattice(A, H, self.L, real, True)
    return sphericalTransformAdjoint(H, self.L, real)


# the plan for a node set: the NUFFT at arbitrary nodes, the FFT on a Clenshaw-Curtis grid,
# the NUFFT on the nodes of a grid whose polar rule is not equispaced
def wignerPlan(nodes, L, real=True, folds=(1, 1)):
  if isinstance(nodes, quadratureSO3Grid):
    return SO3GridPlan(nodes, L, real) if nodes.scheme == "ClenshawCurtis" else SO3FourierPlan(nodes.nodes(), L, real, nodes.folds)
  return SO3FourierPlan(nodes, L, real, folds)


def sphericalPlan(nodes, L, real=False):
  if isinstance(nodes, quadratureS2Grid):
    return S2GridPlan(nodes, L, real) if nodes.scheme == "ClenshawCurtis" else S2FourierPlan(nodes.nodes(), L, real)
  return S2FourierPlan(nodes, L, real)


# ------------------------------------------------------------------ evaluation and adjoint with the cost rule


# the series at every rotation of a container or on a quadrature grid; the direct sum below
# DIRECT_SUM_NODES nodes, and a grid that cannot hold the bandwidth is evaluated at its nodes
def wignerSeries(fhat, rots, real=None, folds=(1, 1)):
  fhat = np.asarray(fhat)
  L = dim2deg(fhat.shape[-1])
  real = isRealWigner(fhat) if real is None else real
  if isinstance(rots, quadratureSO3Grid):
    if 0 < L <= rots.bandwidth:
      return wignerPlan(rots, L, real)(fhat)
    return wignerSeries(fhat, rots.nodes(), real, rots.folds)
  if rots.size < DIRECT_SUM_NODES or L == 0:
    f = _wignerSeriesDirect(fhat, rots)
    f = np.real(f) if real else f
  else:
    f = SO3FourierPlan(rots, L, real, folds)(fhat).reshape(rots.shape)
  return f[()] if f.ndim == 0 else f


# coefficients from weighted values: the adjoint sum below DIRECT_SUM_NODES nodes; a
# quadrature grid brings the weights of its rule
def wignerSeriesAdjoint(rots, values, L, weights=None, folds=(1, 1)):
  values = np.asarray(values).ravel()
  if isinstance(rots, quadratureSO3Grid):
    weights = rots.weights if weights is None else weights
    if 0 < L <= rots.bandwidth:
      return wignerPlan(rots, L, values.dtype.kind != "c").adjoint(values, weights)
    return wignerSeriesAdjoint(rots.nodes(), values, L, weights, rots.folds)
  n = rots.size
  weights = np.full(n, 1 / n) if weights is None else np.asarray(weights).ravel()
  if n < DIRECT_SUM_NODES or L == 0:
    (_, _, _), dtype = _euler(rots)
    fhat = np.zeros(deg2dim(L + 1), dtype=_complexType(np.result_type(dtype, values.dtype)))
    D = WignerD(rots.flatten(), bandwidth=L)
    for n_ in range(L + 1):
      degreeBlock(fhat, n_)[...] = math.sqrt(2 * n_ + 1) * np.einsum("i,ikl->kl", weights * values, np.conj(D[n_]))
    return fhat
  return SO3FourierPlan(rots, L, values.dtype.kind != "c", folds).adjoint(values, weights)


# the direct spherical sum, degree by degree through the harmonics of F0
def _sphericalSeriesDirect(fhat, v):
  L = math.isqrt(fhat.shape[-1]) - 1
  total = np.zeros(v.shape, dtype=_complexType(fhat.dtype))
  for l in range(L + 1):
    total += np.tensordot(fhat[l * l:(l + 1) ** 2], sphericalY(l, v), axes=(0, 0))
  return total


# the bivariate Fourier series of H7 summed directly at a few nodes, where a NUFFT plan
# costs more than the sum: f(theta, rho) = sum G[m, j] e^{i(m rho - j theta)}
def _sphericalSeriesTrig(fhat, v, real):
  L = math.isqrt(fhat.shape[-1]) - 1
  G = sphericalTransform(fhat, L, real)
  theta, rho = v.flatten().theta, v.flatten().rho
  m = np.arange(L + 1) if real else np.arange(-L, L + 1)
  Et = np.exp(-1j * np.outer(theta, np.arange(-L, L + 1)))
  Er = np.exp(1j * np.outer(rho, m))
  f = np.einsum("nm,nm->n", Er, Et @ G.T)
  return (2 * np.real(f) if real else f).reshape(v.shape)


def sphericalSeries(fhat, v, real=None):
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
  elif v.size < DIRECT_TRIG_NODES_S2:
    f = _sphericalSeriesTrig(fhat, v, real)
  else:
    f = S2FourierPlan(v, L, real)(fhat).reshape(v.shape)
  return f[()] if f.ndim == 0 else f


def sphericalSeriesAdjoint(v, values, L, weights=None, real=False):
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
      fhat[l * l:(l + 1) ** 2] = np.conj(sphericalY(l, v)) @ (weights * values)
    return fhat
  return S2FourierPlan(v, L, real).adjoint(values, weights)


def fourierSeries(chat, x, real=None):
  chat = np.asarray(chat)
  L = (chat.shape[-1] - 1) // 2
  real = isRealFourier(chat) if real is None else real
  if isinstance(x, quadratureS1Grid):
    x = x.nodes()
  x = np.asarray(x)
  if x.size < DIRECT_SUM_NODES:
    f = np.exp(1j * np.outer(x.ravel(), np.arange(-L, L + 1))) @ chat
  else:
    f = S1FourierPlan(x, L)(chat)
  f = f.reshape(x.shape)
  f = np.real(f) if real else f
  return f[()] if f.ndim == 0 else f


def fourierSeriesAdjoint(x, values, L, weights=None):
  if isinstance(x, quadratureS1Grid):
    weights = x.weights if weights is None else weights
    x = x.nodes()
  x = np.asarray(x).ravel()
  values = np.asarray(values).ravel()
  weights = np.full(x.size, 1 / x.size) if weights is None else np.asarray(weights).ravel()
  if x.size < DIRECT_SUM_NODES:
    return np.exp(-1j * np.outer(np.arange(-L, L + 1), x)) @ (weights * values)
  return S1FourierPlan(x, L).adjoint(values, weights)


__all__ = ["halfPiDegrees", "foldedDims", "usableFolds", "wignerTransform", "wignerTransformAdjoint",
           "sphericalTransform", "sphericalTransformAdjoint", "isRealWigner", "isRealSpherical", "isRealFourier",
           "SO3FourierPlan", "S2FourierPlan", "S1FourierPlan", "SO3GridPlan", "S2GridPlan", "adjointTransform",
           "wignerPlan", "sphericalPlan", "wignerSeries", "wignerSeriesAdjoint", "sphericalSeries",
           "sphericalSeriesAdjoint", "fourierSeries", "fourierSeriesAdjoint"]
