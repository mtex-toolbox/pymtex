# the automatic halfwidth of a kernel density estimate on the sphere and on SO(3), selected
# from the vector of empirical degree energies (docs/plans/halfwidth.md)
#
#   E = degreeEnergies(v | ori, bandwidth=32, weights=, groups=, folds=8, rng=)
#   h = selectHalfwidth(E, method='UCV' | 'conservative', halfwidths=, noiseFloor=True, z=, tau=, tauMax=, tauRel=)
#   hUCV, hC2, hC3 = ucvHalfwidth(E), lowerBoundHalfwidth(E), noiseFloorHalfwidth(E)
#
# In harmonics orthonormal for the uniform probability a density is f = 1 + sum_l f_l, the
# degree energies A_l = ||f_l||^2, and a zonal kernel with multipliers b_l gives the risk
#   MISE(b) = sum_{l>=1} (1 - b_l)^2 A_l + b_l^2 V_l,   V_l = E ||mu_l||^2 - A_l,
# V_l = s2 (c_l - A_l) for independent points of weights w (s2 = sum w^2, c_l the mean of the
# diagonal of the degree l reproducing kernel of the invariant functions, d_l without symmetry).
from __future__ import annotations

import functools
import math

import numpy as np

from ..geometry.kernels import qconj
from ..geometry.kernels import qmul
from ..geometry.orientation import orientation
from ..geometry.quaternion import quaternion, rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, asVector3d
from .kernelfunctions import S2DeLaValleePoussinKernel, SO3DeLaValleePoussinKernel
from .quadrature import legendreP
from .transforms import S2FourierPlan, SO3FourierPlan
from ..geometry.display import aligned

# the points the symmetric traces c_l are averaged over
TRACE_SAMPLE = 2000
# the points the maximum of a pilot estimate is taken from, and the ones it is taken at
PILOT_SAMPLE = 20000
PILOT_POINTS = 200
# a multiplier below this counts as zero: the degree extent of a kernel and of its fluctuations
NEGLIGIBLE = 1e-3
# the tolerance of the transforms: the energies move by about this relative amount
TRANSFORM_EPS = 1e-6


# the characters chi_l(omega) = U_2l(cos omega/2) = sin((2l+1) omega/2) / sin(omega/2) at the
# cosine of half the angle, degree first, by U_{2l+2} = (4t^2 - 2) U_2l - U_{2l-2}
def _characters(t, L):
  U = np.empty((L + 1,) + t.shape)
  U[0] = 1
  y = 4 * t * t - 2
  if L >= 1:
    U[1] = y + 1
  for l in range(1, L):
    U[l + 1] = y * U[l] - U[l - 1]
  return U


def kernelMultipliers(psi, L):
  """the multipliers b_l = A_l / ((2l+1) A_0) of a zonal kernel, the same on S2 (Legendre) and on
  SO(3) (characters), up to the degree L"""
  A = np.asarray(psi._rule(int(L)), dtype=float)
  return A / ((2 * np.arange(L + 1) + 1) * A[0])


def kernelDegree(psi, cap=4096):
  """the degree beyond which the multipliers of a kernel stay below NEGLIGIBLE"""
  L = min(int(psi.bandwidth), cap)
  b = kernelMultipliers(psi, L)
  big = np.flatnonzero(np.abs(b) >= NEGLIGIBLE)
  return int(big[-1]) if big.size else 0


def checkIndependent(x, groups):
  """an automatic halfwidth on the pixels of a map needs their grains (docs/plans/halfwidth.md,
  section 10): neighbouring pixels repeat one orientation and would pass for independent draws"""
  if isinstance(x, orientation) and x.notIID and groups is None:
    raise ValueError("the orientations come from a map, whose neighbouring pixels are not independent: pass their "
                     "grain ids (groups=ebsd.grainId), the map with its grains computed, or the grain mean "
                     "orientations, or choose the halfwidth explicitly")


class degreeEnergies:
  """The empirical degree energies of a sample on S2 or SO(3), cross-fitted over folds.

  energy[l]  ||mu_l||^2 of the whole weighted sample
  A[l]       the unbiased estimate of the degree energy of the density, from the products of
             disjoint folds (grain-disjoint when groups are given)
  se[l]      its jackknife error over the folds, NaN without folds
  s2, wmax   sum w_i^2 and max w_i over the independent units (points, or groups)

  folds < 2 computes independent points in one transform, the energies then being the
  U-statistic (||mu_l||^2 - sum_i w_i^2 c_l(X_i)) / (1 - s2) with the traces averaged over a
  subsample; it serves UCV, which needs no errors.
  """

  # the sample picks the space: orientations, or a map of them, are on SO(3), directions on S2
  def __new__(cls, x, *args, **kwargs):
    if cls is degreeEnergies:
      cls = SO3DegreeEnergies if isinstance(x, quaternion) or _isMap(x) else S2DegreeEnergies
    return super().__new__(cls)

  def __init__(self, x, bandwidth=32, *, weights=None, groups=None, folds=8, antipodal=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    # a map: the orientations of its one phase, folded by its grains
    if _isMap(x):
      if groups is None and "grainId" in getattr(x, "prop", {}):
        groups = np.asarray(x.prop["grainId"]).reshape(-1)
      x = x.orientations
    checkIndependent(x, groups)
    self.L = L = int(bandwidth)
    pts, w, groups = self._points(x, weights, groups, antipodal, L)
    n = len(w)
    if n < 2:
      raise ValueError("at least two points are needed to estimate the degree energies")
    w = w / np.sum(w)
    self.n, self.grouped = n, groups is not None
    # the independent units: the points, or the groups, whose weights enter the variance
    if groups is None:
      unit, W = np.arange(n), w
    else:
      _, unit = np.unique(groups, return_inverse=True)
      W = np.bincount(unit, w)
    self.nUnits = len(W)
    if self.nUnits < 2:
      raise ValueError("at least two independent groups are needed to estimate the degree energies")
    self.s2, self.wmax = float(np.sum(W * W)), float(np.max(W))
    self.neff = 1 / self.s2
    # samples of the weighted measure: for the symmetric traces by w^2, the weight of a point in
    # the variance, and for the maximum of the pilot estimate by w, over the symmetric copies
    k = self._draw(rng, w * w / np.sum(w * w), TRACE_SAMPLE, weights is None)
    self._sample = pts[k]
    self._sampleW = w[k] ** 2 / np.sum(w[k] ** 2) if len(k) == n else np.full(len(k), 1 / len(k))
    self._dims = None
    m = max(PILOT_SAMPLE // self.nSym, 2000)
    if n <= m:
      self._pilot, self._pilotW = pts, w
    else:
      self._pilot, self._pilotW = pts[self._draw(rng, w, m, weights is None)], np.full(m, 1 / m)
    F = self.nFolds = int(min(folds or 1, self.nUnits))
    if F < 2:
      if groups is not None:
        raise ValueError("grouped points need folds: their energies come from products of disjoint groups")
      c = self._coefficients(pts, w, L)
      self.energy = np.add.reduceat(np.abs(c) ** 2, self._degreeStarts(L))
      self.A = (self.energy - self.s2 * self.dims(L)[0]) / (1 - self.s2)
      self.se = np.full(L + 1, np.nan)
      self.A[0], self.se[0] = 1.0, 0.0
      return
    # equal points fall into folds at random; groups and weighted points in random order are cut
    # into folds of about equal weight
    if groups is None and weights is None:
      fold = rng.integers(0, F, n, dtype=np.uint16)
    else:
      order = rng.permutation(self.nUnits)
      mid = np.cumsum(W[order]) - W[order] / 2
      foldOfUnit = np.empty(self.nUnits, dtype=int)
      foldOfUnit[order] = np.minimum((F * mid).astype(int), F - 1)
      fold = foldOfUnit[unit].astype(np.uint16)
    # the points sorted by fold, each fold one slice; a radix sort on the small integers
    order = np.argsort(fold, kind="stable")
    cut = np.concatenate([[0], np.cumsum(np.bincount(fold, minlength=F))])
    ps, ws = pts[order], w[order]
    C = np.stack([self._coefficients(ps[cut[f]:cut[f + 1]], ws[cut[f]:cut[f + 1]], L) for f in range(F)])
    Wf = np.bincount(fold, w, minlength=F)
    self._crossFit(C, Wf)

  # ---- the sample

  # m indices drawn from the weighted measure: all of them when there are no more, distinct
  # ones for equal weights, else with replacement by weight
  @staticmethod
  def _draw(rng, w, m, equal):
    n = len(w)
    if n <= m:
      return np.arange(n)
    return rng.choice(n, m, replace=False) if equal else rng.choice(n, m, p=w)

  # the points with their weights and groups, those that are not there or weigh nothing left out
  def _points(self, x, weights, groups, antipodal, L):
    pts = self._read(x, antipodal, L)
    w = np.ones(len(pts)) if weights is None else np.asarray(weights, dtype=float).reshape(-1).copy()
    if len(w) != len(pts):
      raise ValueError("one weight per point")
    keep = np.isfinite(pts[:, 0]) & (w > 0)
    if groups is not None:
      groups = np.asarray(groups).reshape(-1)
      if len(groups) != len(pts):
        raise ValueError("one group id per point")
    if np.all(keep):
      return pts, w, groups
    return pts[keep], w[keep], None if groups is None else groups[keep]

  # ---- the energies

  # the fold Gram matrices G[l, f, g] = Re <C_f,l, C_g,l>, the cross-fitted energies and their
  # jackknife over the folds: E <C_f,l, C_g,l> = W_f W_g A_l for disjoint folds
  def _crossFit(self, C, Wf):
    L, F = self.L, len(Wf)
    P = np.real(C[:, None, :] * np.conj(C[None, :, :]))
    starts = self._degreeStarts(L)
    G = np.add.reduceat(P, starts, axis=2).transpose(2, 0, 1)
    self.energy = G.sum(axis=(1, 2))
    diag = np.einsum("lff->lf", G)
    S = self.energy - diag.sum(axis=1)
    D = 1 - np.sum(Wf ** 2)
    self.A = S / D
    # leave one fold out: the products among the others
    rows = G.sum(axis=2) - diag
    Sk = S[:, None] - 2 * rows
    Dk = (1 - Wf) ** 2 - (np.sum(Wf ** 2) - Wf ** 2)
    Ak = Sk / Dk[None, :]
    self.se = np.sqrt((F - 1) / F * np.sum((Ak - Ak.mean(axis=1, keepdims=True)) ** 2, axis=1))
    self.A[0], self.se[0] = 1.0, 0.0

  # ---- the symmetric traces

  def dims(self, L):
    """the mean of the traces weighted by w^2, E c_l, and their maximum over the subsample, computed once
    to a degree beyond the one asked and cut"""
    L = int(L)
    if self._dims is None or len(self._dims[0]) <= L:
      c = self._traces(max(L, 2 * (len(self._dims[0]) - 1) if self._dims is not None else L, 64))
      self._dims = (c @ self._sampleW, c.max(axis=1))
    return self._dims[0][:L + 1], self._dims[1][:L + 1]

  def estimateMax(self, h):
    """the largest value of the estimate of halfwidth h at the pilot points, the symmetrised kernel
    summed over the pilot sample"""
    psi = self.kernel(halfwidth=float(h))
    at = self._pilot[:PILOT_POINTS]
    copies = self._copies()
    # beyond three halfwidths every member is below 2e-3 of its peak; the arguments are raised
    # to that cut, where the powers would otherwise run into slow denormal numbers, and zeroed
    t0 = self._cosineOf(min(3 * float(h), np.pi))
    f = 0
    for g in copies:
      t = self._cosines(at, g)
      f = f + (psi._eval(np.maximum(t, t0)) * (t > t0)) @ self._pilotW
    return float(np.max(f) / len(copies) / psi.mean())

  # the symmetric copies of the pilot sample
  def _copies(self):
    return self._symmetricCopies(self._pilot)

  # ---- the variance term

  def variance(self, a, L):
    """V_l for l = 0..L given energies a (zero above their length): s2 (c_l - a_l) for independent
    points; for groups the measured ||mu_l||^2 - A_l where the energies were computed, and above
    the iid form times the design effect of the highest measured degrees"""
    cbar = self.dims(L)[0]
    if not self.grouped:
      aa = np.zeros(L + 1)
      k = min(len(a), L + 1)
      aa[:k] = a[:k]
      V = self.s2 * np.maximum(cbar - aa, 0)
    else:
      # the unbiased energies, not the ones given: counting the uncertainty of lower bounds in
      # the variance too widened the selection by half again (plan, H2)
      k = min(self.L, L) + 1
      A = np.zeros(L + 1)
      A[:k] = np.maximum(self.A[:k], 0)
      V = self.s2 * np.maximum(cbar - A, 0)
      V[:k] = np.maximum(self.energy[:k] - self.A[:k], V[:k])
      V[k:] *= self.designEffect()
    V[0] = 0
    return V

  def designEffect(self):
    """the ratio of the measured variance to the iid one over the upper quarter of the degrees"""
    cbar = self.dims(self.L)[0]
    lo = max(1, (3 * self.L) // 4)
    iid = self.s2 * np.maximum(cbar[lo:] - self.A[lo:], 1e-300)
    return float(max(1.0, np.median((self.energy[lo:] - self.A[lo:]) / iid)))

  def __repr__(self):
    return (aligned(f"degreeEnergies on {self.space}, bandwidth {self.L}, {self.n} points, {self.nUnits} "
            f"{'groups' if self.grouped else 'units'}, {self.nFolds} folds, n_eff {self.neff:.0f}"))


class S2DegreeEnergies(degreeEnergies):
  """the degree energies of a sample of directions, degreeEnergies on S2"""
  space = "S2"
  kernel = S2DeLaValleePoussinKernel

  def _read(self, x, antipodal, L):
    v = asVector3d(x)
    pts = v.data.reshape(-1, 3)
    pts = pts / np.sqrt(np.einsum("ij,ij->i", pts, pts))[:, None]
    self.antipodal = bool(v.antipodal) if antipodal is None else bool(antipodal)
    sym = getattr(v.frame, "sym", None)
    self.frame = v.frame
    # the group acting on directions, as orthogonal matrices, the antipode among them
    M = np.eye(3)[None] if sym is None else sym.rot.matrix().reshape(-1, 3, 3)
    if self.antipodal and not any(np.allclose(m, -np.eye(3)) for m in M):
      M = np.concatenate([M, -M])
    self._group, self._sym = M, sym
    return pts

  # the coefficients of the weighted point measure of one fold, orthonormal for the uniform
  # probability, projected onto the invariant functions as the estimate is
  def _coefficients(self, pts, w, L):
    from .s2fun import evenize, projectSpherical
    c = np.zeros((L + 1) ** 2, dtype=complex)
    if len(w):
      c = math.sqrt(4 * np.pi) * S2FourierPlan(vector3d(pts), L, real=True, eps=TRANSFORM_EPS).adjoint(np.ones(len(w)), w)
    if self._sym is not None and self._sym.numSym() > 1:
      projectSpherical(c, self._sym)
    return evenize(c) if self.antipodal else c

  @property
  def nSym(self):
    """the number of symmetric copies of a point, the antipode included"""
    return len(self._group)

  # the symmetric copies s x of points
  def _symmetricCopies(self, pts):
    return [pts @ m.T for m in self._group]

  def _degreeStarts(self, L):
    return np.arange(L + 1) ** 2

  # the dimension of degree l without symmetry
  def _fullDims(self, L):
    return 2 * np.arange(L + 1) + 1.0

  # c_l(x) at the subsample, degree first: the diagonal of the projected reproducing kernel,
  # (2l+1) mean_s P_l(x . s x)
  def _traces(self, L):
    M = self._group
    if len(M) == 1:
      return np.broadcast_to(self._fullDims(L)[:, None], (L + 1, len(self._sample)))
    t = np.einsum("ni,gij,nj->ng", self._sample, M, self._sample)
    P = legendreP(L, np.clip(t, -1, 1))
    return (2 * np.arange(L + 1) + 1.0)[:, None] * P.mean(axis=2)

  # the kernel's argument at an angle, and between points and copies: the cosine of the angle
  def _cosineOf(self, angle):
    return math.cos(angle)

  def _cosines(self, at, g):
    return np.clip(at @ g.T, -1, 1)


class SO3DegreeEnergies(degreeEnergies):
  """the degree energies of a sample of orientations, degreeEnergies on SO(3)"""
  space = "SO3"
  kernel = SO3DeLaValleePoussinKernel

  def _read(self, x, antipodal, L):
    o = x if isinstance(x, orientation) else orientation(x)
    self.antipodal = bool(o.antipodal) if antipodal is None else bool(antipodal)
    self.CS, self.SS = o.CS, o.SS
    self._group = (o.SS.sym.properGroup().rotations().data.reshape(-1, 4),
                   o.CS.sym.properGroup().rotations().data.reshape(-1, 4))
    from .harmonics import invariantProjection
    self._projections = (invariantProjection(o.SS, L), invariantProjection(o.CS, L))
    return o.data.reshape(-1, 4)

  # the coefficients of the weighted point measure of one fold, orthonormal for the uniform
  # probability, projected onto the invariant functions as the estimate is
  def _coefficients(self, pts, w, L):
    from .harmonics import deg2dim, project
    from .so3fun import invertedCoefficients
    from .quadrature import folds
    from .transforms import foldedDims
    c = np.zeros(deg2dim(L + 1), dtype=complex)
    if len(w):
      # the z folds of the groups shrink the lattice, which pays where the FFT outweighs the
      # spreading, with fewer points than a quarter of the lattice; 1.25 oversampling, since
      # finufft spreads a thin lattice at 2.0 on many threads up to a hundred times slower
      fd = folds(self.CS, self.SS) if len(w) < np.prod(foldedDims(L, True, (1, 1))) / 4 else (1, 1)
      up = 1.25 if fd != (1, 1) else None
      c = SO3FourierPlan(rotation._fromData(pts), L, real=True, folds=fd, eps=TRANSFORM_EPS, upsampfac=up).adjoint(np.ones(len(w)), w)
    if self.nSym > 1:
      c = project(c, *self._projections)
    return (c + invertedCoefficients(c)) / 2 if self.antipodal else c

  @property
  def nSym(self):
    """the number of symmetric copies of a point, the antipode included"""
    return len(self._group[0]) * len(self._group[1]) * (2 if self.antipodal else 1)

  # the symmetric copies a x b of points, and a x^-1 b for antipodal ones
  def _symmetricCopies(self, pts):
    GS, GC = self._group
    sides = [pts, qconj(pts)] if self.antipodal else [pts]
    return [qmul(qmul(a, y), b) for y in sides for a in GS for b in GC]

  def _degreeStarts(self, L):
    from .harmonics import deg2dim
    return deg2dim(np.arange(L + 1))

  # the dimension of degree l without symmetry
  def _fullDims(self, L):
    return (2 * np.arange(L + 1) + 1.0) ** 2

  # c_l(x) at the subsample, degree first: the diagonal of the projected reproducing kernel,
  # (2l+1) mean_{a,b} chi_l(x^-1 a x b)
  def _traces(self, L):
    if self.nSym == 1:
      return np.broadcast_to(self._fullDims(L)[:, None], (L + 1, len(self._sample)))
    x = self._sample
    t = np.stack([np.abs(np.sum(x * y, axis=1)) for y in self._symmetricCopies(x)], axis=1)
    # the mean character over the copies, degree by degree through U_{2l+2} = (4t^2-2) U_2l - U_{2l-2}
    c = np.empty((L + 1, len(x)))
    y2 = 4 * t * t - 2
    u0, u1 = np.ones_like(t), y2 + 1
    c[0] = 1
    for l in range(1, L + 1):
      c[l] = (2 * l + 1) * u1.mean(axis=1)
      u0, u1 = u1, y2 * u1 - u0
    return c

  # the kernel's argument at an angle, and between points and copies: the cosine of half the angle
  def _cosineOf(self, angle):
    return math.cos(angle / 2)

  def _cosines(self, at, g):
    return np.minimum(np.abs(at @ g.T), 1)


# a map of orientations, which the estimate reads the orientations and the grains of
def _isMap(x):
  return not isinstance(x, (quaternion, vector3d)) and hasattr(x, "orientations")


# ------------------------------------------------------------------ the selectors


def _family(space):
  return S2DegreeEnergies.kernel if space == "S2" else SO3DegreeEnergies.kernel


@functools.lru_cache(maxsize=4096)
def familyMultipliers(space, h):
  """the multipliers of the family's kernel of halfwidth h up to its degree extent, computed once
  per halfwidth"""
  psi = _family(space)(halfwidth=float(h))
  b = kernelMultipliers(psi, max(kernelDegree(psi), 1))
  b.flags.writeable = False
  return b


def candidateHalfwidths(space, hmin=None, hmax=None, H=60):
  """the candidate halfwidths, geometric between hmin and hmax"""
  hmin = (1 if space == "S2" else 2) * degree if hmin is None else hmin
  hmax = (60 if space == "S2" else 40) * degree if hmax is None else hmax
  return np.geomspace(hmin, hmax, int(H))


def riskCurve(E, halfwidths, a):
  """the risk R(h; a, V) at every candidate, the energies a zero above their length; above the
  kernel's extent the multipliers are zero and the energies are all bias"""
  a = np.asarray(a, dtype=float)
  tail = np.concatenate([np.cumsum(a[::-1])[::-1], [0.0]])
  out = np.empty(len(halfwidths))
  for k, h in enumerate(halfwidths):
    b = familyMultipliers(E.space, h)
    Lk = len(b) - 1
    aa = np.zeros(Lk + 1)
    m = min(len(a), Lk + 1)
    aa[:m] = a[:m]
    out[k] = np.sum(((1 - b) ** 2 * aa + b * b * E.variance(a, Lk))[1:]) + tail[min(Lk + 1, len(a))]
  return out


# the largest minimiser, the side of the tie the conservative argument needs
def _largestArgmin(r):
  return len(r) - 1 - int(np.argmin(r[::-1]))


# the candidates whose multipliers vanish above the bandwidth of the energies
def _resolved(E, hs):
  return np.array([len(familyMultipliers(E.space, h)) - 1 <= E.L for h in hs], dtype=bool)


def ucvHalfwidth(E, halfwidths=None, returnCurve=False):
  """C1: the unbiased risk estimate, over the candidates whose multipliers vanish above the
  bandwidth of the energies"""
  hs = candidateHalfwidths(E.space) if halfwidths is None else np.asarray(halfwidths, dtype=float)
  ok = _resolved(E, hs)
  if not np.any(ok):
    raise ValueError(f"no candidate halfwidth is resolved at bandwidth {E.L}; raise the bandwidth of the energies")
  hs = hs[ok]
  r = riskCurve(E, hs, E.A)
  h = float(hs[_largestArgmin(r)])
  return (h, hs, r) if returnCurve else h


def ucvAtEdge(E, halfwidths=None):
  """whether UCV chose the narrowest candidate the energies resolve while narrower ones exist:
  then the bandwidth of the energies, not the data, set the choice"""
  hs = candidateHalfwidths(E.space) if halfwidths is None else np.asarray(halfwidths, dtype=float)
  ok = _resolved(E, hs)
  return bool(np.any(ok) and not np.all(ok) and ucvHalfwidth(E, hs) <= np.min(hs[ok]))


def bonferroniZ(alpha, L0):
  """the one-sided normal quantile for a simultaneous bound over the degrees 1..L0 (Bonferroni)"""
  from scipy.special import ndtri
  return float(ndtri(1 - alpha / max(int(L0), 1)))


def lowerBoundHalfwidth(E, halfwidths=None, z=2.0, alpha=0.05, returnCurve=False):
  """C2: the risk with lower confidence bounds of the energies, zero above the bandwidth; if the
  bounds hold for every degree the largest minimiser is at least the MISE-optimal halfwidth.
  z=None takes the Bonferroni quantile of alpha over the degrees, for which that is a stated
  probability; the default z = 2 never fell below the oracle in the study of the plan"""
  hs = candidateHalfwidths(E.space) if halfwidths is None else np.asarray(halfwidths, dtype=float)
  if np.any(np.isnan(E.se)):
    raise ValueError("the lower bounds need the errors of the energies: compute them with folds")
  z = bonferroniZ(alpha, E.L) if z is None else z
  a = np.maximum(E.A - z * E.se, 0)
  a[0] = 1
  r = riskCurve(E, hs, a)
  h = float(hs[_largestArgmin(r)])
  return (h, hs, r) if returnCurve else h


def footprints(E, h):
  """the number of independent kernel footprints in the fundamental region: the region over the
  ball of the halfwidth, 2/(1 - cos h) on S2 and pi/(h - sin h) on SO(3), where a ball of
  radius h carries the Haar probability (h - sin h)/pi"""
  n = 2 / (1 - math.cos(h)) if E.space == "S2" else math.pi / (h - math.sin(h))
  return max(n / E.nSym, 1.0)


def bennettDeviation(sigma2, b, Lam):
  """Bennett's bound on a sum of independent summands below b with variance sigma2: the
  deviation t exceeded with probability exp(-Lam), from (sigma2/b^2) g(b t / sigma2) = Lam,
  g(u) = (1+u) log(1+u) - u, solved by bisection on u in [0, c + 10], where g exceeds c"""
  sigma2 = np.maximum(np.asarray(sigma2, dtype=float), 1e-300)
  c = Lam * b * b / sigma2
  lo, hi = np.zeros_like(c), c + 10
  for _ in range(60):
    mid = (lo + hi) / 2
    big = (1 + mid) * np.log1p(mid) - mid > c
    hi, lo = np.where(big, mid, hi), np.where(big, lo, mid)
  return sigma2 / b * hi


def noiseFloorHalfwidth(E, halfwidths=None, tau=None, tauMax=0.2, tauRel=0.5, alpha=0.05, pilot=None, returnCurve=False):
  """C3: the smallest halfwidth at which, with probability about 1 - alpha, no fluctuation of the
  estimate exceeds tau + tauRel K_h*f, tau = tauMax M unless given, M the maximum of the pilot
  estimate. The fluctuation at x is a sum of independent summands below b = wmax B (B the
  peak of the symmetrised kernel) with variance s2 R y (R its squared norm, y = K_h*f(x)),
  bounded by Bennett's inequality at every one of N independent kernel footprints,
  Lam = log(2 N / alpha); y runs over (0, M]. Realistic rather than proven constants: the
  footprints stand for the supremum, and R y for the local variance of a smooth density"""
  hs = candidateHalfwidths(E.space) if halfwidths is None else np.asarray(halfwidths, dtype=float)
  if pilot is None:
    pilot = ucvHalfwidth(E, hs) if np.any(np.isnan(E.se)) else lowerBoundHalfwidth(E, hs)
  M = E.estimateMax(pilot)
  tau = tauMax * M if tau is None else tau
  y = M * np.geomspace(1e-3, 1, 48)
  excess = np.empty(len(hs))
  for k, h in enumerate(hs):
    b = familyMultipliers(E.space, h)
    cbar, cmax = E.dims(len(b) - 1)
    B, R = float(np.sum(b * cmax)), float(np.sum(b * b * cbar))
    t = bennettDeviation(E.s2 * R * y, E.wmax * B, math.log(2 * footprints(E, h) / alpha))
    excess[k] = np.max(t - tauRel * y) - tau
  ok = np.flatnonzero(excess <= 0)
  h = float(hs[ok[0]]) if ok.size else float(hs[-1])
  return (h, hs, excess) if returnCurve else h


def selectHalfwidth(E, method="UCV", halfwidths=None, noiseFloor=True, z=2.0, alpha=0.05, tau=None, tauMax=0.2,
                    tauRel=0.5):
  """Select the angular halfwidth of a de la Vallee Poussin kernel for density estimation.

    from mtex.functions.halfwidth import degreeEnergies, selectHalfwidth
    E = degreeEnergies(v, bandwidth=256)
    h = selectHalfwidth(E)                         # select from the degree energies of directions
    h = selectHalfwidth(E, method='conservative')   # favour wider kernels when the energies are uncertain
    h = selectHalfwidth(E, noiseFloor=False)        # use the selected method without the noise floor

  The halfwidth controls how much the density estimate is smoothed: narrow kernels retain
  finer detail, while wide kernels smooth out more of the sampling noise. It is the angle
  at which the kernel falls to half its peak value. The result is in radians; divide by
  `degree` to express it in degrees.

  Both methods balance smoothing bias against sampling variance using the harmonic degree
  energies in E. 'UCV' (unbiased cross validation) minimises an unbiased estimate of the
  integrated squared error. 'conservative' uses lower confidence bounds on the energies,
  favouring wider kernels when fine detail is uncertain. It needs energies computed with
  folds and finite error estimates in E.se; the default eight folds are suitable.

  By default, a noise floor can widen either choice to reduce false peaks caused by
  sampling noise. It is the smallest candidate halfwidth whose estimated fluctuations
  stay below `tau + tauRel * localDensity`. Unless tau is supplied, tau is tauMax
  times the maximum of a pilot density estimate. This is an approximate noise criterion;
  alpha does not give a rigorous confidence guarantee for the entire density estimate.
  If no candidate meets it, the widest candidate is used.

  UCV considers only candidates whose kernels are resolved by E's harmonic bandwidth.
  Increase the bandwidth when the choice is limited by this resolution; `ucvAtEdge`
  detects that case. An error is raised if no candidate is resolved. The conservative
  method treats the density's energies above E's bandwidth as zero.

  Compute E with `degreeEnergies` from orientations, directions or an EBSD map. Pass
  weights and grain groups there; pixels within one grain must not be counted as
  independent observations. To select a kernel directly from a sample, use `calcKernel`.

  Input
    E - degreeEnergies object containing the sample's harmonic energies and statistics
  Keywords
    method - 'UCV' or 'conservative', case insensitive
    halfwidths - candidate halfwidths in radians, in increasing order; by default 60
        geometrically spaced values from 1 to 60 degrees for directions, or 2 to 40
        degrees for orientations
    noiseFloor - apply the additional lower bound that limits estimated sampling noise
    z - number of standard errors subtracted from the energies for 'conservative';
        None uses a Bonferroni normal quantile based on alpha
    alpha - nominal error probability for the approximate noise bound and, when z=None,
        the simultaneous lower confidence bounds on the energies
    tau - absolute noise tolerance in density units, relative to a uniform density of
        one; None uses tauMax times the pilot maximum
    tauMax - fraction of the pilot maximum used as the absolute tolerance when tau=None
    tauRel - additional noise tolerance as a fraction of the local density; increasing
        either tolerance allows narrower kernels
  Output
    h - selected halfwidth as a float in radians, chosen from the candidates

  Example
  Select a kernel halfwidth for grouped direction measurements and use it in a spherical
  density estimate. Two measurements per group share a synthetic grain direction.

    import numpy as np
    import matplotlib.pyplot as plt
    from mtex import vector3d, degree, calcDensity
    from mtex.functions.halfwidth import degreeEnergies, selectHalfwidth

    rng = np.random.default_rng(4)
    grainDirections = rng.normal([0, 0, 1], [0.3, 0.3, 0.05], size=(40, 3))
    measurements = np.repeat(grainDirections, 2, axis=0) + rng.normal(0, 0.01, size=(80, 3))
    v = vector3d(measurements).normalize()
    groups = np.repeat(np.arange(40), 2)
    energies = degreeEnergies(v, groups=groups, bandwidth=64, folds=8, rng=np.random.default_rng(5))
    candidates = np.linspace(5, 40, 36)*degree
    halfwidth = selectHalfwidth(energies, halfwidths=candidates, noiseFloor=False)
    density = calcDensity(v, halfwidth=halfwidth)
    theta = np.linspace(-60, 60, 121)*degree
    probe = vector3d(np.sin(theta), np.zeros_like(theta), np.cos(theta))
    print('Selected halfwidth (degrees):', round(float(halfwidth/degree), 2))
    plt.plot(theta/degree, density(probe))
    plt.xlabel('Angle from z in the x-z plane (degrees)')
    plt.ylabel('Direction density (mean one on the sphere)')
    plt.show()

  Group ids make the forty grain groups the independent sampling units for width selection.
  The noise floor is switched off here to isolate UCV and keep this demonstration short; it is
  enabled by default. Increase the energy bandwidth when considering narrower kernels.

  See also
    degreeEnergies, calcKernel, ucvHalfwidth, lowerBoundHalfwidth, noiseFloorHalfwidth, ucvAtEdge
  """
  m = str(method).lower()
  if m == "ucv":
    h = ucvHalfwidth(E, halfwidths)
  elif m == "conservative":
    h = lowerBoundHalfwidth(E, halfwidths, z=z, alpha=alpha)
  else:
    raise ValueError(f"the halfwidth selection method is 'UCV' or 'conservative', not {method!r}")
  if noiseFloor:
    h = max(h, noiseFloorHalfwidth(E, halfwidths, tau=tau, tauMax=tauMax, tauRel=tauRel, alpha=alpha, pilot=h))
  return h


__all__ = ["checkIndependent", "degreeEnergies", "selectHalfwidth", "ucvHalfwidth", "ucvAtEdge", "lowerBoundHalfwidth", "noiseFloorHalfwidth",
           "bennettDeviation", "footprints",
           "candidateHalfwidths", "kernelMultipliers", "riskCurve"]
