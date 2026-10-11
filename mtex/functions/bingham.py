# the Bingham distributions, on SO(3) as f(q) = w/C(kappa) exp(sum kappa_i <q, a_i>^2) with four
# orthonormal quaternions a_i, on the sphere as f(v) = exp(sum Z_i <v, a_i>^2) / C(Z); the
# constant C is the hypergeometric function of matrix argument 1F1(1/2; s/2; diag kappa),
# the mean of the exponential over the unit sphere of R^s, by the series of Hillier, Kan
# and Wang (2009) (MATLAB @SO3FunBingham, BinghamODF, calcBinghamODF, @S2FunBingham;
# docs/plans/functions.md F5)
from __future__ import annotations

from typing import TYPE_CHECKING
import math

import numpy as np

from .s2funclass import S2Fun
from ..geometry.frames import referenceFrame
from ..geometry.orientation import orientation
from ..geometry.quaternion import orthogonalTransform, quaternion, rotation
from ..geometry.session import getMTEXpref
from ..geometry.symreduce import reductionGroups
from ..geometry.vector3d import vector3d, asVector3d
from ..geometry.display import frameLabel, framesLabel
from .funops import _antipodalPole, _rotatedFrame
from .handles import S2FunHandle, SO3FunGeneric
from .rbf import _framesAmong, _rotationsFor
from .s2fun import S2FunHarmonic
from ..geometry.display import aligned
from ..geometry.framepair import installFramePair


# the coefficients c_1 .. c_s of prod (x - kappa_i) = x^s + c_1 x^{s-1} + ... + c_s
def _charpolyCoefficients(kappa):
  c = np.array([1.0])
  for k in kappa:
    c = np.concatenate([c, [0.0]]) - np.concatenate([[0.0], k * c])
  return c[1:]


def mhyper(kappa):
  """1F1(1/2; s/2; diag kappa) for a vector kappa of length s >= 2"""
  kappa = np.asarray(kappa, dtype=float).ravel()
  s = len(kappa)
  if s < 2:
    raise ValueError("the argument has at least two entries")
  # the mean moves by exp(c) with kappa + c: the series runs on non-negative entries, whose terms do not cancel
  c = float(np.min(kappa))
  kappa = kappa - c
  p = _charpolyCoefficients(kappa)[::-1]
  beta1 = s / 2
  d = np.ones(s) / beta1
  y = 1.0
  i = 0
  for j in range(1, s):
    i = j
    d[i] = np.sum(-(np.arange(i, 2 * i) / (2 * i)) * p[s - i:s] * d[:i])
    y += d[i]
    d /= beta1 + i
  circ = np.arange(1, s + 1) % s
  kp = np.arange(-s, 0) / 2 * p
  while np.linalg.norm(d) > np.finfo(float).eps * y:
    i += 1
    d[0] = np.sum(-(kp / i + p) * d)
    y += d[0]
    d = d[circ] / (beta1 + i)
  return float(math.exp(c) * y)


# d log 1F1 / d kappa_i by central differences, the moments E[x_i^2] of the distribution
def _logMhyperGradient(kappa, h=1e-5):
  kappa = np.asarray(kappa, dtype=float)
  out = np.empty(len(kappa))
  for i in range(len(kappa)):
    e = np.zeros(len(kappa))
    e[i] = h
    out[i] = (math.log(mhyper(kappa + e)) - math.log(mhyper(kappa - e))) / (2 * h)
  return out


# the concentrations with the given moments lambda (summing to one), the smallest fixed at zero
def _concentrationsFor(lam, iterations=50, tol=1e-10):
  lam = np.asarray(lam, dtype=float)
  s = len(lam)
  kappa = np.maximum(-1 / (2 * lam) - np.min(-1 / (2 * lam)), 0.0)
  kappa[s - 1] = 0
  free = slice(0, s - 1)
  for _ in range(iterations):
    r = _logMhyperGradient(kappa)[free] - lam[free]
    if np.linalg.norm(r) < tol:
      break
    J = np.zeros((s - 1, s - 1))
    for j in range(s - 1):
      e = np.zeros(s)
      e[j] = 1e-4
      J[:, j] = (_logMhyperGradient(kappa + e)[free] - _logMhyperGradient(kappa - e)[free]) / 2e-4
    kappa[free] -= np.linalg.solve(J, r)
  return kappa


def _axesMatrix(A):
  if isinstance(A, orientation):
    return rotation(A).data.reshape(-1, 4).T, (A.CS, A.SS), A.antipodal
  if isinstance(A, quaternion):
    return A.data.reshape(-1, 4).T, None, False
  return np.asarray(A, dtype=float), None, False


def binghamSample(kappa, A, n, rng):
  """n quaternions of the density exp(q' A diag(kappa) A' q) on S^3, exactly: rejection from the
  angular central Gaussian envelope of Kent, Ganeiber and Mardia (arXiv 1310.8110), written for
  exp(-x' B x) with B = A diag(max(kappa) - kappa) A' positive semidefinite"""
  from scipy.optimize import brentq
  lam = 2 * (np.max(kappa) - np.asarray(kappa, dtype=float))
  # b solves sum 1 / (b + lam_i) = 1 in (0, 4]
  b = 4.0 if np.all(lam < 1e-12) else brentq(lambda b: np.sum(1 / (b + lam)) - 1, 1e-10, 4.0)
  omega = 1 + lam / b
  logM = -(4 - b) / 2 + 2 * np.log(4 / b)
  out = np.empty((0, 4))
  while len(out) < n:
    m = 2 * (n - len(out)) + 16
    y = rng.standard_normal((m, 4)) / np.sqrt(omega)
    x = y / np.linalg.norm(y, axis=1, keepdims=True)
    xBx = (x ** 2) @ (lam / 2)
    xOx = (x ** 2) @ omega
    accept = np.log(rng.random(m)) < -xBx + 2 * np.log(xOx) - logM
    out = np.concatenate([out, x[accept]])
  return out[:n] @ np.asarray(A, dtype=float).T


class SO3FunBingham(SO3FunGeneric):
  """the Bingham distribution on SO(3): the density exp(q' A diag(kappa) A' q) on the quaternions,
  symmetrised over the crystal and specimen frames, with four concentrations kappa and four
  orthonormal quaternion axes A; BinghamODF builds it from axes, orientations or a fibre

    SO3FunBingham(kappa, A, cs, ss)
    SO3FunBingham(kappa, A, cs, cs, antipodal=True)      with exchange symmetry"""

  def __init__(self, kappa, A, *frames, weight=1.0, antipodal=False):
    A, framesOf, exchange = _axesMatrix(A)
    frames = frames if frames or framesOf is None else framesOf
    self.CS, self.SS = _framesAmong(frames)
    self.kappa = np.asarray(kappa, dtype=float).ravel()
    if len(self.kappa) != 4 or A.shape != (4, 4):
      raise ValueError("four concentrations and a 4x4 matrix of axes")
    if abs(abs(np.linalg.det(A)) - 1) > 1e-6:
      raise ValueError("the axes must be orthonormal")
    antipodal = bool(antipodal or exchange)
    if antipodal and self.CS is not self.SS:
      raise ValueError("exchange symmetry needs the same frame on both sides")
    self.A, self.weight, self.antipodal = A, float(weight), antipodal
    self.C0 = mhyper(self.kappa)

  @property
  def bandwidth(self):
    """the bandwidth of the harmonic expansion, the preference maxSO3Bandwidth"""
    return getMTEXpref("maxSO3Bandwidth")

  @property
  def axes(self):
    """the four principal quaternion axes as orientations, MATLAB's odf.A"""
    return orientation._fromRotation(self.A.T.copy(), (self.CS, self.SS))

  def mean(self):
    """the mean value of the function over SO(3), its weight"""
    return self.weight

  # the axes moved by every pair of the groups, <l q r, a> = <q, l^-1 a r^-1>, and the
  # conjugates under exchange symmetry, one 4x4 matrix per copy
  def _symmetricAxes(self):
    L, R = reductionGroups(orientation(rotation.id, self.CS, self.SS))
    axes = rotation(self.A.T)
    copies = []
    for l in rotation(L):
      for r in rotation(R):
        moved = l.inv() * axes * r.inv()
        copies.append(moved.data.T)
        if self.antipodal:
          copies.append(moved.inv().data.T)
    return np.stack(copies)

  def eval(self, x):
    """the values of the distribution at orientations, the mean over the symmetric copies of its axes"""
    r = _rotationsFor(self, x)
    q = r.data.reshape(-1, 4)
    copies = self._symmetricAxes()
    from ..geometry.kernels import backend
    if backend() == "numba":
      from .kernels import binghamSum
      acc = binghamSum(q, copies, self.kappa)
    else:
      acc = np.zeros(len(q))
      for M in copies:
        acc += np.exp(((q @ M) ** 2) @ self.kappa)
    return (self.weight / self.C0 * acc / len(copies)).reshape(r.shape)

  def __mul__(self, a):
    if isinstance(a, (int, float, np.number)):
      return SO3FunBingham(self.kappa, self.A, self.CS, self.SS, weight=a * self.weight, antipodal=self.antipodal)
    from .so3fun import SO3FunHarmonic
    return SO3FunHarmonic(self) * a

  __rmul__ = __mul__

  def rotate(self, r, side="left"):
    """the distribution turned by a rotation, from the left (the specimen side) or, with
    side='right', from the right (the crystal side)"""
    rot = rotation(r.data.reshape(4)) if not isinstance(r, orientation) else rotation(r)
    axes = rotation(self.A.T)
    if side == "left":
      moved, CS, SS = rot * axes, self.CS, _rotatedFrame(self.SS, rot)
    elif side == "right":
      moved, CS, SS = axes * rot, _rotatedFrame(self.CS, rot), self.SS
    else:
      raise ValueError(f"the side is 'left' or 'right', not {side!r}")
    return SO3FunBingham(self.kappa, moved.data.T, CS, SS, weight=self.weight, antipodal=self.antipodal and CS is SS)

  # the pole figure of one crystal direction by Kunze and Schaeben: on the fibre q(phi) =
  # cos(phi/2) q1 + sin(phi/2) q2 with q1 h = r and q2 = q1 (h, pi), the exponent is a + b cos phi
  # + c sin phi and the mean over the fibre is exp(a) I0(sqrt(b^2 + c^2))
  def _poleFigure(self, h, bandwidth=None):
    from scipy.special import ive
    h = asVector3d(h)
    if h.size != 1:
      raise ValueError("one crystal direction at a time")
    if h.frame is not None and h.frame is not self.CS:
      raise ValueError("the crystal direction must live in the crystal frame of the function")
    hn = vector3d(h.normalize().data.reshape(3))
    copies = self._symmetricAxes()
    flip = rotation.byAxisAngle(hn, np.pi)

    def value(rs):
      rn = vector3d(rs.data.reshape(-1, 3)).normalize()
      q1 = rotation.map(hn, rn)
      q1 = rotation(q1.data.reshape(-1, 4))
      q2 = (q1 * flip).data.reshape(-1, 4)
      q1 = q1.data
      acc = np.zeros(len(q1))
      for M in copies:
        A1, A2 = q1 @ M, q2 @ M
        a = ((A1 ** 2 + A2 ** 2) @ self.kappa) / 2
        b = ((A1 ** 2 - A2 ** 2) @ self.kappa) / 2
        c = (A1 * A2) @ self.kappa
        acc += np.exp(a) * ive(0, np.hypot(b, c)) * np.exp(np.hypot(b, c))
      return (self.weight / self.C0 * acc / len(copies)).reshape(rs.shape)

    L = min(64, getMTEXpref("maxS2Bandwidth")) if bandwidth is None else int(bandwidth)
    return S2FunHarmonic(S2FunHandle(value, self.SS, bandwidth=L, antipodal=_antipodalPole(h, self.CS)))


  # the inverse pole figure of one specimen direction, through the harmonic form
  def _inversePoleFigure(self, r, bandwidth=None):
    from .so3fun import SO3FunHarmonic
    return SO3FunHarmonic(self, bandwidth=bandwidth)._inversePoleFigure(r)

  def radon(self, h=None, r=None, superposition=None, bandwidth=None):
    """the Radon transform: radon(h) the pole figure, in closed form, radon(r=r) the inverse pole
    figure, radon(h, r) the values; several h superposed, weighted by superposition
    (MATLAB's pdf, ipdf and calcPDF)"""
    from .funops import radon
    return radon(self, h, r, superposition, bandwidth)
  def __repr__(self):
    lines = [f"SO3FunBingham ({framesLabel(self.CS, self.SS)})", f"  kappa: {np.array2string(self.kappa, precision=3)}", f"  weight: {self.weight:.4g}"]
    if self.antipodal:
      lines.append("  antipodal: true")
    return aligned("\n".join(lines))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    from .polefigure import calcPoleFigure
    from ..analysis.texture import calcTensor, calcLankford, calcKearnsFactor
    from ..analysis.optimalsample import optimalSample
    from ..plotting.sphericalplot import plotPF, plotIPF
    from ..plotting.sectionplot import plotSection
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


# the four quaternion axes of a fibre: the quaternions along which h is sent to r, and
# those across (MATLAB fibre2A)
def _fibreAxes(h, r):
  h = h.normalize().data.reshape(3)
  r = r.normalize().data.reshape(3)
  qr, qh = np.array([0.0, *r]), np.array([0.0, *h])
  from ..geometry.kernels import qmul
  rh = qmul(qr[None], qh[None]).reshape(4)
  q1 = np.array([1.0, 0, 0, 0]) - rh
  q2 = qr + qh
  if np.linalg.norm(h - r) < 1e-10:
    v1 = np.cross(h, [1.0, 0, 0] if abs(h[0]) < 0.9 else [0, 1.0, 0])
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(h, v1)
    q3, q4 = np.array([0.0, *v1]), np.array([0.0, *v2])
  else:
    q3 = np.array([1.0, 0, 0, 0]) + rh
    q4 = qr - qh
  A = np.stack([q1, q2, q3, q4], axis=1)
  return A / np.linalg.norm(A, axis=0)


def BinghamODF(kappa, A=None, *frames, **kwargs):
  """BinghamODF(kappa, U), BinghamODF(kappa, cs), BinghamODF(20, ori), BinghamODF(20, f), BinghamODF([10, 10, 10], quaternion.id, cs)

  A Bingham distribution from four concentrations and four orthogonal quaternion axes, the
  axes as a 4x4 matrix, four orientations, or one orientation standing for the identity
  basis; fewer concentrations are padded with zeros; a fibre takes its two axes with the
  same concentration, as MATLAB's constructor does.
  """
  from ..geometry.fibre import fibre
  from ..geometry.frames import referenceFrame
  kappa = np.asarray(kappa, dtype=float).ravel()
  # BinghamODF(kappa, cs): the identity basis
  if isinstance(A, referenceFrame):
    A, frames = None, (A,) + frames
  if isinstance(A, fibre):
    kappa = np.concatenate([kappa[:1], kappa[:1]]) if kappa.size == 1 else kappa
    frames = frames if frames else (A.CS, A.SS)
    A = _fibreAxes(A.h, A.r)
  elif isinstance(A, vector3d) and frames and isinstance(frames[0], vector3d):
    kappa = np.concatenate([kappa[:1], kappa[:1]]) if kappa.size == 1 else kappa
    A, frames = _fibreAxes(A, frames[0]), frames[1:]
  elif isinstance(A, quaternion) and A.size == 1:
    # one orientation: the identity basis, its frames
    if isinstance(A, orientation):
      frames = frames if frames else (A.CS, A.SS)
    A = np.eye(4)
  elif A is None:
    A = np.eye(4)
  if kappa.size < 4:
    kappa = np.concatenate([kappa, np.zeros(4 - kappa.size)])
  return SO3FunBingham(kappa, A, *frames, **kwargs)


def calcBinghamODF(o, weights=None, exact=False):
  """the estimate from orientations: the eigen decomposition of the quaternion tensor of the
  sample projected to the fundamental region about its mean gives the axes, and the
  concentrations follow from the eigenvalues, by the large concentration approximation
  kappa_i = -1/(2 lambda_i) shifted to a smallest value of zero, or exactly by Newton on the moments"""
  o = o if isinstance(o, orientation) else orientation(o)
  m = o.mean(weights=weights)
  q = o.projectIntoFundamentalRegion(m).data.reshape(-1, 4)
  w = np.full(len(q), 1 / len(q)) if weights is None else np.asarray(weights, dtype=float).ravel() / np.sum(weights)
  T = (q * w[:, None]).T @ q
  lam, A = np.linalg.eigh(T)
  order = np.argsort(lam)[::-1]
  lam, A = lam[order], A[:, order]
  if exact:
    kappa = _concentrationsFor(lam)
  else:
    kappa = -1 / (2 * lam)
    kappa = kappa - np.min(kappa)
  return SO3FunBingham(kappa, A, o.CS, o.SS, antipodal=o.antipodal)


# ------------------------------------------------------------------ the sphere


class S2FunBingham(S2Fun):
  """the Bingham distribution on the sphere: the density exp(v' a diag(Z) a' v) of the axes v, with
  three shape parameters Z and three orthonormal principal axes a, symmetrised over a frame

    S2FunBingham(Z)                       the coordinate axes
    S2FunBingham(Z, a)                    the axes as directions or a 3x3 matrix
    S2FunBingham.fit(v)                   the fit to directions"""
  isReal = True
  antipodal = True

  def __init__(self, Z, a=None, frame=None):
    if isinstance(a, referenceFrame):
      a, frame = None, a
    if isinstance(a, vector3d):
      frame = a.frame if frame is None else frame
      a = a.data.reshape(-1, 3).T
    self.Z = np.asarray(Z, dtype=float).ravel()
    self._A = np.eye(3) if a is None else np.asarray(a, dtype=float)
    if len(self.Z) != 3 or self._A.shape != (3, 3):
      raise ValueError("three shape parameters and a 3x3 matrix of axes")
    if abs(abs(np.linalg.det(self._A)) - 1) > 1e-6:
      raise ValueError("the axes must be orthonormal")
    self.frame = frame
    self.C = mhyper(self.Z)

  @property
  def CS(self):
    """the frame the distribution lives in, its symmetry"""
    return self.frame

  @property
  def a(self):
    """the principal axes as directions, the modal axis a[2], as MATLAB's a"""
    return vector3d(self._A.T, frame=self.frame)

  @property
  def bandwidth(self):
    """the bandwidth of the harmonic expansion, the preference maxS2Bandwidth"""
    return getMTEXpref("maxS2Bandwidth")

  def mean(self):
    """the mean value over the sphere, one: the distribution is normalised"""
    return 1.0

  def _symmetricAxes(self):
    if self.frame is None:
      return self._A[None]
    els = self.frame.sym.rot.flatten().rot
    return np.stack([(els[i].inv() * vector3d(self._A.T)).data.T for i in range(els.size)])

  def __call__(self, x):
    return self.eval(x)

  def eval(self, x):
    """the values of the density at directions, the mean over the symmetric copies of its axes"""
    v = asVector3d(x)
    d = v.normalize().data.reshape(-1, 3)
    copies = self._symmetricAxes()
    acc = np.zeros(len(d))
    for M in copies:
      acc += np.exp(((d @ M) ** 2) @ self.Z)
    return (acc / (self.C * len(copies))).reshape(v.shape)

  def _harmonicS2(self, bandwidth=None):
    L = self.bandwidth if bandwidth is None else int(bandwidth)
    return S2FunHarmonic(lambda v: self.eval(v), self.frame, bandwidth=L, antipodal=True, real=True)

  def rotate(self, r):
    """the distribution turned by a rotation, its axes and its frame"""
    rot = rotation(r.data.reshape(4))
    return S2FunBingham(self.Z, (rot * vector3d(self._A.T)).data.T, None if self.frame is None else _rotatedFrame(self.frame, rot))

  @classmethod
  def fit(cls, v, weights=None, confidenceEllipse=False, p=0.95):
    """the fit to directions or axes: the eigen decomposition of the scatter matrix gives the
    axes, Newton on the moments the shape parameters, ascending to zero at the mode, the
    third axis, as in MTEX; confidenceEllipse=True returns with it the semi-axes ab, in
    radians, and the rotation rot of the confidence ellipse at the level p of the modal axis
    (Tanaka 1999), for ellipse(rot, ab[0], ab[1])"""
    v = asVector3d(v)
    d = v.normalize().data.reshape(-1, 3)
    w = np.full(len(d), 1 / len(d)) if weights is None else np.asarray(weights, dtype=float).ravel() / np.sum(weights)
    S = (d * w[:, None]).T @ d
    lam, a = np.linalg.eigh(S)
    order = np.argsort(lam)[::-1]
    kappa = _concentrationsFor(lam[order])
    f = cls((kappa - kappa[0])[::-1], a[:, order[::-1]], v.frame)
    if not confidenceEllipse:
      return f
    from scipy.stats import chi2
    # the asymptotic covariance of the modal axis in the tangent plane of the other two
    A = f._A
    Y = d @ A
    lam = lam[order[::-1]] / lam.sum()
    g1, g2 = lam[2] - lam[0], lam[2] - lam[1]
    n = len(d)
    Sigma = np.array([[np.mean(Y[:, 0] ** 2 * Y[:, 2] ** 2) / (n * g1 ** 2), np.mean(Y[:, 0] * Y[:, 1] * Y[:, 2] ** 2) / (n * g1 * g2)],
                      [0.0, np.mean(Y[:, 1] ** 2 * Y[:, 2] ** 2) / (n * g2 ** 2)]])
    Sigma[1, 0] = Sigma[0, 1]
    E, V = np.linalg.eigh(Sigma)
    ab = np.sqrt(chi2.ppf(p, 2) * E)
    # the tangent basis turned onto the axes of the ellipse, the modal axis kept
    B = A.T.copy()
    B[:2] = V.T @ B[:2]
    if np.linalg.det(B) < 0:
      B[0] = -B[0]
    return f, ab, rotation.byMatrix(B.T)

  def __repr__(self):
    info = "" if self.frame is None else f" ({frameLabel(self.frame)})"
    return aligned(f"S2FunBingham{info}\n  Z: {np.array2string(self.Z, precision=3)}")

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from .characteristics import max_ as max, min_ as min, entropy, textureIndex, discreteSample
    from .discrepancy import discrepancy
    from ..analysis.optimalsample import optimalSample
    from ..plotting.sphericalplot import plot, contourf, contour, pcolor
    from ..plotting.spaceplot import plot3d


# the two frames under their names of the Julia port, frameA and frameB
installFramePair(SO3FunBingham)

__all__ = ["mhyper", "SO3FunBingham", "BinghamODF", "calcBinghamODF", "S2FunBingham"]
