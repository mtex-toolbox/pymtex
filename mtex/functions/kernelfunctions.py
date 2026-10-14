# radially symmetric kernels on SO(3), S2 and S1 as parameter objects; the coefficients of
# their series are derived from the rule of the family and never stored (MATLAB
# SO3KernelFunctions, S2KernelFunctions, S1KernelFunctions; docs/plans/functions.md F1)
#
#   SO(3): psi(t) = sum_n A_n U_{2n}(t),  t = cos(omega/2), Haar measure normalised, mean psi = A_0
#   S2:    psi(t) = sum_n A_n P_n(t),     t = cos theta,    surface measure normalised, mean psi = A_0
#   S1:    psi(x) = sum_k c_k e^{ikx},    c_{-k} = c_k, mean psi = c_0
#
#   psi = SO3DeLaValleePoussinKernel(20), SO3DeLaValleePoussinKernel(halfwidth=10*degree)
#   psi.A, psi.bandwidth, psi.calcFourier(L), psi.halfwidth()
#   psi(t), psi(rot), psi(rot1, rot2), psi.eval(t), psi.grad(t), psi.grad()
#   radon(psi), conv(psi1, psi2), 2*psi + psi2, norm(psi), mean(psi)
#
# A kernel is called on the cosine argument t (the angle x on S1) in the precision of the
# argument; the base classes SO3Kernel(A), S2Kernel(A), S1Kernel(fhat) are the kernels given
# by their coefficients and evaluate by Clenshaw, the families override the evaluation with
# their closed form where there is one.
from __future__ import annotations

import math
from functools import cached_property

import numpy as np

from ..geometry.display import numString
from ..geometry.quaternion import quaternion
from ..geometry.session import getMTEXpref
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .quadrature import gaussLegendre, legendreP
from ..geometry.display import aligned

KERNEL_MAX_BANDWIDTH = 2048


def _asFloat(t):
  t = np.asarray(t)
  return t if t.dtype.kind == "f" else t.astype(float)


def ive(n, x):
  """scipy.special is loaded on first use, never at import"""
  from scipy.special import ive
  return ive(n, x)


def _beta(a, b):
  from scipy.special import beta
  return beta(a, b)


# ------------------------------------------------------------------ the common part


class _Kernel:
  _label = "custom"

  # the rule of the family: the coefficients of degree 0 .. L
  def _rule(self, L):
    raise NotImplementedError

  # the coefficients at the kernel's bandwidth, computed once: a kernel is immutable
  @cached_property
  def _coefficients(self):
    return self._rule(self.bandwidth)

  # the closed form at the cosine argument, else the series
  def _eval(self, t):
    return self._clenshaw(self._coefficients, t)

  # ---- bandwidth: MATLAB's cutoff rule, the first degree n >= 1 at which A_n / (n+1)^2
  # drops to max(min(min_n A_n / (n+1)^2, 10 eps), eps), eps = FFTAccuracy / 150

  @cached_property
  def bandwidth(self):
    """the degree the series of the kernel is cut at, MATLAB's rule: the first coefficient that
    drops below the accuracy FFTAccuracy asks for"""
    return self._cutoffBandwidth()

  # over the 1000 coefficients MATLAB computes before it cuts
  def _cutoffBandwidth(self, cap=KERNEL_MAX_BANDWIDTH):
    eps = getMTEXpref("FFTAccuracy", 1e-2) / 150
    A = self._rule(min(1000, cap))
    s = np.real(A) / (np.arange(len(A)) + 1) ** 2
    if len(s) < 2:
      return len(s) - 1
    threshold = max(min(np.min(s[1:]), 10 * eps), eps)
    small = np.flatnonzero(s[1:] <= threshold)
    return int(small[0]) + 1 if small.size else len(s) - 1

  # ---- evaluation

  def __call__(self, t, other=None):
    return self.eval(t, other)

  def eval(self, t, other=None):
    """the kernel at the cosine argument t (the angle on S1), or at a rotation or direction, or
    between two of them; MATLAB's eval

      v = psi.eval(t)               # at the cosines t, cos(omega/2) on SO(3), cos(theta) on S2
      v = psi(t)                    # the same
      v = psi.eval(rot)             # at the angles of rotations, or the polar angles of directions
      v = psi.eval(rot1, rot2)      # at the angles between two rotations or two directions

    A kernel on SO(3) is a function of the cosine of half the rotation angle, one on the sphere
    of the cosine of the polar angle, one on the circle of the angle itself. Rotations or
    directions are reduced to that argument first: |a| of a rotation, the dot product of two;
    the z coordinate of a direction, the cosine between two.

    Input
      psi - SO3Kernel, S2Kernel or S1Kernel
      t - ndarray of arguments, or rotation (SO3Kernel), or vector3d (S2Kernel)
    Output
      v - ndarray, the values
    Keywords
      other - a second rotation or direction: the kernel at the angle between t and it
    See also
      SO3Kernel.grad, SO3Kernel.calcFourier"""
    return self._eval(self._cosine(t, other))

  def _cosine(self, t, other):
    return _asFloat(t)

  # ---- halfwidth: the first angle at which the kernel falls to half its value at zero,
  # bracketed on a 0.05 degree grid and bisected

  def halfwidth(self):
    """the halfwidth of the kernel: the angle at which it falls to half its value at zero"""
    psi0 = float(self._eval(self._argument(np.asarray(0.0))))
    if psi0 == 0:
      return 0.0
    grid = np.linspace(0, np.pi, 3601)
    below = np.flatnonzero(self._eval(self._argument(grid)) <= psi0 / 2)
    if below.size == 0:
      return math.pi
    lo, hi = grid[below[0] - 1], grid[below[0]]
    for _ in range(50):
      mid = (lo + hi) / 2
      if self._eval(self._argument(np.asarray(mid))) > psi0 / 2:
        lo = mid
      else:
        hi = mid
    return (lo + hi) / 2

  # ---- the algebra: scalar multiples and sums are kernels given by their coefficients

  def __mul__(self, a):
    return self._domain(a * self._coefficients)

  __rmul__ = __mul__

  def __truediv__(self, a):
    return self * (1 / a)

  def __neg__(self):
    return self * -1

  def __add__(self, other):
    if not isinstance(other, self._domain):
      return NotImplemented
    L = max(self.bandwidth, other.bandwidth)
    return self._domain(self._rule(L) + other._rule(L))

  def __sub__(self, other):
    return self + (-other)

  # MATLAB eq: the same coefficients up to the smaller bandwidth
  def __eq__(self, other):
    if not isinstance(other, self._domain):
      return NotImplemented
    L = min(self.bandwidth, other.bandwidth)
    A, B = self._rule(L), other._rule(L)
    return bool(np.linalg.norm(A - B) <= 1e-6 * np.linalg.norm(A))

  __hash__ = None

  def mean(self):
    """the mean of the kernel over its domain, its coefficient of degree zero"""
    return float(self._rule(0)[0])

  # ---- display

  def char(self):
    """the kernel as a short text: its name and halfwidth"""
    return f"{self._label}, halfwidth {numString(self.halfwidth() / degree)}{getMTEXpref('degreeChar', '°')}"

  def __str__(self):
    return self.char()

  def __repr__(self):
    return aligned(f"{type(self).__name__}\n  bandwidth: {self.bandwidth}\n  halfwidth: {numString(self.halfwidth() / degree)}{getMTEXpref('degreeChar', '°')}")


# the parameter of a one parameter family with a given halfwidth, by bisection on the
# halfwidth, which is monotone in the parameter within the bracket
def _parameterForHalfwidth(K, hw, lo, hi, iterations=60):
  def f(p):
    return K(p).halfwidth() - hw
  flo = f(lo)
  for _ in range(iterations):
    mid = (lo + hi) / 2
    if (f(mid) > 0) == (flo > 0):
      lo, flo = mid, f(mid)
    else:
      hi = mid
  return (lo + hi) / 2


# ------------------------------------------------------------------ SO(3)


class SO3Kernel(_Kernel):
  """a kernel on SO(3) given by its Chebyshev coefficients A"""

  def __init__(self, A):
    self._A = np.asarray(A, dtype=float).reshape(-1)
    self.bandwidth = len(self._A) - 1

  def _rule(self, L):
    A = np.zeros(L + 1)
    n = min(L + 1, len(self._A))
    A[:n] = self._A[:n]
    return A

  # sum_n A_n U_{2n}(t) through U_{2n+2} = (4t^2 - 2) U_{2n} - U_{2n-2}
  @staticmethod
  def _clenshaw(A, t):
    t = _asFloat(t)
    A = A.astype(t.dtype, copy=False)
    y = 4 * t * t - 2
    b1 = b2 = np.zeros(t.shape, dtype=t.dtype)
    for n in range(len(A) - 1, 0, -1):
      b1, b2 = A[n] + y * b1 - b2, b1
    return A[0] + (y + 1) * b1 - b2

  @staticmethod
  def _argument(omega):
    return np.cos(omega / 2)

  # the cosine of half the angle: |a| of a rotation, the dot product of two
  def _cosine(self, t, other):
    if isinstance(t, quaternion):
      return np.abs(t.a) if other is None else t.dot(other)
    return _asFloat(t)

  def calcFourier(self, L=None):
    """the Chebyshev coefficients A_0 .. A_L of the kernel, to its bandwidth by default; MATLAB's
    calcFourier

      A = psi.calcFourier()         # up to the bandwidth of the kernel, psi.A
      A = psi.calcFourier(64)       # up to the degree 64

    The kernel is the series psi(cos(omega/2)) = sum_n A_n U_2n(cos(omega/2)) of Chebyshev
    polynomials of the second kind; the coefficients come in closed form or by a recurrence for
    the kernel families and by quadrature for a kernel given as a function. MATLAB's
    'GaussKronrod' and maxAngle are not taken.

    Input
      psi - SO3Kernel
    Output
      A - ndarray of L + 1 coefficients
    Keywords
      L - the largest degree, the bandwidth of the kernel when not given
    See also
      SO3Kernel.eval, S2Kernel.calcFourier, S1Kernel.calcFourier"""
    return self._rule(self.bandwidth if L is None else int(L))

  @cached_property
  def A(self):
    """the Chebyshev coefficients of the kernel up to its bandwidth"""
    return self.calcFourier()

  def norm(self):
    """the L2 norm of the kernel, the Euclidean norm of its coefficients

      t = norm(psi)"""
    return float(np.linalg.norm(self.A))

  def radon(self):
    """the Radon transform is the S2 kernel with the same coefficients"""
    return S2Kernel(self.A)

  def conv(self, other):
    """convolution multiplies the coefficients and divides by 2n+1, on S2 when either factor is"""
    if not isinstance(other, SO3Kernel):
      return other.conv(self)
    L = min(self.bandwidth, other.bandwidth)
    return SO3Kernel(self._rule(L) * other._rule(L) / (2 * np.arange(L + 1) + 1))

  def grad(self, t=None, other=None):
    """the derivative with respect to the rotation angle omega at the cosine argument
    t = cos(omega/2); MATLAB's grad

      d = psi.grad(t)               # at the cosines of the half angles t
      d = psi.grad(rot)             # at the angles of rotations
      d = psi.grad(rot1, rot2)      # at the angles between two rotations
      dpsi = psi.grad()             # the derivative as a kernel, an SO3KernelHandle

    The derivative follows from sin(omega/2) psi = sum_n A_n sin((2n+1) omega/2); it vanishes at
    omega = 0. MATLAB calls it the magnitude of the gradient: the gradient of the function
    g -> psi(g) has this length.

    Input
      psi - SO3Kernel
      t - ndarray of cosines cos(omega/2), or rotation
    Output
      d - ndarray, or SO3KernelHandle without an argument
    Keywords
      other - a second rotation: the derivative at the angle between t and it
    See also
      SO3Kernel.eval, SO3FunRBF.grad"""
    if t is None:
      return SO3KernelHandle(lambda t: self.grad(t), bandwidth=self.bandwidth)
    return self._grad(self._cosine(t, other))

  def _grad(self, t):
    s = np.sqrt(np.maximum(1 - t * t, 0))
    omega = 2 * np.arccos(np.clip(t, -1, 1))
    A = self._coefficients
    num = np.zeros(t.shape, dtype=t.dtype)
    for n in range(len(A)):
      num = num + A[n] * (2 * n + 1) / 2 * np.cos((2 * n + 1) * omega / 2)
    num = num - t * self._clenshaw(A, t) / 2
    return np.where(s < 1e-12, 0.0, num / np.where(s < 1e-12, 1.0, s))


SO3Kernel._domain = SO3Kernel


def _dlvpConstant(kappa):
  return _beta(1.5, 0.5) / _beta(1.5, kappa + 0.5)


# the de la Vallee Poussin recurrence, A_n = (2n+1) a_n, shared by the SO(3) and S2 members
def _dlvpRule(kappa, L):
  a = np.ones(L + 1)
  if L >= 1:
    a[1] = kappa / (kappa + 2)
  for l in range(1, L):
    a[l + 1] = ((kappa - l + 1) * a[l - 1] - (2 * l + 1) * a[l]) / (kappa + l + 2)
  return a * (2 * np.arange(L + 1) + 1)


def _dlvpKappa(halfwidth):
  return 0.5 * math.log(0.5) / math.log(math.cos(halfwidth / 2))


class SO3DeLaValleePoussinKernel(SO3Kernel):
  """the de la Vallee Poussin kernel on SO(3), B(3/2, 1/2) / B(3/2, kappa + 1/2) t^(2 kappa):
  nonnegative and a finite series, the kernel recommended for texture analysis

    SO3DeLaValleePoussinKernel(20)
    SO3DeLaValleePoussinKernel(halfwidth=10*degree)"""
  _label = "de la Vallee Poussin"

  def __init__(self, kappa=90, *, halfwidth=None):
    self.kappa = float(_dlvpKappa(halfwidth) if halfwidth is not None else kappa)

  def _rule(self, L):
    return _dlvpRule(self.kappa, L)

  @cached_property
  def bandwidth(self):
    """the degree the series is cut at, MATLAB's rule over at most kappa coefficients"""
    return self._cutoffBandwidth(cap=max(1, math.ceil(self.kappa)))

  def _eval(self, t):
    return _dlvpConstant(self.kappa) * np.abs(t) ** (2 * self.kappa)

  def _grad(self, t):
    return -_dlvpConstant(self.kappa) * self.kappa * np.sqrt(np.maximum(1 - t * t, 0)) * np.sign(t) * np.abs(t) ** (2 * self.kappa - 1)

  def halfwidth(self):
    """the halfwidth in closed form, 2 acos(2^(-1 / (2 kappa)))"""
    return 2 * math.acos(0.5 ** (1 / (2 * self.kappa)))

  def radon(self):
    """the Radon transform: the de la Vallee Poussin kernel on the sphere of the same kappa"""
    return S2DeLaValleePoussinKernel(self.kappa)


class SO3AbelPoissonKernel(SO3Kernel):
  """the Abel Poisson kernel on SO(3), sum_n (2n+1) kappa^(2n) U_2n(t), kappa in (0, 1)

    SO3AbelPoissonKernel(0.4)
    SO3AbelPoissonKernel(halfwidth=5*degree)"""
  _label = "Abel Poisson"

  def __init__(self, kappa=0.9, *, halfwidth=None):
    self.kappa = float(_parameterForHalfwidth(SO3AbelPoissonKernel, halfwidth, 1e-3, 1 - 1e-6) if halfwidth is not None else kappa)

  def _rule(self, L):
    n = np.arange(L + 1)
    return (2 * n + 1) * self.kappa ** (2 * n)

  def _eval(self, t):
    k = self.kappa
    return ((1 - k * k) / (1 - 2 * k * t + k * k) ** 2 + (1 - k * k) / (1 + 2 * k * t + k * k) ** 2) / 2

  def _radonEval(self, t):
    k = self.kappa
    return (1 - k ** 4) / (1 - 2 * k * k * t + k ** 4) ** 1.5

  def radon(self):
    """the Radon transform: the kernel on the sphere of the same coefficients, in closed form"""
    return S2RadonKernel(self)


# e^{kappa cos omega} / (I_0(kappa) - I_1(kappa)) with the exponential scaling of the Bessel functions
def _vmfConstant(kappa):
  return 1 / (ive(0, kappa) - ive(1, kappa))


class SO3vonMisesFisherKernel(SO3Kernel):
  """the von Mises Fisher kernel on SO(3), proportional to e^(kappa cos omega), kappa > 0

    SO3vonMisesFisherKernel(90)
    SO3vonMisesFisherKernel(halfwidth=10*degree)"""
  _label = "von Mises Fisher"

  def __init__(self, kappa=90, *, halfwidth=None):
    self.kappa = float(math.log(2) / (1 - math.cos(halfwidth)) if halfwidth is not None else kappa)

  def _rule(self, L):
    b = ive(np.arange(L + 2), self.kappa)
    A = (b[:-1] - b[1:]) / (b[0] - b[1])
    A[0] = 1
    return A

  def _eval(self, t):
    return _vmfConstant(self.kappa) * np.exp(self.kappa * (2 * t * t - 2))

  def _grad(self, t):
    return -self.kappa * np.sin(2 * np.arccos(np.clip(t, -1, 1))) * self._eval(t)

  def halfwidth(self):
    """the halfwidth in closed form, acos(1 - log(2) / kappa)"""
    return math.acos(1 - math.log(2) / self.kappa)

  def _radonEval(self, t):
    k = self.kappa
    return _vmfConstant(k) * ive(0, k * (1 + t) / 2) * np.exp(k * (t - 1))

  def radon(self):
    """the Radon transform: the kernel on the sphere of the same coefficients, in closed form"""
    return S2RadonKernel(self)


class SO3GaussWeierstrassKernel(SO3Kernel):
  """the Gauss Weierstrass kernel on SO(3), sum_n (2n+1) e^(-n(n+1) kappa) U_2n(t), kappa > 0

    SO3GaussWeierstrassKernel(0.1)
    SO3GaussWeierstrassKernel(halfwidth=10*degree)"""
  _label = "Gauss Weierstrass"

  def __init__(self, kappa=0.1, *, halfwidth=None):
    self.kappa = float(_parameterForHalfwidth(SO3GaussWeierstrassKernel, halfwidth, 1e-6, 10.0) if halfwidth is not None else kappa)

  def _rule(self, L):
    n = np.arange(L + 1)
    return (2 * n + 1) * np.exp(-n * (n + 1) * self.kappa)


# the Legendre functions of the second kind Q_0(z) .. Q_L(z), z > 1, the minimal solution
# of the Legendre recurrence, so computed backwards
def _legendreQ(L, z):
  # the ratio of the two solutions settles after (N - L) acosh(z) exceeds 18
  N = L + 40 + min(math.ceil(18 / math.acosh(z)), 200_000)
  q = np.zeros(N + 2)
  q[N] = 1.0
  for n in range(N, 0, -1):
    q[n - 1] = ((2 * n + 1) * z * q[n] - (n + 1) * q[n + 1]) / n
    # only the ratios matter: the tail is scaled down before it overflows
    if abs(q[n - 1]) > 1e150:
      q[n - 1:] *= 1e-150
  return q[:L + 1] * (math.log((z + 1) / (z - 1)) / 2 / q[0])


class SO3SquareSingularityKernel(SO3Kernel):
  """the square singularity kernel on SO(3), kappa in (0, 1), its coefficients the Legendre
  functions of the second kind

    SO3SquareSingularityKernel(0.9)
    SO3SquareSingularityKernel(halfwidth=10*degree)"""
  _label = "square singularity"

  def __init__(self, kappa=0.9, *, halfwidth=None):
    self.kappa = float(_parameterForHalfwidth(SO3SquareSingularityKernel, halfwidth, 1e-3, 0.999) if halfwidth is not None else kappa)

  # A_n = (2n+1) Q_n(z) / Q_0(z) with z = (1 + kappa^2) / (2 kappa)
  def _rule(self, L):
    z = (1 + self.kappa ** 2) / (2 * self.kappa)
    Q = _legendreQ(L, z)
    return (2 * np.arange(L + 1) + 1) * Q / Q[0]

  def _radonEval(self, t):
    k = self.kappa
    return 2 * k / math.log((1 + k) / (1 - k)) / (1 - 2 * k * t + k * k)

  def radon(self):
    """the Radon transform: the kernel on the sphere of the same coefficients, in closed form"""
    return S2RadonKernel(self)


class SO3BumpKernel(SO3Kernel):
  """the bump kernel on SO(3): constant on the rotations of angle below delta, zero beyond, of
  mean one

    SO3BumpKernel(40*degree)
    SO3BumpKernel(halfwidth=10*degree, bandwidth=64)"""
  _label = "bump"

  def __init__(self, delta=10 * degree, *, halfwidth=None, bandwidth=None):
    self.delta = float(delta if halfwidth is None else halfwidth)
    self._L = bandwidth

  @cached_property
  def bandwidth(self):
    """the discontinuity needs many degrees: MATLAB keeps maxS1Bandwidth of them unless told"""
    return int(getMTEXpref("maxS1Bandwidth") if self._L is None else self._L)

  # the ball of radius delta, A_n = (2/pi) c int_0^delta sin((2n+1) omega/2) sin(omega/2) d omega
  def _rule(self, L):
    d = self.delta
    n = np.arange(1, L + 1)
    A = np.ones(L + 1)
    A[1:] = (np.sin(n * d) / n - np.sin((n + 1) * d) / (n + 1)) / (d - math.sin(d))
    return A

  def _eval(self, t):
    return np.where(t > math.cos(self.delta / 2), math.pi / (self.delta - math.sin(self.delta)), 0.0).astype(t.dtype)

  def grad(self, t=None, other=None):
    """the derivative of the bump kernel, zero wherever it exists; without an argument the zero kernel

      d = psi.grad(t)               # zeros of the shape of t
      dpsi = psi.grad()             # the zero kernel

    Input
      psi - SO3BumpKernel
      t - ndarray of cosines cos(omega/2), or rotation
    Output
      d - ndarray of zeros, or SO3Kernel without an argument
    Keywords
      other - a second rotation: the angle between t and it
    See also
      SO3Kernel.grad"""
    return SO3Kernel(0) if t is None else np.zeros_like(self._cosine(t, other))

  def halfwidth(self):
    """the halfwidth, the radius delta of the bump"""
    return self.delta

  def _radonEval(self, t):
    with np.errstate(divide="ignore", invalid="ignore"):
      s = math.cos(self.delta / 2) / np.sqrt((1 + t) / 2)
    return np.where(s <= 1, 2 / (self.delta - math.sin(self.delta)) * np.arccos(np.minimum(s, 1)), 0.0)

  def radon(self):
    """the Radon transform: the kernel on the sphere of the same coefficients, in closed form"""
    return S2RadonKernel(self)


class SO3DirichletKernel(SO3Kernel):
  """the Dirichlet kernel on SO(3), sum_n (2n+1) U_2n(t) up to the degree N, of integral one:
  the kernel for physical properties

    SO3DirichletKernel(35)
    SO3DirichletKernel()        N the preference maxSO3Bandwidth"""
  _label = "Dirichlet"

  def __init__(self, N=None):
    self.N = int(getMTEXpref("maxSO3Bandwidth") if N is None else N)

  def _rule(self, L):
    return np.where(np.arange(L + 1) <= self.N, 2.0 * np.arange(L + 1) + 1, 0.0)

  @property
  def bandwidth(self):
    """the degree N of the kernel"""
    return self.N

  def _eval(self, t):
    N = self.N
    omega = 2 * np.arccos(np.clip(t, -1, 1))
    small = omega < 1e-6
    s = np.sin(np.where(small, 1.0, omega) / 2)
    value = ((3 + 2 * N) * np.sin((1 + 2 * N) * omega / 2) - (1 + 2 * N) * np.sin((3 + 2 * N) * omega / 2)) / (4 * s ** 3)
    return np.where(small, (1 + N) * (1 + 2 * N) * (3 + 2 * N) / 3, value)

  def radon(self):
    """the Radon transform: the Dirichlet kernel on the sphere of the same degree"""
    return S2DirichletKernel(self.N)


class SO3RestrictedDistanceKernel(SO3Kernel):
  """the restricted distance kernel on SO(3), -sqrt(8) sin(acos(t)), cut at the degree N: negative,
  a distance between orientations that makes points repel each other"""
  _label = "restricted distance"

  def __init__(self, N=None):
    self.N = int(getMTEXpref("maxSO3Bandwidth") if N is None else N)

  def _rule(self, L):
    n = np.arange(L + 1)
    return np.where(n <= self.N, 16 * math.sqrt(2) / math.pi / ((2 * n - 1) * (2 * n + 1) * (2 * n + 3)), 0.0)

  @property
  def bandwidth(self):
    """the degree N the kernel is cut at"""
    return self.N

  def _eval(self, t):
    return -math.sqrt(8) * np.sqrt(np.maximum(1 - t * t, 0))


class SO3LaplaceKernel(SO3Kernel):
  """the Laplace kernel on SO(3), sum_n (2n+1) / (4 n^2 (2n+2)^2) U_2n(t), without a constant term"""
  _label = "Laplace"

  def __init__(self):
    pass

  def _rule(self, L):
    n = np.arange(L + 1)
    with np.errstate(divide="ignore"):
      A = (2 * n + 1) / (4 * n ** 2 * (2 * n + 2) ** 2)
    A[0] = 0
    return A


class SO3SobolevKernel(SO3Kernel):
  """the Sobolev kernel on SO(3), sum_n (2n+1) (n(n+1))^s U_2n(t) up to the bandwidth

    SO3SobolevKernel(1.1, bandwidth=20)"""
  _label = "Sobolev"

  def __init__(self, s=1, *, bandwidth=None):
    self.s = float(s)
    self.N = int(getMTEXpref("maxSO3Bandwidth") if bandwidth is None else bandwidth)

  def _rule(self, L):
    n = np.arange(L + 1)
    return np.where(n <= self.N, (2 * n + 1) * n ** self.s * (n + 1) ** self.s, 0.0)

  @property
  def bandwidth(self):
    """the degree N the kernel is cut at"""
    return self.N


class fibreVonMisesFisherKernel(SO3Kernel):
  """the von Mises Fisher kernel of a fibre, what fibreODF convolves with; kappa the
  concentration, larger is sharper

    fibreVonMisesFisherKernel(90)
    fibreVonMisesFisherKernel(halfwidth=10*degree)"""
  _label = "fibre von Mises Fisher"

  def __init__(self, kappa=90, *, halfwidth=None):
    self.kappa = float(math.log(2) / (1 - math.cos(halfwidth)) if halfwidth is not None else kappa)

  # (2n+1) I_{n+1/2}(kappa) / sinh(kappa) with the exponential scaling cancelled
  def _rule(self, L):
    k = self.kappa
    n = np.arange(L + 1)
    A = (2 * n + 1) * math.sqrt(math.pi * k / 2) * ive(n + 0.5, k) * 2 / (1 - math.exp(-2 * k))
    A[0] = 1
    return A

  def halfwidth(self):
    """the halfwidth in closed form, acos(1 - log(2) / kappa)"""
    return math.acos(1 - math.log(2) / self.kappa)

  def _radonEval(self, t):
    k = self.kappa
    return 2 * k * np.exp(k * (t - 1)) / (1 - math.exp(-2 * k))

  def radon(self):
    """the Radon transform: the kernel on the sphere of the same coefficients, in closed form"""
    return S2RadonKernel(self)


class SO3KernelHandle(SO3Kernel):
  """a kernel given by a closed form, its coefficients by Gauss-Legendre quadrature"""

  def __init__(self, fun, *, bandwidth=None):
    self.fun = fun
    self.N = int(getMTEXpref("maxS1Bandwidth") if bandwidth is None else bandwidth)

  def _rule(self, L):
    x, w = gaussLegendre(max(4 * L + 8, 128))
    omega = (x + 1) * (math.pi / 2)
    f = self.fun(np.cos(omega / 2)) * np.sin(omega / 2) * w
    return np.array([np.sum(f * np.sin((2 * n + 1) * omega / 2)) for n in range(L + 1)])

  @property
  def bandwidth(self):
    """the degree the coefficients are computed to, the preference maxS1Bandwidth unless given"""
    return self.N

  def _eval(self, t):
    return self.fun(np.clip(t, -1, 1))


# ------------------------------------------------------------------ S2


class S2Kernel(_Kernel):
  """a kernel on the sphere given by its Legendre coefficients A"""

  def __init__(self, A):
    self._A = np.asarray(A, dtype=float).reshape(-1)
    self.bandwidth = len(self._A) - 1

  def _rule(self, L):
    A = np.zeros(L + 1)
    n = min(L + 1, len(self._A))
    A[:n] = self._A[:n]
    return A

  # sum_n A_n P_n(t)
  @staticmethod
  def _clenshaw(A, t):
    t = _asFloat(t)
    A = A.astype(t.dtype, copy=False)
    b1 = b2 = np.zeros(t.shape, dtype=t.dtype)
    for n in range(len(A) - 1, 0, -1):
      b1, b2 = A[n] + (2 * n + 1) / (n + 1) * t * b1 - (n + 1) / (n + 2) * b2, b1
    return A[0] + t * b1 - b2 / 2

  @staticmethod
  def _argument(theta):
    return np.cos(theta)

  # the cosine of the polar angle of a direction, the cosine between two
  def _cosine(self, t, other):
    if isinstance(t, vector3d):
      return t.z / t.norm() if other is None else t.dot(other) / (t.norm() * other.norm())
    return _asFloat(t)

  def calcFourier(self, L=None):
    """the Legendre coefficients A_0 .. A_L of the kernel, to its bandwidth by default; MATLAB's
    calcFourier

      A = psi.calcFourier()         # up to the bandwidth of the kernel, psi.A
      A = psi.calcFourier(64)       # up to the degree 64

    The kernel is the series psi(t) = sum_n A_n P_n(t) of Legendre polynomials in the cosine t
    of the polar angle.

    Input
      psi - S2Kernel
    Output
      A - ndarray of L + 1 coefficients
    Keywords
      L - the largest degree, the bandwidth of the kernel when not given
    See also
      S2Kernel.eval, SO3Kernel.calcFourier"""
    return self._rule(self.bandwidth if L is None else int(L))

  @cached_property
  def A(self):
    """the Legendre coefficients of the kernel up to its bandwidth"""
    return self.calcFourier()

  def norm(self):
    """the L2 norm of the kernel, sqrt(sum_n A_n^2 / (2n+1))

      t = norm(psi)"""
    A = self.A
    return float(np.sqrt(np.sum(A * A / (2 * np.arange(len(A)) + 1))))

  def radon(self):
    """the Funk transform: A_n P_n(0)"""
    return S2Kernel(self.A * legendreP(self.bandwidth, 0.0))

  def conv(self, other):
    """the convolution with another kernel on the sphere or SO(3): the coefficients multiplied and
    divided by 2n+1, a kernel on the sphere"""
    if not isinstance(other, (S2Kernel, SO3Kernel)):
      return other.conv(self)
    L = min(self.bandwidth, other.bandwidth)
    return S2Kernel(self._rule(L) * other._rule(L) / (2 * np.arange(L + 1) + 1))

  def grad(self, t=None, other=None, *, polynomial=False):
    """the derivative of a kernel on the sphere with respect to the polar angle, or to its cosine

      d = psi.grad(t)                      # dpsi/dtheta at the cosines t
      d = psi.grad(v, zvector)             # at the angles between the directions v and the z axis
      d = psi.grad(t, polynomial=True)     # dpsi/dt, the derivative in the cosine
      dpsi = psi.grad()                    # the derivative as a kernel

    The kernel is a Legendre series psi(t) in the cosine t of the angle; its derivative in t is
    a Legendre series of its own, and the derivative in the angle theta is -sin(theta) dpsi/dt.
    Without an argument the result is a kernel: a handle of the angular derivative, or the
    Legendre series of dpsi/dt for polynomial=True.

    Input
      psi - S2Kernel
      t - ndarray of cosines, or vector3d
      other - vector3d, the direction the angles of t are measured from
    Output
      d - ndarray, or S2Kernel / S2KernelHandle without an argument
    Keywords
      polynomial - the derivative with respect to the cosine t instead of the angle
    See also
      S2Kernel, S2FunHarmonic.grad, S2Kernels"""
    if t is None:
      return self._derivativeSeries() if polynomial else S2KernelHandle(lambda t: self.grad(t), bandwidth=self.bandwidth)
    t = self._cosine(t, other)
    return self._gradT(t) if polynomial else self._grad(t)

  def _grad(self, t):
    return -np.sqrt(np.maximum(1 - t * t, 0)) * self._derivativeSeries()(t)

  # dpsi/dt from the angular derivative where a closed form exists, the series at the poles
  def _gradT(self, t):
    inner = np.abs(t) < 1 - 1e-8
    safe = np.where(inner, t, 0.0)
    return np.where(inner, -self._grad(safe) / np.sqrt(1 - safe * safe), self._derivativeSeries()(t))

  def _derivativeSeries(self):
    A = self._coefficients
    L = len(A) - 1
    B = np.array([(2 * k + 1) * np.sum(A[k + 1::2]) for k in range(L)])
    return S2Kernel(B if L else [0.0])


S2Kernel._domain = S2Kernel


class S2DeLaValleePoussinKernel(S2Kernel):
  """the de la Vallee Poussin kernel on the sphere, (1 + kappa) ((1 + t) / 2)^kappa: nonnegative and
  a finite series, the kernel recommended for texture analysis

    S2DeLaValleePoussinKernel(20)
    S2DeLaValleePoussinKernel(halfwidth=10*degree)"""
  _label = "de la Vallee Poussin"

  def __init__(self, kappa=90, *, halfwidth=None):
    self.kappa = float(_dlvpKappa(halfwidth) if halfwidth is not None else kappa)

  def _rule(self, L):
    return _dlvpRule(self.kappa, L)

  @cached_property
  def bandwidth(self):
    """the degree the series is cut at, MATLAB's rule over at most kappa coefficients"""
    return self._cutoffBandwidth(cap=max(1, math.ceil(self.kappa)))

  def _eval(self, t):
    return (1 + self.kappa) * ((1 + t) / 2) ** self.kappa

  def _grad(self, t):
    return -np.sqrt(np.maximum(1 - t * t, 0)) * self.kappa * (1 + self.kappa) / 2 * ((1 + t) / 2) ** (self.kappa - 1)

  def halfwidth(self):
    """the halfwidth in closed form, acos(2 * 2^(-1 / kappa) - 1)"""
    return math.acos(2 * 0.5 ** (1 / self.kappa) - 1)


class S2DirichletKernel(S2Kernel):
  """the Dirichlet kernel on the sphere, sum_n (2n+1) P_n(t) up to the degree N, of integral one

    S2DirichletKernel(35)
    S2DirichletKernel()        N the preference maxS2Bandwidth"""
  _label = "Dirichlet"

  def __init__(self, N=None):
    self.N = int(getMTEXpref("maxS2Bandwidth") if N is None else N)

  def _rule(self, L):
    return np.where(np.arange(L + 1) <= self.N, 2.0 * np.arange(L + 1) + 1, 0.0)

  @property
  def bandwidth(self):
    """the degree N of the kernel"""
    return self.N


class S2BumpKernel(S2Kernel):
  """the bump kernel on the sphere: 2 / (1 - cos delta) on the cap of radius delta, zero beyond,
  a density of mean one

    S2BumpKernel(10*degree)
    S2BumpKernel(halfwidth=10*degree, bandwidth=64)"""
  _label = "bump"

  def __init__(self, delta=10 * degree, *, halfwidth=None, bandwidth=None):
    self.delta = float(delta if halfwidth is None else halfwidth)
    self._L = None if bandwidth is None else int(bandwidth)

  @cached_property
  def bandwidth(self):
    """the degree the expansion is cut at, given or by MATLAB's rule"""
    return self._cutoffBandwidth() if self._L is None else self._L

  # the cap of radius delta scaled to a density, A_n = (P_{n-1}(c) - P_{n+1}(c)) / (1 - c)
  def _rule(self, L):
    c = math.cos(self.delta)
    P = legendreP(L + 1, c)
    A = np.ones(L + 1)
    A[1:] = (P[:L] - P[2:]) / (1 - c)
    return A

  def _eval(self, t):
    return np.where(t > math.cos(self.delta), 2 / (1 - math.cos(self.delta)), 0.0).astype(t.dtype)

  def grad(self, t=None, other=None, *, polynomial=False):
    """the derivative of the bump kernel, zero wherever it exists

      d = psi.grad(t)              # zero at every cosine t
      d = psi.grad(v, zvector)     # zero at every direction v
      dpsi = psi.grad()            # the zero kernel

    The bump kernel is constant on its cap and beyond it, so its derivative vanishes but on the
    rim of the cap, where it does not exist; the derivative in the angle and in its cosine
    agree.

    Input
      psi - S2BumpKernel
      t - ndarray of cosines, or vector3d
      other - vector3d, the direction the angles of t are measured from
    Output
      d - ndarray of zeros, or the zero S2Kernel without an argument
    Keywords
      polynomial - the derivative with respect to the cosine, the same zero
    See also
      S2Kernel.grad, S2BumpKernel"""
    return S2Kernel(0) if t is None else np.zeros_like(self._cosine(t, other))

  def halfwidth(self):
    """the halfwidth, the radius delta of the cap"""
    return self.delta


class S2RestrictedDistanceKernel(S2Kernel):
  """the restricted distance kernel on the sphere, -2 sin(acos(t) / 2), cut at the degree N:
  negative, a distance between directions that makes points repel each other"""
  _label = "restricted distance"

  def __init__(self, N=None):
    self.N = int(getMTEXpref("maxS2Bandwidth") if N is None else N)

  def _rule(self, L):
    n = np.arange(L + 1)
    return np.where(n <= self.N, 4 / ((2 * n - 1) * (2 * n + 3)), 0.0)

  @property
  def bandwidth(self):
    """the degree N the kernel is cut at"""
    return self.N

  def _eval(self, t):
    return -2 * np.sqrt(np.maximum((1 - t) / 2, 0))


class S2KernelHandle(S2Kernel):
  """a kernel on the sphere given by a function of the cosine of the polar angle, its Legendre
  coefficients by Gauss-Legendre quadrature

    S2KernelHandle(lambda t: t**2, bandwidth=16)"""

  def __init__(self, fun, *, bandwidth=None):
    self.fun = fun
    self.N = int(getMTEXpref("maxS2Bandwidth") if bandwidth is None else bandwidth)

  def _rule(self, L):
    x, w = gaussLegendre(max(2 * L + 8, 128))
    P = legendreP(L, x)
    return (2 * np.arange(L + 1) + 1) / 2 * (P @ (self.fun(x) * w))

  @property
  def bandwidth(self):
    """the degree the coefficients are computed to, the preference maxS2Bandwidth unless given"""
    return self.N

  def _eval(self, t):
    return self.fun(np.clip(t, -1, 1))


class S2RadonKernel(S2Kernel):
  """the Radon transform of an SO(3) kernel: the S2 kernel with the same coefficients, evaluated
  by the closed form of the family where MTEX gives one"""

  def __init__(self, psi):
    self.psi = psi
    self._label = "Radon transform of " + psi._label

  def _rule(self, L):
    return self.psi._rule(L)

  @property
  def bandwidth(self):
    """the bandwidth of the SO(3) kernel transformed"""
    return self.psi.bandwidth

  def _eval(self, t):
    radonEval = getattr(self.psi, "_radonEval", None)
    return self._clenshaw(self._coefficients, t) if radonEval is None else radonEval(t)


# ------------------------------------------------------------------ S1


class S1Kernel(_Kernel):
  """a kernel on the circle given by its Fourier coefficients fhat, degrees -N .. N"""

  def __init__(self, fhat):
    fhat = np.asarray(fhat, dtype=float).reshape(-1)
    self._c = fhat[len(fhat) // 2:]
    self.bandwidth = len(self._c) - 1

  @classmethod
  def _fromHalf(cls, c):
    psi = cls.__new__(cls)
    psi._c = np.asarray(c, dtype=float).reshape(-1)
    psi.bandwidth = len(psi._c) - 1
    return psi

  # the rule gives c_0 .. c_L; fhat is the two sided array MATLAB stores
  def _rule(self, L):
    c = np.zeros(L + 1)
    n = min(L + 1, len(self._c))
    c[:n] = self._c[:n]
    return c

  def calcFourier(self, L=None):
    """the Fourier coefficients of the kernel, degrees -L .. L, to its bandwidth by default;
    MATLAB's calcFourier

      fhat = psi.calcFourier()      # up to the bandwidth of the kernel, psi.fhat
      fhat = psi.calcFourier(16)    # the degrees -16 .. 16

    The kernel is the even series psi(x) = sum_k c_k e^{ikx}, c_-k = c_k; the coefficients are
    returned two sided, beyond the bandwidth as zeros.

    Input
      psi - S1Kernel
    Output
      fhat - ndarray of 2L + 1 coefficients
    Keywords
      L - the largest degree, the bandwidth of the kernel when not given
    See also
      S1Kernel.grad, SO3Kernel.calcFourier"""
    c = self._rule(self.bandwidth if L is None else int(L))
    return np.concatenate([c[:0:-1], c])

  @cached_property
  def fhat(self):
    """the Fourier coefficients of the kernel, degrees -N .. N up to its bandwidth"""
    return self.calcFourier()

  # c_0 + 2 sum_k c_k cos(kx) as a Chebyshev series in cos x
  @staticmethod
  def _clenshaw(c, x):
    x = _asFloat(x)
    c = c.astype(x.dtype, copy=False)
    t = np.cos(x)
    b1 = b2 = np.zeros(x.shape, dtype=x.dtype)
    for k in range(len(c) - 1, 0, -1):
      b1, b2 = 2 * c[k] + 2 * t * b1 - b2, b1
    return c[0] + t * b1 - b2

  @staticmethod
  def _argument(x):
    return x

  def __mul__(self, a):
    return S1Kernel._fromHalf(a * self._coefficients)

  __rmul__ = __mul__

  def __add__(self, other):
    if not isinstance(other, S1Kernel):
      return NotImplemented
    L = max(self.bandwidth, other.bandwidth)
    return S1Kernel._fromHalf(self._rule(L) + other._rule(L))

  def norm(self):
    """the L2 norm of the kernel, the Euclidean norm of its two sided coefficients

      t = norm(psi)"""
    c = self._coefficients
    return float(np.sqrt(c[0] ** 2 + 2 * np.sum(c[1:] ** 2)))

  def conv(self, other):
    """the convolution with another kernel on the circle: the coefficients multiplied"""
    if not isinstance(other, S1Kernel):
      return other.conv(self)
    L = min(self.bandwidth, other.bandwidth)
    return S1Kernel._fromHalf(self._rule(L) * other._rule(L))

  def grad(self, x=None):
    """the derivative of a kernel on the circle, -2 sum_k k c_k sin(kx)

      d = psi.grad(x)               # at the angles x
      dpsi = psi.grad()             # the derivative as a kernel, an S1KernelHandle

    Input
      psi - S1Kernel
    Output
      d - ndarray, or S1KernelHandle without an argument
    Keywords
      x - the angles; without them the derivative is returned as a kernel
    See also
      S1Kernel.calcFourier, S1FunHarmonic.grad"""
    if x is None:
      return S1KernelHandle(lambda x: self.grad(x), bandwidth=self.bandwidth)
    x = _asFloat(x)
    c = self._coefficients
    value = np.zeros(x.shape, dtype=x.dtype)
    for k in range(1, len(c)):
      value = value - 2 * k * c[k] * np.sin(k * x)
    return value


S1Kernel._domain = S1Kernel


# the angle reduced to [-pi, pi]
def _wrap(x):
  return np.mod(x + np.pi, 2 * np.pi) - np.pi


class S1DeLaValleePoussinKernel(S1Kernel):
  """the n-th de la Vallee Poussin kernel on the circle, 2 F_(2n-1) - F_(n-1) with F the Fejer
  kernels: coefficients one up to n, falling to zero at 2n

    S1DeLaValleePoussinKernel(20)
    S1DeLaValleePoussinKernel(halfwidth=5*degree)"""
  _label = "de la Vallee Poussin"

  def __init__(self, n=None, *, halfwidth=None):
    if n is None:
      n = 1.2 / (2.9 * degree if halfwidth is None else halfwidth) + 1 / 16
    self.n = max(1, int(round(n)))

  # 1 on |k| <= n, falling linearly to 0 at 2n
  def _rule(self, L):
    k = np.arange(L + 1)
    return np.clip(2 - k / self.n, 0, 1)

  @property
  def bandwidth(self):
    """the degree 2n beyond which the coefficients vanish"""
    return 2 * self.n

  def _eval(self, x):
    n = self.n
    x = _wrap(x)
    small = np.abs(x) < 1e-8
    s = np.sin(np.where(small, 1.0, x) / 2)
    return np.where(small, 3 * n, (np.sin(n * x) ** 2 - np.sin(n * x / 2) ** 2) / (n * s * s))


class S1DirichletKernel(S1Kernel):
  """the Dirichlet kernel on the circle, sum_(k = -n .. n) e^(ikx), of integral one"""
  _label = "Dirichlet"

  def __init__(self, n=25):
    self.n = int(n)

  def _rule(self, L):
    return np.where(np.arange(L + 1) <= self.n, 1.0, 0.0)

  @property
  def bandwidth(self):
    """the degree n of the kernel"""
    return self.n

  def _eval(self, x):
    x = _wrap(x)
    small = np.abs(x) < 1e-8
    s = np.sin(np.where(small, 1.0, x) / 2)
    return np.where(small, 2 * self.n + 1, np.sin((2 * self.n + 1) * x / 2) / s)


class S1FejerKernel(S1Kernel):
  """the Fejer kernel on the circle, the mean of the Dirichlet kernels D_0 .. D_n: nonnegative,
  of integral one"""
  _label = "Fejer"

  def __init__(self, n=25):
    self.n = int(n)

  def _rule(self, L):
    return np.maximum(0, 1 - np.arange(L + 1) / (self.n + 1))

  @property
  def bandwidth(self):
    """the degree n of the kernel"""
    return self.n

  def _eval(self, x):
    x = _wrap(x)
    small = np.abs(x) < 1e-8
    s = np.sin(np.where(small, 1.0, x) / 2)
    return np.where(small, self.n + 1, (np.sin((self.n + 1) * x / 2) / s) ** 2 / (self.n + 1))


class S1BumpKernel(S1Kernel):
  """the indicator of the interval |x| < delta scaled to a density"""
  _label = "bump"

  def __init__(self, delta=10 * degree, *, halfwidth=None):
    self.delta = float(delta if halfwidth is None else halfwidth)

  def _rule(self, L):
    k = np.arange(1, L + 1)
    c = np.ones(L + 1)
    c[1:] = np.sin(k * self.delta) / (k * self.delta)
    return c

  def _eval(self, x):
    return np.where(np.abs(_wrap(x)) < self.delta, math.pi / self.delta, 0.0).astype(x.dtype)

  def grad(self, x=None):
    """the derivative of the bump kernel, zero wherever it exists; without an argument the zero kernel

      d = psi.grad(x)               # zeros of the shape of x
      dpsi = psi.grad()             # the zero kernel

    Input
      psi - S1BumpKernel
    Output
      d - ndarray of zeros, or S1Kernel without an argument
    Keywords
      x - the angles; without them the zero kernel is returned
    See also
      S1Kernel.grad"""
    if x is None:
      return S1Kernel([0.0])
    return np.zeros_like(_asFloat(x))

  def halfwidth(self):
    """the halfwidth, the half length delta of the interval"""
    return self.delta


class S1GaussKernel(S1Kernel):
  """the heat kernel of the circle, a wrapped Gaussian: what smooths a density of angles"""
  _label = "Gauss"

  def __init__(self, sigma=None, *, halfwidth=None):
    if halfwidth is not None:
      sigma = halfwidth / math.sqrt(2 * math.log(2))
    self.sigma = float(5 * degree if sigma is None else sigma)

  def _rule(self, L):
    k = np.arange(L + 1)
    return np.exp(-k * k * self.sigma ** 2 / 2)

  @property
  def bandwidth(self):
    """the modes beyond 6 / sigma are below e^{-18}"""
    return max(1, math.ceil(6 / self.sigma))

  def halfwidth(self):
    """the halfwidth in closed form, sigma sqrt(2 log 2)"""
    return self.sigma * math.sqrt(2 * math.log(2))


class S1KernelHandle(S1Kernel):
  """a kernel on the circle given by a function of the angle, its Fourier coefficients by the
  trapezoidal rule

    S1KernelHandle(lambda x: np.cos(x)**2, bandwidth=8)"""

  def __init__(self, fun, *, bandwidth=None):
    self.fun = fun
    self.N = int(getMTEXpref("maxS1Bandwidth") if bandwidth is None else bandwidth)

  def _rule(self, L):
    M = max(4 * L + 8, 128)
    x = 2 * np.pi * np.arange(M) / M
    f = self.fun(x)
    return np.array([np.sum(f * np.cos(k * x)) / M for k in range(L + 1)])

  @property
  def bandwidth(self):
    """the degree the coefficients are computed to, the preference maxS1Bandwidth unless given"""
    return self.N

  def _eval(self, x):
    return self.fun(x)


__all__ = ["SO3Kernel", "SO3DeLaValleePoussinKernel", "SO3AbelPoissonKernel", "SO3vonMisesFisherKernel",
           "SO3GaussWeierstrassKernel", "SO3SquareSingularityKernel", "SO3BumpKernel", "SO3DirichletKernel",
           "SO3RestrictedDistanceKernel", "SO3LaplaceKernel", "SO3SobolevKernel", "fibreVonMisesFisherKernel",
           "SO3KernelHandle", "S2Kernel", "S2DeLaValleePoussinKernel", "S2DirichletKernel", "S2BumpKernel",
           "S2RestrictedDistanceKernel", "S2KernelHandle", "S1Kernel", "S1DeLaValleePoussinKernel",
           "S1DirichletKernel", "S1FejerKernel", "S1BumpKernel", "S1GaussKernel", "S1KernelHandle"]
