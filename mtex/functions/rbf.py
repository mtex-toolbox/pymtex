# the radial basis and fibre representations of functions on SO(3) and the named ODFs
# (MATLAB @SO3FunRBF, @SO3FunCBF, uniformODF, unimodalODF, fibreODF; docs/plans/functions.md F4)
#
#   SO3FunRBF(center, psi, weights, c0)   f(R) = c0 + sum_j w_j K(R, c_j), K the kernel averaged
#                                         over the proper rotations the frames reduce over (the
#                                         groups of P1's dot) and the inverses under exchange
#   SO3FunCBF(h, r, weights, psi, SS)     f(R) = sum_i w_i mean_{s,t} psi(<s R t h_i, r_i>), a fibre component
#   uniformODF(CS, SS), unimodalODF(ori, halfwidth=), fibreODF(h, r, SS, halfwidth=), fibreODF(fibre)
#
# Every other operation goes through the harmonic form, SO3FunHarmonic(f, bandwidth=L).
from __future__ import annotations

import math

import numpy as np

from ..geometry.fibre import fibre
from ..geometry.frames import crystalFrame, referenceFrame, specimenFrame
from ..geometry.miller import Miller
from ..geometry.orientation import orientation
from ..geometry.quaternion import orthogonalTransform, quaternion, rotation
from ..geometry.session import getMTEXpref
from ..geometry.symreduce import reductionGroups
from ..geometry.tangent import SO3TangentVector
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .funops import _antipodalPole, _rotatedFrame, conv
from .harmonics import deg2dim, degreeBlock, _sphericalY
from .kernelfunctions import S2DeLaValleePoussinKernel, S2Kernel, SO3DeLaValleePoussinKernel, SO3Kernel, SO3vonMisesFisherKernel
from .quadrature import folds
from .s2fun import S2FunHarmonic
from .so3fun import SO3FunHarmonic
from .transforms import wignerSeriesAdjoint

CHUNK = 1024
# above this many kernel terms an evaluation goes through the harmonic form
HARMONIC_EVAL = 2_000_000


def _qmul(p, q):
  a1, b1, c1, d1 = np.moveaxis(p, -1, 0)
  a2, b2, c2, d2 = np.moveaxis(q, -1, 0)
  return np.stack([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, b1 * a2 + a1 * b2 - d1 * c2 + c1 * d2,
                   c1 * a2 + d1 * b2 + a1 * c2 - b1 * d2, d1 * a2 - c1 * b2 + b1 * c2 + a1 * d2], axis=-1)


def _conj(q):
  return q * np.array([1.0, -1.0, -1.0, -1.0])


# the cosine of the half angle beyond which the kernel stays below tol of its peak, in
# closed form for the families with one and found by a scan otherwise; a kernel that never
# drops that far is summed in full
def supportCosine(psi, tol=1e-9):
  if isinstance(psi, SO3DeLaValleePoussinKernel):
    return tol ** (1 / (2 * psi.kappa))
  if isinstance(psi, SO3vonMisesFisherKernel):
    x = 1 + math.log(tol) / (2 * psi.kappa)
    return math.sqrt(x) if x > 0 else -1.0
  omega = np.linspace(0, np.pi, 361)
  values = np.abs(psi(np.cos(omega / 2)))
  small = values <= tol * psi(1.0)
  # the first angle from which every later value is small
  tail = np.cumsum(~small[::-1])[::-1] == 0
  i = np.flatnonzero(tail)
  return math.cos(omega[i[0]] / 2) if i.size else -1.0


# one frame, with or without its group: a symmetric function is also asked at the
# orientations of the frame stripped of it, as MATLAB's stripSym gives them
def _sameFrame(a, b):
  return a is b or (a is not None and b is not None and a.stripSym() is b.stripSym())


# the rotations of a query in the frames of a function
def _rotationsFor(f, x):
  if isinstance(x, orientation):
    if not (_sameFrame(x.CS, f.CS) and _sameFrame(x.SS, f.SS)):
      raise ValueError("the orientation lives in other frames than the function")
    return x
  if isinstance(x, orthogonalTransform):
    return x.rot
  if isinstance(x, quaternion):
    return x
  raise TypeError("a function on SO(3) is evaluated at rotations or orientations")


class SO3FunRBF:
  isReal = True

  # the sum of kernels through values at orientations, MATLAB's SO3FunRBF.interpolate
  @staticmethod
  def interpolate(nodes, values, **kwargs):
    from .rbfinterpolation import interpolateRBF
    return interpolateRBF(nodes, values, **kwargs)

  def __init__(self, center, psi=None, weights=None, c0=0.0):
    if not isinstance(center, orientation):
      center = orientation(center)
    self.center = center.flatten()
    self.psi = SO3DeLaValleePoussinKernel(halfwidth=10 * degree) if psi is None else psi
    n = self.center.size
    self.weights = np.full(n, 1 / (n * self.psi.mean()) if n else 0.0) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if self.weights.size != n:
      raise ValueError("one weight per centre")
    self.c0 = float(c0)

  @property
  def CS(self):
    return self.center.CS

  @property
  def SS(self):
    return self.center.SS

  @SS.setter
  def SS(self, fr):
    self.center = orientation._fromRotation(self.center.data, (self.CS, fr), self.antipodal)

  @property
  def antipodal(self):
    return self.center.antipodal

  @property
  def bandwidth(self):
    return self.psi.bandwidth

  def mean(self):
    return self.c0 + float(np.sum(self.weights)) * self.psi.mean()

  # the equivalents s c t of every centre with the weight w / m, the inverses included
  # under exchange symmetry, as (M, 4) quaternions
  def _expandedCentres(self):
    L, R = reductionGroups(self.center)
    c = self.center.data.reshape(-1, 4)
    if self.antipodal:
      c = np.concatenate([c, _conj(c)])
    q = _qmul(_qmul(L[:, None, None, :], c[None, :, None, :]), R[None, None, :, :])
    m = len(L) * len(R) * (2 if self.antipodal else 1)
    w = np.repeat(self.weights, 2) if self.antipodal else self.weights
    return q.transpose(1, 0, 2, 3).reshape(-1, 4), np.repeat(w, len(L) * len(R)) / m

  def __call__(self, x):
    return self.eval(x)

  # the kernel sum over the centres and their symmetric copies; a large sum goes through the
  # harmonic form, exact for a kernel of finite bandwidth, as MTEX does
  def eval(self, x):
    r = _rotationsFor(self, x)
    q = r.data.reshape(-1, 4)
    L, R = reductionGroups(self.center)
    if self.center.size * len(L) * len(R) * len(q) > HARMONIC_EVAL and self.psi.bandwidth <= getMTEXpref("maxSO3Bandwidth"):
      if getattr(self, "_harmonicCache", None) is None:
        self._harmonicCache = self._harmonic()
      return np.real(self._harmonicCache.eval(r)).reshape(r.shape)
    cs, w = self._expandedCentres()
    cut = supportCosine(self.psi)
    out = np.full(len(q), self.c0)
    for start in range(0, len(q), CHUNK):
      D = np.abs(q[start:start + CHUNK] @ cs.T)
      mask = D > cut
      values = np.zeros(D.shape)
      values[mask] = self.psi(np.minimum(D[mask], 1.0))
      out[start:start + CHUNK] += values @ w
    return out.reshape(r.shape)

  # the harmonic form: the point measure of the centres convolved with the kernel,
  # projected; the transform runs at the narrower of the two bandwidths, and the mean is
  # known exactly
  def _harmonic(self, bandwidth=None):
    L = min(self.psi.bandwidth, getMTEXpref("maxSO3Bandwidth") if bandwidth is None else int(bandwidth))
    if self.center.size == 0:
      fhat = np.zeros(deg2dim(L + 1), dtype=complex)
    else:
      fhat = wignerSeriesAdjoint(self.center, np.ones(self.center.size), L, weights=self.weights, folds=folds(self.CS, self.SS))
    A = self.psi.calcFourier(L)
    for n in range(L + 1):
      degreeBlock(fhat, n)[...] *= A[n] / (2 * n + 1)
    fhat[0] = self.mean()
    return SO3FunHarmonic(fhat, self.CS, self.SS, real=True, antipodal=self.antipodal, _internal=True)

  # the gradient: minus the kernel's angular derivative along the geodesic to every
  # equivalent within its support; a query with none in reach takes the nearest centre alone
  # the gradient at orientations, or without them the field of gradients of the harmonic
  # expansion, as MATLAB's
  def grad(self, x=None, side="left"):
    if x is None:
      from .so3fun import SO3FunHarmonic
      return SO3FunHarmonic(self).grad(side=side)
    if isinstance(x, str):
      raise TypeError("the gradient of a kernel function is evaluated at orientations")
    if side not in ("left", "right"):
      raise ValueError(f"the side is 'left' or 'right', not {side!r}")
    r = _rotationsFor(self, x)
    q = r.data.reshape(-1, 4)
    cs, w = self._expandedCentres()
    cut = supportCosine(self.psi)
    out = np.zeros((len(q), 3))
    for start in range(0, len(q), CHUNK):
      qb = q[start:start + CHUNK]
      D = np.abs(qb @ cs.T)
      mask = D > cut
      none = ~np.any(mask, axis=1)
      mask[none, np.argmax(D[none], axis=1)] = True
      i, j = np.nonzero(mask)
      qi = _conj(qb[i])
      p = _qmul(cs[j], qi) if side == "left" else _qmul(qi, cs[j])
      n = np.linalg.norm(p[:, 1:], axis=1)
      ok = n > 1e-12
      f0 = np.zeros(len(i))
      f0[ok] = w[j[ok]] * self.psi.grad(np.minimum(D[i[ok], j[ok]], 1.0)) * np.where(p[ok, 0] < 0, -1.0, 1.0) / n[ok]
      np.add.at(out, start + i, -f0[:, None] * p[:, 1:])
    return SO3TangentVector(vector3d(out.reshape(*r.shape, 3)), x, side)

  # the pole figure: the Radon transform of the kernel centred at the images of h under the
  # centres and the crystal rotations, projected by the specimen frame
  def pdf(self, h, bandwidth=None):
    h = h if isinstance(h, vector3d) else vector3d(h)
    if h.size != 1:
      raise ValueError("one crystal direction at a time")
    if h.frame is not None and h.frame is not self.CS:
      raise ValueError("the crystal direction must live in the crystal frame of the function")
    L = self.psi.bandwidth if bandwidth is None else int(bandwidth)
    ant = _antipodalPole(h, self.CS)
    if self.center.size == 0:
      return S2FunHarmonic([math.sqrt(4 * np.pi) * self.c0], self.SS, real=True, antipodal=ant, _internal=True)
    _, R = reductionGroups(self.center)
    hs = rotation(R) * vector3d(h.normalize().data.reshape(3))
    c = self.center.data.reshape(-1, 4)
    if self.antipodal:
      c = np.concatenate([c, _conj(c)])
    pts = rotation(c[:, None, :]) * hs
    w = np.repeat(self.weights, 2) / 2 if self.antipodal else self.weights
    w = np.repeat(w, hs.size) / hs.size
    s = S2FunHarmonic.quadrature(vector3d(pts.data.reshape(-1, 3), frame=self.SS), np.ones(pts.size), bandwidth=L, weights=4 * np.pi * w, real=True, antipodal=ant)
    return conv(s, self.psi.radon()) + self.c0

  def ipdf(self, r, bandwidth=None):
    return self.inv().pdf(r, bandwidth)

  def radon(self, h=None, r=None):
    if r is None:
      return self.pdf(h)
    if h is None:
      return self.ipdf(r)
    return self.pdf(h)(r)

  def calcPDF(self, h, r=None, superposition=None):
    from .density import calcPDF
    return calcPDF(self, h, r, superposition=superposition)

  def inv(self):
    return SO3FunRBF(self.center.inv(), self.psi, self.weights, self.c0)

  def rotate(self, r, side="left"):
    rot = rotation(r.data.reshape(4)) if not isinstance(r, orientation) else rotation(r)
    if side == "left":
      center = orientation._fromRotation((rot * rotation(self.center)).data, (self.CS, _rotatedFrame(self.SS, rot)), self.antipodal)
    elif side == "right":
      center = orientation._fromRotation((rotation(self.center) * rot).data, (_rotatedFrame(self.CS, rot), self.SS), self.antipodal)
    else:
      raise ValueError(f"the side is 'left' or 'right', not {side!r}")
    return SO3FunRBF(center, self.psi, self.weights, self.c0)

  def conv(self, psi):
    if isinstance(psi, SO3Kernel):
      return SO3FunRBF(self.center, self.psi.conv(psi), self.weights, self.c0 * psi.mean())
    return conv(SO3FunHarmonic(self), psi)

  def __mul__(self, a):
    if isinstance(a, (int, float, np.number)):
      return SO3FunRBF(self.center, self.psi, a * self.weights, a * self.c0)
    return SO3FunHarmonic(self) * a

  __rmul__ = __mul__

  def __truediv__(self, a):
    return self * (1 / a)

  # two kernel functions with the same kernel stay one, anything else is a composition
  def __add__(self, a):
    if isinstance(a, (int, float, np.number)):
      return SO3FunRBF(self.center, self.psi, self.weights, self.c0 + a)
    if isinstance(a, SO3FunRBF):
      if a.CS is not self.CS or a.SS is not self.SS:
        raise ValueError("the functions live in different frames")
      # a uniform function has no kernel of its own, so it joins any kernel
      if (a.psi == self.psi or a.center.size == 0 or self.center.size == 0) and a.antipodal == self.antipodal:
        center = orientation._fromRotation(np.concatenate([self.center.data, a.center.data]), (self.CS, self.SS), self.antipodal)
        psi = a.psi if self.center.size == 0 else self.psi
        return SO3FunRBF(center, psi, np.concatenate([self.weights, a.weights]), self.c0 + a.c0)
    from .handles import SO3FunComposition
    return SO3FunComposition(self, a)

  __radd__ = __add__

  def __sub__(self, a):
    return self + (-1) * a

  def __neg__(self):
    return self * -1

  def norm(self):
    return SO3FunHarmonic(self).norm()

  def __repr__(self):
    lines = [f"SO3FunRBF ({self.CS.char()} → {self.SS.char()})"]
    if self.antipodal:
      lines.append("  antipodal: true")
    if self.c0:
      lines += ["  uniform component", f"    weight: {self.c0:.4g}"]
    n = self.center.size
    if n:
      lines += ["  unimodal component" if n == 1 else "  multimodal components", f"    kernel: {self.psi}",
                f"    center: {n} orientation{'s' if n > 1 else ''}", f"    weight: {np.sum(self.weights) * self.psi.mean():.4g}"]
    return "\n".join(lines)


# ------------------------------------------------------------------ fibre components


class SO3FunCBF:
  isReal = True
  antipodal = False

  def __init__(self, h, r=None, weights=None, psi=None, SS=None, *, halfwidth=None):
    # from a fibre the positional arguments shift: SO3FunCBF(fibre, weights, psi)
    if isinstance(h, fibre):
      f = h
      weights, psi = (None, r) if isinstance(r, S2Kernel) else (r, weights)
      h, r = vector3d(f.h.data, frame=f.CS), f.r
      SS = f.SS if SS is None else SS
    h = h if isinstance(h, vector3d) else vector3d(h)
    r = r if isinstance(r, vector3d) else vector3d(r)
    self.CS = h.frame if h.frame is not None else crystalFrame("1")
    self.SS = SS if SS is not None else r.frame if r.frame is not None else specimenFrame.default
    self.h = vector3d(h.normalize().data.reshape(-1, 3))
    self.r = vector3d(r.normalize().data.reshape(-1, 3))
    if self.h.size != self.r.size:
      self.h, self.r = [vector3d(x) for x in np.broadcast_arrays(self.h.data, self.r.data)]
    self.psi = psi if psi is not None else S2DeLaValleePoussinKernel(halfwidth=10 * degree if halfwidth is None else halfwidth)
    n = self.h.size
    self.weights = np.full(n, 1 / (n * self.psi.mean())) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if self.weights.size != n:
      raise ValueError("one crystal direction, one specimen direction and one weight per fibre")

  @property
  def bandwidth(self):
    return self.psi.bandwidth

  def mean(self):
    return float(np.sum(self.weights)) * self.psi.mean()

  # the fibre pairs (t h_i, s^-1 r_i) over the proper rotations, with weight w_i / m
  def _expandedFibres(self):
    L, R = reductionGroups(orientation(rotation.id, self.CS, self.SS))
    hs = (rotation(R)[None, :] * self.h[:, None]).data.reshape(-1, 3)
    rs = (rotation(_conj(L))[None, :] * self.r[:, None]).data.reshape(-1, 3)
    m = len(L) * len(R)
    H = np.repeat(hs.reshape(self.h.size, 1, len(R), 3), len(L), axis=1).reshape(-1, 3)
    Rv = np.repeat(rs.reshape(self.h.size, len(L), 1, 3), len(R), axis=2).reshape(-1, 3)
    return H, Rv, np.repeat(self.weights, m) / m

  def __call__(self, x):
    return self.eval(x)

  def eval(self, x):
    r = _rotationsFor(self, x)
    q = rotation(r.data.reshape(-1, 4))
    H, Rv, w = self._expandedFibres()
    out = np.empty(q.size)
    for start in range(0, q.size, CHUNK):
      gh = (q[start:start + CHUNK][:, None] * vector3d(H)[None, :]).data
      t = np.clip(np.sum(gh * Rv[None, :, :], axis=-1), -1, 1)
      out[start:start + CHUNK] = self.psi(t) @ w
    return out.reshape(r.shape)

  # fhat_l^{ka} = A_l 4pi/(2l+1)^{3/2} Y_l^k(r) conj(Y_l^a(h)), from the addition theorem and H4, projected
  def _harmonic(self, bandwidth=None):
    L = min(self.psi.bandwidth, getMTEXpref("maxSO3Bandwidth") if bandwidth is None else int(bandwidth))
    A = self.psi.calcFourier(L)
    fhat = np.zeros(deg2dim(L + 1), dtype=complex)
    for l in range(L + 1):
      Yh, Yr = _sphericalY(l, self.h), _sphericalY(l, self.r)
      degreeBlock(fhat, l)[...] = A[l] * 4 * np.pi / (2 * l + 1) ** 1.5 * np.einsum("i,ki,ai->ka", self.weights, Yr, np.conj(Yh))
    return SO3FunHarmonic(fhat, self.CS, self.SS, real=True, _internal=True)

  # the gradient: psi'(t) (g h x r) on the left, psi'(t) (h x g^-1 r) on the right, over every pair
  def grad(self, x=None, side="left"):
    if x is None:
      from .so3fun import SO3FunHarmonic
      return SO3FunHarmonic(self).grad(side=side)
    if side not in ("left", "right"):
      raise ValueError(f"the side is 'left' or 'right', not {side!r}")
    r = _rotationsFor(self, x)
    q = rotation(r.data.reshape(-1, 4))
    H, Rv, w = self._expandedFibres()
    out = np.empty((q.size, 3))
    for start in range(0, q.size, CHUNK):
      qb = q[start:start + CHUNK]
      gh = (qb[:, None] * vector3d(H)[None, :]).data
      t = np.clip(np.sum(gh * Rv[None, :, :], axis=-1), -1, 1)
      d = w[None, :] * self.psi.grad(t, polynomial=True)
      if side == "left":
        cross = np.cross(gh, Rv[None, :, :])
      else:
        gr = (qb.inv()[:, None] * vector3d(Rv)[None, :]).data
        cross = np.cross(H[None, :, :], gr)
      out[start:start + CHUNK] = np.einsum("ij,ijk->ik", d, cross)
    return SO3TangentVector(vector3d(out.reshape(*r.shape, 3)), x, side)

  def rotate(self, r, side="left"):
    rot = rotation(r.data.reshape(4)) if not isinstance(r, orientation) else rotation(r)
    if side == "left":
      return SO3FunCBF(vector3d(self.h.data, frame=self.CS), rot * self.r, self.weights, self.psi, _rotatedFrame(self.SS, rot))
    if side == "right":
      return SO3FunCBF(vector3d((rot.inv() * self.h).data, frame=_rotatedFrame(self.CS, rot)), self.r, self.weights, self.psi, self.SS)
    raise ValueError(f"the side is 'left' or 'right', not {side!r}")

  def pdf(self, h, bandwidth=None):
    return SO3FunHarmonic(self, bandwidth=bandwidth).pdf(h)

  def ipdf(self, r, bandwidth=None):
    return SO3FunHarmonic(self, bandwidth=bandwidth).ipdf(r)

  def radon(self, h=None, r=None):
    return SO3FunHarmonic(self).radon(h, r)

  def calcPDF(self, h, r=None, superposition=None):
    from .density import calcPDF
    return calcPDF(self, h, r, superposition=superposition)

  def conv(self, psi):
    return conv(SO3FunHarmonic(self), psi)

  def __mul__(self, a):
    if isinstance(a, (int, float, np.number)):
      return SO3FunCBF(vector3d(self.h.data, frame=self.CS), self.r, a * self.weights, self.psi, self.SS)
    return SO3FunHarmonic(self) * a

  __rmul__ = __mul__

  def __truediv__(self, a):
    return self * (1 / a)

  def __add__(self, a):
    from .handles import SO3FunComposition
    if isinstance(a, (int, float, np.number)):
      return SO3FunComposition(self, uniformODF(self.CS, self.SS) * a)
    return SO3FunComposition(self, a)

  __radd__ = __add__

  def __sub__(self, a):
    return self + (-1) * a

  def __neg__(self):
    return -1 * self

  def norm(self):
    return SO3FunHarmonic(self).norm()

  def __repr__(self):
    n = self.h.size
    lines = [f"SO3FunCBF ({self.CS.char()} → {self.SS.char()})", "  fibre component" if n == 1 else "  fibre components", f"    kernel: {self.psi}"]
    for i in range(n):
      lines.append(f"    fibre: ({self.h[i].x:.3g},{self.h[i].y:.3g},{self.h[i].z:.3g}) || ({self.r[i].x:.3g},{self.r[i].y:.3g},{self.r[i].z:.3g}), weight {self.weights[i]:.3g}")
    lines.append(f"    weight: {self.mean():.4g}")
    return "\n".join(lines)


# ------------------------------------------------------------------ the named ODFs


def _readFrames(args):
  frames = [a for a in args if isinstance(a, referenceFrame)]
  return (frames[0] if frames else crystalFrame("1")), (frames[1] if len(frames) > 1 else specimenFrame.default)


def uniformODF(*frames, antipodal=False):
  CS, SS = _readFrames(frames)
  return SO3FunRBF(orientation(rotation.nan(0), CS, SS, antipodal=antipodal), SO3DeLaValleePoussinKernel(halfwidth=10 * degree), np.zeros(0), 1.0)


def _densityKernel(space, kernel, halfwidth):
  if kernel is not None:
    return kernel
  hw = 10 * degree if halfwidth is None else halfwidth
  return SO3DeLaValleePoussinKernel(halfwidth=hw) if space == "SO3" else S2DeLaValleePoussinKernel(halfwidth=hw)


# a unimodal ODF, the weights normalised to a mean of one, the centres of negligible weight
# dropped unless nothinning
def unimodalODF(center=None, *args, kernel=None, halfwidth=None, weights=None, nothinning=False):
  if isinstance(center, (referenceFrame, SO3Kernel)):
    args, center = (center, *args), None
  kernels = [a for a in args if isinstance(a, SO3Kernel)]
  kernel = kernels[0] if kernels else kernel
  CS, SS = _readFrames([a for a in args if isinstance(a, referenceFrame)])
  if center is None:
    center = orientation(rotation.id, CS, SS)
  elif not isinstance(center, orientation):
    center = orientation(center, CS, SS)
  psi = _densityKernel("SO3", kernel, halfwidth)
  n = center.size
  w = np.full(n, 1 / (n * psi.mean())) if weights is None else np.asarray(weights, dtype=float).reshape(-1) / (np.sum(weights) * psi.mean())
  # the centres whose share of the weight stays below a hundredth of a kernel's peak over
  # their number are dropped, as MATLAB thins
  if not nothinning and weights is not None:
    keep = w / np.sum(w) > 1e-2 / np.real(psi.eval(1.0)) / n
    if not np.all(keep):
      center, w = center.flatten()[np.flatnonzero(keep)], w[keep]
      w = w / (np.sum(w) * psi.mean())
  return SO3FunRBF(center, psi, w)


# a fibre ODF from crystal and specimen directions or from a fibre of P1, the weights
# normalised to a mean of one
def fibreODF(h, r=None, *args, kernel=None, halfwidth=None, weights=None):
  kernels = [a for a in args if isinstance(a, S2Kernel)]
  kernel = kernels[0] if kernels else kernel
  frames = [a for a in args if isinstance(a, referenceFrame)]
  psi = _densityKernel("S2", kernel, halfwidth)
  if isinstance(h, fibre):
    f = h
    h, r = vector3d(f.h.data, frame=f.CS), f.r
    SS = f.SS
  else:
    SS = frames[0] if frames else None
  n = max(h.size, 1 if r is None else r.size)
  w = np.full(n, 1 / (n * psi.mean())) if weights is None else np.asarray(weights, dtype=float).reshape(-1) / (np.sum(weights) * psi.mean())
  return SO3FunCBF(h, r, w, psi, SS)


# an ODF by its Fourier coefficients, or another ODF in its harmonic form (MATLAB FourierODF)
def FourierODF(C, *args, **kwargs):
  from .so3fun import SO3FunHarmonic
  return SO3FunHarmonic(C, *args, **kwargs)


# the Santa Fe sample ODF (MATLAB SantaFe): cubic crystal and orthorhombic specimen symmetry,
# a uniform portion of 0.73 and a von Mises Fisher component of 10 degree halfwidth at
# (122)[221] carrying the other 0.27
def SantaFe():
  from ..geometry.plottingconvention import plottingConvention
  cs = crystalFrame("cubic")
  ss = specimenFrame("mmm", plottingConvention("y↑→x"))
  ori = orientation.byMiller(Miller(1, 2, 2, cs), Miller(2, 2, 1, cs), cs, ss)
  return 0.73 + 0.27 * unimodalODF(ori, cs, ss, SO3vonMisesFisherKernel(halfwidth=10 * degree))


__all__ = ["SO3FunRBF", "SO3FunCBF", "uniformODF", "unimodalODF", "fibreODF", "FourierODF", "supportCosine", "SantaFe"]
