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

from typing import TYPE_CHECKING
import math

import numpy as np

from .so3funclass import SO3Fun
from ..geometry.kernels import _qmulNumpy as _qmul, qconj
from ..geometry.fibre import fibre
from ..geometry.frames import crystalFrame, referenceFrame, specimenFrame
from ..geometry.miller import Miller
from ..geometry.orientation import orientation
from ..geometry.quaternion import orthogonalTransform, quaternion, rotation
from ..geometry.representations import quat2mat
from ..geometry.session import getMTEXpref
from ..geometry.symreduce import reductionGroups
from ..geometry.tangent import SO3TangentVector
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, asVector3d
from ..geometry.display import fields, framesLabel, numString
from .funops import _antipodalPole, _rotatedFrame, _side, conv
from .harmonics import _so3PerDegree, deg2dim, degreeBlock, _sphericalY
from .kernelfunctions import S2DeLaValleePoussinKernel, S2Kernel, SO3DeLaValleePoussinKernel, SO3Kernel, SO3vonMisesFisherKernel
from .quadrature import folds
from .s2fun import S2FunHarmonic
from .so3fun import SO3FunHarmonic
from .transforms import GRIDDED_ROTATIONS_PER_L3, wignerSeriesAdjoint, wignerSeriesAdjointGridded
from ..geometry.display import aligned

CHUNK = 1024
# above this many kernel terms an evaluation goes through the harmonic form
HARMONIC_EVAL = 2_000_000
# the largest block of dots the direct sum forms, and the most pairs one batch of the cell list holds
BLOCK = 1 << 22
HITS = 1 << 23
# the query images paired with the centres in one call
IMAGES = 1 << 18


def supportCosine(psi, tol=1e-9):
  """the cosine of the half angle beyond which the kernel stays below tol of its peak, in
  closed form for the families with one and found by a scan otherwise; a kernel that never
  drops that far is summed in full

    c = supportCosine(psi)            # where psi falls below 1e-9 of psi(1)
    c = supportCosine(psi, tol=1e-4)  # a wider cut, a smaller support

  The de la Vallee Poussin and von Mises-Fisher kernels have it in closed form; any other
  kernel is scanned on a grid of 0.5 degree in the rotation angle for the first angle from
  which every value stays below tol times the peak. The kernel sums of SO3FunRBF evaluate
  only the centres whose cosine to the query exceeds it.

  Input
    psi - SO3Kernel
  Output
    c - the cosine cos(omega/2) of the support's radius, -1 when the kernel never drops below
  Keywords
    tol - the bound relative to the value at the identity
  See also
    SO3FunRBF, SO3Kernel.halfwidth"""
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


# the kernel sum by blocks of queries and of expanded centres, each block at most BLOCK dots
def _blockSum(q, cs, w, psi, cut):
  out = np.zeros(len(q))
  rows = max(1, min(CHUNK, BLOCK // max(len(cs), 1)))
  cols = max(1, BLOCK // rows)
  for i in range(0, len(q), rows):
    for j in range(0, len(cs), cols):
      D = np.abs(q[i:i + rows] @ cs[j:j + cols].T)
      mask = D > cut
      values = np.zeros(D.shape)
      values[mask] = psi(np.minimum(D[mask], 1.0))
      out[i:i + rows] += values @ w[j:j + cols]
  return out


# the kernel sum over the pairs of a cell list of the centres, for a support below 120 degree:
# every symmetric image l q r of a query (and its conjugate for antipodal centres) is paired with
# the centres within the support's chord, each pair one term, so that the copies of the centres
# are never formed. The list holds c and -c, and a ball of chord below 1 holds one of the two.
# A kernel of a closed form is summed in the compiled loop, any other over pairs in batches of HITS
def _cellSum(q, c, w, L, R, antipodal, psi, cut):
  from ..geometry.kernels import backend
  from .kernelfunctions import _dlvpConstant, _vmfConstant
  from .kernels import cellKernelSum, cellList, cellPairs
  radius = math.sqrt(2 - 2 * cut)
  # the two families with a closed form are summed inside the compiled loop
  fused = None
  if backend() == "numba" and type(psi) is SO3DeLaValleePoussinKernel:
    fused = (0, _dlvpConstant(psi.kappa), psi.kappa)
  elif backend() == "numba" and type(psi) is SO3vonMisesFisherKernel:
    fused = (1, _vmfConstant(psi.kappa), psi.kappa)
  cells = cellList(np.vstack([c, -c]), radius)
  n, m = len(c), len(L) * len(R) * (2 if antipodal else 1)
  out = np.zeros(len(q))
  step = max(1, IMAGES // m)
  for start in range(0, len(q), step):
    qb = q[start:start + step]
    imgs = _qmul(_qmul(L[:, None, None, :], qb[None, :, None, :]), R[None, None, :, :]).transpose(1, 0, 2, 3)
    imgs = imgs.reshape(len(qb), -1, 4)
    if antipodal:
      imgs = np.concatenate([imgs, qconj(imgs)], axis=1)
    if fused is not None:
      out[start:start + len(qb)] = cellKernelSum(imgs, imgs.shape[1], cells, radius, n, w, cut, *fused)
      continue
    for rows, cols, dots in cellPairs(imgs.reshape(-1, 4), cells, radius, HITS):
      cols = cols % n
      d = np.abs(dots)
      keep = d > cut
      values = psi(np.minimum(d[keep], 1.0)) * w[cols[keep]]
      out[start:start + len(qb)] += np.bincount(rows[keep] // imgs.shape[1], weights=values, minlength=len(qb))
  return out / m


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


class SO3FunRBF(SO3Fun):
  """a function on SO(3) as a sum of radial kernels at centre orientations, plus a constant: a
  radial basis function ODF

    SO3F = SO3FunRBF(ori, psi, weights)          # kernels psi at the orientations ori
    SO3F = SO3FunRBF(ori, psi, weights, c0)      # plus the constant c0
    SO3F = SO3FunRBF(ori)                        # 10 degree kernels, the weights giving a mean of one

  The function is c0 + sum_i w_i psi(angle(g, ori_i)), every centre taken with all its
  symmetrically equivalent orientations, in the crystal and specimen frames of the centres.
  It stays a kernel sum under scaling, adding a constant, adding another kernel sum of the
  same kernel, rotation, inversion and convolution with a kernel on SO(3); the harmonic form
  it is evaluated through is kept until a centre, the kernel, the weights or the constant
  are set. unimodalODF and calcDensity build one; SO3FunRBF.interpolate fits one through
  values. MATLAB's SO3FunRBF(F) approximating another SO3Fun is not ported.

  Input
    center - orientation, the centres of the kernels
    psi - SO3Kernel; by default the de la Vallee Poussin kernel of halfwidth 10 degree
    weights - one weight per centre; by default equal, giving a mean of one
    c0 - the constant added
  Output
    SO3F - SO3FunRBF
  See also
    unimodalODF, SO3FunCBF, SO3FunRBF.interpolate, calcDensity, SO3FunRBF.radon,
    RadialODFs"""
  isReal = True

  @staticmethod
  def interpolate(nodes, values, **kwargs):
    """the sum of kernels through values at orientations, MATLAB's SO3FunRBF.interpolate"""
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

  # the centres, the kernel, the weights and the constant define the function: setting one drops
  # the harmonic form kept for evaluation; an element assigned in place, f.weights[0] = w, does not
  @property
  def center(self):
    """the centres of the kernels, a flat list of orientations"""
    return self._center

  @center.setter
  def center(self, c):
    self._center, self._harmonicCache = c, None

  @property
  def psi(self):
    """the kernel"""
    return self._psi

  @psi.setter
  def psi(self, psi):
    self._psi, self._harmonicCache = psi, None

  @property
  def weights(self):
    """the weight of every centre"""
    return self._weights

  @weights.setter
  def weights(self, w):
    self._weights, self._harmonicCache = w, None

  @property
  def c0(self):
    """the constant added to the sum of kernels"""
    return self._c0

  @c0.setter
  def c0(self, c):
    self._c0, self._harmonicCache = c, None

  def _harmonicForm(self):
    """the harmonic form at the kernel's bandwidth, kept until the function is changed"""
    if getattr(self, "_harmonicCache", None) is None:
      self._harmonicCache = self._harmonic()
    return self._harmonicCache

  @property
  def CS(self):
    """the crystal frame of the function, that of its centres"""
    return self.center.CS

  @property
  def SS(self):
    """the specimen frame of the function, that of its centres"""
    return self.center.SS

  @SS.setter
  def SS(self, fr):
    """sets the specimen frame of the centres, the rotations kept"""
    self.center = orientation._fromRotation(self.center.data, (self.CS, fr), self.antipodal)

  @property
  def antipodal(self):
    """whether the function is antipodal, symmetric under inversion, as its centres are"""
    return self.center.antipodal

  @property
  def bandwidth(self):
    """the harmonic bandwidth of the kernel"""
    return self.psi.bandwidth

  def mean(self):
    """the mean of the function over SO(3), c0 plus the sum of the weights times the kernel's mean"""
    return self.c0 + float(np.sum(self.weights)) * self.psi.mean()

  # the equivalents s c t of every centre with the weight w / m, the inverses included
  # under exchange symmetry, as (M, 4) quaternions
  def _expandedCentres(self):
    L, R = reductionGroups(self.center)
    c = self.center.data.reshape(-1, 4)
    if self.antipodal:
      c = np.concatenate([c, qconj(c)])
    q = _qmul(_qmul(L[:, None, None, :], c[None, :, None, :]), R[None, None, :, :])
    m = len(L) * len(R) * (2 if self.antipodal else 1)
    w = np.concatenate([self.weights, self.weights]) if self.antipodal else self.weights
    return q.transpose(1, 0, 2, 3).reshape(-1, 4), np.repeat(w, len(L) * len(R)) / m

  def __call__(self, x):
    return self.eval(x)

  def eval(self, x):
    """the kernel sum over the centres and their symmetric copies; a large sum goes through the
    harmonic form, exact for a kernel of finite bandwidth, as MTEX does"""
    r = _rotationsFor(self, x)
    q = r.data.reshape(-1, 4)
    L, R = reductionGroups(self.center)
    if self.center.size * len(L) * len(R) * len(q) > HARMONIC_EVAL and self.psi.bandwidth <= getMTEXpref("maxSO3Bandwidth"):
      return np.real(self._harmonicForm().eval(r)).reshape(r.shape)
    cut = supportCosine(self.psi)
    m = len(L) * len(R) * (2 if self.antipodal else 1)
    # many terms of a local kernel through the cell list of the centres, the rest by blocks
    if cut > 0.5 and self.center.size * m * len(q) > BLOCK:
      out = _cellSum(q, self.center.data.reshape(-1, 4), self.weights.reshape(-1), L, R, self.antipodal, self.psi, cut)
    else:
      cs, w = self._expandedCentres()
      out = _blockSum(q, cs, w, self.psi, cut)
    return (self.c0 + out).reshape(r.shape)

  # the harmonic form: the point measure of the centres convolved with the kernel,
  # projected; the transform runs at the narrower of the two bandwidths, and the mean is
  # known exactly
  def _harmonic(self, bandwidth=None, exact=False):
    L = min(self.psi.bandwidth, getMTEXpref("maxSO3Bandwidth") if bandwidth is None else int(bandwidth))
    fo = folds(self.CS, self.SS)
    if self.center.size == 0:
      fhat = np.zeros(deg2dim(L + 1), dtype=complex)
    # many centres are rounded onto a grid, whose FFT costs nothing per centre (above 2 L^3,
    # where MATLAB's SO3FunRBF/calcFourier crosses at 10 L^3); exact asks for the transform at
    # the centres
    elif self.center.size > GRIDDED_ROTATIONS_PER_L3 * L ** 3 and not exact:
      fhat = wignerSeriesAdjointGridded(self.center, np.ones(self.center.size), L, weights=self.weights, folds=fo)
    else:
      fhat = wignerSeriesAdjoint(self.center, np.ones(self.center.size), L, weights=self.weights, folds=fo)
    fhat *= _so3PerDegree(self.psi.calcFourier(L) / (2 * np.arange(L + 1) + 1))
    fhat[0] = self.mean()
    return SO3FunHarmonic(fhat, self.CS, self.SS, real=True, antipodal=self.antipodal, _internal=True)

  def grad(self, x=None, side="left"):
    """the gradient of a kernel sum at orientations, or without them the vector field of the
    gradients of its harmonic expansion, as MATLAB's grad

      g = SO3F.grad(ori)                # the gradients at ori, tangent vectors on the left
      g = SO3F.grad(ori, side='right')  # tangent vectors on the right, in crystal coordinates
      G = SO3F.grad()                   # the gradient field, an SO3VectorFieldHarmonic

    Every centre contributes minus the kernel's angular derivative along the geodesic to every
    symmetric equivalent within its support; a query with none in reach takes the nearest
    centre alone. The left side expresses a tangent vector in specimen coordinates, the right
    one in crystal coordinates.

    Input
      SO3F - SO3FunRBF
    Output
      g - SO3TangentVector at the orientations
      G - SO3VectorFieldHarmonic without them
    Keywords
      x - the orientations; without them the gradient field is returned
      side - 'left' or 'right', the side of the tangent vectors
    See also
      SO3Kernel.grad, SO3FunHarmonic.grad, SO3TangentVector"""
    if x is None:
      from .so3fun import SO3FunHarmonic
      return SO3FunHarmonic(self).grad(side=side)
    if isinstance(x, str):
      raise TypeError("the gradient of a kernel function is evaluated at orientations")
    _side(side)
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
      qi = qconj(qb[i])
      p = _qmul(cs[j], qi) if side == "left" else _qmul(qi, cs[j])
      n = np.linalg.norm(p[:, 1:], axis=1)
      ok = n > 1e-12
      f0 = np.zeros(len(i))
      f0[ok] = w[j[ok]] * self.psi.grad(np.minimum(D[i[ok], j[ok]], 1.0)) * np.where(p[ok, 0] < 0, -1.0, 1.0) / n[ok]
      np.add.at(out, start + i, -f0[:, None] * p[:, 1:])
    return SO3TangentVector(vector3d(out.reshape(*r.shape, 3)), x, side)

  # the pole figure of one crystal direction: the Radon transform of the kernel centred at the
  # images of h under the centres and the crystal rotations, projected by the specimen frame
  def _poleFigure(self, h, bandwidth=None):
    h = asVector3d(h)
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
      c = np.concatenate([c, qconj(c)])
    pts = rotation(c[:, None, :]) * hs
    w = np.concatenate([self.weights, self.weights]) / 2 if self.antipodal else self.weights
    w = np.repeat(w, hs.size) / hs.size
    s = S2FunHarmonic.quadrature(vector3d(pts.data.reshape(-1, 3), frame=self.SS), np.ones(pts.size), bandwidth=L, weights=4 * np.pi * w, real=True, antipodal=ant)
    return conv(s, self.psi.radon()) + self.c0

  # the inverse pole figure of one specimen direction, the pole figure of the inverse function
  def _inversePoleFigure(self, r, bandwidth=None):
    return self.inv()._poleFigure(r, bandwidth)

  def radon(self, h=None, r=None, superposition=None, bandwidth=None):
    """the Radon transform of a kernel sum: the pole figure of a crystal direction, the inverse
    pole figure of a specimen direction, or the pole density of both

      S2F = SO3F.radon(h)                                # the pole figure of h
      S2F = SO3F.radon(r=r)                              # the inverse pole figure of r
      v = SO3F.radon(h, r)                               # the pole figure of h at r
      S2F = SO3F.radon([h1, h2], superposition=[0.3, 0.7])  # the superposed pole figure

    The Radon transform of f at (h, r) is the mean of f over the orientations taking h to r.
    For a kernel sum it is again a sum of radial kernels on the sphere, the kernel's Radon
    transform at the centres applied to h, returned as a harmonic expansion. Several crystal
    directions are superposed with the weights of superposition. It replaces MATLAB's
    calcPDF, pdf and ipdf.

    Input
      SO3F - SO3FunRBF
      h - Miller or vector3d, the crystal direction, or a list of them to superpose
      r - vector3d, the specimen direction
    Output
      S2F - S2FunHarmonic, the pole figure on the specimen sphere or the inverse pole figure
          on the crystal sphere
      v - the values at r, given h and r
    Keywords
      superposition - the weights of the superposed crystal directions
      bandwidth - the harmonic degree of the result, by default the kernel's
    See also
      calcPoleFigure, SO3FunCBF.radon, plotPF, plotIPF"""
    from .funops import radon
    return radon(self, h, r, superposition, bandwidth)

  def inv(self):
    """the inverse function f(g^-1), the centres inverted, crystal and specimen sides exchanged"""
    return SO3FunRBF(self.center.inv(), self.psi, self.weights, self.c0)

  def rotate(self, r, side="left"):
    """the function rotated by r on the left, the specimen side, or with side='right' on the crystal
    side; the frame of the rotated side turns with it; MATLAB's rotate

      SO3F = rotate(SO3F, rot)                # f(rot^-1 g), the specimen side turned
      SO3F = SO3F.rotate(rot, side='right')   # f(g rot^-1), the crystal side turned

    A rotation that is no symmetry of the rotated side's group strips that group from its
    frame, with a warning.

    Input
      SO3F - SO3FunRBF
      r - rotation, one
    Output
      SO3F - SO3FunRBF
    Keywords
      side - 'left' or 'right'
    See also
      SO3FunHarmonic.rotate, rotate"""
    rot = rotation(r.data.reshape(4)) if not isinstance(r, orientation) else rotation(r)
    if _side(side) == "left":
      center = orientation._fromRotation((rot * rotation(self.center)).data, (self.CS, _rotatedFrame(self.SS, rot)), self.antipodal)
    else:
      center = orientation._fromRotation((rotation(self.center) * rot).data, (_rotatedFrame(self.CS, rot), self.SS), self.antipodal)
    return SO3FunRBF(center, self.psi, self.weights, self.c0)

  def conv(self, psi):
    """the convolution with a kernel psi, again a kernel sum when psi is an SO(3) kernel, otherwise
    through the harmonic expansion"""
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
    """the L2 norm of the function, from its harmonic expansion

      t = norm(odf)"""
    return SO3FunHarmonic(self).norm()

  def __repr__(self):
    lines = [f"SO3FunRBF ({framesLabel(self.CS, self.SS)})"]
    if self.antipodal:
      lines.append("  antipodal: true")
    if self.c0:
      lines += ["  uniform component"] + fields([("weight", f"{self.c0:.4g}")], "    ")
    n = self.center.size
    if n:
      lines += ["  unimodal component" if n == 1 else "  multimodal components"]
      lines += fields([("kernel", self.psi), ("center", f"{n} orientation{'s' if n > 1 else ''}"),
                       ("weight", f"{np.sum(self.weights) * self.psi.mean():.4g}")], "    ")
      lines.append(self._centerTable())
    return aligned("\n".join(lines))

  def _centerTable(self):
    """the centres by their Euler angles with their weights, as MATLAB lists them, a long list
    by its first and last rows"""
    from ..geometry.display import table, takeRows
    from ..geometry.representations import EULER_LABELS, eulerConvention, quat2euler
    take, pos = takeRows(self.center.size)
    convention = eulerConvention()
    e = np.stack(quat2euler(self.center.data.reshape(-1, 4)[take], convention), axis=-1) / degree
    w = np.asarray(self.weights, dtype=float).reshape(-1)[take] * self.psi.mean()
    t = table(list(EULER_LABELS[convention]) + ["weight"], [e[:, 0], e[:, 1], e[:, 2], w], tiny=1e-8, rows=pos)
    return "\n".join("  " + line for line in [f"  {convention} Euler angles in degree"] + t.split("\n"))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    from .handles import _integralSO3 as sum
    from .characteristics import (max_ as max, min_ as min, entropy, textureIndex, discreteSample, volume, fibreVolume,
                                  calcComponents, calcMDF, calcAngleDistribution, calcAxisDistribution, calcMIndex,
                                  calcError, centerSpecimen)
    from .polefigure import calcPoleFigure
    from .discrepancy import discrepancy
    from ..analysis.texture import calcTensor, calcLankford, calcKearnsFactor
    from ..analysis.optimalsample import optimalSample
    from ..plotting.sphericalplot import plotPF, plotIPF
    from ..plotting.sectionplot import plotSection
    def __abs__(self) -> Any:
      """the absolute value of the function, pointwise, as a function handle

      SO3F = abs(odf)"""
    def abs(self) -> Any:
      """the absolute value of the function, pointwise, as a function handle

      SO3F = abs(odf)"""
    def sqrt(self) -> Any:
      """the square root of the function, pointwise, as a function handle

      SO3F = sqrt(odf)"""
    def exp(self) -> Any:
      """the exponential of the function, pointwise, as a function handle

      SO3F = exp(odf)"""
    def log(self) -> Any:
      """the logarithm of the function, pointwise, as a function handle

      SO3F = log(odf)"""
    def __pow__(self, p) -> Any:
      ...
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


# ------------------------------------------------------------------ fibre components


class SO3FunCBF(SO3Fun):
  """a fibre ODF: a sum over fibres h || r of a kernel on the sphere of the angle between g h and r

    SO3F = SO3FunCBF(h, r)                              # the fibre h || r, a 10 degree kernel
    SO3F = SO3FunCBF(h, r, weights, psi)                # several fibres, weighted, the kernel psi
    SO3F = SO3FunCBF(f, halfwidth=10*degree)            # from a fibre f

  The function of the orientation g is sum_i w_i psi(g h_i . r_i), every fibre taken with all
  its symmetrically equivalent fibres; it is constant along each fibre. The crystal frame is
  that of h, the specimen frame that of r or SS. The weights default to equal ones giving a
  mean of one. fibreODF builds one with the weights normalised.

  Input
    h - Miller or vector3d, the crystal directions, or a fibre
    r - vector3d, the specimen directions, one per crystal direction
    weights - one weight per fibre; by default a mean of one
    psi - S2Kernel; by default the de la Vallee Poussin kernel
  Output
    SO3F - SO3FunCBF
  Keywords
    SS - the specimen frame, when r has none
    halfwidth - the halfwidth of the default kernel, 10 degree unless given
  See also
    fibreODF, fibre, SO3FunRBF, SO3FunCBF.radon, FibreODFs"""
  isReal = True
  antipodal = False

  def __init__(self, h, r=None, weights=None, psi=None, SS=None, *, halfwidth=None):
    # from a fibre the positional arguments shift: SO3FunCBF(fibre, weights, psi)
    if isinstance(h, fibre):
      f = h
      weights, psi = (None, r) if isinstance(r, S2Kernel) else (r, weights)
      h, r = vector3d(f.h.data, frame=f.CS), f.r
      SS = f.SS if SS is None else SS
    h = asVector3d(h)
    r = asVector3d(r)
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
    """the harmonic bandwidth of the S2 kernel"""
    return self.psi.bandwidth

  def mean(self):
    """the mean of the function over SO(3), the sum of the weights times the kernel's mean"""
    return float(np.sum(self.weights)) * self.psi.mean()

  # a Laue group holds the inversion, so the fibre of h is that of -h too
  @property
  def _laue(self):
    return bool(self.CS.sym.isLaue or self.SS.sym.isLaue)

  # the fibre pairs (t h_i, s^-1 r_i) over the proper rotations, with weight w_i / m, and
  # the pairs (-t h_i, s^-1 r_i) beside them, each at half the weight, for a Laue group
  def _expandedFibres(self):
    L, R = reductionGroups(orientation(rotation.id, self.CS, self.SS))
    if self._laue:
      H, Rv, w = SO3FunCBF._pairs(self, L, R)
      return np.concatenate([H, -H]), np.concatenate([Rv, Rv]), np.concatenate([w, w]) / 2
    return SO3FunCBF._pairs(self, L, R)

  def _pairs(self, L, R):
    hs = (rotation(R)[None, :] * self.h[:, None]).data.reshape(-1, 3)
    rs = (rotation(qconj(L))[None, :] * self.r[:, None]).data.reshape(-1, 3)
    m = len(L) * len(R)
    H = np.repeat(hs.reshape(self.h.size, 1, len(R), 3), len(L), axis=1).reshape(-1, 3)
    Rv = np.repeat(rs.reshape(self.h.size, len(L), 1, 3), len(R), axis=2).reshape(-1, 3)
    return H, Rv, np.repeat(self.weights, m) / m

  def __call__(self, x):
    return self.eval(x)

  def eval(self, x):
    """the values at rotations or orientations, the kernel sum over the fibres and their symmetric
    equivalents"""
    r = _rotationsFor(self, x)
    q = r.data.reshape(-1, 4)
    H, Rv, w = self._expandedFibres()
    # <g h, r> = sum_ij M(g)_ij r_i h_j: the arguments of every fibre at a block of rotations are
    # one product of their matrices with the outer products r h
    P = (Rv[:, :, None] * H[:, None, :]).reshape(-1, 9).T
    from ..geometry.kernels import backend
    if backend() == "numba" and type(self.psi) is S2DeLaValleePoussinKernel:
      from .kernels import fibreSum
      return fibreSum(q, P, w, self.psi.kappa).reshape(r.shape)
    out = np.empty(len(q))
    step = max(1, BLOCK // max(len(w), 1))
    for start in range(0, len(q), step):
      t = np.clip(quat2mat(q[start:start + step]).reshape(-1, 9) @ P, -1, 1)
      out[start:start + step] = self.psi(t) @ w
    return out.reshape(r.shape)

  # fhat_l^{ka} = A_l 4pi/(2l+1)^{3/2} Y_l^k(r) conj(Y_l^a(h)), from the addition theorem and H4, projected
  def _harmonic(self, bandwidth=None):
    L = min(self.psi.bandwidth, getMTEXpref("maxSO3Bandwidth") if bandwidth is None else int(bandwidth))
    A = self.psi.calcFourier(L)
    fhat = np.zeros(deg2dim(L + 1), dtype=complex)
    for l in range(L + 1):
      # the average over h and -h leaves the even degrees of a Laue group
      if self._laue and l % 2:
        continue
      Yh, Yr = _sphericalY(l, self.h), _sphericalY(l, self.r)
      degreeBlock(fhat, l)[...] = A[l] * 4 * np.pi / (2 * l + 1) ** 1.5 * np.einsum("i,ki,ai->ka", self.weights, Yr, np.conj(Yh))
    return SO3FunHarmonic(fhat, self.CS, self.SS, real=True, _internal=True)

  def grad(self, x=None, side="left"):
    """the gradient of a fibre ODF at orientations, or without them the vector field of the
    gradients of its harmonic expansion, as MATLAB's grad

      g = SO3F.grad(ori)                # the gradients at ori, tangent vectors on the left
      g = SO3F.grad(ori, side='right')  # tangent vectors on the right, in crystal coordinates
      G = SO3F.grad()                   # the gradient field, an SO3VectorFieldHarmonic

    Over every pair of a symmetric equivalent h of the crystal direction and the specimen
    direction r, the gradient is psi'(t) (g h x r) on the left and psi'(t) (h x g^-1 r) on
    the right, t = g h . r and psi' the derivative of the kernel in t.

    Input
      SO3F - SO3FunCBF
    Output
      g - SO3TangentVector at the orientations
      G - SO3VectorFieldHarmonic without them
    Keywords
      x - the orientations; without them the gradient field is returned
      side - 'left' or 'right', the side of the tangent vectors
    See also
      S2Kernel.grad, SO3FunHarmonic.grad, SO3TangentVector"""
    if x is None:
      from .so3fun import SO3FunHarmonic
      return SO3FunHarmonic(self).grad(side=side)
    _side(side)
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
    """the function rotated by r on the left, the specimen side, or with side='right' on the crystal
    side; the frame of the rotated side turns with it; MATLAB's rotate

      SO3F = rotate(SO3F, rot)                # f(rot^-1 g), the specimen side turned
      SO3F = SO3F.rotate(rot, side='right')   # f(g rot^-1), the crystal side turned

    A rotation that is no symmetry of the rotated side's group strips that group from its
    frame, with a warning.

    Input
      SO3F - SO3FunCBF
      r - rotation, one
    Output
      SO3F - SO3FunCBF
    Keywords
      side - 'left' or 'right'
    See also
      SO3FunHarmonic.rotate, rotate"""
    rot = rotation(r.data.reshape(4)) if not isinstance(r, orientation) else rotation(r)
    if _side(side) == "left":
      return SO3FunCBF(vector3d(self.h.data, frame=self.CS), rot * self.r, self.weights, self.psi, _rotatedFrame(self.SS, rot))
    return SO3FunCBF(vector3d((rot.inv() * self.h).data, frame=_rotatedFrame(self.CS, rot)), self.r, self.weights, self.psi, self.SS)

  # the pole figure of one crystal direction, from the harmonic expansion
  def _poleFigure(self, h, bandwidth=None):
    return SO3FunHarmonic(self, bandwidth=bandwidth)._poleFigure(h)

  # the inverse pole figure of one specimen direction, from the harmonic expansion
  def _inversePoleFigure(self, r, bandwidth=None):
    return SO3FunHarmonic(self, bandwidth=bandwidth)._inversePoleFigure(r)

  def radon(self, h=None, r=None, superposition=None, bandwidth=None):
    """the Radon transform of a fibre ODF: the pole figure of a crystal direction, the inverse
    pole figure of a specimen direction, or the pole density of both

      S2F = SO3F.radon(h)                   # the pole figure of h
      S2F = SO3F.radon(r=r)                 # the inverse pole figure of r
      v = SO3F.radon(h, r)                  # the pole figure of h at r

    The Radon transform of f at (h, r) is the mean of f over the orientations taking h to r,
    computed through the harmonic expansion of the bandwidth. Several crystal directions are
    superposed with the weights of superposition. It replaces MATLAB's calcPDF, pdf and
    ipdf.

    Input
      SO3F - SO3FunCBF
      h - Miller or vector3d, the crystal direction, or a list of them to superpose
      r - vector3d, the specimen direction
    Output
      S2F - S2FunHarmonic, the pole figure or the inverse pole figure
      v - the values at r, given h and r
    Keywords
      superposition - the weights of the superposed crystal directions
      bandwidth - the harmonic degree of the result, by default the kernel's
    See also
      calcPoleFigure, SO3FunRBF.radon, plotPF, plotIPF"""
    from .funops import radon
    return radon(self, h, r, superposition, bandwidth)

  def conv(self, psi):
    """the convolution with a kernel psi, through the harmonic expansion"""
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
    """the L2 norm of the function, from its harmonic expansion

      t = norm(odf)"""
    return SO3FunHarmonic(self).norm()

  def __repr__(self):
    n = self.h.size
    pairs = [("kernel", self.psi)]
    for i in range(n):
      # the crystal direction by its rounded indices, as MATLAB shows it
      h = Miller(vector3d(self.h[i].data), self.CS).round().char(family=False)
      r = ",".join(numString(float(x)) for x in self.r[i].data.reshape(3))
      pairs.append(("fibre", f"{h} || ({r}), weight {self.weights[i]:.3g}"))
    pairs.append(("weight", f"{self.mean():.4g}"))
    return aligned("\n".join([f"SO3FunCBF ({framesLabel(self.CS, self.SS)})", "  fibre component" if n == 1 else "  fibre components"] + fields(pairs, "    ")))

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    from .handles import _integralSO3 as sum
    from .characteristics import (max_ as max, min_ as min, entropy, textureIndex, discreteSample, volume, fibreVolume,
                                  calcComponents, calcMDF, calcAngleDistribution, calcAxisDistribution, calcMIndex,
                                  calcError, centerSpecimen)
    from .polefigure import calcPoleFigure
    from .discrepancy import discrepancy
    from ..analysis.texture import calcTensor, calcLankford, calcKearnsFactor
    from ..analysis.optimalsample import optimalSample
    from ..plotting.sphericalplot import plotPF, plotIPF
    from ..plotting.sectionplot import plotSection
    def __abs__(self) -> Any:
      """the absolute value of the function, pointwise, as a function handle

      SO3F = abs(odf)"""
    def abs(self) -> Any:
      """the absolute value of the function, pointwise, as a function handle

      SO3F = abs(odf)"""
    def sqrt(self) -> Any:
      """the square root of the function, pointwise, as a function handle

      SO3F = sqrt(odf)"""
    def exp(self) -> Any:
      """the exponential of the function, pointwise, as a function handle

      SO3F = exp(odf)"""
    def log(self) -> Any:
      """the logarithm of the function, pointwise, as a function handle

      SO3F = log(odf)"""
    def __pow__(self, p) -> Any:
      ...
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


# ------------------------------------------------------------------ the named ODFs


# the crystal and the specimen frame among the arguments, the others passed over
def _framesAmong(args):
  frames = [a for a in args if isinstance(a, referenceFrame)]
  return (frames[0] if frames else crystalFrame("1")), (frames[1] if len(frames) > 1 else specimenFrame.default)


def uniformODF(*frames, antipodal=False):
  """the uniform ODF, constant one, between a crystal and a specimen frame; MATLAB's uniformODF

    odf = uniformODF(cs)                       # the specimen frame the default one
    odf = uniformODF(cs, ss)                   # between two frames
    mdf = uniformODF(cs, cs, antipodal=True)  # a uniform misorientation distribution

  The result is an SO3FunRBF without centres whose constant part is one. Without a frame the
  crystal frame has the point group 1.

  Input
    cs, ss - crystalFrame, specimenFrame (or a second crystalFrame)
  Output
    odf - SO3FunRBF
  Keywords
    antipodal - the function symmetric under g -> g^-1, as a misorientation distribution of
        one phase
  See also
    unimodalODF, fibreODF, SO3FunRBF"""
  CS, SS = _framesAmong(frames)
  return SO3FunRBF(orientation(rotation.nan(0), CS, SS, antipodal=antipodal), SO3DeLaValleePoussinKernel(halfwidth=10 * degree), np.zeros(0), 1.0)


def _densityKernel(space, kernel, halfwidth):
  if kernel is not None:
    return kernel
  hw = 10 * degree if halfwidth is None else halfwidth
  return SO3DeLaValleePoussinKernel(halfwidth=hw) if space == "SO3" else S2DeLaValleePoussinKernel(halfwidth=hw)


def unimodalODF(center=None, *args, kernel=None, halfwidth=None, weights=None, thinning=True):
  """a unimodal ODF, radially symmetric about a modal orientation, its shape a kernel on SO(3)

    odf = unimodalODF(ori)                                 # the de la Vallee Poussin kernel of halfwidth 10 degree
    odf = unimodalODF(ori, halfwidth=15*degree)            # another halfwidth
    odf = unimodalODF(rot, cs, ss)                         # the centre given as a rotation, the frames named
    odf = unimodalODF(ori, psi)                            # the kernel psi, also kernel=psi
    odf = unimodalODF(ori, weights=w, thinning=False)      # one weight per centre, all kept

  The ODF is a kernel sum, SO3FunRBF, with the kernel at the modal orientations, the weights
  normalised to a mean of one. Given weights, the centres of negligible weight, a hundredth
  of a kernel's peak shared among the centres, are dropped unless thinning=False. Without a
  centre the mode is the identity.

  Input
    center - orientation, the modal orientations, or a rotation with the frames following
    args - crystalFrame and specimenFrame of a centre given as a rotation, and an SO3Kernel
  Output
    odf - SO3FunRBF
  Keywords
    kernel - SO3Kernel, the shape; by default the de la Vallee Poussin kernel
    halfwidth - the halfwidth of the default kernel, 10 degree unless given
    weights - one weight per centre
    thinning - drop the centres of negligible weight (MATLAB's flag nothinning keeps them)
  See also
    fibreODF, uniformODF, FourierODF, BinghamODF, SO3FunRBF, RadialODFs"""
  if isinstance(center, (referenceFrame, SO3Kernel)):
    args, center = (center, *args), None
  kernels = [a for a in args if isinstance(a, SO3Kernel)]
  kernel = kernels[0] if kernels else kernel
  CS, SS = _framesAmong([a for a in args if isinstance(a, referenceFrame)])
  if center is None:
    center = orientation(rotation.id, CS, SS)
  elif not isinstance(center, orientation):
    center = orientation(center, CS, SS)
  psi = _densityKernel("SO3", kernel, halfwidth)
  n = center.size
  w = np.full(n, 1 / (n * psi.mean())) if weights is None else np.asarray(weights, dtype=float).reshape(-1) / (np.sum(weights) * psi.mean())
  # the centres whose share of the weight stays below a hundredth of a kernel's peak over
  # their number are dropped, as MATLAB thins
  if thinning and weights is not None:
    keep = w / np.sum(w) > 1e-2 / np.real(psi.eval(1.0)) / n
    if not np.all(keep):
      center, w = center.flatten()[np.flatnonzero(keep)], w[keep]
      w = w / (np.sum(w) * psi.mean())
  return SO3FunRBF(center, psi, w)


def fibreODF(h, r=None, *args, kernel=None, halfwidth=None, weights=None):
  """a fibre ODF, symmetric about the fibre of the orientations taking a crystal direction to a
  specimen direction, its shape a kernel on the sphere

    odf = fibreODF(h, r)                         # the de la Vallee Poussin kernel of halfwidth 10 degree
    odf = fibreODF(h, r, halfwidth=15*degree)    # another halfwidth
    odf = fibreODF(h, r, psi)                    # the S2 kernel psi, also kernel=psi
    odf = fibreODF(h, r, ss)                     # the specimen frame named
    odf = fibreODF(f, weights=w)                 # from a fibre f, one weight per fibre

  The ODF is SO3FunCBF, the kernel of the angle between g h and r, the weights normalised to a
  mean of one. Several crystal and specimen directions give several fibres.

  Input
    h - Miller, the crystal directions, or a fibre
    r - vector3d, the specimen directions
    args - specimenFrame and an S2Kernel
  Output
    odf - SO3FunCBF
  Keywords
    kernel - S2Kernel, the shape; by default the de la Vallee Poussin kernel
    halfwidth - the halfwidth of the default kernel, 10 degree unless given
    weights - one weight per fibre
  See also
    unimodalODF, uniformODF, FourierODF, BinghamODF, SO3FunCBF, FibreODFs"""
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


def FourierODF(C, *args, **kwargs):
  """an ODF by its Fourier coefficients, or another ODF in its harmonic form (MATLAB FourierODF)"""
  from .so3fun import SO3FunHarmonic
  # FourierODF(odf, L): the harmonic form up to the bandwidth L
  if args and np.isscalar(args[0]) and not isinstance(args[0], str) and not isinstance(C, (np.ndarray, list, tuple)):
    kwargs.setdefault("bandwidth", int(args[0]))
    args = args[1:]
  return SO3FunHarmonic(C, *args, **kwargs)


def SantaFe():
  """the Santa Fe sample ODF (MATLAB SantaFe): cubic crystal and orthorhombic specimen symmetry,
  a uniform portion of 0.73 and a von Mises Fisher component of 10 degree halfwidth at
  (122)[221] carrying the other 0.27"""
  from ..geometry.plottingconvention import plottingConvention
  cs = crystalFrame("cubic")
  ss = specimenFrame("mmm", plottingConvention("y↑→x"))
  ori = orientation.byMiller(Miller(1, 2, 2, cs), Miller(2, 2, 1, cs), cs, ss)
  return 0.73 + 0.27 * unimodalODF(ori, cs, ss, SO3vonMisesFisherKernel(halfwidth=10 * degree))


__all__ = ["SO3FunRBF", "SO3FunCBF", "uniformODF", "unimodalODF", "fibreODF", "FourierODF", "supportCosine", "SantaFe"]
