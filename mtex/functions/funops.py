# the operations on harmonic functions beyond the linear algebra: rotation, convolution,
# differentiation, the Radon transforms and the pointwise operations; each derived from H4
# and H5 (docs/plans/functions.md F3), the module functions of MATLAB's @SO3Fun and @S2Fun
from __future__ import annotations

import math

import numpy as np

from ..geometry.arguments import refuseFlags

from ..geometry.orientation import orientation
from ..geometry.quaternion import orthogonalTransform, quaternion, rotation
from ..geometry.session import getMTEXpref
from ..geometry.vector3d import vector3d, asVector3d
from ..geometry.representations import quat2euler
from .harmonics import _WignerD, deg2dim, degreeBlock, dim2deg, _s2Degrees, _s2PerDegree, _so3Degrees, _so3PerDegree, _sphericalY, wignerdDegrees
from .kernelfunctions import S1Kernel, S2Kernel, SO3Kernel
from .quadrature import bandwidthFor, legendreP, quadratureS2Grid, quadratureSO3Grid
from .s1fun import S1FunHarmonic
from .s2fun import S2FunHarmonic
from .so3fun import SO3FunHarmonic
from .transforms import wignerSeriesAdjoint, sphericalSeriesAdjoint


def _side(side):
  if side not in ("left", "right"):
    raise ValueError(f"the side is 'left' or 'right', not {side!r}")
  return side


# ------------------------------------------------------------------ rotation
# f'(R) = f(r^-1 R) moves the function with the specimen, fhat <- conj(D(r)) fhat (rows);
# on the right f'(R) = f(R r^-1) moves it with the crystal, fhat <- fhat conj(D(r)) (columns).
# A bare rotation keeps the frame when it normalises the group and strips it otherwise; an
# orientation carries the new frame of the rotated side


# the Wigner-d matrices d_0(beta) .. d_L(beta), from the compiled table or the recursion
def _wignerdAt(beta, L):
  from ..geometry.kernels import backend
  if backend() != "numba":
    return wignerdDegrees(np.float64(beta), L)
  from .kernels import wignerdTable
  T = wignerdTable(beta, L)
  return ((n, T[deg2dim(n):deg2dim(n + 1)].reshape(2 * n + 1, 2 * n + 1)) for n in range(L + 1))


# conj(D_n) = diag(e^{ik alpha}) d_n(beta) diag(e^{il gamma}) with d_n real: the phases scale
# rows and columns and the real d_n multiplies the real and imaginary parts, half the work of
# a complex product, D_n never formed
def _rotatedCoefficients(fhat, r, side):
  L = dim2deg(len(fhat))
  q = (r.rot if isinstance(r, orthogonalTransform) else r).data.reshape(-1, 4)[0]
  alpha, beta, gamma = (float(e) for e in quat2euler(q, "Matthies"))
  out = np.empty_like(fhat)
  for n, d in _wignerdAt(beta, L):
    k = np.arange(-n, n + 1)
    a, c = np.exp(1j * k * alpha), np.exp(1j * k * gamma)
    F = degreeBlock(fhat, n)
    if side == "left":
      X = c[:, None] * F
      degreeBlock(out, n)[...] = a[:, None] * (d @ X.real + 1j * (d @ X.imag))
    else:
      X = F * a[None, :]
      degreeBlock(out, n)[...] = (X.real @ d + 1j * (X.imag @ d)) * c[None, :]
  return out


# does r map the group onto itself
def _normalises(r, pg):
  els = pg.rot.flatten()
  moved = orthogonalTransform(r * els.rot * r.inv(), els.isImproper)
  return all(moved[i] in pg for i in range(els.size))


def _rotatedFrame(fr, r):
  if _normalises(r, fr.sym):
    return fr
  import warnings
  warnings.warn("Rotating an ODF with specimen symmetry will remove the specimen symmetry")
  return fr.stripSym()


def _oneRotation(r):
  if isinstance(r, orthogonalTransform):
    r = r.rot
  if not isinstance(r, quaternion) or r.size != 1:
    raise TypeError("a function is rotated by one rotation")
  return rotation(r.data.reshape(4))


# an SO(3) function turned from the left (the specimen side) or the right (the crystal side); an
# orientation turns it into the other frame of the orientation
def _rotateSO3(f, r, side):
  if isinstance(r, orientation):
    o = r
    rot = _oneRotation(rotation(o))
    if side == "left":
      if o.CS is not f.SS:
        raise ValueError("a left rotation maps out of the specimen frame of the function")
      return SO3FunHarmonic(_rotatedCoefficients(f._fhat, rot, "left"), f.CS, o.SS, real=f.isReal, symmetrise=False, _internal=True)
    if o.SS is not f.CS:
      raise ValueError("a right rotation maps into the crystal frame of the function")
    return SO3FunHarmonic(_rotatedCoefficients(f._fhat, rot, "right"), o.CS, f.SS, real=f.isReal, symmetrise=False, _internal=True)
  rot = _oneRotation(r)
  fhat = _rotatedCoefficients(f._fhat, rot, side)
  CS = _rotatedFrame(f.CS, rot) if side == "right" else f.CS
  SS = _rotatedFrame(f.SS, rot) if side == "left" else f.SS
  return SO3FunHarmonic(fhat, CS, SS, real=f.isReal, antipodal=f.antipodal and CS is SS, symmetrise=False, _internal=True)


# on the sphere f'(v) = f(r^-1 v), fhat_l <- D_l(r) fhat_l
def _rotateS2(f, r):
  if isinstance(r, orientation):
    if f.frame is not None and r.CS is not f.frame:
      raise ValueError("the orientation maps out of the frame of the function")
    # the function lands in the specimen frame of the orientation, its crystal frame left behind
    plain = S2FunHarmonic(f._fhat, None, real=f.isReal, antipodal=f.antipodal, symmetrise=False, _internal=True)
    g = _rotateS2(plain, rotation(r))
    return S2FunHarmonic(g._fhat, r.SS, real=f.isReal, antipodal=f.antipodal, symmetrise=False, _internal=True)
  rot = _oneRotation(r)
  L = f.bandwidth
  D = _WignerD(rot, bandwidth=L, dtype=f._fhat.real.dtype)
  fhat = np.empty_like(f._fhat)
  for l in range(L + 1):
    fhat[l * l:(l + 1) ** 2] = D[l] @ f._fhat[l * l:(l + 1) ** 2]
  frame = None if f.frame is None else _rotatedFrame(f.frame, rot)
  return S2FunHarmonic(fhat, frame, real=f.isReal, antipodal=f.antipodal, symmetrise=False, _internal=True)


# ------------------------------------------------------------------ convolution
#   SO(3) x SO(3): (f * g)(R) = int f(R S^-1) g(S) dS, blocks F_n G_n / sqrt(2n+1), the crystal
#     frame of g and the specimen frame of f, needing g.SS is f.CS
#   SO(3) x kernel: fhat_n <- A_n / (2n+1) fhat_n
#   SO(3) x S2: (f * s)(h) = int f(R) s(R h) dR, a function of h in the crystal frame,
#     coefficients F_l^T shat_l / sqrt(2l+1), needing s.frame is f.SS
#   S2 x S2: (s * t)(R) = int s(xi) t(R xi) dxi, a function on SO(3) with the frame of s on the
#     crystal side and that of t on the specimen side, fhat_l^{ka} = (-1)^k that_l^{-k} shat_l^a / sqrt(2l+1)
#   S2 x kernel: (s * psi)(v) = (1/4pi) int s(xi) psi(<xi, v>) dxi, shat_l <- A_l / (2l+1) shat_l


def transformODF(odfP, p2c, vSel=None):
  """the child texture of a parent texture, the transformation texture; MATLAB's transformODF,
  called as variants(p2c, odf)

    odfC = transformODF(odfP, p2c)          # every variant equally likely
    odfC = transformODF(odfP, p2c, vSel)    # the parents weighted by a function vSel

  The child ODF at a child orientation is the mean of the parent ODF over the parent
  orientations it can come from, one per variant of p2c, expanded to the bandwidth of the
  parent ODF. With vSel the parents are weighted by its values at them, normalised to mean
  one per child orientation. MATLAB's 'variantSelection' takes a weight per variant or the
  number of one variant instead; the port has no such form.

  Input
    odfP - SO3Fun, the parent ODF
    p2c - orientation, the parent to child orientation relationship
  Output
    odfC - SO3FunHarmonic, the child ODF in the child crystal frame
  Keywords
    vSel - an SO3Fun over the parent orientations, the variant selection
  See also
    orientation.variants, calcParent2Child, unimodalODF"""
  from ..geometry.quaternion import qmul
  PV = p2c.variants(parent=True).data.reshape(-1, 4)

  def f(cori):
    q = qmul(cori.data.reshape(-1, 1, 4), PV[None]).reshape(-1, 4)
    pori = orientation._fromRotation(q, (p2c.frameA, odfP.SS))
    val = np.asarray(odfP.eval(pori)).reshape(-1, len(PV))
    if vSel is not None:
      w = np.asarray(vSel.eval(pori)).reshape(-1, len(PV))
      val = val * w / np.mean(w, axis=1, keepdims=True)
    return np.mean(val, axis=1).reshape(cori.shape)

  return SO3FunHarmonic(f, p2c.frameB, odfP.SS, bandwidth=odfP.bandwidth)


def conv(f, g, *args, inverse=False):
  """conv(f, g): the convolution of f with g; with inverse=True of the inverse of f with g, MATLAB's
  conv(f, g, 'inv')"""
  refuseFlags(args, "conv")
  if inverse:
    return conv(f.inv(), g)
  if isinstance(f, (SO3Kernel, S2Kernel, S1Kernel)) and not isinstance(g, (SO3Kernel, S2Kernel, S1Kernel)):
    return conv(g, f)
  if isinstance(f, S1FunHarmonic):
    return f.conv(g)
  if isinstance(f, SO3FunHarmonic):
    # another representation of a function on SO(3) takes its harmonic form
    if not isinstance(g, (SO3FunHarmonic, SO3Kernel)) and hasattr(g, "_harmonic"):
      g = SO3FunHarmonic(g)
    if isinstance(g, SO3FunHarmonic):
      if g.SS is not f.CS:
        raise ValueError("the specimen frame of the second factor must be the crystal frame of the first")
      L = min(f.bandwidth, g.bandwidth)
      fhat = np.zeros(deg2dim(L + 1), dtype=np.result_type(f._fhat, g._fhat))
      for n in range(L + 1):
        degreeBlock(fhat, n)[...] = degreeBlock(f._fhat, n) @ degreeBlock(g._fhat, n) / math.sqrt(2 * n + 1)
      return SO3FunHarmonic(fhat, g.CS, f.SS, real=f.isReal and g.isReal, symmetrise=False, _internal=True)
    if isinstance(g, SO3Kernel):
      L = min(f.bandwidth, g.bandwidth)
      return _convDegrees(f, g.calcFourier(L) / (2 * np.arange(L + 1) + 1))
    if isinstance(g, S2FunHarmonic):
      if g.frame is not None and g.frame is not f.SS:
        raise ValueError("the spherical function must live in the specimen frame of the SO(3) function")
      L = min(f.bandwidth, g.bandwidth)
      fhat = np.zeros((L + 1) ** 2, dtype=np.result_type(f._fhat, g._fhat))
      for l in range(L + 1):
        fhat[l * l:(l + 1) ** 2] = degreeBlock(f._fhat, l).T @ g._fhat[l * l:(l + 1) ** 2] / math.sqrt(2 * l + 1)
      return S2FunHarmonic(fhat, f.CS, real=f.isReal and g.isReal, symmetrise=False, _internal=True)
    if isinstance(g, (np.ndarray, list)):
      return _convDegrees(f, np.asarray(g, dtype=float))
  if isinstance(f, S2FunHarmonic):
    if isinstance(g, S2FunHarmonic):
      L = min(f.bandwidth, g.bandwidth)
      fhat = np.zeros(deg2dim(L + 1), dtype=np.result_type(f._fhat, g._fhat))
      for l in range(L + 1):
        k = np.arange(-l, l + 1)
        t = g._fhat[l * l:(l + 1) ** 2][::-1]
        s = f._fhat[l * l:(l + 1) ** 2]
        # the normalised measure of the sphere, 1 / (4 pi) of its area, as MATLAB's
        degreeBlock(fhat, l)[...] = np.outer(np.where(k % 2, -1.0, 1.0) * t, s) / math.sqrt(2 * l + 1) / (4 * math.pi)
      from ..geometry.frames import specimenFrame
      def frameOf(x):
        return specimenFrame.default if x.frame is None else x.frame
      return SO3FunHarmonic(fhat, frameOf(f), frameOf(g), real=f.isReal and g.isReal, symmetrise=False, _internal=True)
    if isinstance(g, (S2Kernel, SO3Kernel)):
      L = min(f.bandwidth, g.bandwidth)
      A = g.calcFourier(L) / (2 * np.arange(L + 1) + 1)
      return S2FunHarmonic(_s2PerDegree(A) * f.truncate(L)._fhat, f.frame, real=f.isReal, antipodal=f.antipodal, symmetrise=False, _internal=True)
  raise TypeError(f"no convolution of {type(f).__name__} with {type(g).__name__}")


# a multiplier per degree, A[n] on degree n and zero beyond the vector: MATLAB's conv(odf, A)
# with a plain vector, which is how the odd degrees are zeroed or flipped
def _convDegrees(f, A):
  L = min(f.bandwidth, len(A) - 1)
  fhat = f.truncate(L)._fhat * _so3PerDegree(A[:L + 1])
  return SO3FunHarmonic(fhat, f.CS, f.SS, real=f.isReal, antipodal=f.antipodal, symmetrise=False, _internal=True)


# ------------------------------------------------------------------ differentiation
# the gradient lives with the vector fields; a harmonic function gives a harmonic field,
# anything else the central difference handle


def grad(f, x=None, side="left"):
  """the gradient of a function: the derivative of a function on the circle, the tangent vector
  field of a function on the sphere or on SO(3), left or right on SO(3); with x its values at x

    grad(sF)
    grad(odf, ori)
    grad(odf, side='right')"""
  if isinstance(f, S1FunHarmonic):
    return f.diff() if x is None else f.diff()(x)
  if isinstance(f, S2FunHarmonic):
    from .s2vectorfield import grad as gradS2
    return gradS2(f, x)
  from .so3vectorfield import grad as gradSO3
  return gradSO3(f, x, side)


def laplace(f):
  """the Laplace-Beltrami operator: fhat_n <- -n(n+1) fhat_n on both spaces"""
  n = (_so3Degrees if isinstance(f, SO3FunHarmonic) else _s2Degrees)(f.bandwidth)
  return f._rebuild(-(n * (n + 1)) * f._fhat)


# ------------------------------------------------------------------ the Radon transform
# the mean of f over the fibre {R : R h = r},
#   Rf(h, r) = sum_l (4pi/sqrt(2l+1)) sum_{k,m} fhat_l^{km} conj(Y_l^k(r)) Y_l^m(h)
# radon(f, h) is the function of r in the specimen frame, the pole figure of h; radon(f, r=r) the
# function of h in the crystal frame. A pole figure of a Laue crystal or of an axis is axis valued


# a pole figure is axis valued when h and -h are the same pole: an axis, a Laue group, or a
# group with an element sending h to -h
def _antipodalPole(h, fr):
  if h.antipodal:
    return True
  pg = fr.sym
  if pg.isLaue:
    return True
  u = h.normalize().data.reshape(3)
  moved = pg.rot.flatten().matrix() @ u
  return bool(np.any(moved @ u < -1 + 1e-8))


def _oneVector(h, frame, what):
  v = asVector3d(h)
  if v.size != 1:
    raise ValueError(f"one {what} at a time")
  if v.frame is not None and frame is not None and v.frame is not frame:
    raise ValueError(f"the {what} must live in the {'crystal' if what == 'crystal direction' else 'specimen'} frame of the function")
  return v


# the pole figure of one crystal direction h, a function on the sphere in the specimen frame: the
# mean of f over the orientations sending h to each direction; axis valued where h and -h coincide
def _harmonicPoleFigure(f, h):
  h = _oneVector(h, f.CS, "crystal direction")
  L = f.bandwidth
  fhat = np.zeros((L + 1) ** 2, dtype=f._fhat.dtype)
  hv = vector3d(h.normalize().data.reshape(3))
  for l in range(L + 1):
    Y = _sphericalY(l, hv)
    F = degreeBlock(f._fhat, l)
    k = np.arange(-l, l + 1)
    fhat[l * l:(l + 1) ** 2] = 4 * np.pi / math.sqrt(2 * l + 1) * np.where(k % 2, -1.0, 1.0) * (F[::-1] @ Y)
  return S2FunHarmonic(fhat, f.SS, real=f.isReal, antipodal=_antipodalPole(h, f.CS), symmetrise=False, _internal=True)


# the inverse pole figure of one specimen direction r, a function on the sphere in the crystal
# frame: the mean of f over the orientations sending each crystal direction to r
def _harmonicInversePoleFigure(f, r):
  r = _oneVector(r, f.SS, "specimen direction")
  L = f.bandwidth
  fhat = np.zeros((L + 1) ** 2, dtype=f._fhat.dtype)
  rv = vector3d(r.normalize().data.reshape(3))
  for l in range(L + 1):
    Y = _sphericalY(l, rv)
    F = degreeBlock(f._fhat, l)
    fhat[l * l:(l + 1) ** 2] = 4 * np.pi / math.sqrt(2 * l + 1) * (np.conj(Y) @ F)
  return S2FunHarmonic(fhat, f.CS, real=f.isReal, antipodal=_antipodalPole(r, f.SS), symmetrise=False, _internal=True)


def radon(f, h=None, r=None, superposition=None, bandwidth=None):
  """radon(f, h) the pole figure, radon(f, r=r) the inverse pole figure, radon(f, h, r) the pole
  figure at the specimen directions r; several crystal directions h give their superposed pole
  figure, weighted by superposition, equally unless given (MATLAB's pdf, ipdf and calcPDF);
  bandwidth that of the harmonic form a function without one is expanded to;
  on the sphere the Funk transform, the mean over the great circle orthogonal to v,
  shat_l <- P_l(0) shat_l
  radon(sF): the mean over the great circle perpendicular to every direction; radon(sF,
  delta) over the small circle at pi/2 + delta from it, by Funk-Hecke the degree n scaled
  by P_n(cos(pi/2 + delta)); the great circle holds a direction with its antipode"""
  if isinstance(f, S2FunHarmonic):
    delta = 0.0 if h is None else float(h)
    P = legendreP(f.bandwidth, math.cos(math.pi / 2 + delta))
    return S2FunHarmonic(_s2PerDegree(P) * f._fhat, f.frame, real=f.isReal, antipodal=delta == 0 or f.antipodal,
                         symmetrise=False, _internal=True)
  if h is None and r is None:
    raise TypeError("radon needs the crystal direction h, the specimen direction r, or both")
  if h is None or (not isinstance(h, vector3d) and len(h) == 0):
    return f._inversePoleFigure(asVector3d(r), bandwidth)
  if isinstance(h, (list, tuple)):
    hs = [asVector3d(x) for x in h]
  else:
    h = asVector3d(h)
    hs = [h[i] for i in range(h.size)] if h.size > 1 else [h]
  if superposition is None and len(hs) == 1:
    P = f._poleFigure(hs[0], bandwidth)
  else:
    c = np.full(len(hs), 1 / len(hs)) if superposition is None else np.asarray(superposition, dtype=float).reshape(-1)
    if len(c) != len(hs):
      raise ValueError("one superposition coefficient per crystal direction")
    P = sum(ci * f._poleFigure(hi, bandwidth) for ci, hi in zip(c, hs))
  return P if r is None else P(r)


def invRadon(s):
  """the inverse of the Funk transform on the sphere: the even function whose means over the great
  circles are s, the odd degrees zero"""
  P = legendreP(s.bandwidth, 0.0)
  l = _s2Degrees(s.bandwidth)
  odd = l % 2 == 1
  fhat = s._fhat / np.where(odd, 1.0, P[l])
  fhat[..., odd] = 0
  return S2FunHarmonic(fhat, s.frame, real=s.isReal, antipodal=True, symmetrise=False, _internal=True)


# ------------------------------------------------------------------ pointwise operations
# by quadrature at the bandwidth the operation needs, capped by the session preference; the
# operands are harmonic, so they are evaluated on the quadrature grid and take the FFT route


def smoothBandwidth(*fs):
  """an operation that is not polynomial is approximated at twice the bandwidth, at least 32"""
  return max(2 * max(f.bandwidth for f in fs), 32)


# the bandwidth of a pointwise operation: the one given, else the sum of the operands' capped by
# the preference
def _mapBandwidth(fs, bandwidth, preference):
  return min(sum(x.bandwidth for x in fs), getMTEXpref(preference)) if bandwidth is None else int(bandwidth)


def mapFun(op, f, *gs, bandwidth=None):
  """the harmonic function of op applied pointwise to f and gs, by quadrature at the sum of their
  bandwidths, capped by the preference, or at the given bandwidth

    h = mapFun(np.exp, f)                      # exp of one function
    h = mapFun(np.maximum, f, g, bandwidth=64)  # the pointwise maximum of two, of bandwidth 64

  op is evaluated at the nodes of a quadrature grid of the bandwidth, with the values of f and
  gs as its arguments, and the result expanded again; the functions are on SO(3)
  (SO3FunHarmonic), on the sphere (S2FunHarmonic) or on the circle (S1FunHarmonic) and must
  share their frames. A list of functions on SO(3) gives a list, op reducing over the axes
  after the nodes. It is what exp, log, sqrt, max and min of the harmonic functions use.

  Input
    op - a function of arrays, applied to the values
    f - SO3FunHarmonic, S2FunHarmonic or S1FunHarmonic
    gs - further functions of the same kind and frames
  Output
    h - the harmonic function of the kind of f
  Keywords
    bandwidth - the bandwidth of the result; by default the sum of the bandwidths, at most
        the preference maxSO3Bandwidth, maxS2Bandwidth or maxS1Bandwidth
  See also
    SO3FunHarmonic.exp, SO3FunHarmonic.sqrt, max, SO3FunHarmonic.quadrature"""
  if isinstance(f, SO3FunHarmonic):
    for g in gs:
      f._sameFrames(g)
    L0 = _mapBandwidth((f, *gs), bandwidth, "maxSO3Bandwidth")
    L = bandwidthFor(L0, f.CS, f.SS)
    q = quadratureSO3Grid(L, f.CS, f.SS)
    values = np.asarray(op(f(q), *(g(q) for g in gs)))
    # values of a list of functions: the nodes first, then the shape of the list
    shape = values.shape[1:]
    cols = values.reshape(values.shape[0], -1)
    fhat = np.stack([wignerSeriesAdjoint(q, cols[:, k], L)[:deg2dim(L0 + 1)] for k in range(cols.shape[1])]).reshape(shape + (deg2dim(L0 + 1),))
    return SO3FunHarmonic(fhat, f.CS, f.SS, real=values.dtype.kind != "c", antipodal=f.antipodal and all(g.antipodal for g in gs), _internal=True)
  if isinstance(f, S2FunHarmonic):
    for g in gs:
      f._sameFrames(g)
    L = _mapBandwidth((f, *gs), bandwidth, "maxS2Bandwidth")
    q = quadratureS2Grid(L)
    values = op(f(q), *(g(q) for g in gs))
    return S2FunHarmonic(sphericalSeriesAdjoint(q, values, L), f.frame, real=values.dtype.kind != "c", antipodal=f.antipodal and all(g.antipodal for g in gs), _internal=True)
  L = _mapBandwidth((f, *gs), bandwidth, "maxS1Bandwidth")
  return S1FunHarmonic(lambda x: op(f(x), *(g(x) for g in gs)), bandwidth=L)


__all__ = ["mapFun", "transformODF"]
