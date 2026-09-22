# the density estimates on SO(3), the sphere and the circle, the superposed pole figures and
# the least squares approximation from samples (MATLAB calcDensity, calcPDF, interpolate;
# docs/plans/functions.md F4)
#
#   calcDensity(ori, halfwidth=, kernel=, weights=, harmonic=None, bandwidth=)
#   calcDensity(v, halfwidth=, kernel=, weights=, bandwidth=)
#   calcDensity(rho, sigma=, kernel=, weights=, antipodal=)
#   calcPDF(odf, h, r, superposition=)
#   interpolate(ori | v, values, bandwidth=, lambda_=, s=, weights=)
from __future__ import annotations

import math

import numpy as np

from ..geometry.frames import crystalFrame
from ..geometry.orientation import orientation
from ..geometry.quaternion import quaternion, rotation
from ..geometry.session import getMTEXpref
from ..geometry.vector3d import vector3d
from .funops import conv
from .harmonics import deg2dim, dim2deg, invariantProjection, project
from .kernelfunctions import SO3DirichletKernel
from .rbf import SO3FunRBF, _densityKernel, uniformODF
from .s1fun import calcDensity as _calcDensityS1
from .s2fun import S2FunHarmonic
from .so3fun import SO3FunHarmonic, invertedCoefficients
from .transforms import S2FourierPlan, SO3FourierPlan


def calcDensity(x, **kwargs):
  if isinstance(x, quaternion):
    return _calcDensitySO3(x, **kwargs)
  if isinstance(x, vector3d):
    return _calcDensityS2(x, **kwargs)
  # angles are periodic by nature, the flag names what they are
  kwargs.pop("periodic", None)
  return _calcDensityS1(x, **kwargs)


# the ODF of a list of orientations: the kernel form up to 200 orientations, the harmonic
# form above or on request, an empty list the uniform ODF; NaN orientations are dropped
def _calcDensitySO3(o, kernel=None, halfwidth=None, weights=None, harmonic=None, bandwidth=None):
  o = o if isinstance(o, orientation) else orientation(o)
  psi = _densityKernel("SO3", kernel, halfwidth)
  q = o.data.reshape(-1, 4)
  w = np.ones(len(q)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
  keep = ~np.any(np.isnan(q), axis=1)
  q, w = q[keep], w[keep]
  if len(q) == 0:
    return uniformODF(o.CS, o.SS, antipodal=o.antipodal)
  w = w / np.sum(w)
  f = SO3FunRBF(orientation._fromRotation(q, (o.CS, o.SS), o.antipodal), psi, w / psi.mean())
  if harmonic is None:
    harmonic = (len(q) > 200 and psi.bandwidth <= 96) or isinstance(psi, SO3DirichletKernel) or isinstance(o.SS, crystalFrame)
  if not harmonic:
    return f
  return SO3FunHarmonic(f, bandwidth=min(psi.bandwidth, getMTEXpref("maxSO3Bandwidth")) if bandwidth is None else bandwidth)


# the density of directions or axes on the sphere: the point measure convolved with the
# kernel, projected by the frame, even for axes
def _calcDensityS2(v, kernel=None, halfwidth=None, weights=None, bandwidth=None, antipodal=False):
  psi = _densityKernel("S2", kernel, halfwidth)
  # the estimate of axes is even; the flag on the data or on the call says so
  antipodal = bool(antipodal) or bool(v.antipodal)
  d = v.data.reshape(-1, 3)
  w = np.ones(len(d)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
  keep = ~np.any(np.isnan(d), axis=1)
  d, w = d[keep], w[keep]
  if len(d) == 0:
    return S2FunHarmonic([math.sqrt(4 * np.pi)], v.frame, real=True, antipodal=antipodal)
  L = min(getMTEXpref("maxS2Bandwidth") if bandwidth is None else int(bandwidth), psi.bandwidth)
  pts = vector3d(d, frame=v.frame, antipodal=antipodal).normalize()
  s = S2FunHarmonic.quadrature(pts, np.ones(len(d)), bandwidth=L, weights=4 * np.pi * w / np.sum(w), real=True, antipodal=antipodal)
  return conv(s, psi)


# the pole figure of superposed crystal directions, sum_k c_k pdf(f, h_k), as a function or
# at specimen directions r
def calcPDF(f, h, r=None, superposition=None):
  # calcPDF(odf, None, r): the inverse pole density function of the specimen direction r
  if h is None or (not isinstance(h, vector3d) and len(h) == 0):
    return f.ipdf(r if isinstance(r, vector3d) else vector3d(r))
  if isinstance(h, (list, tuple)):
    hs = [x if isinstance(x, vector3d) else vector3d(x) for x in h]
  else:
    h = h if isinstance(h, vector3d) else vector3d(h)
    hs = [h[i] for i in range(h.size)] if h.size > 1 else [h]
  c = np.full(len(hs), 1 / len(hs)) if superposition is None else np.asarray(superposition, dtype=float).reshape(-1)
  if len(c) != len(hs):
    raise ValueError("one superposition coefficient per crystal direction")
  P = sum(ci * f.pdf(hi) for ci, hi in zip(c, hs))
  return P if r is None else P(r)


# ------------------------------------------------------------------ least squares from samples
# min sum w_i |f(R_i) - y_i|^2 + lambda sum_n (1 + n(n+1))^s |fhat_n|^2, by LSQR on the
# operator and its adjoint


def _sobolevWeightsSO3(L, s):
  return np.concatenate([np.full((2 * n + 1) ** 2, (1 + n * (n + 1)) ** s) for n in range(L + 1)])


def _sobolevWeightsS2(L, s):
  return np.concatenate([np.full(2 * l + 1, (1 + l * (l + 1)) ** s) for l in range(L + 1)])


def _lsqr(A, At, b, n, maxit, tol):
  from scipy.sparse.linalg import LinearOperator, lsqr
  op = LinearOperator((len(b), n), matvec=A, rmatvec=At, dtype=complex)
  return lsqr(op, b, atol=tol, btol=tol, iter_lim=maxit)[0]


def interpolate(nodes, values=None, *, bandwidth=None, lambda_=None, s=2, weights=None, maxit=100, tol=1e-6):
  from ..geometry.tangent import SO3TangentVector
  if isinstance(nodes, SO3TangentVector):
    from .so3vectorfield import interpolateField
    return interpolateField(nodes, bandwidth=bandwidth, lambda_=lambda_, s=s, weights=weights, maxit=maxit, tol=tol)
  if isinstance(nodes, vector3d) and isinstance(values, vector3d):
    from .s2vectorfield import interpolateField
    return interpolateField(nodes, values, bandwidth=bandwidth, lambda_=lambda_, s=s, weights=weights, maxit=maxit, tol=tol)
  y = np.asarray(values).ravel()
  real = y.dtype.kind != "c"
  if isinstance(nodes, quaternion):
    o = nodes if isinstance(nodes, orientation) else orientation(nodes)
    lam = 1e-8 if lambda_ is None else lambda_
    q = o.data.reshape(-1, 4)
    nA, nB = o.CS.sym.properGroup().rotations().size, o.SS.sym.properGroup().rotations().size
    nSym = nA * nB * (2 if real else 1)
    L = min(dim2deg(max(1, round(len(q) * nSym / 2))), getMTEXpref("maxSO3Bandwidth")) if bandwidth is None else int(bandwidth)
    w = np.full(len(q), 1 / len(q)) if weights is None else np.asarray(weights, dtype=float).ravel() / np.sum(weights)
    sw = np.sqrt(w)
    p = SO3FourierPlan(rotation(q), L, real=False)
    reg = math.sqrt(lam) * np.sqrt(_sobolevWeightsSO3(L, s))
    # the unknown is a symmetric function, so the fit is posed in the invariant subspace: a
    # free fit projected afterwards is the fit divided by the order of the group wherever
    # the samples cover one fundamental region and leave the copies unconstrained
    P, Q = invariantProjection(o.SS, L), invariantProjection(o.CS, L)

    def sym(x):
      v = project(np.array(x, dtype=complex), P, Q)
      return (v + invertedCoefficients(v)) / 2 if o.antipodal else v

    A = lambda x: (v := sym(x), np.concatenate([sw * p(v), reg * v]))[1]
    At = lambda z: sym(p.adjoint(sw * z[:len(q)], np.ones(len(q))) + reg * z[len(q):])
    b = np.concatenate([sw * y, np.zeros(len(reg))]).astype(complex)
    fhat = sym(_lsqr(A, At, b, deg2dim(L + 1), maxit, tol))
    return SO3FunHarmonic(fhat, o.CS, o.SS, real=real, antipodal=o.antipodal)
  v = (nodes if isinstance(nodes, vector3d) else vector3d(nodes)).flatten().normalize()
  lam = 0.0 if lambda_ is None else lambda_
  L = math.ceil(math.sqrt(v.size / 2)) if bandwidth is None else int(bandwidth)
  w = np.full(v.size, 1 / v.size) if weights is None else np.asarray(weights, dtype=float).ravel() / np.sum(weights)
  sw = np.sqrt(w)
  p = S2FourierPlan(v, L)
  reg = math.sqrt(lam) * np.sqrt(_sobolevWeightsS2(L, s))
  A = lambda x: np.concatenate([sw * p(np.asarray(x, dtype=complex)), reg * x])
  At = lambda z: p.adjoint(sw * z[:v.size], np.ones(v.size)) + reg * z[v.size:]
  b = np.concatenate([sw * y, np.zeros(len(reg))]).astype(complex)
  fhat = _lsqr(A, At, b, (L + 1) ** 2, maxit, tol)
  return S2FunHarmonic(fhat, v.frame, real=real, antipodal=v.antipodal)


__all__ = ["calcDensity", "calcPDF", "interpolate"]
