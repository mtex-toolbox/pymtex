# the density estimates on SO(3), the sphere and the circle, the superposed pole figures and
# the least squares approximation from samples (MATLAB calcDensity, calcPDF, interpolate;
# docs/plans/functions.md F4)
#
#   calcDensity(ori | ebsd, halfwidth=, kernel=, weights=, groups=, harmonic=None, bandwidth=)
#   calcDensity(v, halfwidth=, kernel=, weights=, groups=, bandwidth=)
#   without a halfwidth or kernel the halfwidth is selected, calcKernel(x, 'UCV')
#   calcDensity(rho, periodic=True, sigma=, kernel=, weights=, antipodal=)
#   calcDensity(x, range=, bandwidth=, weights=)    real valued data, one sample per row
#   calcPDF(odf, h, r, superposition=)
#   interpolate(ori | v, values, bandwidth=, lambda_=, s=, weights=)
from __future__ import annotations

import math
import warnings

import numpy as np

from ..geometry.frames import crystalFrame
from ..geometry.kernels import blasThreads
from ..geometry.orientation import orientation
from ..geometry.quaternion import quaternion, rotation
from ..geometry.session import getMTEXpref
from ..geometry.vector3d import vector3d
from .funops import conv
from .harmonics import deg2dim, dim2deg, invariantProjection, project
from .kde import calcDensityReal
from .kernelfunctions import SO3DirichletKernel
from .rbf import SO3FunRBF, _densityKernel, uniformODF
from .s1fun import calcDensity as _calcDensityS1
from .s2fun import S2FunHarmonic
from .so3fun import SO3FunHarmonic, invertedCoefficients
from .transforms import S2FourierPlan, SO3FourierPlan


def calcDensity(x, Fourier=False, **kwargs):
  # a map: the orientations of its one phase, with its grain ids when grains were computed
  if not isinstance(x, (quaternion, vector3d, np.ndarray, list, tuple)) and hasattr(x, "orientations") and hasattr(x, "prop"):
    if kwargs.get("groups") is None and "grainId" in x.prop:
      kwargs["groups"] = np.asarray(x.prop["grainId"]).reshape(-1)
    x = x.orientations
  if isinstance(x, quaternion):
    f = _calcDensitySO3(x, **kwargs)
    # MATLAB's 'Fourier' asks for the harmonic form
    return SO3FunHarmonic(f) if Fourier and not isinstance(f, SO3FunHarmonic) else f
  if isinstance(x, vector3d):
    return _calcDensityS2(x, **kwargs)
  # angles are periodic, the flag or an option of the circle says so; other numbers are real
  # valued data, MATLAB's kernel density estimate on an interval or a box
  if kwargs.pop("periodic", False) or any(k in kwargs for k in ("sigma", "kernel", "antipodal")):
    return _calcDensityS1(x, **kwargs)
  return calcDensityReal(x, **kwargs)


# the ODF of a list of orientations: the kernel form up to 200 orientations, the harmonic
# form above or on request, an empty list the uniform ODF; NaN orientations are dropped
def _calcDensitySO3(o, kernel=None, halfwidth=None, weights=None, groups=None, harmonic=None, bandwidth=None):
  o = o if isinstance(o, orientation) else orientation(o)
  psi = _densityKernel("SO3", kernel, halfwidth) if kernel is not None or halfwidth is not None else _selectedKernel(o, weights, groups)
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
def _calcDensityS2(v, kernel=None, halfwidth=None, weights=None, groups=None, bandwidth=None, antipodal=False):
  # the estimate of axes is even; the flag on the data or on the call says so
  antipodal = bool(antipodal) or bool(v.antipodal)
  psi = _densityKernel("S2", kernel, halfwidth) if kernel is not None or halfwidth is not None else \
      _selectedKernel(vector3d(v.data, frame=v.frame, antipodal=antipodal), weights, groups)
  d = v.data.reshape(-1, 3)
  w = np.ones(len(d)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
  keep = ~np.any(np.isnan(d), axis=1)
  d, w = d[keep], w[keep]
  if len(d) == 0:
    return S2FunHarmonic([math.sqrt(4 * np.pi)], v.frame, real=True, antipodal=antipodal, _internal=True)
  L = min(getMTEXpref("maxS2Bandwidth") if bandwidth is None else int(bandwidth), psi.bandwidth)
  pts = vector3d(d, frame=v.frame, antipodal=antipodal).normalize()
  s = S2FunHarmonic.quadrature(pts, np.ones(len(d)), bandwidth=L, weights=4 * np.pi * w / np.sum(w), real=True, antipodal=antipodal)
  return conv(s, psi)


# the kernel selected from the sample by UCV with its noise floor; MATLAB's 10 degrees for the
# pixels of a map without their grains, with a warning, and for fewer than two points
def _selectedKernel(x, weights, groups):
  from .characteristics import calcKernel
  space = "S2" if isinstance(x, vector3d) else "SO3"
  if isinstance(x, orientation) and x.notIID and groups is None:
    warnings.warn("the orientations come from a map, whose neighbouring pixels are not independent, so the halfwidth "
                  "was not selected and is 10 degrees; pass the map with its grains computed, groups=ebsd.grainId, "
                  "or a halfwidth", stacklevel=4)
    return _densityKernel(space, None, None)
  try:
    return calcKernel(x, "UCV", weights=weights, groups=groups)
  except ValueError:
    return _densityKernel(space, None, None)


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
  return _lsqrFull(A, At, b, n, maxit, tol)[0]


# the solution, scipy's stop reason, the iterations and the residual norm; the operators are
# transforms and the iteration's norms and products of vectors run on BLAS, whose waiting
# threads would take the transforms' cores (docs/research/threads.md)
def _lsqrFull(A, At, b, n, maxit, tol):
  from .lsqr import lsqr
  with blasThreads(1):
    return lsqr(A, At, np.asarray(b, dtype=complex), n, tol=tol, maxit=maxit)


def interpolate(nodes, values=None, *, bandwidth=None, lambda_=None, s=2, weights=None, maxit=100, tol=None,
                withParameters=False):
  from ..geometry.tangent import SO3TangentVector
  if isinstance(nodes, SO3TangentVector):
    from .so3vectorfield import interpolateField
    return interpolateField(nodes, bandwidth=bandwidth, lambda_=lambda_, s=s, weights=weights, maxit=maxit, tol=tol)
  if isinstance(nodes, vector3d) and isinstance(values, vector3d):
    from .s2vectorfield import interpolateField
    return interpolateField(nodes, values, bandwidth=bandwidth, lambda_=lambda_, s=s, weights=weights, maxit=maxit, tol=tol)
  y = np.asarray(values).ravel()
  real = y.dtype.kind != "c"
  # MATLAB's tolerance of LSQR: 1e-3 on SO(3), 1e-6 on the sphere
  tol = (1e-3 if isinstance(nodes, quaternion) else 1e-6) if tol is None else tol
  if isinstance(nodes, quaternion) and np.size(values) != nodes.size:
    # values with component axes after the node axis: one fit per component, a list of functions
    vals = np.asarray(values)
    cols = vals.reshape(nodes.size, -1)
    fits = [interpolate(nodes, cols[:, k], bandwidth=bandwidth, lambda_=lambda_, s=s, weights=weights, maxit=maxit, tol=tol)
            for k in range(cols.shape[1])]
    L = max(f.bandwidth for f in fits)
    fhat = np.stack([f.truncate(L)._fhat for f in fits]).reshape(vals.shape[1:] + (fits[0].truncate(L)._fhat.shape[-1],))
    return fits[0]._rebuild(fhat, _isReal=all(f.isReal for f in fits))
  if isinstance(nodes, quaternion):
    o = nodes if isinstance(nodes, orientation) else orientation(nodes)
    # nodes closer than 0.02 are one node carrying the mean of their values, as MATLAB's
    o, iu = o.flatten().unique(tolerance=0.02, return_inverse=True)
    count = np.bincount(iu, minlength=o.size)
    y = (np.bincount(iu, y.real, o.size) + 1j * np.bincount(iu, y.imag, o.size) if not real else np.bincount(iu, y, o.size)) / count
    if weights is not None and not isinstance(weights, str):
      weights = np.bincount(iu, np.asarray(weights, dtype=float).ravel(), o.size)
    lam = 1e-8 if lambda_ is None else lambda_
    q = o.data.reshape(-1, 4)
    nA, nB = o.CS.sym.properGroup().rotations().size, o.SS.sym.properGroup().rotations().size
    nSym = nA * nB * (2 if real else 1)
    L = min(dim2deg(max(1, round(len(q) * nSym / 2))), getMTEXpref("maxSO3Bandwidth")) if bandwidth is None else int(bandwidth)
    # the nodes weighted by their Voronoi volumes below 10000 of them, else equally, as MATLAB's
    if isinstance(weights, str) and weights == "equal" or weights is None and len(q) >= 1e4:
      w = np.full(len(q), 1 / len(q))
    elif weights is None or isinstance(weights, str):
      from .rbfinterpolation import voronoiVolumes
      w = voronoiVolumes(o)
    else:
      w = np.asarray(weights, dtype=float).ravel() / np.sum(weights)
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
    x, flag, relres, itn = _lsqrFull(A, At, b, deg2dim(L + 1), maxit, tol)
    f = SO3FunHarmonic(sym(x), o.CS, o.SS, real=real, antipodal=o.antipodal, _internal=True)
    if withParameters:
      # MATLAB's lsqr parameters: the flag, 1 when the iterations ran out, the relative
      # residual and the number of iterations
      return f, (flag, relres, itn)
    return f
  v = (nodes if isinstance(nodes, vector3d) else vector3d(nodes)).flatten().normalize()
  # values with component axes after the node axis: one fit per component, a list of functions
  values = np.asarray(values)
  if values.size != v.size:
    rest = values.shape[1:]
    cols = values.reshape(v.size, -1)
    fits = [interpolate(v, cols[:, k], bandwidth=bandwidth, lambda_=lambda_, s=s, weights=weights, maxit=maxit, tol=tol)
            for k in range(cols.shape[1])]
    fhat = np.stack([f._fhat for f in fits]).reshape(rest + fits[0]._fhat.shape)
    return S2FunHarmonic(fhat, v.frame, real=all(f.isReal for f in fits), antipodal=v.antipodal, _internal=True)
  lam = 0.0 if lambda_ is None else lambda_
  L = math.ceil(math.sqrt(v.size / 2)) if bandwidth is None else int(bandwidth)
  # the nodes weighted by the areas of their Voronoi cells, so that dense regions do not
  # dominate the fit, or equally
  if isinstance(weights, str) and weights == "equal":
    w = np.full(v.size, 1 / v.size)
  elif weights is None:
    from .polefigure import _voronoiWeights
    w = _voronoiWeights(v)
    w = np.full(v.size, 1 / v.size) if w is None else np.asarray(w, dtype=float).ravel() / np.sum(w)
  else:
    w = np.asarray(weights, dtype=float).ravel() / np.sum(weights)
  sw = np.sqrt(w)
  p = S2FourierPlan(v, L)
  reg = math.sqrt(lam) * np.sqrt(_sobolevWeightsS2(L, s))
  A = lambda x: np.concatenate([sw * p(np.asarray(x, dtype=complex)), reg * x])
  At = lambda z: p.adjoint(sw * z[:v.size], np.ones(v.size)) + reg * z[v.size:]
  b = np.concatenate([sw * y, np.zeros(len(reg))]).astype(complex)
  fhat = _lsqr(A, At, b, (L + 1) ** 2, maxit, tol)
  return S2FunHarmonic(fhat, v.frame, real=real, antipodal=v.antipodal, _internal=True)


# interp(v, y, 'linear'): a function through values at directions, by the method named,
# MATLAB's vector3d/interp; with directions vi after the values, its values there
def interpS2(v, y, *args, **kwargs):
  from .handles import S2FunHandle
  from .s2funtri import S2FunTri
  vi = next((a for a in args if isinstance(a, vector3d)), None)
  flags = {a for a in args if isinstance(a, str)}
  symmetric = getattr(v.frame, "sym", None) is not None and v.frame.sym.rotations().size > 1
  if symmetric and not flags & {"linear", "nearest", "inverseDistance", "noSymmetry"}:
    flags.add("harmonic")
  if vi is not None and "linear" not in flags and not flags & {"harmonic", "nearest", "inverseDistance"}:
    flags.add("spline")
  if isinstance(y, vector3d):
    if "harmonic" not in flags:
      raise NotImplementedError("a vector field through vectors is interpolated with 'harmonic'")
    sF = interpolate(v, y, **kwargs)
  else:
    y = np.asarray(y).reshape(v.size, -1).squeeze(-1) if np.asarray(y).size == v.size else np.asarray(y).reshape(v.size, -1)
    nodes = v.flatten()
    if "nearest" in flags:
      res = nodes.resolution

      def nearest(w):
        ind, d = nodes.find(w)
        out = y[np.asarray(ind).ravel()].astype(float)
        out[np.asarray(d).ravel() > 2 * res] = np.nan
        return out.reshape(w.shape)
      sF = S2FunHandle(nearest, v.frame)
    elif "harmonic" in flags:
      sF = interpolate(v, y, **kwargs)
    elif flags & {"spline", "inverseDistance"}:
      from .kernelfunctions import S2DeLaValleePoussinKernel
      psi = S2DeLaValleePoussinKernel(halfwidth=nodes.resolution / 2)
      V = nodes.normalize().data.reshape(-1, 3)

      # the mean of the four nearest values, weighted by the kernel or the inverse distance
      def spline(w):
        W = w.normalize().data.reshape(-1, 3)
        omega = np.arccos(np.clip(W @ V.T, -1, 1))
        j = np.argsort(omega, axis=1)[:, :4]
        so = np.take_along_axis(omega, j, axis=1)
        M = np.minimum(1 / so, 1e10) if "inverseDistance" in flags else np.real(psi.eval(np.cos(so)))
        M = M / M.sum(axis=1, keepdims=True)
        return np.sum(M * y[j], axis=1).reshape(w.shape)
      sF = S2FunHandle(spline, v.frame)
    else:
      sF = S2FunTri(v, y)
  return sF.eval(vi) if vi is not None else sF


__all__ = ["calcDensity", "calcPDF", "interpolate"]
