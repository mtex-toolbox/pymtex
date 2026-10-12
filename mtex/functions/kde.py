# kernel density estimation of real valued data (MATLAB calcDensity on doubles, Gaussian;
# the extern kde1d and kdeN of Botev, Grotowski and Kroese)
#
#   f = Gaussian(m, delta);  f(x) = Gaussian(m, delta, x)   exp(-(x-m)^2/delta^2) / (delta sqrt(pi))
#   f = calcDensity(x, range=[lo, hi])           one dimension, the diffusion estimator
#   f = calcDensity(np.column_stack([x, y]), range=[[x0, y0], [x1, y1]])   d dimensions, the EM mixture
#   f(x), f(xGrid, yGrid), f.bandwidth           the density as an interpolant on its grid
from __future__ import annotations

import math

import numpy as np

from ..geometry.kernels import columnMax, columnMin
from ..geometry.display import aligned


def Gaussian(m, delta, x=None):
  """the normal density of mean m and width delta, the standard deviation times sqrt(2)"""
  def f(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-(x - m) ** 2 / delta ** 2) / delta / math.sqrt(math.pi)
  return f if x is None else f(x)


class griddedDensity:
  """a density on a regular grid, evaluated by its tensor product spline as MATLAB's
  griddedInterpolant(..., 'spline'), extrapolating beyond the grid"""

  def __init__(self, grid, values, bandwidth=None):
    from scipy.interpolate import CubicSpline, RegularGridInterpolator
    self.grid, self.values, self.bandwidth = grid, values, bandwidth
    if len(grid) == 1:
      self._f = CubicSpline(grid[0], values)
    else:
      self._f = RegularGridInterpolator(grid, values, method="cubic", bounds_error=False, fill_value=None)

  def __call__(self, *x):
    if len(x) != len(self.grid):
      raise ValueError(f"the density is {len(self.grid)} dimensional, it takes {len(self.grid)} coordinates")
    if len(x) == 1:
      return self._f(np.asarray(x[0], dtype=float))
    x = np.broadcast_arrays(*[np.asarray(c, dtype=float) for c in x])
    return self._f(np.stack([c.ravel() for c in x], axis=-1)).reshape(x[0].shape)

  def __repr__(self):
    dims = " × ".join(str(len(g)) for g in self.grid)
    bw = "" if self.bandwidth is None else f", bandwidth {np.array2string(np.asarray(self.bandwidth), precision=4)}"
    return aligned(f"griddedDensity ({len(self.grid)}D, {dims} grid{bw})")


def calcDensityReal(x, range=None, bandwidth=None, weights=None, rng=None):
  """the density of samples x, one per row, on the rectangle range (the minima in the first row,
  the maxima in the second); one dimension by the diffusion estimator on 2^14 points, whose
  bandwidth may be fixed, more by the regularised EM mixture on round(10^(6/d)) points per axis"""
  x = np.asarray(x, dtype=float)
  if x.ndim == 1 or x.shape[1] == 1:
    x = x.ravel()
    lo, hi = (np.min(x), np.max(x)) if range is None else np.asarray(range, dtype=float).ravel()
    bw, density, mesh = kde1d(x, 2 ** 14, lo, hi, weights=weights, bandwidth=bandwidth)
    return griddedDensity([mesh], density, bw)
  d = x.shape[1]
  r = np.vstack([columnMin(x), columnMax(x)]) if range is None else np.asarray(range, dtype=float).reshape(2, d)
  N = round(1000000 ** (1 / d))
  axes = [np.linspace(lo, hi, N) for lo, hi in zip(r[0], r[1])]
  mesh = np.meshgrid(*axes, indexing="ij")
  density, bw = kdeN(x, np.stack([m.ravel() for m in mesh], axis=-1), rng=rng)
  return griddedDensity(axes, density.reshape(mesh[0].shape), bw)


# ------------------------------------------------------------------ one dimension


def _dct1d(data):
  n = len(data)
  w = np.concatenate([[1], 2 * np.exp(-1j * np.arange(1, n) * np.pi / (2 * n))])
  return np.real(w * np.fft.fft(np.concatenate([data[0::2], data[-1:0:-2]])))


def _idct1d(data):
  n = len(data)
  w = n * np.exp(1j * np.arange(n) * np.pi / (2 * n))
  d = np.real(np.fft.ifft(w * data))
  out = np.empty(n)
  out[0::2] = d[:n // 2]
  out[1::2] = d[n - 1:n // 2 - 1:-1]
  return out


# t minus the plug-in estimate of t, the fixed point of Botev et al., over the coefficients a2
# of the degrees 1, 2, ...
def _fixedPoint(t, N, I, a2):
  l = 7
  f = 2 * np.pi ** (2 * l) * np.sum(I ** l * a2 * np.exp(-I * np.pi ** 2 * t))
  for s in range(l - 1, 1, -1):
    K0 = np.prod(np.arange(1, 2 * s, 2)) / math.sqrt(2 * np.pi)
    const = (1 + 0.5 ** (s + 0.5)) / 3
    time = (2 * const * K0 / N / f) ** (2 / (3 + 2 * s))
    f = 2 * np.pi ** (2 * s) * np.sum(I ** s * a2 * np.exp(-I * np.pi ** 2 * time))
  return t - (2 * N * math.sqrt(np.pi) * f) ** (-2 / 5)


# the root of the fixed point equation in [0, tol], the interval doubled until it holds one
def _root(f, N):
  from scipy.optimize import brentq, fminbound
  N = min(max(N, 50), 1050)
  tol = 1e-12 + 0.01 * (N - 50) / 1000
  f0 = f(0)
  while tol < 0.1:
    if f0 * f(tol) <= 0:
      return brentq(f, 0, tol)
    tol = min(2 * tol, 0.1)
  return fminbound(lambda t: abs(f(t)), 0, 0.1)


def kde1d(data, n=2 ** 14, lo=None, hi=None, weights=None, bandwidth=None):
  """the diffusion estimator: the histogram on n cells of [lo, hi] smoothed by the heat kernel
  whose time t is the fixed point, or fixed by the bandwidth sqrt(t) (hi - lo); the bandwidth,
  the density and the cell centres"""
  data = np.asarray(data, dtype=float).ravel()
  n = 2 ** math.ceil(math.log2(n))
  if lo is None or hi is None:
    R = np.max(data) - np.min(data)
    lo, hi = np.min(data) - R / 2, np.max(data) + R / 2
  R = hi - lo
  edges = lo + np.arange(n + 1) * (R / n)
  mesh = (edges[:-1] + edges[1:]) / 2
  N = len(np.unique(data))
  counts = np.histogram(data, edges, weights=weights)[0].astype(float)
  initial = counts / np.sum(counts)
  a = _dct1d(initial)
  if bandwidth is not None:
    t = (bandwidth / R) ** 2
  else:
    I = np.arange(1, n, dtype=float) ** 2
    a2 = (a[1:] / 2) ** 2
    t = _root(lambda t: _fixedPoint(t, N, I, a2), N)
  density = _idct1d(a * np.exp(-np.arange(n) ** 2 * np.pi ** 2 * t / 2)) / R
  return math.sqrt(t) * R, np.maximum(density, np.finfo(float).eps), mesh


# ------------------------------------------------------------------ several dimensions


# the log of the Gaussian of mean mu and covariance Sig weighted by w at the rows of X, the
# upper Cholesky factor of Sig and the rows whitened by it
def _logGaussian(X, w, mu, Sig):
  L = np.linalg.cholesky(Sig).T
  z = np.linalg.solve(L.T, (X - mu).T).T
  return -0.5 * np.sum(z ** 2, axis=1) - np.sum(np.log(np.diag(L))) + np.log(w) - X.shape[1] * np.log(2 * np.pi) / 2, L, z


def _regEM(w, mu, Sig, dl, X):
  gam, d = mu.shape
  n = X.shape[0]
  logLh = np.zeros((n, gam))
  logSig = np.zeros((n, gam))
  for i in range(gam):
    logG, L, xRinv = _logGaussian(X, w[i], mu[i], Sig[i])
    xSig = np.sum(np.linalg.solve(L, xRinv.T).T ** 2, axis=1) + np.finfo(float).eps
    logLh[:, i] = logG - 0.5 * dl ** 2 * np.sum(np.linalg.inv(L) ** 2)
    logSig[:, i] = logLh[:, i] + np.log(xSig)
  maxll, maxlsig = logLh.max(axis=1), logSig.max(axis=1)
  p = np.exp(logLh - maxll[:, None])
  psig = np.exp(logSig - maxlsig[:, None])
  density, psigd = p.sum(axis=1), psig.sum(axis=1)
  logpdf = np.log(density) + maxll
  logpsigd = np.log(psigd) + maxlsig
  p = p / density[:, None]
  ent = np.sum(logpdf)
  w = p.sum(axis=0)
  for i in np.flatnonzero(w > 0):
    mu[i] = p[:, i] @ X / w[i]
    Xc = np.sqrt(p[:, i])[:, None] * (X - mu[i])
    Sig[i] = Xc.T @ Xc / w[i] + dl ** 2 * np.eye(d)
  w = w / np.sum(w)
  curv = np.mean(np.exp(logpsigd - logpdf))
  dl = 1 / (4 * n * (4 * np.pi) ** (d / 2) * curv) ** (1 / (d + 2))
  return w, mu, Sig, dl, ent


def kdeN(X, grid, gam=None, rng=None):
  """the regularised EM mixture of Botev et al. with ceil(sqrt(n)) Gaussians, fitted on the data
  scaled to the unit box, evaluated at the rows of grid; the density and the bandwidth"""
  rng = np.random.default_rng() if rng is None else rng
  X = np.asarray(X, dtype=float)
  X = X[~np.any(np.isnan(X), axis=1)]
  n, d = X.shape
  lo, hi = columnMin(X), columnMax(X)
  s = hi - lo
  lo, hi = lo - s / 10, hi + s / 10
  s = hi - lo
  X = (X - lo) / s
  mesh = (np.asarray(grid, dtype=float) - lo) / s
  gam = math.ceil(math.sqrt(n)) if gam is None else gam
  dl = 0.1 / n ** (d / (d + 4))
  mu = X[rng.permutation(n)[:gam]].copy()
  w = rng.random(gam)
  w = w / np.sum(w)
  Sig = rng.random((gam, d, d)) * (np.eye(d) * dl)
  ent = -np.inf
  for _ in range(201):
    old = ent
    w, mu, Sig, dl, ent = _regEM(w, mu, Sig, dl, X)
    if abs((ent - old) / ent) < 1e-4:
      break
  pdf = sum(np.exp(_logGaussian(mesh, w[k], mu[k], Sig[k])[0]) for k in range(gam))
  return pdf / np.prod(s), dl * s


__all__ = ["Gaussian", "griddedDensity", "kde1d", "kdeN"]
