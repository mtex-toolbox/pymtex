# LSQR of Paige and Saunders with MATLAB's stopping rules, the least squares solver of the
# harmonic and the radial fits
#
#   x, flag, relres, iterations = lsqr(A, At, b, n, tol=1e-6, maxit=20)
#
# A and At apply the matrix and its adjoint. The iteration stops when the residual has
# fallen to tol times the right hand side, or the residual of the normal equations to tol
# times the norm of the matrix and of the residual, with flag 0; flag 1 when the
# iterations ran out first. relres is the residual relative to the right hand side.
from __future__ import annotations

import math

import numpy as np


def lsqr(A, At, b, n, tol=1e-6, maxit=20):
  b = np.asarray(b)
  normb = float(np.linalg.norm(b))
  # the bidiagonalisation of Golub and Kahan, started at b
  beta = normb
  u = b / beta if normb > 0 else b
  v = np.asarray(At(u))
  x = np.zeros(n, dtype=np.result_type(b.dtype, v.dtype, float))
  if normb == 0:
    return x, 0, 0.0, 0
  alpha = float(np.linalg.norm(v))
  if alpha == 0:
    return x, 0, 1.0, 0
  v = v / alpha
  w = v.copy()
  phibar, rhobar = beta, alpha
  normA = 0.0
  normr = beta
  for k in range(1, int(maxit) + 1):
    u = np.asarray(A(v)) - alpha * u
    beta = float(np.linalg.norm(u))
    if beta > 0:
      u = u / beta
    normA = math.sqrt(normA ** 2 + alpha ** 2 + beta ** 2)
    v = np.asarray(At(u)) - beta * v
    alphaNext = float(np.linalg.norm(v))
    if alphaNext > 0:
      v = v / alphaNext
    # the plane rotation that eliminates the subdiagonal
    rho = math.hypot(rhobar, beta)
    c, s = rhobar / rho, beta / rho
    theta = s * alphaNext
    rhobar = -c * alphaNext
    phi = c * phibar
    phibar = s * phibar
    x = x + (phi / rho) * w
    w = v - (theta / rho) * w
    alpha = alphaNext
    # the residual and the residual of the normal equations
    normr = abs(phibar)
    normAr = abs(phibar * alpha * c)
    if normr <= tol * normb or (normr > 0 and normAr / (normA * normr) <= tol) or normr == 0:
      return x, 0, normr / normb, k
  return x, 1, normr / normb, int(maxit)


__all__ = ["lsqr"]
