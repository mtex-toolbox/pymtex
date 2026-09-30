# the grain boundary normal distribution from the traces of a planar section (MATLAB
# @grainBoundary/calcGBND and calcGBNDKernel): every segment is a trace direction l in the
# section with normal N and the orientations on its two sides; the density of the rotations
# taking N to x and l to z composed with the orientations, convolved with a kernel that
# integrates over the unknown inclination of the plane about the trace, is the density of
# the boundary normals in the crystal frame; conditioned on one misorientation it is the
# grain boundary character distribution
from __future__ import annotations

import functools

import numpy as np

from ..functions.density import calcDensity
from ..functions.kernelfunctions import (S2DeLaValleePoussinKernel, S2Kernel, S2KernelHandle, SO3DeLaValleePoussinKernel,
                                         SO3DirichletKernel)
from ..functions.s2fun import S2FunHarmonic
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, xvector, zvector
from ..maps.boundary import grainBoundary
from ..maps.ebsd import EBSD
from ..geometry.install import install

# the bandwidth MATLAB expands the inclination kernel to
BANDWIDTH = 62


def calcGBNDKernel(halfwidth=10 * degree, nonneg=False, bump=False):
  """calcGBNDKernel(halfwidth=10*degree, nonneg=True): the kernel Psi of the inclination, a
  function of the cosine to the trace, normalised so that a uniform distribution of planes
  gives one, and the spherical function psi(v) = Psi(v.z) sin(angle(v, x)) it is applied as;
  without nonneg or bump Psi is the solution of the integral equation that makes the density
  of the normals the de la Vallee Poussin kernel of that halfwidth, or the S2Kernel given
  in its place"""
  from scipy.integrate import quad
  from scipy.special import ellipe
  if nonneg:
    s = np.sin(halfwidth)
    Psi = S2KernelHandle(lambda t: 2 / np.sqrt(2 * np.pi) / s * np.exp(-0.5 * (np.asarray(t) / s) ** 2) / 0.5437)
  elif bump:
    Psi = S2KernelHandle(lambda t: (np.abs(np.asarray(t)) < np.arcsin(halfwidth)).astype(float))
  else:
    if isinstance(halfwidth, S2Kernel):
      Psi = S2Kernel(_inclinationKernel(halfwidth))
    else:
      Psi = S2Kernel(_defaultInclination(float(halfwidth)))
  # the normalisation against the complete elliptic integral of the second kind, MATLAB's ellip2ke
  c = (np.pi / 2) / quad(lambda t: float(Psi.eval(t)) * ellipe(1 - t * t), 0, 1, limit=200)[0]
  Psi = S2KernelHandle(lambda t, fun=Psi.fun: c * fun(t)) if isinstance(Psi, S2KernelHandle) else S2Kernel(c * Psi.A)
  psi = lambda v: Psi.eval(v.data[..., 2] / np.linalg.norm(v.data, axis=-1)) * np.sin(v.angle(xvector, symmetry=False))
  return Psi, S2FunHarmonic.quadrature(psi, bandwidth=BANDWIDTH, antipodal=True)


# the inclination kernel of the de la Vallee Poussin kernel of a halfwidth, a solve of five
# seconds, computed once
@functools.lru_cache(maxsize=8)
def _defaultInclination(halfwidth):
  return _inclinationKernel(S2DeLaValleePoussinKernel(halfwidth=halfwidth))


# the Legendre coefficients up to degree 127 of the inclination kernel f whose density of
# normals is the kernel K: g(t) = 4 int_0^{pi/2} k(theta, t) f(t sin(theta)) dtheta with
# g(t) = K(sqrt(1 - t^2)), discretised by Nystrom at 1000 Chebyshev points and 120
# Gauss-Legendre nodes in theta, f the interpolant through the solution extended evenly
def _inclinationKernel(K, N=1000, nTheta=120, degree=128):
  from scipy.special import ellipe, ellipk
  t = (1 - np.cos(np.pi * np.arange(N) / (N - 1))) / 2
  w = (-1.0) ** np.arange(N)
  w[0] /= 2
  w[-1] /= 2
  x, wx = np.polynomial.legendre.leggauss(nTheta)
  theta, wth = (x + 1) * np.pi / 4, wx * np.pi / 4

  # the barycentric interpolation rows at the points y
  def rows(y):
    d = y[:, None] - t[None, :]
    hit = np.abs(d) < 1e-14
    d[hit] = 1
    r = w / d
    r /= r.sum(axis=1, keepdims=True)
    exact = hit.any(axis=1)
    r[exact] = hit[exact]
    return r

  # the kernel k(theta, t) = I01(a, b) through the elliptic integrals of parameter m = b / a,
  # scipy's taking m < 0 as it is
  def k(th, tv):
    a = 1 - tv ** 2 + tv ** 2 * np.sin(th) ** 2
    b = 1 - 2 * tv ** 2 + tv ** 2 * np.sin(th) ** 2
    m = b / a
    small = np.abs(m) < 1e-12
    ms = np.where(small, 1.0, m)
    with np.errstate(invalid="ignore"):
      v = np.sqrt(a) * ((ms + 1) * ellipe(ms) + (ms - 1) * ellipk(ms)) / (3 * ms)
    v = np.where(small, np.sqrt(a) * np.pi / 4, v)
    return np.where(m == 1, 2 / 3 * np.sqrt(a), v)

  A = np.empty((N, N))
  for i, ti in enumerate(t):
    A[i] = (4 * k(theta, ti) * wth) @ rows(ti * np.sin(theta))
  f = np.linalg.solve(A, K.eval(np.sqrt(1 - t ** 2))) / 0.32
  # the coefficients of the even extension, (2n + 1) int_0^1 f P_n, exact for the interpolant
  xq, wq = np.polynomial.legendre.leggauss(N)
  xq, wq = (xq + 1) / 2, wq / 2
  fq = rows(xq) @ f
  P = np.polynomial.legendre.legvander(xq, degree - 1)
  c = (2 * np.arange(degree) + 1) * ((wq * fq) @ P)
  c[1::2] = 0
  return c


# the orientations on the two sides of every segment: the measurements beside it or the
# mean orientations of the grains
def _sides(gB, source):
  if isinstance(source, EBSD):
    ids = np.asarray(source.id).reshape(-1)
    order = np.argsort(ids)
    q = source.rotations.data.reshape(-1, 4)
    pix = lambda col: order[np.searchsorted(ids[order], gB.ebsdId[:, col])]
    q1, q2 = q[pix(0)], q[pix(1)]
  elif hasattr(source, "meanRotation"):
    q = source.meanRotation.data.reshape(-1, 4)
    q1, q2 = q[source.id2ind(gB.grainId[:, 0])], q[source.id2ind(gB.grainId[:, 1])]
  else:
    raise TypeError("the second argument is an EBSD map or a grain list")
  return q1, q2


# the proper rotations of a frame as quaternions
def _proper(fr):
  d = fr.properGroup().rot.data
  return d.reshape(-1, d.shape[-1])[:, :4]


def _qmul(a, b):
  a0, a1, a2, a3 = np.moveaxis(a, -1, 0)
  b0, b1, b2, b3 = np.moveaxis(b, -1, 0)
  return np.stack([a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3, a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
                   a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1, a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0], axis=-1)


def _qinv(a):
  return a * np.array([1, -1, -1, -1])


# the angle between rotations given as quaternions, without symmetry
def _qangle(a, b):
  return 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))


def calcGBND(gB, source, moriRef=None, halfwidth=10 * degree, nonneg=False, bump=False, oneSide=False):
  """calcGBND(gB, ebsd, halfwidth=5*degree, nonneg=True): the density of the boundary normals in
  the crystal frame of the one phase of gB; calcGBND(gB, grains, moriRef, ...) the density
  of the normals of the boundaries near that misorientation, the grain boundary character
  distribution, taking both sides of a relation that is its own inverse unless oneSide"""
  both = False
  dSym = None
  if moriRef is not None:
    p1, p2 = gB.CSList.phaseId(moriRef.CS), gB.CSList.phaseId(moriRef.SS)
    gB = gB[((gB.phaseId[:, 0] == p1) & (gB.phaseId[:, 1] == p2)) | ((gB.phaseId[:, 0] == p2) & (gB.phaseId[:, 1] == p1))]
    if p1 != p2:
      gB = gB.flip(gB.phaseId[:, 0] != p1)
    elif moriRef.angle(moriRef.inv()) > 5 * degree:
      omega = np.stack([gB.misorientation.angle(moriRef), gB.misorientation.angle(moriRef.inv())], axis=-1)
      gB = gB.flip(omega[:, 1] < omega[:, 0])
    else:
      both = not oneSide
    cs = moriRef.CS.stripSym()
  else:
    if len(np.unique(gB.phaseId, axis=0)) != 1:
      raise ValueError("the boundary should hold one pair of phases")
    cs = gB.CSList[int(gB.phaseId[0, 0])]
  q1, q2 = _sides(gB, source)
  weights = gB.segLength
  plane = gB.plane
  N = np.cross(plane.e1, plane.e2)
  l = gB.direction.normalize().data.reshape(-1, 3)
  if moriRef is not None:
    ref = moriRef.data.reshape(-1)[:4]
    s1, s2 = _proper(moriRef.CS), _proper(moriRef.SS)
    mori = _qmul(_qinv(q1), q2)
    # the equivalents s1 moriRef s2^-1 of the reference, and the one nearest each misorientation
    eq = _qmul(_qmul(s1[:, None], ref), _qinv(s2)[None])
    d = np.abs(mori @ eq.reshape(-1, 4).T)
    if both:
      keep = 2 * np.arccos(np.clip(d.max(axis=1), 0, 1)) < 2 * halfwidth
      i1, i2 = np.unravel_index(np.argmax(d[keep], axis=1), eq.shape[:2])
      q = np.concatenate([_qmul(q1[keep], s1[i1]), _qmul(q2[keep], s2[i2])])
      l = np.concatenate([l[keep], l[keep]])
      weights = np.concatenate([weights[keep], weights[keep]])
      # the elements that fix the reference, which the result keeps: MATLAB's minimum over
      # the first side of the angles of s1 moriRef s2 to moriRef, taken as elements of s1
      fix = _qangle(_qmul(_qmul(s1[:, None], ref), s2[None]), ref)
      dSym = s1[np.min(fix, axis=0) < 1e-5]
    else:
      # the reference with the crystal side turned, its first side put onto ori1 alone
      e1 = _qmul(ref[None], s1)
      dd = np.abs(_qmul(_qinv(q2), q1) @ e1.T)
      idCS = np.argmax(dd, axis=1)
      best = dd[np.arange(len(dd)), idCS]
      keep = best > np.cos(2 * halfwidth)
      q = _qmul(q1[keep], _qinv(s1[idCS[keep]]))
      psi = SO3DeLaValleePoussinKernel(halfwidth=halfwidth)
      weights = weights[keep] * psi.eval(best[keep])
      l = l[keep]
  else:
    q = np.concatenate([q1, q2])
    l = np.concatenate([l, l])
    weights = np.concatenate([weights, weights])
  # the rotation taking the section normal to x and the trace to z, composed with the orientation
  rot = rotation.map(vector3d(np.broadcast_to(N, l.shape).copy()), xvector, vector3d(l), zvector)
  g = _qmul(rot.data.reshape(-1, 4), q)
  _, psiS2 = calcGBNDKernel(halfwidth, nonneg=nonneg, bump=bump)
  ori = orientation._fromRotation(g, (cs.stripSym(), _specimen()))
  f = calcDensity(ori, weights=weights, kernel=SO3DirichletKernel(psiS2.bandwidth), harmonic=True)
  gbnd = f.conv(psiS2)
  if dSym is not None and len(dSym) > 1:
    from ..functions.funops import rotate
    gbnd = sum((rotate(gbnd, rotation._fromData(s)) for s in dSym[1:]), gbnd) / len(dSym)
  # the estimate carries the symmetry of the phase, the character distribution none; a
  # plane normal is an axis
  return S2FunHarmonic(gbnd._fhat, cs, real=True, antipodal=True, _internal=True)


# the specimen frame the rotated orientations live in, the one the kernel is written in
def _specimen():
  from ..geometry.frames import specimenFrame
  return specimenFrame()


install(grainBoundary, calcGBND)

__all__ = ["calcGBND", "calcGBNDKernel"]
