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

from ..geometry.kernels import bestDot, qconj, qmul
from ..functions.density import calcDensity
from ..functions.kernelfunctions import (S2DeLaValleePoussinKernel, S2Kernel, S2KernelHandle, SO3DeLaValleePoussinKernel,
                                         SO3DirichletKernel)
from ..functions.s2fun import S2FunHarmonic
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from ..maps.boundary import grainBoundary
from ..maps.ebsd import EBSD
from ..maps.grains import grain2d
from ..geometry.install import install

# the bandwidth MATLAB expands the inclination kernel to
BANDWIDTH = 62


def calcGBNDKernel(halfwidth=10 * degree, nonneg=False, bump=False):
  """the kernel of the inclination of a boundary plane about its trace, the one calcGBND
  convolves with

    Psi, psi = calcGBNDKernel()                        # the kernel of halfwidth 10 degree
    Psi, psi = calcGBNDKernel(5*degree, nonneg=True)   # a nonnegative kernel of another halfwidth
    Psi, psi = calcGBNDKernel(S2DeLaValleePoussinKernel(halfwidth=10*degree))  # the kernel for a given density kernel

  A trace in a section leaves the inclination of the plane open. Psi is a function of the
  cosine t between a normal and the trace, and psi(v) = Psi(v.z) sin(angle(v, x)) is the
  spherical function it is applied as, for a trace along z in a section with normal x. Both
  are normalised so that a uniform distribution of planes gives one.

  By default Psi solves the integral equation that makes the density of the normals the de
  la Vallee Poussin kernel of the halfwidth, or the S2Kernel given in its place; that
  solution may be negative. nonneg takes a Gaussian in t instead, bump the indicator of
  |t| < arcsin(halfwidth), MATLAB's bound.

  Output
    Psi - S2Kernel, or S2KernelHandle with nonneg or bump, the kernel of the inclination
    psi - S2FunHarmonic, the spherical function of a trace along z, antipodal
  Keywords
    halfwidth - the halfwidth of the kernel, or an S2Kernel, the density of the normals the
        kernel of the inclination is to give
    nonneg - a nonnegative kernel, a Gaussian in the cosine to the trace
    bump - a bump kernel, one within the halfwidth and zero outside
  See also
    calcGBND, S2DeLaValleePoussinKernel"""
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
  def psi(v):
    return Psi.eval(v.data[..., 2] / np.linalg.norm(v.data, axis=-1)) * np.sin(v.angle(vector3d.X, symmetry=False))
  if not bump:
    return Psi, S2FunHarmonic.quadrature(psi, bandwidth=BANDWIDTH, antipodal=True)
  # the jump of the bump aliases on a quadrature grid of the bandwidth: the expansion is taken
  # on a finer one and cut, and the mean the remaining aliasing moves is set to one again
  sF = S2FunHarmonic.quadrature(psi, bandwidth=4 * BANDWIDTH, antipodal=True).truncate(BANDWIDTH)
  return Psi, sF / np.real(sF.mean())


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
    def pix(col):
      return order[np.searchsorted(ids[order], gB.ebsdId[:, col])]
    q1, q2 = q[pix(0)], q[pix(1)]
  elif isinstance(source, grain2d):
    q = source.meanRotation.data.reshape(-1, 4)
    q1, q2 = q[source.id2ind(gB.grainId[:, 0])], q[source.id2ind(gB.grainId[:, 1])]
  else:
    raise TypeError("the second argument is an EBSD map or a grain list")
  return q1, q2


# the proper rotations of a frame as quaternions
def _proper(fr):
  d = fr.properGroup().rot.data
  return d.reshape(-1, d.shape[-1])[:, :4]


# the angle between rotations given as quaternions, without symmetry
def _qangle(a, b):
  return 2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1))


def calcGBND(gB, source, moriRef=None, halfwidth=10 * degree, nonneg=False, bump=False, oneSide=False):
  """the grain boundary normal distribution in the crystal frame, estimated from the traces of
  the boundaries in a planar section; with a misorientation the grain boundary character
  distribution

    gbnd = calcGBND(gB, ebsd)                          # the normals from the orientations of the pixels
    gbnd = calcGBND(gB, grains)                        # the normals from the mean orientations of the grains
    gbnd = calcGBND(gB, ebsd, halfwidth=5*degree, nonneg=True)  # a smaller halfwidth, a nonnegative kernel
    gbcd = calcGBND(gB, grains, moriRef)               # the character distribution of one misorientation
    gbcd = calcGBND(gB, grains, moriRef, oneSide=True) # only the first side of a symmetric relation

  A section shows only the trace of every boundary; the boundary plane may be inclined about
  it at any angle. The orientations on both sides turn each trace into the crystal frame, and
  a kernel over the unknown inclination spreads it to the compatible normals. Summed over all
  boundaries, weighted by their lengths, this gives an antipodal density of the normals,
  normalised so that a uniform distribution of planes is one.

  The boundaries must hold one pair of phases, gB['Magnesium', 'Magnesium'] for instance.
  With a map, the orientations of the two pixels of every segment are used, which requires
  boundaries that were not resampled: smooth them with smoothBoundary(grains, simplify=False,
  refine=False), or pass the grains to use their mean orientations.

  With a misorientation moriRef, only the boundaries near it count, weighted by their distance
  to it, and the normals are given in the crystal frame of its first side: the grain boundary
  character distribution. A relation that is its own inverse, a twin, takes both sides of
  every boundary unless oneSide=True.

  Input
    gB - grainBoundary, the boundaries of one pair of phases
    source - EBSD, the map the boundaries were computed from, or grain2d, the grains
    moriRef - orientation, the misorientation of the character distribution
  Output
    gbnd - S2FunHarmonic, antipodal, in the crystal frame of the phase, or of the first side
        of moriRef
  Keywords
    halfwidth - the halfwidth of the density estimation, of the kernel in the normals, and,
        with moriRef, how far a misorientation may lie from it
    nonneg - a nonnegative kernel of the inclination, a Gaussian in the cosine to the trace,
        instead of the solution of the integral equation, which may ring and be negative
    bump - a bump kernel of the inclination, constant within the halfwidth
    oneSide - with a moriRef that is its own inverse, take only the first side of every
        boundary
  References
    D.M. Saylor, G.S. Rohrer, Determining crystal habits from observations of planar
        sections, J. Am. Ceram. Soc. 85 (2002) 2799-2804
    R. Hielscher, R. Kilian, E. Wünsche, Efficient computation of the grain boundary normal
        distribution from two dimensional EBSD data, not yet published
  See also
    calcGBNDKernel, grain3Boundary.calcGBND, smoothBoundary, BoundaryNormalDistribution"""
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
    if both:
      # the equivalents s1 moriRef s2^-1 of the reference, and the one nearest each misorientation
      eq = qmul(qmul(s1[:, None], ref), qconj(s2)[None])
      best, k = bestDot(qmul(qconj(q1), q2), eq.reshape(-1, 4))
      keep = 2 * np.arccos(np.clip(best, 0, 1)) < 2 * halfwidth
      i1, i2 = np.unravel_index(k[keep], eq.shape[:2])
      q = np.concatenate([qmul(q1[keep], s1[i1]), qmul(q2[keep], s2[i2])])
      l = np.concatenate([l[keep], l[keep]])
      weights = np.concatenate([weights[keep], weights[keep]])
      # the elements that fix the reference, which the result keeps: MATLAB's minimum over
      # the first side of the angles of s1 moriRef s2 to moriRef, taken as elements of s1
      fix = _qangle(qmul(qmul(s1[:, None], ref), s2[None]), ref)
      dSym = s1[np.min(fix, axis=0) < 1e-5]
    else:
      # the reference with the crystal side turned, its first side put onto ori1 alone
      best, idCS = bestDot(qmul(qconj(q2), q1), qmul(ref[None], s1))
      keep = best > np.cos(2 * halfwidth)
      q = qmul(q1[keep], qconj(s1[idCS[keep]]))
      psi = SO3DeLaValleePoussinKernel(halfwidth=halfwidth)
      weights = weights[keep] * psi.eval(best[keep])
      l = l[keep]
  else:
    q = np.concatenate([q1, q2])
    l = np.concatenate([l, l])
    weights = np.concatenate([weights, weights])
  # the rotation taking the section normal to x and the trace to z, composed with the orientation
  rot = rotation.map(vector3d(np.broadcast_to(N, l.shape).copy()), vector3d.X, vector3d(l), vector3d.Z)
  g = qmul(rot.data.reshape(-1, 4), q)
  _, psiS2 = calcGBNDKernel(halfwidth, nonneg=nonneg, bump=bump)
  ori = orientation._fromRotation(g, (cs.stripSym(), _specimen()))
  f = calcDensity(ori, weights=weights, kernel=SO3DirichletKernel(psiS2.bandwidth), harmonic=True)
  gbnd = f.conv(psiS2)
  if dSym is not None and len(dSym) > 1:
    gbnd = sum((gbnd.rotate(rotation._fromData(s)) for s in dSym[1:]), gbnd) / len(dSym)
  # the estimate carries the symmetry of the phase, the character distribution none; a
  # plane normal is an axis
  return S2FunHarmonic(gbnd._fhat, cs, real=True, antipodal=True, _internal=True)


# the specimen frame the rotated orientations live in, the one the kernel is written in
def _specimen():
  from ..geometry.frames import specimenFrame
  return specimenFrame()


install(grainBoundary, calcGBND)

__all__ = ["calcGBND", "calcGBNDKernel"]
