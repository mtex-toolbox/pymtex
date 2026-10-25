# the texture characteristics of an ODF (MATLAB @SO3Fun: textureindex, entropy, volume,
# fibreVolume, max, min, calcComponents, calcMDF, calcAngleDistribution, calcAxisDistribution,
# calcMIndex, discreteSample, calcError, calcKernel, centerSpecimen; docs/plans/functions.md F6)
#
#   textureIndex(odf), entropy(odf), volume(odf, center, radius), fibreVolume(odf, h, r, radius)
#   values, modes = max(odf, numLocal=, resolution=, accuracy=);  min(odf)
#   modes, weights, centerId = calcComponents(odf, seed=, weights=, tol=, radius=, exact=)
#   calcMDF(odf1, odf2, bandwidth=), calcAngleDistribution(odf | cs1, cs2 | mori), calcAxisDistribution(odf, v)
#   calcMIndex(odf), discreteSample(odf, n, rng=), discreteSample(fun, n, range=), calcError(odf1, odf2, 'l2'), calcKernel(ori | v | ebsd, 'UCV')
from __future__ import annotations

import math

import numpy as np

from ..geometry._array import MtexArray
from ..geometry.display import framesLabel
from ..geometry.fibre import _uniqueDirections, fibre
from ..geometry.frames import crystalFrame, referenceFrame
from ..geometry.kernels import backend, qconj, qmul
from ..geometry.grids.localgrids import localOrientationGrid, localS2Grid
from ..geometry.grids.fibonacci import fibonacciS2Grid
from ..geometry.grids.s2grid import equispacedS2Grid
from ..geometry.grids.so3grid import equispacedSO3Grid
from ..geometry.orientation import orientation
from ..geometry.quaternion import qrot, quaternion, rotation
from ..geometry.representations import quat2mat
from ..geometry.session import getMTEXpref
from ..geometry.symreduce import reductionGroups, symDots
from ..geometry.tangent import SO3TangentVector
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, xvector, zvector, asVector3d
from .bingham import S2FunBingham, SO3FunBingham, binghamSample
from .kernels import rbfSample
from .funops import conv
from .handles import S1FunHandle, S2FunHandle, S2FunSum, SO3FunComposition, SO3FunGeneric, SO3FunHandle
from .kernelfunctions import SO3DeLaValleePoussinKernel
from .quadrature import bandwidthFor, quadratureS2Grid, quadratureSO3Grid
from .rbf import SO3FunCBF, SO3FunRBF, _rotationsFor, uniformODF
from .s1fun import S1FunHarmonic
from .s2fun import S2FunHarmonic
from .s2funclass import S2Fun
from .s2funtri import S2FunTri
from .so3fun import SO3FunHarmonic
from .so3funclass import SO3Fun
from ..geometry.install import install

SO3_KINDS = (SO3FunHarmonic, SO3FunRBF, SO3FunCBF, SO3FunGeneric)
S2_KINDS = (S2FunHarmonic, S2FunHandle, S2FunSum, S2FunBingham, S2FunTri)


def _ori(r, f):
  return orientation._fromRotation(r.data if isinstance(r, quaternion) else np.asarray(r), (f.CS, f.SS), f.antipodal)


# ------------------------------------------------------------------ scalars


def textureIndex(f):
  """the texture index of a function on SO(3) or on the sphere, the squared L2 norm of its harmonic
  expansion"""
  if isinstance(f, SO3Fun):
    return (f if isinstance(f, SO3FunHarmonic) else SO3FunHarmonic(f)).norm() ** 2
  return S2FunHarmonic(f).norm() ** 2


# -v log v, zero where v is not positive
def _negEntropy(v):
  return np.where(v > 0, -v * np.log(np.where(v > 0, v, 1.0)), 0.0)


def entropy(f, bandwidth=None):
  """the entropy -int f log f of a function on SO(3) or on the sphere; MATLAB's entropy

    e = odf.entropy()                 # at twice the bandwidth, at least 64 (128 on the sphere)
    e = odf.entropy(bandwidth=128)    # the quadrature of another bandwidth

  The integral is computed by the exact quadrature rule of a raised bandwidth, since f log f
  is not bandlimited; points where f is not positive contribute zero. On the sphere it is the
  mean over the sphere, as MATLAB divides by 4 pi.

  Input
    f - SO3Fun or S2Fun
  Output
    e - the entropy, a number
  Keywords
    bandwidth - the bandwidth of the quadrature grid, at most twice the preference
        maxSO3Bandwidth (maxS2Bandwidth on the sphere)
  See also
    textureIndex, SO3FunHarmonic.norm"""
  if isinstance(f, SO3Fun):
    L0 = max(2 * f.bandwidth, 64) if bandwidth is None else int(bandwidth)
    L = bandwidthFor(min(L0, 2 * getMTEXpref("maxSO3Bandwidth")), f.CS, f.SS)
    g = quadratureSO3Grid(L, f.CS, f.SS)
    v = np.real(f(g if isinstance(f, SO3FunHarmonic) else g.nodes()))
    return float(np.sum(g.weights * _negEntropy(v)))
  L0 = max(2 * f.bandwidth, 128) if bandwidth is None else int(bandwidth)
  g = quadratureS2Grid(min(L0, 2 * getMTEXpref("maxS2Bandwidth")))
  v = np.real(f(g if isinstance(f, S2FunHarmonic) else g.nodes()))
  return float(np.sum(g.weights * _negEntropy(v)) / (4 * np.pi))


def volume(f, center, radius, resolution=None):
  """the share of orientations within a radius of a centre: the integral of f over the ball,
  int_{angle(g, c) < r} f(g) dg, relative to the integral over all orientations; MATLAB's volume

    v = volume(odf, ori, 20*degree)                       # around an orientation
    v = volume(odf, ori, 20*degree, resolution=1*degree)  # a finer local grid
    v = volume(odf, f, 10*degree)                         # around a fibre
    v = volume(sF, xvector, 20*degree)                    # a spherical function around a direction
    v = volume(ori, center, 10*degree)                    # the share of a list of orientations

  The region is the union of the symmetric copies of the ball, which coincide where a symmetry
  fixes the centre and overlap for a large radius. The ball is integrated in shells about the
  centre, their axes a Fibonacci lattice, and every point is divided among the copies that
  hold it, so the union counts once: the volume is the mean of f over these shares times the
  measure of all copies. A uniform density gives the share of the
  region itself, so a value above it marks a preferred orientation.

  On the sphere the ball is a spherical cap; an antipodal function over a cap of 90 degree
  gives its mean value.

  The centre may also be:

  - a fibre: the volume of the tube about it, as fibreVolume.
  - a list of centres: one value each, an empty array for none.

  Given a list of orientations instead of a function, the result is the share of them
  within the radius.

  Input
    f - SO3Fun, S2Fun or orientation
    center - orientation, fibre, or vector3d on the sphere
    radius - the radius of the ball, an angle
  Output
    v - the share of the volume, a number, or an array of one per centre
  Keywords
    resolution - the spacing of the local grid, by default the smaller of 1.25 degree and a
        thirtieth of the radius (on the sphere 0.1 degree and a two hundredth)
  See also
    fibreVolume, entropy, SO3FunHarmonic.mean"""
  if isinstance(f, orientation):
    if isinstance(center, fibre):
      return float(np.count_nonzero(center.angle(f) <= radius) / f.size)
    return float(np.count_nonzero(f.angle(center) <= radius) / f.size)
  if isinstance(center, fibre):
    return fibreVolume(f, center, radius)
  if center.size != 1:
    return np.array([volume(f, center[i], radius, resolution) for i in range(center.size)], dtype=float).reshape(center.shape)
  if radius <= 0:
    return 0.0
  if isinstance(f, S2Fun):
    return _volumeS2(f, center, radius, resolution)
  _rotationsFor(f, center)
  res = min(1.25 * degree, radius / 30) if resolution is None else resolution
  L, R = reductionGroups(center)
  q, w = _ballShells(center.data.reshape(4), min(radius, np.pi), res)
  g = orientation._fromRotation(q, (center.frameA, center.frameB), center.antipodal)
  copies = qmul(qmul(L[:, None, :], center.data.reshape(1, 1, 4)), R[None, :, :]).reshape(-1, 4)
  if f.antipodal:
    copies = np.concatenate([copies, qconj(copies)])
  m = _coverCount(q, copies, center.data.reshape(4), radius)
  share = len(copies) * (radius - math.sin(radius)) / math.pi
  return float(np.sum(w * np.real(f(g)).reshape(-1) / m) / np.sum(w) * share)


# the ball of a radius about the rotation c as shells of equal width, the axes of a shell a
# Fibonacci lattice of the spacing res, every rotation weighted by the Haar measure it stands for
def _ballShells(c, radius, res):
  m = max(int(np.ceil(radius / res)), 1)
  domega = radius / m
  omega = (np.arange(m) + 0.5) * domega
  counts = np.maximum(np.round(16 * np.pi * np.sin(omega / 2) ** 2 / res**2).astype(int), 1)
  shell = np.repeat(np.arange(m), counts)
  k = np.arange(len(shell)) - np.repeat(np.cumsum(counts) - counts, counts)
  n = counts[shell]
  z = 1 - (2 * k + 1) / n
  rho = np.sqrt(1 - z**2)
  phi = k * np.pi * (3 - math.sqrt(5))
  half = omega[shell] / 2
  axis = np.sin(half)[:, None] * np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)
  q = np.concatenate([np.cos(half)[:, None], axis], axis=1)
  w = (2 * np.sin(omega / 2) ** 2 * domega / np.pi / counts)[shell]
  return qmul(q, np.broadcast_to(c, q.shape)), w


# how many of the copies of the ball about c hold each rotation q; only the copies within twice
# the radius of c can, and c's own ball holds every q
def _coverCount(q, copies, c, radius):
  near = copies[np.abs(copies @ c) >= math.cos(min(radius, np.pi / 2)) - 1e-12]
  m = np.empty(len(q), dtype=np.int64)
  step = max(1, 2**24 // max(len(near), 1))
  for i in range(0, len(q), step):
    m[i:i + step] = np.count_nonzero(np.abs(q[i:i + step] @ near.T) >= math.cos(radius / 2) - 1e-12, axis=1)
  return np.maximum(m, 1)


# the cap of a radius about the unit vector c as rings of equal width, every point weighted by the
# area it stands for
def _capRings(c, radius, res):
  m = max(int(np.ceil(radius / res)), 1)
  dtheta = radius / m
  theta = (np.arange(m) + 0.5) * dtheta
  counts = np.maximum(np.round(2 * np.pi * np.sin(theta) / dtheta).astype(int), 1)
  th = np.repeat(theta, counts)
  ph = np.concatenate([np.linspace(0, 2 * np.pi, k, endpoint=False) for k in counts])
  w = np.repeat(np.sin(theta) * dtheta / counts, counts)
  local = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], axis=1)
  return qrot(np.tile(rotation.map(np.array([0.0, 0.0, 1.0]), c).data.reshape(1, 4), (len(local), 1)), local), w


# the share of a spherical density within the caps about the distinct directions of the orbit
# of the centre, both ends of an axis among them: every point of the cap about the centre, in
# rings of their area, is divided among the caps that hold it, times the area of all caps,
# S2Fun/volume
def _volumeS2(f, center, radius, resolution=None):
  res = min(0.25 * degree, radius / 60) if resolution is None else resolution
  c = center.normalize().data.reshape(3)
  u, w = _capRings(c, radius, res)
  orbit = center.symmetrise(unique=True) if getattr(getattr(f, "frame", None), "sym", None) is not None else center
  dirs = orbit.normalize().data.reshape(-1, 3)
  dirs = _uniqueDirections(np.concatenate([dirs, -dirs]) if f.antipodal else dirs)
  cover = np.count_nonzero(u @ dirs.T >= math.cos(radius) - 1e-12, axis=1)
  vals = np.real(f(center._new(u))).reshape(-1)
  return float(np.sum(w * vals / np.maximum(cover, 1)) / np.sum(w) * len(dirs) * (1 - math.cos(radius)) / 2)


def fibreVolume(f, h, r=None, radius=None, resolution=None):
  """the volume of a fibre: the share of an orientation density within a radius of the fibre
  of the orientations that map the crystal direction h onto the specimen direction r

    v = fibreVolume(odf, h, r, radius)                       # the volume within radius of the fibre h || r
    v = fibreVolume(odf, fibre(h, r), radius)                 # the fibre given as a fibre
    v = fibreVolume(odf, h, r, radius, resolution=2*degree)  # a coarser quadrature

  The function is averaged over the tube about the fibre, the orientations g with g h within
  the radius of r: the directions u on the cap of the radius about r, and for each the turns
  about u a degree apart, or the resolution where it is coarser. The region is the union of the tubes of the symmetrically equivalent
  pairs of h and r, the orientations whose angle to the fibre (fibre.angle) is below the
  radius; the tubes overlap where the equivalents of h or of r lie within twice the radius, so
  every point of the tube is divided among the tubes that hold it. The mean of these shares
  times the measure of all tubes is the volume; for the uniform density without overlap it is
  (1 - cos radius) / 2 times the number of the pairs.

  Input
    f - SO3Fun, an orientation density
    h - Miller, the crystal direction, or a fibre, whose h and r are taken
    r - vector3d, the specimen direction; with a fibre the radius
    radius - the angle of the tube about the fibre in radians
  Output
    v - float, the volume, between 0 and 1 for a density
  Keywords
    resolution - the spacing of the quadrature of the tube; by default the smaller of one
        degree and a thirtieth of the radius
  See also
    SO3Fun.volume, SO3Fun.entropy, fibre, ODFCharacteristics"""
  if isinstance(h, fibre):
    h, r, radius = h.h, h.r, r
  res = min(1 * degree, radius / 30) if resolution is None else resolution
  rd = vector3d(r.data.reshape(3)).normalize().data.reshape(3)
  u, wcap = _capRings(rd, radius, res)
  hd = vector3d(h.data.reshape(3)).normalize().data.reshape(3)
  # the circle g h = u as the map of h onto u after the turns about h, a degree apart, which
  # resolves a kernel of a few degree along the circle
  n = max(int(np.ceil(2 * np.pi / max(res, 1 * degree))), 36)
  turn = rotation.byAxisAngle(vector3d(hd), np.linspace(0, 2 * np.pi, n, endpoint=False)).data.reshape(1, n, 4)
  q = qmul(rotation.map(np.broadcast_to(hd, u.shape), u).data.reshape(-1, 1, 4), turn).reshape(-1, 4)
  g = orientation._fromRotation(q, (f.CS, f.SS), getattr(f, "antipodal", False))
  # the equivalent pairs of h and r, and how many of their tubes hold each point of this one
  L, R = _framesOf(f)
  hs = _uniqueDirections(qrot(R, hd.reshape(1, 3)))
  rs = _uniqueDirections(qrot(L, rd.reshape(1, 3)))
  vals = (np.real(f(g)).reshape(-1) / _tubeCoverCount(q, hs, rs, hd, rd, radius)).reshape(len(u), n).mean(axis=1)
  return float(np.sum(wcap * vals) / np.sum(wcap) * len(hs) * len(rs) * (1 - math.cos(radius)) / 2)


# how many of the tubes g h' . r' > cos(radius) of the pairs of hs and rs hold each rotation q of
# the tube of h and r: a tube can meet it only where the angle of h to h' matches that of r to r'
# within twice the radius, and r' . g h' is the matrix of q against the outer product r' h'
def _tubeCoverCount(q, hs, rs, h, r, radius):
  ah, ar = np.arccos(np.clip(hs @ h, -1, 1)), np.arccos(np.clip(rs @ r, -1, 1))
  pairs = np.argwhere(np.abs(ah[:, None] - ar[None, :]) < 2 * radius + 1e-9)
  B = np.einsum("pi,pj->pij", rs[pairs[:, 1]], hs[pairs[:, 0]]).reshape(len(pairs), 9)
  M = quat2mat(q).reshape(-1, 9)
  m = np.empty(len(q), dtype=np.int64)
  step = max(1, 2**24 // max(len(pairs), 1))
  for i in range(0, len(q), step):
    m[i:i + step] = np.count_nonzero(M[i:i + step] @ B.T >= math.cos(radius) - 1e-12, axis=1)
  return np.maximum(m, 1)


# ------------------------------------------------------------------ misorientations


def calcMDF(f1, f2=None, bandwidth=32):
  """the uncorrelated misorientation distribution function of one or two ODFs; MATLAB's calcMDF

    mdf = calcMDF(odf)                         # of one ODF, with grain exchange symmetry
    mdf = calcMDF(odf1, odf2, bandwidth=16)    # of two ODFs, at another bandwidth

  The density of inv(o1) o2 for independent grains, (inv f1 * f2)(g) = int f1(S g^-1)
  f2(S) dS, computed by the convolution of the harmonic expansions; for one ODF the result
  has the exchange symmetry of the misorientations within one phase.

  Input
    f1 - SO3Fun
    f2 - SO3Fun, the second phase; f1 when omitted
  Output
    mdf - SO3FunHarmonic, from the crystal frame of f1 to that of f2
  Keywords
    bandwidth - the bandwidth the expansions are truncated to
  See also
    calcMIndex, calcAngleDistribution, calcAxisDistribution, SO3FunHarmonic.conv"""
  h1 = SO3FunHarmonic(f1, bandwidth=min(bandwidth, f1.bandwidth)).truncate(bandwidth)
  h2 = h1 if f2 is None or f2 is f1 else SO3FunHarmonic(f2, bandwidth=min(bandwidth, f2.bandwidth)).truncate(bandwidth)
  m = conv(h1.conj().inv(), h2)
  return m.withExchange() if f2 is None or f2 is f1 else m


# the rotations whose own angle is the reduced one, decided in bulk against the identity.
# The orbit of a generic rotation has |L| such members, its conjugates by the specimen
# group, which carry the images of its axis under that group, so integrating over this set
# with the weight |R| gives the specimen-symmetric axis density and the angle density
def _minimalAngle(q, L, R):
  return np.abs(q[:, 0]) >= symDots(q, np.array([1.0, 0, 0, 0]), L, R) - 1e-10


def _framesOf(f):
  return reductionGroups(orientation(rotation.id, f.CS, f.SS))


def calcAngleDistribution(f, *args, omega=None, edges=None, resolution=0.5 * degree):
  """the distribution of the misorientation angle: of a misorientation density, of a list of
  misorientations, or of the uniform density of two crystal frames

    density, omega = calcAngleDistribution(mdf)                            # the angle distribution of a misorientation density
    density, omega = calcAngleDistribution(mdf, omega=np.linspace(0, 60, 31)*degree)  # at the angles given
    density, omega = calcAngleDistribution(mori)                           # a histogram of misorientations
    density, omega = calcAngleDistribution(mori, edges=np.linspace(0, 63, 22)*degree)  # with the bin edges given
    ad, omega = calcAngleDistribution(cs)                                  # the uniform distribution of one crystal frame
    ad, omega = calcAngleDistribution(cs, cs2)                             # the uniform one of two crystal frames
    ad, omega = calcAngleDistribution(cs, np.linspace(0, 60, 61)*degree)   # at the angles given

  The result depends on the input:

  - a crystal frame, or two: the distribution of the uniform density, the area of the axis
  sector of the fundamental region at each angle times sin^2(omega/2), scaled to mean one
  over the angles.
  - an SO3Fun: that uniform distribution times the mean of the function over the axes at each
  angle, scaled by the same factor, so that a uniform density gives the uniform distribution.
  - misorientations: a histogram of the disorientation angles in percent, summing to 100,
  with omega the centres of the bins, zero in every bin for no misorientations.

  The angles are in radians and end at the largest angle of the fundamental region; angles
  given beyond it are dropped. They may also be given positionally, after the second frame.
  MATLAB returns the same two outputs in this order.

  Input
    f - SO3Fun, a misorientation density; orientation, misorientations; or crystalFrame
    cs2 - crystalFrame, the second frame of the uniform distribution (positional)
  Output
    density - ndarray, the values of the distribution
    omega - ndarray, the angles they belong to
  Keywords
    omega - the angles at which the distribution is computed, 200 up to the largest angle by
        default (SO3Fun and frames)
    edges - the bin edges of the histogram, by default between 4 and 49 bins up to the largest
        angle, one per 20 misorientations (misorientations)
    resolution - the spacing of the axes sampled at each angle (SO3Fun)
  See also
    calcAxisDistribution, calcMDF, calcMisorientation, plotAngleDistribution,
    MisorientationDistributionFunction"""
  from ..geometry.orientationregion import fundamentalRegion
  if args and isinstance(args[0], referenceFrame):
    SS, args = args[0], args[1:]
  else:
    SS = None
  if args and omega is None:
    omega = np.asarray(args[0], dtype=float).ravel()
  if isinstance(f, referenceFrame):
    oR = fundamentalRegion(f, SS)
    w = np.linspace(0, oR.maxAngle(), 200) if omega is None else omega[omega < oR.maxAngle() + 1e-8]
    # the area of the axis sector at each angle, counted on a Fibonacci grid, which has no
    # rings to alias with the circular caps of the sector
    V = fibonacciS2Grid(1000000).data.reshape(-1, 3)
    V = V[oR.axisSector(0.0).checkInside(V)]
    ad = np.array([np.count_nonzero(oR.axisSector(x).checkInside(V)) for x in w]) * np.sin(w / 2) ** 2
    return ad / np.mean(ad), w
  if isinstance(f, orientation):
    top = fundamentalRegion(f.CS, f.SS).maxAngle()
    e = np.linspace(0, top, int(min(50, max(5, f.size / 20)))) if edges is None else np.asarray(edges, dtype=float)
    counts, _ = np.histogram(f.angle().ravel(), e)
    return 100 * counts / max(np.sum(counts), 1), (e[:-1] + e[1:]) / 2
  L, R = _framesOf(f)
  top = fundamentalRegion(f.CS, f.SS).maxAngle()
  w = np.linspace(0, top, 200) if omega is None else omega[omega < top + 1e-8]
  rots, which = [], []
  for k, x in enumerate(w):
    n = max(4, round(4 / 3 * math.sin(x / 2) ** 2 / resolution ** 2))
    axes = fibonacciS2Grid(n)
    rots.append(rotation.byAxisAngle(vector3d(axes.data), x).data.reshape(-1, 4))
    which.append(np.full(len(rots[-1]), k))
  q, which = np.concatenate(rots), np.concatenate(which)
  inside = _minimalAngle(q, L, R)
  v = np.zeros(len(q))
  v[inside] = np.maximum(np.real(f(_ori(q[inside], f))), 0)
  counts = np.bincount(which, minlength=len(w))
  uniform = np.sin(w / 2) ** 2 * np.bincount(which, weights=inside.astype(float), minlength=len(w)) / counts
  density = np.sin(w / 2) ** 2 * np.bincount(which, weights=v, minlength=len(w)) / counts
  return density / np.mean(uniform), w


def calcMIndex(f, resolution=1 * degree):
  """the M-index of an ODF by Skemer et al., computed from the ODF instead of a discrete
  number of orientations; MATLAB's calcMIndex

    mi = calcMIndex(odf)                         # the M-index
    mi = calcMIndex(odf, resolution=2*degree)    # angle distributions on a coarser grid

  The M-index is half the total variation between the misorientation angle distributions of
  the uniform and of the uncorrelated misorientation distribution, calcMDF(odf): zero for a
  random texture, one for a single crystal.

  Input
    odf - SO3Fun
  Output
    mi - the M-index, a number between 0 and 1
  Keywords
    resolution - the step of the angle distributions
  References
    P. Skemer, I. Katayama, Z. Jiang, S.-I. Karato, The misorientation index: development of
        a new method for calculating the strength of lattice-preferred orientation,
        Tectonophysics 411 (2005) 157-167
    H. Schaeben, Comment on "The misorientation index", Tectonophysics 441 (2007) 115-117
    D. Mainprice, F. Bachmann, R. Hielscher, H. Schaeben, Descriptive tools for the analysis
        of texture projects with large datasets using MTEX: strength, symmetry and components,
        Geological Society, London, Special Publications 409 (2014)
  See also
    calcMDF, calcAngleDistribution, textureIndex"""
  m = calcMDF(f)
  pu, _ = calcAngleDistribution(uniformODF(f.CS, f.CS, antipodal=True), resolution=resolution)
  pm, _ = calcAngleDistribution(m, resolution=resolution)
  return float(np.sum(np.abs(pu / np.sum(pu) - pm / np.sum(pm))) / 2)


# the density of the rotation axis: adf(a) = 2 copies/pi int f(a, omega) sin^2(omega/2) d omega
# over the angles for which (a, omega) lies in the fundamental region and in the window asked,
# by a rectangle rule; a spherical function in the crystal frame when the frames coincide
def _axisDensity(f, resolution=2.5 * degree, minAngle=0.0, maxAngle=np.pi):
  L, R = _framesOf(f)
  copies = len(R)
  omega = np.arange(resolution / 2, np.pi, resolution)
  window = (omega >= max(minAngle, 0)) & (omega <= min(maxAngle, np.pi))

  def value(vs):
    d = vector3d(vs.data.reshape(-1, 3)).normalize()
    q = rotation.byAxisAngle(d[None, :], omega[:, None]).data.reshape(-1, 4)
    use = (_minimalAngle(q, L, R).reshape(len(omega), d.size)) & window[:, None]
    v = np.zeros((len(omega), d.size))
    idx = np.flatnonzero(use.ravel())
    if idx.size:
      v.ravel()[idx] = np.real(f(_ori(q[idx], f)))
    out = 2 * copies / np.pi * np.sum(v * np.sin(omega / 2)[:, None] ** 2, axis=0) * resolution
    return out.reshape(vs.shape)

  return value


# the axis density of uniformly distributed rotations in the fundamental region: the angle
# integral of sin^2(omega/2) up to the largest angle about the axis, (w - sin w) / 2, which
# the planes of the region give in closed form; mean one over the sphere
def _uniformAxisDensity(oR):
  def wmax(xyz):
    w = np.full(len(xyz), np.pi)
    for n in oR.N:
      a = n[1:]
      s = np.linalg.norm(a)
      c = xyz @ a / s
      pos = c > 1e-9
      w[pos] = np.minimum(w[pos], 2 * np.arctan(-n[0] / s / c[pos]))
    return w

  V = fibonacciS2Grid(200000).data.reshape(-1, 3)
  w = wmax(V)
  scale = 1 / np.mean(w - np.sin(w))

  def value(vs):
    xyz = vector3d(vs.data.reshape(-1, 3)).normalize().data.reshape(-1, 3)
    w = wmax(xyz)
    return (scale * (w - np.sin(w))).reshape(vs.shape)

  return value


def calcAxisDistribution(f, v=None, *args, bandwidth=64, **kwargs):
  """the axis distribution of an SO(3) function, an S2 function of the rotation axes symmetric under
  the rotations the two frames share; with v its values at the axes v; for two frames the axis
  distribution of the uniform distribution, exactly; MATLAB's calcAxisDistribution

    adf = calcAxisDistribution(mdf)                                          # the axis distribution
    value = calcAxisDistribution(mdf, v)                                     # its values at the axes v
    adf = calcAxisDistribution(mdf, minAngle=20*degree, maxAngle=40*degree)  # only rotations by 20 to 40 degree
    value = calcAxisDistribution(mdf, v, resolution=1*degree)                # a finer angle quadrature
    adf = calcAxisDistribution(cs1, cs2)                                     # of the uniform distribution

  By default the rotation angle is integrated over the full range the fundamental region
  allows for the axis in question, by a rectangle rule of the step resolution (2.5 degree).
  The keywords minAngle and maxAngle restrict that range to a window, which is what the axis
  angle sections show; an axis for which the window lies outside the fundamental region
  contributes zero. The quadrature keeps its step, so a narrow window is integrated with
  few points. The distribution is antipodal for a function with grain exchange.

  Input
    f - SO3Fun, or a crystal frame cs1 for the uniform distribution
    v - vector3d, the axes to evaluate at, or the second frame cs2
  Output
    adf - S2FunHarmonic, or S2FunHandle for the uniform distribution
    value - the values at the axes v
  Keywords
    bandwidth - the bandwidth of the harmonic expansion of adf
  See also
    calcAngleDistribution, calcMDF, axisAngleSections"""
  if isinstance(f, referenceFrame):
    # the uniform distribution exactly, not as a truncated expansion
    from ..geometry.orientationregion import fundamentalRegion
    CS = f
    SS = v if isinstance(v, referenceFrame) else CS
    v = None if isinstance(v, referenceFrame) else v
    value = _uniformAxisDensity(fundamentalRegion(CS, SS))
    if v is not None:
      return value(asVector3d(v))
    return S2FunHandle(value, CS if CS is SS else None, antipodal=True)
  if v is not None:
    return _axisDensity(f, **kwargs)(asVector3d(v))
  # the axes live in the first crystal frame, symmetric under the rotations the two frames
  # share, and antipodal only with grain exchange
  antipodal = bool(getattr(f, "antipodal", False))
  frame = None
  if isinstance(f.CS, crystalFrame) and isinstance(f.SS, crystalFrame):
    g = f.CS.sym.disjoint(f.SS.sym).properGroup()
    frame = f.CS if f.CS is f.SS and not antipodal else f.CS.sibling(g.Laue() if antipodal else g)
  return S2FunHarmonic(S2FunHandle(_axisDensity(f, **kwargs), frame, bandwidth=bandwidth, antipodal=antipodal))


def calcError(f, g, mode=None, resolution=5 * degree, epsilon=None):
  """the distance between two orientation density functions, or the error of an ODF against
  pole figures

    e = calcError(odf1, odf2)                                # half the mean absolute difference
    e = calcError(odf1, odf2, 'l2', resolution=2.5*degree)  # the relative L2 error on a finer grid
    e = calcError(odf, pf, 'RP')                             # the RP error against pole figures

  Two functions are compared at the nodes of an equispaced grid of the resolution in the
  fundamental region of f; their crystal groups and their specimen groups must be of the same
  Laue class. The mode chooses the measure:

  - 'l1', the default: half the mean absolute difference. For two densities it is the fraction
  of the volume that would have to move to turn one into the other, between 0 and 1.
  - 'l2': the norm of the difference over the norm of g, on the grid; MATLAB computes it from
  the harmonic series.

  With pole figures second the result is PoleFigure.calcError, one number per pole figure,
  with its own modes ('relative', the default, 'RP', 'l1', 'l2') and the threshold epsilon.
  MATLAB's 'L0', the volume where the two differ by more than epsilon, is not provided.

  Input
    f - SO3Fun
    g - SO3Fun, or PoleFigure
    mode - 'l1' or 'l2' for two functions, 'relative', 'RP', 'l1' or 'l2' with pole figures
  Output
    e - float, or one float per pole figure
  Keywords
    resolution - the spacing of the grid on which two functions are compared
    epsilon - the threshold of the error against pole figures
  See also
    PoleFigure.calcError, calcODF, PoleFigure2ODF"""
  from .polefigure import PoleFigure
  if isinstance(g, PoleFigure):
    return g.calcError(f, "relative" if mode is None else mode, epsilon)
  if epsilon is not None:
    raise ValueError("epsilon is the threshold of the error of pole figures")
  mode = "l1" if mode is None else str(mode).lower()
  if f.CS.sym.Laue() != g.CS.sym.Laue() or f.SS.sym.Laue() != g.SS.sym.Laue():
    raise ValueError(f"the functions have different symmetries, {framesLabel(f.CS, f.SS)} and {framesLabel(g.CS, g.SS)}")
  grid = equispacedSO3Grid(f.CS, f.SS, resolution=resolution)
  d1 = np.real(f(grid))
  d2 = np.real(g(orientation._fromRotation(grid.data, (g.CS, g.SS), g.antipodal)))
  if mode == "l1":
    return float(np.sum(np.abs(d1 - d2)) / d1.size / 2)
  if mode == "l2":
    return float(np.linalg.norm(d1 - d2) / np.linalg.norm(d2))
  raise ValueError(f"the error mode is 'l1' or 'l2', not {mode!r}")


# ------------------------------------------------------------------ extrema
# the best seeds of an equispaced grid, refined on a shrinking local grid around each until
# the resolution reaches the accuracy asked, then merged within the seed resolution;
# derivative free, so every representation works


# greedy clusters by the symmetry reduced angle: each orientation joins the cluster of the first
# of their number within the tolerance of it; the neighbourhoods from one radius search of the
# quaternion tree over the symmetric images, of the orientations among only when given (the
# pairs of two others are taken as apart). An orientation without neighbours leads its own
# cluster, so the loop runs over the others and the clusters are numbered by their leaders
def _clusterOrientations(o, tol, among=None):
  from scipy import sparse
  from ..geometry.grids.orientationtree import OrientationTree
  o = o.flatten()
  n = o.size
  among = np.arange(n) if among is None else np.asarray(among, dtype=int)
  I, _ = OrientationTree(o).radius(o[among], tol)
  I = I.tocoo()
  rows, cols = among[I.row], I.col
  apart = rows != cols
  A = sparse.csr_matrix((np.ones(int(apart.sum()), dtype=bool), (rows[apart], cols[apart])), shape=(n, n))
  A = (A + A.T).tocsr()
  indptr, indices = A.indptr, A.indices
  leader = np.arange(n)
  done = np.zeros(n, dtype=bool)
  for i in np.flatnonzero(np.diff(indptr)):
    if done[i]:
      continue
    near = indices[indptr[i]:indptr[i + 1]]
    near = near[~done[near]]
    leader[near], done[near], done[i] = i, True, True
  return np.unique(leader, return_inverse=True)[1]


# the basins of a function on an SO(3) grid: every node points to its highest neighbour within
# the angle, itself at a peak, the larger number deciding a tie so that no pointers cycle; the
# pointers followed to their peaks by doubling
def _gridBasins(grid, v, angle):
  from ..geometry.grids.orientationtree import gridRadius
  I, _ = gridRadius(grid, grid, angle)
  n = grid.size
  rank = np.empty(n, dtype=int)
  rank[np.lexsort((np.arange(n), v))] = np.arange(n)
  byRank = np.argsort(rank)
  best = rank.copy()
  rowOf = np.repeat(np.arange(n), np.diff(I.indptr))
  np.maximum.at(best, rowOf, rank[I.indices])
  up = byRank[best]
  while True:
    nxt = up[up]
    if np.array_equal(nxt, up):
      return up
    up = nxt


# the member of each cluster with the largest value, the first of their number on a tie
def _bestOfClusters(ids, vals, nc):
  order = np.lexsort((np.arange(len(ids)), -vals, ids))
  first = np.concatenate([[True], ids[order][1:] != ids[order][:-1]])
  rep = np.empty(nc, dtype=int)
  rep[ids[order][first]] = order[first]
  return rep


# the seeds of the ascent: the nodes that beat every neighbour within the angle whose cosine
# is c, the neighbourhood taken over the copies of the grid under the group (a node at the
# rim of the fundamental region has its uphill neighbour across it) and over the sign of a
# quaternion or an axis; the best 10 numLocal of them
def _bestSeeds(cand, v, numLocal, c, copies=None):
  from scipy.spatial import cKDTree
  tree = cKDTree(cand)
  r = math.sqrt(max(2 - 2 * c, 0.0))
  best = np.full(len(v), -np.inf)
  for pts in ([cand] if copies is None else copies):
    for sign in (1.0, -1.0):
      pairs = cKDTree(sign * pts).sparse_distance_matrix(tree, r, output_type="ndarray")
      np.maximum.at(best, pairs["i"], v[pairs["j"]])
  order = np.argsort(v)[::-1]
  return order[v[order] >= best[order]][:10 * numLocal]


# the copies of the grid nodes under the group elements, one array per element
def _copiesSO3(cand, f):
  # the proper rotations of the groups: an improper element maps no rotation onto the grid
  def proper(fr):
    return (fr.properGroup() if hasattr(fr, "properGroup") else fr).sym.rot.flatten()
  Ls, Rs = proper(f.CS), proper(f.SS)
  q = rotation(cand)
  return [(rotation(Ls[i]) * q * rotation(Rs[j])).data.reshape(-1, 4) for i in range(Ls.size) for j in range(Rs.size)]


def _copiesS2(cand, fr):
  if not isinstance(fr, referenceFrame):
    return None
  els = fr.sym.rot.flatten()
  return [(els[i] * vector3d(cand)).data.reshape(-1, 3) for i in range(els.size)]


def _localMaximaSO3(f, numLocal=1, resolution=5 * degree, accuracy=0.25 * degree, startingNodes=None):
  g = equispacedSO3Grid(f.CS, f.SS, resolution=resolution) if startingNodes is None else startingNodes
  cand = rotation(g).data.reshape(-1, 4)
  v = np.real(f(_ori(cand, f))).ravel()
  keep = _bestSeeds(cand, v, numLocal, math.cos(1.5 * resolution / 2), _copiesSO3(cand, f))
  modes, vals = cand[keep].copy(), v[keep].copy()
  res = resolution
  while res > accuracy:
    res /= 1.25
    shifts = rotation(localOrientationGrid(f.CS, f.SS, maxAngle=2 * res, resolution=res / 2, respectSymmetry=False)).data.reshape(-1, 4)
    # the shifted grids of every mode in one evaluation, a row per mode
    trial = qmul(shifts[None, :, :], modes[:, None, :])
    tv = np.real(f(_ori(trial.reshape(-1, 4), f))).reshape(len(modes), -1)
    i = np.argmax(tv, axis=1)
    better = tv[np.arange(len(modes)), i] > vals
    modes[better], vals[better] = trial[better, i[better]], tv[better, i[better]]
  o = _ori(modes, f)
  ids = _clusterOrientations(o, 1.5 * resolution)
  reps = _bestOfClusters(ids, vals, ids.max() + 1)
  order = np.argsort(vals[reps])[::-1][:min(numLocal, len(reps))]
  sel = np.array(reps)[order]
  # the values of the modes found, from the function itself: a sum of many kernels is
  # searched through its harmonic form, which can undershoot the function
  best = _ori(modes[sel], f)
  return np.real(f(best)).ravel(), best


def _localMaximaS2(f, numLocal=1, resolution=2.5 * degree, accuracy=0.05 * degree):
  fr, antipodal = f.frame, f.antipodal
  g = equispacedS2Grid(resolution=resolution, antipodal=antipodal) if fr is None else equispacedS2Grid(fr, resolution=resolution, antipodal=antipodal)
  cand = g.data.reshape(-1, 3)
  v = np.real(f(vector3d(cand))).ravel()
  keep = _bestSeeds(cand, v, numLocal, math.cos(1.5 * resolution), _copiesS2(cand, fr))
  modes, vals = cand[keep].copy(), v[keep].copy()
  res = resolution
  while res > accuracy:
    res /= 1.25
    for k in range(len(modes)):
      cap = localS2Grid(vector3d(modes[k]), maxTheta=2 * res, resolution=res / 2, respectSymmetry=False)
      trial = cap.data.reshape(-1, 3)
      tv = np.real(f(vector3d(trial))).ravel()
      i = int(np.argmax(tv))
      if tv[i] > vals[k]:
        modes[k], vals[k] = trial[i], tv[i]
  # merge within the seed resolution, over the images under the group since the refinement
  # from seeds either side of a sector boundary can reach one mode and its copy
  els = None if fr is None else fr.sym.rot.flatten()
  c = math.cos(1.5 * resolution)

  def near(a, b):
    imgs = a[None, :] if els is None else (els * vector3d(a)).data.reshape(-1, 3)
    d = imgs @ b
    return bool(np.any(d > c) or (antipodal and np.any(-d > c)))

  reps = []
  for i in np.argsort(vals)[::-1]:
    if not any(near(modes[r], modes[i]) for r in reps):
      reps.append(i)
  reps = reps[:min(numLocal, len(reps))]
  pts = vector3d(modes[reps], frame=fr, antipodal=antipodal)
  return vals[reps], pts


# the largest value and where it is taken; numLocal > 1 gives the numLocal best local maxima
# max(f, g), min(f, 1): the pointwise extremum of two functions, or of a function and a number,
# expanded by quadrature at the larger bandwidth, antipodal when both are
def _pointwiseS2(op, f, g):
  isFun = isinstance(g, S2Fun)
  L = max(f.bandwidth, g.bandwidth) if isFun else f.bandwidth
  fun = (lambda v: op(np.real(f(v)), np.real(g(v)))) if isFun else (lambda v: op(np.real(f(v)), g))
  antipodal = f.antipodal and (g.antipodal if isFun else True)
  return S2FunHarmonic(fun, f.frame, bandwidth=L, antipodal=antipodal, real=True)


# the pointwise extremum of two functions on SO(3), or of one and a number, a handle
# evaluated when asked, as MATLAB's
def _pointwiseSO3(op, f, g):
  isFun = isinstance(g, SO3Fun)
  fun = (lambda o: op(np.real(f.eval(o)), np.real(g.eval(o)))) if isFun else (lambda o: op(np.real(f.eval(o)), g))
  L = max(f.bandwidth, g.bandwidth) if isFun else f.bandwidth
  return SO3FunHandle(fun, f.CS, f.SS, bandwidth=L)


# a list of functions on SO(3): along an axis the pointwise extremum, MATLAB's min(f, [], d),
# else the extremum of every function, the values and the positions
def _listExtremaSO3(f, op, one, numLocal, kwargs):
  from .funops import mapFun
  axis = kwargs.pop("axis", None)
  if axis is not None:
    return mapFun(lambda x: op(np.real(x), axis=1 + int(axis)), f, bandwidth=f.bandwidth)
  pairs = [one(f[k], numLocal=numLocal, **kwargs) for k in np.ndindex(f.shape)]
  return np.array([p[0] for p in pairs]).reshape(f.shape), [p[1] for p in pairs]


# a handle of several values reduced over one axis of its values, which follow the axes of
# the directions it is evaluated at
def _reduceHandle(f, op, axis):
  def fun(v):
    vals = np.asarray(f.eval(v))
    return op(vals, axis=np.ndim(v.data) - 1 + axis)
  return S2FunHandle(fun, f.frame, bandwidth=f.bandwidth, antipodal=f.antipodal, real=f.isReal, symmetrise=False)


def max_(f, *args, numLocal=1, **kwargs):
  """the maximum of a function on SO(3) or the sphere, the value and the position, numLocal local maxima
  from a grid refined down to the accuracy; with a second function or a number the pointwise
  maximum; with axis= the maximum of a list of functions along an axis; MATLAB's max

    value, ori = max(odf)                                                            # the global maximum and where it is
    values, oris = max(odf, numLocal=5, resolution=5*degree, accuracy=0.25*degree)  # the five largest local maxima
    f = max(odf1, odf2)                                                              # the pointwise maximum of two functions
    v, pos = max(sF)                                                                 # on the sphere, the direction
    values, positions = max(sF, numLocal=3)                                          # three local maxima on the sphere
    sF = max(sF, 0.5)                                                                # the pointwise maximum with a number

  The local maxima are searched from a grid of the spacing resolution (5 degree on SO(3), 2.5
  on the sphere) and refined down to accuracy (0.25 and 0.05 degree); the keywords
  resolution, accuracy and, on SO(3), startingNodes, the orientations to start from, go to
  that search. Maxima closer than the resolution are merged, so fewer than numLocal can come
  back. A function on the circle and a pole figure take their own max. Where MATLAB writes
  max(SO3F, [], dim), the port writes max(SO3F, axis=dim - 1).

  Input
    f - SO3Fun, S2Fun or S1Fun; a second function or a number for the pointwise maximum
  Output
    value - the largest value, or numLocal values in descending order
    ori - the position: orientation, vector3d, or the angle on the circle
  Keywords
    numLocal - the number of local maxima returned
  See also
    min, calcComponents, SO3FunHarmonic.mean"""
  return _extremum(f, args, numLocal, kwargs, largest=True)


def min_(f, *args, numLocal=1, **kwargs):
  """the minimum of a function on SO(3) or the sphere, the value and the position, numLocal local minima;
  with a second function or a number the pointwise minimum; with axis= along an axis of a list;
  MATLAB's min

    value, ori = min(odf)                        # the global minimum and where it is
    values, oris = min(odf, numLocal=3)          # three local minima
    f = min(odf, 0)                              # the pointwise minimum with a number

  The minimum is the maximum of -f, the keywords resolution, accuracy and startingNodes as
  for max.

  Input
    f - SO3Fun, S2Fun or S1Fun; a second function or a number for the pointwise minimum
  Output
    value - the smallest value, or numLocal values in ascending order
    ori - the position: orientation, vector3d, or the angle on the circle
  Keywords
    numLocal - the number of local minima returned
  See also
    max"""
  return _extremum(f, args, numLocal, kwargs, largest=False)


# the largest or the smallest values, by the kind of f: a pole figure's own, along an axis of a
# list, pointwise with a second function or a number, per function of a list, a function on the
# circle's own, or the local extrema searched from a grid; a minimum is the maximum of -f
def _extremum(f, args, numLocal, kwargs, largest):
  name, reduce, pointwise = ("max", np.max, np.maximum) if largest else ("min", np.min, np.minimum)
  if hasattr(f, "allI"):
    return getattr(f, name)(*args)
  axis = kwargs.get("axis")
  if axis is not None and isinstance(f, S2FunHarmonic):
    return f._reduce(reduce, axis)
  if axis is not None and isinstance(f, S2FunHandle):
    return _reduceHandle(f, reduce, axis)
  if args and isinstance(f, S2Fun) and (isinstance(args[0], S2Fun) or np.isscalar(args[0])):
    return _pointwiseS2(pointwise, f, args[0])
  if args and isinstance(f, SO3Fun) and (isinstance(args[0], SO3Fun) or np.isscalar(args[0])):
    return _pointwiseSO3(pointwise, f, args[0])
  if isinstance(f, SO3FunHarmonic) and f.shape:
    return _listExtremaSO3(f, reduce, max_ if largest else min_, numLocal, kwargs)
  if isinstance(f, (S1FunHarmonic, S1FunHandle)):
    return getattr(f, name)(*args, numLocal=numLocal, **kwargs)
  if not isinstance(f, (SO3Fun, S2Fun)):
    return (max if largest else min)(f, *args, **kwargs)
  if not largest:
    v, p = _extremum(-1 * f, (), numLocal, kwargs, largest=True)
    return -v, p
  v, p = (_localMaximaSO3 if isinstance(f, SO3Fun) else _localMaximaS2)(f, numLocal=numLocal, **kwargs)
  return (float(v[0]), p[0]) if numLocal == 1 else (v, p)


# ------------------------------------------------------------------ components
# steepest ascent from seeds along the gradient with a fixed step ladder that halves when no
# step improves, clusters merged within a tolerance, the weights of the seeds accumulated per mode.
# Unless exact, a seed lighter than a mode that is kept and on a slope below FLAT_SLOPE of the
# steepest at the start stops where it is: the ripples of a harmonic expansion are such seeds

FLAT_SLOPE = 1e-3


# the gradient that drives the ascent, as a function of orientations: a radial function with
# many centres through its harmonic form, as its evaluation goes; the field of a harmonic
# function built once
def _gradientField(f):
  if isinstance(f, SO3FunRBF):
    from .rbf import HARMONIC_EVAL
    L, R = reductionGroups(f.center)
    if f.center.size ** 2 * len(L) * len(R) > HARMONIC_EVAL and f.psi.bandwidth <= getMTEXpref("maxSO3Bandwidth"):
      return f._harmonicForm().grad()
  if isinstance(f, (SO3FunRBF, SO3FunCBF)):
    return f.grad
  return (f if isinstance(f, SO3FunHarmonic) else SO3FunHarmonic(f)).grad()


# the step ladder from the modes along the gradient field fg, the modes that meet joined; labels
# say which mode each seed has flowed into
def _ascend(f, fg, modes, vals, labels, w, steps, exact, tol, maxIter):
  active = np.ones(len(modes), dtype=bool)
  steepest = None
  for _ in range(maxIter):
    idx = np.flatnonzero(active)
    if idx.size == 0:
      break
    o = _ori(modes[idx], f)
    g = fg(o)
    n = np.linalg.norm(g.data, axis=-1)
    steepest = n.max() if steepest is None else steepest
    if not exact:
      flat = (np.bincount(labels, w, minlength=len(modes))[idx] < 0.01) & (n < FLAT_SLOPE * steepest)
      if flat.any():
        active[idx[flat]] = False
        idx, o, g, n = idx[~flat], o[~flat], g[~flat], n[~flat]
        if idx.size == 0:
          break
    dirs = g.data / np.where(n > 0, n, 1.0)[:, None]
    trial1 = SO3TangentVector(vector3d((2 * steps[idx] / 3)[:, None] * dirs), o, "left").exp()
    trial2 = SO3TangentVector(vector3d(steps[idx][:, None] * dirs), o, "left").exp()
    t1, t2 = rotation(trial1).data.reshape(-1, 4), rotation(trial2).data.reshape(-1, 4)
    # the modes and both trials in one evaluation: a large one goes through the harmonic form
    # and a small one is summed, and a step must be judged by values of one kind
    v0, v1, v2 = np.split(np.real(f(_ori(np.concatenate([modes[idx], t1, t2]), f))).ravel(), 3)
    vals[idx] = v0
    best = np.argmax(np.stack([vals[idx], v1, v2]), axis=0)
    # on a tie in a flat region the gradient still says where the mass is
    best[(best == 0) & (v1 == vals[idx]) & (v2 == vals[idx]) & (n > 0)] = 2
    stay = best == 0
    halve = stay & (steps[idx] > 0.05 * degree)
    steps[idx[halve]] /= 2
    active[idx[stay & ~halve]] = False
    for b, t, v in ((1, t1, v1), (2, t2, v2)):
      go = best == b
      modes[idx[go]], vals[idx[go]] = t[go], v[go]
    if stay.all():
      if not np.any(active):
        break
      continue
    # join the modes that have come within the tolerance of each other: a pair that has not
    # moved was looked at before
    ids = _clusterOrientations(_ori(modes, f), tol, among=idx[~stay])
    nc = ids.max() + 1
    if nc < len(modes):
      rep = _bestOfClusters(ids, vals, nc)
      keepActive = np.bincount(ids, weights=active, minlength=nc) > 0
      keepStep = np.zeros(nc)
      np.maximum.at(keepStep, ids, steps)
      labels = ids[labels]
      modes, vals, steps, active = modes[rep], vals[rep], keepStep, keepActive
  return modes, vals, labels


def calcComponents(f, seed=None, weights=None, tol=1.5 * degree, maxIter=100, radius=np.inf, exact=False, verbose=True):
  """the modal orientations of an orientation density, the components: their orientations, their
  volumes and the mode every seed flowed into

    modes, weights, centerId = calcComponents(odf)                       # the components from a grid of seeds
    modes, weights, centerId = calcComponents(odf, seed=ori)             # the modes the orientations ascend to
    modes, weights, centerId = calcComponents(odf, seed=ori, radius=10*degree, exact=True)  # volumes within a radius, every mode kept

  Every seed climbs by steepest ascent to a local maximum of the function. Maxima closer than
  tol are one mode, and its weight is the sum of the weights of the seeds that reached it,
  only of those within radius of it when a radius is given. The modes are ordered by
  decreasing weight.

  The seeds are, in this order of preference:

  - the orientations given as seed, weighted by the function's values there unless weights
  are given.
  - the centres of a radial function, with its weights.
  - the nodes of a 2.5 degree grid, with equal weights, which first climb to their peaks on
  the grid; the weight of a mode is then the share of the orientation space that flows into
  it, not its volume.

  Seeds of weight zero are left out, so centerId has one entry per seed of positive weight.
  Modes of less than one percent of the weight are dropped unless exact=True, and the seeds
  that reached them get centerId -1. No seed gives no modes. With verbose the number of seeds
  and of the components found are reported.

  Input
    f - SO3Fun
  Output
    modes - orientation, the modal orientations
    weights - ndarray, the weight of every mode, the volume of the component
    centerId - ndarray, the index of the mode every seed flowed into, numbered from 0, -1 for a
        dropped mode
  Keywords
    seed - orientation, the starting points of the ascent
    weights - the weights of the seeds, by default the values of the function at them, equal
        for the nodes of the grid
    tol - the angle within which modes merge, and the accuracy of the ascent
    maxIter - the largest number of steps of the ascent
    radius - only the seeds within this angle of their mode count to its weight; MATLAB's
        'angle'
    exact - keep the modes of less than one percent
    verbose - report the progress
  See also
    SO3Fun.max, SO3Fun.volume, ODFComponents"""
  grid = None
  if seed is None and isinstance(f, SO3FunRBF):
    seed, weights = f.center, f.weights
  elif seed is None:
    grid = seed = equispacedSO3Grid(f.CS, f.SS, resolution=2.5 * degree)
  rots = rotation(seed).data.reshape(-1, 4)
  if len(rots) == 0:
    return _ori(rots, f), np.zeros(0), np.zeros(0, dtype=int)
  if weights is None:
    w = np.ones(len(rots)) if grid is not None else np.real(f(_ori(rots, f))).ravel()
  else:
    w = np.asarray(weights, dtype=float).ravel()
  keep = w > 0
  rots, w = rots[keep], w[keep] / np.sum(w[keep])
  fg = _gradientField(f)
  # which mode each seed has flowed into; seeds that meet are one mode from then on, so the
  # working set shrinks as the ascent proceeds. The nodes of the grid climb to their peaks on
  # the grid first, and only the peaks ascend
  if grid is not None:
    v = np.real(f(grid)).ravel()
    peaks, labels = np.unique(_gridBasins(grid, v, 1.5 * grid.res), return_inverse=True)
    modes, vals = rots[peaks], v[peaks]
  else:
    modes, labels = rots.copy(), np.arange(len(rots))
    vals = np.real(f(_ori(modes, f))).ravel()
  if verbose:
    print(f"  finding ODF components: {len(modes)} seeds ascending")
  modes, vals, labels = _ascend(f, fg, modes, vals, labels, w, np.full(len(modes), 5 * degree), exact, tol, maxIter)
  # a radial or fibre function ascended on its harmonic form climbs on to its peaks on its own
  # gradient: near a peak the truncated gradient points off the ridge and the ladder stops short
  if isinstance(f, (SO3FunRBF, SO3FunCBF)) and fg != f.grad:
    modes, vals, labels = _ascend(f, f.grad, modes, vals, labels, w, np.full(len(modes), 0.5 * degree), True, tol, maxIter)
  # clusters, the representative with the largest value, the weights within the radius
  ids = _clusterOrientations(_ori(modes, f), tol)
  nclust = ids.max() + 1
  modeRots = modes[_bestOfClusters(ids, vals, nclust)]
  wsum = np.zeros(nclust)
  seedIds = ids[labels]
  if np.isinf(radius):
    np.add.at(wsum, seedIds, w)
  else:
    d = _ori(rots, f).angle(_ori(modeRots[seedIds], f))
    np.add.at(wsum, seedIds[d < radius], w[d < radius])
  order = np.argsort(wsum)[::-1]
  remap = np.empty(nclust, dtype=int)
  remap[order] = np.arange(nclust)
  centerId = remap[seedIds]
  wsum, modeRots = wsum[order], modeRots[order]
  if not exact:
    thr = min(0.01, 0.5 * (wsum.max() if wsum.size else 0.0))
    keepm = wsum > thr
    newId = np.cumsum(keepm) - 1
    centerId = np.where(keepm[centerId], newId[centerId], -1)
    wsum, modeRots = wsum[keepm], modeRots[keepm]
  if verbose:
    print(f"  {len(wsum)} components found")
  return _ori(modeRots, f), wsum, centerId


# ------------------------------------------------------------------ sampling


def discreteSample(f, n, resolution=None, rng=None, withoutReplacement=False, range=(0, 1)):
  """a random sample of n orientations drawn from an orientation density, of directions from a
  density on the sphere, of numbers from a density on an interval, or n random members of a list

    ori = discreteSample(odf, n)                                # n orientations of the density
    ori = discreteSample(odf, n, resolution=2*degree)           # from a coarser grid
    v = discreteSample(sF, n)                                   # n directions of a density on the sphere
    x = discreteSample(lambda x: np.exp(-x), n, range=(0, 5))   # n numbers of a density on an interval
    ori = discreteSample(ori, n, withoutReplacement=True)       # n members of a list, none twice
    ebsd = discreteSample(ebsd, n)                              # n measurements of a map, of any phases

  How the sample is drawn depends on what is sampled:

  - a density on SO(3) or the sphere: grid nodes drawn with probability proportional to the
  values, each moved by a random rotation of up to 1.5 times the grid resolution and turned by
  a random element of each symmetry group. A finer resolution gives a more exact sample.
  - a radial function (unimodalODF, calcDensity): exactly, a centre by its weight, the angle
  by the distribution of the kernel and a uniform axis; a sum of radial functions term by term.
  - a Bingham distribution: exactly, by rejection.
  - any other callable of numbers (a def, a lambda, a NumPy ufunc such as np.exp, an
  S1Fun): its values at 10000 points of the range.
  - a list, a map, grains or boundary segments: n of its members, with replacement unless
  withoutReplacement=True, which returns all members when n exceeds their number; an empty
  list gives none without replacement and is refused otherwise.

  MATLAB's 'compact' sampling and the indices of the members drawn from a list are not
  provided.

  Input
    f - SO3Fun, S2Fun, a callable of numbers, or a list of orientations or directions,
        an EBSD map, grains or grain boundary segments
    n - int, the size of the sample
  Output
    ori - orientation, vector3d, ndarray or the class of the list
  Keywords
    resolution - the spacing of the grid sampled, by default 1.5 degree on SO(3) and 1 degree
        on the sphere
    rng - numpy.random.Generator, the random numbers
    withoutReplacement - draw no member of a list twice (a list)
    range - the interval of a callable of numbers

  Example
  Draw a reproducible orientation sample from a model texture, then compare its pole
  positions with the model pole density. No experimental data are needed.

    import numpy as np
    import matplotlib.pyplot as plt
    from mtex import (crystalFrame, orientation, unimodalODF, degree, discreteSample,
                      Miller, plotPF, mtexColorbar)

    cs = crystalFrame('m-3m')
    centre = orientation.byEuler(25*degree, 35*degree, 45*degree, cs)
    odf = unimodalODF(centre, halfwidth=12*degree)
    sample = discreteSample(odf, 300, rng=np.random.default_rng(9))
    print('Sample size:', sample.size)
    print('Median angle to centre (degrees):', round(float(np.median(sample.angle(centre))/degree), 2))
    h = Miller(1, 0, 0, cs)
    plotPF(odf, h, antipodal=True, resolution=5*degree)
    mtexColorbar(title='Model pole density (mrd)')
    plotPF(sample, h, antipodal=True, markerSize=3)
    plt.show()

  The first figure is the model density; the second contains sampled orientation poles.
  The seed makes this draw reproducible. Sampling from a density differs from selecting members
  of an existing list, where withoutReplacement=True prevents selecting the same member twice.

  See also
    SO3Fun.calcComponents, calcDensity, RandomSampling, S2FunSampling"""
  rng = np.random.default_rng() if rng is None else rng
  n = int(n)
  # a list of orientations or directions, an EBSD map of any phases, grains or boundary segments is
  # sampled from itself, MATLAB's discreteSample(ori, n) and discreteSample(ebsd, n)
  if isinstance(f, MtexArray) or hasattr(f, "CSList"):
    flat = f.flatten() if isinstance(f, MtexArray) else f
    if flat.size == 0 and n > 0 and not withoutReplacement:
      raise ValueError("an empty list has no members to draw")
    k = rng.choice(flat.size, size=min(n, flat.size) if withoutReplacement else n, replace=not withoutReplacement)
    return flat[k]
  if isinstance(f, S2Fun):
    return _discreteSampleS2(f, n, 1 * degree if resolution is None else resolution, rng)
  # any other callable that is no function on SO(3) is a density of real numbers, sampled on
  # 10000 points of range
  if callable(f) and not isinstance(f, SO3Fun):
    x = np.linspace(range[0], range[1], 10000)
    p = np.maximum(np.real(np.asarray(f(x))).astype(float).ravel(), 0)
    return x[rng.choice(len(x), size=n, p=p / p.sum())]
  if isinstance(f, SO3FunComposition) and all(isinstance(t, SO3FunRBF) for t in f.components):
    m = np.array([t.mean() for t in f.components])
    counts = np.bincount(rng.choice(len(m), size=n, p=m / m.sum()), minlength=len(m))
    parts = [discreteSample(t, c, rng=rng).data.reshape(-1, 4) for t, c in zip(f.components, counts) if c > 0]
    return _ori(np.concatenate(parts) if parts else np.zeros((0, 4)), f)
  L, R = _framesOf(f)
  if isinstance(f, SO3FunRBF):
    omega = np.linspace(0, np.pi, 4001)
    pdf = f.psi(np.cos(omega / 2)) * np.sin(omega / 2) ** 2
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    probs = np.concatenate([f.weights * f.psi.mean(), [f.c0]])
    cs = f.center.data.reshape(-1, 4)
    if backend() == "numba":
      return _ori(rbfSample(n, rng, cs, probs, omega, cdf, L, R, f.antipodal), f)
    k = rng.choice(len(probs), size=n, p=probs / probs.sum())
    out = rotation.rand(n, rng=rng).data
    fromCentre = k < len(cs)
    w = omega[np.minimum(np.searchsorted(cdf, rng.random(n)), len(omega) - 1)]
    turned = rotation(cs[k[fromCentre]]) * rotation.byAxisAngle(vector3d.rand(int(fromCentre.sum()), rng=rng), w[fromCentre])
    out[fromCentre] = turned.data
  elif isinstance(f, SO3FunBingham):
    out = binghamSample(f.kappa, f.A, n, rng)
  else:
    res = 1.5 * degree if resolution is None else resolution
    g = equispacedSO3Grid(f.CS, f.SS, resolution=res)
    p = np.maximum(np.real(f(g)).ravel(), 0)
    idx = rng.choice(g.size, size=n, p=p / p.sum())
    w = 1.5 * res * np.cbrt(rng.random(n))
    out = (rotation.byAxisAngle(vector3d.rand(n, rng=rng), w) * rotation(g.data.reshape(-1, 4)[idx])).data
  # an exact sample of a function with exchange symmetry takes the inverse with probability one half
  if isinstance(f, (SO3FunRBF, SO3FunBingham)) and f.antipodal:
    flip = rng.random(n) < 0.5
    out[flip] = rotation(out[flip]).inv().data
  left = rotation(L[rng.integers(len(L), size=n)])
  right = rotation(R[rng.integers(len(R), size=n)])
  return _ori((left * rotation(out) * right).data, f)


def _discreteSampleS2(f, n, resolution, rng):
  fr, antipodal = f.frame, f.antipodal
  g = equispacedS2Grid(resolution=resolution) if fr is None else equispacedS2Grid(fr, resolution=resolution, antipodal=antipodal)
  cand = g.data.reshape(-1, 3)
  p = np.maximum(np.real(f(vector3d(cand))).ravel(), 0)
  idx = rng.choice(len(cand), size=n, p=p / p.sum())
  v = rotation.byAxisAngle(vector3d.rand(n, rng=rng), 1.5 * resolution * np.sqrt(rng.random(n))) * vector3d(cand[idx])
  if fr is not None:
    els = (fr.sym.Laue() if antipodal else fr.sym).rot.flatten()
    v = els[rng.integers(els.size, size=n)] * v
  return vector3d(v.data, frame=fr, antipodal=antipodal)


# ------------------------------------------------------------------ kernel selection and specimen centering


def calcKernel(o, method="UCV", samplingSize=1000, rng=None, *, groups=None, weights=None, bandwidth=None, halfwidths=None,
               noiseFloor=True):
  """the kernel a density estimate of orientations or directions should use, a de la Vallee
  Poussin kernel of the halfwidth chosen from the sample

    psi = calcKernel(ori)                                     # by unbiased cross validation of the degree energies
    psi = calcKernel(ebsd)                                    # a map, folded by its grain ids after calcGrains
    psi = calcKernel(ori, 'conservative', groups=grainId)    # the conservative choice, pixels grouped by grain
    psi = calcKernel(ori, 'klcv', samplingSize=500)          # Kullback-Leibler cross validation, as MATLAB's default
    psi = calcKernel(v, noiseFloor=False)                    # an S2 kernel for directions, no lower bound

  The method chooses the halfwidth:

  - 'UCV', the default: the halfwidth that minimises an unbiased estimate of the error of the
  density, computed from the degree energies of the sample. Orientations and directions.
  - 'conservative': the same error with lower confidence bounds on the energies, which gives a
  wider kernel that does not undersmooth. Orientations and directions.
  - 'klcv': Kullback-Leibler cross validation of ten halfwidths, MATLAB's default.
  - 'ruleOfThumb': from the distances of every orientation to its nearest neighbour.
  - 'magicRule': from the sample size and the symmetries alone.

  'UCV' and 'conservative' never go below the halfwidth at which the noise of the sample would
  show as false peaks, unless noiseFloor=False.

  Neighbouring pixels of one grain are no independent draws. For a map, 'UCV' and
  'conservative' therefore group the pixels by their grain, which needs the grain ids that
  calcGrains writes into the map. 'klcv', 'ruleOfThumb' and 'magicRule' refuse orientations
  taken from a map; select from the grain mean orientations instead.

  Above 5000 orientations, 'klcv', 'ruleOfThumb' and 'magicRule' score a random subsample and
  scale the result to the whole sample. MATLAB's 'LSCV' and 'BCV' are not provided.

  Input
    o - orientation, vector3d, or EBSD
    method - 'UCV', 'conservative', 'klcv', 'ruleOfThumb' or 'magicRule', in any case
  Output
    psi - SO3DeLaValleePoussinKernel, or S2DeLaValleePoussinKernel for directions
  Keywords
    samplingSize - the number of orientations 'klcv' scores
    rng - numpy.random.Generator of the subsamples; the energies use a fixed seed by default
    groups - the grain id of every orientation, by which the energies fold
    weights - the weights of the sample
    bandwidth - the degree up to which the energies are computed, by default 64 on SO(3)
        (doubled when the choice lies at the edge), 256 on the sphere, 32 for 'conservative'
    halfwidths - the candidate halfwidths of the energies
    noiseFloor - bound the halfwidth below by the noise of the energies
  See also
    calcDensity, EBSD2ODF, DensityEstimation"""
  from .halfwidth import degreeEnergies, selectHalfwidth, ucvAtEdge
  from .kernelfunctions import S2DeLaValleePoussinKernel
  if not isinstance(o, (quaternion, vector3d)) and hasattr(o, "orientations"):
    groups = o.prop.get("grainId") if groups is None and "grainId" in o.prop else groups
    o = o.orientations
  directions = isinstance(o, vector3d)
  # the method however it is capitalised, MATLAB's 'RuleOfThumb' included
  method = {"klcv": "klcv", "ruleofthumb": "ruleOfThumb", "magicrule": "magicRule", "conservative": "conservative",
            "ucv": "UCV"}.get(str(method).lower(), method)
  if method in ("conservative", "UCV"):
    # the selection draws subsamples and folds; a fixed seed makes the default reproducible
    rng = np.random.default_rng(0) if rng is None else rng
    L = (32 if method == "conservative" else (256 if directions else 64)) if bandwidth is None else int(bandwidth)
    # UCV needs no errors of the energies, so independent points take one transform
    folds = 1 if method == "UCV" and groups is None else 8
    E = degreeEnergies(o, L, weights=weights, groups=groups, folds=folds, rng=rng)
    # on SO(3) UCV starts at bandwidth 64 and doubles once when its choice sits on the narrowest
    # halfwidth that bandwidth resolves; 128 resolves every candidate
    if method == "UCV" and bandwidth is None and not directions and ucvAtEdge(E, halfwidths):
      E = degreeEnergies(o, 2 * L, weights=weights, groups=groups, folds=folds, rng=rng)
    h = selectHalfwidth(E, method, halfwidths, noiseFloor=noiseFloor)
    return S2DeLaValleePoussinKernel(halfwidth=h) if directions else SO3DeLaValleePoussinKernel(halfwidth=h)
  rng = np.random.default_rng() if rng is None else rng
  if directions:
    raise ValueError(f"the kernel of directions is selected by 'UCV' or 'conservative', not {method!r}")
  # the cross validation and the rules count pixels as independent draws
  if isinstance(o, orientation) and o.notIID:
    raise ValueError(f"{method!r} counts the pixels of a map as independent draws, which neighbouring pixels of one "
                     "grain are not: select with 'conservative' or 'UCV' and the grain ids, or from the grain mean "
                     "orientations")
  o = o.flatten()
  N = o.size
  if N == 0:
    raise ValueError("no orientations to select a kernel from")
  # above five thousand a subsample is scored and the concentration scaled for the whole
  fak = 1.0
  if N > 5000:
    fak = (N / 5000) ** (1 / 7)
    o = o[rng.permutation(N)[:5000]]
  n = o.size
  nL = o.SS.sym.properGroup().rotations().size
  nR = o.CS.sym.properGroup().rotations().size
  if method == "magicRule":
    return SO3DeLaValleePoussinKernel(3 * (nL * nR * N) ** (2 / 7))
  if method == "ruleOfThumb":
    nearest = np.empty(n)
    for k in range(n):
      d = o.angle(o[k]).ravel()
      d[k] = np.pi
      nearest[k] = np.min(np.where(d < 0.005, np.pi, d))
    res = np.quantile(nearest, 0.9)
    psi = SO3DeLaValleePoussinKernel(halfwidth=max(1.5 * res, 2 * degree))
    return SO3DeLaValleePoussinKernel(fak * psi.kappa)
  if method == "klcv":
    candidates = [SO3DeLaValleePoussinKernel(halfwidth=30 * degree / 2 ** (k / 3)) for k in range(1, 11)]
    sN = min(n, samplingSize)
    t = np.stack([np.cos(o.angle(o[j]).ravel() / 2) for j in range(sN)])
    scores = []
    for psi in candidates:
      vals = psi(t)
      vals[np.arange(sN), np.arange(sN)] = 0
      fj = np.sum(vals, axis=1) / n / nR
      scores.append(np.mean(np.log(np.maximum(fj, np.finfo(float).tiny))))
    best = candidates[int(np.argmax(scores))]
    return SO3DeLaValleePoussinKernel(fak * best.kappa)
  raise ValueError(f"the kernel selection method is 'UCV', 'conservative', 'klcv', 'ruleOfThumb' or 'magicRule', not {method!r}")


def centerSpecimen(odf, v0=None, *args, delta=45 * degree, bandwidth=16, verbose=False):
  """an orientation density of an assumed orthotropic texture rotated so that its two fold axes
  lie on the specimen axes

    odr, rot, v1, v2 = centerSpecimen(odf)                                   # starting at the x axis
    odr, rot, v1, v2 = centerSpecimen(odf, yvector, delta=30*degree, verbose=True)  # another start, a smaller search

  The function finds the two fold axes of a texture that should have orthorhombic specimen
  symmetry but was measured slightly turned, and turns it back. A candidate axis is scored
  by how much the density changes when turned by 180 degree about it; the fit verbose
  reports is one minus that change, relative to the norm of the density.

  The first axis is searched on a cap of opening delta about v0 and refined, the second on
  the fan perpendicular to the first. The rotation carries the two axes onto the specimen
  axes nearest to them; v2 is returned made exactly perpendicular to v1.

  The search and the result use the harmonic series of the given bandwidth, so odr is
  truncated to it. MATLAB's flags 'Fourier' and 'silent' change nothing; its 'SO3Grid' and
  'resolution' are not taken.

  Input
    odf - SO3Fun
    v0 - vector3d, the start of the search for the first axis, the x axis by default
  Output
    odr - SO3FunHarmonic, the rotated density
    rot - rotation, the rotation applied, odr = rotate(odf, rot)
    v1, v2 - vector3d, the two fold axes found
  Keywords
    delta - the opening angle of the cap about v0 searched for the first axis
    bandwidth - the degree of the harmonic series the search compares
    verbose - report the fit of every step
  See also
    SO3Fun.rotate, DetectionOfSampleSymmetry"""
  # MATLAB's flags 'Fourier' and 'silent' change nothing here: the search runs on the harmonic form
  if isinstance(v0, str):
    v0 = None
  from ..geometry.frames import specimenFrame
  f0 = SO3FunHarmonic(odf, bandwidth=bandwidth)
  # the specimen group is what is being looked for, so the series drops it
  f = SO3FunHarmonic(f0._fhat, f0.CS, specimenFrame.default, real=f0.isReal, antipodal=f0.antipodal, symmetrise=False, _internal=True)
  scale = f.norm()
  v0 = xvector if v0 is None else v0

  def mismatch(vs):
    return np.array([(f - f.rotate(rotation.byAxisAngle(vector3d(v), np.pi))).norm() / scale for v in vs.data.reshape(-1, 3)])

  def best(cands):
    vals = mismatch(cands)
    k = int(np.argmin(vals))
    if verbose:
      print(f"  fit: {(1 - vals[k]) * 100:.1f}%")
    return vector3d(cands.data.reshape(-1, 3)[k])

  def cap(center, opening, resolution):
    return rotation.map(zvector, center) * vector3d(equispacedS2Grid(resolution=resolution, maxTheta=opening).data)

  if verbose:
    print("  searching for a first two fold symmetry axis")
  v1 = best(cap(v0, delta, 5 * degree))
  v1 = best(cap(v1, 2.5 * degree, 0.75 * degree))
  if verbose:
    print("  searching for a second two fold symmetry axis")
  o = v1.orth()

  def fan(rng_):
    return rotation.byAxisAngle(v1, rng_) * o

  v2 = best(fan(np.arange(-45, 46, 5) * degree))
  o = v2
  v2 = best(fan(np.arange(-5, 5.5, 0.5) * degree))
  # the rotation carrying the two axes onto the specimen axes nearest to them
  axes = np.concatenate([np.eye(3), -np.eye(3)])
  def closest(u):
    return axes[int(np.argmax(axes @ u))]
  u1, u2 = v1.data.reshape(3), v2.data.reshape(3)
  a1, a2 = closest(u1), closest(u2)
  w2 = u2 - np.dot(u2, u1) * u1
  w2 /= np.linalg.norm(w2)
  M = np.stack([a1, a2, np.cross(a1, a2)], axis=1) @ np.stack([u1, w2, np.cross(u1, w2)], axis=1).T
  rot = rotation.byMatrix(M)
  return f0.rotate(rot), rot, v1, vector3d(w2)


# ------------------------------------------------------------------ the methods of the function kinds


install(SO3_KINDS + S2_KINDS, max_, "max")
install(SO3_KINDS + S2_KINDS, min_, "min")
for _f in (entropy, textureIndex, discreteSample):
  install(SO3_KINDS + S2_KINDS, _f)
install(SO3_KINDS + S2_KINDS, volume)
for _f in (fibreVolume, calcComponents, calcMDF, calcAngleDistribution, calcAxisDistribution, calcMIndex, calcError, centerSpecimen):
  install(SO3_KINDS, _f)

__all__ = ["textureIndex", "entropy", "volume", "fibreVolume", "calcMDF", "calcAngleDistribution", "calcMIndex",
           "calcAxisDistribution", "calcError", "calcComponents", "discreteSample", "calcKernel", "centerSpecimen"]
