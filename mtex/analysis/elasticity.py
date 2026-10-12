# elasticity: Hooke's law, the Christoffel tensor and the wave velocities, the moduli and
# the Hashin-Shtrikman bounds (MATLAB TensorAnalysis/@stiffnessTensor, @complianceTensor;
# docs/plans/functions.md P4d)
#
#   sigma = C * eps                                      Hooke's law, a stressTensor
#   ChristoffelTensor(C, n), vp, vs1, vs2, pp, ps1, ps2 = velocity(C, n, density=rho)
#   YoungsModulus(C, v), YoungsModulus(C)                the value, and the function on the sphere
#   bulkModulus(C), bulkModulus(C, 'Hill'), shearModulus(C, h, u), PoissonRatio(C, x, y)
#   HashinShtrikmanModulus(C, K0, G0), hashinShtrikmanBounds(C)
#
# Every modulus is a contraction of the compliance with one or two directions; a stiffness
# is taken through its inverse. Voigt and Reuss are the isotropic bounds of the polycrystal,
# the Hashin-Shtrikman estimate of a reference medium is a lower bound where C - C0 is
# positive definite and an upper bound where it is negative definite, and the best bounds
# sit on the boundary of those references.
from __future__ import annotations

import math
from collections import namedtuple

import numpy as np

from ..geometry._array import MtexArray
from ..geometry.arguments import refuseFlags
from ..geometry.frames import referenceFrame
from ..geometry.vector3d import vector3d
from ..functions.s2funclass import S2Fun
from ..tensors.tensor import (ChristoffelTensor, EinsteinSum, Kelvin, Voigt, complianceTensor, stiffnessTensor, _axesIn,
                              _inFrameOf)
from ..tensors.kernels import eigh3
from ..geometry.install import install

Bounds = namedtuple("Bounds", ["voigt", "reuss", "hill"])
Velocities = namedtuple("Velocities", ["vp", "vs1", "vs2", "pp", "ps1", "ps2"])
HashinShtrikman = namedtuple("HashinShtrikman", ["lower", "upper"])
Moduli = namedtuple("Moduli", ["K", "G"])


def _compliance(T):
  if isinstance(T, complianceTensor):
    return T
  if isinstance(T, stiffnessTensor):
    return T.inv()
  raise TypeError("the moduli are of a stiffnessTensor or a complianceTensor")


def _stiffness(T):
  if isinstance(T, stiffnessTensor):
    return T
  if isinstance(T, complianceTensor):
    return T.inv()
  raise TypeError("a stiffnessTensor or a complianceTensor is needed")


# a direction in the frame of the tensor, as a unit vector; a bare vector is taken as is
def _unit(T, v):
  return _inFrameOf(T, v).normalize()


def _pick(bounds, which):
  if which is None or which == "Voigt":
    return bounds.voigt
  if which == "Reuss":
    return bounds.reuss
  if which == "Hill":
    return bounds.hill
  if which == "all":
    return bounds
  raise ValueError("the bound is 'Voigt', 'Reuss', 'Hill' or 'all'")


# a function on the sphere of a loading direction, in the frame the tensor is written in;
# one with a fixed second direction is not invariant under the group and is written in
# the frame without it
def _sphericalFunction(fun, T, invariant=True):
  from ..functions.handles import S2FunHandle
  fr = T.frame if invariant else T.frame.stripSym()
  return S2FunHandle(lambda v: np.asarray(fun(vector3d(v.data))).reshape(v.shape), fr, antipodal=True, bandwidth=32, symmetrise=False)


# ------------------------------------------------------------------ the Christoffel tensor and the waves

# E_ik = C_ijkl n_j n_l, whose eigenvalues over the density are the squared wave velocities
# in the direction n; ChristoffelTensor(C, n) is the class constructor
def _christoffel(C, n):
  C = _stiffness(C)
  return _christoffelMatrix(C, _unit(C, n)), C.frame, C.density


# E_ik = C_ijkl u_j u_l; for one tensor the products u_j u_l against C read as a 9 x 9 matrix
def _christoffelMatrix(C, u):
  if C.ndim:
    return EinsteinSum(C, [1, -1, 2, -2], u, [-1], u, [-2]).M
  x = u.data
  Cp = np.transpose(C.M, (0, 2, 1, 3)).reshape(9, 9)
  return ((x[..., :, None] * x[..., None, :]).reshape(x.shape[:-1] + (9,)) @ Cp.T).reshape(x.shape[:-1] + (3, 3))


def velocity(C, n=None, *args, density=None, harmonic=False, bandwidth=48):
  """the elastic wave velocities and their polarisations from the Christoffel tensor

    vp, vs1, vs2, pp, ps1, ps2 = velocity(C, n, density=rho)   in the direction n, or at a list of them
    vp, vs1, vs2, pp, ps1, ps2 = velocity(C, density=rho)      as functions on the sphere and axis fields
    vp, vs1, vs2, pp, ps1, ps2 = velocity(C, 'harmonic')       as harmonic expansions of bandwidth 48

  A stiffness in GPa and a density in g/cm^3 give velocities in km/s; the density stored
  on the tensor is used when none is given, and MATLAB's velocity(C, n, rho) passes it
  third.
  """
  C = _stiffness(C)
  if isinstance(n, str):
    n, args = None, (n,) + args
  refuseFlags(args, "velocity")
  numbers = [a for a in args if not isinstance(a, str)]
  if density is not None:
    rho = density
  elif numbers:
    rho = numbers[0]
  else:
    rho = 1.0 if C.density is None else C.density
  if not np.isfinite(rho) or rho <= 0:
    raise ValueError("the density has to be finite and positive")
  if n is None:
    return _velocityHarmonics(C, rho, bandwidth) if harmonic else _velocityFields(C, rho)
  E = _christoffelMatrix(C, _unit(C, n))
  lam, vecs = eigh3(E)
  v = np.sqrt(np.maximum(lam, 0) / rho)
  def axes(k):
    return _axesIn(np.moveaxis(vecs, -1, -2)[..., k, :], C.frame)
  return Velocities(v[..., 2], v[..., 1], v[..., 0], axes(2), axes(1), axes(0))


def _velocityFields(C, rho):
  from ..functions.s2axisfield import S2AxisFieldHandle
  def speed(k):
    return _sphericalFunction(lambda v: velocity(C, v, density=rho)[k], C)
  def field(k):
    return S2AxisFieldHandle(lambda v: velocity(C, vector3d(v.data), density=rho)[k].data, C.frame, bandwidth=32)
  return Velocities(speed(0), speed(1), speed(2), field(3), field(4), field(5))


# the speeds and the polarisations sampled once on a quadrature grid and expanded
def _velocityHarmonics(C, rho, L):
  from ..functions.quadrature import quadratureS2Grid
  from ..functions.s2axisfield import S2AxisFieldHarmonic
  from ..functions.s2fun import S2FunHarmonic
  g = quadratureS2Grid(L)
  x = vector3d(g.nodes().data.reshape(-1, 3))
  w = velocity(C, x, density=rho)
  def speed(k):
    return S2FunHarmonic.quadrature(g, w[k], C.frame, bandwidth=L, antipodal=True)
  def field(k):
    return S2AxisFieldHarmonic(x, w[k].data, C.frame, bandwidth=L, weights=g.weights)
  return Velocities(speed(0), speed(1), speed(2), field(3), field(4), field(5))


def energyVector(C, n, speed=None, polarization=None, *, density=None):
  """the energy velocity of a lossless wave, C_ijkl p_j n_k p_l / (rho v), whose projection on
  the propagation direction is the phase speed; with a speed function and a polarisation
  field the vector field on the sphere"""
  C = _stiffness(C)
  rho = C.density if density is None else density
  if rho is None or not np.isfinite(rho) or rho <= 0:
    raise ValueError("the density has to be finite and positive")
  if n is None or isinstance(speed, S2Fun):
    from ..functions.s2vectorfield import S2VectorFieldHandle
    def field(v):
      x = vector3d(v.data, frame=C.frame)
      return energyVector(C, vector3d(v.data), np.real(speed(x)), polarization(x), density=rho).data
    return S2VectorFieldHandle(field, C.frame, bandwidth=32)
  u = _unit(C, n)
  a = _unit(C, polarization)
  out = EinsteinSum(C, [1, -2, -3, -4], a, [-2], u, [-3], a, [-4])
  data = out.data if isinstance(out, MtexArray) else np.asarray(out)
  v = vector3d(data / (rho * np.asarray(speed, dtype=float)[..., None]), frame=C.frame)
  v.dispStyle = "xyz"
  return v


# ------------------------------------------------------------------ the moduli

def YoungsModulus(T, v=None):
  """Young's modulus in the direction v, the reciprocal of S_ijkl v_i v_j v_k v_l; without a
  direction the function on the sphere"""
  S = _compliance(T)
  if v is None:
    return _sphericalFunction(lambda x: YoungsModulus(S, x), S)
  # the magnitude takes the direction of any length
  return 1 / S.directionalMagnitude(_inFrameOf(S, v))


def linearCompressibility(T, v=None):
  """the length change per pressure in the direction v, S_ijkk v_i v_j; without a direction
  the function on the sphere"""
  S = _compliance(T)
  if v is None:
    return _sphericalFunction(lambda x: linearCompressibility(S, x), S)
  u = _unit(S, v)
  return EinsteinSum(S, [-1, -2, -3, -3], u, [-1], u, [-2])


def volumeCompressibility(T):
  """the volume change per pressure, S_iikk"""
  return EinsteinSum(_compliance(T), [-1, -1, -2, -2])


def shearModulus(T, h=None, u=None, *args):
  """the shear modulus of the plane h and the direction u, 1 / (4 S_ijkl h_i u_j h_k u_l);
  with the plane alone the function of the direction in it; without both the isotropic
  bound, shearModulus(C, 'Reuss'), 'Hill' or 'all' the others"""
  S = _compliance(T)
  if isinstance(h, str) or (h is None and u is None):
    return _pick(isotropicShearModulus(S), h)
  if u is None:
    hh = _unit(S, h)
    return _sphericalFunction(lambda x: shearModulus(S, hh, x), S, invariant=False)
  a, b = _unit(S, h), _unit(S, u)
  return 0.25 / EinsteinSum(S, [-1, -2, -3, -4], a, [-1], b, [-2], a, [-3], b, [-4])


def PoissonRatio(T, x=None, y=None):
  """the Poisson ratio of the pulling direction x and the transverse direction y, minus the
  transverse over the longitudinal strain; with x alone the function of the transverse
  direction; without both the isotropic ratio of the Voigt bounds"""
  S = _compliance(T)
  if x is None:
    K, G = bulkModulus(S), shearModulus(S)
    return (3 * K - 2 * G) / (2 * (3 * K + G))
  if y is None:
    p = _unit(S, x)
    return _sphericalFunction(lambda v: PoissonRatio(S, p, v), S, invariant=False)
  a, b = _unit(S, x), _unit(S, y)
  return -EinsteinSum(S, [-1, -2, -3, -4], a, [-1], a, [-2], b, [-3], b, [-4]) / EinsteinSum(S, [-1, -2, -3, -4], a, [-1], a, [-2], a, [-3], a, [-4])


def bulkModulus(T, which=None):
  """the bulk modulus of a polycrystal: the Voigt bound, or 'Reuss', 'Hill' or 'all' three"""
  S = _compliance(T)
  KV = EinsteinSum(S.inv(), [-1, -1, -2, -2]) / 9
  KR = 1 / volumeCompressibility(S)
  return _pick(Bounds(KV, KR, (KV + KR) / 2), which)


def isotropicShearModulus(T):
  """the isotropic shear modulus of a polycrystal, the Voigt, Reuss and Hill bounds"""
  S = _compliance(T)
  C = Voigt(S.inv())
  V = Voigt(S)
  GV = ((C[0, 0] + C[1, 1] + C[2, 2]) - (C[0, 1] + C[1, 2] + C[2, 0]) + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15
  GR = 15 / (4 * (V[0, 0] + V[1, 1] + V[2, 2]) - 4 * (V[0, 1] + V[1, 2] + V[2, 0]) + 3 * (V[3, 3] + V[4, 4] + V[5, 5]))
  return Bounds(GV, GR, (GV + GR) / 2)


# ------------------------------------------------------------------ the Hashin-Shtrikman bounds
# in the form of Watt and Peselnick (1980): an isotropic comparison medium C0 = 3 K0 P_h +
# 2 G0 P_d is taken off, R = C - C0, and B = (R^-1 - S0)^-1 is written as (I - R S0)^-1 R so
# it stays finite where R turns singular, which is where the best bounds sit

_e = np.array([1.0, 1, 1, 0, 0, 0])
KELVIN_HYDROSTATIC = np.outer(_e, _e) / 3
KELVIN_DEVIATORIC = np.eye(6) - KELVIN_HYDROSTATIC


# the symmetric residual C - C0 of every reference K0, G0, a 6 x 6 matrix over their shape
def _residual(Ck, K0, G0):
  R = Ck - 3 * np.asarray(K0)[..., None, None] * KELVIN_HYDROSTATIC - 2 * np.asarray(G0)[..., None, None] * KELVIN_DEVIATORIC
  return (R + np.swapaxes(R, -1, -2)) / 2


# the estimate of the residuals R of the references K0, G0, over their shape
def _estimate(R, K0, G0):
  alpha = -3 / (3 * K0 + 4 * G0)
  beta = -3 * (K0 + 2 * G0) / (5 * G0 * (3 * K0 + 4 * G0))
  gamma = (alpha - 3 * beta) / 9
  S0 = np.asarray(beta)[..., None, None] * np.eye(6) + 3 * np.asarray(gamma)[..., None, None] * KELVIN_HYDROSTATIC
  B = np.linalg.solve(np.eye(6) - R @ S0, R)
  sB1 = np.sum(B[..., :3, :3], axis=(-2, -1))
  sB2 = np.trace(B, axis1=-2, axis2=-1)
  B1 = (2 * sB1 - sB2) / 15
  B2 = (3 * sB2 - sB1) / 30
  return Moduli(K0 + (3 * B1 + 2 * B2) / (3 + alpha * (3 * B1 + 2 * B2)), G0 + B2 / (1 + 2 * beta * B2))


def HashinShtrikmanModulus(C, K0, G0):
  """the Hashin-Shtrikman estimate of the isotropic moduli of a polycrystal from a reference
  medium K0, G0, over arrays of references: the bulk and shear moduli, and the definiteness
  of C - C0 (1 where the estimate is a lower bound, -1 an upper one, 0 where it is neither
  and the moduli are NaN)"""
  Ck = Kelvin(_stiffness(C))
  K0, G0 = np.broadcast_arrays(np.asarray(K0, dtype=float), np.asarray(G0, dtype=float))
  khs, ghs = np.full(K0.shape, np.nan), np.full(K0.shape, np.nan)
  R = _residual(Ck, K0, G0)
  lam = np.linalg.eigvalsh(R)
  definite = np.where(np.all(lam > 0, axis=-1), 1, np.where(np.all(lam < 0, axis=-1), -1, 0))
  # the estimate only where the residual is definite, where it is a bound
  ok = definite != 0
  khs[ok], ghs[ok] = _estimate(R[ok], K0[ok], G0[ok])
  if K0.ndim == 0:
    return float(khs), float(ghs), int(definite)
  return khs, ghs, definite


def _bisect(pred, lo, hi, tol=1e-10):
  while hi - lo > tol * max(1, abs(hi)):
    mid = (lo + hi) / 2
    if pred(mid):
      lo = mid
    else:
      hi = mid
  return lo


def _goldenSection(f, lo, hi, tol=1e-10):
  phi = (math.sqrt(5) - 1) / 2
  a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
  fa, fb = f(a), f(b)
  while hi - lo > tol * max(1, abs(hi)):
    if fa < fb:
      lo, a, fa = a, b, fb
      b = lo + phi * (hi - lo)
      fb = f(b)
    else:
      hi, b, fb = b, a, fa
      a = hi - phi * (hi - lo)
      fa = f(a)
  return (lo + hi) / 2


def hashinShtrikmanBounds(C):
  """the Hashin-Shtrikman bounds of the isotropic moduli, the tightest pair for a
  quasihomogeneous polycrystal: (lower=(K, G), upper=(K, G)), each the extreme estimate
  over the references whose residual is semidefinite, found on the boundary of that
  convex set by a golden section over the shear and a bisection over the bulk modulus"""
  C = _stiffness(C)
  Ck = Kelvin(C)
  KV = bulkModulus(C)
  GV = isotropicShearModulus(C).voigt
  top = float(np.max(np.linalg.eigvalsh((Ck + Ck.T) / 2)))
  def positive(K0, G0):
    return np.min(np.linalg.eigvalsh(_residual(Ck, K0, G0))) >= 0
  def negative(K0, G0):
    return np.max(np.linalg.eigvalsh(_residual(Ck, K0, G0))) <= 0
  def estimate(K0, G0):
    return _estimate(_residual(Ck, K0, G0), K0, G0)
  # the lower bound: R >= 0 admits the shear moduli up to Gmax and, for each, the bulk moduli up to Kmax
  Gmax = _bisect(lambda G0: positive(0, G0), 0, GV)
  def Kmax(G0):
    return _bisect(lambda K0: positive(K0, G0), 0, KV)
  def lower(G0):
    return estimate(Kmax(G0), G0)
  Glo, Ghi = 1e-6 * GV, Gmax
  KL = lower(_goldenSection(lambda G0: lower(G0).K, Glo, Ghi)).K
  GL = lower(_goldenSection(lambda G0: lower(G0).G, Glo, Ghi)).G
  # the upper bound: R <= 0 admits the shear moduli from Gmin on and, for each, the bulk moduli from Kmin on
  Gmin = -_bisect(lambda G0: negative(10 * top, -G0), -top / 2, -GV)
  def Kmin(G0):
    return -_bisect(lambda K0: negative(-K0, G0), -10 * top, 0)
  def upper(G0):
    return estimate(Kmin(G0), G0)
  KU = upper(_goldenSection(lambda G0: -upper(G0).K, Gmin, top / 2)).K
  GU = upper(_goldenSection(lambda G0: -upper(G0).G, Gmin, top / 2)).G
  return HashinShtrikman(Moduli(KL, GL), Moduli(KU, GU))


# ------------------------------------------------------------------ installation

for _f in (ChristoffelTensor, velocity, energyVector, YoungsModulus, linearCompressibility, volumeCompressibility, shearModulus,
           PoissonRatio, bulkModulus, HashinShtrikmanModulus, hashinShtrikmanBounds):
  install((stiffnessTensor, complianceTensor), _f)

__all__ = ["velocity", "energyVector", "YoungsModulus", "linearCompressibility", "volumeCompressibility",
           "shearModulus", "PoissonRatio", "bulkModulus", "isotropicShearModulus", "HashinShtrikmanModulus", "hashinShtrikmanBounds"]
