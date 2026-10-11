# texture and tensors: the average of a tensor over an ODF, orientations or a map, the
# Lankford coefficients, the Kearns factor, the shear banding function and the transport
# of a texture (MATLAB SO3Fun/calcTensor, EBSD/calcTensor, orientation/calcTensor,
# ODFAnalysis/calcLankford, calcKearnsFactor, SO3FunSBF, doEulerStep; docs/plans/functions.md P4e)
#
#   CV = calcTensor(odf, C), CH = calcTensor(odf, C, 'Hill'), CV, CR, CH = calcTensor(ebsd, C1, C2, 'all')
#   R, M, minM = calcLankford(ori, sS, theta=, rho=, weights=)
#   calcKearnsFactor(odf, h, N), SO3FunSBF(sS, E), doEulerStep(spin, ori, n)
#
# The average of a single crystal tensor over an ODF is the integral of the rotated tensor
# against the density: every component is a polynomial of degree R in the rotation, so
# only the coefficients of the ODF up to R contribute and a quadrature rule exact for
# degree 2R integrates the product exactly.
from __future__ import annotations

import math

import numpy as np

from ..functions.so3funclass import SO3Fun
from ..geometry.frames import crystalFrame, referenceFrame, specimenFrame
from ..geometry.orientation import orientation
from ..geometry.quaternion import quaternion, rotation
from ..geometry.slipsystem import slipSystem
from ..geometry.vector3d import vector3d
from .taylor import _taylorMany, TaylorModel, calcTaylor
from ..tensors.tensor import strainTensor, tensor
from ..geometry.install import install
from ..geometry.display import framesLabel
from ..geometry.display import aligned

AVERAGES = ("Voigt", "Reuss", "Hill", "all")


def _average(args, average):
  strings = [a for a in args if isinstance(a, str)]
  which = strings[0] if strings else average
  if which not in AVERAGES:
    raise ValueError("the average is 'Voigt', 'Reuss', 'Hill' or 'all'")
  return which


def _weights(w, n):
  if n == 0:
    raise ValueError("a tensor average needs at least one orientation")
  if w is None:
    return np.full(n, 1 / n)
  w = np.asarray(w, dtype=float).ravel()
  if w.size != n:
    raise ValueError("one weight per orientation is needed")
  if not (np.all(np.isfinite(w)) and np.all(w >= 0) and np.sum(w) > 0):
    raise ValueError("the weights have to be finite, nonnegative and of positive sum")
  return w / np.sum(w)


def _isMap(x):
  return hasattr(x, "phaseId") and hasattr(x, "CSList") and hasattr(x, "rotations")


# ------------------------------------------------------------------ the average of a tensor

def calcTensor(x, *args, average="Voigt", weights=None):
  """the texture average of a single crystal tensor

    calcTensor(odf, T)                       over an ODF, the Voigt average
    calcTensor(odf, T, 'Reuss'), 'Hill'      the inverse averaged and inverted, and the mean of the two
    calcTensor(ori, T, weights=w)            over a list of orientations
    calcTensor(ebsd, T1, T2, 'all')          over a map, one tensor per phase, the triple Voigt, Reuss, Hill
  """
  which = _average(args, average)
  tensors = [a for a in args if isinstance(a, tensor)]
  if _isMap(x):
    return _calcTensorMap(x, tensors, which, weights)
  if not tensors:
    raise TypeError("calcTensor needs a tensor")
  T = tensors[0]
  if which == "all":
    V = calcTensor(x, T, "Voigt", weights=weights)
    R = calcTensor(x, T, "Reuss", weights=weights)
    return V, R, (V + R) / 2
  if which == "Reuss":
    return calcTensor(x, T.inv(), "Voigt", weights=weights).inv()
  if which == "Hill":
    return (calcTensor(x, T, "Voigt", weights=weights) + calcTensor(x, T, "Reuss", weights=weights)) / 2
  if isinstance(x, orientation):
    w = _weights(weights, x.size)
    fast = T._turnedMean(x.flatten(), np.asarray(w, dtype=float).ravel())
    return fast if fast is not None else (x.flatten() * T).mean(weights=w)
  if isinstance(x, SO3Fun):
    return _calcTensorODF(x, T)
  raise TypeError("calcTensor averages over an ODF, a list of orientations or a map")


# the quadrature of degree 2R of the rotated tensor against the ODF truncated to degree R
def _calcTensorODF(f, T):
  from ..functions.quadrature import quadratureSO3Grid
  from ..functions.so3fun import SO3FunHarmonic
  R = T.rank
  g = quadratureSO3Grid(2 * R)
  nodes = orientation(rotation(g.nodes().data), f.CS, f.SS)
  fh = SO3FunHarmonic(f, bandwidth=R)
  vals = np.real(fh(nodes))
  Ts = nodes * T
  out = np.einsum("n,n,nk->k", g.weights, vals, Ts.data.reshape(nodes.size, -1))
  return T._as(type(T), out, f.SS)


# a map: every indexed phase takes the tensor of its crystal, matched by frame, mineral and
# Laue class, the tensors rotated by the measurements and averaged with the densities
def _calcTensorMap(ebsd, tensors, which, weights):
  phaseId = np.asarray(ebsd.phaseId)
  rots = ebsd.rotations
  indexed = np.asarray(ebsd.isIndexed) if hasattr(ebsd, "isIndexed") else phaseId > 0
  ss = getattr(rots, "frame", None)
  ss = specimenFrame.default if not isinstance(ss, referenceFrame) else ss
  w = None if weights is None else np.asarray(weights, dtype=float).ravel()
  phases = []
  for p in np.unique(phaseId[indexed]):
    cs = ebsd.CSList[int(p)]
    T = next((t for t in tensors if _sameCrystal(t.frame, cs)), None)
    if T is None:
      raise ValueError(f"no tensor for the phase {getattr(cs, 'mineral', cs.char())}")
    mask = phaseId == p
    phases.append((T, orientation(rotation(rots.data[mask]), cs, ss), np.full(int(mask.sum()), 1.0) if w is None else w[mask]))
  ws = np.concatenate([x[2] for x in phases])
  _weights(ws, len(ws))
  total = np.sum(ws)
  # the weighted sums of the tensors turned by every phase's orientations, and of their inverses,
  # since the inverse of a turned tensor is the turned inverse: no tensor of a pixel is formed
  # and none is inverted
  def turned(inverse):
    acc = 0.0
    for T, o, wp in phases:
      Tp = T.inv() if inverse else T
      m = Tp._turnedMean(o, wp)
      acc = acc + (m.M.ravel() if m is not None else (o * Tp).data.reshape(o.size, -1).T @ (wp / np.sum(wp))) * np.sum(wp)
    # the mean of the inverses is of the inverse's class, a compliance for a stiffness
    T0 = tensors[0].inv() if inverse else tensors[0]
    return T0._as(type(T0), acc / total, ss)
  dens = np.concatenate([np.full(o.size, np.nan if T.density is None else T.density) for T, o, _ in phases])
  density = float(ws @ dens / total) if np.all(np.isfinite(dens)) else None
  voigt = turned(False)
  voigt.density = density
  if which == "Voigt":
    return voigt
  reuss = turned(True).inv()
  reuss.density = density
  if which == "Reuss":
    return reuss
  hill = (voigt + reuss) / 2
  hill.density = density
  return hill if which == "Hill" else (voigt, reuss, hill)


def _sameCrystal(a, b):
  if a is b:
    return True
  if not (isinstance(a, crystalFrame) and isinstance(b, crystalFrame)):
    return False
  try:
    if a.isclose(b):
      return True
  except Exception:
    pass
  return (getattr(a, "mineral", None) == getattr(b, "mineral", None) and a.Laue().pointGroup == b.Laue().pointGroup
          and np.allclose(a.abc, b.abc, rtol=1e-2) and np.allclose(a.abg, b.abg, atol=1e-2))


# ------------------------------------------------------------------ the sheet quantities

def calcLankford(x, sS, *args, theta=None, rho=None, RD=None, ND=None, weights=None, bandwidth=16):
  """the Lankford coefficients of a sheet texture: for every angle theta to the rolling
  direction the normalised plastic work is minimised over the strain ratio rho, and
  R = rho/(1 - rho) at the minimum; returns R per angle, the work over rho and theta, and
  its minimum per angle

    R, M, minM = calcLankford(ori, sS, theta=, rho=, RD=xvector, ND=zvector, weights=)
    R, M, minM = calcLankford(odf, sS, theta=, rho=)
  """
  from .. import geometry as _g
  th = np.linspace(0, np.pi / 2, 19) if theta is None else np.asarray(theta, dtype=float).ravel()
  rh = np.linspace(0, 1, 11) if rho is None else np.asarray(rho, dtype=float).ravel()
  for a in args:
    if isinstance(a, np.ndarray):
      th = a.ravel()
  model = sS if isinstance(sS, TaylorModel) else TaylorModel(sS)
  rd = (vector3d(1, 0, 0) if RD is None else vector3d(RD)).normalize().data.reshape(3)
  nd = (vector3d(0, 0, 1) if ND is None else vector3d(ND)).normalize().data.reshape(3)
  if abs(rd @ nd) > 1e-8:
    raise ValueError("the rolling and the normal direction of a sheet are orthogonal")
  td = np.cross(rd, nd)
  M = np.zeros((rh.size, th.size))
  if isinstance(x, orientation):
    o = x.flatten()
    w = _weights(weights, o.size)
    Q = rotation(o.data).matrix()
    for k, t in enumerate(th):
      r = rotation.byAxisAngle(vector3d(nd), -t).matrix()
      for l, p in enumerate(rh):
        e = np.outer(rd, rd) - p * np.outer(td, td) - (1 - p) * np.outer(nd, nd)
        E = r @ e @ r.T
        # the strain in every crystal frame, Q^T E Q, solved in one call
        M[l, k] = _taylorMany(np.einsum("nji,jk,nkl->nil", Q, E, Q), model, True, model.spinSelection)[0] @ w
  elif isinstance(x, SO3Fun):
    from ..functions.so3fun import SO3FunHarmonic
    turned = [SO3FunHarmonic(x.rotate(rotation.byAxisAngle(vector3d(nd), t)), bandwidth=bandwidth) for t in th]
    fr = x.SS
    for l, p in enumerate(rh):
      e = np.outer(rd, rd) - p * np.outer(td, td) - (1 - p) * np.outer(nd, nd)
      work = calcTaylor(strainTensor(e, fr), model, plasticWork=True, bandwidth=bandwidth).M
      for k in range(th.size):
        M[l, k] = float(np.real((work * turned[k]).mean()))
  else:
    raise TypeError("calcLankford takes orientations or an ODF")
  idx = np.argmin(M, axis=0)
  R = rh[idx] / (1 - rh[idx])
  return R, M, M[idx, np.arange(th.size)]


def calcKearnsFactor(f, h=None, N=None):
  """the Kearns factor, the mean squared alignment of a crystal direction with N: the
  integral of the pole figure of h against (r . N)^2"""
  from ..functions.quadrature import quadratureS2Grid
  if isinstance(f, SO3Fun):
    pdf = f.radon(h)
  else:
    pdf, N = f, h if N is None else N
  n = (vector3d(0, 0, 1) if N is None else vector3d(N)).normalize().data.reshape(3)
  g = quadratureS2Grid(pdf.bandwidth + 2)
  v = g.nodes()
  vals = np.real(pdf(vector3d(v.data, frame=pdf.frame)))
  return float(np.sum(g.weights * vals * (v.data @ n) ** 2) / (4 * np.pi))


# ------------------------------------------------------------------ the shear banding function

def SO3FunSBF(sS, E, bandwidth=None):
  """the shear banding function of a slip system and a strain (rate) tensor: the closed form
  ODF of Schaeben that a single slip system produces under that strain, as a handle"""
  from ..geometry.session import getMTEXpref
  from ..functions.handles import SO3FunHandle

  # the handle that says which system and which strain it stands for, as MATLAB's display
  class _SBF(SO3FunHandle):
    def __repr__(self):
      lam = np.round(np.real(E.eig()[0])[::-1], 4)
      return aligned("\n".join([f"SO3FunSBF ({framesLabel(self.CS, self.SS)})",
                        f"  slip system: {sS.n.char()}{sS.b.char()}",
                        "  strain:      " + " ".join(f"{x:g}" for x in lam)]))

  cs = sS.CS
  if not isinstance(cs, crystalFrame):
    raise ValueError("the slip system has to live in a crystal frame")
  if len(sS) != 1:
    raise ValueError("one slip system at a time")
  lam, V = E.eig()
  Mv = V.data.reshape(3, 3).T.copy()
  if np.linalg.det(Mv) < 0:
    Mv[:, 0] = -Mv[:, 0]
  Q = rotation.byMatrix(Mv.T)
  n = vector3d(sS.n.data.reshape(3)).normalize()
  b = vector3d(sS.b.data.reshape(3)).normalize()
  mori = rotation.map(n, vector3d(0, 1, 0), b, vector3d(1, 0, 0))
  R1, R2 = lam[0] - lam[1], lam[1] - lam[2]

  def fun(o):
    q = rotation(o.data.reshape(-1, 4))
    phi1, Phi, phi2 = (rotation(Q.data) * q * mori).Euler("Bunge")
    F = -np.sin(2 * phi1) * np.cos(Phi)
    G = np.sin(phi1) ** 2 * np.cos(Phi) ** 2 - np.cos(phi1) ** 2
    H = -np.sin(Phi) ** 2
    chi = np.arctan2(R1 * G + R2 * H, R1 * F) / 2
    Gamma = np.sqrt(R1 ** 2 * F ** 2 + (R1 * G + R2 * H) ** 2)
    return (1 / (np.cosh(Gamma) - np.sin(2 * (phi2 - chi)) * np.sinh(Gamma))).reshape(o.shape)

  fr = E.frame if isinstance(E.frame, referenceFrame) and not isinstance(E.frame, crystalFrame) else specimenFrame.default
  return _SBF(fun, cs, fr, bandwidth=getMTEXpref("maxSO3Bandwidth") if bandwidth is None else bandwidth)


# ------------------------------------------------------------------ the transport of a texture

# the continuity equation d f / dt = -div(f spin) on a fixed harmonic space: the rule exact
# for the product of the density and the spin, the spin sampled on it once, the plan of
# the density reused by every step
class _TextureTransport:

  def __init__(self, spin, f, bandwidth=None):
    from ..functions.harmonics import deg2dim
    from ..functions.quadrature import bandwidthFor, folds, quadratureSO3Grid
    from ..functions.so3fun import SO3FunHarmonic
    from ..functions.so3vectorfield import componentFrames, _tangentValues
    from ..functions.transforms import wignerPlan
    self.f = SO3FunHarmonic(f) if bandwidth is None else SO3FunHarmonic(f, bandwidth=bandwidth)
    self.L = self.f.bandwidth
    Ls = getattr(spin, "bandwidth", self.L)
    self.side = spin.tangentSpace
    ca, cb = componentFrames(self.f.CS, self.f.SS, self.side)
    rA, rB = folds(ca, cb)
    self.grid = quadratureSO3Grid(bandwidthFor(max(1, self.L + Ls), rA, rB), ca, cb)
    nodes = orientation._fromRotation(self.grid.nodes().data, (self.f.CS, self.f.SS))
    self.spin = _tangentValues(spin(nodes), self.side)
    self.plan = wignerPlan(self.grid, self.L, True)
    self.dim = deg2dim(self.L + 1)

  def derivative(self, fhat):
    """the coefficients of -div(f spin), projected onto the invariant functions"""
    from ..functions.funcs import div
    from ..functions.so3fun import SO3FunHarmonic
    from ..functions.so3vectorfield import SO3VectorFieldHarmonic, _adjointColumns
    values = np.real(self.plan(fhat))
    coeff = _adjointColumns(self.grid, values[:, None] * self.spin, self.L, self.grid.weights)
    flux = SO3VectorFieldHarmonic(coeff, self.f.CS, self.f.SS, self.side, symmetrise=False)
    d = SO3FunHarmonic(div(flux), self.f.CS, self.f.SS, bandwidth=self.L, real=True)._fhat
    d[0] = 0
    return -d


def doEulerStep(spin, x, n=1, bandwidth=None):
  """the orientations or the density after unit time in n Euler steps under the rotation
  velocity field spin: orientations move along the exponential of the field, a density by
  the explicit continuity equation in its harmonic space"""
  n = int(n)
  if n <= 0:
    raise ValueError("the step count is positive")
  if isinstance(x, orientation):
    o = x
    for _ in range(n):
      o = (spin(o) / n).exp()
    return o
  if not isinstance(x, SO3Fun):
    raise TypeError("doEulerStep moves orientations or a density")
  from ..functions.so3fun import SO3FunHarmonic
  p = _TextureTransport(spin, x, bandwidth)
  fhat = p.f._fhat.astype(complex)
  for _ in range(n):
    fhat = fhat + p.derivative(fhat) / n
  return SO3FunHarmonic(fhat, p.f.CS, p.f.SS, real=True, symmetrise=False, _internal=True)


def _install():
  from ..functions.so3fun import SO3FunHarmonic
  from ..functions.handles import SO3FunGeneric
  from ..functions.rbf import SO3FunCBF, SO3FunRBF
  from ..functions.bingham import SO3FunBingham
  functions = (SO3FunHarmonic, SO3FunGeneric, SO3FunRBF, SO3FunCBF, SO3FunBingham)
  install(functions + (orientation,), calcTensor)
  install(functions + (orientation,), calcLankford)
  install(functions, calcKearnsFactor)


_install()

__all__ = ["calcTensor", "calcLankford", "calcKearnsFactor", "SO3FunSBF", "doEulerStep"]
