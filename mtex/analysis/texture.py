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
from ..maps.ebsd import EBSD
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


# ------------------------------------------------------------------ the average of a tensor

def calcTensor(x, *args, average="Voigt", weights=None):
  """the texture average of a single crystal tensor over an ODF, a list of orientations or a
  map; MATLAB's calcTensor

    TV = calcTensor(odf, T)                       # over an ODF, the Voigt average
    TR = calcTensor(odf, T, 'Reuss')              # the inverse averaged and inverted
    TH = calcTensor(odf, T, average='Hill')       # the mean of the Voigt and the Reuss average
    TV = calcTensor(ori, T, weights=w)            # over a weighted list of orientations
    TV, TR, TH = calcTensor(ebsd, T1, T2, 'all')  # over a map, one tensor per phase

  The average decides how the rotated tensors are combined:

  - 'Voigt', the default: the mean of the tensor rotated by the orientations.
  - 'Reuss': the inverse of the mean of the rotated inverses.
  - 'Hill': the mean of the Voigt and the Reuss average.
  - 'all': the three of them.

  Over an ODF the average is exact, since a rotated tensor of rank R is a polynomial of degree
  R in the rotation. MATLAB's geometric mean ('geometric') is not offered.

  Over a map every indexed phase takes the tensor of its crystal frame, the tensors given in
  any order, and a phase without a tensor is refused: select the phases first,
  ebsd[['Forsterite', 'Enstatite']].

  Input
    x - SO3Fun, orientation or EBSD
    T - tensor, the single crystal tensor; for a map one per phase
  Output
    T - the average, of the class of T; with 'all' the three averages Voigt, Reuss, Hill
  Keywords
    average - 'Voigt', 'Reuss', 'Hill' or 'all'; also given as a string after the tensors
    weights - one weight per orientation or pixel
  See also
    tensor.mean, calcODF, TensorAverage"""
  which = _average(args, average)
  tensors = [a for a in args if isinstance(a, tensor)]
  if isinstance(x, EBSD):
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
    fast = T._turnedMean(x.flatten(), w)
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
  indexed = np.asarray(ebsd.isIndexed)
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
      acc = acc + (m.M.ravel() * np.sum(wp) if m is not None else wp @ (o * Tp).data.reshape(o.size, -1))
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
  if a.isclose(b):
    return True
  return (getattr(a, "mineral", None) == getattr(b, "mineral", None) and a.Laue().pointGroup == b.Laue().pointGroup
          and np.allclose(a.abc, b.abc, rtol=1e-2) and np.allclose(a.abg, b.abg, atol=1e-2))


# ------------------------------------------------------------------ the sheet quantities

def calcLankford(x, sS, *args, theta=None, rho=None, RD=None, ND=None, weights=None, bandwidth=16):
  """the Lankford coefficients, the R-values, of a sheet from its orientations or its ODF

    R, M, minM = calcLankford(ori, sS)                 # R at 0, 5, ..., 90 degree to the rolling direction
    R, M, minM = calcLankford(ori, sS, theta, weights=w)  # weighted orientations, chosen angles
    R, M, minM = calcLankford(odf, sS, rho=np.linspace(0, 1, 21))  # from an ODF, a finer ratio

  The R-value is the ratio of the true width strain to the true thickness strain of a strip
  pulled at the angle theta to the rolling direction. For every theta the strip is strained
  along the pull and contracts by the fraction rho across it and 1 - rho through the
  thickness; the Taylor model gives the normalised plastic work M of every rho, averaged over
  the texture, and the strip takes the rho of least work, R = rho / (1 - rho).

  The mean Rbar of R over the directions tells how well a sheet resists thinning, Rbar >= 1
  favouring deep drawing; the planar anisotropy deltaR indicates the fourfold earing of a cup.

  The angles theta may also be given as a positional array after sS, as MATLAB.

  Input
    x - orientation, the crystal orientations of the sheet, or SO3Fun, its ODF
    sS - slipSystem, or a TaylorModel, the systems the crystals deform by
  Output
    R - the Lankford coefficient at the least plastic work, one per theta
    M - the normalised plastic work, an array of rho by theta
    minM - the least normalised plastic work, one per theta
  Keywords
    theta - the angles of the pull to the rolling direction in radians
    rho - the fractions of the contraction across the strip, between 0 and 1
    RD - vector3d, the rolling direction, the pull at theta = 0
    ND - vector3d, the normal of the sheet, orthogonal to RD
    weights - the weights of the orientations, the area of grains for instance
    bandwidth - the harmonic bandwidth the Taylor factor and the ODF are expanded to (an ODF)
  See also
    calcTaylor, TaylorModel, slipSystem, Lankford"""
  th =np.linspace(0, np.pi / 2, 19) if theta is None else np.asarray(theta, dtype=float).ravel()
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
  # the sheet strain of every ratio rho: the extension along RD, the contractions along TD and ND
  e = np.outer(rd, rd) - rh[:, None, None] * np.outer(td, td) - (1 - rh)[:, None, None] * np.outer(nd, nd)
  M = np.zeros((rh.size, th.size))
  if isinstance(x, orientation):
    o = x.flatten()
    w = _weights(weights, o.size)
    Q = rotation(o.data).matrix()
    for k, t in enumerate(th):
      r = rotation.byAxisAngle(vector3d(nd), -t).matrix()
      # the strains of every ratio in every crystal frame, Q^T r e r^T Q, solved in one call
      E = np.einsum("nji,ljk,nkm->lnim", Q, r @ e @ r.T, Q)
      M[:, k] = _taylorMany(E, model, True, model.spinSelection)[0].reshape(rh.size, -1) @ w
  elif isinstance(x, SO3Fun):
    from ..functions.so3fun import SO3FunHarmonic
    turned = [SO3FunHarmonic(x.rotate(rotation.byAxisAngle(vector3d(nd), t)), bandwidth=bandwidth) for t in th]
    fr = x.SS
    for l in range(rh.size):
      work = calcTaylor(strainTensor(e[l], fr), model, plasticWork=True, bandwidth=bandwidth).M
      for k in range(th.size):
        M[l, k] = float(np.real((work * turned[k]).mean()))
  else:
    raise TypeError("calcLankford takes orientations or an ODF")
  idx = np.argmin(M, axis=0)
  R = rh[idx] / (1 - rh[idx])
  return R, M, M[idx, np.arange(th.size)]


def calcKearnsFactor(f, h=None, N=None):
  """the Kearns factor, the mean squared alignment of a crystal direction with the specimen
  direction N; MATLAB's calcKearnsFactor

    k = calcKearnsFactor(odf, h, N)    # of the crystal direction h with N
    k = calcKearnsFactor(odf, h)       # with the z direction
    k = calcKearnsFactor(pdf, N)       # of a pole figure

  The integral of the pole figure of h against (r . N)^2 over the sphere, normalised so that
  a uniform texture gives 1/3. MATLAB's order is calcKearnsFactor(odf, N, h), the arguments
  told apart by their class; here h comes first. MATLAB's default h = [0001] is not taken:
  an ODF needs h.

  Input
    f - SO3Fun, an ODF, or S2Fun, a pole figure
    h - Miller, the crystal direction (with an ODF); with a pole figure the place of N
    N - vector3d, the specimen direction, z when left out
  Output
    k - float
  See also
    radon, calcLankford"""
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
  ODF of Schaeben that a single slip system produces under that strain, as a handle; MATLAB's
  SO3FunSBF

    odf = SO3FunSBF(sS, E)                  # the deformation texture of one slip system
    odf = SO3FunSBF(sS, E, bandwidth=32)    # with the bandwidth of its harmonic conversion

  The texture follows from the finite strain and the slip system without integrating its
  evolution; it is evaluated analytically and has no bandwidth of its own, the bandwidth is
  the one a harmonic conversion uses. The function maps the crystal frame of the slip system
  to the frame of E, the default specimen frame when E has none.

  Input
    sS - slipSystem, a single one, in a crystal frame
    E - strainTensor or strainRateTensor
  Output
    odf - SO3FunHandle
  Keywords
    bandwidth - the bandwidth of the handle, by default the preference maxSO3Bandwidth
  References
    H. Schaeben et al., An analytical finite-strain parametrization for texture evolution in
        deforming olivine polycrystals, Geophys. J. Int. 216 (2019)
  See also
    doEulerStep, SingleSlipModel, slipSystem, strainTensor"""
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
  velocity field spin; MATLAB's doEulerStep

    ori = doEulerStep(spin, ori0, 10)                # orientations in 10 steps
    odf = doEulerStep(spin, odf0, 40)                # a density in 40 steps
    odf = doEulerStep(spin, odf0, 40, bandwidth=16)  # the density in harmonics up to 16

  Orientations move along the exponential of the field, one step of a n-th of the time at a
  time. A density follows the continuity equation, d f / dt = -div(f spin), integrated by the
  explicit Euler method in its harmonic space. MATLAB's implicit variants ('implicit',
  'implicit_fixPoint') are not offered.

  Input
    spin - SO3VectorField or a callable of orientations, the rotation velocity
    x - orientation or SO3Fun, the start
  Output
    x - orientation or SO3FunHarmonic after unit time
  Keywords
    n - the number of steps
    bandwidth - the bandwidth of the density, by default that of SO3FunHarmonic(x)
  See also
    SO3FunSBF, calcTaylor, SingleSlipModel, TaylorModel"""
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
