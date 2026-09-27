# the pole figure to ODF inversion (MATLAB PoleFigureAnalysis/@MLSSolver, @zeroRangeMethod;
# docs/plans/functions.md P4b)
#
#   odf = calcODF(pf, resolution=5*degree, halfwidth=5*degree, zeroRange=True)
#   solver = MLSSolver(pf); odf = solver.calcODF(); solver.alpha
#   zrm = zeroRangeMethod(pf); zrm.checkZeroRange(ori)
#   odf = calcODFIterative(pf, halfwidth=2.5*degree)
#
# The ODF is the radial basis function c0 + sum_l c_l psi(., g_l) on a grid of orientations,
# and the pole figure of one component is the Radon transform of the kernel centred at the
# images of h under g_l. The operator taking the coefficients to the intensities is a
# convolution on the sphere and is never assembled: the point measure of the coefficients
# is transformed to spherical harmonics, multiplied by the kernel diagonal and evaluated at
# the specimen directions; the adjoint is the same two plans in the other order. The
# coefficients come from the modified least squares algorithm of Hielscher and Schaeben,
# J. Appl. Cryst. 41 (2008), 1024.
from __future__ import annotations

import math

import numpy as np

from ..geometry.grids.so3grid import equispacedSO3Grid
from ..geometry.kernels import blasThreads
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.session import getMTEXpref
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from ..functions.funops import conv
from ..functions.kernelfunctions import S2DeLaValleePoussinKernel, SO3DeLaValleePoussinKernel
from ..functions.polefigure import PoleFigure, calcError, calcQuadratureWeights
from ..functions.rbf import SO3FunRBF, uniformODF
from ..functions.s2fun import S2FunHarmonic
from ..functions.transforms import S2FourierPlan
from ..geometry.display import display, framesLabel, header


# ------------------------------------------------------------------ the forward operator

# the crystal directions of one pole figure expanded over the crystal group, up to sign
# where the measurement cannot tell r from -r, with the structure coefficient of each
def _symmetrisedDirections(h, c, antipodal):
  hs, refl = [], []
  for k in range(h.size):
    imgs = h[k].symmetrise(unique=True, antipodal=antipodal)
    hs.append(imgs.data.reshape(-1, 3))
    refl.append(np.full(imgs.size, c[k] / imgs.size))
  return np.concatenate(hs), np.concatenate(refl)


class PoleFigureOperator:
  """the discretised operator Psi_i of one pole figure: the plan on the nodes g_l h, the
  plan on the nodes s r_k, the kernel diagonal and the structure coefficient of every
  symmetrised crystal direction"""

  def __init__(self, grid, psi, h, c, r, bandwidth=None):
    L = min(psi.bandwidth, getMTEXpref("maxS2Bandwidth")) if bandwidth is None else int(bandwidth)
    hs, self.refl = _symmetrisedDirections(h, c, r.antipodal)
    A = psi.radon().calcFourier(L)
    if r.antipodal:
      A[1::2] = 0
    l = np.arange(L + 1)
    self.A = np.repeat(4 * np.pi * A / (2 * l + 1), 2 * l + 1)
    self.n, self.nH, self.nR = grid.size, len(hs), r.size
    gh = (rotation(grid.data.reshape(-1, 1, 4)) * vector3d(hs)).data.reshape(-1, 3)
    sym = r.frame.properGroup().rotations().flatten()
    self.nSym = sym.size
    rs = (rotation(sym.data.reshape(-1, 1, 4)) * vector3d(r.flatten().normalize().data)).data.reshape(-1, 3)
    self.planGH = S2FourierPlan(vector3d(gh), L, real=True)
    self.planR = S2FourierPlan(vector3d(rs), L, real=True)

  def forward(self, c):
    """Psi c: the pole figure the coefficients predict"""
    cext = np.outer(np.asarray(c, dtype=float), self.refl).ravel()
    fhat = self.planGH.adjoint(cext, np.ones(cext.size))
    I = self.planR(fhat * self.A)
    return I.reshape(self.nSym, self.nR).mean(axis=0)

  def backward(self, I):
    """Psi^T I: the same plans in the other order, the adjoint by construction"""
    Iext = np.tile(np.asarray(I, dtype=float), self.nSym) / self.nSym
    fhat = self.planR.adjoint(Iext, np.ones(Iext.size))
    d = self.planGH(fhat * self.A)
    return d.reshape(self.n, self.nH) @ self.refl


# ------------------------------------------------------------------ the zero range method

# sum_i w_i psi(., v_i), the density without the normalisation calcDensity applies
def _kernelSum(v, w, psi, L):
  s = S2FunHarmonic.quadrature(v, np.ones(v.size), bandwidth=L, weights=4 * np.pi * w, real=True, antipodal=v.antipodal)
  return conv(s, psi)


class zeroRangeMethod:
  """a pole figure that vanishes on a region says the ODF vanishes on every fibre through
  it: the density of the measurement points says where the statement is backed by data,
  the density weighted by alpha (I > bg) - 1 where the data is zero

    zrm = zeroRangeMethod(pf, halfwidth=, threshold=, delta=, bg=, alpha=)
    zrm.checkZeroRange(ori)          the orientations every pole figure position of which is zero
    zrm.checkZeroRange(v, i)         the positions of the i-th pole figure that are zero
  """

  def __init__(self, pf, *args, halfwidth=None, kernel=None, threshold=None, delta=0.001, bg=None, alpha=10, bandwidth=None):
    self.pf = pf
    kernels = [a for a in args if hasattr(a, "calcFourier")]
    hw = 2 * pf.resolution if halfwidth is None else halfwidth
    self.psi = kernels[0] if kernels else S2DeLaValleePoussinKernel(halfwidth=hw) if kernel is None else kernel
    self.threshold = 0.1 * float(self.psi(1.0)) if threshold is None else float(threshold)
    self.delta, self.alpha = float(delta), float(alpha)
    self.bg = self.delta * pf.max() if bg is None else np.broadcast_to(np.asarray(bg, dtype=float), (pf.numPF,))
    L = min(self.psi.bandwidth, getMTEXpref("maxS2Bandwidth")) if bandwidth is None else int(bandwidth)
    self.density, self.pdf, self._h = [], [], []
    for k in range(pf.numPF):
      r, I = pf.allR[k].flatten(), pf.allI[k].ravel()
      w = self.alpha * (I > self.bg[k]) - 1.0
      self.density.append(_kernelSum(r, np.ones(r.size), self.psi, L))
      self.pdf.append(_kernelSum(r, w, self.psi, L))
      hs, _ = _symmetrisedDirections(pf.allH[k], pf.c[k], pf.antipodal)
      self._h.append(vector3d(hs, frame=pf.CS))

  def checkZeroRange(self, x, i=None):
    """which orientations the zero range method rules out: those some pole figure is zero at in
    one of their symmetric poles; with i, which directions the i-th pole figure is zero at

      zrm.checkZeroRange(ori)
      zrm.checkZeroRange(v, i)"""
    if i is not None:
      v = vector3d(x.flatten().normalize().data)
      d = np.real(self.density[i](v))
      p = np.real(self.pdf[i](v))
      with np.errstate(divide="ignore", invalid="ignore"):
        return (d > self.threshold) & (p / d < -0.1)
    o = x.flatten()
    isZero = np.zeros(o.size, dtype=bool)
    q = rotation(o.data.reshape(-1, 1, 4))
    for i, hs in enumerate(self._h):
      for k in range(hs.size):
        v = q * vector3d(hs[k].data.reshape(3))
        isZero |= self.checkZeroRange(v, i)
    return isZero

  def __repr__(self):
    return display("zeroRangeMethod", [("pole figures", self.pf.numPF), ("kernel", self.psi), ("threshold", f"{self.threshold:.4g}")])


# ------------------------------------------------------------------ the solver

class MLSSolver:
  """the modified least squares solver of the pole figure inversion: the ansatz functions
  sit on an equispaced grid and have the shape of a kernel, by default the de la Vallee
  Poussin kernel of the grid's resolution

    solver = MLSSolver(pf, resolution=5*degree, halfwidth=7.5*degree)
    odf = solver.calcODF()
    solver.alpha                     the scaling of each pole figure
  """

  def __init__(self, pf, *args, resolution=None, halfwidth=None, kernel=None, S3G=None, zeroRange=False, iterMin=5,
               iterMax=10, c0=None, noGhostCorrection=False, quadratureWeights=False, intensityWeights=False,
               regularisation=0.0, silent=True):
    # the intensities non negative, a pole figure ten times below the largest lifted
    data = pf.max(0)
    mm = np.max(data.max())
    factors = np.array([mm / (10 * m) if mm > 10 * m > 0 else 1.0 for m in data.max()])
    self.pf = data * factors
    grids = [a for a in args if isinstance(a, orientation)]
    kernels = [a for a in args if hasattr(a, "calcFourier")]
    res = self.pf.resolution if resolution is None else resolution
    self.S3G = grids[0] if grids else S3G if S3G is not None else equispacedSO3Grid(self.pf.CS, self.pf.SS, resolution=res)
    hw = getattr(self.S3G, "resolution", res) if halfwidth is None else halfwidth
    self.psi = kernels[0] if kernels else SO3DeLaValleePoussinKernel(halfwidth=hw) if kernel is None else kernel
    self.zrm = zeroRange if isinstance(zeroRange, zeroRangeMethod) else zeroRangeMethod(self.pf) if zeroRange else None
    self.iterMin, self.iterMax = int(iterMin), max(int(iterMax), int(iterMin))
    self.c = None if c0 is None else np.asarray(c0, dtype=float).ravel()
    self.c0 = 0.0
    self.ghostCorrection = not noGhostCorrection
    self.silent = silent
    if quadratureWeights and self.pf.SS.numProper() == 1:
      self.weights = [calcQuadratureWeights(r).ravel() for r in self.pf.allR]
    else:
      self.weights = [np.full(I.size, 1 / I.size) for I in self.pf.allI]
    if intensityWeights:
      self.weights = [w * (1 + I.ravel()) ** -0.5 for w, I in zip(self.weights, self.pf.allI)]
    self.lambda_ = float(regularisation)
    self.RM = None
    self.ops, self.alpha, self.a, self.u = [], np.zeros(0), [], []

  @property
  def CS(self):
    """the crystal frame of the pole figures the solver inverts"""
    return self.pf.CS

  @property
  def SS(self):
    """the specimen frame of the pole figures the solver inverts"""
    return self.pf.SS

  def init(self):
    """the operators on the grid the zero range method leaves"""
    if self.zrm is not None:
      keep = ~self.zrm.checkZeroRange(self.S3G)
      if not self.silent:
        print(f"  the zero range method drops {np.count_nonzero(~keep)} of {keep.size} orientations")
      if np.any(keep):
        self.S3G = self.S3G.subGrid(keep)
      elif not self.silent:
        print("  the zero range method would drop every orientation of the grid; the grid is kept")
    self.c0 = 0.0
    self.ops = [PoleFigureOperator(self.S3G, self.psi, h, c, r) for h, c, r in zip(self.pf.allH, self.pf.c, self.pf.allR)]
    if self.lambda_ > 0:
      self.initRegularisation()
    return self

  def initRegularisation(self, epsilon=None):
    """a discrete Laplacian on the grid: minus the dots within two halfwidths off the
    diagonal, the row sums on it"""
    from scipy import sparse
    eps = 2 * self.psi.halfwidth() if epsilon is None else epsilon
    from ..geometry.grids.orientationtree import gridRadius
    _, C = gridRadius(self.S3G, self.S3G.flatten(), eps)
    C = C.tocoo()
    A = sparse.csr_matrix((-C.data, (C.row, C.col)), shape=C.shape)
    A.setdiag(0)
    A.eliminate_zeros()
    d = np.asarray(A.sum(axis=0)).ravel() - 1
    A = A + sparse.diags(-1 - d)
    self.RM = self.lambda_ * A.tocsr()
    return self

  def initIter(self):
    """the state the iteration starts from: the normalisation directions a_i, the scalings
    alpha_i and the weighted residuals u_i"""
    n = len(self.ops)
    if self.c is None or self.c.size != self.S3G.size:
      self.c = np.full(self.S3G.size, 1 / self.S3G.size)
    reg = self.RM is not None
    self.a, self.u = [None] * (n + reg), [None] * (n + reg)
    self.alpha = np.zeros(n + reg)
    for i, op in enumerate(self.ops):
      I = self.pf.allI[i].ravel()
      self.a[i] = op.backward(np.ones(I.size)) / np.sum(I)
      self.alpha[i] = 1 / (self.a[i] @ self.c)
      self.u[i] = (op.forward(self.c) * self.alpha[i] - I) * self.weights[i]
    if reg:
      self.alpha[n] = 1 / np.sum(self.c)
      self.u[n] = self.alpha[n] * (self.RM @ self.c)
    return self

  def doIter(self):
    """one step of the modified projected gradient; the gradient is tangent to the
    normalisation, <c, v> = 0, so the regularisation block, the Dirichlet energy of the
    normalised coefficients, moves linearly along the direction and keeps its scaling"""
    n = len(self.ops)
    reg = self.RM is not None
    c, u, alpha, a = self.c, self.u, self.alpha, self.a
    v = alpha[n] * u[n] - alpha[n] ** 2 * (u[n] * c) if reg else np.zeros(c.size)
    for i, op in enumerate(self.ops):
      vti = alpha[i] * op.backward(u[i] * self.weights[i])
      v = v - vti + alpha[i] * (vti @ c) * a[i]
    # the feasible direction and the state it would lead to
    tc = c * v
    talpha = np.zeros(n)
    tu = [None] * len(u)
    for i, op in enumerate(self.ops):
      talpha[i] = 1 / (a[i] @ tc)
      tu[i] = (talpha[i] * op.forward(tc) - self.pf.allI[i].ravel()) * self.weights[i]
    if reg:
      tu[n] = alpha[n] * (self.RM @ tc)
    # the largest step keeping the coefficients non negative, and the best step below it
    neg = tc < 0
    tauMax = float(np.min(-c[neg] / tc[neg])) if np.any(neg) else 1.0
    tau = self._lineSearch(tc, tu, talpha, tauMax)
    self.c = np.maximum(c + tau * tc, 0)
    denom = alpha[:n] * tau + talpha
    for i in range(n):
      u[i] = (talpha[i] / denom[i]) * u[i] + (tau * alpha[i] / denom[i]) * tu[i]
    if reg:
      u[n] = u[n] + tau * tu[n]
    self.alpha[:n] = alpha[:n] * talpha / denom
    return self

  # the step minimising J(tau) = sum_i (A_i + 2 tau B_i + tau^2 C_i) / (tau alpha_i + talpha_i)^2
  # plus the energy of the regularisation on a geometric ladder below the largest
  # feasible step
  def _lineSearch(self, tc, tu, talpha, tauMax, candidates=200):
    n = len(self.ops)
    A, B, C = np.zeros(n), np.zeros(n), np.zeros(n)
    for j in range(n):
      A[j] = talpha[j] ** 2 * np.sum(self.u[j] ** 2)
      B[j] = talpha[j] * self.alpha[j] * (self.u[j] @ tu[j])
      C[j] = self.alpha[j] ** 2 * np.sum(tu[j] ** 2)
    tau = tauMax * 0.95 ** np.arange(candidates)
    t = tau[:, None]
    J = np.sum((A + 2 * t * B + t ** 2 * C) / (t * self.alpha[:n] + talpha) ** 2, axis=1)
    if self.RM is not None:
      Ar, Br, Cr = self.alpha[n] * (self.c @ self.u[n]), self.alpha[n] * (tc @ self.u[n]), self.alpha[n] * (tc @ tu[n])
      J = J + Ar + 2 * tau * Br + tau ** 2 * Cr
    return float(tau[np.argmin(J)])

  def residual(self):
    """the relative residual of each pole figure, and the norm of the vector of them"""
    e = np.array([np.linalg.norm(self.u[i]) / np.linalg.norm(self.pf.allI[i].ravel() * self.weights[i]) for i in range(len(self.ops))])
    return e, float(np.sqrt(np.sum(e ** 2)))

  def iterate(self):
    """the iteration, stopped when the residual improves by less than five percent; its BLAS
    calls are products of vectors, and OpenBLAS's waiting threads would take the cores of the
    transforms between them"""
    with blasThreads(1):
      return self._iterate()

  def _iterate(self):
    self.initIter()
    e, last = self.residual()
    if not self.silent:
      print("   0 | " + "  ".join(f"{x:.2f}" for x in e))
    for it in range(1, self.iterMax + 1):
      self.doIter()
      e, err = self.residual()
      if not self.silent:
        print(f"  {it:2d} | " + "  ".join(f"{x:.2f}" for x in e))
      if (last - err) / err < 0.05 and it > self.iterMin:
        break
      last = err
    return self

  @property
  def odf(self):
    """the ODF the current coefficients describe"""
    if self.c0 > 0.99:
      return uniformODF(self.CS, self.SS)
    return SO3FunRBF(self.S3G, self.psi, (1 - self.c0) * self.c / np.sum(self.c) / self.psi.mean(), self.c0)

  def calcODF(self, iterMin=None, iterMax=None, noGhostCorrection=None, silent=None):
    """the reconstruction, with the uniform portion the data allows fixed by the ghost
    correction and put into the constant instead of the components"""
    if iterMin is not None:
      self.iterMin = int(iterMin)
    if iterMax is not None:
      self.iterMax = max(int(iterMax), self.iterMin)
    if silent is not None:
      self.silent = silent
    ghost = self.ghostCorrection if noGhostCorrection is None else not noGhostCorrection
    self.init()
    self.iterate()
    n = len(self.ops)
    alpha = self.alpha[:n].copy()
    c0 = min(np.quantile(np.maximum(I.ravel(), 0), 0.01) / alpha[i] for i, I in enumerate(self.pf.allI))
    if ghost and 0.1 < c0 < 0.99:
      if not self.silent:
        print(f"  ghost correction: the uniform portion is fixed to {c0:.3f}")
      self.pf = (self.pf - alpha * c0).max(0)
      self.c0 = float(c0)
      self.c = None
      self.iterate()
      self.alpha[:n] = alpha
    return self.odf

  def __repr__(self):
    return display(header("MLSSolver", framesLabel(self.pf.CS, self.pf.SS)),
                   [("pole figures", self.pf.numPF), ("kernel", self.psi), ("grid", f"{self.S3G.size} orientations")])


def calcODF(pf, *args, resolution=None, halfwidth=None, kernel=None, S3G=None, zeroRange=False, iterMin=5, iterMax=10, c0=None,
            noGhostCorrection=False, quadratureWeights=False, intensityWeights=False, regularisation=0.0, silent=True):
  """an ODF reconstructed from pole figure intensities, an SO3FunRBF

    calcODF(pf)
    calcODF(pf, halfwidth=5*degree, resolution=5*degree)
    calcODF(pf, zeroRange=True)                     drop the orientations the data says are empty
    calcODF(pf, noGhostCorrection=True)             keep the uniform portion in the components
  """
  solvers = [a for a in args if isinstance(a, MLSSolver)]
  if solvers:
    return solvers[0].calcODF()
  return MLSSolver(pf, *args, resolution=resolution, halfwidth=halfwidth, kernel=kernel, S3G=S3G, zeroRange=zeroRange,
                   iterMin=iterMin, iterMax=iterMax, c0=c0, noGhostCorrection=noGhostCorrection,
                   quadratureWeights=quadratureWeights, intensityWeights=intensityWeights,
                   regularisation=regularisation, silent=silent).calcODF()


# ------------------------------------------------------------------ successive refinement

def calcODFIterative(pf, *args, resolution=None, halfwidth=None, kernel=None, maxSteps=5, thinning=0.025, nothinning=False,
                     silent=True, **kwargs):
  """the reconstruction refined successively (MATLAB calcODFIterative): a ladder of kernel
  halfwidths from sqrt(e)^(steps-1) times the target down to the target, those above 45
  degree dropped, each solved on the grid of its own width and started from the previous
  ODF evaluated on that grid, the nodes below the thinning fraction of the mean weight
  removed before the solve"""
  kernels = [a for a in args if hasattr(a, "calcFourier")]
  res = pf.resolution if resolution is None else resolution
  hw = res if halfwidth is None else halfwidth
  psi = kernels[0] if kernels else kernel
  if psi is not None:
    hw = psi.halfwidth()
  make = (lambda h: type(psi)(halfwidth=h)) if psi is not None else (lambda h: SO3DeLaValleePoussinKernel(halfwidth=h))
  res = min(res, hw)
  ladder = [math.sqrt(math.e) ** (k - 1) for k in range(maxSteps, 0, -1)]
  steps = [(x * hw, x * res) for x in ladder if x * res <= 45 * degree]
  odf = uniformODF(pf.CS, pf.SS)
  for k, (h, r) in enumerate(steps):
    grid = equispacedSO3Grid(pf.CS, pf.SS, resolution=r)
    c = np.maximum(np.real(odf(grid)), 0)
    c = c / np.sum(c)
    if thinning > 0 and not nothinning:
      keep = c > min(1.0, thinning) / c.size
      grid = grid.subGrid(keep)
      c = c[keep]
    solver = MLSSolver(pf, S3G=grid, kernel=make(h), c0=c, silent=silent, **kwargs)
    odf = solver.calcODF()
    if not silent:
      e = calcError(pf, odf)
      print(f"  {k + 1:2d} : {grid.size:7d} | " + "  ".join(f"{x:.2f}" for x in e))
  return odf


__all__ = ["MLSSolver", "zeroRangeMethod", "calcODF", "calcODFIterative"]
