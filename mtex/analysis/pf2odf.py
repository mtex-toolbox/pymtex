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

from typing import TYPE_CHECKING
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
from ..functions.kernelfunctions import S2DeLaValleePoussinKernel, S2Kernel, SO3DeLaValleePoussinKernel, SO3Kernel
from ..functions.polefigure import PoleFigure, calcError, calcQuadratureWeights
from ..functions.rbf import SO3FunRBF, uniformODF
from ..functions.s2fun import S2FunHarmonic
from ..functions.transforms import S2FourierPlan
from ..geometry.display import display, framesLabel, header
from ..geometry.framepair import installFramePair


# ------------------------------------------------------------------ the forward operator

# the crystal directions of one pole figure expanded over the crystal group, up to sign
# where the measurement cannot tell r from -r, with the structure coefficient of each
def _symmetrisedDirections(h, c, antipodal):
  hs, refl = [], []
  for k in range(h.size):
    imgs = h[k].symmetrise(unique=True, antipodal=antipodal or None)
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

# the directions checkZeroRange evaluates in one call, orientations times symmetric poles
ZERO_RANGE_CHUNK = 1 << 20


# sum_i w_i psi(., v_i), the density without the normalisation calcDensity applies
def _kernelSum(v, w, psi, L):
  s = S2FunHarmonic.quadrature(v, np.ones(v.size), bandwidth=L, weights=4 * np.pi * w, real=True, antipodal=v.antipodal)
  return conv(s, psi)


class zeroRangeMethod:
  """the zero range method of the pole figure inversion: where a pole figure vanishes, the ODF
  vanishes on every fibre through it

    zrm = zeroRangeMethod(pf)                                     # the defaults
    zrm = zeroRangeMethod(pf, halfwidth=10*degree, delta=0.01)    # a wider kernel, a higher background
    isZero = zrm.checkZeroRange(ori)                              # the orientations ruled out
    isZero = zrm.checkZeroRange(v, i)                             # the zero positions of the i-th pole figure

  A direction of a pole figure is in its zero range where enough measurements surround it
  and most of them lie below the background. Both are kernel sums over the measured
  directions: their density, which must exceed the threshold, and a sum weighting an
  intensity above the background bg by alpha - 1 and one below it by -1, which must be
  below -0.1 times the density.

  An orientation is ruled out when one of its symmetric poles falls into the zero range of
  some pole figure. calcODF(pf, zeroRange=True) drops those orientations from the grid of
  the ansatz functions; calcODF(pf, zeroRange=zrm) uses the method set up here.

  Input
    pf - PoleFigure
    psi - S2Kernel, the smoothing kernel, given positionally
  Output
    zrm - zeroRangeMethod
  Keywords
    halfwidth - the halfwidth of the de la Vallee Poussin smoothing kernel, by default twice
        the resolution of the pole figures
    kernel - S2Kernel, the smoothing kernel instead
    threshold - the density below which the data says nothing, by default a tenth of the
        kernel's maximum
    delta - the background as a fraction of the largest intensity, when bg is not given
    bg - the background intensity, one number or one per pole figure
    alpha - the weight of an intensity above the background against one below it
    bandwidth - the bandwidth of the smoothed densities, by default the kernel's
  See also
    calcODF, MLSSolver, PoleFigureDubna"""

  def __init__(self, pf, *args, halfwidth=None, kernel=None, threshold=None, delta=0.001, bg=None, alpha=10, bandwidth=None):
    self.pf = pf
    kernels = [a for a in args if isinstance(a, S2Kernel)]
    if kernels:
      self.psi = kernels[0]
    elif kernel is not None:
      self.psi = kernel
    else:
      self.psi = S2DeLaValleePoussinKernel(halfwidth=2 * pf.resolution if halfwidth is None else halfwidth)
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
    """which orientations the zero range method rules out, those some pole figure is zero at in
    one of their symmetric poles; MATLAB's checkZeroRange

      isZero = zrm.checkZeroRange(ori)     # the orientations ruled out by any pole figure
      isZero = zrm.checkZeroRange(v, i)    # the directions the i-th pole figure is zero at

    A direction is in the zero range of a pole figure where the kernel density of its
    measured directions exceeds the threshold and the kernel sum, positive at intensities
    above the background and negative below, is negative there.

    Input
      zrm - zeroRangeMethod
      ori - orientation, or v - vector3d, the specimen directions with an index
      i - the index of the pole figure, counted from zero
    Output
      isZero - bool array, one entry per orientation or direction
    See also
      zeroRangeMethod, MLSSolver, calcODF"""
    if i is not None:
      v = vector3d(x.flatten().normalize().data)
      d = np.real(self.density[i](v))
      p = np.real(self.pdf[i](v))
      with np.errstate(divide="ignore", invalid="ignore"):
        return (d > self.threshold) & (p / d < -0.1)
    q = x.flatten().data.reshape(-1, 4)
    isZero = np.zeros(len(q), dtype=bool)
    for i, hs in enumerate(self._h):
      h = vector3d(hs.data.reshape(-1, 3))
      # the orientations in chunks, every symmetric pole of a chunk evaluated in one call
      step = max(1, ZERO_RANGE_CHUNK // h.size)
      for a in range(0, len(q), step):
        v = rotation(q[a:a + step, None]) * h
        isZero[a:a + step] |= self.checkZeroRange(v, i).reshape(-1, h.size).any(axis=1)
    return isZero

  def __repr__(self):
    return display("zeroRangeMethod", [("pole figures", self.pf.numPF), ("kernel", self.psi), ("threshold", f"{self.threshold:.4g}")])


# ------------------------------------------------------------------ the solver

class MLSSolver:
  """the modified least squares solver of the pole figure inversion, an ODF as a weighted sum of
  unimodal components fitted to scattered pole figure intensities

    solver = MLSSolver(pf, resolution=5*degree, halfwidth=7.5*degree)   # set up the solver
    odf = solver.calcODF()                                              # and run it
    solver = MLSSolver(pf, psi, S3G=grid)                               # a kernel and the centres given
    solver.alpha                                                        # the scaling of each pole figure

  The ansatz functions have the shape of a kernel, by default the de la Vallee Poussin kernel,
  and sit on a grid of orientations, by default an equispaced grid of the resolution. The
  coefficients minimise the squared misfit between the pole figures of the ODF and the
  measured intensities, every pole figure with its own scaling alpha, under the constraint
  that they are non negative. calcODF(pf, ...) builds the solver and runs it at once.

  The kernel and the grid may also be passed positionally. The solver keeps its state:
  solver.S3G the centres, solver.psi the kernel, solver.c the coefficients, solver.alpha the
  scaling factors, solver.zrm the zero range method; solver.odf the ODF they describe.

  Input
    pf - PoleFigure
    psi - SO3Kernel, the shape of the ansatz functions
    grid - orientation, the centres of the ansatz functions
  Output
    solver - MLSSolver
  Keywords
    resolution - the spacing of the grid the ansatz functions sit on, by default the
        resolution of the pole figures (MATLAB: 3/2 of it)
    halfwidth - the halfwidth of the ansatz functions, by default the resolution of the grid
        (MATLAB: 2/3 of it)
    kernel - SO3Kernel, the shape of the ansatz functions instead of the de la Vallee Poussin
        kernel of the halfwidth
    S3G - orientation, the centres of the ansatz functions instead of the equispaced grid
    zeroRange - apply the zero range method: drop the centres where some pole figure says the
        ODF vanishes; a zeroRangeMethod gives its parameters
    minIter - the least number of iterations
    maxIter - the largest number of iterations (MATLAB iterMax)
    c0 - the coefficients of the ansatz functions the iteration starts from, by default all equal
    ghostCorrection - fix the uniform portion the data allows and reconstruct only the rest;
        MATLAB flag noGhostCorrection is ghostCorrection=False
    quadratureWeights - weight the residuals with the quadrature weights of the measured
        directions (no specimen symmetry)
    intensityWeights - weight the residuals with (1 + I)^(-1/2), the counting statistics of the
        intensities I
    regularisation - the weight of a penalty on the Dirichlet energy, a discrete Laplacian on
        the grid; zero for none
    verbose - report the fit of every pole figure in every iteration
  References
    R. Hielscher, H. Schaeben, A novel pole figure inversion method: specification of the MTEX
        algorithm, J. Appl. Cryst. 41 (2008) 1024-1037
  See also
    calcODF, MLSSolver.calcODF, calcODFIterative, zeroRangeMethod, PoleFigureTutorial,
    PoleFigure2ODFAmbiguity, PoleFigure2ODFGhostCorrection"""

  def __init__(self, pf, *args, resolution=None, halfwidth=None, kernel=None, S3G=None, zeroRange=False, minIter=5,
               maxIter=10, c0=None, ghostCorrection=True, quadratureWeights=False, intensityWeights=False,
               regularisation=0.0, verbose=False):
    # the intensities non negative, a pole figure ten times below the largest lifted
    data = pf.max(0)
    mm = np.max(data.max())
    factors = np.array([mm / (10 * m) if mm > 10 * m > 0 else 1.0 for m in data.max()])
    self.pf = data * factors
    grids = [a for a in args if isinstance(a, orientation)]
    kernels = [a for a in args if isinstance(a, SO3Kernel)]
    res = self.pf.resolution if resolution is None else resolution
    if grids:
      self.S3G = grids[0]
    elif S3G is not None:
      self.S3G = S3G
    else:
      self.S3G = equispacedSO3Grid(self.pf.CS, self.pf.SS, resolution=res)
    if kernels:
      self.psi = kernels[0]
    elif kernel is not None:
      self.psi = kernel
    else:
      self.psi = SO3DeLaValleePoussinKernel(halfwidth=getattr(self.S3G, "resolution", res) if halfwidth is None else halfwidth)
    if isinstance(zeroRange, zeroRangeMethod):
      self.zrm = zeroRange
    else:
      self.zrm = zeroRangeMethod(self.pf) if zeroRange else None
    self.minIter, self.maxIter = int(minIter), max(int(maxIter), int(minIter))
    self.c = None if c0 is None else np.asarray(c0, dtype=float).ravel()
    self.c0 = 0.0
    self.ghostCorrection = ghostCorrection
    self.verbose = verbose
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
      if self.verbose:
        print(f"  the zero range method drops {np.count_nonzero(~keep)} of {keep.size} orientations")
      if np.any(keep):
        self.S3G = self.S3G.subGrid(keep)
      elif self.verbose:
        print("  the zero range method would drop every orientation of the grid; the grid is kept")
    self.c0 = 0.0
    self.ops = [PoleFigureOperator(self.S3G, self.psi, h, c, r) for h, c, r in zip(self.pf.allH, self.pf.c, self.pf.allR)]
    if self.lambda_ > 0:
      self.initRegularisation()
    return self

  def initRegularisation(self, epsilon=None):
    """sets the regularisation matrix of the solver, its weight times a discrete Laplacian on
    the grid; MATLAB's initRegularisation

      solver.initRegularisation()                    # the neighbours within two halfwidths
      solver.initRegularisation(epsilon=15*degree)   # within 15 degree

    Off the diagonal the matrix holds minus the dots of the grid orientations closer than
    epsilon, on the diagonal minus the sum of the row's other entries. init calls it when the
    solver has a regularisation weight.

    Input
      solver - MLSSolver, initialised
    Output
      solver - the solver, its RM set
    Keywords
      epsilon - the radius of the neighbours, by default twice the halfwidth of the kernel
    See also
      MLSSolver, calcODF"""
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
    if self.verbose:
      print("   0 | " + "  ".join(f"{x:.2f}" for x in e))
    for it in range(1, self.maxIter + 1):
      self.doIter()
      e, err = self.residual()
      if self.verbose:
        print(f"  {it:2d} | " + "  ".join(f"{x:.2f}" for x in e))
      if (last - err) / err < 0.05 and it > self.minIter:
        break
      last = err
    return self

  @property
  def odf(self):
    """the ODF the current coefficients describe"""
    if self.c0 > 0.99:
      return uniformODF(self.CS, self.SS)
    return SO3FunRBF(self.S3G, self.psi, (1 - self.c0) * self.c / np.sum(self.c) / self.psi.mean(), self.c0)

  def calcODF(self, minIter=None, maxIter=None, ghostCorrection=None, verbose=None):
    """the ODF the solver reconstructs, with the ghost correction

      odf = solver.calcODF()                                   # with the solver's settings
      odf = solver.calcODF(minIter=2, maxIter=4)               # another number of iterations
      odf = solver.calcODF(ghostCorrection=False, verbose=True)  # no ghost correction, reported

    The zero range method, when set, thins the grid first; then the coefficients are iterated.
    With the ghost correction the uniform portion the data allows, the smallest intensity of
    the scaled pole figures, is subtracted, the rest reconstructed again and the portion kept
    as the constant of the ODF rather than spread over the components. The keywords replace
    the solver's settings for this and later calls. The scaling of every pole figure,
    MATLAB's second output alpha, is solver.alpha afterwards.

    Input
      solver - MLSSolver
    Output
      odf - SO3FunRBF, or the uniform ODF when the data says the ODF is uniform
    Keywords
      minIter - the least number of iterations
      maxIter - the largest number of iterations
      ghostCorrection - fix the uniform portion and reconstruct only the rest
      verbose - report the fit of every pole figure in every iteration
    See also
      MLSSolver, calcODF, PoleFigure2ODFGhostCorrection"""
    if minIter is not None:
      self.minIter = int(minIter)
    if maxIter is not None:
      self.maxIter = max(int(maxIter), self.minIter)
    if verbose is not None:
      self.verbose = verbose
    ghost = self.ghostCorrection if ghostCorrection is None else ghostCorrection
    self.init()
    self.iterate()
    n = len(self.ops)
    alpha = self.alpha[:n].copy()
    c0 = min(np.quantile(np.maximum(I.ravel(), 0), 0.01) / alpha[i] for i, I in enumerate(self.pf.allI))
    if ghost and 0.1 < c0 < 0.99:
      if self.verbose:
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

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from typing import Any
    @property
    def frameA(self) -> Any:
      ...
    @frameA.setter
    def frameA(self, value) -> None:
      ...
    @property
    def frameB(self) -> Any:
      ...
    @frameB.setter
    def frameB(self, value) -> None:
      ...


def calcODF(pf, *args, resolution=None, halfwidth=None, kernel=None, S3G=None, zeroRange=False, minIter=5, maxIter=10, c0=None,
            ghostCorrection=True, quadratureWeights=False, intensityWeights=False, regularisation=0.0, verbose=False):
  """an ODF reconstructed from pole figure intensities by the modified least squares method,
  with ghost correction and, on request, the zero range method

    odf = calcODF(pf)                                            # the defaults
    odf = calcODF(pf, halfwidth=7.5*degree, resolution=5*degree)  # the halfwidth and the grid
    odf = calcODF(pf, psi)                                       # the ansatz functions of the kernel psi
    odf = calcODF(pf, zeroRange=True)                            # drop the orientations the data says are empty
    odf = calcODF(pf, ghostCorrection=False)                     # keep the uniform portion in the components
    odf = calcODF(pf, solver)                                    # run an MLSSolver set up before

  The ODF is a weighted sum of many unimodal ODFs on a grid, whose pole figures are fitted to
  the intensities in the least squares sense, every pole figure scaled by its own factor.
  The ghost correction fixes the uniform portion the data allows before the fit; the zero
  range method removes the components where the measured intensities vanish. MATLAB's
  second output, the scaling factors alpha, comes from an MLSSolver, solver.alpha after
  solver.calcODF(). A kernel or a grid of orientations may be passed positionally.

  Input
    pf - PoleFigure
    psi - SO3Kernel, the shape of the ansatz functions
    solver - MLSSolver, run with its own settings
  Output
    odf - SO3FunRBF
  Keywords
    resolution - the spacing of the grid the ansatz functions sit on, by default the
        resolution of the pole figures (MATLAB: 3/2 of it)
    halfwidth - the halfwidth of the ansatz functions, by default the resolution of the grid
        (MATLAB: 2/3 of it)
    kernel - SO3Kernel, the shape of the ansatz functions instead of the de la Vallee Poussin
        kernel of the halfwidth
    S3G - orientation, the centres of the ansatz functions instead of the equispaced grid
    zeroRange - apply the zero range method: drop the centres where some pole figure says the
        ODF vanishes; a zeroRangeMethod gives its parameters
    minIter - the least number of iterations
    maxIter - the largest number of iterations (MATLAB iterMax)
    c0 - the coefficients of the ansatz functions the iteration starts from, by default all equal
    ghostCorrection - fix the uniform portion the data allows and reconstruct only the rest;
        MATLAB flag noGhostCorrection is ghostCorrection=False
    quadratureWeights - weight the residuals with the quadrature weights of the measured
        directions (no specimen symmetry)
    intensityWeights - weight the residuals with (1 + I)^(-1/2), the counting statistics of the
        intensities I
    regularisation - the weight of a penalty on the Dirichlet energy, a discrete Laplacian on
        the grid; zero for none
    verbose - report the fit of every pole figure in every iteration

  Example
  Reconstruct a model ODF from three simulated pole figures, then compare the input and
  reconstructed intensities. The sampling grid is deliberately coarse, and the fit uses 10-degree orientation spacing.

    import numpy as np
    import matplotlib.pyplot as plt
    from mtex import (crystalFrame, orientation, degree, unimodalODF, Miller,
                      equispacedS2Grid, calcODF, calcError, plotPF)

    cs = crystalFrame('m-3m')
    centre = orientation.byEuler(35*degree, 45*degree, 10*degree, cs)
    model = unimodalODF(centre, halfwidth=20*degree)
    h = [Miller(1, 0, 0, cs), Miller(1, 1, 0, cs), Miller(1, 1, 1, cs)]
    r = equispacedS2Grid(resolution=20*degree, antipodal=True)
    pf = model.calcPoleFigure(h, r)
    reconstructed = calcODF(pf, resolution=10*degree, halfwidth=20*degree,
                            minIter=5, maxIter=12, ghostCorrection=False)
    fitted = reconstructed.calcPoleFigure(h, r)
    print('Relative pole-figure errors:', np.round(calcError(pf, reconstructed), 3))
    limit = float(np.max(pf.intensities))
    plotPF(pf, colorRange=(0, limit))
    plotPF(fitted, colorRange=(0, limit))
    plt.show()

  The first figure contains the simulated measurements and the second their reconstruction,
  on the same colour scale. The printed residuals include the coarse-grid approximation. A small
  pole-figure residual alone does not establish that an ODF is uniquely determined by the data.

  References
    R. Hielscher, H. Schaeben, A novel pole figure inversion method: specification of the MTEX
        algorithm, J. Appl. Cryst. 41 (2008) 1024-1037
  See also
    MLSSolver, calcODFIterative, calcError, PoleFigure2ODF, PoleFigure2ODFGhostCorrection,
    PoleFigureDubna, PoleFigureTutorial"""
  solvers = [a for a in args if isinstance(a, MLSSolver)]
  if solvers:
    return solvers[0].calcODF()
  return MLSSolver(pf, *args, resolution=resolution, halfwidth=halfwidth, kernel=kernel, S3G=S3G, zeroRange=zeroRange,
                   minIter=minIter, maxIter=maxIter, c0=c0, ghostCorrection=ghostCorrection,
                   quadratureWeights=quadratureWeights, intensityWeights=intensityWeights,
                   regularisation=regularisation, verbose=verbose).calcODF()


# ------------------------------------------------------------------ successive refinement

def calcODFIterative(pf, *args, resolution=None, halfwidth=None, kernel=None, maxSteps=5, thinning=0.025,
                     verbose=False, **kwargs):
  """an ODF reconstructed from pole figures by a ladder of kernel widths, from wide to the
  target, each step started from the last

    odf = calcODFIterative(pf)                          # the defaults
    odf = calcODFIterative(pf, halfwidth=5*degree)      # the target halfwidth
    odf = calcODFIterative(pf, thinning=0)              # every node of every grid kept

  Where the pole figures are sampled very irregularly, a coarse model fills the gaps with
  volume along the fibres it justifies, and finer models refine it. The method is an
  elaborate start for calcODF and acts as a regularisation.

  The halfwidths run from sqrt(e)^(maxSteps-1) times the target down to the target, the
  steps whose grid would be coarser than 45 degree left out. Each step solves on an
  equispaced grid of its own resolution, starting from the previous ODF evaluated on that
  grid (MATLAB interpolates it), the first from the uniform ODF. Before every solve the
  nodes whose share is below thinning times the uniform one are removed.

  MATLAB's second output alpha is not returned; MATLAB's flag nothinning is thinning=0.
  Further keywords go to MLSSolver.

  Input
    pf - PoleFigure
  Output
    odf - SO3FunRBF
  Keywords
    resolution - the resolution of the finest grid, by default that of the pole figures
    halfwidth - the target halfwidth of the ansatz functions, by default the resolution
    kernel - SO3Kernel, whose type gives the ansatz functions and whose halfwidth the target,
        instead of the de la Vallee Poussin kernel
    maxSteps - the number of halfwidths of the ladder
    thinning - remove the nodes below this fraction of the uniform share; 0 keeps all
    verbose - report the fit of every pole figure in every step
  References
    F. Bachmann, Texturbestimmung aus Beugungsbildern, in: Optimierung der Goniometrie zur
        Texturbestimmung aus Roentgenbeugungsbildern, Springer Spektrum, Wiesbaden (2016)
  See also
    calcODF, MLSSolver, PoleFigureRefinement"""
  kernels = [a for a in args if isinstance(a, SO3Kernel)]
  res = pf.resolution if resolution is None else resolution
  hw = res if halfwidth is None else halfwidth
  psi = kernels[0] if kernels else kernel
  if psi is not None:
    hw = psi.halfwidth()
  kernelType = SO3DeLaValleePoussinKernel if psi is None else type(psi)
  res = min(res, hw)
  ladder = [math.sqrt(math.e) ** (k - 1) for k in range(maxSteps, 0, -1)]
  steps = [(x * hw, x * res) for x in ladder if x * res <= 45 * degree]
  odf = uniformODF(pf.CS, pf.SS)
  for k, (h, r) in enumerate(steps):
    grid = equispacedSO3Grid(pf.CS, pf.SS, resolution=r)
    c = np.maximum(np.real(odf(grid)), 0)
    c = c / np.sum(c)
    if thinning:
      keep = c > min(1.0, thinning) / c.size
      grid = grid.subGrid(keep)
      c = c[keep]
    solver = MLSSolver(pf, S3G=grid, kernel=kernelType(halfwidth=h), c0=c, verbose=verbose, **kwargs)
    odf = solver.calcODF()
    if verbose:
      e = calcError(pf, odf)
      print(f"  {k + 1:2d} : {grid.size:7d} | " + "  ".join(f"{x:.2f}" for x in e))
  return odf


# the two frames under their names of the Julia port, frameA and frameB
installFramePair(MLSSolver)

__all__ = ["MLSSolver", "zeroRangeMethod", "calcODF", "calcODFIterative"]
