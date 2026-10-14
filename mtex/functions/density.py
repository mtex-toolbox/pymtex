# the density estimates on SO(3), the sphere and the circle and the least squares approximation
# from samples (MATLAB calcDensity, interpolate;
# docs/plans/functions.md F4)
#
#   calcDensity(ori | ebsd, halfwidth=, kernel=, weights=, groups=, harmonic=None, bandwidth=)
#   calcDensity(v, halfwidth=, kernel=, weights=, groups=, bandwidth=)
#   without a halfwidth or kernel the halfwidth is selected, calcKernel(x, 'UCV')
#   calcDensity(rho, periodic=True, sigma=, kernel=, weights=, antipodal=)
#   calcDensity(x, range=, bandwidth=, weights=)    real valued data, one sample per row
#   interpolate(ori | v, values, bandwidth=, lambda_=, s=, weights=)
from __future__ import annotations

import math
import warnings

import numpy as np

from ..geometry.arguments import refuseFlags
from ..geometry.frames import crystalFrame
from ..geometry.kernels import blasThreads
from ..geometry.orientation import orientation
from ..geometry.quaternion import quaternion, rotation
from ..geometry.session import getMTEXpref
from ..geometry.vector3d import vector3d, asVector3d
from .funops import conv
from .harmonics import _s2PerDegree, _so3PerDegree, deg2dim, dim2deg, invariantProjection, project
from .kde import calcDensityReal
from .kernels import _groupSums
from .kernelfunctions import SO3DirichletKernel
from .rbf import SO3FunRBF, _densityKernel, uniformODF
from .s1fun import calcDensity as _calcDensityS1
from .s2fun import S2FunHarmonic
from .so3fun import SO3FunHarmonic, invertedCoefficients
from .transforms import S2FourierPlan, SO3FourierPlan


def calcDensity(x, Fourier=False, **kwargs):
  """estimate a smooth density from orientations, directions, angles or real-valued samples.

    odf = calcDensity(ori)                                 # select the smoothing width from the orientations
    odf = calcDensity(ori, halfwidth=5*degree)              # use a fixed angular halfwidth
    odf = calcDensity(ori, kernel=SO3AbelPoissonKernel(halfwidth=10*degree))  # choose another kernel family
    odf = calcDensity(grains.meanOrientation, weights=grains.area)  # weight each grain by its area
    odf = calcDensity(ebsd)                                # use a single-phase map with grain ids
    odf = calcDensity(ori, harmonic=True, bandwidth=16)     # return a harmonic expansion through degree 16
    sF = calcDensity(v, halfwidth=10*degree, weights=w)      # estimate a weighted directional density
    sF = calcDensity(v, halfwidth=10*degree, antipodal=True)  # treat opposite directions as the same axis
    f = calcDensity(rho, periodic=True, sigma=5*degree)     # estimate a density of angles on the circle
    f = calcDensity(x, range=[0, 1])                       # estimate a density of real values on an interval
    f = calcDensity(X, range=[[0, 0], [1, 1]], rng=rng)    # estimate a joint density from rows of (x, y) samples

  The estimate replaces the sample cloud with a smooth function. Narrow kernels retain
  finer detail; wider kernels suppress more sampling noise. Use odf(oriQuery),
  sF(vQuery) or f(xQuery) to evaluate it. For a joint density in two dimensions, use
  f(xQuery, yQuery); the coordinate arrays broadcast together.

  The input determines the domain and the meaning of the density:

  | Input | Estimate | Normalisation |
  |---|---|---|
  | orientation or rotation | orientation distribution function (ODF) on SO(3) | mean one; a value of two is twice the uniform density |
  | vector3d or Miller | density of directions or axes on the sphere | mean one, with the input's frame symmetry |
  | numeric angles with periodic=True | density on the circle, with period 2*pi | mean one, with angles in radians |
  | numeric array without circular options | density of real values, one sample per row | probability density in the units of the data |

  For orientations and directions, an explicit halfwidth uses a de la Vallee Poussin
  kernel; halfwidth is the angle at which the kernel falls to half its peak. Without a
  halfwidth or kernel, `calcKernel` selects the width by unbiased cross validation with
  a noise floor. Angular halfwidths are in radians; write 5*degree for five degrees.

  For an EBSD map, select one phase before calling. Its grain ids are used automatically
  when available, so neighbouring pixels within a grain are grouped during width selection.
  Map orientations without grain groups use a halfwidth of 10 degrees and issue a warning.
  Automatic selection also falls back to 10 degrees when it cannot be performed, for
  example with fewer than two valid points. NaN orientations and directions are omitted;
  if none remain, their density is uniform. These empty-sample rules do not apply to
  numeric angles or real-valued data.

  Numeric arrays are interpreted as real-valued data by default. Use periodic=True for
  angles; supplying sigma, kernel or antipodal also selects the circle estimator, even
  when antipodal=False. Circular data use a Gaussian kernel with Silverman's width rule
  unless sigma or kernel is supplied. Real-valued data use a diffusion estimate in one
  dimension and a fitted Gaussian mixture in several dimensions.

  Additional options depend on the input. Omitted weights give every sample equal weight;
  supplied weights are normalised internally.

  | Option | Meaning and default |
  |---|---|
  | halfwidth | orientations, directions: angular kernel halfwidth in radians; selected automatically if neither halfwidth nor kernel is given |
  | kernel | orientations, directions, angles: SO3Kernel, S2Kernel or S1Kernel matching the domain; takes precedence over halfwidth or sigma |
  | weights | orientations, directions, angles, one-dimensional real data: one nonnegative weight per sample, with a positive total |
  | groups | orientations, directions: one group id per sample for automatic width selection; groups are treated as independent units |
  | antipodal | directions, angles: identify v with -v, or rho with rho+pi; default False, while axes already marked on vector3d remain axes |
  | harmonic | orientations: True returns SO3FunHarmonic; False keeps the kernel sum as SO3FunRBF; by default use harmonics above 200 orientations when the kernel bandwidth is at most 96, with a Dirichlet kernel, or when both frames are crystal frames |
  | bandwidth | orientations, directions, angles: maximum harmonic degree, controlling expansion resolution rather than the smoothing width; defaults depend on the kernel and MTEX bandwidth preferences |
  | bandwidth | one-dimensional real data: Gaussian smoothing width in the units of the data; selected automatically by default |
  | periodic | numeric angles: True selects the circular estimator; default False |
  | sigma | numeric angles: standard deviation of the wrapped Gaussian in radians; selected by Silverman's rule by default |
  | range | real-valued data: [lo, hi] in one dimension, or [minima, maxima] with one column per dimension; defaults to the data's bounding box |
  | rng | multidimensional real data: numpy.random.Generator for the Gaussian mixture's random initialisation; a new generator by default |

  Multidimensional real data currently ignore weights and bandwidth. For orientations,
  bandwidth is used when the estimate is built with harmonic=True; Fourier=True converts
  the result to harmonics afterwards. Use harmonic=True to request a specific degree.
  MATLAB's 'order' is bandwidth; for a Bingham model use `calcBinghamODF`.

  Input
    x - orientations or rotations, directions or Miller indices, a single-phase EBSD map,
        an array of angles, or a real array of shape (n,) or (n, d)
  Output
    f - callable density: SO3FunRBF or SO3FunHarmonic for orientations, S2FunHarmonic for
        directions, S1FunHarmonic for angles, or griddedDensity for real-valued data
  Keywords
    Fourier - convert the orientation density to SO3FunHarmonic after estimation;
        has no effect for other input types; MATLAB's flag 'Fourier'
  See also
    calcKernel, calcBinghamODF, unimodalODF, interpolate, calcDensityReal, DensityEstimation
  """
  # a map: the orientations of its one phase, with its grain ids when grains were computed
  if not isinstance(x, (quaternion, vector3d, np.ndarray, list, tuple)) and hasattr(x, "orientations") and hasattr(x, "prop"):
    if kwargs.get("groups") is None and "grainId" in x.prop:
      kwargs["groups"] = np.asarray(x.prop["grainId"]).reshape(-1)
    x = x.orientations
  if isinstance(x, quaternion):
    f = _calcDensitySO3(x, **kwargs)
    # MATLAB's 'Fourier' asks for the harmonic form
    return SO3FunHarmonic(f) if Fourier and not isinstance(f, SO3FunHarmonic) else f
  if isinstance(x, vector3d):
    return _calcDensityS2(x, **kwargs)
  # angles are periodic, the flag or an option of the circle says so; other numbers are real
  # valued data, MATLAB's kernel density estimate on an interval or a box
  if kwargs.pop("periodic", False) or any(k in kwargs for k in ("sigma", "kernel", "antipodal")):
    return _calcDensityS1(x, **kwargs)
  return calcDensityReal(x, **kwargs)


# the ODF of a list of orientations: the kernel form up to 200 orientations, the harmonic
# form above or on request, an empty list the uniform ODF; NaN orientations are dropped
def _calcDensitySO3(o, kernel=None, halfwidth=None, weights=None, groups=None, harmonic=None, bandwidth=None):
  o = o if isinstance(o, orientation) else orientation(o)
  psi = _densityKernel("SO3", kernel, halfwidth) if kernel is not None or halfwidth is not None else _selectedKernel(o, weights, groups)
  q = o.data.reshape(-1, 4)
  w = np.ones(len(q)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
  keep = ~np.any(np.isnan(q), axis=1)
  q, w = q[keep], w[keep]
  if len(q) == 0:
    return uniformODF(o.CS, o.SS, antipodal=o.antipodal)
  w = w / np.sum(w)
  f = SO3FunRBF(orientation._fromRotation(q, (o.CS, o.SS), o.antipodal), psi, w / psi.mean())
  if harmonic is None:
    harmonic = (len(q) > 200 and psi.bandwidth <= 96) or isinstance(psi, SO3DirichletKernel) or isinstance(o.SS, crystalFrame)
  if not harmonic:
    return f
  return SO3FunHarmonic(f, bandwidth=min(psi.bandwidth, getMTEXpref("maxSO3Bandwidth")) if bandwidth is None else bandwidth)


# the density of directions or axes on the sphere: the point measure convolved with the
# kernel, projected by the frame, even for axes
def _calcDensityS2(v, kernel=None, halfwidth=None, weights=None, groups=None, bandwidth=None, antipodal=False):
  # the estimate of axes is even; the flag on the data or on the call says so
  antipodal = bool(antipodal) or bool(v.antipodal)
  psi = _densityKernel("S2", kernel, halfwidth) if kernel is not None or halfwidth is not None else \
      _selectedKernel(vector3d(v.data, frame=v.frame, antipodal=antipodal), weights, groups)
  d = v.data.reshape(-1, 3)
  w = np.ones(len(d)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
  # the lengths, NaN where a coordinate is, so that one pass drops those and normalises the rest
  r = np.sqrt(np.einsum("ij,ij->i", d, d))
  keep = ~np.isnan(r)
  d, w, r = d[keep], w[keep], r[keep]
  if len(d) == 0:
    return S2FunHarmonic([math.sqrt(4 * np.pi)], v.frame, real=True, antipodal=antipodal, _internal=True)
  L = min(getMTEXpref("maxS2Bandwidth") if bandwidth is None else int(bandwidth), psi.bandwidth)
  pts = vector3d(d / r[:, None], frame=v.frame, antipodal=antipodal)
  s = S2FunHarmonic.quadrature(pts, np.ones(len(d)), bandwidth=L, weights=4 * np.pi * w / np.sum(w), real=True, antipodal=antipodal)
  return conv(s, psi)


# the kernel selected from the sample by UCV with its noise floor; MATLAB's 10 degrees for the
# pixels of a map without their grains, with a warning, and for fewer than two points
def _selectedKernel(x, weights, groups):
  from .characteristics import calcKernel
  space = "S2" if isinstance(x, vector3d) else "SO3"
  if isinstance(x, orientation) and x.notIID and groups is None:
    warnings.warn("the orientations come from a map, whose neighbouring pixels are not independent, so the halfwidth "
                  "was not selected and is 10 degrees; pass the map with its grains computed, groups=ebsd.grainId, "
                  "or a halfwidth", stacklevel=4)
    return _densityKernel(space, None, None)
  try:
    return calcKernel(x, "UCV", weights=weights, groups=groups)
  except ValueError:
    return _densityKernel(space, None, None)


# ------------------------------------------------------------------ least squares from samples
# min sum w_i |f(R_i) - y_i|^2 + lambda sum_n (1 + n(n+1))^s |fhat_n|^2, by LSQR on the
# operator and its adjoint


def _sobolevWeightsSO3(L, s):
  n = np.arange(L + 1)
  return _so3PerDegree((1.0 + n * (n + 1)) ** s)


def _sobolevWeightsS2(L, s):
  l = np.arange(L + 1)
  return _s2PerDegree((1.0 + l * (l + 1)) ** s)


def _lsqr(A, At, b, n, maxIter, tol):
  return _lsqrFull(A, At, b, n, maxIter, tol)[0]


# the solution, LSQR's flag, the relative residual and the iterations; the operators are
# transforms and the iteration's norms and products of vectors run on BLAS, whose waiting
# threads would take the transforms' cores (docs/research/threads.md)
def _lsqrFull(A, At, b, n, maxIter, tol):
  from .lsqr import lsqr
  with blasThreads(1):
    return lsqr(A, At, np.asarray(b, dtype=complex), n, tol=tol, maxIter=maxIter)


def interpolate(nodes, values=None, *, bandwidth=None, lambda_=None, s=2, weights=None, maxIter=100, tol=None,
                withParameters=False):
  """the function in harmonic form fitted by least squares to values at nodes: on SO(3) at
  orientations, on the sphere at directions, a vector field for vector values or tangent vectors

    SO3F = interpolate(ori, y)                                  # an SO(3) function through values at orientations
    SO3F = interpolate(ori, y, bandwidth=8, weights='equal')    # of a given degree, the nodes weighted equally
    SO3F, (flag, residual, nIter) = interpolate(ori, y, withParameters=True)  # with LSQR's flag, residual and iterations
    sF = interpolate(v, y, bandwidth=16, lambda_=1e-4)          # a function on the sphere through values at directions
    sVF = interpolate(v, w)                                     # a vector field on the sphere through vectors w
    SO3VF = interpolate(SO3TangentVector(w, ori), bandwidth=8)  # a vector field on SO(3) through tangent vectors
    SO3VF = interpolate(SO3TangentVector(w, ori), lambda_=0)    # without regularisation

  The harmonic coefficients up to the bandwidth minimise the weighted squared misfit at the
  nodes plus lambda_ times the Sobolev norm of order s, solved by LSQR. Without a bandwidth it
  follows from the number of nodes, on SO(3) at most the preference 'maxSO3Bandwidth'.

  On SO(3) the fit is posed among the functions with the symmetries of the nodes. Nodes closer
  than 0.02 are merged into one carrying the mean of their values.

  Values with further axes after the node axis give one function per component, and the LSQR
  parameters then arrays of one per component. A vector field on SO(3) is fitted componentwise,
  three SO3FunHarmonic, as MATLAB's SO3VectorFieldHarmonic.interpolate; MATLAB passes the
  tangent vectors as values, here they carry their orientations.

  Input
    nodes - orientation or rotation, vector3d, or SO3TangentVector for a vector field on SO(3)
    values - ndarray of one value per node, or vector3d for a vector field on the sphere
  Output
    SO3F - SO3FunHarmonic, S2FunHarmonic, S2VectorFieldHarmonic or SO3VectorFieldHarmonic
  Keywords
    bandwidth - the largest harmonic degree; set by hand it may undersample
    lambda_ - the regularisation parameter, 1e-8 on SO(3) by default, 0 for none; MATLAB's
        'regularization'
    s - the order of the Sobolev norm; MATLAB's 'SobolevIndex'
    weights - the weights of the nodes: by default their Voronoi volumes below 10000 nodes, else
        equal; 'equal', or one number per node
    maxIter - the largest number of LSQR iterations
    tol - the tolerance of LSQR, 1e-3 on SO(3) and 1e-6 on the sphere by default
    withParameters - return also LSQR's flag, 1 when the iterations ran out, the relative
        residual and the number of iterations, for a function on SO(3) or the sphere
  See also
    SO3FunHarmonic.interpolate, S2FunHarmonic.interpolate, SO3FunRBF.interpolate,
    calcDensity, SO3FunVectorValued, HarmonicApproximationTheory"""
  from ..geometry.tangent import SO3TangentVector
  if nodes.size == 0:
    raise ValueError("there are no nodes to interpolate")
  if isinstance(nodes, SO3TangentVector):
    from .so3vectorfield import interpolateField
    return interpolateField(nodes, bandwidth=bandwidth, lambda_=lambda_, s=s, weights=weights, maxIter=maxIter, tol=tol)
  if isinstance(nodes, vector3d) and isinstance(values, vector3d):
    from .s2vectorfield import interpolateField
    return interpolateField(nodes, values, bandwidth=bandwidth, lambda_=lambda_, s=s, weights=weights, maxIter=maxIter, tol=tol)
  y = np.asarray(values).ravel()
  real = y.dtype.kind != "c"
  # MATLAB's tolerance of LSQR: 1e-3 on SO(3), 1e-6 on the sphere
  tol = (1e-3 if isinstance(nodes, quaternion) else 1e-6) if tol is None else tol
  if isinstance(nodes, quaternion) and np.size(values) != nodes.size:
    # values with component axes after the node axis: one fit per component, a list of functions
    vals = np.asarray(values)
    cols = vals.reshape(nodes.size, -1)
    fits = [interpolate(nodes, cols[:, k], bandwidth=bandwidth, lambda_=lambda_, s=s, weights=weights, maxIter=maxIter, tol=tol,
                        withParameters=True) for k in range(cols.shape[1])]
    L = max(f.bandwidth for f, _ in fits)
    fhat = np.stack([f.truncate(L)._fhat for f, _ in fits]).reshape(vals.shape[1:] + (fits[0][0].truncate(L)._fhat.shape[-1],))
    f = fits[0][0]._rebuild(fhat, _isReal=all(f.isReal for f, _ in fits))
    return (f, _parameters(fits, vals.shape[1:])) if withParameters else f
  if isinstance(nodes, quaternion):
    o = nodes if isinstance(nodes, orientation) else orientation(nodes)
    # nodes closer than 0.02 are one node carrying the mean of their values, as MATLAB's
    o, iu = o.flatten().unique(tol=0.02, return_inverse=True)
    count = np.bincount(iu, minlength=o.size)
    y = _groupSums(iu, y, o.size) / count
    if weights is not None and not isinstance(weights, str):
      weights = np.bincount(iu, np.asarray(weights, dtype=float).ravel(), o.size)
    lam = 1e-8 if lambda_ is None else lambda_
    q = o.data.reshape(-1, 4)
    nA, nB = o.CS.sym.properGroup().rotations().size, o.SS.sym.properGroup().rotations().size
    nSym = nA * nB * (2 if real else 1)
    L = min(dim2deg(max(1, round(len(q) * nSym / 2))), getMTEXpref("maxSO3Bandwidth")) if bandwidth is None else int(bandwidth)
    # the nodes weighted by their Voronoi volumes below 10000 of them, else equally, as MATLAB's
    if (isinstance(weights, str) and weights == "equal") or (weights is None and len(q) >= 1e4):
      w = np.full(len(q), 1 / len(q))
    elif weights is None or isinstance(weights, str):
      from .rbfinterpolation import voronoiVolumes
      w = voronoiVolumes(o)
    else:
      w = np.asarray(weights, dtype=float).ravel() / np.sum(weights)
    sw = np.sqrt(w)
    p = SO3FourierPlan(rotation(q), L, real=False)
    reg = math.sqrt(lam) * np.sqrt(_sobolevWeightsSO3(L, s))
    # the unknown is a symmetric function, so the fit is posed in the invariant subspace: a
    # free fit projected afterwards is the fit divided by the order of the group wherever
    # the samples cover one fundamental region and leave the copies unconstrained
    P, Q = invariantProjection(o.SS, L), invariantProjection(o.CS, L)

    def sym(x):
      v = project(np.array(x, dtype=complex), P, Q)
      return (v + invertedCoefficients(v)) / 2 if o.antipodal else v

    def A(x):
      v = sym(x)
      return np.concatenate([sw * p(v), reg * v])

    def At(z):
      return sym(p.adjoint(sw * z[:len(q)], np.ones(len(q))) + reg * z[len(q):])
    b = np.concatenate([sw * y, np.zeros(len(reg))]).astype(complex)
    x, flag, relres, itn = _lsqrFull(A, At, b, deg2dim(L + 1), maxIter, tol)
    f = SO3FunHarmonic(sym(x), o.CS, o.SS, real=real, antipodal=o.antipodal, _internal=True)
    if withParameters:
      # MATLAB's lsqr parameters: the flag, 1 when the iterations ran out, the relative
      # residual and the number of iterations
      return f, (flag, relres, itn)
    return f
  v = asVector3d(nodes).flatten().normalize()
  # values with component axes after the node axis: one fit per component, a list of functions
  values = np.asarray(values)
  if values.size != v.size:
    rest = values.shape[1:]
    cols = values.reshape(v.size, -1)
    fits = [interpolate(v, cols[:, k], bandwidth=bandwidth, lambda_=lambda_, s=s, weights=weights, maxIter=maxIter, tol=tol,
                        withParameters=True) for k in range(cols.shape[1])]
    fhat = np.stack([f._fhat for f, _ in fits]).reshape(rest + fits[0][0]._fhat.shape)
    f = S2FunHarmonic(fhat, v.frame, real=all(f.isReal for f, _ in fits), antipodal=v.antipodal, _internal=True)
    return (f, _parameters(fits, rest)) if withParameters else f
  lam = 0.0 if lambda_ is None else lambda_
  L = math.ceil(math.sqrt(v.size / 2)) if bandwidth is None else int(bandwidth)
  # the nodes weighted by the areas of their Voronoi cells, so that dense regions do not
  # dominate the fit, or equally
  if isinstance(weights, str) and weights == "equal":
    w = np.full(v.size, 1 / v.size)
  elif weights is None:
    from .polefigure import _voronoiWeights
    w = _voronoiWeights(v)
    w = np.full(v.size, 1 / v.size) if w is None else np.asarray(w, dtype=float).ravel() / np.sum(w)
  else:
    w = np.asarray(weights, dtype=float).ravel() / np.sum(weights)
  sw = np.sqrt(w)
  p = S2FourierPlan(v, L)
  reg = math.sqrt(lam) * np.sqrt(_sobolevWeightsS2(L, s))
  def A(x):
    return np.concatenate([sw * p(np.asarray(x, dtype=complex)), reg * x])
  def At(z):
    return p.adjoint(sw * z[:v.size], np.ones(v.size)) + reg * z[v.size:]
  b = np.concatenate([sw * y, np.zeros(len(reg))]).astype(complex)
  fhat, flag, relres, itn = _lsqrFull(A, At, b, (L + 1) ** 2, maxIter, tol)
  f = S2FunHarmonic(fhat, v.frame, real=real, antipodal=v.antipodal, _internal=True)
  return (f, (flag, relres, itn)) if withParameters else f


# the LSQR parameters of the fits of the components, an array of each over the component axes
def _parameters(fits, shape):
  return tuple(np.array([p[i] for _, p in fits]).reshape(shape) for i in range(3))


def interpS2(v, y, *args, symmetry=True, **kwargs):
  """interp(v, y, 'linear'): a function through values at directions, by the method named,
  MATLAB's vector3d/interp; with directions vi after the values, its values there; symmetry=False
  interpolates a function of a crystal frame without its group"""
  from .handles import S2FunHandle
  from .s2funtri import S2FunTri
  vi = next((a for a in args if isinstance(a, vector3d)), None)
  methods = ("linear", "nearest", "harmonic", "spline", "inverseDistance")
  refuseFlags(args, "interp", names=methods)
  flags = {a for a in args if isinstance(a, str)}
  symmetric = symmetry and getattr(v.frame, "sym", None) is not None and v.frame.sym.rotations().size > 1
  if symmetric and not flags & {"linear", "nearest", "inverseDistance"}:
    flags.add("harmonic")
  if vi is not None and "linear" not in flags and not flags & {"harmonic", "nearest", "inverseDistance"}:
    flags.add("spline")
  if isinstance(y, vector3d):
    if "harmonic" not in flags:
      raise NotImplementedError("a vector field through vectors is interpolated with 'harmonic'")
    sF = interpolate(v, y, **kwargs)
  else:
    y = np.asarray(y).reshape(v.size, -1)
    if y.shape[1] == 1:
      y = y[:, 0]
    nodes = v.flatten()
    if "nearest" in flags:
      res = nodes.resolution

      def nearest(w):
        ind, d = nodes.find(w)
        out = y[np.asarray(ind).ravel()].astype(float)
        out[np.asarray(d).ravel() > 2 * res] = np.nan
        return out.reshape(w.shape)
      sF = S2FunHandle(nearest, v.frame)
    elif "harmonic" in flags:
      sF = interpolate(v, y, **kwargs)
    elif flags & {"spline", "inverseDistance"}:
      from .kernelfunctions import S2DeLaValleePoussinKernel
      psi = S2DeLaValleePoussinKernel(halfwidth=nodes.resolution / 2)
      V = nodes.normalize().data.reshape(-1, 3)

      # the mean of the four nearest values, weighted by the kernel or the inverse distance
      def spline(w):
        W = w.normalize().data.reshape(-1, 3)
        omega = np.arccos(np.clip(W @ V.T, -1, 1))
        k = min(4, omega.shape[1])
        j = np.argpartition(omega, k - 1, axis=1)[:, :k]
        so = np.take_along_axis(omega, j, axis=1)
        M = np.minimum(1 / so, 1e10) if "inverseDistance" in flags else np.real(psi.eval(np.cos(so)))
        M = M / M.sum(axis=1, keepdims=True)
        return np.sum(M * y[j], axis=1).reshape(w.shape)
      sF = S2FunHandle(spline, v.frame)
    else:
      sF = S2FunTri(v, y)
  return sF.eval(vi) if vi is not None else sF


__all__ = ["calcDensity", "interpolate"]
