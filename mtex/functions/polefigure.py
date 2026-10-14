# measured pole figure intensities (MATLAB PoleFigureAnalysis/@PoleFigure; docs/plans/functions.md P4a)
#
#   pf = PoleFigure(h, r, I)                                one pole figure, h one or several superposed directions
#   pf = PoleFigure([h1, h23], [r1, r2], [I1, I2], cs, ss, superposition=[1, [0.52, 1.23]])
#   pf.allH, pf.allR, pf.allI, pf.c                         one entry per pole figure
#   pf.h, pf.r, pf.intensities, pf.CS, pf.SS, pf.antipodal, pf.numPF
#   pf.select(i), pf[i], pf[mask]                           a pole figure, the measurements a mask keeps
#   calcPoleFigure(odf, h, r), calcError(pf, odf, 'RP'), correct(pf, background=bg), mean(pf)
#
# One pole figure is one diffraction measurement: the crystal directions it diffracts on
# with their structure coefficients, the specimen directions and the intensities. A data
# set is one container holding its pole figures as parallel lists, MATLAB's allH, allR,
# allI, c; the frames are those of the directions, the crystal frame that of allH and the
# specimen frame that of allR. Diffraction cannot tell r from -r, so the specimen
# directions are axes unless the measurement is complete.
from __future__ import annotations

from typing import TYPE_CHECKING
import math

import numpy as np

from .quadrature import clenshawCurtis
from ..geometry._array import MtexArray
from ..geometry.frames import crystalFrame, referenceFrame, specimenFrame
from ..geometry.grids.s2grid import S2Grid
from ..geometry.kernels import blasThreads
from ..geometry.products import dropGroup
from ..geometry.symmetryops import _images
from ..geometry.quaternion import rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d, asVector3d
from .characteristics import calcError as _calcErrorODF
from .s2fun import S2FunHarmonic
from ..geometry.install import install
from ..geometry.display import display, framesLabel
from ..geometry.display import aligned

ERROR_MODES = ("relative", "RP", "l1", "l2")


# the error measure of pole figures by its name in any case, as MATLAB's flags
def _errorMode(mode):
  names = {m.lower(): m for m in ERROR_MODES}
  if str(mode).lower() not in names:
    raise ValueError(f"the error mode is 'relative', 'RP', 'l1' or 'l2', not {mode!r}")
  return names[str(mode).lower()]


# ------------------------------------------------------------------ the directions

# crystal directions as a flat list in a crystal frame, the frame given taking over
def _millerList(h, CS):
  v = asVector3d(h)
  v = v.flatten()
  if CS is not None:
    v = vector3d(v)
    if v.isCrystalDirection:
      v.CS = CS
    else:
      v.frame = CS
  if not v.isCrystalDirection:
    raise ValueError("the pole figure directions have to be given in a crystal frame")
  return v


# specimen directions as axes unless complete, in a specimen frame, keeping their shape
def _specimenDirections(r, complete, SS):
  v = vector3d(r) if isinstance(r, vector3d) else vector3d(np.asarray(r, dtype=float))
  if SS is not None:
    v.frame = SS
  elif v.frame is None:
    v.frame = specimenFrame.default
  if isinstance(v.frame, crystalFrame):
    raise ValueError("the specimen directions of a pole figure live in a specimen frame")
  if not complete:
    v.antipodal = True
  return v


def _coefficients(c, n):
  if c is None:
    return np.full(n, 1 / n)
  c = np.asarray(c, dtype=float).reshape(-1)
  if c.size != n:
    raise ValueError("one superposition coefficient per crystal direction")
  return c


def _isBoolMask(key):
  if isinstance(key, np.ndarray):
    return key.dtype == bool
  return isinstance(key, (list, tuple)) and len(key) > 0 and all(isinstance(k, (bool, np.bool_)) for k in key)


# the angle the point density of a direction list stands for: the step of a grid, the
# polar step of a list measured on evenly spaced rings, and the area per point otherwise
def _resolutionOf(r):
  if isinstance(r, S2Grid):
    return float(r.resolution)
  theta = np.unique(np.round(r.flatten().theta / 1e-6) * 1e-6)
  steps = np.diff(theta)
  if len(theta) > 1 and np.allclose(steps, steps[0], atol=1e-6) and r.size >= 4 * len(theta):
    return float(steps[0])
  return float(r.resolution)


# ------------------------------------------------------------------ the container

class PoleFigure:
  """measured pole figure intensities, one or several pole figures of one specimen

    PoleFigure(h, r, I)                          h a Miller, one or several superposed directions
    PoleFigure([h1, h23], [r1, r2], [I1, I2])    one pole figure per entry
    PoleFigure(h, r, I, cs, ss, superposition=c, complete=True)
  """

  def __init__(self, h, r, intensities, *frames, superposition=None, complete=False):
    cs = next((f for f in frames if isinstance(f, crystalFrame)), None)
    ss = next((f for f in frames if isinstance(f, referenceFrame) and not isinstance(f, crystalFrame)), None)
    several = isinstance(intensities, (list, tuple))
    allI = list(intensities) if several else [intensities]
    if isinstance(h, (list, tuple)):
      allH = list(h)
    elif several:
      hv = asVector3d(h)
      allH = [hv[i] for i in range(hv.size)] if hv.size > 1 else [hv] * len(allI)
    else:
      allH = [h]
    allR = list(r) if isinstance(r, (list, tuple)) else [r] * len(allH)
    if not len(allH) == len(allR) == len(allI):
      raise ValueError(f"{len(allH)} crystal directions, {len(allR)} specimen direction sets and {len(allI)} intensity sets")
    self.allH = [_millerList(x, cs) for x in allH]
    self.allR = [_specimenDirections(x, complete, ss) for x in allR]
    self.allI = []
    for rr, I in zip(self.allR, allI):
      I = np.asarray(I, dtype=float)
      if I.size != rr.size:
        raise ValueError(f"{I.size} intensities on {rr.size} specimen directions")
      self.allI.append(I.reshape(rr.shape))
    # one entry per pole figure; a single pole figure takes its coefficients directly
    if superposition is None or not isinstance(superposition, (list, tuple, np.ndarray)):
      superposition = [superposition] * len(self.allH)
    elif len(self.allH) == 1 and len(superposition) != 1:
      superposition = [superposition]
    if len(superposition) != len(self.allH):
      raise ValueError("one superposition entry per pole figure")
    self.c = [_coefficients(c, hh.size) for c, hh in zip(superposition, self.allH)]
    if any(hh.frame is not self.allH[0].frame for hh in self.allH):
      raise ValueError("the pole figures live in different crystal frames")
    if any(rr.frame is not self.allR[0].frame for rr in self.allR):
      raise ValueError("the pole figures live in different specimen frames")

  # the same pole figures with other intensities, or other specimen directions
  def _rebuild(self, allI, allR=None):
    return PoleFigure(self.allH, self.allR if allR is None else allR, [np.asarray(I, dtype=float) for I in allI],
                      superposition=self.c, complete=True)

  # ------------------------------------------------------------------ accessors

  @property
  def numPF(self):
    """the number of pole figures"""
    return len(self.allH)

  def __len__(self):
    return self.numPF

  def __iter__(self):
    for i in range(self.numPF):
      yield self.select(i)

  @property
  def CS(self):
    """the crystal frame, that of the crystal directions allH"""
    return self.allH[0].frame

  @property
  def SS(self):
    """the specimen frame, that of the specimen directions allR"""
    return self.allR[0].frame

  @property
  def frame(self):
    """the frame of the specimen directions, pf.SS"""
    return self.SS

  @property
  def how2plot(self):
    """the plotting convention of the specimen directions, the session's for frame-free ones"""
    return self.allR[0].how2plot

  @property
  def antipodal(self):
    """whether the specimen directions are axes, r not told from -r"""
    return self.allR[0].antipodal

  @property
  def h(self):
    """the crystal directions of all pole figures in one vector"""
    return vector3d.cat(*[x.flatten() for x in self.allH])

  @property
  def r(self):
    """the specimen directions of all pole figures in one vector"""
    return vector3d.cat(*[x.flatten() for x in self.allR])

  @property
  def intensities(self):
    """the intensities of all pole figures in one array, in the order of pf.r"""
    return np.concatenate([I.ravel() for I in self.allI])

  @intensities.setter
  def intensities(self, value):
    """pf.intensities = I sets the intensities: one number for all, or one per specimen direction"""
    value = np.asarray(value, dtype=float)
    if value.size == 1:
      self.allI = [np.full(I.shape, float(value)) for I in self.allI]
      return
    if value.size != sum(I.size for I in self.allI):
      raise ValueError("one intensity per specimen direction")
    self.allI = [v.reshape(I.shape) for v, I in zip(self._perFigure(value.ravel()), self.allI)]

  # an array over all measurements cut into the parts of the pole figures
  def _perFigure(self, a):
    return np.split(a, np.cumsum([I.size for I in self.allI])[:-1])

  @property
  def isempty(self):
    """whether the pole figures hold no measurement"""
    return sum(I.size for I in self.allI) == 0

  @property
  def resolution(self):
    """the resolution of the measurement, the finest over the pole figures"""
    return min(_resolutionOf(r) for r in self.allR)

  def select(self, i):
    """the pole figures of the index or indices i, a data set of their own"""
    idx = [i] if isinstance(i, (int, np.integer)) else list(i)
    return PoleFigure([self.allH[k] for k in idx], [self.allR[k] for k in idx], [self.allI[k] for k in idx],
                      superposition=[self.c[k] for k in idx], complete=True)

  # an integer, slice or list picks pole figures; a boolean mask picks measurements
  def __getitem__(self, key):
    if _isBoolMask(key):
      mask = np.asarray(key, dtype=bool).ravel()
      if mask.size != sum(I.size for I in self.allI):
        raise ValueError("the mask has one entry per specimen direction")
      masks = self._perFigure(mask)
      return self._rebuild([I.ravel()[m] for I, m in zip(self.allI, masks)], [r.flatten()[m] for r, m in zip(self.allR, masks)])
    if isinstance(key, slice):
      return self.select(range(*key.indices(self.numPF)))
    return self.select(key)

  # pf[mask] = value: the intensities of the measurements the mask picks, MATLAB's
  # pf(pf.intensities < 0) = 0
  def __setitem__(self, key, value):
    if not _isBoolMask(key):
      raise TypeError("pf[mask] = value takes a mask with one entry per specimen direction")
    I = self.intensities
    I[np.asarray(key, dtype=bool).ravel()] = value
    self.intensities = I

  def __repr__(self):
    lines = [f"PoleFigure ({framesLabel(self.CS, self.SS)})"]
    for h, r in zip(self.allH, self.allR):
      shape = " × ".join(str(s) for s in r.shape) if r.ndim > 1 else str(r.size)
      lines.append(f"  h = {h.char()}, r = {shape} points")
    return aligned("\n".join(lines))

  # ------------------------------------------------------------------ arithmetic
  # a number applies to every pole figure, a sequence one entry per pole figure, another
  # pole figure intensity by intensity

  def _operands(self, other):
    if isinstance(other, PoleFigure):
      if other.numPF != self.numPF or any(a.shape != b.shape for a, b in zip(self.allI, other.allI)):
        raise ValueError("the pole figures are measured on different specimen directions")
      return other.allI
    a = np.asarray(other, dtype=float)
    if a.size == 1:
      return [float(a.ravel()[0])] * self.numPF
    if a.size == self.numPF:
      return list(a.ravel())
    raise ValueError("a factor is one number, or one per pole figure")

  def _apply(self, other, op):
    return self._rebuild([op(I, x) for I, x in zip(self.allI, self._operands(other))])

  def __add__(self, other):
    return self._apply(other, lambda a, b: a + b)

  __radd__ = __add__

  def __sub__(self, other):
    return self._apply(other, lambda a, b: a - b)

  def __rsub__(self, other):
    return self._apply(other, lambda a, b: b - a)

  def __mul__(self, other):
    return self._apply(other, lambda a, b: a * b)

  __rmul__ = __mul__

  def __truediv__(self, other):
    return self._apply(other, lambda a, b: a / b)

  def __rtruediv__(self, other):
    return self._apply(other, lambda a, b: b / a)

  def __neg__(self):
    return self._rebuild([-I for I in self.allI])

  def scale(self, alpha):
    """the intensities multiplied by alpha, one number or one per pole figure"""
    return self * alpha

  def max(self, other=None):
    """the largest intensity of each pole figure, or the pointwise maximum with a number or
    another pole figure; MATLAB's max

      m = max(pf)                   # the largest intensity of every pole figure
      pf = max(pf, 0)               # the intensities clipped from below at zero
      pf = max(pf1, pf2)            # the larger intensity of two pole figures on the same directions

    Input
      pf - PoleFigure
    Output
      m - ndarray, one number per pole figure
      pf - PoleFigure, with another argument
    Keywords
      other - a number, one per pole figure, or a PoleFigure on the same directions
    See also
      PoleFigure.min, PoleFigure.mean"""
    if other is None:
      return np.array([np.max(I) for I in self.allI])
    return self._apply(other, np.maximum)

  def min(self, other=None):
    """the smallest intensity of each pole figure, or the pointwise minimum with a number or
    another pole figure; MATLAB's min

      m = min(pf)                   # the smallest intensity of every pole figure
      pf = min(pf, 100)             # the intensities clipped from above at 100
      pf = min(pf1, pf2)            # the smaller intensity of two pole figures on the same directions

    Input
      pf - PoleFigure
    Output
      m - ndarray, one number per pole figure
      pf - PoleFigure, with another argument
    Keywords
      other - a number, one per pole figure, or a PoleFigure on the same directions
    See also
      PoleFigure.max, PoleFigure.mean"""
    if other is None:
      return np.array([np.min(I) for I in self.allI])
    return self._apply(other, np.minimum)

  def mean(self, weights=None):
    """the mean intensity of each pole figure, the integral over the sphere; MATLAB's mean

      m = mean(pf)                  # by the quadrature weights of the directions
      m = pf.mean(weights=w)        # by weights of one's own, one array per pole figure

    The intensities are summed with the weights calcQuadratureWeights gives the directions of
    each pole figure, which sum to one: the exact weights of a regular grid, the Voronoi areas
    of scattered directions.

    Input
      pf - PoleFigure
    Output
      m - ndarray, one number per pole figure
    Keywords
      weights - a list of weights, one array per pole figure of the shape of its intensities,
          instead of the quadrature weights
    See also
      calcQuadratureWeights, PoleFigure.normalize"""
    if weights is None:
      weights = [calcQuadratureWeights(r) for r in self.allR]
    return np.array([float(np.sum(I.ravel() * np.asarray(w).ravel())) for I, w in zip(self.allI, weights)])

  def normalize(self, odf=None):
    """the pole figures scaled to mean one, or to the pole figures an ODF predicts; MATLAB's
    normalize

      pf = normalize(pf)            # every pole figure of mean intensity one
      pf = normalize(pf, odf)       # scaled to the mass of the pole figures of the ODF

    Input
      pf - PoleFigure
    Output
      pf - PoleFigure
    Keywords
      odf - an SO3Fun: each pole figure divided by the factor calcNormalization fits against
          the pole figure the ODF predicts, instead of by its mean
    See also
      PoleFigure.mean, calcNormalization, PoleFigure.calcError"""
    return self / (self.mean() if odf is None else self.calcNormalization(odf))

  def rotate(self, rot):
    """the specimen directions turned; a rotation that is no element of the specimen group
    strips the group from the frame"""
    if not isinstance(rot, MtexArray):
      rot = rotation.byAxisAngle(vector3d.Z, rot)
    ss = dropGroup(rot, self.SS)
    allR = []
    for r in self.allR:
      v = vector3d(rotation(rot.data) * vector3d(r.data), antipodal=r.antipodal)
      v.frame = ss
      allR.append(v)
    return self._rebuild(self.allI, allR)

  # ------------------------------------------------------------------ the point sets

  def unique(self, tol=1e-6):
    """the duplicated directions removed: equivalent specimen directions under the specimen group
    carry the same pole density, so they are merged and their intensities averaged; MATLAB's
    unique

      pf = unique(pf)               # the directions within 1e-6 merged
      pf = pf.unique(tol=1e-3)      # a coarser tolerance

    Two directions are merged when one of them, mapped by an element of the specimen group,
    has a dot product with the other above 1 - tol (its absolute value for axes).

    Input
      pf - PoleFigure
    Output
      pf - PoleFigure
    Keywords
      tol - the tolerance on the dot product of two directions
    See also
      PoleFigure.union, PoleFigure.cat"""
    allR, allI = [], []
    for r, I in zip(self.allR, self.allI):
      xyz = r.flatten().normalize().data
      imgs = _images(self.SS, xyz)     # (n, |G|, 3)
      d = np.einsum("ngk,mk->gnm", imgs, xyz)
      same = (np.abs(d) if r.antipodal else d) > 1 - tol
      same = np.any(same, axis=0)
      pos = np.full(len(xyz), -1)
      keep = []
      for k in range(len(xyz)):
        if pos[k] < 0:
          later = np.flatnonzero(same[k])
          pos[later[pos[later] < 0]] = len(keep)
          keep.append(k)
      sums = np.bincount(pos, weights=I.ravel(), minlength=len(keep))
      counts = np.bincount(pos, minlength=len(keep))
      allR.append(r.flatten()[keep])
      allI.append(sums / counts)
    return self._rebuild(allI, allR)

  def union(self, *others):
    """the pole figures of several data sets in one; those of equal crystal directions joined"""
    allH, allR, allI, c = list(self.allH), list(self.allR), list(self.allI), list(self.c)
    for pf in others:
      if pf is None:
        continue
      allH += pf.allH
      allR += pf.allR
      allI += pf.allI
      c += pf.c
    i = 1
    while i < len(allH):
      match = [j for j in range(i) if allH[j].size == allH[i].size and np.max(allH[j].angle(allH[i], symmetry=False)) < 1e-4]
      if match:
        j = match[0]
        allR[j] = vector3d.cat(allR[j].flatten(), allR[i].flatten())
        allI[j] = np.concatenate([allI[j].ravel(), allI[i].ravel()])
        del allH[i], allR[i], allI[i], c[i]
      else:
        i += 1
    return PoleFigure(allH, allR, allI, superposition=c, complete=True)

  def isOutlier(self, alpha=2):
    """the measurements whose intensity deviates from the mean of their neighbours by more than
    alpha standard deviations; MATLAB's isOutlier and findOutlier

      ind = isOutlier(pf)           # deviating by more than two standard deviations
      ind = pf.findOutlier(alpha=3) # by more than three
      pf = pf[~isOutlier(pf)]       # the outliers removed

    The neighbours of a direction are the other directions of its pole figure within three
    times the resolution of the grid; the standard deviation is that of all intensities of the
    pole figure.

    Input
      pf - PoleFigure
    Output
      ind - ndarray of booleans, one per measurement, the pole figures one after another
    Keywords
      alpha - the number of standard deviations, MATLAB's 'alpha'
    See also
      PoleFigure.unique"""
    out = []
    for r, I in zip(self.allR, self.allI):
      xyz = r.flatten().normalize().data
      I = I.ravel()
      c = math.cos(3 * _resolutionOf(r))
      d = xyz @ xyz.T
      near = (np.abs(d) if r.antipodal else d) >= c
      np.fill_diagonal(near, False)
      n = near.sum(axis=1)
      m = np.where(n > 0, (near @ I) / np.maximum(n, 1), I)
      out.append(np.abs(m - I) > alpha * np.std(I))
    return np.concatenate(out)

  findOutlier = isOutlier

  def noisepf(self, factor=1, background=0, nonnegative=False, rng=None):
    """simulated diffraction counts: Poisson counts of factor * I + background, the background
    subtracted again

      pfn = pf.noisepf(100)                          # counts of radiation 100
      pfn = pf.noisepf(100, 10)                      # with a background of 10
      pfn = pf.noisepf(100, 10, nonnegative=True)    # the negative counts set to zero

    Every intensity I is replaced by a random count of the Poisson distribution of the mean
    factor * I + background, from which the background is subtracted again: the counts a
    diffractometer measures for a pole figure of intensity I, the factor the radiation, the
    background the counts every direction receives. Where the background exceeds a count the
    result is negative, unless nonnegative=True sets it to zero.

    Input
      pf - PoleFigure
      factor - the radiation, the counts of intensity one; one per pole figure or one for all
      background - the background radiation
    Output
      pfn - PoleFigure, the simulated counts
    Keywords
      nonnegative - set the negative counts to zero (MATLAB's flag NONNEGATIV)
      rng - a numpy.random.Generator, for reproducible counts
    See also
      calcPoleFigure, PoleFigureSimulation"""
    rng = np.random.default_rng() if rng is None else rng
    factor = self._operands(factor)
    allI = []
    for I, f in zip(self.allI, factor):
      counts = rng.poisson(np.maximum(f * I + background, 0)) - background
      allI.append(np.maximum(counts, 0) if nonnegative else counts.astype(float))
    return self._rebuild(allI)

  # ------------------------------------------------------------------ corrections

  def correct(self, background=None, defocusing=None, defocusingBackground=None):
    """the pole figures corrected for background and defocusing

      pf = pf.correct(background=pfBg)                      # the background subtracted
      pf = pf.correct(defocusing=pfDef)                     # the defocusing divided out
      pf = pf.correct(background=pfBg, defocusing=pfDef, defocusingBackground=pfDefBg)  # both, the defocusing without its background

    The background is subtracted from the intensities, and they are divided by the
    defocusing, from which its own background is subtracted first. Each correction is a
    PoleFigure of its own or a number. One measured pole figure serves all, and a correction
    measured on other specimen directions is interpolated onto those of pf: spherically
    where it covers the sphere, by a cubic spline in the polar angle where it is measured
    along a curve, as a defocusing usually is.

    Input
      pf - PoleFigure
    Output
      pf - PoleFigure, the corrected intensities
    Keywords
      background - PoleFigure or number, the background (MATLAB's option bg)
      defocusing - PoleFigure, the defocusing (MATLAB's option def)
      defocusingBackground - PoleFigure or number, the background of the defocusing
          measurement (MATLAB's option defbg)
    See also
      PoleFigure.noisepf, PoleFigureCorrection"""
    out = self if background is None else self - self._adapt(background)
    if defocusing is None:
      return out
    d = self._adapt(defocusing)
    if defocusingBackground is not None:
      d = d - self._adapt(defocusingBackground)
    return out / d

  # a correction on the specimen directions of this data set: one pole figure serves
  # all, values on other directions are interpolated, spherically when the correction
  # covers the sphere and by a natural cubic spline in the polar angle when it is a curve
  def _adapt(self, corr):
    if not isinstance(corr, PoleFigure):
      return self._rebuild([np.full(I.shape, float(corr)) for I in self.allI])
    if corr.numPF == 1 and self.numPF > 1:
      corr = PoleFigure(self.allH, [corr.allR[0]] * self.numPF, [corr.allI[0]] * self.numPF, superposition=self.c, complete=True)
    if corr.numPF != self.numPF:
      raise ValueError(f"{corr.numPF} correction pole figures for {self.numPF} pole figures")
    allI = []
    for r, cr, cI in zip(self.allR, corr.allR, corr.allI):
      if cr.shape == r.shape and np.allclose(cr.data, r.data):
        allI.append(cI)
        continue
      th, rh, _ = cr.flatten().polar()
      if len(np.unique(np.round(rh, 1))) > 17:
        from .density import interpolate
        allI.append(np.real(interpolate(cr.flatten(), cI.ravel())(r.flatten())).reshape(r.shape))
      else:
        from scipy.interpolate import CubicSpline
        order = np.argsort(th)
        x, y = th[order], cI.ravel()[order]
        x, first = np.unique(np.round(x, 12), return_index=True)
        spline = CubicSpline(x, y[first], bc_type="natural") if len(x) > 2 else (lambda t: np.interp(t, x, y[first]))
        allI.append(np.asarray(spline(r.flatten().theta)).reshape(r.shape))
    return self._rebuild(allI)

  # ------------------------------------------------------------------ simulation and errors

  def calcPoleFigure(self, odf):
    """the pole figures an ODF predicts on the measured directions"""
    return calcPoleFigure(odf, self.allH, self.allR, superposition=self.c, complete=True)

  def calcNormalization(self, other):
    """the factor the second pole figure is scaled by to carry the mass of the first"""
    if not isinstance(other, PoleFigure):
      other = self.calcPoleFigure(other)
    out = []
    for r, I1, I2 in zip(self.allR, self.allI, other.allI):
      w = calcQuadratureWeights(r).ravel()
      out.append(float(w @ np.maximum(I1.ravel(), 0) / (w @ np.maximum(I2.ravel(), 0))))
    return np.array(out)

  def calcErrorPF(self, other, mode="relative", epsilon=None):
    """the pole figures of the deviation from a recalculation: the difference between measured
    intensities and those an ODF predicts, or another pole figure on the same directions;
    MATLAB's calcErrorPF

      pfe = calcErrorPF(pf, odf)               # the damped relative error
      pfe = calcErrorPF(pf, pf2)               # against other pole figures
      pfe = pf.calcErrorPF(odf, 'RP', epsilon=0.5)  # |pf - pfcalc| / pfcalc where pfcalc > 0.5
      pfe = pf.calcErrorPF(odf, 'l1')          # |pf - pfcalc|
      pfe = pf.calcErrorPF(odf, 'l2')          # |pf - pfcalc|^2

    The recalculation is first scaled to the mass of the measurement by calcNormalization.
    'relative', the default, is |I - Ic| / max(I + epsilon, Ic + epsilon), which MATLAB computes
    when no measure is named, though its help names RP the default; 'RP' keeps only the
    directions where the recalculation exceeds epsilon, so its pole figures have fewer
    directions; 'l1' and 'l2' are the plain differences. The names are taken in any case.

    Input
      pf - PoleFigure, the measurement
      other - SO3Fun, the ODF of the recalculation, or PoleFigure on the same directions
    Output
      pfe - PoleFigure of the deviations
    Keywords
      mode - 'relative', 'RP', 'l1' or 'l2'
      epsilon - the damping of 'relative' (0.5) or the threshold of 'RP' (1), both times the
          normalisation
    See also
      PoleFigure.calcError, calcNormalization, calcPoleFigure"""
    mode = _errorMode(mode)
    # the RP measure keeps the directions above one, the regularised one damps by a half
    epsilon = (1.0 if mode == "RP" else 0.5) if epsilon is None else float(epsilon)
    if not isinstance(other, PoleFigure):
      other = self.calcPoleFigure(other)
    alpha = self.calcNormalization(other)
    allR, allI = [], []
    for r, I1, I2, a in zip(self.allR, self.allI, other.allI, alpha):
      d1, d2 = I1.ravel(), a * I2.ravel()
      if mode == "l1":
        d = np.abs(d1 - d2)
      elif mode == "l2":
        d = (d1 - d2) ** 2
      elif mode == "RP":
        keep = d2 > epsilon * a
        d = np.abs(d1[keep] - d2[keep]) / d2[keep]
        allR.append(r.flatten()[keep])
        allI.append(d)
        continue
      else:
        d = np.abs(d1 - d2) / np.maximum(d1 + epsilon * a, d2 + epsilon * a)
      allR.append(r)
      allI.append(d.reshape(r.shape))
    return self._rebuild(allI, allR)

  def calcError(self, other, mode="relative", epsilon=None):
    """the mean deviation, one number per pole figure, from a recalculation by an ODF or from
    other pole figures on the same directions; MATLAB's calcError

      e = calcError(pf, odf)                   # the damped relative error
      e = calcError(pf, pf2)                   # against other pole figures on the same directions
      e = calcError(pf, odf, 'RP', epsilon=0.5)  # the RP error above 0.5
      e = calcError(odf, pf, 'RP')             # the ODF first, as MATLAB's
      e = calcError(pf, odf, 'l2')             # the relative mean square error

    The deviations are those of calcErrorPF, averaged over each pole figure: 'relative', the
    default, the relative error damped by epsilon (0.5), which MATLAB computes when no measure is
    named (its documentation calls the default RP); 'RP' the mean of |I - Ic| / Ic over the
    directions where Ic exceeds epsilon (1), MATLAB's 'RP', epsilon; 'l1' and 'l2' the mean
    absolute and squared differences divided by the mean absolute and squared intensities. The
    names are taken in any case.

    Input
      pf - PoleFigure, the measurement
      other - SO3Fun, the ODF of the recalculation, or PoleFigure on the same directions
    Output
      e - ndarray, one number per pole figure; NaN where 'RP' keeps no direction
    Keywords
      mode - 'relative', 'RP', 'l1' or 'l2'
      epsilon - the damping of 'relative' or the threshold of 'RP', both times the normalisation
    See also
      PoleFigure.calcErrorPF, calcNormalization, SO3Fun.calcError"""
    mode = _errorMode(mode)
    e = self.calcErrorPF(other, mode, epsilon)
    out = np.array([float(np.mean(I)) if I.size else np.nan for I in e.allI])
    if mode == "l1":
      out = out / np.array([np.mean(np.abs(I)) for I in self.allI])
    elif mode == "l2":
      out = out / np.array([np.mean(I ** 2) for I in self.allI])
    return out

  # a single pole figure as a function on the sphere, by quadrature over its directions;
  # the bandwidth half of what the point density resolves keeps the counting noise from
  # ringing
  def _harmonicS2(self, bandwidth=None):
    if self.numPF != 1:
      raise ValueError("one pole figure at a time becomes a function on the sphere")
    r, I = self.allR[0].flatten(), self.allI[0].ravel()
    n = max(r.size, 1)
    L = max(1, round(np.pi / (2 * math.sqrt((2 * np.pi if r.antipodal else 4 * np.pi) / n)))) if bandwidth is None else int(bandwidth)
    w = 4 * np.pi * calcQuadratureWeights(r).ravel()
    return S2FunHarmonic.quadrature(r, I, bandwidth=L, weights=w, real=True, antipodal=r.antipodal)

  def calcODF(self, *args, **kwargs):
    """calcODF(pf, ...): an ODF reconstructed from the pole figure intensities, an SO3FunRBF"""
    from ..analysis.pf2odf import calcODF
    return calcODF(self, *args, **kwargs)

  def calcODFIterative(self, *args, **kwargs):
    """calcODFIterative(pf, ...): the reconstruction refined successively over a ladder of halfwidths"""
    from ..analysis.pf2odf import calcODFIterative
    return calcODFIterative(self, *args, **kwargs)

  # ------------------------------------------------------------------ joining data sets

  @staticmethod
  def cat(*pfs, axis=0):
    """the pole figures of several data sets in one, in order; MATLAB's cat

      pf = cat(pf1, pf2, pf3)       # every pole figure of the three, in order
      pf = PoleFigure.cat(pf1, pf2) # the same as a static method

    Unlike union, pole figures of the same crystal direction stay separate; None is skipped.

    Input
      pf1, pf2, pf3 - PoleFigure
    Output
      pf - PoleFigure
    Keywords
      axis - taken for numpy's signature and ignored: the pole figures form one list, MATLAB's
          dim
    See also
      PoleFigure.union"""
    pfs = [pf for pf in pfs if pf is not None]
    return PoleFigure([h for pf in pfs for h in pf.allH], [r for pf in pfs for r in pf.allR], [I for pf in pfs for I in pf.allI],
                      superposition=[c for pf in pfs for c in pf.c], complete=True)

  # declared for the editor: the methods other modules attach (tools/staticapi.py)
  if TYPE_CHECKING:
    from ..io.loadpolefigure import loadPoleFigure
    from ..io.exportpolefigure import exportPoleFigure as export
    from ..plotting.polefigureplot import _plotPoleFigure as plot, plotDiff
    load = staticmethod(loadPoleFigure)


# ------------------------------------------------------------------ the verbs as functions

def union(pf, *others):
  """union(pf, ...) is pf.union(...): the pole figures of several data sets in one, those of equal
  crystal directions joined"""
  return pf.union(*others)


def correct(pf, **kwargs):
  """correct(pf, ...) is pf.correct(...): the background subtracted and the defocusing divided out"""
  return pf.correct(**kwargs)


def noisepf(pf, factor=1, background=0, nonnegative=False, rng=None):
  """simulated diffraction counts: Poisson counts of factor * I + background, the background
  subtracted again; the same as pf.noisepf

    pfn = noisepf(pf, 100)                         # counts of radiation 100
    pfn = noisepf(pf, 100, 10)                     # with a background of 10
    pfn = noisepf(pf, 100, 10, nonnegative=True)   # the negative counts set to zero

  Every intensity I is replaced by a random count of the Poisson distribution of the mean
  factor * I + background, from which the background is subtracted again: the counts a
  diffractometer measures for a pole figure of intensity I, the factor the radiation, the
  background the counts every direction receives. Where the background exceeds a count the
  result is negative, unless nonnegative=True sets it to zero.

  Input
    pf - PoleFigure
    factor - the radiation, the counts of intensity one; one per pole figure or one for all
    background - the background radiation
  Output
    pfn - PoleFigure, the simulated counts
  Keywords
    nonnegative - set the negative counts to zero (MATLAB's flag NONNEGATIV)
    rng - a numpy.random.Generator, for reproducible counts
  See also
    calcPoleFigure, PoleFigureSimulation"""
  return pf.noisepf(factor, background, nonnegative, rng)


def isOutlier(pf, alpha=2):
  """the measurements deviating from the mean of their neighbours by more than alpha standard
  deviations; the same as pf.isOutlier, MATLAB's isOutlier

    ind = isOutlier(pf)             # deviating by more than two standard deviations
    ind = isOutlier(pf, alpha=3)    # by more than three

  The neighbours are the directions within three times the resolution of the grid.

  Input
    pf - PoleFigure
  Output
    ind - ndarray of booleans, one per measurement
  Keywords
    alpha - the number of standard deviations
  See also
    PoleFigure.isOutlier, PoleFigure.unique"""
  return pf.isOutlier(alpha)


def calcNormalization(pf, other):
  """calcNormalization(pf, other) is pf.calcNormalization(other): the factor that scales the second
  pole figure to the mass of the first"""
  return pf.calcNormalization(other)


def calcErrorPF(pf, other, mode="relative", epsilon=None):
  """the pole figures of the deviation from a recalculation; the same as pf.calcErrorPF,
  MATLAB's calcErrorPF

    pfe = calcErrorPF(pf, odf)               # the damped relative error
    pfe = calcErrorPF(pf, odf, 'RP')         # |pf - pfcalc| / pfcalc where pfcalc > 1
    pfe = calcErrorPF(pf, pf2, 'l1')         # |pf - pf2| against other pole figures

  Input
    pf - PoleFigure, the measurement
    other - SO3Fun or PoleFigure, the recalculation
  Output
    pfe - PoleFigure of the deviations
  Keywords
    mode - 'relative', 'RP', 'l1' or 'l2'
    epsilon - the damping of 'relative' (0.5) or the threshold of 'RP' (1)
  See also
    PoleFigure.calcErrorPF, PoleFigure.calcError"""
  return pf.calcErrorPF(other, mode, epsilon)


def calcError(a, b, *args, **kwargs):
  """the error between a measurement and a recalculation, or between two SO(3) functions

    e = calcError(pf, odf)  # one number per pole figure, PoleFigure.calcError
    e = calcError(odf, pf, 'RP')  # the same, in MATLAB's order
    e = calcError(odf1, odf2)  # 'l1'
    e = calcError(odf1, odf2, 'l2', resolution=2.5*degree)"""
  if isinstance(a, PoleFigure):
    return a.calcError(b, *args, **kwargs)
  return _calcErrorODF(a, b, *args, **kwargs)


# ------------------------------------------------------------------ pole figures from an ODF

# the pole density of one crystal direction on the specimen directions, the even part
# where the directions are axes
def _poleDensity(odf, h, r):
  P = odf.radon(h)
  if r.antipodal and not P.antipodal:
    P = P.even()
  return np.real(P(vector3d(r.flatten().data))).reshape(r.shape)


def calcPoleFigure(odf, h, r=None, superposition=None, complete=False, **gridOptions):
  """the pole figures an ODF predicts: simulated diffraction intensities, superposed pole
  figures and any specimen directions

    pf = calcPoleFigure(odf, h, r)                                # one pole figure per element of h
    pf = calcPoleFigure(odf, h, resolution=5*degree)              # on a regular grid of the upper hemisphere
    pf = calcPoleFigure(odf, h, resolution=5*degree, complete=True)  # on a regular grid of the whole sphere
    pf = calcPoleFigure(odf, [h1, h2], [r1, r2])                  # one set of specimen directions per pole figure
    pf = calcPoleFigure(odf, [h1, h23], r, superposition=[1, [0.2, 0.8]])  # h23 two superposed directions
    pf = calcPoleFigure(odf, pfMeasured)                          # on the directions of a measurement

  The intensity of the crystal direction h at the specimen direction r is the mean of the ODF
  over all orientations that take h to r, its radon transform. Where the specimen directions
  are axes, as on the default grid of the upper hemisphere, it is the mean of r and -r.

  A Python list of crystal directions makes one pole figure per entry. An entry of several
  directions is their superposition, weighted by the superposition coefficients, as the
  overlapping reflections a diffractometer measures together.

  Without specimen directions the pole figures are evaluated on regularS2Grid, which takes
  the remaining keywords. noisepf turns the intensities into counts.

  Input
    odf - SO3Fun
    h - Miller or vector3d in the crystal frame of the ODF, or a list of them, one per pole
        figure
  Output
    pf - PoleFigure
  Keywords
    r - vector3d, the specimen directions, or a list of them, one per pole figure; by
        default a regular grid
    superposition - the weights of the superposed directions, a list with one entry per
        pole figure
    complete - the whole sphere instead of the upper hemisphere, the directions without
        antipodal symmetry (MATLAB's flag complete)
  See also
    PoleFigure.noisepf, calcODF, regularS2Grid, PoleFigureSimulation"""
  if isinstance(odf, PoleFigure):
    odf, h = h, odf
  if isinstance(h, PoleFigure):
    return h.calcPoleFigure(odf)
  if isinstance(h, vector3d):
    hs = [h.flatten()[i] for i in range(h.size)] if h.size > 1 else [h.flatten()]
  else:
    hs = [asVector3d(x) for x in h]
  if r is None:
    from ..geometry.grids.s2grid import regularS2Grid
    r = regularS2Grid(antipodal=not complete, **gridOptions)
  rs = list(r) if isinstance(r, (list, tuple)) else [r] * len(hs)
  cs = list(superposition) if isinstance(superposition, (list, tuple)) and len(hs) > 1 else [superposition] * len(hs)
  allH, allR, allI, allC = [], [], [], []
  for hh, rr, c in zip(hs, rs, cs):
    hh = _millerList(hh, odf.CS if hh.frame is None else None)
    if hh.frame is not odf.CS:
      raise ValueError("the pole figure directions have to be given in the crystal frame of the ODF")
    # the specimen directions take the ODF's specimen frame, its symmetry included, as MATLAB's
    rf = getattr(rr, "frame", None)
    own = rf is None or odf.SS is not None and rf.stripSym() is odf.SS.stripSym()
    rr = _specimenDirections(rr, complete, odf.SS if own else None)
    c = _coefficients(c, hh.size)
    allH.append(hh)
    allR.append(rr)
    allC.append(c)
    allI.append(sum(w * _poleDensity(odf, hh[k], rr) for k, w in enumerate(c)))
  return PoleFigure(allH, allR, allI, superposition=allC, complete=True)


# ------------------------------------------------------------------ quadrature weights

# the polar ladder and the azimuth ladder of a product grid, or None
def _productGrid(v, tol=1e-6):
  th, rh, _ = v.flatten().polar()
  rh = np.mod(rh, 2 * np.pi)
  rh[rh > 2 * np.pi - tol] = 0
  th = np.round(th / tol) * tol
  rh = np.round(rh / tol) * tol
  pole = (th < tol) | (th > np.pi - tol)
  uth = np.unique(th)
  urh = np.unique(rh[~pole]) if np.any(~pole) else np.zeros(0)
  nth = np.count_nonzero((uth > tol) & (uth < np.pi - tol))
  pairs = np.unique(np.stack([th[~pole], rh[~pole]], axis=1), axis=0)
  if len(pairs) != np.count_nonzero(~pole) or len(pairs) != nth * len(urh):
    return None
  return uth, urh


# Clenshaw-Curtis weights on the nodes cos(k pi / N), k = 0..N
def calcQuadratureWeights(v, method="voronoi"):
  """weights that integrate a function from samples at the directions, summing to one; MATLAB's
  calcQuadratureWeights

    w = calcQuadratureWeights(r)                     # a regular grid exactly, scattered points by Voronoi areas
    w = calcQuadratureWeights(r, method='harmonic')  # the least norm correction that integrates every harmonic the count allows

  On a regular grid of rings, a polar ladder from the pole in equal steps, the weights are
  the Clenshaw-Curtis weights of the rings, shared by their points. They are exact on the
  sphere and, with the equator ring halved, on a hemisphere.

  For scattered directions the method chooses:

  - 'voronoi', the default and MATLAB's only method: the areas of the spherical Voronoi cells,
  an axis list counted on both hemispheres, duplicates sharing a cell. No cell counts more
  than twice the 80 % quantile of the areas, so isolated points do not dominate.
  - 'harmonic': equal weights corrected by the least change that integrates every spherical
  harmonic the number of points allows.

  Where neither works, as for fewer than four distinct directions, the weights are equal.

  Input
    r - vector3d
  Output
    w - ndarray of the shape of r, summing to one
  Keywords
    method - 'voronoi' or 'harmonic', for scattered directions
  See also
    PoleFigure.mean, S2FunHarmonic.quadrature"""
  v = asVector3d(v)
  n = v.size
  if n == 0:
    return np.zeros(v.shape)
  grid = _productGrid(v)
  if grid is not None and grid[0][0] < 1e-6 and len(grid[0]) > 1 and np.allclose(np.diff(grid[0]), grid[0][1], atol=1e-6):
    # the Clenshaw-Curtis weights of the polar ladder, exact on the sphere and, with the
    # equator ring halved, for the even functions of a hemisphere; a cap keeps the
    # truncated weights as MATLAB does
    uth, urh = grid
    N = int(round(np.pi / uth[1]))
    w = clenshawCurtis(N + 1)[1][:len(uth)]
    if len(uth) == N // 2 + 1 and N % 2 == 0 and abs(uth[-1] - np.pi / 2) < 1e-6:
      w[-1] /= 2
    th = v.flatten().theta
    ring = np.rint(th / uth[1]).astype(int)
    # every point of a ring shares the ring's weight, the pole counted once
    counts = np.bincount(ring, minlength=len(uth))
    w = w[ring] / counts[ring]
    return (w / np.sum(w)).reshape(v.shape)
  w = _harmonicWeights(v) if method == "harmonic" else _voronoiWeights(v)
  if w is None or not np.all(np.isfinite(w)):
    w = np.full(n, 1 / n)
  return w.reshape(v.shape)


# the areas of the spherical Voronoi cells, an axis list doubled and halved, duplicates
# sharing a cell, no cell above twice the 80 % quantile
def _voronoiWeights(v):
  from ..geometry.hull import _cellAreas
  xyz = v.flatten().normalize().data
  if v.antipodal:
    xyz = np.concatenate([xyz, -xyz])
  key = np.round(xyz / 1e-6) * 1e-6
  u, inv = np.unique(key, axis=0, return_inverse=True)
  inv = inv.ravel()
  if len(u) < 4:
    return None
  try:
    area = _cellAreas(u)
  except Exception:
    return None
  counts = np.bincount(inv, minlength=len(u))
  w = area[inv] / counts[inv]
  if v.antipodal:
    w = w[:v.size] + w[v.size:]
  wmax = 2 * np.quantile(w, 0.8)
  if np.any(w > wmax):
    w = np.where(w > wmax, np.max(w[w <= wmax]), w)
  return w / np.sum(w)


# the weights of least distance to equal weights that integrate every spherical harmonic
# up to the degree the count supports, the even ones for axes: the minimum norm solution
# of the real system the harmonics at the points set up, dense for a few thousand points
# and by LSQR on the transform above; they can turn negative on an irregular set
def _harmonicWeights(v):
  from .harmonics import _sphericalY
  x = v.flatten().normalize()
  n = x.size
  L = max(1, int(math.sqrt(n / (2 if v.antipodal else 1))) - 1)
  degrees = range(0, L + 1, 2) if v.antipodal else range(L + 1)
  w0 = np.full(n, 4 * np.pi / n)
  if n <= 4000:
    Y = np.concatenate([_sphericalY(l, vector3d(x.data)) for l in degrees])
    A = np.concatenate([Y.real, Y.imag])
    b = np.zeros(A.shape[0])
    b[0] = math.sqrt(4 * np.pi)
    d = np.linalg.lstsq(A, b - A @ w0, rcond=None)[0]
  else:
    from scipy.sparse.linalg import LinearOperator, lsqr
    from .transforms import S2FourierPlan
    plan = S2FourierPlan(vector3d(x.data), L)
    deg = np.repeat(np.arange(L + 1), 2 * np.arange(L + 1) + 1)
    rows = np.flatnonzero(deg % 2 == 0) if v.antipodal else np.arange(deg.size)
    full = np.zeros(deg.size, dtype=complex)

    def rmatvec(z):
      full[:] = 0
      full[rows] = z
      return plan(full)

    op = LinearOperator((rows.size, n), matvec=lambda w: plan.adjoint(np.asarray(w, dtype=complex), np.ones(n))[rows],
                        rmatvec=rmatvec, dtype=complex)
    target = np.zeros(rows.size, dtype=complex)
    target[0] = math.sqrt(4 * np.pi)
    # BLAS on one thread between the transforms, docs/research/threads.md
    with blasThreads(1):
      d = np.real(lsqr(op, target - op.matvec(w0), atol=1e-10, btol=1e-10, iter_lim=200)[0])
  w = w0 + d
  return w / np.sum(w)


# ------------------------------------------------------------------ refining a point set

def refine(v):
  """the directions with a new one between every three neighbours: the centroids of the
  Delaunay triangles of the points on the sphere, none below the lowest point, so a cap
  stays a cap; returns the joined list and the new points, MATLAB's [v, r] = refine(v)"""
  from scipy.spatial import ConvexHull
  v = asVector3d(v)
  xyz = v.flatten().normalize().data
  # the south pole closes the hull below a cap, and no triangle is wanted on it
  P = np.concatenate([xyz, [[0.0, 0.0, -1.0]]])
  hull = ConvexHull(P, qhull_options="QJ")
  tri = hull.simplices[~np.any(hull.simplices == len(P) - 1, axis=1)]
  c = P[tri].sum(axis=1)
  c /= np.linalg.norm(c, axis=1, keepdims=True)
  maxTheta = np.max(np.arccos(np.clip(xyz[:, 2], -1, 1)))
  c = c[np.arccos(np.clip(c[:, 2], -1, 1)) <= maxTheta + 1e-9]
  new = vector3d(c, antipodal=v.antipodal, frame=v.frame)
  return vector3d.cat(vector3d(xyz, antipodal=v.antipodal, frame=v.frame), new), new


# ------------------------------------------------------------------ the defocusing curve

class SchulzDefocusingKernel:
  """the defocusing of a Bragg-Brentano goniometer (MATLAB SchulzDefocusingKernel): the
  beam of width r is smeared over the receiving slit of width w at the distance d as the
  sample tilts by alpha, and the measured intensity drops by the factor the kernel gives

    psi = SchulzDefocusingKernel(theta=45*degree, r=1, w=2, d=200, dtheta=0.5*degree)
    psi.eval(cos(alpha))                     the factor at the tilt alpha
    psi.defocusing(pf)                       the factor on the specimen directions of a pole figure
  """

  def __init__(self, mu_t=1.0, theta=45 * degree, r=1.0, w=2.0, d=200.0, dtheta=0.5 * degree):
    self.mu_t, self.theta, self.r, self.w, self.d, self.dtheta = float(mu_t), float(theta), float(r), float(w), float(d), float(dtheta)

  def eval(self, t):
    """the defocusing factor at t = cos(alpha) of the tilt alpha, 1 at no tilt, infinite at 90 degrees"""
    from scipy.special import erf
    alpha = np.arccos(np.clip(np.asarray(t, dtype=float), -1, 1))
    tau = self.dtheta / math.sqrt(2 * np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
      a = (self.r / self.d) * math.cos(self.theta) * np.tan(alpha)
      u1, u2 = (a + self.w / 2) / (tau * math.sqrt(2)), (a - self.w / 2) / (tau * math.sqrt(2))
      t1 = tau * math.sqrt(2) / (2 * a * erf(self.w / (2 * tau * math.sqrt(2))))
      value = 1 / (t1 * (u1 * erf(u1) - u2 * erf(u2) + np.exp(-u1 ** 2) / math.sqrt(np.pi) - np.exp(-u2 ** 2) / math.sqrt(np.pi)))
    value = np.where(alpha <= 0, 1.0, value)
    return np.where(alpha >= np.pi / 2, np.inf, value)

  def __call__(self, t):
    return self.eval(t)

  def defocusing(self, pf):
    """the correction on the specimen directions of a pole figure, one at the pole"""
    return pf._rebuild([self.eval(np.cos(r.theta)) for r in pf.allR])

  def __repr__(self):
    return display("SchulzDefocusingKernel", [("mu_t", f"{self.mu_t:g}"), ("theta", f"{self.theta / degree:g}°"), ("r", f"{self.r:g}"),
                                              ("w", f"{self.w:g}"), ("d", f"{self.d:g}"), ("dtheta", f"{self.dtheta / degree:g}°")])


# ------------------------------------------------------------------ installation

def _install():
  from .bingham import SO3FunBingham
  from .handles import SO3FunComposition, SO3FunGeneric, SO3FunHandle
  from .rbf import SO3FunCBF, SO3FunRBF
  from .so3fun import SO3FunHarmonic
  install((SO3FunHarmonic, SO3FunRBF, SO3FunCBF, SO3FunHandle, SO3FunComposition, SO3FunGeneric, SO3FunBingham), calcPoleFigure)
  install(vector3d, calcQuadratureWeights)
  install(vector3d, refine)


_install()

__all__ = ["PoleFigure", "calcPoleFigure", "calcQuadratureWeights", "calcNormalization", "calcErrorPF", "calcError",
           "correct", "noisepf", "isOutlier", "refine", "SchulzDefocusingKernel"]
