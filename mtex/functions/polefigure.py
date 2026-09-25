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

import math

import numpy as np

from ..geometry._array import MtexArray
from ..geometry.frames import crystalFrame, referenceFrame, specimenFrame
from ..geometry.grids.s2grid import S2Grid
from ..geometry.kernels import blasThreads
from ..geometry.products import dropGroup
from ..geometry.quaternion import quaternion, rotation
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from .characteristics import calcError as _calcErrorODF
from .funops import conv
from .s2fun import S2FunHarmonic

ERROR_MODES = ("relative", "RP", "l1", "l2")


# ------------------------------------------------------------------ the directions

# crystal directions as a flat list in a crystal frame, the frame given taking over
def _millerList(h, cs):
  v = h if isinstance(h, vector3d) else vector3d(h)
  v = v.flatten()
  if cs is not None:
    v = vector3d(v)
    if v.isCrystalDirection:
      v.CS = cs
    else:
      v.frame = cs
  if not v.isCrystalDirection:
    raise ValueError("the pole figure directions have to be given in a crystal frame")
  return v


# specimen directions as axes unless complete, in a specimen frame, keeping their shape
def _specimenDirections(r, complete, ss):
  v = vector3d(r) if isinstance(r, vector3d) else vector3d(np.asarray(r, dtype=float))
  if ss is not None:
    v.frame = ss
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
      hv = h if isinstance(h, vector3d) else vector3d(h)
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
    return len(self.allH)

  def __len__(self):
    return self.numPF

  def __iter__(self):
    for i in range(self.numPF):
      yield self.select(i)

  @property
  def CS(self):
    return self.allH[0].frame

  @property
  def SS(self):
    return self.allR[0].frame

  @property
  def frame(self):
    return self.SS

  @property
  def antipodal(self):
    return self.allR[0].antipodal

  @property
  def h(self):
    return vector3d.cat(*[x.flatten() for x in self.allH])

  @property
  def r(self):
    return vector3d.cat(*[x.flatten() for x in self.allR])

  @property
  def intensities(self):
    return np.concatenate([I.ravel() for I in self.allI])

  @intensities.setter
  def intensities(self, value):
    value = np.asarray(value, dtype=float)
    if value.size == 1:
      self.allI = [np.full(I.shape, float(value)) for I in self.allI]
      return
    if value.size != sum(I.size for I in self.allI):
      raise ValueError("one intensity per specimen direction")
    value, out = value.ravel(), []
    start = 0
    for I in self.allI:
      out.append(value[start:start + I.size].reshape(I.shape))
      start += I.size
    self.allI = out

  @property
  def isempty(self):
    return sum(I.size for I in self.allI) == 0

  # the resolution of the measurement, the finest over the pole figures
  @property
  def resolution(self):
    return min(_resolutionOf(r) for r in self.allR)

  def select(self, i):
    idx = [i] if isinstance(i, (int, np.integer)) else list(i)
    return PoleFigure([self.allH[k] for k in idx], [self.allR[k] for k in idx], [self.allI[k] for k in idx],
                      superposition=[self.c[k] for k in idx], complete=True)

  # an integer, slice or list picks pole figures; a boolean mask picks measurements
  def __getitem__(self, key):
    if _isBoolMask(key):
      mask = np.asarray(key, dtype=bool).ravel()
      if mask.size != sum(I.size for I in self.allI):
        raise ValueError("the mask has one entry per specimen direction")
      allR, allI, start = [], [], 0
      for r, I in zip(self.allR, self.allI):
        m = mask[start:start + I.size]
        allR.append(r.flatten()[m])
        allI.append(I.ravel()[m])
        start += I.size
      return self._rebuild(allI, allR)
    if isinstance(key, slice):
      return self.select(range(*key.indices(self.numPF)))
    return self.select(key)

  def __repr__(self):
    lines = [f"PoleFigure ({self.CS.char()} → {self.SS.char()})"]
    for h, r in zip(self.allH, self.allR):
      shape = " x ".join(str(s) for s in r.shape) if r.ndim > 1 else str(r.size)
      lines.append(f"  h = {h.char()}, r = {shape} points")
    return "\n".join(lines)

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
    return self * alpha

  # max(pf) the largest intensity of each pole figure, max(pf, a) the intensities clipped
  def max(self, other=None):
    if other is None:
      return np.array([np.max(I) for I in self.allI])
    return self._apply(other, np.maximum)

  def min(self, other=None):
    if other is None:
      return np.array([np.min(I) for I in self.allI])
    return self._apply(other, np.minimum)

  # the mean intensity of each pole figure, the integral over the sphere
  def mean(self, weights=None):
    if weights is None:
      weights = [calcQuadratureWeights(r) for r in self.allR]
    return np.array([float(np.sum(I.ravel() * np.asarray(w).ravel())) for I, w in zip(self.allI, weights)])

  # scaled to mean one, or to the pole figures the ODF predicts
  def normalize(self, odf=None):
    return self / (self.mean() if odf is None else self.calcNormalization(odf))

  # the specimen directions turned; a rotation that is no element of the specimen group
  # strips the group from the frame
  def rotate(self, rot):
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

  # equivalent specimen directions under the specimen group carry the same pole density,
  # so they are merged and their intensities averaged
  def unique(self, tolerance=1e-6):
    sym = self.SS.properGroup().rotations()
    if self.SS.numSym() <= 2:
      sym = rotation.id
    allR, allI = [], []
    for r, I in zip(self.allR, self.allI):
      xyz = r.flatten().normalize().data
      imgs = (rotation(sym.data.reshape(-1, 1, 4)) * vector3d(xyz)).data     # (|G|, n, 3)
      d = np.einsum("gnk,mk->gnm", imgs, xyz)
      same = (np.abs(d) if r.antipodal else d) > 1 - tolerance
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

  # the pole figures of several data sets in one; those of equal crystal directions joined
  def union(self, *others):
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
      match = [j for j in range(i) if allH[j].size == allH[i].size and np.max(allH[j].angle(allH[i], noSymmetry=True)) < 1e-4]
      if match:
        j = match[0]
        allR[j] = vector3d.cat(allR[j].flatten(), allR[i].flatten())
        allI[j] = np.concatenate([allI[j].ravel(), allI[i].ravel()])
        del allH[i], allR[i], allI[i], c[i]
      else:
        i += 1
    return PoleFigure(allH, allR, allI, superposition=c, complete=True)

  # the measurements whose intensity deviates from the mean of their neighbours within
  # three resolutions by more than alpha standard deviations
  def isOutlier(self, alpha=2):
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

  # Poisson counts of fak * I + background, the background subtracted again
  def noisepf(self, fak=1, background=0, nonnegative=False, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    fak = self._operands(fak)
    allI = []
    for I, f in zip(self.allI, fak):
      counts = rng.poisson(np.maximum(f * I + background, 0)) - background
      allI.append(np.maximum(counts, 0) if nonnegative else counts.astype(float))
    return self._rebuild(allI)

  # ------------------------------------------------------------------ corrections

  # the background subtracted and the defocusing divided out, each correction a pole
  # figure of its own, interpolated onto the measured directions where it was measured
  # elsewhere
  def correct(self, background=None, defocusing=None, defocusingBackground=None):
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

  # the pole figures an ODF predicts on the measured directions
  def calcPoleFigure(self, odf):
    return calcPoleFigure(odf, self.allH, self.allR, superposition=self.c, complete=True)

  # the factor the second pole figure is scaled by to carry the mass of the first
  def calcNormalization(self, other):
    if not isinstance(other, PoleFigure):
      other = self.calcPoleFigure(other)
    out = []
    for r, I1, I2 in zip(self.allR, self.allI, other.allI):
      w = calcQuadratureWeights(r).ravel()
      out.append(float(w @ np.maximum(I1.ravel(), 0) / (w @ np.maximum(I2.ravel(), 0))))
    return np.array(out)

  # the pole figures of the deviation from a recalculation, normalised against each
  # other first; 'relative' is the damped relative error, 'RP', 'l1', 'l2' the others
  def calcErrorPF(self, other, mode="relative", epsilon=None):
    if mode not in ERROR_MODES:
      raise ValueError(f"the error mode is 'relative', 'RP', 'l1' or 'l2', not {mode!r}")
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

  # the mean deviation, one number per pole figure
  def calcError(self, other, mode="relative", epsilon=None):
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
    from ..analysis.pf2odf import calcODF
    return calcODF(self, *args, **kwargs)

  def calcODFIterative(self, *args, **kwargs):
    from ..analysis.pf2odf import calcODFIterative
    return calcODFIterative(self, *args, **kwargs)

  # ------------------------------------------------------------------ joining data sets

  @staticmethod
  def cat(*pfs, axis=0):
    pfs = [pf for pf in pfs if pf is not None]
    return PoleFigure(sum([pf.allH for pf in pfs], []), sum([pf.allR for pf in pfs], []), sum([pf.allI for pf in pfs], []),
                      superposition=sum([pf.c for pf in pfs], []), complete=True)


# ------------------------------------------------------------------ the verbs as functions

def union(pf, *others):
  return pf.union(*others)


def correct(pf, **kwargs):
  return pf.correct(**kwargs)


def noisepf(pf, fak=1, background=0, nonnegative=False, rng=None):
  return pf.noisepf(fak, background, nonnegative, rng)


def isOutlier(pf, alpha=2):
  return pf.isOutlier(alpha)


def calcNormalization(pf, other):
  return pf.calcNormalization(other)


def calcErrorPF(pf, other, mode="relative", epsilon=None):
  return pf.calcErrorPF(other, mode, epsilon)


# the error between a measurement and a recalculation, or between two SO(3) functions
def calcError(a, b, *args, **kwargs):
  if isinstance(a, PoleFigure):
    return a.calcError(b, *args, **kwargs)
  return _calcErrorODF(a, b, *args, **kwargs)


# ------------------------------------------------------------------ pole figures from an ODF

# the pole density of one crystal direction on the specimen directions, the even part
# where the directions are axes
def _poleDensity(odf, h, r):
  P = odf.pdf(h)
  if r.antipodal and not P.antipodal:
    P = P.even()
  return np.real(P(vector3d(r.flatten().data))).reshape(r.shape)


def calcPoleFigure(odf, h, r=None, superposition=None, complete=False, **gridOptions):
  """the pole figures of an ODF for the crystal directions h on the specimen directions r

    calcPoleFigure(odf, h, r)                       one pole figure per element of h
    calcPoleFigure(odf, [h1, h23], r)               a Python list: an entry of several directions is superposed
    calcPoleFigure(odf, [h1, h2], [r1, r2])         one set of specimen directions per pole figure
    calcPoleFigure(odf, h, resolution=5*degree)     on a regular grid
    calcPoleFigure(odf, pf)                         on the directions of a measurement
  """
  if isinstance(odf, PoleFigure):
    odf, h = h, odf
  if isinstance(h, PoleFigure):
    return h.calcPoleFigure(odf)
  if isinstance(h, vector3d):
    hs = [h.flatten()[i] for i in range(h.size)] if h.size > 1 else [h.flatten()]
  else:
    hs = [x if isinstance(x, vector3d) else vector3d(x) for x in h]
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
    rr = _specimenDirections(rr, complete, None)
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
def _clenshawCurtisWeights(N):
  if N == 0:
    return np.array([2.0])
  k = np.arange(N + 1)
  w = np.zeros(N + 1)
  for j in range(0, N // 2 + 1):
    b = 1 if (2 * j == 0 or 2 * j == N) else 2
    w += b * np.cos(2 * j * k * np.pi / N) / (1 - 4 * j * j)
  w *= 2 / N
  w[0] /= 2
  w[-1] /= 2
  return w


def calcQuadratureWeights(v, method="voronoi"):
  """weights that integrate a function from samples at the directions, summing to one

    calcQuadratureWeights(r)                     a regular grid exactly, scattered points by Voronoi areas
    calcQuadratureWeights(r, method='harmonic')  the least norm correction that integrates every harmonic the count allows
  """
  v = v if isinstance(v, vector3d) else vector3d(v)
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
    w = _clenshawCurtisWeights(N)[:len(uth)]
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
  from scipy.spatial import SphericalVoronoi
  xyz = v.flatten().normalize().data
  if v.antipodal:
    xyz = np.concatenate([xyz, -xyz])
  key = np.round(xyz / 1e-6) * 1e-6
  u, inv = np.unique(key, axis=0, return_inverse=True)
  inv = inv.ravel()
  if len(u) < 4:
    return None
  try:
    sv = SphericalVoronoi(u, radius=1.0)
    area = sv.calculate_areas()
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
  v = v if isinstance(v, vector3d) else vector3d(v)
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

  # the correction on the specimen directions of a pole figure, one at the pole
  def defocusing(self, pf):
    return pf._rebuild([self.eval(np.cos(r.theta)) for r in pf.allR])


# ------------------------------------------------------------------ installation

def _install():
  from .bingham import SO3FunBingham
  from .handles import SO3FunComposition, SO3FunGeneric, SO3FunHandle
  from .rbf import SO3FunCBF, SO3FunRBF
  from .so3fun import SO3FunHarmonic
  for cls in (SO3FunHarmonic, SO3FunRBF, SO3FunCBF, SO3FunHandle, SO3FunComposition, SO3FunGeneric, SO3FunBingham):
    cls.calcPoleFigure = lambda self, h, r=None, **kw: calcPoleFigure(self, h, r, **kw)
  vector3d.calcQuadratureWeights = lambda self, method="voronoi": calcQuadratureWeights(self, method)
  vector3d.refine = lambda self: refine(self)


_install()

__all__ = ["PoleFigure", "calcPoleFigure", "calcQuadratureWeights", "calcNormalization", "calcErrorPF", "calcError",
           "correct", "noisepf", "isOutlier", "refine", "SchulzDefocusingKernel"]
