# the grain boundary criteria (MATLAB grainBoundaryCriteria/, Julia criteria.jl)
#
#   gbcAngle(10 * degree)                 a threshold on the misorientation angle
#   gbcAngle([10 * degree, 2 * degree])   a high and a low angle threshold
#   gbcAngle(..., byPhase={'Forsterite': 15 * degree})   a phase with thresholds of its own
#   gbcSoft(10 * degree, width)           the error function transition instead of a threshold
#   gbcEqual('variantId')                 two measurements belong together when a property agrees
#   gbcCustom(f), gbcCustom(values, threshold)
#   crit.evaluate(ebsd, i, j), connectivity(crit, ebsd, i, j)
#
# A criterion answers, for pairs of neighbouring measurements, how connected they are: 1
# no boundary, 0.5 a low angle boundary inside a grain, 0 a boundary. Anything between
# means the same by the same rule, above zero the two are joined and at most one half a
# boundary is drawn, so a continuous criterion needs no separate path. Different phases
# are never joined; two not indexed measurements always are. The angle criterion compares
# the largest |dot| over the proper group against each level transformed once, never an
# angle in the loop.
from __future__ import annotations

import numpy as np

from ..geometry._array import MtexArray
from ..geometry.kernels import qconj, qmul
from .kernels import grainKernel
from ..geometry.units import degree
from ..geometry.vector3d import vector3d
from ..geometry.display import aligned


class grainBoundaryCriterion:
  """the base of the grain boundary criteria: how connected pairs of neighbouring measurements are,
  1 no boundary, 0.5 a low angle boundary inside a grain, 0 a boundary; a subclass implements
  evaluate, and calling the criterion evaluates it"""

  # a criterion that removes the regions below minPixel itself is handed the value by
  # setMinPixel, and calcGrains skips its own cull
  handlesMinPixel = False

  def evaluate(self, ebsd, i, j):
    """the connectivity of the pairs of neighbouring measurements i[k], j[k] of a map, one value in
    [0, 1] per pair"""
    raise NotImplementedError

  def __call__(self, ebsd, i, j):
    return self.evaluate(ebsd, i, j)


def connectivity(crit, ebsd, i, j):
  """connectivity(crit, ebsd, i, j) is crit.evaluate(ebsd, i, j): the connectivity of the pairs of
  neighbouring measurements"""
  return crit.evaluate(ebsd, i, j)


# the levels of a criterion: one angle, or a high and a low angle one, as a tuple
def _levels(x):
  return tuple(float(t) for t in np.atleast_1d(np.asarray(x, dtype=float)))


# whether a threshold names one entry per phase, MATLAB's cell list in CSList order
def _perPhase(x):
  if not isinstance(x, (list, tuple)):
    return False
  # a pair of numbers is [highAngle, lowAngle]; any other list names the phases in turn
  return len(x) != 2 or any(isinstance(t, (list, tuple, np.ndarray)) for t in x)


# the entry of phase p in a list per phase: one entry per phase of CSList, MATLAB's cell, or one per
# indexed phase in their order
def _phaseEntry(x, CSList, p):
  if len(x) == len(CSList):
    return x[p]
  indexed = [q for q in range(len(CSList)) if CSList[q].isIndexed]
  if len(x) == len(indexed) and p in indexed:
    return x[indexed.index(p)]
  raise ValueError(f"{len(x)} thresholds for {len(indexed)} indexed phases: give one per indexed phase, or one per entry of "
                   f"ebsd.CSList ({len(CSList)}), or a single threshold for all phases")


def _pairs(ebsd, i, j):
  i, j = np.asarray(i).reshape(-1), np.asarray(j).reshape(-1)
  if i.shape != j.shape:
    raise ValueError("i and j are equally sized lists of neighbouring measurements")
  return i, j, ebsd.phaseId.reshape(-1), ebsd.rotations.data.reshape(-1, 4)


def phaseGroups(CSList, dtype=np.float64):
  """the proper groups of the phases of a map as one table for the kernels: their rotations
  padded to the largest group, their sizes, and whether a phase has orientations at all"""
  groups = [fr.sym.properGroup().rotations().data.reshape(-1, 4) if fr.isIndexed else np.zeros((0, 4)) for fr in CSList]
  Gall = np.zeros((len(groups), max([len(G) for G in groups] + [1]), 4), dtype=dtype)
  for p, G in enumerate(groups):
    Gall[p, :len(G)] = G
  nG = np.array([len(G) for G in groups], dtype=np.int64)
  indexed = np.array([fr.isIndexed for fr in CSList], dtype=np.bool_)
  return Gall, nG, indexed


# the largest |<q_i, q_j g>| over the proper group of a frame, for pairs of rotations
def _bestDot(fr, q1, q2):
  G = fr.sym.properGroup().rotations().data.reshape(-1, 4).astype(q1.dtype)
  m = qmul(qconj(q2), q1)
  return np.max(np.abs(m @ G.T), axis=1)


# the misorientation angle of pairs of rotations of one frame, from the largest |dot|
def _misorientationAngle(fr, q1, q2):
  return 2 * np.arccos(np.minimum(_bestDot(fr, q1, q2), 1.0))


# the levels byPhase gives the phase p, None where it names other phases only
def _byPhaseLevels(byPhase, CSList, p):
  for key, t in (byPhase or {}).items():
    if CSList.phaseId(key) == p:
      return _levels(t)
  return None


# the connectivity of the pairs phase by phase: value(p, frame, q1, q2) where both sides are the
# indexed phase p, 1 where both are one not indexed phase, since nothing was measured that
# separates them, and 0 across phases
def _byPhase(CSList, i, j, ph, rot, value):
  out = np.zeros(len(i), dtype=rot.dtype)
  for p, fr in enumerate(CSList):
    sel = (ph[i] == p) & (ph[j] == p)
    if sel.any():
      out[sel] = value(p, fr, rot[i[sel]], rot[j[sel]]) if fr.isIndexed else 1
  return out


# ------------------------------------------------------------------ the angle criterion

# the threshold a grain reconstruction takes when none is given, MATLAB's
DEFAULT_ANGLE = 15 * degree


class gbcAngle(grainBoundaryCriterion):
  """the criterion of a threshold on the misorientation angle: one angle, or a high and a low
  angle threshold, the pair counting as the share of the levels it stays below; a list of
  other than two numbers, or of lists, gives every phase its own, one entry per indexed phase
  or per entry of ebsd.CSList as MATLAB's cell; byPhase gives a phase thresholds by its name

    gbcAngle(10*degree)
    gbcAngle([10*degree, 2*degree])
    gbcAngle([[10*degree], [15*degree, 2*degree]])  # one per indexed phase
    gbcAngle(10*degree, byPhase={'Forsterite': 15*degree})

  Input
    threshold - misorientation threshold in radians, a high/low pair, or a per-phase list of thresholds
    byPhase - dictionary of thresholds keyed by phase name, overriding the common thresholds"""

  def __init__(self, threshold=DEFAULT_ANGLE, byPhase=None):
    self.threshold = threshold
    self.byPhase = byPhase

  def levelsFor(self, CSList, p):
    """the levels of a phase: its own from byPhase or the per phase list, else the common ones"""
    own = _byPhaseLevels(self.byPhase, CSList, p)
    if own is not None:
      return own
    if _perPhase(self.threshold):
      return _levels(_phaseEntry(self.threshold, CSList, p))
    return _levels(self.threshold)

  def evaluate(self, ebsd, i, j, backend=None):
    """the connectivity of the pairs i[k], j[k]: the share of the angle levels their misorientation
    stays below, 0 across phases, 1 between not indexed measurements; a compiled loop when
    numba is the backend; MATLAB's gbcAngle.evaluate

      c = gbcAngle(10*degree).evaluate(ebsd, i, j)                    # 1 joined, 0 a boundary
      c = gbcAngle([10*degree, 2*degree]).evaluate(ebsd, i, j)        # 0.5 for a low angle boundary
      c = gbcAngle(10*degree).evaluate(ebsd, i, j, backend='numpy')   # the NumPy loop

    With a pair [high, low] the value is 1 below low, 0.5 between low and high and 0 above
    high: the share of the levels the misorientation angle stays below. calcGrains joins two
    pixels whose value is 1 into one grain.

    Input
      ebsd - EBSD
      i, j - the flat indices of the pairs of measurements, equally long
    Output
      c - the connectivity of every pair, between 0 and 1
    Keywords
      backend - 'numba' or 'numpy', by default the preference 'kernels'
    See also
      gbcSoft, gbcEqual, calcGrains"""
    i, j, ph, rot = _pairs(ebsd, i, j)
    # all pairs in one compiled loop over the tables of the phases, the loop below its fallback
    CSList = ebsd.CSList
    loop = grainKernel("angleCriterion", backend)
    if loop is not None:
      Gall, nG, indexed = phaseGroups(CSList, rot.dtype)
      levels = [self.levelsFor(CSList, p) if CSList[p].isIndexed else (0.0,) for p in range(len(CSList))]
      cosT = np.zeros((len(CSList), max(len(lv) for lv in levels)), dtype=rot.dtype)
      for p, lv in enumerate(levels):
        cosT[p, :len(lv)] = np.cos(np.array(lv, dtype=rot.dtype) / 2)
      out = np.empty(len(i), dtype=rot.dtype)
      loop(np.ascontiguousarray(rot), np.ascontiguousarray(ph, dtype=np.int64), np.ascontiguousarray(i, dtype=np.int64),
           np.ascontiguousarray(j, dtype=np.int64), Gall, nG, indexed, cosT, np.array([len(lv) for lv in levels], dtype=np.int64), out)
      return out

    # the share of the levels of the phase the half cosine of the pair exceeds
    def share(p, fr, q1, q2):
      cosT = np.cos(np.array(self.levelsFor(CSList, p), dtype=rot.dtype) / 2)
      return np.sum(_bestDot(fr, q1, q2)[:, None] > cosT[None, :], axis=1) / len(cosT)

    return _byPhase(CSList, i, j, ph, rot, share)

  def __repr__(self):
    t = self.threshold if _perPhase(self.threshold) else np.asarray(self.threshold) / degree
    return aligned(f"gbcAngle (threshold {t}°" + (", by phase" if self.byPhase else "") + ")")


# ------------------------------------------------------------------ the soft criterion

class gbcSoft(grainBoundaryCriterion):
  """a smooth transition instead of a threshold, so a pair near the threshold is joined or
  not by how far it is from it rather than by which side it fell on

  Input
    threshold - centre angle of the smooth boundary transition in radians; default 10*degree
    width - angular width of the transition in radians; None uses half the threshold, or the second entry of a threshold/width pair
    byPhase - dictionary of threshold/width settings keyed by phase name"""

  def __init__(self, threshold=10 * degree, width=None, byPhase=None):
    t = _levels(threshold) if not _perPhase(threshold) else (float(threshold[0][0]),)
    self.threshold = t[0]
    self.width = float(width) if width is not None else (t[1] if len(t) > 1 else t[0] / 2)
    self.byPhase = byPhase

  def levelsFor(self, CSList, p):
    """the threshold and the width of the transition for phase p, its own from byPhase or the
    common ones"""
    lv = _byPhaseLevels(self.byPhase, CSList, p)
    if lv is not None:
      return lv[0], (lv[1] if len(lv) > 1 else self.width)
    return self.threshold, self.width

  def evaluate(self, ebsd, i, j):
    """the connectivity of the pairs i[k], j[k]: one minus the error function step of width w about
    the threshold at their misorientation angle, 0 across phases"""
    from scipy.special import erf
    i, j, ph, rot = _pairs(ebsd, i, j)

    # one minus the error function step about the threshold of the phase
    def step(p, fr, q1, q2):
      theta, w = self.levelsFor(ebsd.CSList, p)
      return 1 - (1 + erf(2 * (_misorientationAngle(fr, q1, q2) - theta) / w)) / 2

    return _byPhase(ebsd.CSList, i, j, ph, rot, step)

  def __repr__(self):
    return aligned(f"gbcSoft (threshold {self.threshold / degree:.4g}°, width {self.width / degree:.4g}°)")


# ------------------------------------------------------------------ by a property

class gbcEqual(grainBoundaryCriterion):
  """two measurements belong together when the values of a property agree, which is how a
  segmentation by variant, by parent grain or by any imported label is done

  Input
    key - name of an EBSD property whose equal labels define connected measurements"""

  def __init__(self, key):
    self.key = key

  def evaluate(self, ebsd, i, j):
    """the connectivity of the pairs i[k], j[k]: 1 where the property agrees, 0 where it differs"""
    i, j, _, rot = _pairs(ebsd, i, j)
    v = getattr(ebsd, self.key) if isinstance(self.key, str) else np.asarray(self.key)
    v = v.reshape((-1,) + v.shape[ebsd.ndim:]) if isinstance(self.key, str) else v.reshape((ebsd.size, -1))
    same = np.all((v[i] == v[j]).reshape(len(i), -1), axis=1)
    return same.astype(rot.dtype)

  def __repr__(self):
    return aligned(f"gbcEqual ({self.key!r})" if isinstance(self.key, str) else "gbcEqual (values)")


class gbcCustom(grainBoundaryCriterion):
  """a criterion given as a function of the map and the two measurement indices, or as a
  per-pixel property, its values or the name of a property of the map, with the largest
  difference that still counts as one grain, gbcCustom('grainId', 0.5)

  Input
    values - callable of (ebsd, i, j), per-pixel values, or the name of a pixel property
    threshold - largest allowed difference for values to remain connected; not used for a callable criterion
    antipodal - treat opposite vectors or axes alike in angular comparisons; default False"""

  def __init__(self, values, threshold=None, antipodal=False):
    if callable(values):
      self.f, self.values, self.threshold = values, None, None
    else:
      self.f, self.values, self.threshold = None, values, float(threshold)
    self.antipodal = antipodal

  def evaluate(self, ebsd, i, j):
    """the connectivity of the pairs i[k], j[k]: the function's value, or 1 where the property
    differs by at most the threshold, as an angle for vectors and orientations, 0 otherwise"""
    i, j, _, rot = _pairs(ebsd, i, j)
    if self.f is not None:
      return np.asarray(self.f(ebsd, i, j), dtype=rot.dtype).reshape(len(i))
    v = ebsd.prop[self.values] if isinstance(self.values, str) else self.values
    if isinstance(v, MtexArray):
      flat = v.flatten()
      d = flat[i].angle(flat[j], antipodal=self.antipodal or None) if isinstance(v, vector3d) else flat[i].angle(flat[j])
    else:
      v = np.asarray(v).reshape(ebsd.size, -1)
      d = np.linalg.norm(v[i] - v[j], axis=1)
    return (d <= self.threshold).astype(rot.dtype)

  def __repr__(self):
    return aligned("gbcCustom (function)" if self.f is not None else f"gbcCustom (threshold {self.threshold:.4g})")


__all__ = ["grainBoundaryCriterion", "gbcAngle", "gbcSoft", "gbcEqual", "gbcCustom", "connectivity"]
