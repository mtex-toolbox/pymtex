# orientations defined by directions, the named orientation relationships, the variants of
# a relationship and the coincidence site lattices (MATLAB orientation/byMiller,
# orientation/variants, the static methods of orientation.m)
from __future__ import annotations


import numpy as np

from .frames import referenceFrame
from .kernels import qconj
from .frames import crystalFrame, specimenFrame
from .install import install
from .miller import Miller
from .orientation import orientation
from .quaternion import qmul, rotation
from .symmetryops import projectIntoFundamentalRegionOrientation, symmetrise
from .symreduce import _firstOfClusters, reductionGroups, symReduce
from .units import degree
from .vector3d import vector3d, xvector, zvector


def phaseFrame(x):
  """the group of a direction's phase: a direction written group-free for one plane still
  belongs to the phase, which the register knows as the largest group over that basis"""
  if not isinstance(x, vector3d) or x.frame is None:
    return None
  if x.isCrystalDirection and x.frame.numSym() == 1:
    return x.frame.fullSym()
  return x.frame


def byMiller(hkl, uvw, *frames):
  """the orientation with ori * hkl parallel to z and ori * uvw parallel to x, from the
  symmetric equivalent of uvw closest to perpendicular to hkl; among those equally close
  the one closest to uvw itself, so the choice does not depend on the element order"""
  from .orientation import orientation
  frames = [f for f in frames if isinstance(f, referenceFrame)]
  if not isinstance(hkl, vector3d):
    cs = frames.pop(0)
    hkl, uvw = Miller(*np.asarray(hkl, dtype=float), cs), Miller(*np.asarray(uvw, dtype=float), cs, "uvw")
  # the frame of hkl given again is the crystal frame, the next one the specimen frame
  frames = [f for f in frames if f is not phaseFrame(hkl)]
  ss = frames[0] if frames else specimenFrame.default
  n = hkl.normalize().data
  u = uvw.normalize().data
  cands = symmetrise(uvw).data.reshape(-1, 3)
  cands = cands / np.linalg.norm(cands, axis=1, keepdims=True)
  dev = np.round(np.abs(np.pi / 2 - np.arctan2(np.linalg.norm(np.cross(cands, n), axis=1), cands @ n)), 8)
  score = np.lexsort((-np.round(cands[:, 2], 8), -np.round(cands[:, 1], 8), -np.round(cands[:, 0], 8), -np.round(cands @ u, 8), dev))
  d = cands[score[0]]
  if dev[score[0]] > 1 * degree:
    import warnings
    warnings.warn(f"the Miller indices are not orthogonal, deviation {dev[score[0]] / degree:.3g}°")
  d = d - (d @ n) * n
  return orientation(rotation.map(vector3d(n), zvector, vector3d(d), xvector), phaseFrame(hkl), ss)


def relationship(nParent, nChild, dParent, dChild):
  """each relationship maps a plane and a direction of the parent onto those of the child"""
  from .orientation import orientation
  return orientation.map(nParent, nChild, dParent, dChild, phaseFrame(nParent), phaseFrame(nChild))


def Bain(csG, csA):
  """the Bain orientation relationship from fcc to bcc, (100) || (100) and [010] || [011], a
  misorientation from the parent frame csG to the child frame csA

    p2c = orientation.Bain(csGamma, csAlpha)"""
  return relationship(Miller(1, 0, 0, csG), Miller(1, 0, 0, csA), Miller(0, 1, 0, csG, "uvw"), Miller(0, 1, 1, csA, "uvw"))


def KurdjumovSachs(csG, csA):
  """the Kurdjumov-Sachs orientation relationship from fcc to bcc, (111) || (011) and
  [-101] || [-1-11], a misorientation from the parent frame csG to the child frame csA

    p2c = orientation.KurdjumovSachs(csGamma, csAlpha)"""
  return relationship(Miller(1, 1, 1, csG), Miller(0, 1, 1, csA), Miller(-1, 0, 1, csG, "uvw"), Miller(-1, -1, 1, csA, "uvw"))


def NishiyamaWassermann(csG, csA):
  """the Nishiyama-Wassermann orientation relationship from fcc to bcc, (111) || (011) and
  [11-2] || [0-11], a misorientation from the parent frame csG to the child frame csA

    p2c = orientation.NishiyamaWassermann(csGamma, csAlpha)"""
  return relationship(Miller(1, 1, 1, csG), Miller(0, 1, 1, csA), Miller(1, 1, -2, csG, "uvw"), Miller(0, -1, 1, csA, "uvw"))


def Pitsch(csG, csA):
  """the Pitsch orientation relationship from fcc to bcc, (010) || (101) and [101] || [-111], a
  misorientation from the parent frame csG to the child frame csA

    p2c = orientation.Pitsch(csGamma, csAlpha)"""
  return relationship(Miller(0, 1, 0, csG), Miller(1, 0, 1, csA), Miller(1, 0, 1, csG, "uvw"), Miller(-1, 1, 1, csA, "uvw"))


def GreningerTrojano(csG, csA):
  """the Greninger-Trojano orientation relationship from fcc to bcc, by the Bunge Euler angles
  (2.7, 46.6, 7.5) degrees of the inverse, a misorientation from csG to csA

    p2c = orientation.GreningerTrojano(csGamma, csAlpha)"""
  from .orientation import orientation
  return orientation.byEuler(2.7 * degree, 46.6 * degree, 7.5 * degree, csA, csG).inv()


def Bagaryatsky(csT, csA):
  """the Bagaryatsky orientation relationship between cementite and ferrite, (100) || (1-10) and
  [010] || [111], a misorientation from the cementite frame csT to the ferrite frame csA

    mori = orientation.Bagaryatsky(csTheta, csAlpha)"""
  return relationship(Miller(1, 0, 0, csT), Miller(1, -1, 0, csA), Miller(0, 1, 0, csT, "uvw"), Miller(1, 1, 1, csA, "uvw"))


def Burgers(csParent, csChild):
  """the Burgers orientation relationship between bcc and hcp, (110) || (0001) and
  [-11-1] || [2-1-10], from csParent to csChild; a hexagonal parent gives the inverse of the one
  from the child

    p2c = orientation.Burgers(csBeta, csAlpha)"""
  if csParent.lattice.isTriHex:
    return Burgers(csChild, csParent).inv()
  return relationship(Miller(1, 1, 0, csParent), Miller(0, 0, 0, 1, csChild), Miller(-1, 1, -1, csParent, "uvw"), Miller(2, -1, -1, 0, csChild, "uvw"))


def PitschSchrader(csParent, csChild):
  """the Pitsch-Schrader orientation relationship between bcc and hcp, (011) || (0001) and
  [100] || [2-1-10], a misorientation from csParent to csChild

    p2c = orientation.PitschSchrader(csParent, csChild)"""
  return relationship(Miller(0, 1, 1, csParent), Miller(0, 0, 0, 1, csChild), Miller(1, 0, 0, csParent, "uvw"), Miller(2, -1, -1, 0, csChild, "uvw"))


def Potter(csParent, csChild):
  """the Potter orientation relationship between bcc and hcp, (011) || (01-11) and
  [11-1] || [2-1-10], a misorientation from csParent to csChild

    p2c = orientation.Potter(csParent, csChild)"""
  return relationship(Miller(0, 1, 1, csParent), Miller(0, 1, -1, 1, csChild), Miller(1, 1, -1, csParent, "uvw"), Miller(2, -1, -1, 0, csChild, "uvw"))


def ShojiNishiyama(csG, csE):
  """the Shoji-Nishiyama orientation relationship from fcc to hcp, (111) || (0001) and
  [10-1] || [2-1-10], a misorientation from the parent frame csG to the child frame csE

    p2c = orientation.ShojiNishiyama(csGamma, csEps)"""
  return relationship(Miller(1, 1, 1, csG), Miller(0, 0, 0, 1, csE), Miller(1, 0, -1, csG, "uvw"), Miller(2, -1, -1, 0, csE, "uvw"))


def variants(p2c, second=None, ids=None, parent=False):
  """the child variants of a parent-to-child relationship: the distinct p2c * r over the parent
  group modulo the child group; the parent variants with the sides swapped; the child
  orientations of parents, all of them or one chosen per parent by its id; the parent
  orientations of children, whose frame is the child's. The variants are numbered as MTEX
  numbers them, which is what a variant id means

    p2cVariants = variants(p2c)                      # the child variants
    p2pVariants = variants(p2c, parent=True)         # the parent variants
    oriChild = variants(p2c, oriParent)              # every child variant of every parent
    oriChild = variants(p2c, oriParent, variantId)   # one per parent, chosen by its id
    oriParent = variants(p2c, oriChild)              # the parents of children
    odfChild = variants(p2c, odfParent)              # the child texture"""
  from .orientation import orientation
  if isinstance(second, str):
    parent, second = second == "parent", None
  if isinstance(ids, str):
    parent, ids = ids == "parent", None
  if second is None:
    return p2c._new(_variantData(p2c, parent).copy())
  # variants(p2c, odf): the child texture, by the function spaces above this layer
  if not isinstance(second, orientation) and hasattr(second, "eval"):
    from ..functions.funops import transformODF
    return transformODF(second, p2c, ids)
  par = second.data.reshape(-1, 4)
  if second.frameA is p2c.frameB and second.frameA is not p2c.frameA:
    PV = variants(p2c, parent=True).data
    PV = PV if ids is None else PV[np.asarray(ids).reshape(-1) - 1]
    q = qmul(par[:, None, :], PV[None, :, :]) if ids is None else qmul(par, PV)
    return orientation(rotation._fromData(q.reshape(-1, 4)), p2c.frameA, second.frameB)
  # a variant id counts from the representative of the parent in the fundamental region
  par = projectIntoFundamentalRegionOrientation(second).data.reshape(-1, 4)
  vs = variants(p2c)
  if ids is None:
    q = qmul(par[:, None, :], qconj(vs.data)[None, :, :])
    return orientation(rotation._fromData(q.reshape(-1, 4)), p2c.frameB, second.frameB)
  ids = np.asarray(ids).reshape(-1) - 1
  return orientation(rotation._fromData(qmul(par, qconj(vs.data[ids]))), p2c.frameB, second.frameB)


# MTEX's numbering of the 24 child variants of a cubic parent, Morito's for Kurdjumov-Sachs
KS_ORDER = np.array([1, 17, 2, 16, 3, 18, 8, 22, 9, 24, 7, 23, 19, 11, 20, 10, 21, 12, 6, 15, 4, 14, 5, 13])


def mtexEnumeration(sym):
  """the proper rotations of a group in the order MTEX enumerates them, the order its variant
  ids are counted in: the products r1 r2 r3 of the cyclic groups about the generating axes
  of its Laue class, the first factor running fastest; the closure order where no
  generator list of MTEX spans the group"""
  G = sym.properGroup().rotations().data.reshape(-1, 4)
  a, b, c = sym.axes
  for gens in ([], [(a, 2)], [(b, 2)], [(c, 2)], [(a, 2), (c, 2)], [(c, 3)], [(a, 2), (c, 3)], [(a - b, 2), (c, 3)], [(c, 4)],
               [(a, 2), (c, 4)], [(c, 6)], [(a, 2), (c, 6)], [(a + b + c, 3), (a, 2), (c, 2)], [(a + b + c, 3), (a + b, 2), (c, 4)]):
    if int(np.prod([n for _, n in gens])) != len(G):
      continue
    E = np.array([[1.0, 0, 0, 0]])
    for v, n in gens:
      w = np.arange(n) * 2 * np.pi / n
      r = np.c_[np.cos(w / 2), np.sin(w / 2)[:, None] * (v / np.linalg.norm(v))]
      E = qmul(E[None, :, :], r[:, None, :]).reshape(-1, 4)
    if np.all(np.max(np.abs(E @ G.T), axis=1) > 1 - 1e-6) and np.all(np.max(np.abs(G @ E.T), axis=1) > 1 - 1e-6):
      return E
  return G


# the variants of a relationship by its rotation, its frames and the side, computed once: a
# reconstruction asks for them once per grain
_VARIANTS = {}


def _variantData(p2c, parent):
  key = (p2c.data.tobytes(), p2c.frameA, p2c.frameB, bool(parent))
  if key not in _VARIANTS:
    if len(_VARIANTS) >= 64:
      _VARIANTS.clear()
    if parent:
      G = mtexEnumeration(p2c.frameB.sym)
      cand = p2c._new(qmul(G, p2c.data[None]))
      keep = _distinctModulo(cand, "right")
    else:
      G = mtexEnumeration(p2c.frameA.sym)
      if len(G) == 24:
        G = G[KS_ORDER - 1]
      cand = p2c._new(qmul(p2c.data[None], G))
      keep = _distinctModulo(cand, "left")
    _VARIANTS[key] = cand.data[keep]
  return _VARIANTS[key]


# the candidates distinct modulo the group of one side, the first of every class kept
def _distinctModulo(cand, side):
  L, R = reductionGroups(cand)
  one = np.array([[1.0, 0, 0, 0]])
  L, R = (L, one) if side == "left" else (one, R)
  q = cand.data.reshape(-1, 4)
  return np.flatnonzero(_firstOfClusters(symReduce(q[:, None], q[None, :], L, R)[0] > 1 - 1e-4))


def copper(CS, *ss):
  """(112)[-1-11] is Bunge (90, 35, 45), (112)[11-1] its twin (270, 35, 45)"""
  return byMiller([1, 1, 2], [-1, -1, 1], CS, *ss)


def copper2(CS, *ss):
  """the second copper component, (112)[11-1], Bunge (270, 35, 45) degrees, in cs and the
  specimen frame

    ori = orientation.copper2(cs)"""
  return byMiller([1, 1, 2], [1, 1, -1], CS, *ss)


# the other named components of rolling textures, by their Bunge Euler angles in degrees
STANDARD_EULER = {"cube": (0, 0, 0), "cubeND22": (22, 0, 0), "cubeND45": (45, 0, 0), "cubeRD": (0, 22, 0), "goss": (0, 45, 0),
                  "SR": (53, 35, 63), "SR2": (233, 35, 63), "SR3": (307, 35, 27), "SR4": (127, 35, 27), "brass": (35, 45, 0),
                  "brass2": (325, 45, 0), "PLage": (65, 45, 0), "PLage2": (245, 45, 0), "QLage": (65, 20, 0), "QLage2": (245, 20, 0),
                  "QLage3": (115, 160, 0), "QLage4": (295, 160, 0), "invGoss": (90, 45, 0)}


def _byEulerDegrees(name):
  def build(CS, *ss):
    from .orientation import orientation
    from .units import degree
    a, b, c = STANDARD_EULER[name]
    return orientation.byEuler(a * degree, b * degree, c * degree, CS, *ss)
  build.__name__ = name
  build.__doc__ = "the {} component, Bunge Euler angles ({}°, {}°, {}°), in cs and the specimen frame".format(name, *STANDARD_EULER[name])
  return build


cube = _byEulerDegrees("cube")
cubeND22 = _byEulerDegrees("cubeND22")
cubeND45 = _byEulerDegrees("cubeND45")
cubeRD = _byEulerDegrees("cubeRD")
goss = _byEulerDegrees("goss")
SR = _byEulerDegrees("SR")
SR2 = _byEulerDegrees("SR2")
SR3 = _byEulerDegrees("SR3")
SR4 = _byEulerDegrees("SR4")
brass = _byEulerDegrees("brass")
brass2 = _byEulerDegrees("brass2")
PLage = _byEulerDegrees("PLage")
PLage2 = _byEulerDegrees("PLage2")
QLage = _byEulerDegrees("QLage")
QLage2 = _byEulerDegrees("QLage2")
QLage3 = _byEulerDegrees("QLage3")
QLage4 = _byEulerDegrees("QLage4")
invGoss = _byEulerDegrees("invGoss")


# the named components and the orientation relationships as constructors, orientation.brass(cs)
for _f in (copper, copper2, *(globals()[n] for n in STANDARD_EULER), Bain, KurdjumovSachs, NishiyamaWassermann, Pitsch,
           GreningerTrojano, Bagaryatsky, Burgers, PitschSchrader, Potter, ShojiNishiyama):
  install(orientation, staticmethod(_f))
del _f


# the standard orientations and the orientation relationships are reached as orientation.copper(cs),
# orientation.KurdjumovSachs(csP, csC); the module itself exports what has no class to hang on
__all__ = ["variants", "STANDARD_EULER"]
