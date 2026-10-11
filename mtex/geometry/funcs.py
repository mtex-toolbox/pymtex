# the verbs as functions: angle(o1, o2) and o1.angle(o2) are one call, and a plain array
# falls through to NumPy
from __future__ import annotations

import builtins

import numpy as np

from ._array import MtexArray, cat
from .arguments import refuseFlags


def _method(name, fallback=None):
  def f(x, *args, **kwargs):
    if isinstance(x, MtexArray) or hasattr(x, name) and not isinstance(x, np.ndarray):
      return getattr(x, name)(*args, **kwargs)
    if fallback is None:
      raise TypeError(f"{name} is not defined for {type(x).__name__}")
    return fallback(x, *args, **kwargs)
  f.__name__ = name
  return f


angle = _method("angle", np.angle)
dot = _method("dot", np.dot)
cross = _method("cross", np.cross)
norm = _method("norm", np.linalg.norm)
normalize = _method("normalize")
inv = _method("inv", np.linalg.inv)
mean = _method("mean", np.mean)
std = _method("std", np.std)
unique = _method("unique", np.unique)
discretize = _method("discretize")
displacement = _method("displacement")
fitStage = _method("fitStage")
matrix = _method("matrix")
Euler = _method("Euler")
Rodrigues = _method("Rodrigues")
axis = _method("axis")
log = _method("log", np.log)
homochoric = _method("homochoric")
cubochoric = _method("cubochoric")
orth = _method("orth")
perp = _method("perp")
polar = _method("polar")
orthProj = _method("orthProj")
det = _method("det", np.linalg.det)
conj = _method("conj", np.conj)
isnan = _method("isnan", np.isnan)
Laue = _method("Laue")
properGroup = _method("properGroup")
properSubGroup = _method("properSubGroup")
disjoint = _method("disjoint")
union = _method("union", np.union1d)
factor = _method("factor")
nfold = _method("nfold")
numSym = _method("numSym")
numProper = _method("numProper")
LaueName = _method("LaueName")
sim = _method("sim")
isAligned = _method("isAligned")
isCompatible = _method("isCompatible")
transformationMatrix = _method("transformationMatrix")
alignment = _method("alignment")
stripSym = _method("stripSym")
fullSym = _method("fullSym")
dspacing = _method("dspacing")
symmetrise = _method("symmetrise")
multiplicity = _method("multiplicity")
projectIntoFundamentalRegion = _method("projectIntoFundamentalRegion")
transformReferenceFrame = _method("transformReferenceFrame")
fundamentalSector = _method("fundamentalSector")
conventionalSector = _method("conventionalSector")
fundamentalRegion = _method("fundamentalRegion")


def round(x, *args, **kwargs):
  """round(x) of an MTEX object by its own round, of an array, a list or with an axis as NumPy
  rounds, of a number as Python rounds, so that from mtex import * keeps Python's round"""
  if isinstance(x, MtexArray) or hasattr(x, "round") and not isinstance(x, (np.ndarray, np.generic)):
    return x.round(*args, **kwargs)
  if isinstance(x, (np.ndarray, list, tuple)) or kwargs:
    return np.round(x, *args, **kwargs)
  return builtins.round(x, *args)


def sum(x, *args, **kwargs):
  """sum(x) of an MTEX object by its own sum, of an array or with an axis as NumPy sums, of any
  other iterable as Python sums, so that from mtex import * keeps Python's sum"""
  if isinstance(x, MtexArray) or hasattr(x, "sum") and not isinstance(x, (np.ndarray, np.generic)):
    return x.sum(*args, **kwargs)
  if isinstance(x, (np.ndarray, np.generic)) or kwargs:
    return np.sum(x, *args, **kwargs)
  return builtins.sum(x, *args)


def checkInside(region, x, **kw):
  return region.checkInside(x, **kw)
find = _method("find", np.flatnonzero)
subGrid = _method("subGrid")


exp = _method("exp", np.exp)
isclose = _method("isclose", np.isclose)
merge = _method("merge")


def rotate(v, rot, *args, **kwargs):
  """rotate(x, rot) is x.rotate(rot), or rot * x for an object without that method; 'left' and
  'right' name the side, a flag is a keyword

    rotate(ebsd, rot, keepEuler=True)
    rotate(odf, rot, 'right')"""
  refuseFlags(args, "rotate", names=("left", "right"))
  side = next((a for a in args if isinstance(a, str)), None)
  args = tuple(a for a in args if not isinstance(a, str))
  if side is not None:
    kwargs["side"] = side
  return v.rotate(rot, *args, **kwargs) if hasattr(v, "rotate") else rot * v


__all__ = ["angle", "dot", "cross", "orthProj", "norm", "normalize", "inv", "mean", "sum", "std", "unique", "matrix", "Euler",
           "Rodrigues", "axis", "log", "exp", "isclose", "homochoric", "cubochoric", "orth", "perp", "polar",
           "det", "conj", "isnan", "rotate", "cat", "merge", "Laue", "properGroup", "properSubGroup", "disjoint", "union", "factor", "nfold", "numSym", "numProper", "LaueName", "sim", "isAligned", "isCompatible", "transformationMatrix", "alignment", "stripSym", "fullSym", "dspacing", "round", "symmetrise", "multiplicity", "projectIntoFundamentalRegion", "transformReferenceFrame", "fundamentalSector", "conventionalSector", "fundamentalRegion", "checkInside", "find", "subGrid", "discretize", "displacement", "fitStage"]
