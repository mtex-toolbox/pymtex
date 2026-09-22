# the crystal axes a, b, c in Cartesian coordinates (MATLAB @crystalFrame/private/calcAxis.m)
#
#   crystalAxes(lattice, abc, abg, alignment=['X||a', 'Z||c*'], rotate=rot)
#
# The default alignment is X||a*, Z||c. Any other is asked for by naming which lattice
# direction a Cartesian axis lies along: a, b, c, their reciprocals a*, b*, c*, m = a + b
# and d = a + b + c; 'EDAX' (also 'TSL', 'OIM') is the vendor alignment X||a for the
# lattices where a and a* differ. One construction serves every lattice: the axes from
# the metric with a* along x and c along z, then turned into the alignment asked for.
# Axes are (3, 3) arrays, one axis per row.
from __future__ import annotations

import re

import numpy as np

from ._array import MtexArray

from .latticetype import latticeType

ALIGNMENT_VENDORS = ("EDAX", "TSL", "OIM")


# missing lengths and angles are filled the way MTEX fills them
def completeCell(abc, abg):
  a, b, c = (float(x) for x in abc)
  if c == 0:
    c = max(a, b, c)
  if c == 0:
    c = 1.0
  if a == 0:
    a = b
  if b == 0:
    b = a
  if b == 0:
    a = b = c
  alpha, beta, gamma = (float(x) for x in abg)
  if alpha == 0:
    if beta == 0:
      beta = np.pi / 2
    alpha = np.pi - beta
  if beta == 0:
    beta = np.pi - alpha
  if gamma == 0:
    gamma = np.pi / 2
  return (a, b, c), (alpha, beta, gamma)


# a lattice constrains its cell
def checkCell(lattice, abc, abg, uniqueAxis=1):
  a, b, c = abc
  if lattice is latticeType.monoclinic:
    if not all(abs(abg[k] - np.pi / 2) < 1e-5 for k in range(3) if k != uniqueAxis):
      raise ValueError("for a monoclinic lattice the angles with the symmetry axis have to be 90 degree")
  elif lattice in (latticeType.trigonal, latticeType.hexagonal, latticeType.tetragonal):
    if a != b:
      raise ValueError(f"for a {lattice} lattice a and b must be equal")
  elif lattice in (latticeType.cubic, latticeType.icosahedral):
    if not a == b == c:
      raise ValueError(f"for a {lattice} lattice a, b and c must be equal")


# the axes in the default alignment X||a*, Z||c, from the metric
def defaultAxes(abc, abg):
  (a, b, c), (alpha, beta, gamma) = abc, abg
  ca, cb, cg, sa = np.cos(alpha), np.cos(beta), np.cos(gamma), np.sin(alpha)
  V = np.sqrt(1 + 2 * ca * cb * cg - ca ** 2 - cb ** 2 - cg ** 2)
  return np.array([[a * V / sa, a * (cg - ca * cb) / sa, a * cb], [0, b * sa, b * ca], [0, 0, c]])


# the reciprocal axes a*, b*, c*
def reciprocalAxes(axes):
  axes = np.asarray(axes, dtype=float)
  a, b, c = axes
  d = np.dot(a, np.cross(b, c))
  return np.array([np.cross(b, c), np.cross(c, a), np.cross(a, b)]) / d


# 'X||a', 'a||X', 'z||c*' -> ('X', 'a'), ('Z', 'c*')
def parseAlignment(s):
  m = re.match(r"^\s*([xyzXYZ])\s*\|\|\s*([abcdmABCDM]\*?)\s*$", s)
  if m is not None:
    return m.group(1).upper(), m.group(2).lower()
  m = re.match(r"^\s*([abcdmABCDM]\*?)\s*\|\|\s*([xyzXYZ])\s*$", s)
  if m is not None:
    return m.group(2).upper(), m.group(1).lower()
  raise ValueError(f"'{s}' is not an alignment like X||a or Z||c*")


def isAlignment(s):
  return isinstance(s, str) and (s.upper() in ALIGNMENT_VENDORS or re.match(r"^\s*([xyzXYZ]\s*\|\|\s*[abcdmABCDM]\*?|[abcdmABCDM]\*?\s*\|\|\s*[xyzXYZ])\s*$", s) is not None)


# the lattice direction named by an alignment token
def latticeDirection(token, axes, dual):
  a, b, c = axes
  table = {"a": a, "b": b, "c": c, "a*": dual[0], "b*": dual[1], "c*": dual[2], "m": a + b, "d": a + b + c}
  try:
    return table[token]
  except KeyError:
    raise ValueError(f"unknown lattice direction '{token}'") from None


def _unit(v):
  return v / np.linalg.norm(v)


# the axes turned into the requested alignment
def alignAxes(axes, alignment):
  spec = dict(parseAlignment(s) for s in alignment)
  # with no Z and at most one of X, Y given, Z follows c - or c* when a direct axis was named
  if "Z" not in spec and sum(k in spec for k in ("X", "Y")) <= 1:
    spec["Z"] = "c*" if any(t in ("a", "b", "c") for t in spec.values()) else "c"
  # with Z alone given, X takes the partner that keeps the usual setting
  if "Z" in spec and "X" not in spec and "Y" not in spec:
    partner = {"a": "c*", "c": "a*", "a*": "c", "c*": "a"}
    if spec["Z"] in partner:
      spec["X"] = partner[spec["Z"]]
  dual = reciprocalAxes(axes)
  dirs = {k: np.asarray(latticeDirection(t, axes, dual), dtype=float) for k, t in spec.items()}
  # a missing axis is the cross product of the two others, in cyclic order
  cyc = {"X": ("Y", "Z"), "Y": ("Z", "X"), "Z": ("X", "Y")}
  for k in ("X", "Y", "Z"):
    if k not in dirs:
      p, q = cyc[k]
      dirs[k] = np.cross(dirs[p], dirs[q])
  M = np.array([_unit(dirs[k]) for k in ("X", "Y", "Z")])   # rows: the new x, y, z in old coordinates
  if np.linalg.norm(M @ M.T - np.eye(3)) >= 1e-6:
    raise ValueError("bad alignment options: the requested axes are not orthogonal")
  if np.linalg.det(M) < 0:
    M[1] = -M[1]
  return axes @ M.T


def crystalAxes(lattice, abc, abg=None, alignment=(), rotate=None, uniqueAxis=1, check=True):
  abg = lattice.defaultAngles if abg is None else abg
  abc, abg = completeCell(abc, abg)
  if check:
    checkCell(lattice, abc, abg, uniqueAxis)
  axes = defaultAxes(abc, abg)
  alignment = [str(s) for s in alignment]
  if any(s.upper() in ALIGNMENT_VENDORS for s in alignment):
    alignment = [s for s in alignment if s.upper() not in ALIGNMENT_VENDORS]
    # the vendor aligns x with a wherever a and a* differ in the a-b plane
    if not alignment and lattice in (latticeType.triclinic, latticeType.trigonal, latticeType.hexagonal):
      alignment.append("X||a")
  if alignment:
    axes = alignAxes(axes, alignment)
  if rotate is not None:
    from .vector3d import vector3d
    axes = (rotate * vector3d(axes)).data if isinstance(rotate, MtexArray) else axes @ np.asarray(rotate).T
  return axes


# the alignment of the Cartesian axes as strings, e.g. ['X||a', 'Z||c*']: nothing for an
# orthogonal basis, and the direct axis where it coincides with a reciprocal one
def alignment(axes, names=("a", "b", "c")):
  axes = np.asarray(axes, dtype=float)
  n = axes / np.linalg.norm(axes, axis=1, keepdims=True)
  if np.all(np.abs(n @ n.T - np.eye(3)) < 1e-6):
    return []
  dual = reciprocalAxes(axes)
  dirs = np.concatenate([n, dual / np.linalg.norm(dual, axis=1, keepdims=True)])
  labels = list(names) + [x + "*" for x in names]
  out = []
  for X, e in zip(("X", "Y", "Z"), np.eye(3)):
    hits = np.flatnonzero(np.abs(dirs @ e) > 1 - 1e-6)
    if hits.size:
      out.append(X + "||" + labels[hits[0]])
  return out


__all__ = ["crystalAxes", "reciprocalAxes"]
