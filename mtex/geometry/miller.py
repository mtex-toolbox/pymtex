# crystal directions and lattice planes (MATLAB geometry/Miller.m)
#
#   Miller(h, k, l, cs)                  a plane normal, indices in the reciprocal basis
#   Miller(h, k, i, l, cs)               four index form, i = -(h + k)
#   Miller(u, v, w, cs, 'uvw')           a lattice direction
#   Miller(U, V, T, W, cs, 'UVTW')       Weber indices
#   Miller([[h, k, l], ...], cs)         a list of indices, one per row
#   Miller('(hkl)', cs), '[uvw]'         one plane or direction: the group-free sibling of cs
#   Miller('{hkl}', cs), '<uvw>'         the whole symmetrically equivalent set: cs itself
#   Miller(v, cs)                        Cartesian components restated in cs
#
# A crystal direction is a vector3d written in a crystalFrame: it stores Cartesian
# coordinates and reads hkl, hkil, uvw and UVTW off the reciprocal and direct axes of the
# frame. A crystal direction carries no antipodal state (Julia ADR 0002): 'antipodal'
# means the Laue sibling of its frame.
from __future__ import annotations

import re

import numpy as np

from .frames import crystalFrame
from .session import getMTEXpref
from .vector3d import vector3d

STYLES = ("hkl", "hkil", "uvw", "UVTW", "xyz")
# MATLAB reads the style names without regard to case, 'HKIL' for 'hkil'
_STYLE = {s.lower(): s for s in STYLES}


def Miller(*args, antipodal=False, dispStyle=None):
  """crystal directions and lattice plane normals, a vector3d in a crystal frame

  antipodal=True adds the antipodal symmetry, dispStyle= the indices the display shows ('hkl',
  'hkil', 'uvw', 'UVTW', 'xyz')

    m = Miller(h, k, l, cs)                    # a lattice plane normal
    m = Miller(h, k, i, l, cs)                 # four indices, hexagonal and trigonal lattices
    m = Miller(u, v, w, cs, 'uvw')             # a lattice direction, 'UVTW' with four indices
    m = Miller([[1, 0, 0], [1, 1, 1]], cs)     # many at once, the indices along the last axis
    m = Miller('{111}', cs)                    # (hkl) or [uvw] one, {hkl} or <uvw> the whole family
    m = Miller('[100]\\[011]', cs)             # the plane holding both directions, the zone law
    m = Miller('(100)\\(011)', cs)             # the direction both planes hold
    m = Miller(v, cs)                          # a vector3d expressed in the crystal frame
  """
  style = dispStyle
  for a in args:
    if isinstance(a, str) and a.lower() in _STYLE:
      style = _STYLE[a.lower()]
  args = [a for a in args if not (isinstance(a, str) and a.lower() in _STYLE)]
  frames = [a for a in args if isinstance(a, crystalFrame)]
  if not frames:
    if args and isinstance(args[0], vector3d) and args[0].isCrystalDirection:
      frames = [args[0].frame]
    else:
      raise ValueError("no crystal frame has been specified when defining a crystal direction")
  cs = frames[0]
  args = [a for a in args if not isinstance(a, crystalFrame)]
  if antipodal:
    cs = cs.Laue()
  if not args:
    m = vector3d(frame=cs)
  elif isinstance(args[0], vector3d):
    m = vector3d(args[0])
    m.frame = cs
    # a plain vector, or a tangent vector, becomes Miller indices as MATLAB's Miller(v, cs)
    if style is not None:
      m.dispStyle = style
    elif args[0].dispStyle == "xyz" or hasattr(args[0], "oriRef"):
      m.dispStyle = "hkil" if cs.lattice.isTriHex else "hkl"
    return m
  elif isinstance(args[0], str):
    m, single = _parse(args[0], cs)
    if single:
      m.frame = cs.stripSym()
    return m
  else:
    idx = [np.asarray(a, dtype=float) for a in args]
    if len(idx) == 1:
      idx = list(np.moveaxis(np.atleast_2d(idx[0]), -1, 0))
    if len(idx) not in (3, 4):
      raise ValueError("you need three or four Miller indices")
    idx = np.stack(np.broadcast_arrays(*idx), axis=-1)
    direct = style in ("uvw", "UVTW")
    if idx.shape[-1] == 4 and np.any(np.abs(idx[..., 0] + idx[..., 1] + idx[..., 2]) > 1e-8):
      import warnings
      warnings.warn(("convention U + V + T = 0 violated" if direct else "convention h + k + i = 0 violated") + ", assuming the third index from the first two")
    m = vector3d(np.zeros(idx.shape[:-1] + (3,)), frame=cs)
    if style == "xyz":
      m.data[...] = idx[..., :3]
      m.dispStyle = "xyz"
      return m
    if direct:
      if idx.shape[-1] == 3 and style != "UVTW":
        m.uvw = idx
      else:
        m.UVTW = idx
    else:
      m.hkl = idx
    # four indices given are shown as four, whichever of the two names asked for them
    if idx.shape[-1] == 4 and style is not None:
      style = "UVTW" if direct else "hkil"
  if style is not None:
    m.dispStyle = style
  return m


# '(hkl)', '{hkl}', '[uvw]', '<uvw>': the bracket says direct or reciprocal, and whether the
# direction stands for itself or for its whole family; '[uvw]\\[uvw]' is the plane (hkl) holding
# both directions and '(hkl)\\(hkl)' the direction [uvw] both planes hold, the zone law
_BRACKET = r"\s*([(\[{<⟨])\s*([-\d,\s]+)\s*([)\]}>⟩])\s*"


def _parse(s, CS):
  m = re.fullmatch(_BRACKET + r"(?:\\" + _BRACKET + ")?", s)
  if m is None:
    raise ValueError(f"cannot read '{s}' as Miller indices, write (hkl), {{hkl}}, [uvw], <uvw>, [uvw]\\[uvw] or (hkl)\\(hkl)")
  parts = [_indices(s, m.group(1), m.group(2), CS)]
  if m.group(4) is not None:
    parts.append(_indices(s, m.group(4), m.group(5), CS))
  direct = [d for _, d, _ in parts]
  single = any(o in ("(", "[") for o in (m.group(1), m.group(4)))
  if len(parts) == 1:
    return parts[0][0], single
  if direct[0] != direct[1]:
    raise ValueError(f"'{s}' joins a direction and a plane; the zone law takes two of a kind")
  # the cross product of the three index forms is the other kind, shown as many indices as given
  a, b = (p[0].uvw if direct[0] else p[0].hkl for p in parts)
  four = any(n == 4 for _, _, n in parts)
  idx = np.cross(a, b)
  style = ("hkil" if four else "hkl") if direct[0] else ("UVTW" if four else "uvw")
  out = Miller(*idx, CS, "hkl" if direct[0] else "uvw")
  out.dispStyle = style
  # the three index forms of four indices are thirds; the result is put back to integers
  return (out.round() if four else out), single


# the indices of one bracket, whether they are direct and how many were written
def _indices(s, open_, body, CS):
  idx = [float(x) for x in body.split(",")] if "," in body else [float(x) for x in re.findall(r"-?\d", body)]
  if len(idx) not in (3, 4):
    raise ValueError(f"'{s}' does not hold three or four indices")
  direct = open_ in ("[", "<", "⟨")
  style = ("UVTW" if len(idx) == 4 else "uvw") if direct else ("hkil" if len(idx) == 4 else "hkl")
  return Miller(*idx, CS, style), direct, len(idx)


# ------------------------------------------------------------------ rounding

def roundIndices(v, style=None, maxHKL=12):
  """the smallest multiple whose indices are close to integers, up to maxHKL; UVTW is reduced
  by its gcd, the smallest integer four index form"""
  # a direction without a lattice: its Cartesian coordinates to the nearest small integers
  if not v.isCrystalDirection:
    m = v.data.reshape(-1, 3)
    style = "xyz"
  else:
    v._needsCrystal()
    style = v.dispStyle if style is None else style
    if style == "xyz":
      style = "hkl"
    m = getattr(v, style).reshape(-1, len(style))
  if style == "hkil":
    m = m[:, [0, 1, 3]]
  # every row scaled by n = 1 .. maxHKL, the first n of the least relative rounding error kept
  mMax = np.max(np.abs(m), axis=1, keepdims=True)
  unit = m / np.where(mMax == 0, 1, mMax)
  scaled = unit[:, None, :] * np.arange(1, maxHKL + 1)[:, None]
  with np.errstate(invalid="ignore"):
    err = np.round(np.sum((scaled - np.round(scaled)) ** 2, axis=-1) / np.sum(scaled ** 2, axis=-1) * 1e7) * 1e-7
  best = np.argmin(err, axis=1) + 1
  # adding zero turns the negative zero of a rounded small component into zero
  out = np.where(mMax == 0, m, np.round(unit * best[:, None]) + 0.0)
  if style == "UVTW":
    g = np.gcd.reduce(np.abs(np.nan_to_num(out)).astype(int), axis=1)
    out = out / np.maximum(g, 1)[:, None]
  if style == "xyz":
    return v._new(out.reshape(v.shape + (3,)).astype(float))
  result = v.copy()
  if style == "hkil":
    out = np.concatenate([out[:, :2], -out[:, :1] - out[:, 1:2], out[:, 2:]], axis=1)
  setattr(result, style, out.reshape(v.shape + (out.shape[-1],)))
  result.dispStyle = v.dispStyle
  return result


# ------------------------------------------------------------------ display

BAR = "̅"


def indexString(idx, utf8):
  """indices as text: integers barred when negative under UTF-8, others at two decimals"""
  parts = []
  for x in idx:
    isInt = abs(x - round(x)) < 1e-6
    r = str(int(round(x))) if isInt else f"{x:.2f}"
    if utf8 and isInt and x < -0.5:
      r = "".join(c + BAR for c in r[1:])
    parts.append(r)
  sep = " " if any(abs(x) > 9 or abs(x - round(x)) > 1e-3 for x in idx) else ""
  return sep.join(parts)


def brackets(style, family):
  """(hkl) and [uvw] for one plane or direction, {hkl} and <uvw> for the family a group makes"""
  if style in ("uvw", "UVTW"):
    return ("<", ">") if family else ("[", "]")
  if style in ("hkl", "hkil"):
    return ("{", "}") if family else ("(", ")")
  return ("", "")


def indexChar(v, style=None, utf8=None, family=None):
  """the family brackets {hkl}, <uvw> for a direction in a frame with symmetry, the plain ones
  (hkl), [uvw] when family is False, as a label on a plot names one direction"""
  # char(m, 'LaTex'): the indices as mathematics, a bar over the negative ones
  latex = isinstance(style, str) and style.lower() == "latex"
  utf8 = getMTEXpref("UTF8Output") if utf8 is None else utf8
  style = v.dispStyle if style is None or latex else style
  if family is None:
    family = v.frame is not None and v.frame.numSym() > 1
  left, right = brackets(style, family)
  coord = v.data if style == "xyz" else getattr(v, style)
  coord = np.where(np.abs(coord) < 1e-8, 0.0, coord).reshape(-1, coord.shape[-1])
  if latex:
    tex = {"{": r"\{", "}": r"\}", "<": r"\langle ", ">": r"\rangle"}
    return "".join("$" + tex.get(left, left) + _latexIndices(row) + tex.get(right, right) + "$" for row in coord)
  return "".join(left + indexString(row, utf8) + right for row in coord)


def _latexIndices(idx):
  parts = []
  for x in idx:
    isInt = abs(x - round(x)) < 1e-6
    r = str(int(round(x))) if isInt else f"{x:.2f}"
    parts.append(r"\bar{" + r[1:] + "}" if isInt and x < -0.5 else r)
  return ("\\," if any(abs(x) > 9 for x in idx) else "").join(parts)


__all__ = ["Miller"]
