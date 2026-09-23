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
from math import gcd

import numpy as np

from .frames import crystalFrame
from .session import getMTEXpref
from .vector3d import vector3d

STYLES = ("hkl", "hkil", "uvw", "UVTW", "xyz")
# MATLAB reads the style names without regard to case, 'HKIL' for 'hkil'
_STYLE = {s.lower(): s for s in STYLES}


def Miller(*args, antipodal=False, dispStyle=None):
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
# direction stands for itself or for its whole family
def _parse(s, cs):
  m = re.match(r"^\s*([(\[{<⟨])\s*([-\d,\s]+)\s*([)\]}>⟩])\s*$", s)
  if m is None:
    raise ValueError(f"cannot read '{s}' as Miller indices, write (hkl), {{hkl}}, [uvw] or <uvw>")
  open_, body = m.group(1), m.group(2)
  idx = [float(x) for x in body.split(",")] if "," in body else [float(x) for x in re.findall(r"-?\d", body)]
  if len(idx) not in (3, 4):
    raise ValueError(f"'{s}' does not hold three or four indices")
  direct = open_ in ("[", "<", "⟨")
  single = open_ in ("(", "[")
  style = ("UVTW" if len(idx) == 4 else "uvw") if direct else ("hkil" if len(idx) == 4 else "hkl")
  return Miller(*idx, cs, style), single


# ------------------------------------------------------------------ rounding

# the smallest multiple whose indices are close to integers, up to maxHKL; UVTW is reduced
# by its gcd, the smallest integer four index form
def roundIndices(v, style=None, maxHKL=12):
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
  out = np.empty_like(m)
  for r, row in enumerate(m):
    mMax = np.max(np.abs(row))
    if mMax == 0:
      out[r] = row
      continue
    best, bestErr = 1, np.inf
    for n in range(1, maxHKL + 1):
      mNew = row / mMax * n
      err = np.round(np.sum((mNew - np.round(mNew)) ** 2) / np.sum(mNew ** 2) * 1e7) * 1e-7
      if err < bestErr:
        best, bestErr = n, err
    # adding zero turns the negative zero of a rounded small component into zero
    rounded = np.round(row / mMax * best) + 0.0
    if style == "UVTW":
      ints = [int(x) for x in rounded]
      g = max(gcd(gcd(gcd(abs(ints[0]), abs(ints[1])), abs(ints[2])), abs(ints[3])), 1)
      rounded = rounded / g
    out[r] = rounded
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


# indices as text: integers barred when negative under UTF-8, others at two decimals
def indexString(idx, utf8):
  parts = []
  for x in idx:
    isInt = abs(x - round(x)) < 1e-6
    r = str(int(round(x))) if isInt else f"{x:.2f}"
    if utf8 and isInt and x < -0.5:
      r = "".join(c + BAR for c in r[1:])
    parts.append(r)
  sep = " " if any(abs(x) > 9 or abs(x - round(x)) > 1e-3 for x in idx) else ""
  return sep.join(parts)


# (hkl) and [uvw] for one plane or direction, {hkl} and <uvw> for the family a group makes
def brackets(style, family):
  if style in ("uvw", "UVTW"):
    return ("<", ">") if family else ("[", "]")
  if style in ("hkl", "hkil"):
    return ("{", "}") if family else ("(", ")")
  return ("", "")


# the family brackets {hkl}, <uvw> for a direction in a frame with symmetry, the plain ones
# (hkl), [uvw] when family is False, as a label on a plot names one direction
def indexChar(v, style=None, utf8=None, family=None):
  # char(m, 'LaTex'): the indices as mathematics, a bar over the negative ones
  latex = isinstance(style, str) and style.lower() == "latex"
  utf8 = getMTEXpref("UTF8Output") if utf8 is None else utf8
  style = v.dispStyle if style is None or latex else style
  family = v.frame.numSym() > 1 if family is None else family
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
