# changing the frame data is expressed in, without moving the data (MATLAB ADR 0008 rule 8;
# orientation/align, byScreenAlignment, vector3d/transformReferenceFrame,
# orientation/transformReferenceFrame)
#
# The relation between two frames comes three ways, most certain first: an orientation
# given outright, the assertion that the frames were drawn alike (byScreenAlignment), or
# the two bases (align). A frame change composes that rotation onto the data and relabels
# the frame; rotating is a different verb.
from __future__ import annotations

import numpy as np

from .frames import crystalFrame
from .quaternion import qmul, qrot, rotation
from .symreduce import _conj
from .vector3d import vector3d


# the orientation two frames stand in by their axes: a point at v in frA sits at
# align(frA, frB) * v in frB; the transition has to be a rotation
def align(frA, frB, tolerance=1e-1):
  from .orientation import orientation
  M = frA.transformationMatrix(frB)
  dev = np.linalg.norm(np.eye(3) - M @ M.T)
  if dev > tolerance:
    raise ValueError(f"the frames {frA.char()} and {frB.char()} are not related by a rotation: the transition deviates from one by {dev:.3g}")
  return orientation(rotation.byMatrix(M), frA.stripSym(), frB.stripSym())


# the orientation implied by two frames rendering alike: a point at v renders at
# inv(rot) * v in screen coordinates, so vB = rotB * inv(rotA) * vA
def byScreenAlignment(frA, frB):
  from .orientation import orientation
  if frA.how2plot is None or frB.how2plot is None:
    raise ValueError("both frames need a plotting convention of their own; an empty one follows the session default")
  return orientation(frB.how2plot.rot * frA.how2plot.rot.inv(), frA.stripSym(), frB.stripSym())


# the rotation taking coordinates in `from` to coordinates in `to`
def frameTransition(fromFr, toFr, rule=None, tolerance=1e-1):
  if rule is None or rule == "byAxes":
    return align(fromFr, toFr, tolerance)
  if rule == "byScreenAlignment":
    return byScreenAlignment(fromFr, toFr)
  if hasattr(rule, "frameA"):
    return rule
  raise ValueError("the frame transition rule is 'byAxes', 'byScreenAlignment' or an orientation")


# a direction restated in another frame, keeping its meaning
def transformVector(v, to, rule=None, tolerance=1e-1):
  if v.frame is to:
    return v
  t = frameTransition(v.frame, to, rule, tolerance)
  out = v._new(qrot(t.data, v.data), _frame=to, dispStyle="xyz")
  if isinstance(to, crystalFrame):
    style = v.dispStyle
    out.dispStyle = ("hkil" if to.lattice.isTriHex else "hkl") if style in ("xyz", "hkl", "hkil") else ("UVTW" if to.lattice.isTriHex else "uvw")
  return out


# an orientation restated with another crystal frame: o' = o R^-1 for the transition R from
# the old crystal frame to the new one, so that o'(x_new) = o(x_old); both sides of a
# misorientation when two frames are given
def transformOrientation(o, to, toB=None, rule=None, tolerance=1e-1):
  if toB is not None:
    return transformOrientation(transformOrientation(o, to, rule=rule, tolerance=tolerance).inv(), toB, rule=rule, tolerance=tolerance).inv()
  if o.frameA is to:
    return o
  t = frameTransition(o.frameA, to, rule, tolerance)
  return o._new(qmul(o.data, _conj(t.data)), frameA=to)


__all__ = ["align", "byScreenAlignment"]
