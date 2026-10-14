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

from .kernels import qconj
from .frames import crystalFrame
from .quaternion import qmul, qrot, rotation
from .vector3d import vector3d


def align(frA, frB, tol=1e-1):
  """the orientation two frames stand in by their axes, MATLAB's orientation.align

    ori = align(cs1, cs2)              # the rotation from the axes of cs1 to those of cs2
    ori = align(cs1, cs2, tol=1e-3)    # a transition nearer a rotation demanded

  A point sitting at v in frA sits at ori * v in frB. Unlike orientation.byScreenAlignment,
  which infers the rotation from how the two frames are drawn, this reads the two bases: it is
  the rotation of referenceFrame.transformationMatrix, which vector3d.transformReferenceFrame
  applies to move data from one frame into the other. The orientation is between the frames
  without their groups (stripSym).

  Input
    frA - referenceFrame, the source
    frB - referenceFrame, the target
  Output
    ori - orientation from frA to frB
  Keywords
    tol - how far, as the norm of I - M M', the transition M may be from a rotation; a
        larger deviation is refused
  See also
    orientation.byScreenAlignment, referenceFrame.isCompatible, transformReferenceFrame"""
  from .orientation import orientation
  M = frA.transformationMatrix(frB)
  dev = np.linalg.norm(np.eye(3) - M @ M.T)
  if dev > tol:
    raise ValueError(f"the frames {frA.char()} and {frB.char()} are not related by a rotation: the transition deviates from one by {dev:.3g}")
  return orientation(rotation.byMatrix(M), frA.stripSym(), frB.stripSym())


def byScreenAlignment(frA, frB):
  """the orientation implied by two frames rendering alike: a point at v renders at
  inv(rot) * v in screen coordinates, so vB = rotB * inv(rotA) * vA"""
  from .orientation import orientation
  if frA.how2plot is None or frB.how2plot is None:
    raise ValueError("both frames need a plotting convention of their own; an empty one follows the session default")
  return orientation(frB.how2plot.rot * frA.how2plot.rot.inv(), frA.stripSym(), frB.stripSym())


def frameTransition(fromFr, toFr, rule=None, tol=1e-1):
  """the rotation taking coordinates in `from` to coordinates in `to`"""
  if rule is None or rule == "byAxes":
    return align(fromFr, toFr, tol)
  if rule == "byScreenAlignment":
    return byScreenAlignment(fromFr, toFr)
  from .orientation import orientation
  if isinstance(rule, orientation):
    return rule
  raise ValueError("the frame transition rule is 'byAxes', 'byScreenAlignment' or an orientation")


def transformVector(v, to, rule=None, tol=1e-1):
  """a direction restated in another frame, keeping its meaning"""
  if v.frame is to:
    return v
  t = frameTransition(v.frame, to, rule, tol)
  out = v._new(qrot(t.data, v.data), _frame=to, dispStyle="xyz")
  if isinstance(to, crystalFrame):
    style = v.dispStyle
    out.dispStyle = ("hkil" if to.lattice.isTriHex else "hkl") if style in ("xyz", "hkl", "hkil") else ("UVTW" if to.lattice.isTriHex else "uvw")
  return out


def transformOrientation(o, to, toB=None, rule=None, tol=1e-1):
  """an orientation restated with another crystal frame: o' = o R^-1 for the transition R from
  the old crystal frame to the new one, so that o'(x_new) = o(x_old); both sides of a
  misorientation when two frames are given"""
  if toB is not None:
    return transformOrientation(transformOrientation(o, to, rule=rule, tol=tol).inv(), toB, rule=rule, tol=tol).inv()
  if o.frameA is to:
    return o
  t = frameTransition(o.frameA, to, rule, tol)
  return o._new(qmul(o.data, qconj(t.data)), frameA=to)


__all__ = ["align", "byScreenAlignment"]
