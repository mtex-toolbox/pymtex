# the correction of a pseudo symmetry (MATLAB EBSDAnalysis/cleanUpPseudoSym.m, Julia
# pseudosym.jl): an indexer alternating between the true orientation and a rotated
# alternative cuts one crystal into grains separated by the pseudo-symmetric misorientation,
# and their boundaries are ragged where a physical interface is not
#
#   ebsd, grainsM, numChanged = cleanUpPseudoSym(ebsd, grains, psSym, delta=2*degree, threshold=1.5)
from __future__ import annotations

import warnings

import numpy as np

from ..geometry.kernels import qmul
from ..geometry.orientation import orientation
from ..geometry.quaternion import rotation
from ..geometry.units import degree


# the operators an indexer could have applied: the cosets g psSym of the proper group G,
# without the group itself and without a coset twice
def pseudoOperators(psSym, tol=1e-4):
  G = psSym.CS.sym.properGroup().rotations().data.reshape(-1, 4)
  p = psSym.data.reshape(-1, 4)[0]
  kept = []
  for g in G:
    c = qmul(g, p)
    # the coset c G holds the identity exactly when c is in G
    if np.max(np.abs(qmul(c[None], G)[:, 0])) > np.cos(tol / 2):
      continue
    if any(np.max(np.abs(qmul(c[None], G) @ k)) > np.cos(tol / 2) for k in kept):
      continue
    kept.append(c)
  return np.array(kept).reshape(-1, 4)


# the map corrected for the pseudo symmetry psSym, a misorientation of one phase with itself:
# the boundaries of that phase within delta of it are grouped into connected components, the
# components longer than threshold times the diagonal of their bounding box and of more than
# four segments are taken for indexing artefacts, the grains across them are merged, and
# every measurement of a grain that absorbed another is moved to the operator that brings it
# nearest the merged grain's mean orientation. As calcGrains, it alters the map it is given,
# its rotations and grain ids, and grains gets the parentId of the merge; MATLAB's three
# outputs are returned
def cleanUpPseudoSym(ebsd, grains, psSym, delta=2 * degree, threshold=1.5):
  if not isinstance(psSym, orientation):
    raise ValueError("the pseudo symmetry must be an orientation: its crystal frame says which phase to correct")
  if psSym.CS is not psSym.SS:
    raise ValueError("the pseudo symmetry is a misorientation of one phase with itself (psSym.CS is psSym.SS)")
  pid = ebsd.CSList.phaseId(psSym.CS)
  if pid is None or pid <= 0:
    warnings.warn("the phase of the pseudo symmetry is not in the map", stacklevel=2)
    return ebsd, grains, 0
  gB = grains.boundary[psSym.CS, psSym.CS]
  if len(gB):
    gB = gB[np.asarray(gB.misorientation.angle(psSym)).reshape(-1) < delta]
  # the tortuosity of every component: its length against the diagonal of its bounding box
  cid = gB.componentId
  mid = gB.midPoint.data.reshape(-1, 3)
  n = int(cid.max()) if len(cid) else 0
  lo, hi = np.full((n + 1, 2), np.inf), np.full((n + 1, 2), -np.inf)
  np.minimum.at(lo, cid, mid[:, :2])
  np.maximum.at(hi, cid, mid[:, :2])
  length = np.bincount(cid, weights=gB.segLength, minlength=n + 1)
  with np.errstate(divide="ignore", invalid="ignore"):
    tortuosity = (length / np.linalg.norm(hi - lo, axis=1))[cid]
  artefact = (tortuosity > threshold) & (gB.componentSize > 4)
  grainsM = grains.merge(gB[np.flatnonzero(artefact)])
  parent = np.asarray(grains.parentId).reshape(-1)
  gid = ebsd.grainId.reshape(-1)
  has = gid > 0
  gid[has] = parent[grains.id2ind(gid[has])]
  ops = pseudoOperators(psSym)
  if len(ops) == 0:
    warnings.warn("the pseudo symmetry is a true symmetry of the crystal, there is nothing to correct", stacklevel=2)
    return ebsd, grainsM, 0
  # the measurements of the phase in grains that absorbed another
  members = np.bincount(parent, minlength=grainsM.size + 1)
  todo = np.flatnonzero(has & (ebsd.phaseId.reshape(-1) == pid) & (members[np.maximum(gid, 0)] > 1))
  if len(todo) == 0:
    return ebsd, grainsM, 0
  q = ebsd.rotations.data.reshape(-1, 4)
  mean = orientation(rotation(grainsM.meanRotation.data.reshape(-1, 4)[grainsM.id2ind(gid[todo])]), psSym.CS, ebsd.frame)
  cand = np.concatenate([[[1.0, 0, 0, 0]], ops])
  d = np.stack([np.asarray(orientation(rotation(qmul(q[todo], c[None])), psSym.CS, ebsd.frame).angle(mean)).reshape(-1)
                for c in cand], axis=1)
  best = np.argmin(d, axis=1)
  moved = best > 0
  q[todo[moved]] = qmul(q[todo[moved]], cand[best[moved]])
  return ebsd, grainsM, int(moved.sum())


__all__ = ["cleanUpPseudoSym"]
