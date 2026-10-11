# the faces of polyhedral grains turned consistently (MATLAB grain3d/orientFaces, ../master)
#
#   grains = orientFaces(grains)
#
# A face enters every grain it bounds as a half face; two half faces of one grain that share
# an edge are consistent when they run along it in opposite senses. Each half face is a pair
# of nodes, kept and turned, and every shared edge links the nodes that agree; the connected
# parts come in mirrored pairs, a patch of the surface and its turned copy, and of each pair
# the one enclosing a positive volume is the outward orientation. The corners of the faces
# are not changed: the signs of I_GF are, and the sides of the boundary follow them.
from __future__ import annotations

import warnings

import numpy as np


def orientFaces(grains):
  """the grains with their faces turned consistently, the normals pointing out of every grain:
  the signs of I_GF and the sides of the boundary change, the corners do not"""
  from scipy.sparse import coo_matrix, csr_matrix
  from scipy.sparse.csgraph import connected_components
  gB = grains.boundary
  C = grains.I_GF.tocoo()
  order = np.lexsort((C.row, C.col))
  gId, fId = C.row[order], C.col[order]
  nHF = len(gId)
  if nHF == 0:
    return grains
  # the edges of every half face in the sense of its face
  counts = gB.numCorners[fId]
  hf = np.repeat(np.arange(nHF), counts)
  k = np.arange(len(hf)) - np.repeat(np.cumsum(counts) - counts, counts)
  start = gB._ptr[fId][hf]
  eA = gB._idx[start + k]
  eB = gB._idx[start + (k + 1) % counts[hf]]
  eDir = np.where(eA > eB, -1, 1)
  nV = len(gB.allV)
  _, eId = np.unique(np.minimum(eA, eB) * nV + np.maximum(eA, eB), return_inverse=True)
  _, gE = np.unique(gId[hf] * (eId.max() + 1) + eId, return_inverse=True)
  pair = np.flatnonzero(np.bincount(gE)[gE] == 2)
  pair = pair[np.argsort(gE[pair], kind="stable")]
  e1, e2 = pair[0::2], pair[1::2]
  n1, n2 = hf[e1], hf[e2]
  w = -eDir[e1] * eDir[e2]
  src = np.concatenate([n1, n1 + nHF])
  dst = np.concatenate([n2 + nHF * (w < 0), n2 + nHF * (w > 0)])
  G = coo_matrix((np.ones(len(src)), (src, dst)), shape=(2 * nHF, 2 * nHF))
  _, comp = connected_components(G, directed=False)
  cPlus, cMinus = comp[:nHF], comp[nHF:]
  if np.any(cPlus == cMinus):
    warnings.warn(f"the grain boundary contains {int(np.sum(cPlus == cMinus))} faces that can not be oriented consistently")
  x = np.where(cPlus > cMinus, -1, 1)
  _, patch = np.unique(np.minimum(cPlus, cMinus), return_inverse=True)
  fV = _faceVolume(gB)
  patchVol = np.bincount(patch, weights=x * fV[fId])
  x = x * np.where(patchVol[patch] < 0, -1, 1)
  nG = np.bincount(fId, minlength=len(gB))
  bad = (nG == 2) & (np.bincount(fId, weights=x, minlength=len(gB)) != 0)
  if bad.any():
    warnings.warn(f"{int(bad.sum())} inner faces could not be oriented consistently; this happens for grains that completely "
                  "enclose another grain, whose normals have to be flipped by hand")
  I = csr_matrix((x.astype(float), (gId, fId)), shape=grains.I_GF.shape)
  new = np.zeros((len(gB), 2), dtype=np.int64)
  new[fId[x > 0], 0] = grains.id[gId[x > 0]]
  new[fId[x < 0], 1] = grains.id[gId[x < 0]]
  # a face whose sides only changed places is flipped; one that got other sides, as a face
  # given with the same sign for both grains, takes them with their phases and misorientation
  changed = np.any(new != gB.grainId, axis=1)
  swapped = changed & np.all(new == gB.grainId[:, ::-1], axis=1)
  gB = gB.flip(np.flatnonzero(swapped))
  other = np.flatnonzero(changed & ~swapped)
  if len(other):
    from ..geometry.kernels import qconj, qmul
    pos = np.zeros(int(max(grains.id.max(initial=0), new.max(initial=0))) + 1, dtype=np.int64)
    pos[grains.id] = np.arange(len(grains))
    g = new[other]
    gB.grainId[other] = g
    gB.ebsdId[other] = -1
    gB.phaseId[other] = np.where(g > 0, grains.phaseId[pos[np.maximum(g, 0)]], -1)
    q = grains.meanRotation.data.reshape(-1, 4)
    mori = gB.misrotation.data.reshape(-1, 4)
    both = np.all(g > 0, axis=1)
    mori[other] = np.nan
    mori[other[both]] = qmul(qconj(q[pos[g[both, 1]]]), q[pos[g[both, 0]]])
  return grains._new(np.arange(len(grains)), I, gB)


# six times the signed volume of the cone from the origin over every face
def _faceVolume(gB):
  T, face = gB._fan()
  a, b, c = T[:, 0], T[:, 1], T[:, 2]
  return gB._sumByFace(np.einsum("ij,ij->i", a, np.cross(b - a, c - a)), face)


__all__ = ["orientFaces"]
