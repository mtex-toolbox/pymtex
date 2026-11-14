# the kinds of the notIndexed parts the closing leaves open (MATLAB's kindOfParts in
# spatialDecompositionGrid.m)
#
#   kindP, parts, lab = kindOfParts(closed, present, outside, pixelOf, connects, override, stencil, A)
#   labelCells(mask, stencil)       the connected components of a raster mask over the stencil
#   kindCodes(kinds)                the codes of kinds given by name
#
# The closing with alpha fills every notIndexed region narrower than 2 alpha cells; what it leaves
# open are the thick parts, each a pore, a pocket, a crack or a void, and calcGrains closes each
# kind with a radius of its own. A part is a crack when its Feret diameter squared is at least five
# times its area and at least half of its boundary cells touch grains rather than notIndexed cells,
# a void when it touches the raster outside the convex hull of the indexed cells, a pore when at
# least 85 percent of its rim is held by cells the criterion connects, a pocket otherwise; crack
# before void before pore before pocket. The closing cuts a crack where it narrows, so what lies
# between two parts of a crack is crack: the parts and the connected pieces of filled cells are
# the nodes of a graph with a contact as an edge, and pruning the leaves that are no crack part
# leaves the nodes between crack parts.
from __future__ import annotations

import numpy as np

from ..geometry.hull import convexHull

KINDS = np.array(["pore", "pocket", "crack", "void", "speck", "indexed"])
PORE, POCKET, CRACK, VOID, SPECK, INDEXED = range(1, 7)


def labelCells(mask, stencil):
  """the connected components of a raster mask over the stencil adjacency, numbered from 1 and 0
  off the mask, and their number"""
  from scipy import ndimage
  structure = np.zeros((3, 3), dtype=bool)
  structure[1 + stencil[:, 0], 1 + stencil[:, 1]] = True
  structure[1, 1] = True
  return ndimage.label(mask, structure=structure)


def kindCodes(kinds):
  """the codes of kinds given per pixel by name, 1 pore, 2 pocket, 3 crack, 4 void and 0 for the
  rest, or as codes already; None stays None"""
  if kinds is None or np.issubdtype(np.asarray(kinds).dtype, np.number):
    return None if kinds is None else np.asarray(kinds).reshape(-1).astype(np.int64)
  m = np.asarray(kinds).reshape(-1)[:, None] == KINDS[None, :4]
  return (m.argmax(axis=1) + 1) * m.any(axis=1)


# the (part, cell) pairs of the contacts once each, in the order of the part and then of the cell
# as MATLAB counts the raster, down the columns, so the first cell of a rim piece is the same
def _contacts(px, nb, ok, n1, n2):
  N = n1 * n2
  p, c = np.broadcast_to(px[:, None], nb.shape)[ok], nb[ok]
  u = np.unique(p * N + (c % n2) * n1 + c // n2)
  part, key = u // N, u % N
  return part, (key % n1) * n2 + key // n1


def kindOfParts(closed, present, outside, pixelOf, connects, override, stencil, A):
  """the kind of every raster cell of the notIndexed parts the closing leaves open, 1 pore, 2
  pocket, 3 crack, 4 void and 0 elsewhere, the filled cells between two parts of a crack marked
  crack, the table of the parts with their measures and the part of every cell; connects(i, j) is
  the criterion on pairs of pixels, override a code per pixel replacing the computed kind of its part"""
  from .segment import components

  # the criterion on pairs of raster cells, none to judge when there are no pairs
  def same(i, j):
    return connects(i, j) if len(i) else np.zeros(0, dtype=bool)

  n1, n2 = closed.shape
  N = n1 * n2
  so = (stencil[:, 0] * n2 + stencil[:, 1]).astype(np.int64)
  pr, out = present.reshape(-1), outside.reshape(-1)
  lab, nP = labelCells(~closed & ~outside, stencil)
  lab = lab.reshape(-1).astype(np.int64)
  area = np.bincount(lab, minlength=nP + 1)[1:]
  # the cells of a part next to one that is not open carry its border contact, Feret diameter
  # and contacts
  x = np.flatnonzero(lab)
  x = x[(lab[x[:, None] + so] == 0).any(axis=1)]
  px = lab[x] - 1
  nb = x[:, None] + so
  border = np.bincount(px, out[nb].any(axis=1), minlength=nP) > 0
  # the cells around every part that are not open, one contact per part and cell: the notIndexed
  # ones, the outside and the filled cells, are its gap, the indexed its rim
  part, cell = _contacts(px, nb, lab[nb] == 0, n1, n2)
  isRim = pr[cell]
  gap = np.bincount(part[~isRim], minlength=nP)
  part, cell = part[isRim], cell[isRim]
  # the Feret diameter of a part from the hull of its ideal cell positions
  xy = np.column_stack(np.divmod(x, n2)).astype(float) @ np.asarray(A, dtype=float).T
  order = np.argsort(px, kind="stable")
  starts = np.searchsorted(px[order], np.arange(nP + 1))
  feret = np.zeros(nP)
  for p in np.flatnonzero(area >= 3):
    q = xy[order[starts[p]:starts[p + 1]]]
    q = q[convexHull(q)]
    feret[p] = np.sqrt(np.sum((q[:, None] - q[None]) ** 2, axis=2).max())
  elong = feret ** 2 / (area * abs(np.linalg.det(A)))
  # adjacent rim cells of one part joined where the criterion connects them
  nC = len(part)
  key = part * N + cell
  o = np.argsort(key)
  a, b = [], []
  for s in so:
    k2 = key + s
    j = o[np.minimum(np.searchsorted(key[o], k2), nC - 1)] if nC else np.zeros(0, dtype=np.int64)
    tf = (key[j] == k2) & (np.arange(nC) < j)
    a.append(np.flatnonzero(tf))
    b.append(j[tf])
  a, b = np.concatenate(a), np.concatenate(b)
  pix = pixelOf.reshape(-1)
  tf = same(pix[cell[a]], pix[cell[b]])
  cl = components(nC, a[tf], b[tf])
  # the rim pieces of one part joined once more when their first cells connect, except on the
  # border, where a part is a void whatever its rim
  _, first, piece = np.unique(cl, return_index=True, return_inverse=True)
  piece = piece.reshape(-1)
  pp = part[first]
  po = np.argsort(pp, kind="stable")
  cnt = np.bincount(pp, minlength=nP)
  pairs = [np.column_stack(np.triu_indices(c, 1)) + s for c, s in zip(np.where(border, 0, cnt), np.cumsum(cnt) - cnt) if c > 1]
  pairs = po[np.vstack(pairs)] if pairs else np.zeros((0, 2), dtype=np.int64)
  tf = same(pix[cell[first[pairs[:, 0]]]], pix[cell[first[pairs[:, 1]]]])
  cl = components(len(first), pairs[tf, 0], pairs[tf, 1])[piece]
  rim = np.bincount(part, minlength=nP)
  share = np.zeros(nP)
  np.maximum.at(share, part, np.bincount(cl)[cl])
  share /= np.maximum(rim, 1)
  kind = np.full(nP, POCKET)
  kind[share >= 0.85] = PORE
  kind[border] = VOID
  kind[(elong >= 5) & (gap <= rim)] = CRACK
  # a part takes the mode of the kinds given for its pixels, the smallest on a tie
  if override is not None:
    y = np.flatnonzero(lab)
    k = np.where(pix[y] >= 0, override[np.maximum(pix[y], 0)], 0)
    k = np.where((k >= PORE) & (k <= VOID), k, 0)
    votes = np.zeros((nP, 5), dtype=np.int64)
    np.add.at(votes, (lab[y[k > 0]] - 1, k[k > 0]), 1)
    given = votes[:, 1:].any(axis=1)
    kind[given] = votes[given, 1:].argmax(axis=1) + 1
  # what lies between two parts of a crack is part of it: the leaves of the contact graph of the
  # parts and the filled pieces that are no crack part pruned, the parts and pieces left in a
  # component with a crack part are between
  kindP = np.zeros(N, dtype=np.int8)
  if (kind == CRACK).any():
    thin = np.flatnonzero(~pr & ~out & closed.reshape(-1))
    ct, nT = labelCells((~present & ~outside & closed), stencil)
    ct = ct.reshape(-1).astype(np.int64)
    tn = thin[:, None] + so
    ok = lab[tn] > 0
    e = np.unique((nP + ct[thin] - 1)[:, None].repeat(len(so), axis=1)[ok] * (nP + nT) + lab[tn[ok]] - 1)
    ea, eb = e // (nP + nT), e % (nP + nT)
    isCrack = np.concatenate([kind == CRACK, np.zeros(nT, dtype=bool)])
    keep = np.ones(nP + nT, dtype=bool)
    while True:
      kk = keep[ea] & keep[eb]
      deg = np.bincount(ea[kk], minlength=nP + nT) + np.bincount(eb[kk], minlength=nP + nT)
      leaf = keep & ~isCrack & (deg <= 1)
      if not leaf.any():
        break
      keep[leaf] = False
    comp = components(nP + nT, ea[kk], eb[kk])
    between = keep & ~isCrack & np.isin(comp, comp[isCrack])
    kind[between[:nP]] = CRACK
    kindP[thin[between[nP + ct[thin] - 1]]] = CRACK
  kindP[lab > 0] = kind[lab[lab > 0] - 1]
  parts = dict(id=np.arange(1, nP + 1), kind=KINDS[kind - 1], area=area, feret=feret, elong=elong, border=border,
               gap=gap, rim=rim, share=share)
  return kindP.reshape(n1, n2), parts, lab.reshape(n1, n2)


__all__ = []
