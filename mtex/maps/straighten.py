# the shortest grain boundaries through the cells the alpha closing filled (MATLAB
# EBSD/private/straightenInFill)
#
#   c, seg = straightenInFill(c, seg)    the complex with its boundary vertices in the fill moved
#
# Through filled cells a grain boundary follows the bisectors between the measured pixels around
# them, which bends it around the middle of a crack or a cluster of notIndexed pixels. Inside the
# filled cells the boundaries form a network: chains of segments between the junctions and the
# points where a boundary enters the fill. It is replaced by the shortest network of the same
# chains that passes every measured, unfilled and exterior cell on the same side, as for an
# isotropic boundary energy: every chain is pulled taut, so it is straight between the cell corners
# it bends around, and every junction sits where the sum of the distances to the first bends of its
# chains is least, which makes its chains meet at 120 degrees unless a cell is in the way. A region
# of filled cells at least half of which lie in a gap one cell wide between measured pixels of two
# grains is a boundary the measurement missed, as where the beam spans two grains; it keeps its
# bisectors. Every measured pixel stays in its grain, no boundary crosses another one, and the
# outer boundary never leaves the closing.
from __future__ import annotations

import numpy as np

from .decomposition import cellComplex
from .kernels import pixelKernel
from .segment import components, segmentation

# the fixed shuffle of the points of a chain, by the golden ratio of their number from 1
GOLD = 0.6180339887498949


def straightenInFill(c, seg):
  """the complex with the boundary segments that enter the filled cells split there and the
  boundaries inside them the shortest network of their chains, and the segmentation with the
  split faces; the complex keeps the vertices before they moved as V0"""
  inFill = None if c.inFill is None else missedBoundaries(c, seg.grainId)
  if inFill is None:
    return c, seg
  tol, dxy = c.dxy / 100, c.dxy
  # the boundary: the faces between two grains or against the outside, the segmentation's outer ones
  V, F, isB = c.V.astype(float), c.F, seg.isOuter.copy()

  # the fill test at every boundary vertex once
  used = np.zeros(len(V), dtype=bool)
  used[F[isB]] = True
  dV = np.full(len(V), np.nan)
  dV[used] = inFill(V[used])
  # a boundary segment gets a vertex where it enters the filled cells, by bisection, twice, so the
  # pieces are cut again; src is the face every face was cut from
  loop = pixelKernel("cut") if inFill.rigid else None
  src, look = np.arange(len(F)), isB.copy()
  for _ in range(2):
    L = np.flatnonzero(look)
    d0, d1 = dV[F[L, 0]], dV[F[L, 1]]
    isC = (np.maximum(d0, d1) > tol) & (np.minimum(d0, d1) < -tol)
    k = ~isC & (np.maximum(d0, d1) > tol)
    c2, a2, b2 = fillEntry(V, F, L[k], d0[k], inFill, tol, dxy)
    cut = np.concatenate([L[isC], c2])
    a, b = np.vstack([V[F[L[isC], 0]], a2]), np.vstack([V[F[L[isC], 1]], b2])
    if loop is not None:
      t = np.empty((len(cut), 1))
      loop(a, b, *inFill.lattice, t[:, 0])
    else:
      lo, hi = np.zeros((len(cut), 1)), np.ones((len(cut), 1))
      aIn = inFill(a) > 0
      for _ in range(16):
        mid = (lo + hi) / 2
        isA = (inFill(a + mid * (b - a)) > 0) == aIn
        lo[isA], hi[~isA] = mid[isA], mid[~isA]
      t = (lo + hi) / 2
    nV, nF = len(V), len(F)
    new = nV + np.arange(len(cut))
    V = np.vstack([V, a + t * (b - a)])
    dV = np.concatenate([dV, inFill(V[nV:])])
    # the cut faces end at the new vertices, the new faces continue to their old ends
    F = np.vstack([F, np.column_stack([new, F[cut, 1]])])
    F[cut, 1] = new
    src, isB = np.concatenate([src, src[cut]]), np.concatenate([isB, np.ones(len(cut), dtype=bool)])
    look = np.zeros(len(F), dtype=bool)
    look[cut] = look[nF:] = True

  V, F, src, V0 = shortestNetwork(V, F, src, np.flatnonzero(isB), dV > tol, fillGeometry(inFill), dxy)
  return (cellComplex(V.astype(c.V.dtype, copy=False), F.astype(c.F.dtype, copy=False), c.cells[src], c.sites, c.A, c.plane, c.shape,
                      V0.astype(c.V.dtype, copy=False)),
          segmentation(seg.grainId, seg.nGrains, seg.connect[src], seg.isInner[src], seg.isOuter[src]))


class chainPoints:
  """the points of the polylines: their lattice coordinates X, the point before and after each, nb
  (-1 for none), whether they are to be looked at, dirty, the junction a chain end belongs to, jOf
  (-1 for none), and the number n in use; the arrays grow by doubling"""

  def __init__(self, X, nb, dirty, jOf):
    self.X, self.nb, self.dirty, self.jOf, self.n = X, nb, dirty, jOf, len(X)

  def add(self, k):
    """the indices of k new points"""
    c = self.n + np.arange(k)
    self.n += k
    grow = 2 * self.n - len(self.X) if self.n > len(self.X) else 0
    self.X = np.vstack([self.X, np.zeros((grow, 2))])
    self.nb = np.vstack([self.nb, np.full((grow, 2), -1)])
    self.dirty = np.concatenate([self.dirty, np.zeros(grow, dtype=bool)])
    self.jOf = np.concatenate([self.jOf, np.full(grow, -1)])
    return c


def shortestNetwork(V, F, src, segId, isIn, G, dxy):
  """the boundary segments segId of F pulled taut inside the fill and their junctions moved to the
  point of least total length, isIn whether a vertex lies inside the fill; returns the vertices,
  the faces with the vertices a chain needs beyond its inner ones, their source faces, and the
  vertices before they moved"""
  # the boundary segments with an end inside the fill; a vertex there with two of them is an inner
  # vertex of a chain, every other one a node
  e = F[segId]
  nV = len(V)
  isFree = np.zeros(nV, dtype=bool)
  isFree[e.reshape(-1)] = isIn[e.reshape(-1)]
  keep = isFree[e].any(axis=1)
  e, segId = e[keep], segId[keep]
  m = len(e)
  deg = np.bincount(e.reshape(-1), minlength=nV)
  isInner = isFree & (deg == 2)

  # a cycle of inner vertices gets a node, its first vertex
  ii = isInner[e].all(axis=1)
  lab = components(nV, e[ii, 0], e[ii, 1])
  atNode = np.zeros(nV, dtype=bool)
  atNode[lab[e[~ii]]] = True
  cyc = np.flatnonzero(isInner & ~atNode[lab])
  isInner[cyc[np.unique(lab[cyc], return_index=True)[1]]] = False
  isNode = (deg > 0) & ~isInner

  # lattice coordinates, the nodes on the fill edge exactly on the cell corner next to them
  U = G.toU(V)
  for k in G.C:
    r = np.round(U - k) + k
    s = np.all(np.abs(U - r) < 0.1, axis=1) & ~isFree
    U[s] = r[s]

  # the two segments of every inner vertex
  ev = np.concatenate([e[:, 0], e[:, 1]])
  o = np.argsort(ev, kind="stable")
  vv, eid = ev[o], np.tile(np.arange(m), 2)[o]
  ve = np.full((nV, 2), -1)
  s = np.flatnonzero(isInner[vv])
  ve[vv[s], np.arange(len(s)) % 2] = eid[s]

  # walk from every node along every segment to the next node
  wc, ws = np.nonzero(isNode[e].T)
  cur, edg = e[ws, 1 - wc], ws.copy()
  act = np.arange(len(ws))
  rw, rv = [act[:0]], [act[:0]]
  while len(act):
    act = act[~isNode[cur[act]]]
    rw.append(act)
    rv.append(cur[act])
    t = ve[cur[act]]
    ne = np.where(t[:, 0] == edg[act], t[:, 1], t[:, 0])
    cur[act] = e[ne, 0] + e[ne, 1] - cur[act]
    edg[act] = ne
  rw, rv = np.concatenate(rw), np.concatenate(rv)

  # every chain once, as walked from the smaller of its two end segments: start and end node, last
  # segment, and the inner vertices iv in order with their chain ic
  hs = ws + m * wc
  he = edg + m * (e[edg, 1] == cur)
  isC = hs < he
  ch = np.cumsum(isC) - 1
  C = int(isC.sum())
  cS, cE, cLast = e.T.reshape(-1)[hs[isC]], cur[isC], segId[edg[isC]]
  s = np.flatnonzero(isC[rw])
  o = np.argsort(ch[rw[s]], kind="stable")
  ic, iv = ch[rw[s]][o], rv[s[o]]
  nI = np.bincount(ic, minlength=C)

  # the points of the polylines: the inner vertices, then a start and an end point for every chain
  sS, sE = nV + np.arange(C), nV + C + np.arange(C)
  nb = np.full((nV + 2 * C, 2), -1)
  first, last = np.diff(ic, prepend=-1) > 0, np.diff(ic, append=C) > 0
  nb[iv, 0] = np.roll(iv, 1)
  nb[iv[first], 0] = sS[ic[first]]
  nb[iv, 1] = np.roll(iv, -1)
  nb[iv[last], 1] = sE[ic[last]]
  nb[sS, 1] = sE
  nb[sS[ic[first]], 1] = iv[first]
  nb[sE, 0] = sS
  nb[sE[ic[last]], 0] = iv[last]
  dirty = np.zeros(nV + 2 * C, dtype=bool)
  dirty[iv] = True

  # the junctions J, the free nodes, and the chain ends sJ at junction jS
  J = np.flatnonzero(isNode & isFree)
  jOf = np.full(nV, -1)
  jOf[J] = np.arange(len(J))
  a, b = jOf[cS] >= 0, jOf[cE] >= 0
  sJ, jS = np.concatenate([sS[a], sE[b]]), np.concatenate([jOf[cS[a]], jOf[cE[b]]])
  isStart = np.arange(len(sJ)) < a.sum()
  jOf = np.full(nV + 2 * C, -1)
  jOf[sJ] = jS
  P = chainPoints(np.vstack([U, U[cS], U[cE]]), nb, dirty, jOf)
  Xj, nJ = U[J], len(J)

  tighten(P, G)
  Ltot = [np.inf]
  for it in range(1, 12):
    if nJ == 0:
      break
    f = P.nb[sJ].max(axis=1)
    ln = np.linalg.norm((P.X[sJ] - P.X[f]) @ G.M, axis=1)
    jf = P.jOf[f]
    j = jf >= 0
    a = np.flatnonzero(P.nb[:P.n, 1] >= 0)
    Ltot.append(np.linalg.norm((P.X[a] - P.X[P.nb[a, 1]]) @ G.M, axis=1).sum())
    isLast = it > 10 or Ltot[-2] - Ltot[-1] < 5e-4 * Ltot[-1]
    if not isLast:
      # the junctions with the least sum of squared lengths, each weighted by the inverse of the present length
      from scipy.sparse import coo_matrix, identity
      from scipy.sparse.linalg import spsolve
      w = 1 / np.sqrt(ln ** 2 + (1e-3 * dxy) ** 2)
      A = coo_matrix((np.concatenate([w, -w[j]]), (np.concatenate([jS, jS[j]]), np.concatenate([jS, jf[j]]))), shape=(nJ, nJ)).tocsc()
      rhs = np.column_stack([np.bincount(jS[~j], weights=w[~j] * P.X[f[~j], k], minlength=nJ) for k in (0, 1)])
      d = spsolve(A + 1e-9 * identity(nJ, format="csc"), rhs + 1e-9 * Xj).reshape(-1, 2) - Xj
      d[np.linalg.norm(d, axis=1) < 1e-2] = 0
    else:
      # at last junctions a tenth of a pixel apart become one point
      a = j & (ln < 0.1 * dxy)
      lab = components(nJ, jS[a], jf[a])
      d = np.column_stack([np.bincount(lab, Xj[:, k], minlength=nJ) for k in (0, 1)]) / np.maximum(np.bincount(lab, minlength=nJ), 1)[:, None]
      d = d[lab] - Xj

    # as far as the fill reaches, along the side of a cell where that blocks
    dn = clipStep(Xj, d, G)[:, None] * d
    k = np.flatnonzero((np.linalg.norm(dn, axis=1) < 1e-6) & (np.linalg.norm(d, axis=1) > 1e-6))
    for sd in G.side:
      da = (d[k] @ sd)[:, None] * sd / (sd @ sd)
      da = clipStep(Xj[k], da, G)[:, None] * da
      s = np.linalg.norm(da, axis=1) > np.linalg.norm(dn[k], axis=1)
      dn[k[s]] = da[s]
    dn[np.linalg.norm(dn, axis=1) < 1e-9] = 0
    Y = Xj + dn

    # the old place stays as a point of every chain whose end sweeps over a cell that is no fill
    mvd = dn[jS].any(axis=1)
    for st in (1, 0):
      a = np.flatnonzero(mvd & (isStart == st))
      f = P.nb[sJ[a]].max(axis=1)
      jf = P.jOf[f]
      Yf = np.where((jf >= 0)[:, None], Y[np.maximum(jf, 0)], P.X[f])
      Q = np.stack([Y[jS[a]], Xj[jS[a]], P.X[f], Yf])
      s = anyObst(Q.min(axis=0), Q.max(axis=0), G)
      a, f = a[s], f[s]
      c = P.add(len(a))
      P.X[c] = Xj[jS[a]]
      P.dirty[c] = True
      P.nb[c] = np.column_stack([sJ[a], f] if st else [f, sJ[a]])
      P.nb[sJ[a], st] = c
      P.nb[f, 1 - st] = c

    # the first point of every chain of a moved junction is looked at again
    f = P.nb[sJ[mvd]].max(axis=1)
    P.dirty[f[(P.nb[f] >= 0).all(axis=1)]] = True
    P.X[sJ] = Y[jS]
    Xj = Y
    tighten(P, G)
    if isLast:
      break

  # the bends ib of every chain in order, cb their chain
  cur = P.nb[sS, 1].copy()
  act = np.flatnonzero(cur != sE)
  cb, ib = [act[:0]], [act[:0]]
  while len(act):
    cb.append(act)
    ib.append(cur[act])
    cur[act] = P.nb[cur[act], 1]
    act = act[cur[act] != sE[act]]
  cb = np.concatenate(cb)
  o = np.argsort(cb, kind="stable")
  cb, ib = cb[o], np.concatenate(ib)[o]
  nB = np.bincount(cb, minlength=C)

  # a chain with more bends than inner vertices gets the missing ones on its last segment
  r = np.maximum(nB - nI, 0)
  other = F[cLast, 0] + F[cLast, 1] - cE
  g = np.repeat(np.arange(C), r)
  i = np.arange(1, r.sum() + 1) - np.repeat(np.cumsum(r) - r, r)
  w = nV + np.arange(r.sum())
  t = (i / (r[g] + 1))[:, None]
  V0 = np.vstack([V, (1 - t) * V[other[g]] + t * V[cE[g]]])
  V = V0.copy()
  isL = F[cLast, 1] == cE
  e0, e1 = i == 1, i == r[g]
  Fn = np.column_stack([w, np.where(e1, cE[g], w + 1)])
  Fp = np.column_stack([w, np.where(e0, other[g], w - 1)])
  F = F.copy()
  F[cLast[isL & (r > 0)], 1] = w[e0 & isL[g]]
  F[cLast[~isL & (r > 0)], 1] = w[e1 & ~isL[g]]
  F = np.vstack([F, np.where(isL[g][:, None], Fn, Fp)])
  src = np.concatenate([src, src[cLast[g]]])
  o = np.argsort(np.concatenate([ic, g]), kind="stable")
  iv = np.concatenate([iv, w])[o]
  nI = nI + r

  # the points of every chain from start to end, kb the number of the inner vertex on each of them:
  # proportional to the arc length, at least one vertex further than the point before
  pc = np.concatenate([np.arange(C), cb, np.arange(C)])
  po = np.concatenate([np.zeros(C, dtype=np.int64), np.arange(1, len(cb) + 1) - np.repeat(np.cumsum(nB) - nB, nB), nB + 1])
  pX = np.vstack([P.X[sS], P.X[ib], P.X[sE]])
  o = np.lexsort((po, pc))
  pc, po, pX = pc[o], po[o], pX[o]
  al = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pX, axis=0) @ G.M, axis=1))])
  al = al - np.repeat(al[po == 0], nB + 2)
  with np.errstate(invalid="ignore", divide="ignore"):
    kb = np.floor(al / np.repeat(al[po > nB[pc]], nB + 2) * (nI[pc] + 1) + 0.5)
  kb = np.where(np.isnan(kb), 0, kb).astype(np.int64)
  kb = np.maximum.accumulate(np.maximum(kb - po, 0) + 10 ** 7 * pc) - 10 ** 7 * pc
  kb = po + np.minimum(kb, nI[pc] - nB[pc])
  kb[po == 0] = 0
  kb[po > nB[pc]] = nI + 1

  # the inner vertices evenly on the straight pieces
  a = np.flatnonzero(po <= nB[pc])
  n = kb[a + 1] - kb[a]
  q = np.repeat(a, n)
  t = np.arange(1, n.sum() + 1) - np.repeat(np.cumsum(n) - n, n)
  s = kb[q] + t <= nI[pc[q]]
  q, t = q[s], t[s]
  off = np.cumsum(nI) - nI
  V[iv[off[pc[q]] + kb[q] + t - 1]] = G.toP(pX[q] + (t / (kb[q + 1] - kb[q]))[:, None] * (pX[q + 1] - pX[q]))
  V[J] = G.toP(Xj)
  return V, F, src, V0


def tighten(P, G):
  """every dirty point dropped or replaced by the cell corners between its neighbours, until none is
  left or, where points take turns on a row of collinear corners, none fewer for 50 passes"""
  rank = lambda i: np.mod((i + 1) * GOLD, 1)
  best, stall = np.inf, 0
  while True:
    # the dirty points before their dirty neighbours in a fixed shuffle, so no two are neighbours
    v = np.flatnonzero(P.dirty[:P.n])
    best, stall = (len(v), 0) if len(v) < best else (best, stall + 1)
    if len(v) == 0 or stall > 50:
      break
    p, n = P.nb[v, 0], P.nb[v, 1]
    s = (~P.dirty[p] | (rank(v) > rank(p))) & (~P.dirty[n] | (rank(v) > rank(n)))
    v, p, n = v[s], p[s], n[s]
    hk, hq = hullChains(P.X[p], P.X[v], P.X[n], G)
    cnt = np.bincount(hk, minlength=len(v))
    h1 = np.flatnonzero(np.diff(hk, prepend=-1) > 0)
    P.dirty[v] = False

    # a point on its only corner stays
    same = np.zeros(len(v), dtype=bool)
    same[hk[h1]] = (cnt[hk[h1]] == 1) & (np.linalg.norm(hq[h1] - P.X[v[hk[h1]]], axis=1) < 1e-9)

    # without a corner between its neighbours it goes
    dl = cnt == 0
    P.nb[p[dl], 1] = n[dl]
    P.nb[n[dl], 0] = p[dl]
    P.nb[v[dl]] = -1

    # else it moves to the first corner, new points take the others
    mv = (cnt > 0) & ~same
    P.X[v[mv]] = hq[h1[mv[hk[h1]]]]
    ex = np.flatnonzero(mv[hk] & (np.diff(hk, prepend=-1) == 0))
    c = P.add(len(ex))
    P.X[c] = hq[ex]
    isF, isL = np.diff(hk[ex], prepend=-1) > 0, np.diff(hk[ex], append=len(v)) > 0
    P.nb[c] = np.column_stack([c - 1, c + 1])
    P.nb[c[isF], 0] = v[hk[ex[isF]]]
    P.nb[v[hk[ex[isF]]], 1] = c[isF]
    P.nb[c[isL], 1] = n[hk[ex[isL]]]
    P.nb[n[hk[ex[isL]]], 0] = c[isL]

    # the neighbours of a point that went or moved are looked at again
    t = np.concatenate([p[dl | mv], n[dl | mv]])
    P.dirty[t[(P.nb[t] >= 0).all(axis=1)]] = True


def hullChains(Pp, Q, N, G):
  """the cell corners hq the shortest path from Pp to N bends around in place of Pp - Q - N, in
  order, hk the triangle: the hull of the cells that are no fill inside the triangle, on the side of Q"""
  B = N - Pp
  sg = np.sign(B[:, 0] * (Q[:, 1] - Pp[:, 1]) - B[:, 1] * (Q[:, 0] - Pp[:, 0]))
  T = np.stack([Pp, Q, N], axis=2)
  s, c = obstCells(T.min(axis=2), T.max(axis=2), sg != 0, G)

  # the cells that overlap the triangle by more than a touch, by the separating axes of the two
  isIn = np.ones(len(s), dtype=bool)
  nr = []
  for a in range(3):
    D = T[:, :, (a + 1) % 3] - T[:, :, a]
    nr.append(-sg[:, None] * np.column_stack([D[:, 1], -D[:, 0]]) / np.maximum(np.linalg.norm(D, axis=1), np.finfo(float).tiny)[:, None])
    isIn &= np.sum((c - T[s, :, a]) * nr[a][s], axis=1) < (1 - 2e-7) * (nr[a][s] @ G.C.T).max(axis=1, initial=-np.inf)
  for ax, wd in zip(G.axis, G.width):
    pj = np.einsum("nkj,k->nj", T, ax)[s]
    cp, h = c @ ax, (1 - 2e-7) * wd
    isIn &= (cp + h > pj.min(axis=1, initial=np.inf)) & (cp - h < pj.max(axis=1, initial=-np.inf))
  s, c = s[isIn], c[isIn]

  # their corners in the triangle, or just outside where the lattice is distorted
  pt = (c[None] + G.C[:, None]).reshape(-1, 2)
  pk = np.tile(s, len(G.C))
  isIn = np.ones(len(pk), dtype=bool)
  for a in range(3):
    isIn &= np.sum((pt - T[pk, :, a]) * nr[a][pk], axis=1) < 0.05
  pt, pk = pt[isIn], pk[isIn]

  # quickhull for all triangles at once: an edge with a corner above it is split at the farthest one,
  # key and width order the edges of a triangle from Pp to N
  uk, pe = np.unique(pk, return_inverse=True)
  EA, EB, EK = Pp[uk], N[uk], uk
  key, width = np.zeros(len(uk)), np.ones(len(uk))
  hk, hkey, hq = [uk[:0]], [key[:0]], [EA[:0]]
  while True:
    h = above(pt, EA[pe], EB[pe], sg[EK[pe]])
    pt, pe, h = pt[h > 0], pe[h > 0], h[h > 0]
    if len(pe) == 0:
      break
    o = np.lexsort((-h, pe))
    f = o[np.diff(pe[o], prepend=-1) > 0]
    ee, cp = pe[f], pt[f]
    right = np.full(len(EA), -1)
    right[ee] = len(EA) + np.arange(len(ee))
    width[ee] /= 2
    hk.append(EK[ee])
    hkey.append(key[ee] + width[ee])
    hq.append(cp)
    EA, EB, EK = np.vstack([EA, cp]), np.vstack([EB, EB[ee]]), np.concatenate([EK, EK[ee]])
    key, width = np.concatenate([key, key[ee] + width[ee]]), np.concatenate([width, width[ee]])
    EB[ee] = cp
    s = above(pt, EA[right[pe]], EB[right[pe]], sg[EK[pe]]) > 0
    pe[s] = right[pe[s]]
  hk, hkey, hq = np.concatenate(hk), np.concatenate(hkey), np.vstack(hq)
  o = np.lexsort((hkey, hk))
  return hk[o], hq[o]


def above(pt, A, B, sg):
  """how far the points pt lie above the edges A - B, on the side sg, less a tolerance"""
  D = B - A
  return sg * (D[:, 0] * (pt[:, 1] - A[:, 1]) - D[:, 1] * (pt[:, 0] - A[:, 0])) - 1e-9 * np.linalg.norm(D, axis=1)


def clipStep(x, d, G):
  """the fraction of every step d from x that stays in the fill"""
  s, c = obstCells(np.minimum(x, x + d), np.maximum(x, x + d), d.any(axis=1), G)

  # a cell side next to a filled cell is drawn in, a side next to another such cell pushed out
  tin, tout = np.full(len(s), -np.inf), np.full(len(s), np.inf)
  xs, ds = x[s] - c, d[s]
  for k in range(len(G.C)):
    num = G.dist[k] + 1e-9 * (2 * isObst(c + G.next[k], G) - 1) - xs @ G.normal[k]
    den = ds @ G.normal[k]
    tin[(np.abs(den) < 1e-14) & (num <= 0)] = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
      t = num / den
    neg, pos = den < -1e-14, den > 1e-14
    tin[neg] = np.maximum(tin[neg], t[neg])
    tout[pos] = np.minimum(tout[pos], t[pos])
  hit = (tout > np.maximum(tin, 0)) & (tin < 1)
  lam = np.ones(len(x))
  np.minimum.at(lam, s[hit], np.maximum(tin[hit], 0))
  return lam


def obstCells(lo, hi, isOn, G):
  """the cells c that are no fill around the boxes lo..hi with isOn, s the box"""
  t = np.flatnonzero(isOn & anyObst(lo, hi, G))
  lo, hi = np.ceil(lo[t] - G.ext).astype(np.int64), np.floor(hi[t] + G.ext).astype(np.int64)
  w = hi[:, 0] - lo[:, 0] + 1
  n = w * (hi[:, 1] - lo[:, 1] + 1)
  s = np.repeat(t, n)
  r = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
  w, lo = np.repeat(w, n), np.repeat(lo, n, axis=0)
  c = np.column_stack([lo[:, 0] + r % w, lo[:, 1] + r // w])
  isO = isObst(c, G)
  return s[isO], c[isO]


def isObst(c, G):
  """whether the cells c are no fill, as every cell off the raster"""
  isO = np.ones(len(c), dtype=bool)
  inR = np.all((c >= 0) & (c < G.sz), axis=1)
  isO[inR] = ~G.fill[c[inR, 0], c[inR, 1]]
  return isO


def anyObst(lo, hi, G):
  """whether the boxes of cells around lo..hi hold a cell that is no fill, from the summed area table"""
  lo, hi = np.ceil(lo - G.ext).astype(np.int64), np.floor(hi + G.ext).astype(np.int64)
  has = np.any((lo < 0) | (hi >= G.sz), axis=1)
  lo, hi = np.clip(lo, 0, G.sz - 1), np.clip(hi, 0, G.sz - 1) + 1
  S = G.sum
  return has | (S[hi[:, 0], hi[:, 1]] - S[lo[:, 0], hi[:, 1]] - S[hi[:, 0], lo[:, 1]] + S[lo[:, 0], lo[:, 1]] > 0)


def missedBoundaries(c, grainId):
  """the fill test without the regions of filled cells mostly in gaps one cell wide between two
  grains, None where no filled cell is left"""
  fd, sites = c.inFill, c.sites
  fill = fd.fill.copy()
  k = np.flatnonzero((sites.pixel >= 0) & ~sites.isHole)
  ij = sites.ij[k] - fd.lo
  Gm = np.zeros(fill.shape, dtype=np.int64)
  Gm[ij[:, 0], ij[:, 1]] = grainId[sites.pixel[k]]

  # the neighbours across an edge and, on a square grid, across a corner
  O = np.array([[1, 0], [0, 1], [1, 1], [1, -1]])
  mid = np.floor(np.array(fill.shape) / 2 + 0.5) - 1
  d = np.linalg.norm(fd.cell2pos(mid + O) - fd.cell2pos(mid), axis=1)
  ci, cj = np.nonzero(fill)
  thin = np.zeros(len(ci), dtype=bool)
  for o in O[d < 1.5 * d.min()]:
    g1, g2 = Gm[ci - o[0], cj - o[1]], Gm[ci + o[0], cj + o[1]]
    thin |= (g1 > 0) & (g2 > 0) & (g1 != g2)

  # the regions: filled cells joined across an edge
  idr = np.full(fill.shape, -1)
  idr[ci, cj] = np.arange(len(ci))
  t = [idr[ci + o[0], cj + o[1]] for o in O[d < 1.1 * d.min()]]
  r = components(len(ci), np.concatenate([np.flatnonzero(x >= 0) for x in t]), np.concatenate([x[x >= 0] for x in t]))
  share = np.bincount(r, thin, minlength=len(ci)) / np.maximum(np.bincount(r, minlength=len(ci)), 1)
  s = share[r] >= 0.5
  fill[ci[s], cj[s]] = False
  return fd.masked(fill) if fill.any() else None


class fillGeometry:
  """the lattice of the filled cells: in lattice coordinates cell (i, j) of the raster fill is the
  Voronoi cell of the point (i, j); the summed area table of the cells that are no fill, the maps
  to positions and back, the lattice basis M from the middle of the raster, the corners C of a cell
  counterclockwise around its centre and its sides: outward normal, distance from the centre, the
  cell beyond, and half of them as the separating axes"""

  def __init__(self, inFill):
    from scipy.spatial import Voronoi
    self.fill, self.sz = inFill.fill, np.array(inFill.fill.shape)
    self.sum = np.zeros(self.sz + 1)
    self.sum[1:, 1:] = np.cumsum(np.cumsum(~self.fill, axis=0), axis=1)
    self.toP = inFill.cell2pos
    c = np.floor(self.sz / 2 + 0.5) - 1
    self.M = self.toP(c + np.eye(2)) - self.toP(c)
    self.Minv = np.linalg.inv(self.M)
    self.p0 = self.toP(c)[0] - c @ self.M
    i, j = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3), indexing="ij")
    nbr = np.column_stack([i.ravel(order="F"), j.ravel(order="F")])
    vor = Voronoi(np.vstack([[0, 0], nbr[nbr.any(axis=1)] @ self.M]))
    C = vor.vertices[vor.regions[vor.point_region[0]]] @ self.Minv
    self.C = C[np.argsort(np.arctan2(C[:, 1], C[:, 0]), kind="stable")]
    self.ext = np.abs(C).max()
    D = np.roll(self.C, -1, axis=0) - self.C
    self.normal = np.column_stack([D[:, 1], -D[:, 0]]) / np.linalg.norm(D, axis=1)[:, None]
    self.dist = np.sum(self.C * self.normal, axis=1)
    self.next = np.round(self.C + np.roll(self.C, -1, axis=0)).astype(np.int64)
    h = len(C) // 2
    self.axis, self.width, self.side = self.normal[:h], self.dist[:h], D[:h]

  def toU(self, p):
    """the lattice coordinates of the positions p, exact after one step on a rigid lattice"""
    u = (p - self.p0) @ self.Minv
    for _ in range(4):
      u = u + (p - self.toP(u)) @ self.Minv
    return u


def fillEntry(V, F, k, d0, inFill, tol, dxy):
  """of the segments k with an end inside the fill (d0 the fill test at their first) those that
  leave it in between: that end, the first if both are, and the sample outside nearest to it,
  samples a third of a pixel apart"""
  in1 = (d0 > tol)[:, None]
  a = np.where(in1, V[F[k, 0]], V[F[k, 1]])
  b = np.where(in1, V[F[k, 1]], V[F[k, 0]])
  T = np.maximum(2, np.ceil(3 * np.linalg.norm(b - a, axis=1) / dxy)).astype(np.int64)
  s = np.repeat(np.arange(len(k)), T - 1)
  t = (np.arange(1, len(s) + 1) - np.repeat(np.cumsum(T - 1) - (T - 1), T - 1)) / T[s]
  isO = inFill(a[s] + t[:, None] * (b[s] - a[s])) < -tol
  tm = np.full(len(k), np.inf)
  np.minimum.at(tm, s[isO], t[isO])
  o = np.isfinite(tm)
  return k[o], a[o], a[o] + tm[o, None] * (b[o] - a[o])


__all__ = []
