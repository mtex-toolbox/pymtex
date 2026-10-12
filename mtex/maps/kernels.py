# the compiled kernels of the maps layer: the local Voronoi clipping of the decomposition
# as one numba loop over chunks of sites, each chunk writing its faces into its own slice
# of a buffer so that the chunks run in parallel without a shared counter; the union-find
# of the segmentation; the tracing of the grain loops; the fused gradient and kernel
# average misorientation over the lattice; and the dense simplex of the dislocation fit.
# The NumPy forms in decomposition.py, grains.py, intragranular.py and dislocations.py are
# the fallbacks without numba and the oracles of the tests.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import _jit, _parallelKernel
from ..geometry.session import getMTEXpref

prange = range
_COMPILED = {}


# whether the compiled loops run: the backend asked for, else the preference 'kernels', is numba
# and numba is installed
def _useNumba(backend=None):
  return (getMTEXpref("kernels", "numba") if backend is None else backend) == "numba" and bool(_jit())


# a loop compiled once, plain or parallel over the prange of the module it is written in; None
# where the compiled loops do not run
def _compiledLoop(f, parallel=False, backend=None, **options):
  if not _useNumba(backend):
    return None
  key = f"{f.__module__}.{f.__name__}"
  if key not in _COMPILED:
    nb = _jit()
    if parallel:
      f.__globals__["prange"] = nb.prange
    _COMPILED[key] = _parallelKernel(f) if parallel else nb.njit(cache=True, **options)(f)
  return _COMPILED[key]
K = 256            # the vertices a cell polygon may hold while it is clipped
NO_SITE = -1


# clip the polygon (vs, gs) of m vertices against the half plane of the points at least as
# close to p as to q, through the scratch (ws, hs); returns the new count, or -1 when the
# polygon outgrew its buffer
def _clip(vs, gs, m, ws, hs, px, py, qx, qy, gen):
  nx, ny = qx - px, qy - py
  c = (nx * qx + ny * qy + nx * px + ny * py) / 2
  cnt = 0
  for i in range(m):
    i2 = 0 if i == m - 1 else i + 1
    v1x, v1y = vs[i, 0], vs[i, 1]
    v2x, v2y = vs[i2, 0], vs[i2, 1]
    d1 = nx * v1x + ny * v1y - c
    d2 = nx * v2x + ny * v2y - c
    if d1 <= 0:
      if cnt + 2 > K:
        return -1
      ws[cnt, 0] = v1x
      ws[cnt, 1] = v1y
      hs[cnt] = gs[i]
      cnt += 1
      if d2 > 0:
        t = d1 / (d1 - d2)
        ws[cnt, 0] = v1x + t * (v2x - v1x)
        ws[cnt, 1] = v1y + t * (v2y - v1y)
        hs[cnt] = gen
        cnt += 1
    elif d2 <= 0:
      if cnt + 1 > K:
        return -1
      t = d1 / (d1 - d2)
      ws[cnt, 0] = v1x + t * (v2x - v1x)
      ws[cnt, 1] = v1y + t * (v2y - v1y)
      hs[cnt] = gs[i]
      cnt += 1
  for i in range(cnt):
    vs[i, 0] = ws[i, 0]
    vs[i, 1] = ws[i, 1]
    gs[i] = hs[i]
  return cnt


# the cell of site s against one candidate list: the seed box of half width W clipped by
# the sites the offsets reach; returns the vertex count, whether an edge of the box
# survived and whether the buffer overflowed
def _cellOf(vs, gs, ws, hs, pos, ij, lookup, off0, off1, s, offs, dIdeal, W, distortTol):
  px, py = pos[s, 0], pos[s, 1]
  a0 = ij[s, 0] - off0
  b0 = ij[s, 1] - off1
  n1, n2 = lookup.shape
  vs[0, 0], vs[0, 1] = px - W, py - W
  vs[1, 0], vs[1, 1] = px + W, py - W
  vs[2, 0], vs[2, 1] = px + W, py + W
  vs[3, 0], vs[3, 1] = px - W, py + W
  for i in range(4):
    gs[i] = NO_SITE
  m = 4
  rmax = W * np.sqrt(2.0)
  stillOpen = True
  for k in range(offs.shape[0]):
    if not stillOpen and dIdeal[k] * (1 - distortTol) > 2 * rmax:
      break
    a = a0 + offs[k, 0]
    b = b0 + offs[k, 1]
    if a < 0 or a >= n1 or b < 0 or b >= n2:
      continue
    s2 = lookup[a, b]
    if s2 < 0:
      continue
    m = _clip(vs, gs, m, ws, hs, px, py, pos[s2, 0], pos[s2, 1], s2)
    if m < 0:
      return 0, False, True
    rmax = 0.0
    stillOpen = False
    for i in range(m):
      dx, dy = vs[i, 0] - px, vs[i, 1] - py
      r = np.sqrt(dx * dx + dy * dy)
      if r > rmax:
        rmax = r
      if gs[i] == NO_SITE:
        stillOpen = True
  return m, stillOpen, False


# the faces of every site, chunk by chunk: cellsBuf[t, k] the site pair and vBuf[t, k]
# the two vertices of face k of chunk t, counts[t] how many, overflow[t] whether the
# chunk ran out of room, unbounded[t] how many cells of the map stayed open
def _localVoronoiLoop(pos, ij, pixel, isHole, lookup, off0, off1, offs1, d1, W1, offs2, d2, W2, offs3, d3, W3,
                      distortTol, chunkSize, cellsBuf, vBuf, counts, overflow, unbounded):
  ns = pos.shape[0]
  cap = cellsBuf.shape[1]
  for t in prange(counts.shape[0]):
    vs = np.empty((K, 2), pos.dtype)
    ws = np.empty((K, 2), pos.dtype)
    gs = np.empty(K, np.int32)
    hs = np.empty(K, np.int32)
    cnt = 0
    unb = 0
    ovf = False
    for s in range(t * chunkSize, min(ns, (t + 1) * chunkSize)):
      isMap = pixel[s] >= 0 or isHole[s]
      m = 0
      stillOpen = True
      for lv in range(3):
        if not (stillOpen and (lv == 0 or isMap)):
          break
        if lv == 0:
          m, stillOpen, bad = _cellOf(vs, gs, ws, hs, pos, ij, lookup, off0, off1, s, offs1, d1, W1, distortTol)
        elif lv == 1:
          m, stillOpen, bad = _cellOf(vs, gs, ws, hs, pos, ij, lookup, off0, off1, s, offs2, d2, W2, distortTol)
        else:
          m, stillOpen, bad = _cellOf(vs, gs, ws, hs, pos, ij, lookup, off0, off1, s, offs3, d3, W3, distortTol)
        if bad:
          ovf = True
      for i in range(m):
        g = gs[i]
        if g == NO_SITE:
          if isMap:
            unb += 1
          continue
        if g > s:
          if cnt >= cap:
            ovf = True
            break
          i2 = 0 if i == m - 1 else i + 1
          cellsBuf[t, cnt, 0] = s
          cellsBuf[t, cnt, 1] = g
          vBuf[t, cnt, 0] = vs[i, 0]
          vBuf[t, cnt, 1] = vs[i, 1]
          vBuf[t, cnt, 2] = vs[i2, 0]
          vBuf[t, cnt, 3] = vs[i2, 1]
          cnt += 1
    counts[t] = cnt
    overflow[t] = ovf
    unbounded[t] = unb


# ------------------------------------------------------------------ union-find

# the root of every node after uniting the edges (a, b), with path halving and union by size
def _unionFindLoop(parent, size, a, b):
  for k in range(a.shape[0]):
    x = a[k]
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    y = b[k]
    while parent[y] != y:
      parent[y] = parent[parent[y]]
      y = parent[y]
    if x == y:
      continue
    if size[x] < size[y]:
      x, y = y, x
    parent[y] = x
    size[x] += size[y]
  for k in range(parent.shape[0]):
    x = k
    while parent[x] != x:
      x = parent[x]
    parent[k] = x


def unionFind(n, a, b, backend=None):
  """the component root of every node of a graph on n nodes, or None without the compiled loops"""
  loop = _compiledLoop(_unionFindLoop, backend=backend)
  if loop is None:
    return None
  parent = np.arange(n, dtype=np.int32)
  size = np.ones(n, dtype=np.int32)
  loop(parent, size, np.ascontiguousarray(a, dtype=np.int32), np.ascontiguousarray(b, dtype=np.int32))
  return parent


# ------------------------------------------------------------------ the weld of the vertices

# the point stored under the bucket b and the fine grid cell (k0[p], k1[p]): a bucket keeps
# the first point of each of its cells in a short list through nxt, and p is stored when its
# cell is new
def _gridInsert(head, nxt, k0, k1, b, p):
  q = head[b]
  while q >= 0:
    if k0[q] == k0[p] and k1[q] == k1[p]:
      return q
    q = nxt[q]
  nxt[p] = head[b]
  head[b] = p
  return p


def _root(parent, x):
  while parent[x] != x:
    parent[x] = parent[parent[x]]
    x = parent[x]
  return x


# the first point on the fine grid cell of every flagged point, after the shift h: the fine
# cells and their coarse buckets computed once, then the threads share the coarse rows so
# that no bucket is written by two of them, and the first point of a cell is the smallest
# index on it whatever the schedule
def _firstOnCell(xy, lo0, lo1, c, h, M, n1, flag, head, nxt, k0, k1, bucket, owner, out, nT):
  n = xy.shape[0]
  for i in prange(n):
    a = np.int64(np.floor((xy[i, 0] - lo0) / c + h))
    b = np.int64(np.floor((xy[i, 1] - lo1) / c + h))
    k0[i], k1[i] = a, b
    r = a // M + 1
    bucket[i] = r * n1 + b // M + 1
    owner[i] = r % nT
  for t in prange(nT):
    for i in range(n):
      if owner[i] == t and flag[i]:
        out[i] = _gridInsert(head, nxt, k0, k1, bucket[i], i)


# the vertex groups of meshVertices before its tolerance grid: the points on one cell of the
# fine grid are one group, a point within rounding of a boundary of that grid joins the first
# point on its cell of the half-shifted grid, and the groups this links are one; returns the
# group of every point, numbered by first occurrence, and the first point of every group
def _weldLoop(xy, lo0, lo1, c, M, nT, group, first):
  n = xy.shape[0]
  k0max = k1max = np.int64(0)
  for i in range(n):
    k0max = max(k0max, np.int64(np.floor((xy[i, 0] - lo0) / c + 0.5)))
    k1max = max(k1max, np.int64(np.floor((xy[i, 1] - lo1) / c + 0.5)))
  # the coarse cells grow until there are no more buckets than a few per point
  while (k0max // M + 3) * (k1max // M + 3) > 4 * n + 16:
    M *= 2
  n1 = k1max // M + 3
  head = np.full((k0max // M + 3) * n1, -1, np.int32)
  nxt = np.empty(n, np.int32)
  k0, k1 = np.empty(n, np.int64), np.empty(n, np.int64)
  bucket, owner = np.empty(n, np.int32), np.empty(n, np.int32)
  near = np.ones(n, np.bool_)
  rep = np.empty(n, np.int32)
  _firstOnCell(xy, lo0, lo1, c, 0.0, M, n1, near, head, nxt, k0, k1, bucket, owner, rep, nT)
  # the points within rounding of a boundary of the fine grid
  for i in prange(n):
    u0, u1 = (xy[i, 0] - lo0) / c, (xy[i, 1] - lo1) / c
    f0, f1 = u0 - np.floor(u0), u1 - np.floor(u1)
    near[i] = f0 < 1e-3 or f0 > 1 - 1e-3 or f1 < 1e-3 or f1 > 1 - 1e-3
  head[:] = -1
  mate = nxt.copy()
  _firstOnCell(xy, lo0, lo1, c, 0.5, M, n1, near, head, nxt, k0, k1, bucket, owner, mate, nT)
  # the groups the half-shifted grid links, rooted at their smallest point, then numbered
  # by first occurrence; bucket is free again and holds the numbers
  parent = owner
  for i in prange(n):
    parent[i] = i
  for i in range(n):
    if near[i]:
      a, b = _root(parent, rep[i]), _root(parent, rep[mate[i]])
      if a != b:
        if a < b:
          parent[b] = a
        else:
          parent[a] = b
  label = bucket
  label[:] = -1
  m = 0
  for i in range(n):
    r = _root(parent, rep[i])
    if label[r] < 0:
      label[r] = m
      first[m] = i
      m += 1
    group[i] = label[r]
  return m


def weldNumba(xy, lo, c, cell, backend=None):
  """the groups and first points of meshVertices through the compiled weld, or None without the
  compiled loops; the coarse grid has cells of about cell in the units of xy"""
  if not _useNumba(backend):
    return None
  _bindHelpers(("_gridInsert", "_root"))
  _bindHelpers(("_firstOnCell",), parallel=True)
  loop = _compiledLoop(_weldLoop, parallel=True, backend=backend)
  xy = np.ascontiguousarray(xy, dtype=np.float64)
  group = np.empty(len(xy), np.int32)
  first = np.empty(len(xy), np.int32)
  M = max(int(cell / c), 1)
  m = loop(xy, float(lo[0]), float(lo[1]), float(c), np.int64(M), np.int64(_jit().get_num_threads()), group, first)
  return group, first[:m]


# ------------------------------------------------------------------ compilation

# the helpers the loops of this module call, compiled once and bound in place of their Python
# forms, so that a loop compiled after them calls the compiled ones
def _bindHelpers(names, parallel=False):
  nb = _jit()
  globals()["prange"] = nb.prange
  for name in names:
    if name not in _COMPILED:
      _COMPILED[name] = nb.njit(cache=True, parallel=parallel)(globals()[name])
      globals()[name] = _COMPILED[name]


def localVoronoiNumba(sites, levels, distortTol=0.05, chunkSize=4096, backend=None):
  """the faces of every cell through the compiled loop, or None without the compiled loops"""
  if not _useNumba(backend):
    return None
  _bindHelpers(("_clip", "_cellOf"))
  loop = _compiledLoop(_localVoronoiLoop, parallel=True, backend=backend)
  pos = np.ascontiguousarray(sites.pos)
  ij = np.ascontiguousarray(sites.ij, dtype=np.int32)
  lookup = np.ascontiguousarray(sites.lookup, dtype=np.int32)
  ns = len(pos)
  nChunks = max(1, (ns + chunkSize - 1) // chunkSize)
  args = []
  for offs, d, W in levels:
    args += [np.ascontiguousarray(offs, dtype=np.int64), np.ascontiguousarray(d, dtype=pos.dtype), pos.dtype.type(W)]
  cap = 4 * chunkSize + 256
  while True:
    cellsBuf = np.empty((nChunks, cap, 2), dtype=np.int32)
    vBuf = np.empty((nChunks, cap, 4), dtype=pos.dtype)
    counts = np.zeros(nChunks, dtype=np.int64)
    overflow = np.zeros(nChunks, dtype=np.bool_)
    unbounded = np.zeros(nChunks, dtype=np.int64)
    loop(pos, ij, sites.pixel, sites.isHole, lookup, np.int64(sites.offset[0]), np.int64(sites.offset[1]), *args,
         pos.dtype.type(distortTol), np.int64(chunkSize), cellsBuf, vBuf, counts, overflow, unbounded)
    if not overflow.any():
      break
    if cap > 64 * chunkSize:
      return None
    cap *= 2
  keep = np.flatnonzero(np.arange(cap)[None, :] < counts[:, None])
  return cellsBuf.reshape(-1, 2).take(keep, axis=0), vBuf.reshape(-1, 4).take(keep, axis=0), int(unbounded.sum())



# ------------------------------------------------------------------ loop tracing

# the first and one past the last position of a value in a sorted array
def _range(sorted_, v):
  lo, hi = 0, sorted_.shape[0]
  while lo < hi:
    mid = (lo + hi) // 2
    if sorted_[mid] < v:
      lo = mid + 1
    else:
      hi = mid
  first = lo
  hi = sorted_.shape[0]
  while lo < hi:
    mid = (lo + hi) // 2
    if sorted_[mid] <= v:
      lo = mid + 1
    else:
      hi = mid
  return first, lo


# the loops of every grain from its directed edges, tails ea and heads eb sorted by grain
# with the offsets goff: a walk leaves a vertex by the one edge that starts there, or by the
# one that turns most tightly clockwise where the grain touches itself, which keeps the face
# on the left. The grains are walked in parallel, each into its own slice [goff[g], goff[g+1])
# of the scratch, which holds its loops since a loop has as many vertices as edges: the
# vertices into vtmp, the start of every loop into ltmp, twice its signed area into atmp, the
# counts of loops and vertices into nlg and nvg; a loop of fewer than three vertices is dropped
def _traceGrainsLoop(V, ea, eb, goff, used, vtmp, ltmp, atmp, nlg, nvg):
  for g in prange(goff.shape[0] - 1):
    s, e = goff[g], goff[g + 1]
    m = e - s
    nlg[g] = 0
    nvg[g] = 0
    if m == 0:
      continue
    order = np.argsort(ea[s:e]) + s
    tails = ea[order]
    nv = s
    nl = s
    for st in range(m):
      k = s + st
      if used[k]:
        continue
      start = nv
      area = 0.0
      while True:
        used[k] = True
        vtmp[nv] = ea[k]
        nv += 1
        area += V[ea[k], 0] * V[eb[k], 1] - V[eb[k], 0] * V[ea[k], 1]
        lo, hi = _range(tails, eb[k])
        nxt = -1
        if hi - lo == 1:
          c = order[lo]
          if not used[c]:
            nxt = c
        elif hi - lo > 1:
          ax, ay = V[eb[k], 0] - V[ea[k], 0], V[eb[k], 1] - V[ea[k], 1]
          best = -np.inf
          for t in range(lo, hi):
            c = order[t]
            if used[c]:
              continue
            bx, by = V[eb[c], 0] - V[ea[c], 0], V[eb[c], 1] - V[ea[c], 1]
            turn = np.arctan2(ax * by - ay * bx, ax * bx + ay * by)
            if turn > best:
              best, nxt = turn, c
        if nxt < 0:
          break
        k = nxt
      if nv - start >= 3:
        ltmp[nl] = start
        atmp[nl] = area
        nl += 1
      else:
        nv = start
    nlg[g] = nl - s
    nvg[g] = nv - s


# the slices of the grains moved together in grain order, the loop starts made flat
def _compactLoopsLoop(goff, nlg, nvg, loopOff, vertOff, vtmp, ltmp, atmp, loopV, loff, lgrain, larea):
  for g in prange(goff.shape[0] - 1):
    s = goff[g]
    for j in range(nvg[g]):
      loopV[vertOff[g] + j] = vtmp[s + j]
    for j in range(nlg[g]):
      loff[loopOff[g] + j] = ltmp[s + j] - s + vertOff[g]
      lgrain[loopOff[g] + j] = g
      larea[loopOff[g] + j] = atmp[s + j]


def traceLoopsNumba(V, ea, eb, goff):
  """the loops of every grain through the compiled walk: the flat vertices, the loop offsets,
  the grain and twice the signed area of every loop, in grain order; None without the compiled loops"""
  trace, compact = _pixelLoop(_traceGrainsLoop, True), _pixelLoop(_compactLoopsLoop, True)
  if trace is None:
    return None
  n, G = len(ea), len(goff) - 1
  goff = np.ascontiguousarray(goff, dtype=np.int64)
  ea, eb = np.ascontiguousarray(ea, dtype=np.int32), np.ascontiguousarray(eb, dtype=np.int32)
  vtmp, ltmp, atmp = np.empty(n, dtype=np.int32), np.empty(n, dtype=np.int64), np.empty(n)
  nlg, nvg = np.empty(G, dtype=np.int64), np.empty(G, dtype=np.int64)
  trace(np.ascontiguousarray(V, dtype=np.float64), ea, eb, goff, np.zeros(n, dtype=np.bool_), vtmp, ltmp, atmp, nlg, nvg)
  loopOff = np.concatenate([[0], np.cumsum(nlg)])
  vertOff = np.concatenate([[0], np.cumsum(nvg)])
  nl, nv = int(loopOff[-1]), int(vertOff[-1])
  loopV, loff = np.empty(nv, dtype=np.int32), np.empty(nl + 1, dtype=np.int64)
  lgrain, larea = np.empty(nl, dtype=np.int64), np.empty(nl)
  compact(goff, nlg, nvg, loopOff, vertOff, vtmp, ltmp, atmp, loopV, loff, lgrain, larea)
  loff[nl] = nv
  return loopV, loff, lgrain, larea


# the segments at every vertex in increasing segment order, a counting sort that is the
# stable argsort of F.reshape(-1): entry i of that list is segment i // 2
def _adjacencyLoop(F, offset, at):
  ptr = offset[:-1].copy()
  for i in range(2 * F.shape[0]):
    v = F[i // 2, i % 2]
    at[ptr[v]] = i // 2
    ptr[v] += 1


# the directed edges of every grain, a counting sort by grain that is the stable argsort of
# the edges of the first sides followed by those of the second: a segment runs forward for
# its first grain and backward for its second, grains outside ids (pos -1) left out
def _edgesByGrainLoop(F, gid, posOf, goff, ea, eb):
  ptr = goff[:-1].copy()
  n = F.shape[0]
  for side in range(2):
    for k in range(n):
      g = posOf[gid[k, side]]
      if g >= 0:
        ea[ptr[g]] = F[k, side]
        eb[ptr[g]] = F[k, 1 - side]
        ptr[g] += 1


def adjacencyNumba(F, offset):
  """the adjacency of chainOrder, or None without the compiled loops"""
  loop = _compiledLoop(_adjacencyLoop)
  if loop is None:
    return None
  at = np.empty(2 * len(F), dtype=np.int32)
  loop(np.ascontiguousarray(F, dtype=np.int32), np.ascontiguousarray(offset, dtype=np.int64), at)
  return at


def edgesByGrainNumba(F, gid, posOf, nIds):
  """the directed edges of the grains in ids sorted by grain with their offsets, or None without
  the compiled loops"""
  loop = _compiledLoop(_edgesByGrainLoop)
  if loop is None:
    return None
  gid = np.ascontiguousarray(gid, dtype=np.int64)
  g0, g1 = posOf[gid[:, 0]], posOf[gid[:, 1]]
  counts = np.bincount(g0[g0 >= 0], minlength=nIds) + np.bincount(g1[g1 >= 0], minlength=nIds)
  goff = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
  ea = np.empty(goff[-1], dtype=np.int32)
  eb = np.empty(goff[-1], dtype=np.int32)
  loop(np.ascontiguousarray(F, dtype=np.int32), gid, np.ascontiguousarray(posOf, dtype=np.int64), goff, ea, eb)
  return ea, eb, goff


# ------------------------------------------------------------------ the gradient and the KAM

# the image of q under the element g of the group, q g, closest to p, with the sign that
# puts it on p's side of the sphere
def _closestTo(q0, q1, q2, q3, p0, p1, p2, p3, G, nG):
  best = -1.0
  c0 = c1 = c2 = c3 = 0.0
  for gi in range(nG):
    g0, g1, g2, g3 = G[gi, 0], G[gi, 1], G[gi, 2], G[gi, 3]
    r0 = q0 * g0 - q1 * g1 - q2 * g2 - q3 * g3
    r1 = q0 * g1 + q1 * g0 + q2 * g3 - q3 * g2
    r2 = q0 * g2 - q1 * g3 + q2 * g0 + q3 * g1
    r3 = q0 * g3 + q1 * g2 - q2 * g1 + q3 * g0
    d = r0 * p0 + r1 * p1 + r2 * p2 + r3 * p3
    if abs(d) > best:
      best = abs(d)
      sgn = -1.0 if d < 0 else 1.0
      c0, c1, c2, c3 = sgn * r0, sgn * r1, sgn * r2, sgn * r3
  return c0, c1, c2, c3


# the angle from p to the closest image q g of q: <q g, p> = <g, m> for m = inv(q) p, so one
# product and a dot per element find it, and the chord |m -/+ g| measures it, accurate near zero
def _closestAngle(q0, q1, q2, q3, p0, p1, p2, p3, G, nG):
  m0 = q0 * p0 + q1 * p1 + q2 * p2 + q3 * p3
  m1 = q0 * p1 - q1 * p0 - q2 * p3 + q3 * p2
  m2 = q0 * p2 + q1 * p3 - q2 * p0 - q3 * p1
  m3 = q0 * p3 - q1 * p2 + q2 * p1 - q3 * p0
  best, d, k = -1.0, 0.0, 0
  for gi in range(nG):
    t = m0 * G[gi, 0] + m1 * G[gi, 1] + m2 * G[gi, 2] + m3 * G[gi, 3]
    if abs(t) > best:
      best, d, k = abs(t), t, gi
  s = -1.0 if d < 0 else 1.0
  h0, h1, h2, h3 = m0 - s * G[k, 0], m1 - s * G[k, 1], m2 - s * G[k, 2], m3 - s * G[k, 3]
  return 4 * np.arcsin(min(np.sqrt(h0 * h0 + h1 * h1 + h2 * h2 + h3 * h3) / 2, 1.0))


# the neighbour of k at the offset (oi, oj) of the lattice, -1 where the map has none
def _neighbour(lk, ij, k, oi, oj):
  i, j = ij[k, 0] + oi, ij[k, 1] + oj
  if i < 0 or j < 0 or i >= lk.shape[0] or j >= lk.shape[1]:
    return -1
  return lk[i, j]


def _comparable(indexed, phase, grain, hasGrain, a, b):
  if b < 0 or not indexed[a] or not indexed[b] or phase[a] != phase[b]:
    return False
  return (not hasGrain) or grain[a] == grain[b]


# the gradient of every pixel from the normal equations over its neighbours: the one sided
# stencil steps backwards where the map ends and gives up where a neighbour is missing
# inside it, the wider ones answer from what is left; a pixel whose neighbours do not span
# two directions is NaN
def _gradientLoop(rot, indexed, phase, grain, hasGrain, lk, ij, offs, D, oneSided, Gall, nG, gx, gy):
  n = rot.shape[0]
  for k in prange(n):
    m11 = m12 = m22 = 0.0
    r1x = r1y = r1z = r2x = r2y = r2z = 0.0
    bad = not indexed[k]
    p = phase[k]
    q0, q1, q2, q3 = rot[k, 0], rot[k, 1], rot[k, 2], rot[k, 3]
    for o in range(offs.shape[0]):
      if bad:
        break
      oi, oj = offs[o, 0], offs[o, 1]
      dx, dy = D[o, 0], D[o, 1]
      i, j = ij[k, 0] + oi, ij[k, 1] + oj
      inside = i >= 0 and j >= 0 and i < lk.shape[0] and j < lk.shape[1]
      if oneSided and not inside:
        oi, oj, dx, dy = -oi, -oj, -dx, -dy
      nb = _neighbour(lk, ij, k, oi, oj)
      if not _comparable(indexed, phase, grain, hasGrain, k, nb):
        if oneSided:
          bad = True
        continue
      c0, c1, c2, c3 = _closestTo(rot[nb, 0], rot[nb, 1], rot[nb, 2], rot[nb, 3], q0, q1, q2, q3, Gall[p], nG[p])
      # the left tangent vector c conj(q), as a rotation vector
      m0 = c0 * q0 + c1 * q1 + c2 * q2 + c3 * q3
      m1 = -c0 * q1 + c1 * q0 - c2 * q3 + c3 * q2
      m2 = -c0 * q2 + c1 * q3 + c2 * q0 - c3 * q1
      m3 = -c0 * q3 - c1 * q2 + c2 * q1 + c3 * q0
      nrm = np.sqrt(m1 * m1 + m2 * m2 + m3 * m3)
      f = 0.0 if nrm < 1e-15 else (2 * np.arctan2(nrm, abs(m0)) / nrm) * (-1.0 if m0 < 0 else 1.0)
      hx, hy, hz = f * m1, f * m2, f * m3
      m11 += dx * dx
      m12 += dx * dy
      m22 += dy * dy
      r1x += dx * hx
      r1y += dx * hy
      r1z += dx * hz
      r2x += dy * hx
      r2y += dy * hy
      r2z += dy * hz
    det = m11 * m22 - m12 * m12
    if (not bad) and det > 0 and det > 1e-10 * m11 * m22:
      gx[k, 0] = (m22 * r1x - m12 * r2x) / det
      gx[k, 1] = (m22 * r1y - m12 * r2y) / det
      gx[k, 2] = (m22 * r1z - m12 * r2z) / det
      gy[k, 0] = (m11 * r2x - m12 * r1x) / det
      gy[k, 1] = (m11 * r2y - m12 * r1y) / det
      gy[k, 2] = (m11 * r2z - m12 * r1z) / det
    else:
      for c in range(3):
        gx[k, c] = np.nan
        gy[k, c] = np.nan


# the kernel average misorientation of every pixel over the offsets and their weights: the
# angle to the closest equivalent from the chord, the pairs above the threshold dropped
def _kamLoop(rot, indexed, phase, grain, hasGrain, lk, ij, offs, w, threshold, maximal, Gall, nG, out):
  n = rot.shape[0]
  for k in prange(n):
    acc = cnt = best = 0.0
    found = False
    if indexed[k]:
      p = phase[k]
      q0, q1, q2, q3 = rot[k, 0], rot[k, 1], rot[k, 2], rot[k, 3]
      for o in range(offs.shape[0]):
        nb = _neighbour(lk, ij, k, offs[o, 0], offs[o, 1])
        if not _comparable(indexed, phase, grain, hasGrain, k, nb):
          continue
        ang = _closestAngle(rot[nb, 0], rot[nb, 1], rot[nb, 2], rot[nb, 3], q0, q1, q2, q3, Gall[p], nG[p])
        if ang > threshold:
          continue
        found = True
        if ang > best:
          best = ang
        acc += w[o] * ang
        cnt += w[o]
    out[k] = (best if maximal else acc / cnt) if found else np.nan


# the grain reference orientation deviation of every pixel: the reference taken over the
# equivalent closest to the pixel, and the deviation inv(ref) rot
def _grodLoop(rot, ref, indexed, G, nG, out):
  n = rot.shape[0]
  for k in prange(n):
    if (not indexed[k]) or np.isnan(ref[k, 0]):
      for c in range(4):
        out[k, c] = np.nan
      continue
    q0, q1, q2, q3 = rot[k, 0], rot[k, 1], rot[k, 2], rot[k, 3]
    c0, c1, c2, c3 = _closestTo(ref[k, 0], ref[k, 1], ref[k, 2], ref[k, 3], q0, q1, q2, q3, G, nG)
    out[k, 0] = c0 * q0 + c1 * q1 + c2 * q2 + c3 * q3
    out[k, 1] = c0 * q1 - c1 * q0 - c2 * q3 + c3 * q2
    out[k, 2] = c0 * q2 + c1 * q3 - c2 * q0 - c3 * q1
    out[k, 3] = c0 * q3 - c1 * q2 + c2 * q1 - c3 * q0


# ------------------------------------------------------------------ the simplex

# one pivot of the tableau on (r, c)
def _pivot(T, basis, r, c):
  piv = T[r, c]
  for j in range(T.shape[1]):
    T[r, j] /= piv
  for i in range(T.shape[0]):
    if i == r:
      continue
    f = T[i, c]
    if f != 0.0:
      for j in range(T.shape[1]):
        T[i, j] -= f * T[r, j]
  basis[r] = c


# the reduced costs of the tableau for a cost vector over the first ncols columns
def _reducedCosts(T, basis, cost, ncols):
  m = basis.shape[0]
  last = T.shape[0] - 1
  for j in range(T.shape[1]):
    T[last, j] = cost[j] if j < ncols else 0.0
  for i in range(m):
    cb = cost[basis[i]] if basis[i] < ncols else 0.0
    if cb != 0.0:
      for j in range(T.shape[1]):
        T[last, j] -= cb * T[i, j]


# the simplex iteration over the first ncols columns with Bland's rule: 0 optimal, 1
# unbounded, 2 the iteration limit
def _simplexSolve(T, basis, ncols, tol, maxIter):
  m = basis.shape[0]
  last = T.shape[0] - 1
  rhs = T.shape[1] - 1
  for _ in range(maxIter):
    c = -1
    for j in range(ncols):
      if T[last, j] < -tol:
        c = j
        break
    if c < 0:
      return 0
    r = -1
    ratio = np.inf
    for i in range(m):
      if T[i, c] > tol:
        q = T[i, rhs] / T[i, c]
        if q < ratio - 1e-12 or (abs(q - ratio) <= 1e-12 and r >= 0 and basis[i] < basis[r]):
          ratio, r = q, i
    if r < 0:
      return 1
    _pivot(T, basis, r, c)
  return 2


# min c x under A x = b, x >= 0, by the two phase simplex on a dense tableau: the rows are
# scaled and signed so the artificial basis is feasible, phase one drives the artificials
# out, phase two runs on the original columns. Returns 0 with x written, or a code
def _linearProgram(A, b, c, tol, maxIter, x):
  m, n = A.shape
  T = np.zeros((m + 1, n + m + 1))
  basis = np.empty(m, dtype=np.int64)
  for i in range(m):
    scale = abs(b[i])
    for j in range(n):
      scale = max(scale, abs(A[i, j]))
    if scale == 0.0:
      scale = 1.0
    if b[i] < 0:
      scale = -scale
    for j in range(n):
      T[i, j] = A[i, j] / scale
    T[i, n + m] = b[i] / scale
    T[i, n + i] = 1.0
    basis[i] = n + i
  cost = np.zeros(n + m)
  for j in range(n, n + m):
    cost[j] = 1.0
  _reducedCosts(T, basis, cost, n + m)
  status = _simplexSolve(T, basis, n + m, tol, maxIter)
  if status != 0:
    return 3
  infeas = 0.0
  for i in range(m):
    if basis[i] >= n:
      infeas += T[i, n + m]
  if infeas > 10 * tol:
    return 4
  # an artificial still basic at zero is pivoted out on any original column, or its row
  # is redundant and ignored
  alive = np.ones(m, dtype=np.bool_)
  for i in range(m):
    if basis[i] >= n:
      col = -1
      for j in range(n):
        if abs(T[i, j]) > tol:
          inBasis = False
          for t in range(m):
            if basis[t] == j:
              inBasis = True
          if not inBasis:
            col = j
            break
      if col < 0:
        alive[i] = False
        for j in range(T.shape[1]):
          T[i, j] = 0.0
      else:
        _pivot(T, basis, i, col)
  scale = 1e-300
  for j in range(n):
    scale = max(scale, abs(c[j]))
  cost2 = np.zeros(n + m)
  for j in range(n):
    cost2[j] = c[j] / scale
  _reducedCosts(T, basis, cost2, n)
  status = _simplexSolve(T, basis, n, tol, maxIter)
  if status != 0:
    return 5
  for j in range(n):
    x[j] = 0.0
  for i in range(m):
    if alive[i] and basis[i] < n:
      x[basis[i]] = max(T[i, n + m], 0.0)
  return 0


# M z = r for a symmetric positive semidefinite M by Cholesky, the diagonal raised by 1e-16 of
# its largest entry, so that dependent rows of a program leave it solvable; more would shift
# the steps of the badly conditioned last iterations off the constraints
def _choleskySolve(M, r):
  m = M.shape[0]
  jitter = 0.0
  for i in range(m):
    jitter = max(jitter, M[i, i])
  jitter = 1e-16 * jitter if jitter > 0 else 1e-300
  L = np.zeros((m, m))
  for i in range(m):
    for j in range(i + 1):
      t = M[i, j] + (jitter if i == j else 0.0)
      for k in range(j):
        t -= L[i, k] * L[j, k]
      L[i, j] = np.sqrt(max(t, jitter)) if i == j else t / L[j, j]
  z = np.empty(m)
  for i in range(m):
    t = r[i]
    for k in range(i):
      t -= L[i, k] * z[k]
    z[i] = t / L[i, i]
  for i in range(m - 1, -1, -1):
    t = z[i]
    for k in range(i + 1, m):
      t -= L[k, i] * z[k]
    z[i] = t / L[i, i]
  return z


# min c x under A x = b, x >= 0, at the analytic centre of the optimal face, the solution a
# symmetric program keeps symmetric, by the primal dual path following method: Newton steps
# towards the central path at 0.3 mu, kept inside the positive orthant, the normal
# equations A diag(x / s) A^T by Cholesky. Returns 0 with x written, 1 where the residuals and
# the duality gap x.s, the error of the objective, do not fall below tol relative to the data
# within maxIter
def _centreProgram(A, b, c, tol, maxIter, x):
  m, n = A.shape
  s, dx, ds, d, w = np.ones(n), np.empty(n), np.empty(n), np.empty(n), np.empty(n)
  lam, dlam, rp, rhs = np.zeros(m), np.empty(m), np.empty(m), np.empty(m)
  rd = np.empty(n)
  M = np.empty((m, m))
  for j in range(n):
    x[j] = 1.0
  nb = np.sqrt(b @ b)
  nc = np.sqrt(c @ c)
  for _ in range(maxIter):
    # the residuals, the duality gap and the objective
    np_, nd, gap, cx = 0.0, 0.0, 0.0, 0.0
    for i in range(m):
      t = b[i]
      for j in range(n):
        t -= A[i, j] * x[j]
      rp[i] = t
      np_ += t * t
    for j in range(n):
      t = c[j] - s[j]
      for i in range(m):
        t -= A[i, j] * lam[i]
      rd[j] = t
      nd += t * t
      gap += x[j] * s[j]
      cx += c[j] * x[j]
    if np.sqrt(np_) <= tol * (1 + nb) and np.sqrt(nd) <= tol * (1 + nc) and gap <= tol * (1 + abs(cx)):
      return 0
    # the Newton step towards the central path at 0.3 mu, by the normal equations
    target = 0.3 * gap / n
    for j in range(n):
      d[j] = x[j] / s[j]
      w[j] = (target - x[j] * s[j] - x[j] * rd[j]) / s[j]
    for i in range(m):
      t = rp[i]
      for j in range(n):
        t -= A[i, j] * w[j]
      rhs[i] = t
      for k in range(i + 1):
        t = 0.0
        for j in range(n):
          t += A[i, j] * d[j] * A[k, j]
        M[i, k] = t
        M[k, i] = t
    dlam[:] = _choleskySolve(M, rhs)
    ap, ad = 1.0, 1.0
    for j in range(n):
      t = rd[j]
      for i in range(m):
        t -= A[i, j] * dlam[i]
      ds[j] = t
      dx[j] = (target - x[j] * s[j] - x[j] * t) / s[j]
      if dx[j] < 0:
        ap = min(ap, -0.95 * x[j] / dx[j])
      if ds[j] < 0:
        ad = min(ad, -0.95 * s[j] / ds[j])
    for j in range(n):
      x[j] += ap * dx[j]
      s[j] += ad * ds[j]
    for i in range(m):
      lam[i] += ad * dlam[i]
  return 1


# the dislocation fit of every pixel: the systems in the frame of the curvature, one set for
# all pixels or one per pixel, the curvature of each system on the known components as the
# columns of the program, both signs of every system, the line energies as the cost, solved at
# the centre of its optimal face; a pixel with an unknown curvature or systems is left NaN
def _gndLoop(kappa, known, b, l, u, tol, maxIter, gnd, rho):
  n = kappa.shape[0]
  ns = b.shape[1]
  m = known.shape[0]
  for k in prange(n):
    gnd[k] = np.nan
    for j in range(ns):
      rho[k, j] = np.nan
    kb = np.int64(k) if b.shape[0] > 1 else np.int64(0)
    ok = True
    for r in range(m):
      if np.isnan(kappa[k, known[r]]):
        ok = False
    for j in range(ns):
      if np.isnan(b[kb, j, 0]) or np.isnan(l[kb, j, 0]):
        ok = False
    if not ok:
      continue
    A = np.empty((m, 2 * ns))
    bvec = np.empty(m)
    for r in range(m):
      bvec[r] = kappa[k, known[r]]
    for j in range(ns):
      bx, by, bz = b[kb, j, 0], b[kb, j, 1], b[kb, j, 2]
      lx, ly, lz = l[kb, j, 0], l[kb, j, 1], l[kb, j, 2]
      half = (bx * lx + by * ly + bz * lz) / 2
      for r in range(m):
        idx = known[r]
        a, bcol = idx // 3, idx % 3
        la = lx if a == 0 else (ly if a == 1 else lz)
        bb = bx if bcol == 0 else (by if bcol == 1 else bz)
        val = la * bb - (half if a == bcol else 0.0)
        A[r, j] = val
        A[r, ns + j] = -val
    cvec = np.empty(2 * ns)
    for j in range(ns):
      cvec[j] = u[kb, j]
      cvec[ns + j] = u[kb, j]
    x = np.empty(2 * ns)
    if _centreProgram(A, bvec, cvec, tol, maxIter, x) != 0:
      continue
    total = 0.0
    for j in range(ns):
      rho[k, j] = x[j] - x[ns + j]
      total += abs(rho[k, j]) * u[kb, j]
    gnd[k] = total



# ------------------------------------------------------------------ the window filters

# the window filter of every measured pixel over the offsets of its window, within its own
# component: mode 0 the mean, 1 the componentwise median, 2 Kuwahara, the mean of the most
# homogeneous of the four quadrants quad marks per offset
def _windowLoop(u, comp, isData, lk, ij, offs, quad, mode, out):
  n = u.shape[0]
  no = offs.shape[0]
  for k in prange(n):
    for c in range(3):
      out[k, c] = u[k, c]
    if comp[k] == 0 or not isData[k]:
      continue
    buf = np.empty((no, 3))
    which = np.empty(no, dtype=np.int64)
    cnt = 0
    for o in range(no):
      nb = _neighbour(lk, ij, k, offs[o, 0], offs[o, 1])
      if nb < 0 or comp[nb] != comp[k] or not isData[nb]:
        continue
      for c in range(3):
        buf[cnt, c] = u[nb, c]
      which[cnt] = o
      cnt += 1
    if mode == 0:
      for c in range(3):
        s = 0.0
        for i in range(cnt):
          s += buf[i, c]
        out[k, c] = s / cnt
    elif mode == 1:
      col = np.empty(cnt)
      for c in range(3):
        for i in range(cnt):
          col[i] = buf[i, c]
        col.sort()
        out[k, c] = col[cnt // 2] if cnt % 2 == 1 else (col[cnt // 2 - 1] + col[cnt // 2]) / 2
    else:
      bestVar = np.inf
      for q in range(4):
        m = 0
        s0 = s1 = s2 = 0.0
        for i in range(cnt):
          if quad[q, which[i]]:
            m += 1
            s0 += buf[i, 0]
            s1 += buf[i, 1]
            s2 += buf[i, 2]
        if m == 0:
          continue
        mu0, mu1, mu2 = s0 / m, s1 / m, s2 / m
        v = 0.0
        for i in range(cnt):
          if quad[q, which[i]]:
            v += (buf[i, 0] - mu0) ** 2 + (buf[i, 1] - mu1) ** 2 + (buf[i, 2] - mu2) ** 2
        v /= m
        if v < bestVar:
          bestVar = v
          out[k, 0], out[k, 1], out[k, 2] = mu0, mu1, mu2


# ------------------------------------------------------------------ the graph solver

# y = A x for the operator of the variational filters over the graph: the data weights wd on
# the diagonal and alpha K^T W K, with K the difference along the edges and W the weight per
# edge (order 1), or K the Laplacian of the graph and W the weight per pixel (order 2). The
# edges at every pixel are listed in CSR form: ptr, the neighbour adj and the edge eid.
# Returns x . y over the free pixels, which the conjugate gradient needs of the same pass.
# Only the free pixels are evaluated, and for the Laplacian the active ones, the neighbours
# a free pixel reads; a component that has converged is neither and costs nothing
def _applyOperator(x, wd, wr, alpha, order, ptr, adj, eid, free, active, y, tmp2):
  n = x.shape[0]
  d = x.shape[1]
  dot = 0.0
  if order == 1:
    for i in prange(n):
      if not free[i]:
        continue
      for c in range(d):
        y[i, c] = wd[i] * x[i, c]
      for t in range(ptr[i], ptr[i + 1]):
        j = adj[t]
        w = alpha * wr[eid[t]]
        for c in range(d):
          y[i, c] += w * (x[i, c] - x[j, c])
      for c in range(d):
        dot += x[i, c] * y[i, c]
  else:
    for i in prange(n):
      if not active[i]:
        continue
      for c in range(d):
        tmp2[i, c] = x[i, c] * (ptr[i + 1] - ptr[i])
      for t in range(ptr[i], ptr[i + 1]):
        j = adj[t]
        for c in range(d):
          tmp2[i, c] -= x[j, c]
      for c in range(d):
        tmp2[i, c] *= wr[i]
    for i in prange(n):
      if not free[i]:
        continue
      for c in range(d):
        y[i, c] = wd[i] * x[i, c] + alpha * tmp2[i, c] * (ptr[i + 1] - ptr[i])
      for t in range(ptr[i], ptr[i + 1]):
        j = adj[t]
        for c in range(d):
          y[i, c] -= alpha * tmp2[j, c]
      for c in range(d):
        dot += x[i, c] * y[i, c]
  return dot


# the conjugate gradient on the operator above from the start x, the fixed pixels held at
# their values and taken off the free ones through the residual, Jacobi preconditioned, in
# three passes over the map per step; stops at a relative residual of tol or after maxIter
# steps. Returns the steps taken
def _cgLoop(x, b, wd, wr, alpha, order, ptr, adj, eid, free, active, tol, maxIter, work):
  n, d = x.shape
  y, tmp2, r, z, p = work[0], work[1], work[2], work[3], work[4]
  # the diagonal of the operator, for the preconditioner
  diag = np.empty(n)
  for i in prange(n):
    acc = 0.0
    if not free[i]:
      diag[i] = 1.0
      continue
    if order == 1:
      for t in range(ptr[i], ptr[i + 1]):
        acc += wr[eid[t]]
    else:
      deg = ptr[i + 1] - ptr[i]
      acc = deg * deg * wr[i]
      for t in range(ptr[i], ptr[i + 1]):
        acc += wr[adj[t]]
    diag[i] = wd[i] + alpha * acc
    if diag[i] <= 0:
      diag[i] = 1.0
  _applyOperator(x, wd, wr, alpha, order, ptr, adj, eid, free, active, y, tmp2)
  # the work arrays are zero where a pixel is not free, and only the free entries are ever
  # written, so a pixel that stops being free leaves nothing a neighbour still reads
  bnorm = 0.0
  rz = 0.0
  rnorm = 0.0
  for i in prange(n):
    if free[i]:
      for c in range(d):
        r[i, c] = b[i, c] - y[i, c]
        bnorm += b[i, c] * b[i, c]
        rnorm += r[i, c] * r[i, c]
        z[i, c] = r[i, c] / diag[i]
        rz += r[i, c] * z[i, c]
        p[i, c] = z[i, c]
  bnorm = np.sqrt(bnorm)
  if bnorm == 0.0:
    bnorm = 1.0
  it = 0
  while it < maxIter and np.sqrt(rnorm) > tol * bnorm:
    pAp = _applyOperator(p, wd, wr, alpha, order, ptr, adj, eid, free, active, y, tmp2)
    if pAp <= 0.0:
      break
    a = rz / pAp
    rzNew = 0.0
    rnorm = 0.0
    for i in prange(n):
      if free[i]:
        for c in range(d):
          x[i, c] += a * p[i, c]
          r[i, c] -= a * y[i, c]
          z[i, c] = r[i, c] / diag[i]
          rzNew += r[i, c] * z[i, c]
          rnorm += r[i, c] * r[i, c]
    beta = rzNew / rz if rz > 0 else 0.0
    rz = rzNew
    for i in prange(n):
      if free[i]:
        for c in range(d):
          p[i, c] = z[i, c] + beta * p[i, c]
    it += 1
  return it


# the weights of one reweighting step: per edge or pixel from the residual norm t, zero
# above the threshold and Huber's eps / sqrt(t^2 + eps^2) below it where the penalty is
# absolute; per pixel the data weight where the field is not pinned; and the right hand side
def _irlsWeightsLoop(t, x, f, isData, pin, l1Data, l1Reg, threshold, eps, edges, active, wr, wd, b):
  for e in prange(t.shape[0]):
    if edges.shape[0] == t.shape[0] and not active[edges[e, 0]]:
      continue
    if t.shape[0] == x.shape[0] and edges.shape[0] != t.shape[0] and not active[e]:
      continue
    if t[e] > threshold:
      wr[e] = 0.0
    elif l1Reg:
      wr[e] = eps / np.sqrt(t[e] * t[e] + eps * eps)
    else:
      wr[e] = 1.0
  n, d = x.shape
  for i in prange(n):
    if not active[i]:
      continue
    w = 0.0
    if (not pin) and isData[i]:
      if l1Data:
        td = 0.0
        for c in range(d):
          td += (x[i, c] - f[i, c]) ** 2
        w = eps / np.sqrt(td + eps * eps)
      else:
        w = 1.0
    wd[i] = w
    for c in range(d):
      b[i, c] = w * f[i, c]


# the norm of K x per edge (order 1) or per pixel (order 2), the residual the reweighting
# reads, for the active pixels and the edges between them; the rest is left as it was
def _residualNormLoop(x, order, edges, ptr, adj, active, out):
  d = x.shape[1]
  if order == 1:
    for e in prange(edges.shape[0]):
      a, b = edges[e, 0], edges[e, 1]
      if not active[a]:
        continue
      acc = 0.0
      for c in range(d):
        acc += (x[a, c] - x[b, c]) ** 2
      out[e] = np.sqrt(acc)
  else:
    for i in prange(x.shape[0]):
      if not active[i]:
        continue
      acc = 0.0
      for c in range(d):
        v = 0.0
        for t in range(ptr[i], ptr[i + 1]):
          v += x[i, c] - x[adj[t], c]
        acc += v * v
      out[i] = np.sqrt(acc)


# every pixel of a component as the left tangent vector at its reference, over the
# equivalent closest to it: u = log(closest(rot) conj(ref)), and the closest image itself
def _tangentLoop(rot, ref, indexed, phase, Gall, nG, img, u):
  n = rot.shape[0]
  for k in prange(n):
    if not indexed[k]:
      for c in range(3):
        u[k, c] = 0.0
      for c in range(4):
        img[k, c] = np.nan
      continue
    p = phase[k]
    q0, q1, q2, q3 = ref[k, 0], ref[k, 1], ref[k, 2], ref[k, 3]
    c0, c1, c2, c3 = _closestTo(rot[k, 0], rot[k, 1], rot[k, 2], rot[k, 3], q0, q1, q2, q3, Gall[p], nG[p])
    img[k, 0], img[k, 1], img[k, 2], img[k, 3] = c0, c1, c2, c3
    # conj(ref) c, the vector of the rotation from the reference to the pixel
    m0 = q0 * c0 + q1 * c1 + q2 * c2 + q3 * c3
    m1 = q0 * c1 - q1 * c0 - q2 * c3 + q3 * c2
    m2 = q0 * c2 + q1 * c3 - q2 * c0 - q3 * c1
    m3 = q0 * c3 - q1 * c2 + q2 * c1 - q3 * c0
    nrm = np.sqrt(m1 * m1 + m2 * m2 + m3 * m3)
    f = 0.0 if nrm < 1e-15 else (2 * np.arctan2(nrm, abs(m0)) / nrm) * (-1.0 if m0 < 0 else 1.0)
    u[k, 0], u[k, 1], u[k, 2] = f * m1, f * m2, f * m3


# the rotation ref exp(v) of every pixel: the reference turned by the tangent vector
def _expLoop(ref, v, out):
  n = v.shape[0]
  for k in prange(n):
    w = np.sqrt(v[k, 0] ** 2 + v[k, 1] ** 2 + v[k, 2] ** 2)
    if w < 1e-15:
      e0, e1, e2, e3 = 1.0, 0.0, 0.0, 0.0
    else:
      sc = np.sin(w / 2) / w
      e0, e1, e2, e3 = np.cos(w / 2), sc * v[k, 0], sc * v[k, 1], sc * v[k, 2]
    r0, r1, r2, r3 = ref[k, 0], ref[k, 1], ref[k, 2], ref[k, 3]
    out[k, 0] = r0 * e0 - r1 * e1 - r2 * e2 - r3 * e3
    out[k, 1] = r0 * e1 + r1 * e0 + r2 * e3 - r3 * e2
    out[k, 2] = r0 * e2 - r1 * e3 + r2 * e0 + r3 * e1
    out[k, 3] = r0 * e3 + r1 * e2 - r2 * e1 + r3 * e0


# ------------------------------------------------------------------ the compilation of these

# the grain of every pixel the closing left without a cell: the grain of the site nearest
# to each nudged corner of its footprint, searched over the lattice cells around the pixel,
# kept only when all corners agree; calcgrains.absorbInteriorPixels is the NumPy form
def _absorbLoop(todo, P, ij, corners, lookup, off0, off1, sitePos, siteGrain, reach, out):
  n1, n2 = lookup.shape
  for t in prange(len(todo)):
    k = todo[t]
    a0, b0 = ij[k, 0] - off0, ij[k, 1] - off1
    g = 0
    ok = True
    for c in range(corners.shape[0]):
      qx, qy = P[k, 0] + corners[c, 0], P[k, 1] + corners[c, 1]
      best, bestD = -1, np.inf
      for a in range(max(0, a0 - reach), min(n1, a0 + reach + 1)):
        for b in range(max(0, b0 - reach), min(n2, b0 + reach + 1)):
          s = lookup[a, b]
          if s >= 0:
            d = (sitePos[s, 0] - qx) ** 2 + (sitePos[s, 1] - qy) ** 2
            if d < bestD:
              best, bestD = s, d
      gs = 0 if best < 0 else siteGrain[best]
      if gs == 0 or (g != 0 and gs != g):
        ok = False
        break
      g = gs
    if ok:
      out[k] = g


_PIXEL_HELPERS = ("_range", "_closestTo", "_closestAngle", "_neighbour", "_comparable", "_pivot", "_reducedCosts", "_simplexSolve", "_linearProgram",
                  "_choleskySolve", "_centreProgram")
_PARALLEL_HELPERS = ("_applyOperator",)


# the helpers of the pixel and grain loops compiled and bound, and numba; None without numba
def _preparePixel():
  nb = _jit()
  if not nb:
    return None
  _bindHelpers(_PIXEL_HELPERS)
  _bindHelpers(_PARALLEL_HELPERS, parallel=True)
  return nb


# a loop of the pixels or the grains compiled once, after the helpers they call; None where the
# compiled loops do not run
def _pixelLoop(f, parallel, backend=None):
  if not _useNumba(backend):
    return None
  _preparePixel()
  return _compiledLoop(f, parallel=parallel, backend=backend)


# the curvature tensor of every pixel from the two derivatives in the plane, kappa = gx e1^T +
# gy e2^T, a column NaN where unknown marks it, each entry written once
def _planeTensorLoop(gx, gy, e1, e2, unknown, out):
  for k in prange(gx.shape[0]):
    for i in range(3):
      for j in range(3):
        out[k, i, j] = np.nan if unknown[j] else gx[k, i] * e1[j] + gy[k, i] * e2[j]


# the extremal widths of every hull, its vertices H[offsets[k]:offsets[k + 1]] in order: the
# narrowest over the normals of its edges, the widest between two of its vertices, each the
# first in the order of the edges and the points on a tie; an outline of one vertex keeps zeros
def _extremalWidthLoop(H, offsets, shortest, longest):
  for k in prange(len(offsets) - 1):
    a, L = offsets[k], offsets[k + 1] - offsets[k]
    best = np.inf
    for i in range(L):
      ex, ey = H[a + (i + 1) % L, 0] - H[a + i, 0], H[a + (i + 1) % L, 1] - H[a + i, 1]
      el = np.sqrt(ex * ex + ey * ey)
      if el == 0:
        continue
      nx, ny = -ey / el, ex / el
      lo, hi = np.inf, -np.inf
      for j in range(L):
        p = H[a + j, 0] * nx + H[a + j, 1] * ny
        lo, hi = min(lo, p), max(hi, p)
      if hi - lo < best:
        best = hi - lo
        shortest[k, 0], shortest[k, 1] = best * nx, best * ny
    far = -1.0
    for i in range(L):
      for j in range(L):
        dx, dy = H[a + j, 0] - H[a + i, 0], H[a + j, 1] - H[a + i, 1]
        d = dx * dx + dy * dy
        if d > far:
          far = d
          longest[k, 0], longest[k, 1] = dx, dy


_PIXEL_LOOPS = {"gradient": _gradientLoop, "kam": _kamLoop, "gnd": _gndLoop, "grod": _grodLoop, "window": _windowLoop, "cg": _cgLoop,
                "residualNorm": _residualNormLoop, "irlsWeights": _irlsWeightsLoop, "tangent": _tangentLoop, "exp": _expLoop,
                "absorb": _absorbLoop, "planeTensor": _planeTensorLoop, "extremalWidths": _extremalWidthLoop}


def pixelKernel(name, backend=None):
  """the compiled numba loop of the map's pixels named (gradient, kam, gnd, grod, window, ...),
  compiled on first call; None where the backend, else the preference 'kernels', is not numba or
  numba is missing"""
  return _pixelLoop(_PIXEL_LOOPS[name], True, backend)


# ------------------------------------------------------------------ the criterion and the grain table

# the angle criterion on every pair of neighbours: the largest |<q_j^-1 q_i, g>| over the
# proper group of their phase, the share of the phase's levels whose half cosine it exceeds;
# a pair of two phases is 0, a pair of a phase without orientations 1; criteria.gbcAngle's
# NumPy form is the oracle
def _angleCriterionLoop(rot, ph, i, j, Gall, nG, indexed, cosT, nLevels, out):
  for k in prange(len(i)):
    a, b = i[k], j[k]
    p = ph[a]
    if ph[b] != p:
      out[k] = 0.0
      continue
    if not indexed[p]:
      out[k] = 1.0
      continue
    a1, b1, c1, d1 = rot[b, 0], -rot[b, 1], -rot[b, 2], -rot[b, 3]
    a2, b2, c2, d2 = rot[a, 0], rot[a, 1], rot[a, 2], rot[a, 3]
    m0 = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    m1 = b1 * a2 + a1 * b2 - d1 * c2 + c1 * d2
    m2 = c1 * a2 + d1 * b2 + a1 * c2 - b1 * d2
    m3 = d1 * a2 - c1 * b2 + b1 * c2 + a1 * d2
    best = -1.0
    for g in range(nG[p]):
      d = abs(m0 * Gall[p, g, 0] + m1 * Gall[p, g, 1] + m2 * Gall[p, g, 2] + m3 * Gall[p, g, 3])
      if d > best:
        best = d
    cnt = 0
    for l in range(nLevels[p]):
      if best > cosT[p, l]:
        cnt += 1
    out[k] = cnt / nLevels[p]


# every pixel of a grain of an indexed phase moved to its equivalent closest to the reference
# of its grain, signed towards it, as geometry.kernels.closestImage with the identity on the
# left; with chord the chord angle to the reference in dev instead of the image in q
def _toReferenceLoop(rot, ph, gid, refs, Gall, nG, indexed, chord, q, dev):
  for k in prange(len(gid)):
    g = gid[k]
    p = ph[k]
    if g <= 0 or not indexed[p]:
      continue
    t0, t1, t2, t3 = refs[g, 0], refs[g, 1], refs[g, 2], refs[g, 3]
    c0, c1, c2, c3 = rot[k, 0], -rot[k, 1], -rot[k, 2], -rot[k, 3]
    m0 = c0 * t0 - c1 * t1 - c2 * t2 - c3 * t3
    m1 = c1 * t0 + c0 * t1 - c3 * t2 + c2 * t3
    m2 = c2 * t0 + c3 * t1 + c0 * t2 - c1 * t3
    m3 = c3 * t0 - c2 * t1 + c1 * t2 + c0 * t3
    best, br = -1.0, 0
    for r in range(nG[p]):
      d = abs(m0 * Gall[p, r, 0] + m1 * Gall[p, r, 1] + m2 * Gall[p, r, 2] + m3 * Gall[p, r, 3])
      if d > best:
        best, br = d, r
    b0, b1, b2, b3 = rot[k, 0], rot[k, 1], rot[k, 2], rot[k, 3]
    r0, r1, r2, r3 = Gall[p, br, 0], Gall[p, br, 1], Gall[p, br, 2], Gall[p, br, 3]
    p0 = b0 * r0 - b1 * r1 - b2 * r2 - b3 * r3
    p1 = b1 * r0 + b0 * r1 - b3 * r2 + b2 * r3
    p2 = b2 * r0 + b3 * r1 + b0 * r2 - b1 * r3
    p3 = b3 * r0 - b2 * r1 + b1 * r2 + b0 * r3
    if p0 * t0 + p1 * t1 + p2 * t2 + p3 * t3 < 0:
      p0, p1, p2, p3 = -p0, -p1, -p2, -p3
    if chord:
      h = np.sqrt((t0 - p0) ** 2 + (t1 - p1) ** 2 + (t2 - p2) ** 2 + (t3 - p3) ** 2)
      dev[k] = 4 * np.arcsin(min(h / 2, 1.0))
    else:
      q[k, 0], q[k, 1], q[k, 2], q[k, 3] = p0, p1, p2, p3


# the first measurement of every grain, the images summed per grain in the order of the
# measurements, and the chord angles summed likewise: the serial halves of the grain table
def _firstPixelLoop(gid, first):
  for k in range(len(gid)):
    if first[gid[k]] < 0:
      first[gid[k]] = k


def _sumByGrainLoop(gid, ph, indexed, x, acc):
  for k in range(len(gid)):
    g = gid[k]
    if g > 0 and indexed[ph[k]]:
      for c in range(x.shape[1]):
        acc[g, c] += x[k, c]


# the grains numbered in order of first appearance of their root over the measurements that
# have a cell, segment.segmentFrom's np.unique form the oracle
def _relabelLoop(raw, hasCell, label, grainId):
  cnt = 0
  for k in range(len(raw)):
    if hasCell[k]:
      r = raw[k]
      if label[r] == 0:
        cnt += 1
        label[r] = cnt
      grainId[k] = label[r]
  return cnt


# the coarse seeds of one FMC step, in the given order over the columns of a CSC coupling
# matrix: a node becomes a seed when its coupling to the seeds before it stays below its
# share alphaSum, fmc.coarseSeeds's loop the fallback
def _fmcSeedsLoop(order, indptr, indices, data, alphaSum, seed):
  for k in order:
    s = 0.0
    for e in range(indptr[k], indptr[k + 1]):
      if seed[indices[e]]:
        s += data[e]
    seed[k] = s < alphaSum[k]


# the sums over the edges of every loop into its outline's row of out: twice the area, the
# length of the outer loop, the length of all loops, the first moments times six and the
# second times twelve, twelve and twenty four, the edge list of polygon.loopEdges its oracle
def _loopSumsLoop(V, flat, start, length, owner, first, out):
  for l in range(start.shape[0]):
    k, s, n = owner[l], start[l], length[l]
    for i in range(n):
      a, b = flat[s + i], flat[s + (i + 1) % n]
      px, py, qx, qy = V[a, 0], V[a, 1], V[b, 0], V[b, 1]
      w = px * qy - qx * py
      d = np.sqrt((qx - px) ** 2 + (qy - py) ** 2)
      out[k, 0] += w
      if first[l]:
        out[k, 1] += d
      out[k, 2] += d
      out[k, 3] += (px + qx) * w
      out[k, 4] += (py + qy) * w
      out[k, 5] += (py * py + py * qy + qy * qy) * w
      out[k, 6] += (px * px + px * qx + qx * qx) * w
      out[k, 7] += (px * qy + 2 * px * py + 2 * qx * qy + qx * py) * w


# the widths of the point runs H[offsets[k]:offsets[k + 1]] seen from every direction of D,
# summed over the runs: a direction is one pass over the points, no projection is stored
def _projectionLengthLoop(H, offsets, D, out):
  for w in prange(D.shape[0]):
    dx, dy = D[w, 0], D[w, 1]
    acc = 0.0
    for k in range(offsets.shape[0] - 1):
      s, e = offsets[k], offsets[k + 1]
      if e > s:
        lo = hi = H[s, 0] * dx + H[s, 1] * dy
        for i in range(s + 1, e):
          p = H[i, 0] * dx + H[i, 1] * dy
          if p < lo:
            lo = p
          elif p > hi:
            hi = p
        acc += hi - lo
    out[w] = acc


_GRAIN_LOOPS = {"angleCriterion": (_angleCriterionLoop, True), "toReference": (_toReferenceLoop, True),
                "firstPixel": (_firstPixelLoop, False), "sumByGrain": (_sumByGrainLoop, False), "relabel": (_relabelLoop, False),
                "fmcSeeds": (_fmcSeedsLoop, False), "loopSums": (_loopSumsLoop, False), "projectionLength": (_projectionLengthLoop, True)}


def grainKernel(name, backend=None):
  """the compiled numba loop of the segmentation and the grain table named (angleCriterion,
  sumByGrain, relabel, ...), compiled on first call; None where the backend, else the preference
  'kernels', is not numba or numba is missing"""
  return _pixelLoop(*_GRAIN_LOOPS[name], backend)


# ------------------------------------------------------------------ the grid of a scan

# the lattice index of every position of a scan written row by row, counts[r] positions in row r
# with x running fastest, counted from the least one, and the largest distance of a position from
# its site under the basis (a00, a01; 0, a11) from the first one; a hexagonal row r starts r // 2
# steps back along a1. Returns the distance and the least index, the first position's negated
def _rowOrderLoop(x, y, counts, hexagonal, sx, sy, a00, a01, a11, ij):
  x0, y0 = x[0], y[0]
  k = 0
  resid = 0.0
  i0, j0 = 0, 0
  for r in range(counts.shape[0]):
    for c in range(counts[r]):
      i = c - r // 2 if hexagonal else c
      j = r
      if sx < 0:
        i = -i - (j if hexagonal else 0)
      if sy < 0:
        j = -j
      ij[k, 0] = i
      ij[k, 1] = j
      dx = abs(x[k] - (x0 + a00 * i + a01 * j))
      dy = abs(y[k] - (y0 + a11 * j))
      if dx > resid:
        resid = dx
      if dy > resid:
        resid = dy
      if i < i0:
        i0 = i
      if j < j0:
        j0 = j
      k += 1
  for k in range(ij.shape[0]):
    ij[k, 0] -= i0
    ij[k, 1] -= j0
  return resid, i0, j0


# the column and the row of every lattice index under the scan's inverse basis (b00, b01; b10,
# b11) and row shift s, rows counted from the least; returns the least row, the least column and
# the largest column and row
def _scanCellsLoop(ij, b00, b01, b10, b11, s, cells):
  n = ij.shape[0]
  r0 = 1 << 62
  for k in range(n):
    r = b10 * ij[k, 0] + b11 * ij[k, 1]
    if r < r0:
      r0 = r
  c0, c1, r1 = 1 << 62, -(1 << 62), -(1 << 62)
  for k in range(n):
    m = b00 * ij[k, 0] + b01 * ij[k, 1]
    r = b10 * ij[k, 0] + b11 * ij[k, 1]
    c = m + np.int64(np.floor((r - r0) * s + 0.25))
    cells[k, 0] = c
    cells[k, 1] = r
    if c < c0:
      c0 = c
    if c > c1:
      c1 = c
    if r > r1:
      r1 = r
  return r0, c0, c1, r1


# the lattice index of every cell of a (rows, columns) array from the row r0 and the column c0 on,
# under the scan's basis (b00, b01; b10, b11) and row shift s; returns the least index
def _scanSitesLoop(nr, nc, r0, c0, b00, b01, b10, b11, s, ij):
  i0, j0 = 1 << 62, 1 << 62
  k = 0
  for a in range(nr):
    r = r0 + a
    shift = np.int64(np.floor(a * s + 0.25))
    for b in range(nc):
      m = c0 + b - shift
      i = b00 * m + b01 * r
      j = b10 * m + b11 * r
      ij[k, 0] = i
      ij[k, 1] = j
      if i < i0:
        i0 = i
      if j < j0:
        j0 = j
      k += 1
  return i0, j0


# ------------------------------------------------------------------ the expansion of the clustering

# rows a:b of the Markov clustering step A A, every entry raised to the power p and dropped at
# or below cut: the products summed in a dense accumulator in the order of the rows of A, the
# columns of a row written in the reverse order they were first touched, which is the order
# and the sums of scipy's product; the buffers grow by doubling
def _mclExpandLoop(ptr, idx, val, a, b, p, cut):
  n = ptr.shape[0] - 1
  acc = np.zeros(n)
  seen = np.full(n, -1, dtype=np.int64)
  cols = np.empty(n, dtype=np.int64)
  rowNnz = np.zeros(b - a, dtype=np.int64)
  cap = max(16, 4 * (ptr[b] - ptr[a]))
  outIdx = np.empty(cap, dtype=np.int32)
  outVal = np.empty(cap)
  m = 0
  for i in range(a, b):
    length = 0
    for jj in range(ptr[i], ptr[i + 1]):
      j, v = idx[jj], val[jj]
      for kk in range(ptr[j], ptr[j + 1]):
        k = idx[kk]
        if seen[k] != i:
          seen[k] = i
          cols[length] = k
          length += 1
          acc[k] = 0.0
        acc[k] += v * val[kk]
    if m + length > cap:
      cap = max(2 * cap, m + length)
      grownIdx, grownVal = np.empty(cap, dtype=np.int32), np.empty(cap)
      grownIdx[:m], grownVal[:m] = outIdx[:m], outVal[:m]
      outIdx, outVal = grownIdx, grownVal
    start = m
    for t in range(length - 1, -1, -1):
      k = cols[t]
      w = acc[k] ** p
      if w > cut and w != 0.0:
        outIdx[m] = k
        outVal[m] = w
        m += 1
    rowNnz[i - a] = m - start
  return rowNnz, outIdx[:m], outVal[:m]


def mclExpandNumba(A, p, cut):
  """the expansion and inflation of a Markov clustering step, A A with every entry raised to the
  power p and those at or below cut dropped, as a CSR matrix over the threads of the preference;
  None without the compiled loops"""
  from scipy import sparse

  from ..geometry.kernels import ROW_COST, _overRows
  loop = _compiledLoop(_mclExpandLoop, nogil=True)
  if loop is None:
    return None
  A = A.tocsr()
  n = A.shape[0]
  ptr = np.ascontiguousarray(A.indptr, dtype=np.int64)
  idx = np.ascontiguousarray(A.indices, dtype=np.int64)
  val = np.ascontiguousarray(A.data, dtype=np.float64)
  parts = {}

  # the rows a to b expanded on one thread
  def expandRows(a, b):
    parts[a] = loop(ptr, idx, val, a, b, float(p), float(cut))

  _overRows(n, n * ROW_COST + A.nnz * ROW_COST, expandRows)
  parts = [parts[a] for a in sorted(parts)]
  rowNnz = np.concatenate([q[0] for q in parts])
  indptr = np.concatenate([[0], np.cumsum(rowNnz)])
  return sparse.csr_matrix((np.concatenate([q[2] for q in parts]), np.concatenate([q[1] for q in parts]), indptr), shape=A.shape)


_GRID_LOOPS = {"rowOrder": _rowOrderLoop, "scanCells": _scanCellsLoop, "scanSites": _scanSitesLoop}


def gridKernel(name, backend=None):
  """the compiled numba loop of the grid of a scan named (rowOrder, scanCells, scanSites), compiled
  on first call; None where the backend, else the preference 'kernels', is not numba or numba is
  missing"""
  return _compiledLoop(_GRID_LOOPS[name], backend=backend)


__all__ = []
