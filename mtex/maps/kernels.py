# the compiled kernels of the maps layer: the local Voronoi clipping of the decomposition
# as one numba loop over chunks of sites, each chunk writing its faces into its own slice
# of a buffer so that the chunks run in parallel without a shared counter; the union-find
# of the segmentation; the tracing of the grain loops; the fused gradient and kernel
# average misorientation over the lattice; and the dense simplex of the dislocation fit.
# The NumPy forms in decomposition.py, grains.py, intragranular.py and dislocations.py are
# the fallbacks without numba and the oracles of the tests.
from __future__ import annotations

import numpy as np

from ..geometry.kernels import _jit

prange = range
_COMPILED = {}
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


# the component root of every node of a graph on n nodes, or None without numba
def unionFind(n, a, b):
  nb = _jit()
  if not nb:
    return None
  if "unionFind" not in _COMPILED:
    _COMPILED["unionFind"] = nb.njit(cache=True)(_unionFindLoop)
  parent = np.arange(n, dtype=np.int32)
  size = np.ones(n, dtype=np.int32)
  _COMPILED["unionFind"](parent, size, np.ascontiguousarray(a, dtype=np.int32), np.ascontiguousarray(b, dtype=np.int32))
  return parent


# ------------------------------------------------------------------ compilation

_HELPERS = ("_clip", "_cellOf")


def _compile():
  nb = _jit()
  if not nb:
    return None
  if "localVoronoi" not in _COMPILED:
    globals()["prange"] = nb.prange
    for name in _HELPERS:
      _COMPILED[name] = nb.njit(cache=True)(globals()[name])
      globals()[name] = _COMPILED[name]
    _COMPILED["localVoronoi"] = nb.njit(cache=True, parallel=True)(_localVoronoiLoop)
  return _COMPILED["localVoronoi"]


# the faces of every cell through the compiled loop, or None without numba
def localVoronoiNumba(sites, levels, distortTol=0.05, chunkSize=4096):
  loop = _compile()
  if loop is None:
    return None
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
  keep = np.concatenate([np.arange(t * cap, t * cap + counts[t]) for t in range(nChunks)])
  return cellsBuf.reshape(-1, 2)[keep], vBuf.reshape(-1, 4)[keep], int(unbounded.sum())



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
# on the left. The vertices of the loops go into loopV, the start of every loop into loff,
# its grain into lgrain and twice its signed area into larea; a loop of fewer than three
# vertices is dropped. Returns the count of loops and of vertices
def _traceLoopsLoop(V, ea, eb, goff, loopV, loff, lgrain, larea):
  nl = 0
  nv = 0
  for g in range(goff.shape[0] - 1):
    s, e = goff[g], goff[g + 1]
    m = e - s
    if m == 0:
      continue
    order = np.argsort(ea[s:e]) + s
    tails = ea[order]
    used = np.zeros(m, dtype=np.bool_)
    for st in range(m):
      if used[st]:
        continue
      k = s + st
      start = nv
      area = 0.0
      while True:
        used[k - s] = True
        loopV[nv] = ea[k]
        nv += 1
        area += V[ea[k], 0] * V[eb[k], 1] - V[eb[k], 0] * V[ea[k], 1]
        lo, hi = _range(tails, eb[k])
        nxt = -1
        if hi - lo == 1:
          c = order[lo]
          if not used[c - s]:
            nxt = c
        elif hi - lo > 1:
          ax, ay = V[eb[k], 0] - V[ea[k], 0], V[eb[k], 1] - V[ea[k], 1]
          best = -np.inf
          for t in range(lo, hi):
            c = order[t]
            if used[c - s]:
              continue
            bx, by = V[eb[c], 0] - V[ea[c], 0], V[eb[c], 1] - V[ea[c], 1]
            turn = np.arctan2(ax * by - ay * bx, ax * bx + ay * by)
            if turn > best:
              best, nxt = turn, c
        if nxt < 0:
          break
        k = nxt
      if nv - start >= 3:
        loff[nl] = start
        lgrain[nl] = g
        larea[nl] = area
        nl += 1
      else:
        nv = start
  loff[nl] = nv
  return nl, nv


# the loops of every grain through the compiled walk: the flat vertices, the loop offsets,
# the grain and twice the signed area of every loop; None without numba
def traceLoopsNumba(V, ea, eb, goff):
  loop = _compileLoops()
  if loop is None:
    return None
  n = len(ea)
  loopV = np.empty(n, dtype=np.int32)
  loff = np.empty(n + 1, dtype=np.int64)
  lgrain = np.empty(n, dtype=np.int64)
  larea = np.empty(n)
  nl, nv = loop(np.ascontiguousarray(V, dtype=np.float64), np.ascontiguousarray(ea, dtype=np.int32), np.ascontiguousarray(eb, dtype=np.int32),
                np.ascontiguousarray(goff, dtype=np.int64), loopV, loff, lgrain, larea)
  return loopV[:nv], loff[:nl + 1], lgrain[:nl], larea[:nl]


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
        c0, c1, c2, c3 = _closestTo(rot[nb, 0], rot[nb, 1], rot[nb, 2], rot[nb, 3], q0, q1, q2, q3, Gall[p], nG[p])
        h = np.sqrt((q0 - c0) ** 2 + (q1 - c1) ** 2 + (q2 - c2) ** 2 + (q3 - c3) ** 2)
        ang = 4 * np.arcsin(min(h / 2, 1.0))
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
def _simplexSolve(T, basis, ncols, tol, maxit):
  m = basis.shape[0]
  last = T.shape[0] - 1
  rhs = T.shape[1] - 1
  for _ in range(maxit):
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
def _linearProgram(A, b, c, tol, maxit, x):
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
  status = _simplexSolve(T, basis, n + m, tol, maxit)
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
  status = _simplexSolve(T, basis, n, tol, maxit)
  if status != 0:
    return 5
  for j in range(n):
    x[j] = 0.0
  for i in range(m):
    if alive[i] and basis[i] < n:
      x[basis[i]] = max(T[i, n + m], 0.0)
  return 0


# the dislocation fit of every pixel: the systems in the frame of the curvature, one set for
# all pixels or one per pixel, the curvature of each system on the known components as the
# columns of the program, both signs of every system, the line energies as the cost; a pixel
# with an unknown curvature or systems is left NaN
def _gndLoop(kappa, known, b, l, u, tol, maxit, gnd, rho):
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
    if _linearProgram(A, bvec, cvec, tol, maxit, x) != 0:
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
# three passes over the map per step; stops at a relative residual of tol or after maxit
# steps. Returns the steps taken
def _cgLoop(x, b, wd, wr, alpha, order, ptr, adj, eid, free, active, tol, maxit, work):
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
  while it < maxit and np.sqrt(rnorm) > tol * bnorm:
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


_PIXEL_HELPERS = ("_range", "_closestTo", "_neighbour", "_comparable", "_pivot", "_reducedCosts", "_simplexSolve", "_linearProgram")
_PARALLEL_HELPERS = ("_applyOperator",)


def _preparePixel():
  nb = _jit()
  if not nb:
    return None
  if "_pixelHelpers" not in _COMPILED:
    globals()["prange"] = nb.prange
    for name in _PIXEL_HELPERS:
      _COMPILED[name] = nb.njit(cache=True)(globals()[name])
      globals()[name] = _COMPILED[name]
    for name in _PARALLEL_HELPERS:
      _COMPILED[name] = nb.njit(cache=True, parallel=True)(globals()[name])
      globals()[name] = _COMPILED[name]
    _COMPILED["_pixelHelpers"] = True
  return nb


def _compileLoops():
  nb = _preparePixel()
  if nb is None:
    return None
  if "traceLoops" not in _COMPILED:
    _COMPILED["traceLoops"] = nb.njit(cache=True)(_traceLoopsLoop)
  return _COMPILED["traceLoops"]


def pixelKernel(name):
  nb = _preparePixel()
  if nb is None:
    return None
  if name not in _COMPILED:
    _COMPILED[name] = nb.njit(cache=True, parallel=True)({"gradient": _gradientLoop, "kam": _kamLoop, "gnd": _gndLoop, "grod": _grodLoop, "window": _windowLoop, "cg": _cgLoop, "residualNorm": _residualNormLoop, "irlsWeights": _irlsWeightsLoop, "tangent": _tangentLoop, "exp": _expLoop, "absorb": _absorbLoop}[name])
  return _COMPILED[name]


# the linear program of one problem through the compiled simplex, None without numba, and
# None again where the program has no solution
def linearProgramNumba(c, A, b, tol=1e-9, maxit=10000):
  if _preparePixel() is None:
    return None
  x = np.empty(len(c))
  status = _linearProgram(np.ascontiguousarray(A, dtype=float), np.ascontiguousarray(b, dtype=float), np.ascontiguousarray(c, dtype=float), tol, maxit, x)
  return x if status == 0 else None


__all__ = []
