# the plane geometry of a partition given by its boundary: point location, the overlay of two
# partitions, distances between their boundaries
#
# A partition is an array of rows (x1, y1, x2, y2, left, right), one per boundary segment, the
# region `left` on the left of the walk from (x1, y1) to (x2, y2) and 0 outside; the truth and
# every reconstruction are written in this form, so all scores are computed by the code below.
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

EPS = 1e-7


def lengths(S):
  return np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1])


def subdivide(S, h):
  """every segment cut into equal pieces no longer than h"""
  n = np.maximum(1, np.ceil(lengths(S) / h).astype(np.int64))
  k = np.repeat(np.arange(len(S)), n)
  t0 = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) / n[k]
  t1 = t0 + 1 / n[k]
  P, D = S[k, :2], S[k, 2:4] - S[k, :2]
  return np.column_stack([P + t0[:, None] * D, P + t1[:, None] * D, S[k, 4:]])


def locate(S, xy, h=1.0):
  """the region at every point: the first segment a ray towards +x crosses names it, by its left
  region where it runs upwards and its right one where it runs downwards; 0 where none is crossed"""
  xy = np.asarray(xy, float).reshape(-1, 2)
  out = np.zeros(len(xy), np.int64)
  S = S[S[:, 1] != S[:, 3]]
  if not len(S) or not len(xy):
    return out
  # the query heights off every vertex height of a lattice, so that no ray runs through a vertex
  X, Y = xy[:, 0], xy[:, 1] + np.pi * 1e-9
  lo, hi = np.minimum(S[:, 1], S[:, 3]), np.maximum(S[:, 1], S[:, 3])
  # the segments by horizontal strips of height h
  y0 = min(lo.min(), Y.min())
  s0, s1 = ((lo - y0) // h).astype(np.int64), ((hi - y0) // h).astype(np.int64)
  n = s1 - s0 + 1
  seg = np.repeat(np.arange(len(S)), n)
  strip = np.repeat(s0, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
  order = np.argsort(strip, kind="stable")
  seg, strip = seg[order], strip[order]
  qs = ((Y - y0) // h).astype(np.int64)
  qorder = np.argsort(qs, kind="stable")
  bounds = np.flatnonzero(np.diff(qs[qorder])) + 1
  for q in np.split(qorder, bounds):
    c = seg[np.searchsorted(strip, qs[q[0]]):np.searchsorted(strip, qs[q[0]], side="right")]
    if not len(c):
      continue
    for qq in np.array_split(q, max(1, len(q) * len(c) // 2_000_000)):
      a, b = S[c, 1], S[c, 3]
      y = Y[qq, None]
      cross = (a > y) != (b > y)
      xc = S[c, 0] + (y - a) * (S[c, 2] - S[c, 0]) / (b - a)
      xc = np.where(cross & (xc > X[qq, None]), xc, np.inf)
      k = np.argmin(xc, axis=1)
      hit = np.isfinite(xc[np.arange(len(qq)), k])
      up = b[k] > a[k]
      out[qq[hit]] = np.where(up, S[c[k], 4], S[c[k], 5])[hit]
  return out


def _splits(T, S, tol=1e-9):
  """the parameters at which the segments of T and of S meet one another"""
  mT, mS = (T[:, :2] + T[:, 2:4]) / 2, (S[:, :2] + S[:, 2:4]) / 2
  r = (lengths(T).max() + lengths(S).max()) / 2 + tol
  pairs = cKDTree(mT).sparse_distance_matrix(cKDTree(mS), r, output_type="ndarray")
  i, j = pairs["i"], pairs["j"]
  p, rr = T[i, :2], T[i, 2:4] - T[i, :2]
  q, ss = S[j, :2], S[j, 2:4] - S[j, :2]
  cr = lambda u, v: u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
  den, qp = cr(rr, ss), q - p
  Lr, Ls = np.hypot(*rr.T), np.hypot(*ss.T)
  par = np.abs(den) <= tol * Lr * Ls
  with np.errstate(divide="ignore", invalid="ignore"):
    t, u = cr(qp, ss) / den, cr(qp, rr) / den
  meet = ~par & (t >= -tol) & (t <= 1 + tol) & (u >= -tol) & (u <= 1 + tol)
  ti, tv, sj, sv = [i[meet]], [t[meet]], [j[meet]], [u[meet]]
  # collinear overlaps: the ends of each projected onto the other
  col = par & (np.abs(cr(qp, rr)) <= tol * Lr * np.maximum(1, np.abs(qp).max(axis=1)))
  proj = lambda e, P, R: np.einsum("ij,ij->i", e - P, R) / np.einsum("ij,ij->i", R, R)
  for e in (q[col], (q + ss)[col]):
    ti.append(i[col]); tv.append(proj(e, p[col], rr[col]))
  for e in (p[col], (p + rr)[col]):
    sj.append(j[col]); sv.append(proj(e, q[col], ss[col]))
  return (np.concatenate(ti), np.concatenate(tv)), (np.concatenate(sj), np.concatenate(sv))


def _cut(S, idx, t, tol=1e-9):
  """the segments of S cut at the parameters t of segments idx"""
  ok = (t > tol) & (t < 1 - tol)
  idx = np.concatenate([idx[ok], np.arange(len(S)), np.arange(len(S))])
  t = np.concatenate([t[ok], np.zeros(len(S)), np.ones(len(S))])
  o = np.lexsort((t, idx))
  idx, t = idx[o], t[o]
  same = (idx[1:] == idx[:-1]) & (t[1:] - t[:-1] > tol)
  k, a, b = idx[:-1][same], t[:-1][same], t[1:][same]
  P, D = S[k, :2], S[k, 2:4] - S[k, :2]
  return np.column_stack([P + a[:, None] * D, P + b[:, None] * D, S[k, 4:6]])


def _sides(S, other, eps=EPS):
  """the regions of the other partition just left and right of the middle of every segment"""
  m = (S[:, :2] + S[:, 2:4]) / 2
  d = S[:, 2:4] - S[:, :2]
  n = np.column_stack([-d[:, 1], d[:, 0]]) / np.hypot(*d.T)[:, None]
  return locate(other, m + eps * n), locate(other, m - eps * n)


def overlay(T, S, h=1.0):
  """the area of every intersection of a region of T and one of S, as a sparse matrix indexed by
  their labels, and the pieces both boundaries are cut into, each with the regions of the other
  partition on its two sides. Green's theorem: the boundary of T_i & S_j is the part of the
  boundary of T_i inside S_j and the part of the boundary of S_j inside T_i, where the two run
  along one another counted once, from T"""
  T, S = subdivide(T[lengths(T) > 0], h), subdivide(S[lengths(S) > 0], h)
  (ti, tv), (sj, sv) = _splits(T, S)
  T, S = _cut(T, ti, tv), _cut(S, sj, sv)
  rL, rR = _sides(T, S)
  tL, tR = _sides(S, T)
  # the integral of x dy along every piece
  cT = (T[:, 0] + T[:, 2]) / 2 * (T[:, 3] - T[:, 1])
  cS = (S[:, 0] + S[:, 2]) / 2 * (S[:, 3] - S[:, 1])
  free = tL == tR
  i = np.concatenate([T[:, 4], T[:, 5], tL[free], tL[free]]).astype(np.int64)
  j = np.concatenate([rL, rR, S[free, 4], S[free, 5]]).astype(np.int64)
  v = np.concatenate([cT, -cT, cS[free], -cS[free]])
  shape = (int(max(T[:, 4:6].max(), 0)) + 1, int(max(S[:, 4:6].max(), 0)) + 1)
  A = coo_matrix((v, (i, j)), shape=shape).tocsr()
  return A, np.column_stack([T, rL, rR]), np.column_stack([S, tL, tR])


def areas(S):
  """the area of every region, from the boundary alone"""
  c = (S[:, 0] + S[:, 2]) / 2 * (S[:, 3] - S[:, 1])
  n = int(S[:, 4:6].max()) + 1
  return np.bincount(S[:, 4].astype(np.int64), c, n) - np.bincount(S[:, 5].astype(np.int64), c, n)


def nearest(S, xy, u=None, h=0.25):
  """the distance from every point to the nearest segment of S; with unit directions u, also the
  angle in degrees between u and the nearest segment, the smallest where several are nearest"""
  S = subdivide(S[lengths(S) > 0], h)
  xy = np.asarray(xy, float).reshape(-1, 2)
  tree = cKDTree((S[:, :2] + S[:, 2:4]) / 2)
  dist, ang = np.full(len(xy), np.inf), np.zeros(len(xy))
  todo, k = np.arange(len(xy)), 8
  while len(todo):
    k = min(k, len(S))
    dm, c = tree.query(xy[todo], k)
    dm, c = dm.reshape(len(todo), -1), c.reshape(len(todo), -1)
    P, D = S[c, :2], S[c, 2:4] - S[c, :2]
    t = np.clip(np.einsum("ijk,ijk->ij", xy[todo, None] - P, D) / np.maximum(np.einsum("ijk,ijk->ij", D, D), 1e-300), 0, 1)
    d = np.hypot(*(P + t[..., None] * D - xy[todo, None]).transpose(2, 0, 1))
    dist[todo] = d.min(axis=1)
    if u is not None:
      cos = np.abs(np.einsum("ijk,ik->ij", D, u[todo])) / np.hypot(*D.transpose(2, 0, 1))
      cos = np.where(d <= dist[todo, None] + 1e-9, cos, 0)
      ang[todo] = np.degrees(np.arccos(np.clip(cos.max(axis=1), 0, 1)))
    # the nearest segment's middle lies within h/2 of its nearest point, so it is among the
    # candidates once the farthest of them is further than that
    todo = todo[(dm[:, -1] < dist[todo] + h / 2) & (k < len(S))]
    k *= 4
  return dist, ang


def sample(S, h=1 / 32):
  """the points and weights of Simpson's rule along every segment, on pieces at most h long, and
  the unit direction of the segment every point lies on"""
  S = S[lengths(S) > 0]
  L = lengths(S)
  n = np.ceil(L / h).astype(np.int64)
  k = np.repeat(np.arange(len(S)), 2 * n + 1)
  r = np.arange(len(k)) - np.repeat(np.cumsum(2 * n + 1) - 2 * n - 1, 2 * n + 1)
  w = np.where((r == 0) | (r == 2 * n[k]), 1, np.where(r % 2, 4, 2)) * L[k] / (6 * n[k])
  d = S[k, 2:4] - S[k, :2]
  return S[k, :2] + (r / (2 * n[k]))[:, None] * d, w, d / L[k, None]


def clipToBox(S, box, labelAt):
  """the segments clipped to the box (x0, x1, y0, y1), and the box's own edges between the points
  where they end on it, the region inside on their left and 0 outside"""
  x0, x1, y0, y1 = box
  P, D = S[:, :2], S[:, 2:4] - S[:, :2]
  t0, t1 = np.zeros(len(S)), np.ones(len(S))
  # Liang-Barsky: every side of the box bounds the parameter range of every segment
  for p, q in ((-D[:, 0], P[:, 0] - x0), (D[:, 0], x1 - P[:, 0]), (-D[:, 1], P[:, 1] - y0), (D[:, 1], y1 - P[:, 1])):
    with np.errstate(divide="ignore", invalid="ignore"):
      r = q / p
    t0 = np.where(p < 0, np.maximum(t0, r), t0)
    t1 = np.where(p > 0, np.minimum(t1, r), t1)
    t1 = np.where((p == 0) & (q < 0), -1, t1)
  keep = t1 > t0
  C = np.column_stack([P + t0[:, None] * D, P + t1[:, None] * D, S[:, 4:6]])[keep]
  # the box walked anticlockwise, parametrised by the arc length from (x0, y0)
  w, hgt = x1 - x0, y1 - y0
  corner = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
  cum = np.array([0, w, w + hgt, 2 * w + hgt, 2 * (w + hgt)])
  def arc(p):
    x, y, tol = p[:, 0], p[:, 1], 1e-9 * max(w, hgt)
    return np.select([np.abs(y - y0) < tol, np.abs(x - x1) < tol, np.abs(y - y1) < tol, np.abs(x - x0) < tol],
                     [x - x0, w + y - y0, w + hgt + x1 - x, 2 * w + hgt + y1 - y], np.nan)
  def point(s):
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, 3)
    return corner[k] + (s - cum[k])[:, None] * np.sign(corner[k + 1] - corner[k])
  s = np.concatenate([arc(C[:, :2]), arc(C[:, 2:4]), cum])
  s = np.unique(s[np.isfinite(s)])
  a, b = point(s[:-1]), point(s[1:])
  mid = (a + b) / 2
  inward = np.column_stack([-(b - a)[:, 1], (b - a)[:, 0]])
  inward /= np.hypot(*inward.T)[:, None]
  lab = labelAt(mid + 1e-6 * inward)
  return np.vstack([C, np.column_stack([a, b, lab, np.zeros(len(a))])])


def junctions(S):
  """the points where three or more regions meet"""
  P = np.vstack([S[:, :2], S[:, 2:4], S[:, :2], S[:, 2:4]])
  lab = np.concatenate([S[:, 4], S[:, 4], S[:, 5], S[:, 5]])
  V, v = np.unique(P, axis=0, return_inverse=True)
  vl = np.unique(np.column_stack([v.ravel(), lab])[lab > 0], axis=0)
  return V[np.bincount(vl[:, 0].astype(np.int64), minlength=len(V)) >= 3]
