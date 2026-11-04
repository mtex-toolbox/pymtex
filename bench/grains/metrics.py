# the scores of a reconstruction against the truth of its instance (WP2 of the research plan of
# ../master/docs/grain-reconstruction-research-plan.md): the partition on the pixels, the geometry of
# the boundary on the grains truth and reconstruction share
#
# A reconstruction is the grain id of every pixel and its boundary in the form of planar.py. Every
# length is in pixel steps, every area in square steps, every angle in degrees.
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

import planar as pl

BINS = (0, 2, 3, 5, 10, 15, 180)


def contingency(a, b):
  """the overlap counts of two labellings as a sparse matrix over their compressed labels"""
  ua, ia = np.unique(a, return_inverse=True)
  ub, ib = np.unique(b, return_inverse=True)
  return coo_matrix((np.ones(len(a)), (ia, ib)), shape=(len(ua), len(ub))).tocsr(), ua, ub


def pairStats(n, ni, nj):
  """ARI and variation of information (bits) of a contingency table given as its nonzero counts and margins"""
  c2 = lambda x: (x * (x - 1) / 2).sum()
  N = ni.sum()
  si, sj = c2(ni), c2(nj)
  e = si * sj / c2(np.array([N]))
  ari = 1.0 if si + sj == 2 * e else (c2(n) - e) / ((si + sj) / 2 - e)
  H = lambda x: -(x / N * np.log2(x / N)).sum()
  return ari, 2 * H(n) - H(ni) - H(nj)


def matching(n, ni, nj, rows, cols):
  """F1 of the grain matching by IoU at 0.5 and 0.75, the splits and the merges"""
  iou = n / (ni[rows] + nj[cols] - n)
  f1 = {t: 2 * (iou > t).sum() / (len(ni) + len(nj)) for t in (0.5, 0.75)}
  # a reconstructed grain lying mostly in one true grain is a piece of it, and the other way round
  splits = np.bincount(rows[n >= nj[cols] / 2], minlength=len(ni))
  merges = np.bincount(cols[n >= ni[rows] / 2], minlength=len(nj))
  return f1, int(np.maximum(splits - 1, 0).sum()), int(np.maximum(merges - 1, 0).sum())


def misorientationAngles(grainQ, pairs):
  if not len(pairs):
    return np.zeros(0)
  from mtex import angle, crystalFrame, degree, orientation, rotation
  cs = crystalFrame("m-3m", mineral="Iron")
  o1, o2 = (orientation(rotation(grainQ[pairs[:, k]]), cs) for k in (0, 1))
  return np.asarray(angle(o1, o2)) / degree


def partition(inst, gid):
  """the partition scores on the indexed pixels that are not wild and lie in a reconstructed grain"""
  valid = (inst["phase"] > 0) & ~inst["wild"] & (gid > 0)
  t, g = inst["label"][valid], gid[valid]
  C, ut, ug = contingency(t, g)
  ni, nj = np.asarray(C.sum(axis=1)).ravel(), np.asarray(C.sum(axis=0)).ravel()
  Cc = C.tocoo()
  ari, vi = pairStats(Cc.data, ni, nj)
  f1, splits, merges = matching(Cc.data, ni, nj, Cc.row, Cc.col)
  # detection: a true boundary is found when the grains most of its two sides lie in differ;
  # separated is the share of its adjacent pixel pairs put into different grains
  dom = np.asarray(C.argmax(axis=1)).ravel()
  pos = inst["pos"][valid]
  pp = cKDTree(pos).query_pairs(1.01, output_type="ndarray")
  ta, tb = np.searchsorted(ut, t[pp[:, 0]]), np.searchsorted(ut, t[pp[:, 1]])
  cross = ta != tb
  key = np.sort(np.column_stack([ta, tb])[cross], axis=1)
  pairs, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
  sep = np.bincount(inv.ravel(), g[pp[cross, 0]] != g[pp[cross, 1]], len(pairs))
  ang = misorientationAngles(inst["grainQ"], ut[pairs])
  found = dom[pairs[:, 0]] != dom[pairs[:, 1]]
  out = dict(ari=ari, vi=vi, f1_50=f1[0.5], f1_75=f1[0.75], splits=splits, merges=merges, nTrue=len(ut), nRecon=len(ug))
  for lo, hi in zip(BINS[:-1], BINS[1:]):
    s = (ang >= lo) & (ang < hi)
    out[f"det{lo}_n"], out[f"det{lo}_k"] = int(s.sum()), int(found[s].sum())
    out[f"sep{lo}"] = sep[s].sum() / max(cnt[s].sum(), 1)
  # the size distributions: Wasserstein distance of the log equivalent circle diameters
  out["sizeW1"] = wasserstein_distance(np.log(ni), np.log(nj))
  return out


def directionW1(u, wu, v, wv, nb=180):
  """the Wasserstein distance on the circle of two length weighted line direction distributions"""
  h = lambda d, w: np.bincount((np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180 * nb / 180).astype(int) % nb, w, nb) / w.sum()
  D = np.cumsum(h(u, wu) - h(v, wv))
  return np.abs(D - np.median(D)).sum() * 180 / nb


def geometry(inst, S, h=1 / 32):
  """the geometry scores of a boundary on the grains matched by an area IoU above 0.5"""
  T = inst["S"]
  A, tp, sp = pl.overlay(T, S)
  aT, aR = np.asarray(A[:, 1:].sum(axis=1)).ravel(), np.asarray(A[1:, :].sum(axis=0)).ravel()
  Ac = A[1:, 1:].tocoo()
  i, j, a = Ac.row + 1, Ac.col + 1, Ac.data
  iou = a / (aT[i] + aR[j] - a)
  m, mt = np.zeros(A.shape[0], np.int64), np.zeros(A.shape[1], np.int64)
  hit = iou > 0.5
  m[i[hit]], mt[j[hit]] = j[hit], i[hit]
  # the true boundary inside the map, each piece once, and the perimeter of every grain there
  L = lambda X: np.hypot(X[:, 2] - X[:, 0], X[:, 3] - X[:, 1])
  tl, tr = tp[:, 4].astype(np.int64), tp[:, 5].astype(np.int64)
  inW = (tp[:, 6] > 0) & (tp[:, 7] > 0) & (tl > 0) & (tr > 0) & (tl != tr)
  per = np.bincount(tl[inW], L(tp[inW]), len(aT)) + np.bincount(tr[inW], L(tp[inW]), len(aT))
  I = np.flatnonzero(m)
  sym = aT[I] + aR[m[I]] - 2 * np.asarray(A[I, m[I]]).ravel()
  out = dict(matched=len(I), disp=sym.sum() / per[I].sum(), areaBias=np.mean((aR[m[I]] - aT[I]) / aT[I]))
  # the boundaries between matched grains, in truth and reconstruction
  sl, sr = S[:, 4].astype(np.int64), S[:, 5].astype(np.int64)
  tB = tp[inW & (m[tl] > 0) & (m[tr] > 0), :4]
  sB = S[(mt[sl] > 0) & (mt[sr] > 0) & (mt[sl] != mt[sr])]
  out["lengthRatio"] = L(sB).sum() / L(tB).sum()
  tIn = T[(T[:, 4] > 0) & (T[:, 5] > 0) & (T[:, 4] != T[:, 5])]
  sIn = S[(sl > 0) & (sr > 0) & (sl != sr)]
  ps, ws, us = pl.sample(sB, h)
  pt, wt, ut = pl.sample(np.column_stack([tB, np.zeros((len(tB), 2))]), h)
  dRT, aRT = pl.nearest(tIn, ps, us)
  dTR, _ = pl.nearest(sIn, pt)
  for k, d, w in (("rt", dRT, ws), ("tr", dTR, wt)):
    o = np.argsort(d)
    c = np.cumsum(w[o]) / w.sum()
    out[f"d_{k}_mean"], out[f"d_{k}_p95"], out[f"d_{k}_max"] = (d * w).sum() / w.sum(), d[o][min(np.searchsorted(c, 0.95), len(d) - 1)], d.max()
  out["dirErr"] = (aRT * ws).sum() / ws.sum()
  out["dirW1"] = directionW1(us, ws, ut, wt)
  # the true junctions two steps inside the map, and the nearest reconstructed one
  lo, hi = inst["pos"].min(axis=0) + 2, inst["pos"].max(axis=0) - 2
  tJ = inst["junctions"][np.all((inst["junctions"] > lo) & (inst["junctions"] < hi), axis=1)]
  rJ = pl.junctions(S)
  dJ = cKDTree(rJ).query(tJ)[0] if len(rJ) and len(tJ) else np.full(len(tJ), np.inf)
  out["junctionErr"] = float(np.mean(dJ[dJ < 2])) if (dJ < 2).any() else np.nan
  out["junctionFound"] = float(np.mean(dJ < 2)) if len(tJ) else np.nan
  # pixel consistency: the pixel centres of matched grains on the wrong side of the reconstruction
  lab = inst["label"]
  rg = pl.locate(S, inst["pos"])
  ok = m[lab] > 0
  out["pixelWrong"] = float(np.mean(rg[ok] != m[lab[ok]]))
  # the boundary must leave every reconstructed grain a positive area, together the map's
  ar = pl.areas(S)[np.setdiff1d(np.unique(S[:, 4:6]), [0]).astype(np.int64)]
  out["orientOK"] = bool(ar.min() > 0 and abs(ar.sum() - aR[1:].sum()) < 1e-6 * ar.sum())
  return out
