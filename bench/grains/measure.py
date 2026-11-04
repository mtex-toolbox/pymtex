# the measurement model: a structure of structures.py, the orientation field of its grains, and a
# square or hexagonal grid turned and shifted against it at random, sampled at the pixel centre or
# by an elliptic spot, with boundaries of finite width, orientation noise, wild, misindexed and
# notIndexed pixels; and the files of one instance
#
#   <name>.bin, <name>.phases   the measured map in the neutral format of bench/calcgrains.py
#   <name>.labels.bin           the true label at every pixel centre, 0 where the map is notIndexed
#   <name>.mix.bin              the share of the measured grain in the spot of every pixel, 1 for point sampling
#   <name>.truth.npz            per pixel label, qTrue, share, label2, share2, wild, misindexed; the
#                               boundary network S (planar.py), its junctions; per grain grainQ,
#                               grainPhase, edge (cut by the map window), parent
#   <name>.json                 the specification, its seed and what was drawn from it
from __future__ import annotations

import functools
import json
import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import ndtri

import planar as pl
import structures as st

# the phases: point group, a b c, alpha beta gamma, alignment, and as (crystal axis, degrees) the
# twin and the pseudo-symmetric solution misindexing gives
PHASES = {
  "Iron": ("m-3m", (1, 1, 1), (90, 90, 90), "", ((1, 1, 1), 60), None),
  "Titanium": ("6/mmm", (2.95, 2.95, 4.68), (90, 90, 120), "X||a", ((1, 0, 0), 85.03), ((0, 0, 1), 30)),
  "Forsterite": ("mmm", (4.762, 10.225, 5.994), (90, 90, 90), "", None, None),
  "TiAl": ("4/mmm", (4.0, 4.0, 4.08), (90, 90, 90), "", ((1, 1, 1), 60), ((1, 0, 0), 90)),
}


def qmul(p, q):
  a1, b1, c1, d1 = np.moveaxis(p, -1, 0)
  a2, b2, c2, d2 = np.moveaxis(q, -1, 0)
  return np.stack([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                   a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2], axis=-1)


def expq(v):
  """the rotation of every rotation vector"""
  w = np.linalg.norm(v, axis=-1, keepdims=True)
  return np.concatenate([np.cos(w / 2), np.sinc(w / 2 / np.pi) / 2 * v], axis=-1)


def axisAngle(axis, degrees):
  a = np.asarray(axis, float)
  return expq(a / np.linalg.norm(a) * np.radians(degrees))


def randq(rng, n):
  q = rng.standard_normal((n, 4))
  return q / np.linalg.norm(q, axis=1, keepdims=True)


def randAxes(rng, n):
  v = rng.standard_normal((n, 3))
  return v / np.linalg.norm(v, axis=1, keepdims=True)


@functools.lru_cache
def frame(name):
  from mtex import crystalFrame
  pg, abc, ang, al = PHASES[name][:4]
  return crystalFrame(pg, list(abc), list(ang), mineral=name, **({"alignment": al.split(",")} if al else {}))


@functools.lru_cache
def proper(name):
  """the rotations of the point group, as quaternions"""
  R = np.asarray(frame(name).rot.data)
  return R[R[:, 4] > 0, :4]


def nearestImage(qa, qb, sym):
  """qb times the symmetry that brings it nearest to qa, on the side of qa"""
  c = qmul(qb[:, None], sym[None])
  d = np.einsum("ik,ijk->ij", qa, c)
  k = np.argmax(np.abs(d), axis=1)
  n = np.arange(len(qa))
  return c[n, k] * np.sign(d[n, k])[:, None]


def slerp(qa, qb, t):
  d = np.clip(np.einsum("ij,ij->i", qa, qb), -1, 1)
  w = np.arccos(d)
  s = np.where(w > 1e-9, np.sin(w), 1)
  a = np.where(w > 1e-9, np.sin((1 - t) * w) / s, 1 - t)
  b = np.where(w > 1e-9, np.sin(t * w) / s, t)
  q = a[:, None] * qa + b[:, None] * qb
  return q / np.linalg.norm(q, axis=1, keepdims=True)


def grid(kind, nx, ny):
  """the pixel centres of unit spacing, a hexagonal grid in rows shifted by half a step"""
  i, j = np.meshgrid(np.arange(nx), np.arange(ny))
  if kind == "hex":
    return np.column_stack([(i + (j % 2) / 2).ravel(), (j * np.sqrt(3) / 2).ravel()])
  return np.column_stack([i.ravel(), j.ravel()]).astype(float)


def window(pos, kind):
  """the box the pixel cells cover, the hexagonal zigzag rounded outwards"""
  r = np.array([0.5, 1 / np.sqrt(3) if kind == "hex" else 0.5])
  return np.concatenate([[pos[:, 0].min() - r[0], pos[:, 0].max() + r[0]], [pos[:, 1].min() - r[1], pos[:, 1].max() + r[1]]])


def spot(radius, aspect=1.0, k=37):
  """a sunflower of k points filling an ellipse, radius along the tilt axis x and aspect times
  radius across it"""
  t = np.arange(k)
  r = np.sqrt((t + 0.5) / k)
  return radius * np.column_stack([r * np.cos(t * 2.399963229728653), aspect * r * np.sin(t * 2.399963229728653)])


def randomField(rng, lo, hi, cov, ncomp, h=0.5):
  """a stationary Gaussian random field of ncomp components with covariance cov(r) on the box
  lo..hi, by circulant embedding on a grid of spacing h; returns its values at given points,
  bilinear between the grid nodes"""
  n = np.ceil((hi - lo) / h).astype(int) + 2
  N = 2 * n
  i, j = (np.minimum(np.arange(m), m - np.arange(m)) * h for m in N)
  lam = np.maximum(np.fft.fft2(cov(np.hypot(*np.meshgrid(i, j, indexing="ij")))).real, 0)
  F = np.stack([np.fft.ifft2(np.sqrt(lam) * np.fft.fft2(rng.standard_normal(N))).real[:n[0], :n[1]] for _ in range(ncomp)], -1)
  def at(xy):
    u = (xy - lo) / h
    k = np.clip(np.floor(u).astype(int), 0, n - 2)
    f = u - k
    g = lambda a, b: F[k[:, 0] + a, k[:, 1] + b]
    return ((1 - f[:, :1]) * ((1 - f[:, 1:]) * g(0, 0) + f[:, 1:] * g(0, 1)) +
            f[:, :1] * ((1 - f[:, 1:]) * g(1, 0) + f[:, 1:] * g(1, 1)))
  return at


def orientations(rng, k, pairs, phase, lowAngle=0.0, twin=0.0, lowRange=(1, 15)):
  """uniform grain orientations, a share of them low angle or twin to a neighbour of their phase
  drawn before them"""
  q = randq(rng, k + 1)
  nb = [[] for _ in range(k + 1)]
  for a, b in pairs:
    if phase[a] == phase[b]:
      nb[a].append(b)
      nb[b].append(a)
  done = np.zeros(k + 1, bool)
  for g in rng.permutation(np.arange(1, k + 1)):
    u, cand = rng.random(), [n for n in nb[g] if done[n]]
    tw = PHASES[phase[g]][4]
    if cand and u < lowAngle + twin and (u < lowAngle or tw):
      n = cand[rng.integers(len(cand))]
      ax = randAxes(rng, 1)[0]
      q[g] = qmul(q[n], axisAngle(ax, rng.uniform(*lowRange)) if u < lowAngle else axisAngle(*tw))
    done[g] = True
  return q


def gradients(rng, k, label, pos, kind, degrees):
  """a linear orientation gradient in every grain: about a random axis, along a random direction in
  the plane, turning by degrees[0] to degrees[1] across the grain's equivalent diameter; the rotation
  of every pixel position as seen from the grain it is given"""
  n = np.bincount(label, minlength=k + 1)
  c = np.column_stack([np.bincount(label, pos[:, i], k + 1) for i in (0, 1)]) / np.maximum(n, 1)[:, None]
  ax = randAxes(rng, k + 1)
  th = rng.uniform(0, 2 * np.pi, k + 1)
  diam = 2 * np.sqrt(np.maximum(n, 1) * (np.sqrt(3) / 2 if kind == "hex" else 1) / np.pi)
  rate = np.radians(rng.uniform(*degrees, k + 1)) / diam
  def turn(gr, xy):
    s = ((xy - c[gr]) * np.column_stack([np.cos(th[gr]), np.sin(th[gr])])).sum(axis=1)
    return expq(ax[gr] * (rate[gr] * s)[:, None])
  return turn


def misindex(rng, label, pos, qg, phase, share, scale=2.0, pseudo=1.0):
  """patches of pixels indexed as another orientation: the upper tail of a Gaussian random field of
  correlation length scale, as large in each grain as its susceptibility asks, which rises from 0
  to 2 times share as the specimen normal z approaches the crystal's highest-order axis; a patch
  takes the grain's pseudo-symmetric solution with probability pseudo, else one random orientation"""
  lo, hi = pos.min(axis=0) - 1, pos.max(axis=0) + 1
  F = randomField(rng, lo, hi, lambda r: np.exp(-(r / scale) ** 2), 1)(pos)[:, 0]
  F /= F.std()
  # the specimen normal z in crystal coordinates against the symmetric images of the main axis
  zc = qmul(qmul(qg * [1, -1, -1, -1], [0, 0, 0, 1]), qg)[:, 1:]
  sus = np.zeros(len(qg))
  for name in set(phase[1:]):
    a = np.asarray({"m-3m": (1, 1, 1), "6/mmm": (0, 0, 1), "4/mmm": (0, 0, 1)}.get(PHASES[name][0], (0, 0, 1)), float)
    R = proper(name)
    imgs = qmul(qmul(R, np.r_[0, a / np.linalg.norm(a)]), R * [1, -1, -1, -1])[:, 1:]
    sel = np.flatnonzero(phase == name)
    sus[sel] = np.max(np.abs(zc[sel] @ imgs.T), axis=1) ** 2
  p = np.clip(2 * share * (sus - sus[1:].min()) / max(np.ptp(sus[1:]), 1e-12), 0, 0.95)
  hit = F > ndtri(1 - np.maximum(p[label], 1e-12))
  # the patches: connected pixels of one grain
  pp = cKDTree(pos).query_pairs(1.01, output_type="ndarray")
  pp = pp[hit[pp[:, 0]] & hit[pp[:, 1]] & (label[pp[:, 0]] == label[pp[:, 1]])]
  _, patch = connected_components(coo_matrix((np.ones(len(pp)), (pp[:, 0], pp[:, 1])), shape=(len(pos),) * 2), directed=False)
  ids, inv = np.unique(patch[hit], return_inverse=True)
  owner = label[hit][np.unique(inv, return_index=True)[1]]
  wrong = randq(rng, len(ids))
  ps = rng.random(len(ids)) < pseudo
  for n, g in enumerate(owner):
    ax = PHASES[phase[g]][5]
    if ps[n] and ax:
      wrong[n] = qmul(qg[g], axisAngle(*ax))
  return hit, wrong[inv]


def instance(spec):
  """the measured map and its truth for one specification"""
  sStruct, sOri, sMeas = (np.random.default_rng(s) for s in np.random.SeedSequence(spec["seed"]).spawn(3))
  g, o, m = spec["grid"], spec.get("orientation", {}), spec.get("measure", {})
  kind = g.get("kind", "square")
  pos = grid(kind, g["nx"], g["ny"])
  # the structure in a box that still covers the map once turned about its centre
  lo, hi = pos.min(axis=0) - 1, pos.max(axis=0) + 1
  c, half = (lo + hi) / 2, np.hypot(*(hi - lo)) / 2 + 2
  phi, offset = sMeas.uniform(0, 2 * np.pi), sMeas.uniform(0, 1, 2)
  S, extra = st.generate(sStruct, (c[0] - half, c[0] + half, c[1] - half, c[1] + half), **spec["structure"])
  S = st.transform(S, phi, c, offset)
  k = int(S[:, 4:6].max())
  names = spec.get("phases", ["Iron"])
  phase = np.array(names, dtype=object)[np.asarray(extra.get("phase", np.ones(k + 1, int))) - 1]
  qg = orientations(sOri, k, st.neighbours(S), phase, o.get("lowAngle", 0.0), o.get("twin", 0.0))
  label = pl.locate(S, pos)
  turn = gradients(sOri, k, label, pos, kind, o["gradient"]) if o.get("gradient") else None
  field = lambda gr, xy: qg[gr] if turn is None else qmul(qg[gr], turn(gr, xy))
  # spot sampling: the majority of the grains in the spot sets the label, its share the confidence
  label2, share2 = np.zeros(len(pos), np.int64), np.zeros(len(pos))
  if m.get("spot"):
    L = pl.locate(S, (pos[:, None] + spot(m["spot"], m.get("spotAspect", 1.0))).reshape(-1, 2)).reshape(len(pos), -1)
    pix = np.repeat(np.arange(len(pos)), L.shape[1])
    u, cnt = np.unique(np.column_stack([pix, L.ravel()]), axis=0, return_counts=True)
    o2 = np.lexsort((-cnt, u[:, 0]))
    first = np.r_[True, u[o2[1:], 0] != u[o2[:-1], 0]]
    rank = np.arange(len(o2)) - np.maximum.accumulate(np.where(first, np.arange(len(o2)), 0))
    measured, share = u[o2[first], 1], cnt[o2[first]] / L.shape[1]
    sec = o2[rank == 1]
    label2[u[sec, 0]], share2[u[sec, 0]] = u[sec, 1], cnt[sec] / L.shape[1]
  else:
    measured, share = label, np.ones(len(pos))
  # a boundary of width w: within w/2 of it the orientation turns towards the grain across,
  # half way at the boundary itself, wherever both grains are of one phase
  dist, _, LR = pl.nearest(S[(S[:, 4] > 0) & (S[:, 5] > 0) & (S[:, 4] != S[:, 5])], pos)
  across = np.where(LR[:, 0] == label, LR[:, 1], LR[:, 0])
  own = (LR == label[:, None]).any(axis=1)
  qTrue = field(label, pos)
  w = m.get("width", 0.0)
  if w > 0:
    t = np.clip(0.5 - dist / w, 0, 0.5) * (phase[across] == phase[label]) * own
    on = t > 0
    qb = field(across[on], pos[on])
    for name in set(phase[label[on]]):
      sel = phase[label[on]] == name
      idx = np.flatnonzero(on)[sel]
      qTrue[idx] = slerp(qTrue[idx], nearestImage(qTrue[idx], qb[sel], proper(name)), t[idx])
  q = np.where((measured == label)[:, None], qTrue, field(measured, pos))
  mis = np.zeros(len(pos), bool)
  if m.get("misindex"):
    mis, wrong = misindex(sMeas, measured, pos, qg, phase, m["misindex"], m.get("misindexScale", 2.0), m.get("pseudo", 1.0))
    q[mis] = wrong
  # noise: exp(v) with v ~ N(0, s^2 I), whose angle has the mean 2 s sqrt(2 / pi) = noise
  q = qmul(q, expq(sMeas.standard_normal((len(pos), 3)) * np.radians(m.get("noise", 0.0)) * np.sqrt(np.pi / 8)))
  wild = sMeas.random(len(pos)) < m.get("wild", 0.0)
  q[wild] = randq(sMeas, wild.sum())
  bad = (share < m.get("minShare", 0.0)) | (sMeas.random(len(pos)) < m.get("notIndexed", 0.0))
  bad |= (dist < m.get("band", 0.75)) & (sMeas.random(len(pos)) < m.get("boundaryNotIndexed", 0.0))
  # clusters of notIndexed pixels: discs of radius 1 to 4 until they cover their share of the map
  tree, cl = cKDTree(pos), np.zeros(len(pos), bool)
  while cl.mean() < m.get("clusters", 0.0):
    cl[tree.query_ball_point(sMeas.uniform(lo, hi), sMeas.uniform(1, 4))] = True
  bad |= cl
  q[bad] = np.nan
  # a grain is cut by the map when part of it lies outside the window the pixel cells cover
  A = pl.overlay(S, pl.clipToBox(np.zeros((0, 6)), window(pos, kind), lambda xy: np.ones(len(xy))))[0].toarray()
  edge = (A[:, 0] > 1e-9) & (A[:, 1] > 1e-9)
  head = dict(spec, phi=float(phi), offset=offset.tolist(), grains=int(len(np.unique(label))), notIndexed=float(bad.mean()),
              wild=float(wild.mean()), misindexed=float(mis.mean()), tiltAxis="x",
              noiseModel="exp(v), v ~ N(0, s^2 I), mean angle 2 s sqrt(2/pi) = noise")
  grainPhase = np.array([names.index(p) + 1 for p in phase])
  return dict(pos=pos, phase=np.where(bad, 0, grainPhase[measured]).astype(float), q=q, label=label, qTrue=qTrue, share=share,
              label2=label2, share2=share2, wild=wild, misindexed=mis, S=S, junctions=pl.junctions(S), grainQ=qg,
              grainPhase=grainPhase, edge=edge,
              parent=np.asarray(extra.get("parent", np.arange(k + 1))), phases=np.array(names), head=head)


def write(out, name, inst):
  os.makedirs(out, exist_ok=True)
  f = lambda ext: os.path.join(out, name + ext)
  np.column_stack([inst["pos"], inst["phase"], inst["q"]]).astype("<f8").tofile(f(".bin"))
  lines = []
  for p in inst["phases"]:
    pg, abc, ang, al = PHASES[p][:4]
    lines.append("\t".join([p, pg, " ".join(map(str, abc)), " ".join(map(str, ang)), al]))
  open(f(".phases"), "w").write("\n".join(lines) + "\n")
  np.where(inst["phase"] > 0, inst["label"], 0).astype("<f8").tofile(f(".labels.bin"))
  inst["share"].astype("<f8").tofile(f(".mix.bin"))
  np.savez_compressed(f(".truth.npz"), **{k: v for k, v in inst.items() if k not in ("pos", "phase", "q", "head")})
  json.dump(inst["head"], open(f(".json"), "w"), indent=1)


def read(out, name):
  f = lambda ext: os.path.join(out, name + ext)
  A = np.fromfile(f(".bin"), "<f8").reshape(-1, 7)
  return dict(np.load(f(".truth.npz"), allow_pickle=False), pos=A[:, :2], phase=A[:, 2], q=A[:, 3:], head=json.load(open(f(".json"))))
