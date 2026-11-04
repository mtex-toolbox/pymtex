# the measurement model: a structure of structures.py, grain orientations, and a square or hexagonal
# grid turned and shifted against it at random, sampled point or spot wise, with intragranular
# orientation gradients, orientation noise,
# wild and notIndexed pixels; and the files of one instance
#
#   <name>.bin, <name>.phases   the measured map in the neutral format of bench/calcgrains.py
#   <name>.labels.bin           the true label at every pixel centre, 0 where the map is notIndexed
#   <name>.truth.npz            label, qTrue, share, wild per pixel; the boundary network S (planar.py),
#                               its junctions, the orientation of every grain
#   <name>.json                 the specification, its seed and what was drawn from it
from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial import cKDTree

import planar as pl
import structures as st

PHASE = ("Iron", "m-3m")
SIGMA3 = np.array([np.sqrt(3) / 2, 0.5 / np.sqrt(3), 0.5 / np.sqrt(3), 0.5 / np.sqrt(3)])


def qmul(p, q):
  a1, b1, c1, d1 = np.moveaxis(p, -1, 0)
  a2, b2, c2, d2 = np.moveaxis(q, -1, 0)
  return np.stack([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                   a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2], axis=-1)


def expq(v):
  """the rotation of every rotation vector"""
  w = np.linalg.norm(v, axis=-1, keepdims=True)
  return np.concatenate([np.cos(w / 2), np.sinc(w / 2 / np.pi) / 2 * v], axis=-1)


def randq(rng, n):
  q = rng.standard_normal((n, 4))
  return q / np.linalg.norm(q, axis=1, keepdims=True)


def grid(kind, nx, ny):
  """the pixel centres of unit spacing, a hexagonal grid in rows shifted by half a step"""
  i, j = np.meshgrid(np.arange(nx), np.arange(ny))
  if kind == "hex":
    return np.column_stack([(i + (j % 2) / 2).ravel(), (j * np.sqrt(3) / 2).ravel()])
  return np.column_stack([i.ravel(), j.ravel()]).astype(float)


def spot(radius, k=37):
  """a sunflower of k points filling a disc"""
  t = np.arange(k)
  r = radius * np.sqrt((t + 0.5) / k)
  return np.column_stack([r * np.cos(t * 2.399963229728653), r * np.sin(t * 2.399963229728653)])


def orientations(rng, k, pairs, lowAngle=0.0, twin=0.0, lowRange=(1, 15)):
  """uniform grain orientations, a share of them low angle or Sigma 3 to a neighbour drawn before"""
  q = randq(rng, k + 1)
  nb = [[] for _ in range(k + 1)]
  for a, b in pairs:
    nb[a].append(b)
    nb[b].append(a)
  done = np.zeros(k + 1, bool)
  for g in rng.permutation(np.arange(1, k + 1)):
    u, cand = rng.random(), [n for n in nb[g] if done[n]]
    if cand and u < lowAngle + twin:
      n = cand[rng.integers(len(cand))]
      ax = randq(rng, 1)[0, 1:]
      d = expq(ax / np.linalg.norm(ax) * np.radians(rng.uniform(*lowRange))) if u < lowAngle else SIGMA3
      q[g] = qmul(q[n], d)
    done[g] = True
  return q


def gradients(rng, k, label, pos, kind, degrees):
  """a linear orientation gradient in every grain: about a random axis, along a random direction in
  the plane, turning by degrees[0] to degrees[1] across the grain's equivalent diameter; the rotation
  of every pixel position as seen from the grain it is given"""
  n = np.bincount(label, minlength=k + 1)
  c = np.column_stack([np.bincount(label, pos[:, i], k + 1) for i in (0, 1)]) / np.maximum(n, 1)[:, None]
  ax = randq(rng, k + 1)[:, 1:]
  ax /= np.linalg.norm(ax, axis=1, keepdims=True)
  th = rng.uniform(0, 2 * np.pi, k + 1)
  diam = 2 * np.sqrt(np.maximum(n, 1) * (np.sqrt(3) / 2 if kind == "hex" else 1) / np.pi)
  rate = np.radians(rng.uniform(*degrees, k + 1)) / diam
  def turn(gr):
    s = ((pos - c[gr]) * np.column_stack([np.cos(th[gr]), np.sin(th[gr])])).sum(axis=1)
    return expq(ax[gr] * (rate[gr] * s)[:, None])
  return turn


def instance(spec):
  """the measured map and its truth for one specification"""
  sStruct, sOri, sMeas = (np.random.default_rng(s) for s in np.random.SeedSequence(spec["seed"]).spawn(3))
  g, o, m = spec["grid"], spec.get("orientation", {}), spec.get("measure", {})
  pos = grid(g.get("kind", "square"), g["nx"], g["ny"])
  # the structure in a box that still covers the map once turned about its centre
  lo, hi = pos.min(axis=0) - 1, pos.max(axis=0) + 1
  c, half = (lo + hi) / 2, np.hypot(*(hi - lo)) / 2 + 2
  phi, offset = sMeas.uniform(0, 2 * np.pi), sMeas.uniform(0, 1, 2)
  S = st.generate(sStruct, (c[0] - half, c[0] + half, c[1] - half, c[1] + half), **spec["structure"])
  S = st.transform(S, phi, c, offset)
  k = int(S[:, 4:6].max())
  qg = orientations(sOri, k, st.neighbours(S), o.get("lowAngle", 0.0), o.get("twin", 0.0))
  label = pl.locate(S, pos)
  turn = gradients(sOri, k, label, pos, g.get("kind", "square"), o["gradient"]) if o.get("gradient") else None
  # spot sampling: the majority of the grains in the spot sets the label, its share the confidence
  if m.get("spot"):
    L = pl.locate(S, (pos[:, None] + spot(m["spot"])).reshape(-1, 2)).reshape(len(pos), -1)
    pix = np.repeat(np.arange(len(pos)), L.shape[1])
    u, cnt = np.unique(np.column_stack([pix, L.ravel()]), axis=0, return_counts=True)
    o2 = np.lexsort((-cnt, u[:, 0]))
    first = o2[np.r_[True, u[o2[1:], 0] != u[o2[:-1], 0]]]
    measured, share = u[first, 1], cnt[first] / L.shape[1]
  else:
    measured, share = label, np.ones(len(pos))
  qTrue, q = qg[label], qg[measured]
  if turn is not None:
    qTrue, q = qmul(qTrue, turn(label)), qmul(q, turn(measured))
  q = qmul(q, expq(sMeas.standard_normal((len(pos), 3)) * np.radians(m.get("sigma", 0.0)) / np.sqrt(3)))
  wild = sMeas.random(len(pos)) < m.get("wild", 0.0)
  q[wild] = randq(sMeas, wild.sum())
  bad = (share < m.get("minShare", 0.0)) | (sMeas.random(len(pos)) < m.get("notIndexed", 0.0))
  # clusters of notIndexed pixels: discs of radius 1 to 4 until they cover their share of the map
  tree, cl = cKDTree(pos), np.zeros(len(pos), bool)
  while cl.mean() < m.get("clusters", 0.0):
    cl[tree.query_ball_point(sMeas.uniform(lo, hi), sMeas.uniform(1, 4))] = True
  bad |= cl
  q[bad] = np.nan
  head = dict(spec, phi=float(phi), offset=offset.tolist(), grains=int(len(np.unique(label))), notIndexed=float(bad.mean()),
              wild=float(wild.mean()))
  return dict(pos=pos, phase=(~bad).astype(float), q=q, label=label, qTrue=qTrue, share=share, wild=wild,
              S=S, junctions=pl.junctions(S), grainQ=qg, head=head)


def write(out, name, inst):
  os.makedirs(out, exist_ok=True)
  f = lambda ext: os.path.join(out, name + ext)
  np.column_stack([inst["pos"], inst["phase"], inst["q"]]).astype("<f8").tofile(f(".bin"))
  open(f(".phases"), "w").write("\t".join([PHASE[0], PHASE[1], "1 1 1", "90 90 90", ""]) + "\n")
  np.where(inst["phase"] > 0, inst["label"], 0).astype("<f8").tofile(f(".labels.bin"))
  np.savez_compressed(f(".truth.npz"), **{k: v for k, v in inst.items() if k not in ("pos", "phase", "q", "head")})
  json.dump(inst["head"], open(f(".json"), "w"), indent=1)


def read(out, name):
  f = lambda ext: os.path.join(out, name + ext)
  A = np.fromfile(f(".bin"), "<f8").reshape(-1, 7)
  return dict(np.load(f(".truth.npz")), pos=A[:, :2], phase=A[:, 2], q=A[:, 3:], head=json.load(open(f(".json"))))
