# the grain reconstruction benchmark (../master/docs/grain-reconstruction-research-plan.md, M1)
#
#   .venv/bin/python bench/grains/run.py generate T0         the instances of a tier from their seeds
#   .venv/bin/python bench/grains/run.py run T0 [name ...]   this port's candidates
#   .venv/bin/python bench/grains/run.py matlab T0 [name ...]  bench/matlab/bench_grains.m through the bridge
#   .venv/bin/python bench/grains/run.py score T0            every result scored into <tier>/scores.csv
#   .venv/bin/python bench/grains/run.py table T0            the smoke table of docs/research/grain-benchmark.md
#   .venv/bin/python bench/grains/run.py compare T0          the cross-check of the ports, run by run
#   .venv/bin/python bench/grains/checks.py                  the metrics on analytic cases
#
# A candidate segments the measured map (track map, at 5 degree) or joins the pixels of one true
# label (track labels, calcGrains(..., grainId=True)), so that the geometry is scored apart from the
# segmentation, and smooths the boundary or not. Every run writes <port>_<name>_<case>.px.bin, rows
# (x, y, grainId), and .gb.bin, the boundary as rows (x1, y1, x2, y2, left, right) with the grain
# `left` on the left, into <tier>/res; the scores are computed from these files alone.
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure  # noqa: E402
import metrics  # noqa: E402
import tiers  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "out", "grains")
ANGLE = 5
TRACKS = ("map", "labels")
SMOOTH = {"raw": None, "smooth": {}, "noSimplify": dict(simplify=False)}
CASES = [f"{t}-{s}" for t in TRACKS for s in SMOOTH]
PORTS = ("python", "matlabNoStraighten", "matlab")
ALGORITHM = [
  {"name": "grain benchmark: segmentation",
   "entry": {"matlab": ["@EBSD.calcGrains"], "python": ["mtex.maps.calcgrains.calcGrains"]},
   "cases": {"map-raw": "calcGrains at 5 degree of the measured map", "labels-raw": "calcGrains(grainId) of the true labels"}},
  {"name": "grain benchmark: boundary smoothing",
   "entry": {"matlab": ["@grain2d.smoothBoundary"], "python": ["mtex.maps.smoothing.smoothBoundary"]},
   "cases": {"map-smooth": "smoothBoundary(1)", "map-noSimplify": "smoothBoundary(1, 'noSimplify')",
             "labels-smooth": "smoothBoundary(1) on the true labels", "labels-noSimplify": "smoothBoundary(1, 'noSimplify') on the true labels"}},
]


def names(tier, args):
  return args or [s["name"] for s in tiers.TIERS[tier]]


def generate(tier, which):
  for s in tiers.TIERS[tier]:
    if s["name"] in which:
      t0 = time.perf_counter()
      inst = measure.instance(s)
      measure.write(os.path.join(OUT, tier), s["name"], inst)
      h = inst["head"]
      print(f"{s['name']}: {len(inst['pos'])} px, {h['grains']} grains, {100 * h['notIndexed']:.1f} % notIndexed, {time.perf_counter() - t0:.1f} s")


def loadMap(tier, name):
  from mtex import EBSD, crystalFrame, phaseList, rotation
  d = os.path.join(OUT, tier)
  A = np.fromfile(os.path.join(d, name + ".bin"), "<f8").reshape(-1, 7)
  lab = np.fromfile(os.path.join(d, name + ".labels.bin"), "<f8")
  frames = []
  for line in open(os.path.join(d, name + ".phases")).read().splitlines():
    mineral, pg, abc, abg, al = line.split("\t")
    frames.append(crystalFrame(pg, [float(x) for x in abc.split()], [float(x) for x in abg.split()], mineral=mineral,
                               **({"alignment": al.split(",")} if al else {})))
  pos = np.column_stack([A[:, :2], np.zeros(len(A))])
  return EBSD(pos, rotation(A[:, 3:7]), A[:, 2].astype(np.int8), phaseList(*frames), prop=dict(grainId=lab)).gridify()


def writeResult(res, port, name, case, t, g, e):
  os.makedirs(res, exist_ok=True)
  stem = os.path.join(res, f"{port}_{name}_{case}")
  keep = ~e.isPadding.reshape(-1)
  xy = e.pos.data.reshape(-1, 3)[keep, :2]
  np.column_stack([xy, np.asarray(e.grainId).reshape(-1)[keep]]).astype("<f8").tofile(stem + ".px.bin")
  b = g.boundary
  V, F, G = np.asarray(b.V, float)[:, :2], np.asarray(b.F), np.asarray(b.grainId)
  np.column_stack([V[F[:, 0]], V[F[:, 1]], G]).astype("<f8").tofile(stem + ".gb.bin")
  open(stem + ".txt", "w").write(f"time {t}\ngrains {len(g)}\n")


def runPython(tier, which):
  from mtex import calcGrains, degree, smoothBoundary
  res = os.path.join(OUT, tier, "res")
  for name in which:
    e0 = loadMap(tier, name)
    for track in TRACKS:
      e = e0.copy()
      t0 = time.perf_counter()
      g = calcGrains(e, angle=ANGLE * degree) if track == "map" else calcGrains(e, grainId=True)
      t = time.perf_counter() - t0
      for s, kw in SMOOTH.items():
        t1 = time.perf_counter()
        gs = g if kw is None else smoothBoundary(g, **kw)
        writeResult(res, "python", name, f"{track}-{s}", t + time.perf_counter() - t1, gs, e)
    print(f"{name}: {len(g)} grains")


# MATLAB through the bridge, one instance per call; SESSION_NAME names the engine to connect to
def runMatlab(tier, which):
  bridge = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "master", "docs", "agents", "matlab-bridge")
  d = os.path.join(OUT, tier)
  for name in which:
    cmd = f"addpath('{os.path.join(os.path.dirname(HERE), 'matlab')}'); bench_grains('{d}', '{name}', {ANGLE});"
    r = subprocess.run([os.path.join(bridge, ".venv", "bin", "python"), os.path.join(bridge, "matlab_run.py"), cmd],
                       capture_output=True, text=True, timeout=300)
    print(r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else f"{name} FAILED\n{r.stdout[-1500:]}{r.stderr[-1500:]}")


def readResult(inst, stem):
  """the grain id of every pixel of the instance and the boundary of one run"""
  px = np.fromfile(stem + ".px.bin", "<f8").reshape(-1, 3)
  key = lambda xy: np.round(xy * 1000).astype(np.int64) @ np.array([1, 1 << 32])
  where = dict(zip(key(inst["pos"]), range(len(inst["pos"]))))
  gid = np.zeros(len(inst["pos"]), np.int64)
  idx = np.array([where.get(k, -1) for k in key(px[:, :2])])
  ok = (idx >= 0) & np.isfinite(px[:, 2])
  gid[idx[ok]] = px[ok, 2]
  S = np.fromfile(stem + ".gb.bin", "<f8").reshape(-1, 6)
  stats = dict(l.split() for l in open(stem + ".txt").read().splitlines() if l.strip())
  return gid, S, float(stats["time"])


def score(tier):
  d = os.path.join(OUT, tier)
  rows = []
  for s in tiers.TIERS[tier]:
    inst = measure.read(d, s["name"])
    for port in PORTS:
      for case in CASES:
        stem = os.path.join(d, "res", f"{port}_{s['name']}_{case}")
        if not os.path.exists(stem + ".gb.bin"):
          continue
        gid, S, t = readResult(inst, stem)
        rows.append(dict(instance=s["name"], port=port, case=case, time=t) | metrics.partition(inst, gid) | metrics.geometry(inst, S))
  with open(os.path.join(d, "scores.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  print(f"{len(rows)} rows to {os.path.join(d, 'scores.csv')}")


COLUMNS = [("ari", "ARI", 3), ("f1_75", "F1 .75", 3), ("splits", "splits", 0), ("merges", "merges", 0), ("disp", "disp", 3),
           ("d_rt_mean", "d r→t", 3), ("d_tr_p95", "d t→r p95", 3), ("d_rt_max", "d max", 2), ("areaBias", "area %", 2),
           ("lengthRatio", "length", 3), ("dirErr", "dir °", 1), ("dirW1", "dirW1 °", 1), ("junctionErr", "junction", 2),
           ("pixelWrong", "pixel %", 2), ("time", "time s", 2)]


def table(tier, port=None):
  rows = list(csv.DictReader(open(os.path.join(OUT, tier, "scores.csv"))))
  head = ["port", "case"] + [c[1] for c in COLUMNS]
  print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
  for p in PORTS:
    for case in CASES:
      sel = [r for r in rows if r["port"] == p and r["case"] == case]
      if not sel:
        continue
      cells = []
      for k, _, nd in COLUMNS:
        v = np.nanmean([float(r[k]) for r in sel])
        v = 100 * v if k in ("areaBias", "pixelWrong") else v
        cells.append(f"{v:.{nd}f}")
      print(f"| {p} | {case} | " + " | ".join(cells) + " |")


# two ports on the same runs: the share of pixels whose grains differ (each grain of a matched to
# the grain of b most of its pixels lie in) and the Hausdorff distance of the two boundaries
def compare(tier, a="python", b="matlabNoStraighten"):
  import planar as pl
  from metrics import contingency
  d = os.path.join(OUT, tier)
  print(f"| case | instances | same partition | pixels differing | boundary Hausdorff max | median |\n|---|---|---|---|---|---|")
  for case in CASES:
    diff, haus = [], []
    for s in tiers.TIERS[tier]:
      inst = measure.read(d, s["name"])
      stems = [os.path.join(d, "res", f"{p}_{s['name']}_{case}") for p in (a, b)]
      if not all(os.path.exists(x + ".gb.bin") for x in stems):
        continue
      (g1, S1, _), (g2, S2, _) = (readResult(inst, x) for x in stems)
      C = contingency(g1, g2)[0]
      diff.append(1 - np.asarray(C.max(axis=1).todense()).sum() / len(g1))
      inner = lambda S: S[(S[:, 4] > 0) & (S[:, 5] > 0)]
      haus.append(max(pl.nearest(inner(S2), pl.sample(inner(S1))[0])[0].max(), pl.nearest(inner(S1), pl.sample(inner(S2))[0])[0].max()))
    if diff:
      print(f"| {case} | {len(diff)} | {sum(x == 0 for x in diff)} | {max(diff):.2%} | {max(haus):.2g} | {np.median(haus):.2g} |")


if __name__ == "__main__":
  cmd, tier, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
  if cmd == "generate":
    generate(tier, names(tier, rest))
  elif cmd == "run":
    runPython(tier, names(tier, rest))
  elif cmd == "matlab":
    runMatlab(tier, names(tier, rest))
  elif cmd == "score":
    score(tier)
  elif cmd == "table":
    table(tier)
  elif cmd == "compare":
    compare(tier, *rest)
