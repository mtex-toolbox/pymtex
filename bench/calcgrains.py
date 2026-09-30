# calcGrains against MATLAB and the Julia port: one input per map, the grains compared by position
#
#   .venv/bin/python bench/calcgrains.py export             the maps as the neutral input of all three ports
#   .venv/bin/python bench/calcgrains.py run [map ...]      this port: times, counts, per-pixel grain ids
#   julia -t auto --project=../mtexJulia/dev bench/calcgrains.jl [map ...]
#   .venv/bin/python bench/calcgrains.py matlab [map ...]   bench/matlab/bench_calcgrains.m through the bridge
#   .venv/bin/python bench/calcgrains.py compare            the table of docs/research/benchmark-calcgrains.md
#
# A map is read once, by this port's reader, and written as its real measurements - the padding
# of a grid left out - one float64 row (x, y, phase, a, b, c, d) per measurement, the phase an
# index into the phase list with 0 notIndexed, and a text file with one tab separated line per
# phase: mineral, point group, a b c, alpha beta gamma in degrees, the alignment. Every port builds its map from
# these, puts it on its grid itself and segments it, so the three see one input and differ only in
# calcGrains. Each run writes its counts to <out>/<port>_<map>_<case>.txt and its partition as
# float64 rows (x, y, grainId) to <out>/<port>_<map>_<case>.bin, compared by position.
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "calcgrains")
MAPS = ["twins", "copper", "csl", "forsterite", "emsland", "synthetic"]
# large maps run only when named: the 2.5 million pixel steel map of the grain reconstruction benchmarks
FILES = {"1C_1": os.path.expanduser("~/mtex/data/1C_1.ctf")}
# name: keywords of calcGrains, the same in every port
CASES = {
  "a": dict(angle=10, minPixel=1, alpha=3.1, removeQuadruplePoints=False),
  "b": dict(angle=5, minPixel=5, alpha=3.1, removeQuadruplePoints=False),
  "c": dict(angle=10, minPixel=1, alpha=0.0, removeQuadruplePoints=False),
  "d": dict(angle=10, minPixel=1, alpha=3.1, removeQuadruplePoints=True),
}
# the quadruple point split only where the grid is square and the map small enough to matter
CASES_OF = {m: "abcd" if m in ("twins", "forsterite") else "abc" for m in MAPS} | {"1C_1": "ac"}


def exportMaps(names):
  import mtex
  from mtex import loadEBSD, mtexdata
  os.makedirs(OUT, exist_ok=True)
  for name in names:
    e = syntheticMap() if name == "synthetic" else loadEBSD(FILES[name]) if name in FILES else mtexdata(name, verbose=False)
    keep = ~e.isPadding.reshape(-1)
    xy = e.pos.data.reshape(-1, 3)[keep, :2]
    ph = e.phaseId.reshape(-1)[keep].astype(float)
    # preserve the reader's rotations, including NaNs on notIndexed measurements
    q = e.rotations.data.reshape(-1, 4)[keep]
    np.column_stack([xy, ph, q]).astype("<f8").tofile(os.path.join(OUT, f"{name}.bin"))
    lines = []
    for cs in e.CSList.frames[1:]:
      a, b, c = (np.asarray(v.data, float).reshape(3) for v in (cs.aAxis, cs.bAxis, cs.cAxis))
      ang = lambda u, v: np.degrees(np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v)))
      abc = " ".join(f"{x:.6g}" for x in cs.abc)
      abg = " ".join(f"{x:.6g}" for x in (ang(b, c), ang(a, c), ang(a, b)))
      lines.append("\t".join([cs.mineral, cs.pointGroup, abc, abg, ",".join(cs.alignment())]))
    open(os.path.join(OUT, f"{name}.phases"), "w").write("\n".join(lines) + "\n")
    print(f"{name}: {keep.sum()} measurements, {len(lines)} phases")


# the Voronoi map of bench/maps.py: 600 x 600 pixels, 800 seeds, one orientation about z each
def syntheticMap(n=600):
  from mtex import EBSD, crystalFrame, phaseList, rotation, zvector
  ang = np.fromfile(os.path.join(HERE, f"maps_{n}_800.bin"), "<f8")
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  pos = np.stack([i.reshape(-1), j.reshape(-1), np.zeros(n * n)], axis=-1).astype(float)
  return EBSD(pos, rotation.byAxisAngle(zvector, ang), np.ones(n * n, dtype=np.int8), phaseList(crystalFrame("m-3m", mineral="test")))


def loadMap(name):
  from mtex import EBSD, crystalFrame, phaseList, rotation
  A = np.fromfile(os.path.join(OUT, f"{name}.bin"), "<f8").reshape(-1, 7)
  frames = []
  for line in open(os.path.join(OUT, f"{name}.phases")).read().splitlines():
    mineral, pg, abc, abg, al = line.split("\t")
    kw = dict(mineral=mineral)
    if al:
      kw["alignment"] = al.split(",")
    frames.append(crystalFrame(pg, [float(x) for x in abc.split()], [float(x) for x in abg.split()], **kw))
  pos = np.column_stack([A[:, :2], np.zeros(len(A))])
  e = EBSD(pos, rotation(A[:, 3:7]), A[:, 2].astype(np.int8), phaseList(*frames))
  return e.gridify()


def runPort(names):
  from mtex import calcGrains, degree
  os.makedirs(OUT, exist_ok=True)
  for name in names:
    e0 = loadMap(name)
    for case in CASES_OF[name]:
      o = CASES[case]
      kw = dict(angle=o["angle"] * degree, minPixel=o["minPixel"], alpha=o["alpha"], removeQuadruplePoints=o["removeQuadruplePoints"])
      # calcGrains writes into its map, so every call gets a fresh copy, made outside the clock;
      # the best of three after a warm up
      calcGrains(e0.copy(), **kw)
      t = np.inf
      for _ in range(3):
        e = e0.copy()
        t0 = time.perf_counter()
        g = calcGrains(e, **kw)
        t = min(t, time.perf_counter() - t0)
      writeResult("python", name, case, t, g, e)
      print(f"{name} {case}: {t:.3f} s, {len(g)} grains")


def writeResult(port, name, case, t, g, e):
  ind = np.asarray(g.isIndexed).reshape(-1)
  keep = ~e.isPadding.reshape(-1)
  stats = dict(time=t, grains=len(g), indexed=int(ind.sum()), notIndexed=int((~ind).sum()), boundary=len(g.boundary),
               inner=len(g.innerBoundary), triplePoints=len(g.triplePoints), largest=int(np.max(np.asarray(g.numPixel)[ind])) if ind.any() else 0)
  open(os.path.join(OUT, f"{port}_{name}_{case}.txt"), "w").write("\n".join(f"{k} {v}" for k, v in stats.items()) + "\n")
  xy = e.pos.data.reshape(-1, 3)[keep, :2]
  gid = np.asarray(e.grainId).reshape(-1)[keep].astype(float)
  np.column_stack([xy, gid]).astype("<f8").tofile(os.path.join(OUT, f"{port}_{name}_{case}.bin"))


# MATLAB through the bridge of the frameSymmetry checkout, one map and case per call, so that
# no call comes near the five minutes a bridge call may take
def runMatlab(names):
  import subprocess
  mtexRoot = os.path.dirname(os.path.dirname(HERE))
  bridge = os.path.join(mtexRoot, "frameSymmetry", "docs", "agents", "matlab-bridge")
  master = os.path.join(mtexRoot, "master")
  pre = (f"warning('off', 'MATLAB:rmpath:DirNotFound'); rmpath(genpath('{master}')); "
         f"warning('on', 'MATLAB:rmpath:DirNotFound'); clear classes; rehash; addpath('{os.path.join(HERE, 'matlab')}');")
  env = dict(os.environ, SESSION_NAME="mtexfs")
  for name in names:
    for case in CASES_OF[name]:
      cmd = f"{pre} bench_calcgrains('{OUT}', '{name}', '{case}');"
      r = subprocess.run([os.path.join(bridge, ".venv", "bin", "python"), os.path.join(bridge, "matlab_run.py"), cmd],
                         capture_output=True, text=True, env=env, timeout=300)
      print(r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else f"{name} {case} FAILED\n{r.stdout[-1500:]}{r.stderr[-1500:]}")


# ------------------------------------------------------------------ the comparison

PORTS = ["python", "julia", "matlab"]


def readStats(port, name, case):
  f = os.path.join(OUT, f"{port}_{name}_{case}.txt")
  if not os.path.exists(f):
    return None
  return {k: float(v) for k, v in (l.split() for l in open(f).read().splitlines() if l.strip())}


# the partition of a port on the positions of the input, 0 where the port gives no grain
def readPartition(port, name, case, keys):
  f = os.path.join(OUT, f"{port}_{name}_{case}.bin")
  if not os.path.exists(f):
    return None
  A = np.fromfile(f, "<f8").reshape(-1, 3)
  where = {k: i for i, k in enumerate(keys)}
  gid = np.zeros(len(keys), dtype=np.int64)
  for k, g in zip(positionKeys(A[:, :2]), A[:, 2]):
    i = where.get(k)
    if i is not None:
      gid[i] = int(g) if np.isfinite(g) else 0
  return gid


def positionKeys(xy):
  return [tuple(r) for r in np.round(xy * 1000).astype(np.int64)]


# how two partitions of the same pixels agree: the grains with identical pixel sets, the share of
# pixels lying in such grains, and the adjusted Rand index over the pixels both give a grain
def agreement(g1, g2):
  both = (g1 > 0) & (g2 > 0)
  a, b = g1[both], g2[both]
  # a grain is identical when its pixels form exactly one grain of the other partition
  pairs, n = np.unique(np.column_stack([a, b]), axis=0, return_counts=True)
  size1 = dict(zip(*np.unique(g1[g1 > 0], return_counts=True)))
  size2 = dict(zip(*np.unique(g2[g2 > 0], return_counts=True)))
  same = [(p, c) for p, c in zip(map(tuple, pairs), n) if size1[p[0]] == c and size2[p[1]] == c]
  identical = len(same)
  share = sum(c for _, c in same) / max(both.sum(), 1)
  # the pixels that disagree: each grain of the first matched to the grain of the second most of
  # its pixels lie in, a pixel counted where the match fails or where only one gives a grain
  order = np.argsort(-n)
  partner = {}
  for p, c in zip(pairs[order], n[order]):
    partner.setdefault(p[0], p[1])
  matched = np.array([partner[x] for x in a]) == b if len(a) else np.zeros(0, bool)
  differ = int((~matched).sum() + ((g1 > 0) != (g2 > 0)).sum())
  return identical, share, adjustedRand(a, b), int((g1 > 0).sum() - both.sum()), int((g2 > 0).sum() - both.sum()), differ


def adjustedRand(a, b):
  from scipy.special import comb
  _, ia = np.unique(a, return_inverse=True)
  _, ib = np.unique(b, return_inverse=True)
  cont = np.unique(ia.astype(np.int64) * (ib.max() + 1) + ib, return_counts=True)[1]
  sumComb = comb(cont, 2).sum()
  sa = comb(np.bincount(ia), 2).sum()
  sb = comb(np.bincount(ib), 2).sum()
  expected = sa * sb / comb(len(a), 2)
  top = 0.5 * (sa + sb)
  return 1.0 if top == expected else (sumComb - expected) / (top - expected)


# the largest grains of one partition that the other does not reproduce
def mismatches(g1, g2, k=5):
  out = []
  for gid, size in sorted(zip(*np.unique(g1[g1 > 0], return_counts=True)), key=lambda x: -x[1]):
    other, cnt = np.unique(g2[g1 == gid], return_counts=True)
    main = other[np.argmax(cnt)]
    if len(other) > 1 or (g2 == main).sum() != size:
      out.append((int(gid), int(size), int(main), int((g2 == main).sum()), len(other)))
    if len(out) == k:
      break
  return out


def compare(names=MAPS):
  rows, agree, worst = [], [], None
  for name in names:
    A = np.fromfile(os.path.join(OUT, f"{name}.bin"), "<f8").reshape(-1, 7)
    keys = positionKeys(A[:, :2])
    for case in CASES_OF[name]:
      for port in PORTS:
        s = readStats(port, name, case)
        if s:
          rows.append(f"| {name} | {case} | {port} | {s['time']:.3f} | {int(s['grains'])} | {int(s['indexed'])} | {int(s['notIndexed'])} | "
                      f"{int(s['boundary'])} | {int(s['inner'])} | {int(s['triplePoints'])} | {int(s['largest'])} |")
      parts = {p: readPartition(p, name, case, keys) for p in PORTS}
      for p1, p2 in (("python", "matlab"), ("python", "julia"), ("julia", "matlab")):
        if parts[p1] is None or parts[p2] is None:
          continue
        ident, share, ari, only1, only2, differ = agreement(parts[p1], parts[p2])
        n1 = len(np.unique(parts[p1][parts[p1] > 0]))
        agree.append(f"| {name} | {case} | {p1} - {p2} | {ident} of {n1} | {100 * share:.2f} | {ari:.5f} | {only1} / {only2} | {differ} |")
        if p2 == "matlab" and p1 == "python" and (worst is None or share < worst[0]):
          worst = (share, name, case, mismatches(parts[p1], parts[p2]), mismatches(parts[p2], parts[p1]))
  print("| map | case | port | s | grains | indexed | notIndexed | segments | inner | triple points | largest |")
  print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
  print("\n".join(rows))
  print("\n| map | case | ports | identical grains | pixels in identical grains % | ARI | pixels with a grain in one only | pixels that differ |")
  print("|---|---|---|---:|---:|---:|---:|---:|")
  print("\n".join(agree))
  if worst:
    print(f"\nworst python - matlab: {worst[1]} {worst[2]}, {100 * worst[0]:.2f} %")
    print("python grain, pixels, main matlab grain, its pixels, matlab grains it touches:", worst[3])
    print("matlab grain, pixels, main python grain, its pixels, python grains it touches:", worst[4])


if __name__ == "__main__":
  cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
  if cmd == "export":
    exportMaps(sys.argv[2:] or MAPS)
  elif cmd == "run":
    runPort(sys.argv[2:] or MAPS)
  elif cmd == "matlab":
    runMatlab(sys.argv[2:] or MAPS)
  elif cmd == "compare":
    compare(sys.argv[2:] or MAPS)
