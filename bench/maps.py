# the maps layer against the Julia port: one synthetic map, every verb of M3 to M7 timed
#
#   .venv/bin/python bench/maps.py [n] [grains]      writes the map for bench/maps.jl next to it
#   julia -t auto --project=../mtexJulia/dev bench/maps.jl [n] [grains]
#
# The map is n x n pixels of unit spacing, its grains the Voronoi cells of random seeds,
# each grain with its own orientation about z and a gradient of 0.002 rad per pixel along
# x inside, so that the intragranular quantities have something to measure. The angles are
# written to a binary file that the Julia script reads, so both ports see one map. Every
# timing is the best of three after a warm up on a small map.
from __future__ import annotations

import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

import mtex
from mtex import (EBSD, phaseList, crystalFrame, rotation, zvector, degree, calcGrains, curvatureFilter, dislocationSystem, meanFilter, medianFilter,
                  KuwaharaFilter, splineFilter, l1TVFilter)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
NG = int(sys.argv[2]) if len(sys.argv) > 2 else 800
HERE = os.path.dirname(os.path.abspath(__file__))
CS = crystalFrame("m-3m", mineral="test")


def angles(n, ng, seed=0):
  rng = np.random.default_rng(seed)
  seeds = rng.uniform(0, n, (ng, 2))
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  P = np.stack([i.reshape(-1), j.reshape(-1)], 1).astype(float)
  lab = cKDTree(seeds).query(P)[1]
  return rng.uniform(0, 2 * np.pi, ng)[lab] + 0.002 * P[:, 0]


def mapOf(n, ang):
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  pos = np.stack([i.reshape(-1), j.reshape(-1), np.zeros(n * n)], axis=-1).astype(float)
  return EBSD(pos, rotation.byAxisAngle(zvector, ang), np.ones(n * n, dtype=np.int8), phaseList(CS))


def best(f, k=3):
  t = np.inf
  out = None
  for _ in range(k):
    t0 = time.perf_counter()
    out = f()
    t = min(t, time.perf_counter() - t0)
  return t, out


def run(n, ng, report):
  ang = angles(n, ng)
  if report:
    ang.astype(np.float64).tofile(os.path.join(HERE, f"maps_{n}_{ng}.bin"))
  e = mapOf(n, ang)
  rows = []
  t, g = best(lambda: calcGrains(mapOf(n, ang), angle=5 * degree))
  rows.append(("calcGrains", t, f"{len(g)} grains, {len(g.boundary)} segments"))
  e = mapOf(n, ang)
  g = calcGrains(e, angle=5 * degree)
  kam = e.KAM()
  for name, f in [("simplifyBoundary", lambda: g.simplifyBoundary()), ("refineBoundary", lambda: g.refineBoundary()),
                  ("reduceBoundary", lambda: g.reduceBoundary(2)), ("smoothBoundary laplace 5", lambda: g.smoothBoundary(5)),
                  ("smoothBoundary curvature", lambda: g.smoothBoundary(curvatureFilter())),
                  ("aspectRatio", lambda: g.aspectRatio()), ("hull", lambda: g.hull()), ("paris", lambda: g.paris()),
                  ("caliper", lambda: g.caliper()), ("caliper shortest", lambda: g.caliper("shortest")), ("paror", lambda: g.paror()),
                  ("surfor", lambda: g.surfor()), ("characteristicShape", lambda: g.characteristicShape()),
                  ("gradient oneSided", lambda: e.gradient()), ("gradient full", lambda: e.gradient(stencil="full")),
                  ("curvature", lambda: e.curvature()), ("KAM", lambda: e.KAM()), ("KAM order 3", lambda: e.KAM(order=3)),
                  ("calcGROD", lambda: e.calcGROD(g)), ("grainMean", lambda: e.grainMean(kam, g)),
                  ("calcMisorientation 1e5", lambda: e.calcMisorientation(sampleSize=100000)),
                  ("smooth mean", lambda: e.smooth(meanFilter())), ("smooth median", lambda: e.smooth(medianFilter())),
                  ("smooth Kuwahara", lambda: e.smooth(KuwaharaFilter())), ("smooth spline", lambda: e.smooth(splineFilter())),
                  ("smooth l1TV", lambda: e.smooth(l1TVFilter()))]:
    t, out = best(f)
    rows.append((name, t, ""))
  dS = dislocationSystem.fcc(CS)
  sub = e[np.arange(40 * 40)]
  t, out = best(lambda: sub.calcGND(dS), 1)
  rows.append(("calcGND 1600 px", t, f"{int(np.isfinite(out[0]).sum())} finite"))
  if report:
    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, map {n} x {n}, {ng} seeds")
    print("| verb | s | note |\n|---|---:|---|")
    for name, t, note in rows:
      print(f"| {name} | {t:.3f} | {note} |")


if __name__ == "__main__":
  run(60, 8, False)                              # compile the kernels
  run(N, NG, True)
