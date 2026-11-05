# calcGrains against MATLAB develop on measured and synthetic maps: the closing with its corridors,
# the pinches, the minPixel loop and the boundaries drawn straight through the filled cells, or along
# the bisectors (straighten=False)
#
#   .venv/bin/python bench/calcgrains_ops.py export              the crack instance of the grain benchmark
#   .venv/bin/python bench/calcgrains_ops.py run [op ...]        this port
#   .venv/bin/python bench/calcgrains_ops.py matlab [op ...]     bench/matlab/bench_calcgrains_ops.m through the bridge
#   .venv/bin/python bench/calcgrains_ops.py compare
#
# An op is <map>[_<alpha>][_bisectors]: small at 10 degree and minPixel 5, peridotite at alpha 2.1, 6.1
# and 16.1 with minPixel 5, forsterite with minPixel 1 and 5, the T1 instance with cracks at 5 degree.
# The ports number grains their own way, so a result is the grain count, then every segment between
# two grains, of both boundaries, as rows (x1, y1, x2, y2), matched by midpoint. The outer boundary is
# left out: MATLAB's vertices at the map corners lie up to 1e-3 pixels off the lattice.
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "graph"))
import crosscheck  # noqa: E402

OUT = os.path.join(HERE, "out", "calcgrains_ops")
# MTEX's checkout beside this one, or MTEX_ROOT where this one is a worktree elsewhere
PERIDOTITE = os.path.join(os.environ.get("MTEX_ROOT", os.path.join(os.path.dirname(crosscheck.ROOT), "master")), "data", "EBSD", "peridotite.ctf")
CRACKS = "T1_clean_cracks"
MAPS = {"small": ("small", dict(angle=10, minPixel=5)), "peridotite_2.1": ("peridotite", dict(alpha=2.1, minPixel=5)),
        "peridotite_6.1": ("peridotite", dict(alpha=6.1, minPixel=5)), "peridotite_16.1": ("peridotite", dict(alpha=16.1, minPixel=5)),
        "forsterite": ("forsterite", {}), "forsterite_5": ("forsterite", dict(minPixel=5)), "cracks": (CRACKS, dict(angle=5))}
OPS = [m + s for m in MAPS for s in ("", "_bisectors")]
STRAIGHT = "the straightened vertices: the cut points are bisections whose midpoint falls on a lattice edge, decided by 1e-14 of either port"
ALGORITHM = [
  {"name": "grain segmentation",
   "entry": {"matlab": ["@EBSD.calcGrains", "spatialDecompositionGrid", "splitPinches", "remapIFD"],
             "python": ["mtex.maps.calcgrains.calcGrains", "mtex.maps.decomposition.decompose", "mtex.maps.decomposition.splitPinches",
                        "mtex.maps.decomposition.joinHoles"]},
   "cases": {op: f"calcGrains of {MAPS[op[:-10]][0]} {MAPS[op[:-10]][1]}, along the bisectors" for op in OPS if op.endswith("_bisectors")},
   # every vertex at the same place up to rounding, 1e-13 of coordinates up to 3e4
   "tolerance": 3e-9},
  {"name": "boundary straightening",
   "entry": {"matlab": ["@EBSD.calcGrains", "straightenInFill"], "python": ["mtex.maps.calcgrains.calcGrains", "mtex.maps.straighten.straightenInFill"]},
   "cases": {op: f"calcGrains of {MAPS[op][0]} {MAPS[op][1]}, straight through the filled cells" for op in OPS if not op.endswith("_bisectors")},
   # STRAIGHT: a cut point moves by up to 1/2^17 of its segment; where MATLAB's swept-area samples,
   # ceil(3 len / dxy) of a segment of exactly one pixel, flip with the rounding of len, a vertex is held
   # in one port and moved in the other (forsterite: one chain of five segments, peridotite none)
   "tolerance": {op: None for op in OPS}},
]


def export():
  sys.path.insert(0, os.path.join(HERE, "grains"))
  import run
  os.makedirs(OUT, exist_ok=True)
  if not os.path.exists(os.path.join(run.OUT, "T1", CRACKS + ".bin")):
    run.generate("T1", [CRACKS])


def load(name):
  from mtex import loadEBSD, mtexdata
  if name == "peridotite":
    return loadEBSD(PERIDOTITE)
  if name == CRACKS:
    sys.path.insert(0, os.path.join(HERE, "grains"))
    import run
    return run.loadMap("T1", CRACKS)
  return mtexdata(name, verbose=False)


# the grain count, then the segments of both boundaries between two grains
def result(g):
  rows = [[len(g), int(np.sum(g.isIndexed)), 0, 0]]
  for b in (g.boundary, g.innerBoundary):
    V, F = np.asarray(b.V, float)[:, :2], np.asarray(b.F)
    ok = (b.grainId > 0).all(axis=1)
    rows.append(np.column_stack([V[F[ok, 0]], V[F[ok, 1]]]))
  return np.vstack(rows)


def run(ops):
  from mtex import calcGrains, degree
  maps = {m: load(m) for m in {MAPS[op.removesuffix("_bisectors")][0] for op in ops}}

  def call(op):
    name, kw = MAPS[op.removesuffix("_bisectors")]
    kw = {k: v * degree if k == "angle" else v for k, v in kw.items()}
    return lambda: calcGrains(maps[name].copy(), straighten=not op.endswith("_bisectors"), **kw)

  crosscheck.run(OUT, ALGORITHM, {op: (call(op), result) for op in OPS}, ops, reps=3)


def matlab(ops):
  crosscheck.matlab("bench_calcgrains_ops", repr(OUT), crosscheck.cellOf(ops), "1e7")


# the grain counts, then the segments matched by their midpoints: the largest distance of an
# endpoint from its partner, either way round
def deviation(op, a, b):
  from scipy.spatial import cKDTree
  if not np.array_equal(a[0], b[0]) or len(a) != len(b):
    print(f"  {op}: grains {a[0, :2]} and {b[0, :2]}, {len(a) - 1} and {len(b) - 1} segments")
    return float("nan")
  a, b = a[1:], b[1:]
  d, j = cKDTree((b[:, :2] + b[:, 2:]) / 2).query((a[:, :2] + a[:, 2:]) / 2)
  b = b[j]
  same = np.maximum(np.linalg.norm(a[:, :2] - b[:, :2], axis=1), np.linalg.norm(a[:, 2:] - b[:, 2:], axis=1))
  turned = np.maximum(np.linalg.norm(a[:, :2] - b[:, 2:], axis=1), np.linalg.norm(a[:, 2:] - b[:, :2], axis=1))
  dev = np.minimum(same, turned)
  if len(set(j)) != len(j):
    print(f"  {op}: {len(j) - len(set(j))} segments without a partner")
  print(f"  {op}: {len(a)} segments, {int((dev > 1e-9).sum())} apart by more than 1e-9, median of those {np.median(dev[dev > 1e-9]) if (dev > 1e-9).any() else 0:.1e}")
  return float(dev.max())


if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "compare", sys.argv[2:] or OPS
  {"export": export, "run": lambda: run(names), "matlab": lambda: matlab(names),
   "compare": lambda: crosscheck.compare(OUT, ALGORITHM, lambda op: 4, deviation)}[cmd]()
