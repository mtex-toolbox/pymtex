# the boundary smoothing against MATLAB on one synthetic map: the staircase calcGrains draws, the
# default smoothBoundary (Taubin pairs with the vertices held in their pixel cells, then the side
# constrained simplification), the same without the simplification, and Douglas-Peucker on the
# staircase itself, where every tie of the walk independent rule is exercised
#
#   .venv/bin/python bench/smoothing_ops.py export              the map
#   .venv/bin/python bench/smoothing_ops.py run [op ...]        this port
#   .venv/bin/python bench/smoothing_ops.py matlab [op ...]     bench/matlab/bench_smoothing_ops.m through the bridge
#   .venv/bin/python bench/smoothing_ops.py julia [op ...]      bench/smoothing_ops.jl, the Julia port on the same input
#   .venv/bin/python bench/smoothing_ops.py compare
#
# The map: N x N pixels of unit spacing, the grains the Voronoi cells of random seeds, each with its
# orientation about z and a gradient of 0.003 rad per pixel along x inside, which leaves inner
# boundaries with loose ends, and discs of notIndexed pixels inside the map, which the fill model
# gives to the grains around them; written as rows (x, y, angle, phase), phase 0 notIndexed. MATLAB segments
# with 'straighten', false, the bisectors through filled cells the other ports draw. The ports number
# grains their own way, so a result is every segment between two grains, of both boundaries, as rows
# (x1, y1, x2, y2), matched by midpoint.
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "graph"))
import crosscheck  # noqa: E402

OUT = os.path.join(HERE, "out", "smoothing")
N, NG, NHOLES = 160, 150, 12
ALGORITHM = [
  {"name": "grain segmentation",
   "entry": {"matlab": ["@EBSD.calcGrains"], "python": ["mtex.maps.calcgrains.calcGrains"], "julia": ["calcGrains!"]},
   "cases": {"staircase": "calcGrains at 5 degree with notIndexed discs: the boundaries between grains, the input of the others"},
   "tolerance": 1e-9},
  {"name": "boundary smoothing",
   "entry": {"matlab": ["@grain2d.smoothBoundary", "taubinFilter", "@grainBoundary.simplify"],
             "python": ["mtex.maps.smoothing.smoothBoundary", "mtex.maps.smoothing.pixelCells.project",
                        "mtex.maps.boundaryresampling.simplifyBoundary"],
             "julia": ["smoothBoundary"]},
   "cases": {"smooth": "smoothBoundary(grains): 20 Taubin pairs in the pixel cells, the areas restored, simplified at d/10 with the margin d/20",
             "smooth_noSimplify": "smoothBoundary(grains, 'noSimplify'): the Taubin pairs in the cells and the areas restored",
             "simplify_staircase": "simplifyBoundary(grains) of the staircase at d/sqrt(2), decided by ties"},
   # every vertex kept by both at the same place up to rounding, but for the area restoration: MATLAB's
   # convhull holds only the corners and junctions of the map rim, so its area step moves the rim's
   # plain vertices by up to 0.02 px where the port holds them, and the simplification then keeps a
   # vertex or two differently (develop-fixes.md)
   "tolerance": {"smooth": None, "smooth_noSimplify": 0.02, "simplify_staircase": 1e-9}},
]
OPS = ["staircase", "smooth", "smooth_noSimplify", "simplify_staircase"]


def export():
  from scipy.spatial import cKDTree
  os.makedirs(OUT, exist_ok=True)
  rng = np.random.default_rng(1)
  i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="xy")
  P = np.column_stack([i.reshape(-1), j.reshape(-1)]).astype(float)
  grain = cKDTree(rng.uniform(0, N, (NG, 2))).query(P)[1]
  ang = rng.uniform(0, 2 * np.pi, NG)[grain] + 0.003 * P[:, 0]
  holes = cKDTree(rng.uniform(10, N - 10, (NHOLES, 2))).query(P)[0] <= 2
  np.column_stack([P, ang, (~holes).astype(float)]).astype("<f8").tofile(os.path.join(OUT, "map.bin"))


# the segments of both boundaries between two grains
def inner(g):
  rows = []
  for b in (g.boundary, g.innerBoundary):
    V, F = np.asarray(b.V, float)[:, :2], np.asarray(b.F)
    ok = (b.grainId > 0).all(axis=1)
    rows.append(np.column_stack([V[F[ok, 0]], V[F[ok, 1]]]))
  return np.vstack(rows)


def run(ops):
  from mtex import EBSD, calcGrains, crystalFrame, degree, phaseList, rotation, smoothBoundary, zvector
  A = np.fromfile(os.path.join(OUT, "map.bin"), "<f8").reshape(-1, 4)
  pos = np.column_stack([A[:, :2], np.zeros(len(A))])
  e = EBSD(pos, rotation.byAxisAngle(zvector, A[:, 2]), A[:, 3].astype(np.int8), phaseList(crystalFrame("m-3m", mineral="test")))
  s = {}

  def grains():
    s["g"] = calcGrains(e.copy(), angle=5 * degree, straighten=False)
    return s["g"]

  table = {"staircase": (grains, inner),
           "smooth": (lambda: smoothBoundary(s["g"]), inner),
           "smooth_noSimplify": (lambda: smoothBoundary(s["g"], simplify=False), inner),
           "simplify_staircase": (lambda: s["g"].simplifyBoundary(), inner)}
  crosscheck.run(OUT, ALGORITHM, table, ops, always=("staircase",))


def matlab(ops):
  crosscheck.matlab("bench_smoothing_ops", repr(OUT), crosscheck.cellOf(ops), "1e7")


# the segments matched by their midpoints; the largest distance of an endpoint from its partner,
# either way round
def deviation(op, a, b):
  from scipy.spatial import cKDTree
  if len(a) != len(b):
    print(f"  {op}: {len(a)} and {len(b)} segments")
    return float("nan")
  d, j = cKDTree((b[:, :2] + b[:, 2:]) / 2).query((a[:, :2] + a[:, 2:]) / 2)
  if d.max() > 0.05 or len(set(j)) != len(j):
    print(f"  {op}: {int((d > 0.05).sum())} segments without a partner within a twentieth of a pixel")
    return float("nan")
  b = b[j]
  same = np.maximum(np.linalg.norm(a[:, :2] - b[:, :2], axis=1), np.linalg.norm(a[:, 2:] - b[:, 2:], axis=1))
  turned = np.maximum(np.linalg.norm(a[:, :2] - b[:, 2:], axis=1), np.linalg.norm(a[:, 2:] - b[:, :2], axis=1))
  return float(np.minimum(same, turned).max())


if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "compare", sys.argv[2:] or OPS
  {"export": export, "run": lambda: run(names), "matlab": lambda: matlab(names),
   "julia": lambda: crosscheck.julia("smoothing_ops.jl", ALGORITHM, names),
   "compare": lambda: crosscheck.compare(OUT, ALGORITHM, lambda op: 4, deviation)}[cmd]()
