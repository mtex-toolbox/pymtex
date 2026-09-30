# the geometry layer against MATLAB and Julia on one input: the rotation products, the
# symmetry-reduced dot, the projection into the fundamental region and the SO(3) grid search,
# each operation timed, profiled and its results compared element by element
#
#   .venv/bin/python bench/geometry_ops.py export               the input, N = 1e6 of each kind
#   .venv/bin/python bench/geometry_ops.py run [op ...]         this port: best of five, one profiled run, the results
#   .venv/bin/python bench/geometry_ops.py matlab [op ...]      bench/matlab/bench_geometry_ops.m through the bridge of ../master
#   .venv/bin/python bench/geometry_ops.py julia [op ...]       bench/geometry_ops.jl, the Julia port on the same input
#   .venv/bin/python bench/geometry_ops.py compare
#
# The input in bench/out/geometry: four sets of unit quaternions A B C D and two of unit vectors
# V W. r = A, s = B, ori and ori2 are A and B in m-3m, mori and mori2 are C and D from m-3m to
# 6/mmm, v = V, m = W in m-3m, the grid queries are the first 1e4 of ori. Each port writes
# <port>_<op>.txt (the best time), <port>_<op>.bin (the first K results as float64 rows) and
# profile_<port>_<op>.json.
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out", "geometry")
sys.path.insert(0, os.path.join(ROOT, "tools", "graph"))
import crosscheck  # noqa: E402

ALGORITHM = [
  {"name": "rotation products",
   "entry": {"matlab": ["@quaternion.times", "@rotation.times", "@quaternion.mtimes"],
             "python": ["mtex.geometry.quaternion.quaternion.__mul__", "mtex.geometry.quaternion.rotation.__mul__"],
             "julia": ["*"]},
   "cases": {"r_s": "r .* s, 1e6 rotations", "r_v": "r .* v, 1e6 rotations and vectors"},
   "tolerance": 1e-12},
  {"name": "symmetry-reduced dot",
   "entry": {"matlab": ["@orientation.dot"], "python": ["mtex.geometry.orientation.orientation.dot"], "julia": ["dot"]},
   "cases": {"dot_ori_ori2": "dot(ori, ori2), m-3m, 1e6 pairs", "dot_ori_ref": "dot(ori, ref), one reference",
             "dot_mori_mori2": "dot(mori, mori2), m-3m -> 6/mmm", "dot_mori_mref": "dot(mori, mref)"},
   "tolerance": 1e-12},
  {"name": "fundamental region projection",
   "entry": {"matlab": ["@orientation.project2FundamentalRegion", "@vector3d.project2FundamentalRegion",
                        "@Miller.projectIntoFundamentalRegion"],
             "python": ["mtex.geometry.symmetryops.projectIntoFundamentalRegion",
                        "mtex.geometry.orientation.orientation.projectIntoFundamentalRegion"],
             "julia": ["project2FundamentalRegion", "batchedSymReduce"]},
   "cases": {"p2fr_m": "projectIntoFundamentalRegion(m), m-3m", "p2fr_ori_ref": "projectIntoFundamentalRegion(ori, ref)",
             "p2fr_ori_ori2": "projectIntoFundamentalRegion(ori, ori2)", "p2fr_mori_mref": "projectIntoFundamentalRegion(mori, mref)",
             "p2fr_mori_mori2": "projectIntoFundamentalRegion(mori, mori2)"},
   "tolerance": 1e-12},
  {"name": "orientation grid search",
   "entry": {"matlab": ["equispacedSO3Grid", "@SO3Grid.find"],
             "python": ["mtex.geometry.grids.so3grid.equispacedSO3Grid", "mtex.geometry.grids.search.find"],
             "julia": ["equispacedSO3Grid", "nearest", "neighbours"]},
   "cases": {"grid_build": "equispacedSO3Grid(m-3m, 2.5 degree)", "grid_nearest": "find(grid, ori), 1e4 queries",
             "grid_radius": "find(grid, ori, 7.5 degree), 2000 queries"},
   "tolerance": 2e-2},
]
N, K, NQ, NR = 10**6, 20000, 10**4, 2000


def export():
  os.makedirs(OUT, exist_ok=True)
  rng = np.random.default_rng(0)
  for name, k in (("A", 4), ("B", 4), ("C", 4), ("D", 4), ("V", 3), ("W", 3)):
    x = rng.standard_normal((N, k))
    (x / np.linalg.norm(x, axis=1, keepdims=True)).astype("<f8").tofile(os.path.join(OUT, f"{name}.bin"))


def inputs():
  import mtex
  load = lambda name, k: np.fromfile(os.path.join(OUT, f"{name}.bin"), "<f8").reshape(-1, k)
  cs, hx = mtex.crystalFrame("m-3m"), mtex.crystalFrame("6/mmm")
  A, B, C, D = (load(x, 4) for x in "ABCD")
  x = dict(r=mtex.rotation(A), s=mtex.rotation(B), v=mtex.vector3d(load("V", 3)),
           ori=mtex.orientation(mtex.rotation(A), cs), ori2=mtex.orientation(mtex.rotation(B), cs),
           mori=mtex.orientation(mtex.rotation(C), cs, hx), mori2=mtex.orientation(mtex.rotation(D), cs, hx),
           m=mtex.Miller(mtex.vector3d(load("W", 3)), cs), cs=cs)
  x["ref"], x["mref"] = x["ori"][0], x["mori"][0]
  return x


# the operations: the call, and the first K results as float64 rows
def operations(x):
  import mtex
  deg = mtex.degree
  grid = {}

  def build():
    grid["g"] = mtex.equispacedSO3Grid(x["cs"], resolution=2.5 * deg)
    return grid["g"]

  rows = lambda a, k: np.asarray(a.data if hasattr(a, "data") else a, dtype=float).reshape(-1, k)[:K]
  return {
    "r_s": (lambda: x["r"] * x["s"], lambda o: rows(o, 4)),
    "r_v": (lambda: x["r"] * x["v"], lambda o: rows(o, 3)),
    "dot_ori_ori2": (lambda: mtex.dot(x["ori"], x["ori2"]), lambda o: rows(o, 1)),
    "dot_ori_ref": (lambda: mtex.dot(x["ori"], x["ref"]), lambda o: rows(o, 1)),
    "dot_mori_mori2": (lambda: mtex.dot(x["mori"], x["mori2"]), lambda o: rows(o, 1)),
    "dot_mori_mref": (lambda: mtex.dot(x["mori"], x["mref"]), lambda o: rows(o, 1)),
    "p2fr_m": (lambda: mtex.projectIntoFundamentalRegion(x["m"]), lambda o: rows(o, 3)),
    "p2fr_ori_ref": (lambda: mtex.projectIntoFundamentalRegion(x["ori"], x["ref"]), lambda o: rows(o, 4)),
    "p2fr_ori_ori2": (lambda: mtex.projectIntoFundamentalRegion(x["ori"], x["ori2"]), lambda o: rows(o, 4)),
    "p2fr_mori_mref": (lambda: mtex.projectIntoFundamentalRegion(x["mori"], x["mref"]), lambda o: rows(o, 4)),
    "p2fr_mori_mori2": (lambda: mtex.projectIntoFundamentalRegion(x["mori"], x["mori2"]), lambda o: rows(o, 4)),
    "grid_build": (build, lambda o: np.array([[o.size]], dtype=float)),
    # the grids order their points differently: the distance to the nearest point is compared
    # find returns the cosine of half the distance, compared as the angle as in MATLAB's half
    "grid_nearest": (lambda: mtex.find(grid["g"], x["ori"][:NQ]), lambda o: rows(2 * np.arccos(np.minimum(np.asarray(o[1]), 1)), 1)),
    "grid_radius": (lambda: mtex.find(grid["g"], x["ori"][:NR], 7.5 * deg),
                    lambda o: np.diff(o[0].tocsr().indptr).astype(float).reshape(-1, 1)),
  }


def run(ops):
  crosscheck.run(OUT, ALGORITHM, operations(inputs()), ops, always=("grid_build",))


# the Julia half, bench/geometry_ops.jl on the same input
def julia(ops):
  crosscheck.julia("geometry_ops.jl", ALGORITHM, ops)


def deviation(op, a, b):
  # the grids of the ports differ in their points, so a grid result is compared by its mean; the
  # nearest distance by its median, since MATLAB's search misses the nearest point of about one
  # query in a hundred (up to 30 degree on the 2.5 degree grid), counted separately
  if op == "grid_nearest":
    print(f"  grid_nearest: largest distance {np.degrees(np.max(a)):.2f} / {np.degrees(np.max(b)):.2f} degree, "
          f"{int(np.sum(b > np.max(a)))} MATLAB queries beyond this port's largest")
    return abs(float(np.median(a) - np.median(b))) / abs(float(np.median(b)))
  if op.startswith("grid"):
    return abs(float(np.mean(a) - np.mean(b))) / abs(float(np.mean(b)))
  if op in ("r_s",) or op.startswith("p2fr_") and a.shape[1] == 4:
    return float(np.max(1 - np.abs(np.sum(a * b, axis=1))))   # a quaternion and its negative are one rotation
  return float(np.max(np.abs(a - b)))


def cols(op):
  return 1 if op.startswith(("dot", "grid")) else 3 if op in ("r_v", "p2fr_m") else 4


if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "run", sys.argv[2:] or [c for a in ALGORITHM for c in a["cases"]]
  {"export": export, "run": lambda: run(names), "julia": lambda: julia(sys.argv[2:]),
   "matlab": lambda: crosscheck.matlab("bench_geometry_ops", repr(OUT), crosscheck.cellOf(names), str(K), str(NQ), str(NR)),
   "compare": lambda: crosscheck.compare(OUT, ALGORITHM, cols, deviation)}[cmd]()
