# the maps layer against MATLAB on one synthetic map: grain segmentation, grain shape, boundary
# smoothing, the intragranular misorientations and the denoising of the orientations
#
#   .venv/bin/python bench/maps_ops.py export              the map
#   .venv/bin/python bench/maps_ops.py run [op ...]        this port
#   .venv/bin/python bench/maps_ops.py matlab [op ...]     bench/matlab/bench_maps_ops.m through the bridge
#   .venv/bin/python bench/maps_ops.py compare
#
# The map of bench/maps.py: N x N pixels of unit spacing, the grains the Voronoi cells of random
# seeds, each with its orientation about z and a gradient of 0.002 rad per pixel along x inside,
# and on top a noise of 1 degree per axis, written as rows (x, y, true quaternion, noisy
# quaternion); the ports work on the noisy one, and the denoising is also scored against the true.
# The large map is the 2.5 million pixels of steel 1C_1 as bench/calcgrains.py exports it. The
# ports number grains and pixels their own way, so every result row starts with a key, the grain's
# centroid or the pixel's position, and compare matches by it.
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "graph"))
import crosscheck  # noqa: E402

OUT = os.path.join(HERE, "out", "maps")
N, NG = 300, 200
ALGORITHM = [
  {"name": "grain segmentation",
   "entry": {"matlab": ["@EBSD.calcGrains"], "python": ["mtex.maps.calcgrains.calcGrains"]},
   "cases": {"calcgrains": "calcGrains of the 300 x 300 map at 5 degree: the area of every grain",
             "calcgrains_1C_1": "calcGrains of the 2.5 million pixel steel map 1C_1 at 10 degree, one run"},
   # MATLAB's boundary vertices sit up to 1e-6 off the pixel lattice
   "tolerance": 1e-5},
  {"name": "grain shape",
   "entry": {"matlab": ["@grain2d.perimeter", "@grain2d.aspectRatio"],
             "python": ["mtex.maps.polygon.polygonSet.perimeter", "mtex.maps.shape.aspectRatio"]},
   "cases": {"perimeter": "the perimeter of every grain", "aspect": "aspectRatio of every grain"},
   "tolerance": 1e-5},
  {"name": "boundary smoothing",
   "entry": {"matlab": ["@grain2d.smoothBoundary"], "python": ["mtex.maps.smoothing.smoothBoundary"]},
   "cases": {"smooth_boundary": "smoothBoundary(5): the perimeter of every grain after"},
   "tolerance": 1e-5},
  {"name": "intragranular misorientation",
   "entry": {"matlab": ["@EBSD.KAM", "@EBSD.calcGROD"],
             "python": ["mtex.maps.intragranular.KAM", "mtex.maps.intragranular.calcGROD"]},
   "cases": {"kam": "KAM, first order, every pixel", "grod": "the angle of calcGROD, every pixel"},
   # MATLAB takes small angles through acos
   "tolerance": 1e-6},
  {"name": "orientation denoising",
   "entry": {"matlab": ["@EBSD.smooth", "@l1TVFilter.smooth"],
             "python": ["mtex.analysis.denoising.smooth", "mtex.analysis.variational.l1TVFilter.smooth"]},
   "cases": {"smooth_mean": "smooth(meanFilter), every pixel", "smooth_spline": "smooth(splineFilter), every pixel",
             "smooth_l1tv": "smooth(l1TVFilter), every pixel"},
   # the misorientation angle in radians
   "tolerance": 1e-6},
]
COLS = {"calcgrains": 3, "calcgrains_1C_1": 3, "perimeter": 3, "aspect": 3, "smooth_boundary": 3, "kam": 3,
        "grod": 3, "smooth_mean": 6, "smooth_spline": 6, "smooth_l1tv": 6}
LARGE = os.path.join(HERE, "out", "calcgrains", "1C_1.bin")


def export():
  from scipy.spatial import cKDTree
  os.makedirs(OUT, exist_ok=True)
  rng = np.random.default_rng(0)
  i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="xy")
  P = np.column_stack([i.reshape(-1), j.reshape(-1)]).astype(float)
  grain = cKDTree(rng.uniform(0, N, (NG, 2))).query(P)[1]
  ang = rng.uniform(0, 2 * np.pi, NG)[grain] + 0.002 * P[:, 0]
  q = np.column_stack([np.cos(ang / 2), 0 * ang, 0 * ang, np.sin(ang / 2)])
  # the noise: a rotation of normally distributed rotation vector, 1 degree per axis, from the right
  r = rng.normal(0, np.radians(1), (len(P), 3))
  w = np.linalg.norm(r, axis=1)
  n = np.column_stack([np.cos(w / 2), np.sin(w / 2)[:, None] * r / w[:, None]])
  a1, b1, c1, d1 = q.T
  a2, b2, c2, d2 = n.T
  noisy = np.column_stack([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                           a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2])
  np.column_stack([P, q, noisy]).astype("<f8").tofile(os.path.join(OUT, "map.bin"))
  if not os.path.exists(LARGE):
    subprocess.run([sys.executable, os.path.join(HERE, "calcgrains.py"), "export", "1C_1"], check=True)


def run(ops):
  from mtex import EBSD, calcGrains, crystalFrame, degree, l1TVFilter, meanFilter, phaseList, rotation, splineFilter
  sys.path.insert(0, HERE)
  import calcgrains
  A = np.fromfile(os.path.join(OUT, "map.bin"), "<f8").reshape(-1, 10)
  pos = np.column_stack([A[:, :2], np.zeros(len(A))])
  e = EBSD(pos, rotation(A[:, 6:10]), np.ones(len(A), dtype=np.int8), phaseList(crystalFrame("m-3m", mineral="test")))
  xy = np.asarray(e.pos.data, float).reshape(-1, 3)[:, :2]
  s = {}

  def grains():
    s["g"] = calcGrains(e, angle=5 * degree)
    s["c"] = np.asarray(s["g"].centroid.data, float).reshape(-1, 3)[:, :2]
    return s["g"]

  byGrain = lambda v: np.column_stack([s["c"], np.asarray(v, float).reshape(-1)])
  byPixel = lambda v: np.column_stack([xy, np.asarray(v, float).reshape(-1, np.asarray(v).size // len(xy))])
  # a smoothed map comes back on its grid, in the grid's order
  quat = lambda m: np.column_stack([np.asarray(m.pos.data, float).reshape(-1, 3)[:, :2],
                                    np.asarray(m.rotations.data, float).reshape(-1, 4)])
  def large():
    g = calcGrains(s["large"], angle=10 * degree)
    return np.column_stack([np.asarray(g.centroid.data, float).reshape(-1, 3)[:, :2], np.asarray(g.area, float)])

  s["large"] = calcgrains.loadMap("1C_1") if "calcgrains_1C_1" in ops else None
  table = {
    "calcgrains_1C_1": (large, lambda r: r),
    "calcgrains": (grains, lambda g: byGrain(g.area)),
    "perimeter": (lambda: s["g"].perimeter, byGrain),
    "aspect": (lambda: s["g"].aspectRatio(), byGrain),
    "smooth_boundary": (lambda: s["g"].smoothBoundary(5), lambda g: byGrain(g.perimeter)),
    "kam": (lambda: e.KAM(), byPixel),
    "grod": (lambda: e.calcGROD(s["g"]).angle(), byPixel),
    "smooth_mean": (lambda: e.smooth(meanFilter()), quat),
    "smooth_spline": (lambda: e.smooth(splineFilter()), quat),
    "smooth_l1tv": (lambda: e.smooth(l1TVFilter()), quat),
  }
  crosscheck.run(OUT, ALGORITHM, table, ops, reps={"calcgrains_1C_1": 1}, always=("calcgrains",))


# rows matched by their keys (the first two columns, a centroid or a position) to the nearest within
# 1e-3; orientations compared by their misorientation angle in m-3m, the other values relative to the largest
def deviation(op, a, b):
  from scipy.spatial import cKDTree
  d, j = cKDTree(b[:, :2]).query(a[:, :2])
  if len(a) != len(b) or d.max() > 1e-3 or len(set(j)) != len(j):
    return float("nan")
  va, vb = a[:, 2:], b[j, 2:]
  if va.shape[1] == 4:
    from mtex import angle, crystalFrame, orientation, rotation
    cs = crystalFrame("m-3m")
    return float(np.max(np.asarray(angle(orientation(rotation(va), cs), orientation(rotation(vb), cs)))))
  both = ~(np.isnan(va) | np.isnan(vb))
  if (np.isnan(va) != np.isnan(vb)).any():
    print(f"  {op}: NaN in {int(np.isnan(va).sum())} Python and {int(np.isnan(vb).sum())} MATLAB rows")
  return float(np.max(np.abs(va - vb)[both]) / np.max(np.abs(vb[both])))


# the table, and every denoising scored against the true orientations
def compare():
  from scipy.spatial import cKDTree
  from mtex import angle, crystalFrame, orientation, rotation
  crosscheck.compare(OUT, ALGORITHM, lambda op: COLS[op], deviation)
  A = np.fromfile(os.path.join(OUT, "map.bin"), "<f8").reshape(-1, 10)
  cs = crystalFrame("m-3m")
  truth = orientation(rotation(A[:, 2:6]), cs)
  mean = lambda q, idx: np.degrees(np.mean(np.asarray(angle(orientation(rotation(q), cs), truth[idx]))))
  print(f"mean misorientation to the true map, degree: noisy {mean(A[:, 6:10], np.arange(len(A))):.3f}")
  tree = cKDTree(A[:, :2])
  for op in ("smooth_mean", "smooth_spline", "smooth_l1tv"):
    line = f"  {op:18s}"
    for port in ("python", "matlab"):
      f = os.path.join(OUT, f"{port}_{op}.bin")
      if os.path.exists(f):
        r = np.fromfile(f, "<f8").reshape(-1, 6)
        line += f"   {port} {mean(r[:, 2:], tree.query(r[:, :2])[1]):.3f}"
    print(line)


if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "compare", sys.argv[2:] or list(COLS)
  {"export": export, "run": lambda: run(names),
   "matlab": lambda: crosscheck.matlab("bench_maps_ops", repr(OUT), crosscheck.cellOf(names), "1e7", repr(LARGE)),
   "compare": compare}[cmd]()
