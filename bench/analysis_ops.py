# the analysis layer against MATLAB where no other benchmark reaches it: the geometrically
# necessary dislocations, the grain boundary normal distribution, the Schmid factors, the
# birefringence, the clustering of orientations and the optimal sampling of a density
#
#   .venv/bin/python bench/analysis_ops.py export              the input
#   .venv/bin/python bench/analysis_ops.py run [op ...]        this port
#   .venv/bin/python bench/analysis_ops.py matlab [op ...]     bench/matlab/bench_analysis_ops.m through the bridge
#   .venv/bin/python bench/analysis_ops.py compare
#
# The input in bench/out/analysis: 1e5 orientations O and directions V, and four clouds of 500
# orientations each, 3 degree about four centres, in 432. The maps are read by each port itself:
# dc06 for calcGND as the GND page takes it, unsmoothed, and twins with its grains at 5 degree for
# calcGBND as the BoundaryNormalDistribution page. The clustering is compared by the centres, each
# port's nearest to the other's, the optimal sample by its harmonic moments up to the bandwidth,
# since the nodes themselves depend on the start.
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "graph"))
import crosscheck  # noqa: E402

OUT = os.path.join(HERE, "out", "analysis")
ALGORITHM = [
  {"name": "dislocation density",
   "entry": {"matlab": ["@EBSD.calcGND", "@EBSD.curvature", "@curvatureTensor.fitDislocationSystems"],
             "python": ["mtex.analysis.dislocations.calcGND", "mtex.analysis.dislocations.fitDislocationSystems"]},
   "cases": {"gnd": "calcGND(ebsd, dS) of the dc06 map, bcc edge and screw systems, every pixel"},
   # the densities are minima of a linear program, unique in their sum
   "tolerance": 1e-6},
  {"name": "grain boundary normal distribution",
   "entry": {"matlab": ["@grainBoundary.calcGBND"], "python": ["mtex.analysis.gbnd.calcGBND"]},
   "cases": {"gbnd": "calcGBND(gB, ebsd, 5 degree, 'nonneg') of the twins map at 2000 directions"},
   # the boundaries differ in three segments of 3162 where they meet the border of the map
   "tolerance": 1e-3},
  {"name": "Schmid factor",
   "entry": {"matlab": ["@slipSystem.SchmidFactor"], "python": ["mtex.geometry.slipsystem.slipSystem.SchmidFactor"]},
   "cases": {"schmid": "sS.SchmidFactor(inv(ori) * x) of the 24 fcc systems, both slip directions, 1e5 orientations"},
   # the ports list the systems in their own order: every row compared sorted
   "tolerance": 1e-12},
  {"name": "birefringence",
   "entry": {"matlab": ["@refractiveIndexTensor.birefringence"], "python": ["mtex.analysis.optics.birefringence"]},
   "cases": {"birefringence": "the birefringence of the forsterite-fayalite mixture of the BirefringenceDemo page, 1e5 directions"},
   # MATLAB's eig2 is 1.3e-11 off the extended precision difference of the two indices, this port 1e-15
   "tolerance": 1e-9},
  {"name": "clustering",
   "entry": {"matlab": ["@orientation.calcCluster"], "python": ["mtex.analysis.cluster.calcCluster"]},
   "cases": {"cluster": "calcCluster(ori, 'numCluster', 4, 'method', 'hierarchical'), 2000 orientations in 432"},
   # the centres are means of clusters that may differ by a few border orientations
   "tolerance": 1e-2},
  {"name": "discrete sampling",
   "entry": {"matlab": ["@S2Fun.optimalSample"], "python": ["mtex.analysis.optimalsample.optimalSample"]},
   "cases": {"optimal_s2": "optimalSample(1 + 0.6 z, 300, 'bandwidth', 16): the moments of the nodes"},
   # both are approximations of one density, apart by what each misses of it
   "tolerance": 2e-2},
]
N, NG = 10**5, 2000
COLS = {"gnd": 3, "gbnd": 1, "schmid": 24, "birefringence": 1, "cluster": 4, "optimal_s2": 3}
# 1 + 0.6 z in the coefficients of degree 0 and 1
DENSITY = [np.sqrt(4 * np.pi), 0, 0.6 * np.sqrt(4 * np.pi / 3), 0]


def export():
  from mtex import crystalFrame, degree, orientation, rotation, vector3d
  os.makedirs(OUT, exist_ok=True)
  rng = np.random.default_rng(0)
  unit = lambda n, k: (lambda x: x / np.linalg.norm(x, axis=1, keepdims=True))(rng.standard_normal((n, k)))
  for name, k in (("O", 4), ("V", 3)):
    unit(N, k).astype("<f8").tofile(os.path.join(OUT, f"{name}.bin"))
  centres = rotation.byEuler(np.array([0, 40, 80, 20]) * degree, np.array([0, 30, 60, 80]) * degree, np.array([0, 10, 50, 70]) * degree)
  C = rotation.cat(*[rotation(centres.data[k]) * rotation.byAxisAngle(vector3d(unit(500, 3)), 3 * degree * np.abs(rng.standard_normal(500)))
                     for k in range(4)])
  np.asarray(C.data, "<f8").reshape(-1, 4).tofile(os.path.join(OUT, "C.bin"))
  # the directions calcGBND is evaluated at
  unit(NG, 3).astype("<f8").tofile(os.path.join(OUT, "G.bin"))


def load(name, k):
  return np.fromfile(os.path.join(OUT, f"{name}.bin"), "<f8").reshape(-1, k)


def run(ops):
  from mtex import (S2FunHarmonic, calcCluster, calcGBND, calcGND, calcGrains, crystalFrame, degree, dislocationSystem, inv,
                    mtexdata, optimalSample, orientation, refractiveIndexTensor, rotation, slipSystem, vector3d, xvector)
  s = {}
  if "gnd" in ops:
    s["dc06"] = mtexdata("dc06")
  if "gbnd" in ops:
    tw = mtexdata("twins")
    g = calcGrains(tw, angle=5 * degree, minPixel=3)
    s["twins"], s["gB"] = tw, g.boundary["indexed"]

  def gnd():
    e = s["dc06"]
    dS = dislocationSystem.bcc(e.CS)
    dS.u[dS.isEdge] = 1
    dS.u[dS.isScrew] = 1 - 0.3
    return np.column_stack([np.asarray(e.pos.data, float).reshape(-1, 3)[:, :2], np.asarray(calcGND(e, dS)[0], float).reshape(-1)])

  cub = crystalFrame("m-3m")
  ori = orientation(rotation(load("O", 4)), cub)
  sS = slipSystem.fcc(cub).symmetrise()
  ol = crystalFrame("mmm", [4.756, 10.207, 5.980], mineral="Forsterite")
  rI = 0.86 * refractiveIndexTensor(np.diag([1.670, 1.635, 1.651]), ol) + 0.14 * refractiveIndexTensor(np.diag([1.879, 1.827, 1.869]), ol)
  col = lambda x: np.asarray(x, float).reshape(-1, 1)
  quat = lambda o: np.asarray(o.data, float).reshape(-1, 4)
  table = {
    "gnd": (gnd, lambda r: r),
    "gbnd": (lambda: calcGBND(s["gB"], s["twins"], halfwidth=5 * degree, nonneg=True), lambda f: col(np.real(f.eval(vector3d(load("G", 3)))))),
    "schmid": (lambda: sS.SchmidFactor(inv(ori) * xvector), lambda m: np.asarray(m, float).reshape(-1, 24)),
    "birefringence": (lambda: rI.birefringence(vector3d(load("V", 3)))[0], col),
    "cluster": (lambda: calcCluster(orientation(rotation(load("C", 4)), crystalFrame("432")), method="hierarchical", numCluster=4)[1], quat),
    "optimal_s2": (lambda: optimalSample(S2FunHarmonic(np.array(DENSITY, dtype=complex)), 300, bandwidth=16),
                   lambda v: np.asarray(v.data, float).reshape(-1, 3)),
  }
  reps = {"gnd": 1, "gbnd": 1, "cluster": 1, "optimal_s2": 1}
  crosscheck.run(OUT, ALGORITHM, table, ops, reps=reps | {op: 3 for op in COLS if op not in reps})


# the power of every degree 1 to 16 of equally weighted directions: the mean of P_l over all
# pairs, 4 pi / (2l + 1) times the squared norm of the degree's moments, free of any basis
def _moments(v, L=16):
  from scipy.special import eval_legendre
  G = np.clip(v @ v.T, -1, 1)
  return np.array([np.mean(eval_legendre(l, G)) for l in range(1, L + 1)])


def deviation(op, a, b):
  from mtex import angle, crystalFrame, orientation, rotation
  from scipy.spatial import cKDTree
  if op == "cluster":
    cs = crystalFrame("432")
    A, B = orientation(rotation(a), cs), orientation(rotation(b), cs)
    return float(max(min(float(angle(A[i], B[j])) for j in range(B.size)) for i in range(A.size)))
  if op == "optimal_s2":
    ma, mb = _moments(a), _moments(b)
    return float(np.max(np.abs(ma - mb)))
  if op == "gnd":
    d, j = cKDTree(b[:, :2]).query(a[:, :2])
    if len(a) != len(b) or d.max() > 1e-6:
      return float("nan")
    va, vb = a[:, 2], b[j, 2]
    ok = np.isfinite(va) & np.isfinite(vb)
    if (np.isfinite(va) != np.isfinite(vb)).any():
      print(f"  gnd: {int((~np.isfinite(va)).sum())} Python and {int((~np.isfinite(vb)).sum())} MATLAB pixels without a density")
    return float(np.max(np.abs(va[ok] - vb[ok])) / np.max(np.abs(vb[ok])))
  if a.shape != b.shape:
    return float("nan")
  if op == "schmid":
    a, b = np.sort(a, axis=1), np.sort(b, axis=1)
  return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "compare", sys.argv[2:] or list(COLS)
  {"export": export, "run": lambda: run(names),
   "matlab": lambda: [crosscheck.matlab("bench_analysis_ops", repr(OUT), crosscheck.cellOf([op]), "1e7", timeout=290) for op in names],
   "compare": lambda: crosscheck.compare(OUT, ALGORITHM, lambda op: COLS[op], deviation)}[cmd]()
