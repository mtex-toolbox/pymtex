# the analysis layer against MATLAB where no other benchmark reaches it: the geometrically
# necessary dislocations, the grain boundary normal distribution, the Schmid factors, the
# birefringence, the components of an ODF, the clustering of points, directions and
# orientations, the optimal sampling of a density and the selection of a kernel halfwidth
#
#   .venv/bin/python bench/analysis_ops.py export              the input
#   .venv/bin/python bench/analysis_ops.py run [op ...]        this port
#   .venv/bin/python bench/analysis_ops.py matlab [op ...]     bench/matlab/bench_analysis_ops.m through the bridge
#   .venv/bin/python bench/analysis_ops.py julia [op ...]      bench/analysis_ops.jl, the Julia port on the same input
#   .venv/bin/python bench/analysis_ops.py compare
#
# The input in bench/out/analysis: 1e5 orientations O and directions V, and four clouds of 500
# orientations each, 3 degree about four centres, in 432. The maps are read by each port itself:
# dc06 for calcGND as the GND page takes it, unsmoothed, and twins with its grains at 5 degree for
# calcGBND as the BoundaryNormalDistribution page. The clustering is compared by the centres, each
# port's nearest to the other's, the optimal sample by its harmonic moments up to the bandwidth,
# since the nodes themselves depend on the start.
#
# The components: 300 random centres R in m-3m with weights Rw, a 10 degree kernel each, matched up
# to symmetry with their weights. The clustering at scale: five Gaussian clouds X of 2e5 points,
# 0.25 wide, for CLASSIX, five clouds D of 2e4 directions, 3 degree wide, and four cubic clouds Q of 5000
# orientations, 2 degree wide, each with an explicit radius, since the ports estimate the default
# from the neighbours differently; a cluster is compared by its mean and its size. The halfwidths:
# 1e4 independent directions HS of model A of bench/halfwidth_s2.py and 1e4 orientations HO of
# model mix of bench/halfwidth_so3.py; the folds and the subsamples are each port's own draws.
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
             "python": ["mtex.analysis.dislocations.calcGND", "mtex.analysis.dislocations.fitDislocationSystems"],
             "julia": ["calcGND", "curvature", "fitDislocationSystems"]},
   "cases": {"gnd": "calcGND(ebsd, dS) of the dc06 map, bcc edge and screw systems, every pixel"},
   # the densities are minima of a linear program, unique in their sum
   "tolerance": 1e-6},
  {"name": "grain boundary normal distribution",
   "entry": {"matlab": ["@grainBoundary.calcGBND"], "python": ["mtex.analysis.gbnd.calcGBND"], "julia": ["calcGBND"]},
   "cases": {"gbnd": "calcGBND(gB, ebsd, 5 degree, 'nonneg') of the twins map at 2000 directions"},
   # the boundaries differ in three segments of 3162 where they meet the border of the map
   "tolerance": 1e-3},
  {"name": "Schmid factor",
   "entry": {"matlab": ["@slipSystem.SchmidFactor"], "python": ["mtex.geometry.slipsystem.slipSystem.SchmidFactor"],
             "julia": ["SchmidFactor"]},
   "cases": {"schmid": "sS.SchmidFactor(inv(ori) * x) of the 24 fcc systems, both slip directions, 1e5 orientations"},
   # the ports list the systems in their own order: every row compared sorted
   "tolerance": 1e-12},
  {"name": "birefringence",
   "entry": {"matlab": ["@refractiveIndexTensor.birefringence"], "python": ["mtex.analysis.optics.birefringence"], "julia": ["birefringence"]},
   "cases": {"birefringence": "the birefringence of the forsterite-fayalite mixture of the BirefringenceDemo page, 1e5 directions"},
   # MATLAB's eig2 is 1.3e-11 off the extended precision difference of the two indices, this port 1e-15
   "tolerance": 1e-9},
  {"name": "ODF components",
   "entry": {"matlab": ["@SO3Fun.calcComponents", "@SO3Fun.max"],
             "python": ["mtex.functions.characteristics.calcComponents", "mtex.functions.characteristics.max_"],
             "julia": ["calcComponents", "findmax"]},
   "cases": {"components": "calcComponents(odf) of 300 random centres in m-3m, a 10 degree kernel each: the modes and their weights",
             "odf_max": "[v, ori] = max(odf) of the same ODF"},
   # a mode is where the ascent stops, its finest step 0.05 degree, the maximum where the grid refinement
   # stops, at 0.25 degree; a weight is the sum of the seeds that climb to a mode, exact where the basins
   # agree. MATLAB's ascent ends at the first iteration in which no mode gains at its current step, 3.3
   # degree here, before any step is halved to the resolution: its 22 modes are up to 1.8 degree short
   # of the peaks and some not merged, against 16 of this port and the Julia port
   "tolerance": {"components": 2e-3, "odf_max": 5e-3}},
  {"name": "clustering",
   "entry": {"matlab": ["@orientation.calcCluster", "@vector3d.calcCluster"],
             "python": ["mtex.analysis.cluster.calcCluster", "mtex.analysis.cluster.classix"],
             "julia": ["calcCluster", "classix"]},
   "cases": {"cluster": "calcCluster(ori, 'numCluster', 4, 'method', 'hierarchical'), 2000 orientations in 432",
             "classix": "classix(X, 0.1, 2000) of five Gaussian clouds, 1e6 points in 3D",
             "cluster_directions": "calcCluster(v, 'radius', 0.1, 'minPoints', 1000) of five clouds of directions, 1e5",
             "cluster_orientations": "calcCluster(ori, 'method', 'classix', 'radius', 0.1, 'minPoints', 200) of four cubic clouds, 2e4"},
   # the centres are means of clusters that may differ by a few border points, the sizes by as many
   "tolerance": 1e-2},
  {"name": "discrete sampling",
   "entry": {"matlab": ["@S2Fun.optimalSample", "@SO3Fun.optimalSample"], "python": ["mtex.analysis.optimalsample.optimalSample"],
             "julia": ["optimalSample"]},
   "cases": {"optimal_s2": "optimalSample(1 + 0.6 z, 300, 'bandwidth', 16): the discrepancy J of the nodes",
             "optimal_so3": "optimalSample(odf, 200, 'bandwidth', 16) of a 20 degree unimodal ODF in 1: the discrepancy J of the nodes"},
   # the ports' L-BFGS reach equally good minima by different paths, so the nodes agree only as a
   # set; J of each port's nodes, by this port's objective, against MATLAB's relative to J of the
   # target alone
   "tolerance": 2e-2},
  # the Julia port selects no halfwidth
  {"name": "halfwidth selection",
   "entry": {"matlab": ["selectHalfwidth", "@orientation.calcKernel", "@vector3d.calcKernel"],
             "python": ["mtex.functions.halfwidth.selectHalfwidth", "mtex.functions.characteristics.calcKernel"]},
   "cases": {"halfwidth_s2_ucv": "selectHalfwidth(v, 'UCV') of 1e4 directions of model A",
             "halfwidth_s2_cons": "selectHalfwidth(v, 'conservative') of the same directions",
             "halfwidth_so3_ucv": "selectHalfwidth(ori, 'UCV') of 1e4 orientations of model mix in m-3m",
             "halfwidth_so3_cons": "selectHalfwidth(ori, 'conservative') of the same orientations"},
   # one candidate apart, 5 percent on SO(3) and 7 percent on S2, and two for the conservative choice:
   # its energies are lowered by their jackknife error over folds each port draws itself, while UCV's
   # traces and the noise floor's pilot are averages over subsamples each port draws itself
   "tolerance": {"halfwidth_s2_ucv": 0.08, "halfwidth_s2_cons": 0.16, "halfwidth_so3_ucv": 0.06, "halfwidth_so3_cons": 0.11}},
]
N, NG = 10**5, 2000
COLS = {"gnd": 3, "gbnd": 1, "schmid": 24, "birefringence": 1, "components": 5, "odf_max": 5, "cluster": 4,
        "classix": 4, "cluster_directions": 4, "cluster_orientations": 5, "optimal_s2": 3, "optimal_so3": 4,
        "halfwidth_s2_ucv": 1, "halfwidth_s2_cons": 1, "halfwidth_so3_ucv": 1, "halfwidth_so3_cons": 1}
# 1 + 0.6 z in the coefficients of degree 0 and 1
DENSITY = [np.sqrt(4 * np.pi), 0, 0.6 * np.sqrt(4 * np.pi / 3), 0]
# the centres of the Gaussian clouds of CLASSIX and the directions of the clouds of directions
POINTS = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3], [2, 2, 2]], dtype=float)
AXES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, -1, 0]], dtype=float)


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
  exportScale(np.random.default_rng(1))


# the inputs of the components, the clustering at scale and the halfwidth selection
def exportScale(rng):
  from mtex import crystalFrame, degree, discreteSample, orientation, rotation, uniformODF, unimodalODF, vector3d
  import halfwidth_s2
  write = lambda name, x: np.asarray(x, "<f8").tofile(os.path.join(OUT, f"{name}.bin"))
  unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
  write("R", unit(rng.standard_normal((300, 4))))
  write("Rw", rng.uniform(0.5, 1.5, 300))
  write("X", np.concatenate([c + 0.25 * rng.standard_normal((200_000, 3)) for c in POINTS]))
  write("D", np.concatenate([unit(unit(a[None]) + 0.05 * rng.standard_normal((20_000, 3))) for a in AXES]))
  cs = crystalFrame("432")
  oc = rotation.byEuler(np.array([0, 40, 80, 20]) * degree, np.array([0, 30, 60, 80]) * degree, np.array([0, 10, 50, 70]) * degree)
  write("Q", np.concatenate([(rotation(oc.data[k]) * rotation.byAxisAngle(vector3d.rand(5000, rng=rng),
                                                                         2 * degree * np.abs(rng.standard_normal(5000)))).data.reshape(-1, 4)
                             for k in range(4)]))
  write("HS", halfwidth_s2.sample(halfwidth_s2.model("A"), 10_000, rng))
  cub = crystalFrame("m-3m")
  ori = lambda: orientation(rotation.rand(rng=rng), cub)
  mix = 0.5 * uniformODF(cub) + 0.3 * unimodalODF(ori(), halfwidth=7 * degree) + 0.2 * unimodalODF(ori(), halfwidth=15 * degree)
  write("HO", discreteSample(mix, 10_000, rng=rng).data.reshape(-1, 4))


def load(name, k):
  return np.fromfile(os.path.join(OUT, f"{name}.bin"), "<f8").reshape(-1, k)


def run(ops):
  from mtex import (S2FunHarmonic, SO3DeLaValleePoussinKernel, SO3FunRBF, calcCluster, calcComponents, calcGBND, calcGND,
                    calcGrains, calcKernel, classix, crystalFrame, degree, dislocationSystem, inv, mtexdata, optimalSample,
                    orientation, refractiveIndexTensor, rotation, slipSystem, vector3d, xvector)
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
  odf = SO3FunRBF(orientation(rotation(load("R", 4)), cub), SO3DeLaValleePoussinKernel(halfwidth=10 * degree), load("Rw", 1).ravel())
  unimodal = SO3FunRBF(orientation(rotation.byEuler(10 * degree, 20 * degree, 30 * degree), crystalFrame("1")),
                       SO3DeLaValleePoussinKernel(halfwidth=20 * degree), np.ones(1))
  hs, ho = vector3d(load("HS", 3)), orientation(rotation(load("HO", 4)), cub)
  q = load("Q", 4)
  # the clusters as rows of their centre and their size
  sized = lambda centre, cId: np.column_stack([centre, np.bincount(np.asarray(cId).ravel(), minlength=len(centre))])
  table = {
    "gnd": (gnd, lambda r: r),
    "gbnd": (lambda: calcGBND(s["gB"], s["twins"], halfwidth=5 * degree, nonneg=True), lambda f: col(np.real(f.eval(vector3d(load("G", 3)))))),
    "schmid": (lambda: sS.SchmidFactor(inv(ori) * xvector), lambda m: np.asarray(m, float).reshape(-1, 24)),
    "birefringence": (lambda: rI.birefringence(vector3d(load("V", 3)))[0], col),
    "components": (lambda: calcComponents(odf, verbose=False), lambda r: np.column_stack([quat(r[0]), r[1]])),
    "odf_max": (lambda: odf.max(), lambda r: np.column_stack([np.atleast_1d(r[0]), quat(r[1])])),
    "cluster": (lambda: calcCluster(orientation(rotation(load("C", 4)), crystalFrame("432")), method="hierarchical", numCluster=4)[1], quat),
    "classix": (lambda: classix(load("X", 3), 0.1, minPts=2000),
                lambda c: sized(np.stack([load("X", 3)[c == k].mean(axis=0) for k in range(c.max() + 1)]), c)),
    "cluster_directions": (lambda: calcCluster(vector3d(load("D", 3)), radius=0.1, minPoints=1000),
                           lambda r: sized(np.asarray(r[1].data, float).reshape(-1, 3), r[0])),
    "cluster_orientations": (lambda: calcCluster(orientation(rotation(q), crystalFrame("432")), method="classix", radius=0.1, minPoints=200),
                             lambda r: sized(quat(r[1]), r[0])),
    "optimal_s2": (lambda: optimalSample(S2FunHarmonic(np.array(DENSITY, dtype=complex)), 300, bandwidth=16),
                   lambda v: np.asarray(v.data, float).reshape(-1, 3)),
    "optimal_so3": (lambda: optimalSample(unimodal, 200, bandwidth=16), quat),
    "halfwidth_s2_ucv": (lambda: calcKernel(hs, "UCV").halfwidth(), col),
    "halfwidth_s2_cons": (lambda: calcKernel(hs, "conservative").halfwidth(), col),
    "halfwidth_so3_ucv": (lambda: calcKernel(ho, "UCV").halfwidth(), col),
    "halfwidth_so3_cons": (lambda: calcKernel(ho, "conservative").halfwidth(), col),
  }
  reps = {op: 1 for op in COLS if op not in ("schmid", "birefringence")}
  crosscheck.run(OUT, ALGORITHM, table, ops, reps=reps | {op: 3 for op in COLS if op not in reps})


# the discrepancy J of equally weighted nodes against the target, by this port's objective, so that
# every port's nodes are measured alike, and the J of no nodes at all, the size of the target
def _discrepancy(op, x):
  from mtex import S2FunHarmonic, SO3DeLaValleePoussinKernel, SO3FunRBF, crystalFrame, degree, orientation, rotation
  from mtex.analysis.optimalsample import _Problem
  if op == "optimal_s2":
    f = S2FunHarmonic(np.array(DENSITY, dtype=complex))
  else:
    f = SO3FunRBF(orientation(rotation.byEuler(10 * degree, 20 * degree, 30 * degree), crystalFrame("1")),
                  SO3DeLaValleePoussinKernel(halfwidth=20 * degree), np.ones(1))
  p = _Problem(f, x, 16)
  return p.residual(x, np.full(len(x), 1 / len(x)))[0], p.objective(p.target._fhat)


# the angles between two lists of orientations of one symmetry, every pair
def _angles(a, b, cs):
  from mtex import crystalFrame, orientation, rotation
  frame = crystalFrame(cs)
  A, B = orientation(rotation(a), frame), orientation(rotation(b), frame)
  return np.array([[float(A[i].angle(B[j])) for j in range(B.size)] for i in range(A.size)])


# two sets matched each to the other's nearest: the farthest match, and the largest difference of
# the values in the columns from `col` on, relative to `scale`, between nearest partners
def _matched(D, a, b, col=None, scale=1.0, name=""):
  if a.shape != b.shape:
    print(f"  {name}: {len(a)} against MATLAB's {len(b)}")
  d = max(D.min(axis=1).max(), D.min(axis=0).max())
  if col is None:
    return float(d)
  j, i = D.argmin(axis=1), D.argmin(axis=0)
  dv = max(np.abs(a[:, col:] - b[j, col:]).max(), np.abs(a[i, col:] - b[:, col:]).max())
  return float(max(d, dv / scale))


def deviation(op, a, b):
  from scipy.spatial import cKDTree
  if op == "cluster":
    return float(_angles(a, b, "432").min(axis=1).max())
  if op == "components":
    return _matched(_angles(a[:, :4], b[:, :4], "m-3m"), a, b, 4, name=op)
  if op == "odf_max":
    return float(max(abs(a[0, 0] - b[0, 0]) / abs(b[0, 0]), _angles(a[:, 1:], b[:, 1:], "m-3m")[0, 0]))
  if op == "classix":
    return _matched(np.linalg.norm(a[:, None, :3] - b[None, :, :3], axis=-1), a, b, 3, b[:, 3].sum(), op)
  if op == "cluster_directions":
    u = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
    return _matched(np.arccos(np.clip(u(a[:, :3]) @ u(b[:, :3]).T, -1, 1)), a, b, 3, b[:, 3].sum(), op)
  if op == "cluster_orientations":
    return _matched(_angles(a[:, :4], b[:, :4], "432"), a, b, 4, b[:, 4].sum(), op)
  if op in ("optimal_s2", "optimal_so3"):
    (ja, j0), (jb, _) = _discrepancy(op, a), _discrepancy(op, b)
    print(f"  {op}: J {ja:.3e} against MATLAB's {jb:.3e}, of the target's {j0:.3e}")
    return float(abs(ja - jb) / j0)
  if op.startswith("halfwidth"):
    return float(abs(a[0, 0] - b[0, 0]) / b[0, 0])
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
   "julia": lambda: crosscheck.julia("analysis_ops.jl", ALGORITHM, names),
   "matlab": lambda: [crosscheck.matlab("bench_analysis_ops", repr(OUT), crosscheck.cellOf([op]), "1e7", timeout=290) for op in names],
   "compare": lambda: crosscheck.compare(OUT, ALGORITHM, lambda op: COLS[op], deviation)}[cmd]()
