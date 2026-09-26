# the tensors layer against MATLAB on one input: the rotation of a stiffness tensor, its texture
# averages over orientations and an ODF, the elastic wave velocities and the Taylor model
#
#   .venv/bin/python bench/tensors_ops.py export              the input
#   .venv/bin/python bench/tensors_ops.py run [op ...]        this port
#   .venv/bin/python bench/tensors_ops.py matlab [op ...]     bench/matlab/bench_tensors_ops.m through the bridge
#   .venv/bin/python bench/tensors_ops.py compare
#
# The input in bench/out/tensors: 1e5 orientations O and directions V. The stiffness is the olivine
# of Abramson (1997) in mmm with its density, the ODF unimodal of halfwidth 15 degree about the Euler
# angles (10, 20, 30) degree, the Taylor model fcc {111}<110> in m-3m under the plane strain
# diag(1, -1/2, -1/2). A tensor is written as its 81 coordinates in MATLAB's column-major order.
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "graph"))
import crosscheck  # noqa: E402

OUT = os.path.join(HERE, "out", "tensors")
ALGORITHM = [
  {"name": "tensor rotation",
   "entry": {"matlab": ["@tensor.rotate", "@tensor.inv"],
             "python": ["mtex.tensors.tensor.tensor.rotate", "mtex.tensors.tensor.tensor.inv"]},
   "cases": {"rotate": "ori * C of the olivine stiffness, 1e5 orientations",
             "inv": "inv of the 1e5 rotated stiffness tensors"},
   "tolerance": 1e-12},
  {"name": "tensor average",
   "entry": {"matlab": ["@orientation.calcTensor", "@SO3Fun.calcTensor"],
             "python": ["mtex.analysis.texture.calcTensor"]},
   "cases": {"average_ori": "calcTensor(ori, C, 'Hill'), 1e5 orientations",
             "average_odf": "calcTensor(odf, C), the Voigt average over a unimodal ODF"},
   # MATLAB's ODF average is 2.3e-8 off the quadrature of the ODF's own values, this port's 1e-9
   "tolerance": {"average_ori": 1e-10, "average_odf": 5e-8}},
  {"name": "elastic wave velocities",
   "entry": {"matlab": ["@stiffnessTensor.velocity", "@stiffnessTensor.YoungsModulus"],
             "python": ["mtex.analysis.elasticity.velocity", "mtex.analysis.elasticity.YoungsModulus"]},
   "cases": {"velocity": "vp, vs1, vs2 of the olivine stiffness in 1e5 directions",
             "youngs": "YoungsModulus in 1e5 directions"},
   "tolerance": 1e-10},
  {"name": "Taylor model",
   "entry": {"matlab": ["@strainTensor.calcTaylor"], "python": ["mtex.analysis.taylor.calcTaylor"]},
   "cases": {"taylor": "calcTaylor(inv(ori) * eps, sS), the factor of 1e4 orientations",
             "taylor_fun": "calcTaylor(eps, sS), the factor as an SO3Fun, at 1e4 orientations"},
   # MATLAB's linprog stops at its optimality tolerance; the harmonic factor is a fit to a kink
   # function, the two ports on different quadrature grids
   "tolerance": {"taylor": 1e-8, "taylor_fun": 1e-2}},
]
N, NT = 10**5, 10**4
COLS = {"rotate": 81, "inv": 81, "average_ori": 81, "average_odf": 81, "velocity": 3}
OLIVINE = [[320.5, 68.15, 71.6, 0, 0, 0], [68.15, 196.5, 76.8, 0, 0, 0], [71.6, 76.8, 233.5, 0, 0, 0],
           [0, 0, 0, 64, 0, 0], [0, 0, 0, 0, 77, 0], [0, 0, 0, 0, 0, 78.7]]


def export():
  os.makedirs(OUT, exist_ok=True)
  rng = np.random.default_rng(0)
  for name, k in (("O", 4), ("V", 3)):
    x = rng.standard_normal((N, k))
    (x / np.linalg.norm(x, axis=1, keepdims=True)).astype("<f8").tofile(os.path.join(OUT, f"{name}.bin"))


def load(name, k):
  return np.fromfile(os.path.join(OUT, f"{name}.bin"), "<f8").reshape(-1, k)


# the coordinates of every tensor in MATLAB's column-major order
def coordinates(T):
  M = np.asarray(T.M, float).reshape(-1, 3, 3, 3, 3)
  return M.transpose(0, 4, 3, 2, 1).reshape(len(M), 81)


def run(ops):
  from mtex import (calcTaylor, calcTensor, crystalFrame, degree, inv, orientation, rotation, slipSystem,
                    stiffnessTensor, strainTensor, unimodalODF, vector3d)
  ol = crystalFrame("mmm", [4.7646, 10.2296, 5.9942], mineral="Olivine")
  C = stiffnessTensor(np.array(OLIVINE), ol, density=3.355)
  O, v = rotation(load("O", 4)), vector3d(load("V", 3))
  ori = orientation(O, ol)
  odf = unimodalODF(orientation.byEuler(10 * degree, 20 * degree, 30 * degree, ol), halfwidth=15 * degree)
  cub = crystalFrame("m-3m")
  sS = slipSystem.fcc(cub).symmetrise()
  eps = strainTensor(np.diag([1, -0.5, -0.5]))
  oriT = orientation(O[:NT], cub)
  s = {}

  def keep(key, f):
    s[key] = f()
    return s[key]

  col = lambda x: np.asarray(x, float).reshape(-1, 1)
  table = {
    "rotate": (lambda: keep("R", lambda: ori * C), coordinates),
    "inv": (lambda: s["R"].inv(), coordinates),
    "average_ori": (lambda: calcTensor(ori, C, "Hill"), coordinates),
    "average_odf": (lambda: calcTensor(odf, C), coordinates),
    "velocity": (lambda: C.velocity(v), lambda r: np.column_stack([np.asarray(x, float).reshape(-1) for x in r[:3]])),
    "youngs": (lambda: C.YoungsModulus(v), col),
    "taylor": (lambda: calcTaylor(inv(oriT) * eps, sS)[0], col),
    "taylor_fun": (lambda: calcTaylor(eps, sS)[0], lambda F: col(np.real(F.eval(oriT)))),
  }
  crosscheck.run(OUT, ALGORITHM, table, ops, reps=3, always=("rotate",))


def deviation(op, a, b):
  if a.shape != b.shape:
    return float("nan")
  return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "compare", sys.argv[2:] or [c for a in ALGORITHM for c in a["cases"]]
  {"export": export, "run": lambda: run(names),
   "matlab": lambda: crosscheck.matlab("bench_tensors_ops", repr(OUT), crosscheck.cellOf(names), str(N)),
   "compare": lambda: crosscheck.compare(OUT, ALGORITHM, lambda op: COLS.get(op, 1), deviation)}[cmd]()
