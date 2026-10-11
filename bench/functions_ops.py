# the functions layer against MATLAB on one input: the harmonic transforms on SO(3) and the
# sphere, the radial basis functions and their harmonic form, the least squares fits
#
#   .venv/bin/python bench/functions_ops.py export              the input
#   .venv/bin/python bench/functions_ops.py run [op ...]        this port
#   .venv/bin/python bench/functions_ops.py matlab [op ...]     bench/matlab/bench_functions_ops.m through the bridge
#   .venv/bin/python bench/functions_ops.py julia [op ...]      bench/functions_ops.jl, the Julia port on the same input
#   .venv/bin/python bench/functions_ops.py compare
#
# The input in bench/out/functions: 1000 centres C (m-3m) with weights w of a de la Vallee Poussin
# ODF of halfwidth 10 degree, 1e5 orientations O and directions V to evaluate at and fit to, and
# the smooth values a^2 - b c on O and exp(z) + x y on V. rbf_direct evaluates the ODF at the first
# 1e4 of O, MATLAB and Julia by the kernel sum, this port by its harmonic form, since the sum has
# more terms than HARMONIC_EVAL; the slow variant so3_eval is checked against in every port.
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "graph"))
import crosscheck  # noqa: E402

OUT = os.path.join(HERE, "out", "functions")
ALGORITHM = [
  {"name": "harmonic transforms",
   "entry": {"matlab": ["@SO3FunHarmonic.eval", "@SO3FunHarmonic.quadrature", "@S2FunHarmonic.eval", "@S2FunHarmonic.quadrature"],
             "python": ["mtex.functions.so3fun.SO3FunHarmonic.eval", "mtex.functions.so3fun.SO3FunHarmonic.quadrature",
                        "mtex.functions.s2fun.S2FunHarmonic.eval", "mtex.functions.s2fun.S2FunHarmonic.quadrature"],
             "julia": ["SO3FunHarmonic", "S2FunHarmonic"]},
   "cases": {"so3_eval": "SO3FunHarmonic(odf) at 1e5 orientations, L = 64, m-3m",
             "so3_quadrature": "SO3FunHarmonic.quadrature, 1e5 nodes, L = 32, triclinic",
             "s2_quadrature": "S2FunHarmonic.quadrature, 1e5 nodes, L = 128",
             "s2_eval": "the S2 quadrature result at 1e5 directions"},
   "tolerance": 1e-6},
  {"name": "radial basis functions",
   "entry": {"matlab": ["@SO3FunRBF.eval", "@SO3FunRBF.calcFourier"],
             "python": ["mtex.functions.rbf.SO3FunRBF.eval", "mtex.functions.rbf.SO3FunRBF._harmonic"],
             "julia": ["SO3FunRBF", "SO3FunHarmonic"]},
   "cases": {"rbf_direct": "odf(ori), the ODF of 1000 centres at 1e4 orientations: the kernel sum in MATLAB and Julia, "
                           "in this port its harmonic form, cut at bandwidth 25, since the sum has more terms than HARMONIC_EVAL",
             "rbf_harmonic": "SO3FunHarmonic(odf), bandwidth 64"},
   # MATLAB and Julia sum the kernels exactly and agree to 3e-6; this port evaluates the harmonic
   # form there (decided 2026-10-09), which the cut at the kernel's accuracy bandwidth leaves about
   # 5e-4 of the maximum from the sum; the 3e-13 of the slow variant's check here compares the
   # harmonic form with itself
   "tolerance": {"rbf_direct": 1e-3, "rbf_harmonic": 1e-6}},
  {"name": "least squares approximation",
   "entry": {"matlab": ["@SO3FunHarmonic.interpolate", "@S2FunHarmonic.interpolate"],
             "python": ["mtex.functions.density.interpolate"],
             "julia": ["interpolate"]},
   "cases": {"so3_interpolate": "SO3FunHarmonic.interpolate, 2e4 nodes, L = 16, triclinic, at 1e4 orientations",
             "s2_interpolate": "S2FunHarmonic.interpolate, 2e4 nodes, L = 48, at 1e4 directions"},
   # both ports stop LSQR at a relative residual of 1e-3 on SO(3) and 1e-6 on the sphere
   "tolerance": {"so3_interpolate": 5e-2, "s2_interpolate": 1e-5}},
]
K = 10**5
COLS = {"so3_quadrature": 2, "s2_quadrature": 2, "rbf_harmonic": 2}


def export():
  os.makedirs(OUT, exist_ok=True)
  rng = np.random.default_rng(0)
  unit = lambda n, k: (lambda x: x / np.linalg.norm(x, axis=1, keepdims=True))(rng.standard_normal((n, k)))
  C, O, V = unit(1000, 4), unit(10**5, 4), unit(10**5, 3)
  a, b, c, _ = O.T
  x, y, z = V.T
  for name, arr in (("C", C), ("w", rng.random(1000)), ("O", O), ("V", V),
                    ("fO", a * a - b * c), ("fV", np.exp(z) + x * y)):
    np.asarray(arr, dtype="<f8").tofile(os.path.join(OUT, f"{name}.bin"))


def load(name, k=1):
  return np.fromfile(os.path.join(OUT, f"{name}.bin"), "<f8").reshape(-1, k) if k > 1 else np.fromfile(os.path.join(OUT, f"{name}.bin"), "<f8")


# complex coefficients as rows (real, imaginary)
def coefficients(F):
  f = np.asarray(F.fhat).reshape(-1)
  return np.column_stack([f.real, f.imag])


def run(ops):
  import mtex
  from mtex import (S2FunHarmonic, SO3DeLaValleePoussinKernel, SO3FunHarmonic, SO3FunRBF, crystalFrame, degree,
                    orientation, rotation, vector3d)
  cs, tri = crystalFrame("m-3m"), crystalFrame("1")
  O, V = rotation(load("O", 4)), vector3d(load("V", 3))
  odf = SO3FunRBF(orientation(rotation(load("C", 4)), cs), SO3DeLaValleePoussinKernel(halfwidth=10 * degree), load("w"))
  ori, oriT = orientation(O, cs), orientation(O, tri)
  # the ports weigh unweighted nodes differently, MATLAB by 1, Python by the volume over N
  wO, wV = np.full(len(O), 1 / len(O)), np.full(len(V), 4 * np.pi / len(V))
  s = {}

  def keep(key, f):
    s[key] = f()
    return s[key]

  vals = lambda F, x: np.real(np.asarray(F.eval(x))).reshape(-1, 1)
  table = {
    "rbf_harmonic": (lambda: keep("F", lambda: SO3FunHarmonic(odf, bandwidth=64)), coefficients),
    "rbf_direct": (lambda: odf.eval(ori[:10**4]), lambda v: np.real(v).reshape(-1, 1)),
    "so3_eval": (lambda: s["F"].eval(ori), lambda v: np.real(v).reshape(-1, 1)),
    "so3_quadrature": (lambda: SO3FunHarmonic.quadrature(oriT, load("fO"), bandwidth=32, weights=wO), coefficients),
    "s2_quadrature": (lambda: keep("G", lambda: S2FunHarmonic.quadrature(V, load("fV"), bandwidth=128, weights=wV)),
                      coefficients),
    "s2_eval": (lambda: s["G"].eval(V), lambda v: np.real(v).reshape(-1, 1)),
    "so3_interpolate": (lambda: SO3FunHarmonic.interpolate(oriT[:20000], load("fO")[:20000], bandwidth=16),
                        lambda F: vals(F, oriT[:10**4])),
    "s2_interpolate": (lambda: S2FunHarmonic.interpolate(V[:20000], load("fV")[:20000], bandwidth=48),
                       lambda F: vals(F, V[:10**4])),
  }
  crosscheck.run(OUT, ALGORITHM, table, ops, reps=3, always=("rbf_harmonic", "s2_quadrature"))
  # the harmonic form against the exact kernel sum of the radial basis functions
  exact = np.real(odf.eval(ori[:10**4])).ravel()
  d = np.max(np.abs(np.real(s["F"].eval(ori[:10**4])).ravel() - exact)) / np.max(exact)
  print(f"harmonic form against the exact kernel sum, relative to its maximum: {d:.1e}")


def deviation(op, a, b):
  if a.shape != b.shape:
    return float("nan")
  return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "compare", sys.argv[2:] or [c for a in ALGORITHM for c in a["cases"]]
  {"export": export, "run": lambda: run(names),
   "matlab": lambda: crosscheck.matlab("bench_functions_ops", repr(OUT), crosscheck.cellOf(names), str(K)),
   "julia": lambda: crosscheck.julia("functions_ops.jl", ALGORITHM, sys.argv[2:]),
   "compare": lambda: crosscheck.compare(OUT, ALGORITHM, lambda op: COLS.get(op, 1), deviation)}[cmd]()
