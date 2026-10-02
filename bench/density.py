# kernel density estimation, calcDensity, against MATLAB: one input per case, the densities
# compared at fixed points, the time of each port broken down by its profiler
#
#   .venv/bin/python bench/density.py export              the samples and evaluation points of every case
#   .venv/bin/python bench/density.py run [case ...]      this port: best of three, one profiled run, the values
#   .venv/bin/python bench/density.py matlab [case ...]   bench/matlab/bench_density.m through the bridge of ../master
#   .venv/bin/python bench/density.py compare             times and the relative deviation of the densities
#
# Each case writes <out>/<case>.bin, the samples as float64 rows (quaternion a b c d or vector
# x y z) and <out>/<case>_at.bin, the evaluation points; each port writes <port>_<case>.txt
# (the best time), <port>_<case>_values.bin and profile_<port>_<case>.json for tools/graph.
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out", "density")
sys.path.insert(0, os.path.join(ROOT, "tools", "graph"))

ALGORITHM = {
  "name": "density estimation",
  "entry": {"matlab": ["@rotation.calcDensity", "@vector3d.calcDensity"],
            "python": ["mtex.functions.density.calcDensity"]},
  "cases": {
    "so3_1e6": "1e6 orientations, m-3m, three peaks and a uniform part, halfwidth 10 degrees",
    "so3_1e4": "1e4 orientations, m-3m, halfwidth 5 degrees",
    "s2_1e6": "1e6 axes, three peaks, halfwidth 5 degrees",
  },
  "tolerance": 1e-6,
  "stages": 12,
}
CASES = {"so3_1e6": ("SO3", 10**6, 10.0), "so3_1e4": ("SO3", 10**4, 5.0), "s2_1e6": ("S2", 10**6, 5.0)}
NAT = 1000


# orientations around three centres with a uniform quarter, as unit quaternions
def sampleSO3(n, rng):
  axis = rng.standard_normal((n, 3))
  axis /= np.linalg.norm(axis, axis=1, keepdims=True)
  ang = np.abs(rng.normal(0, np.radians(12), n))
  local = np.column_stack([np.cos(ang / 2), np.sin(ang / 2)[:, None] * axis])
  centres = rng.standard_normal((3, 4))
  centres /= np.linalg.norm(centres, axis=1, keepdims=True)
  c = centres[rng.integers(0, 3, n)]
  a1, b1, c1, d1 = c.T
  a2, b2, c2, d2 = local.T
  q = np.column_stack([a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2, a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                       a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2, a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2])
  u = rng.random(n) < 0.25
  q[u] = rng.standard_normal((u.sum(), 4))
  return q / np.linalg.norm(q, axis=1, keepdims=True)


def sampleS2(n, rng):
  centres = rng.standard_normal((3, 3))
  centres /= np.linalg.norm(centres, axis=1, keepdims=True)
  v = centres[rng.integers(0, 3, n)] + rng.normal(0, 0.15, (n, 3))
  u = rng.random(n) < 0.25
  v[u] = rng.standard_normal((u.sum(), 3))
  return v / np.linalg.norm(v, axis=1, keepdims=True)


def export():
  os.makedirs(OUT, exist_ok=True)
  rng = np.random.default_rng(0)
  for case, (dom, n, _) in CASES.items():
    f = sampleSO3 if dom == "SO3" else sampleS2
    f(n, rng).astype("<f8").tofile(os.path.join(OUT, f"{case}.bin"))
    f(NAT, rng).astype("<f8").tofile(os.path.join(OUT, f"{case}_at.bin"))
    print(f"{case}: {n} samples, {NAT} evaluation points")


def load(case):
  dom, n, hw = CASES[case]
  k = 4 if dom == "SO3" else 3
  X = np.fromfile(os.path.join(OUT, f"{case}.bin"), "<f8").reshape(-1, k)
  A = np.fromfile(os.path.join(OUT, f"{case}_at.bin"), "<f8").reshape(-1, k)
  return dom, X, A, hw


def run(cases):
  import profiling
  from mtex import calcDensity, crystalFrame, degree, orientation, rotation, specimenFrame, vector3d
  cs = crystalFrame("m-3m")
  for case in cases:
    dom, X, A, hw = load(case)
    if dom == "SO3":
      x = orientation(rotation(X), cs, specimenFrame.default)
      at = orientation(rotation(A), cs, specimenFrame.default)
      f = lambda: calcDensity(x, halfwidth=hw * degree)
    else:
      x = vector3d(X, antipodal=True)
      at = vector3d(A, antipodal=True)
      f = lambda: calcDensity(x, halfwidth=hw * degree)
    f()
    (_, best, _), cores = profiling.best(profiling.timed(f) for _ in range(3))
    F, wall, recs = profiling.profileCall(f)
    profiling.write(os.path.join(OUT, f"profile_python_{case}.json"), ALGORITHM["name"], "python", case, best, recs, cores,
                    profiling.threads())
    np.asarray(F.eval(at), dtype=float).reshape(-1).astype("<f8").tofile(os.path.join(OUT, f"python_{case}_values.bin"))
    open(os.path.join(OUT, f"python_{case}.txt"), "w").write(f"{best}\n")
    print(f"{case}: {1e3 * best:.1f} ms, profiled {1e3 * wall:.1f} ms")


# MATLAB through the bridge of ../master, one case per call
def matlab(cases):
  bridge = os.path.join(os.path.dirname(ROOT), "master", "docs", "agents", "matlab-bridge")
  for case in cases:
    dom, n, hw = CASES[case]
    cmd = (f"addpath('{os.path.join(HERE, 'matlab')}'); addpath('{os.path.join(ROOT, 'tools', 'graph')}'); "
           f"bench_density('{OUT}', '{case}', '{dom}', {hw});")
    r = subprocess.run([os.path.join(bridge, ".venv", "bin", "python"), os.path.join(bridge, "matlab_run.py"), cmd],
                       timeout=290)
    if r.returncode:
      sys.exit(f"MATLAB failed on {case}")


def compare():
  print(f"{'case':10s} {'python':>10s} {'matlab':>10s}  deviation of the densities at {NAT} points")
  for case in CASES:
    t, v = {}, {}
    for port in ("python", "matlab"):
      p = os.path.join(OUT, f"{port}_{case}.txt")
      if os.path.exists(p):
        t[port] = float(open(p).read())
        v[port] = np.fromfile(os.path.join(OUT, f"{port}_{case}_values.bin"), "<f8")
    dev = (np.max(np.abs(v["python"] - v["matlab"])) / np.max(np.abs(v["matlab"]))) if len(v) == 2 else np.nan
    ok = "ok" if dev <= ALGORITHM["tolerance"] else "DIFFERS" if dev == dev else ""
    print(f"{case:10s} " + " ".join(f"{1e3 * t[p]:8.1f}ms" if p in t else f"{'-':>10s}" for p in ("python", "matlab"))
          + f"  {dev:.2e} {ok}")


if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "compare", sys.argv[2:] or list(CASES)
  {"export": lambda: export(), "run": lambda: run(names), "matlab": lambda: matlab(names),
   "compare": lambda: compare()}[cmd]()
