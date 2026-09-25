# the Dubna pipeline against MATLAB and Julia: pf = mtexdata('dubna'); odf = calcODF(pf);
# ori = odf.discreteSample(1e7); odf2 = calcDensity(ori), each step timed and profiled
#
#   .venv/bin/python bench/dubna_density.py export              the orientations the ODFs are compared at
#   .venv/bin/python bench/dubna_density.py run [step ...]      this port: best of two or three, one profiled run, the values
#   .venv/bin/python bench/dubna_density.py matlab [step ...]   bench/matlab/bench_dubna_density.m through the bridge of ../master
#   .venv/bin/python bench/dubna_density.py julia               bench/dubna_density.jl, times only
#   .venv/bin/python bench/dubna_density.py compare             times, and the ports' ODFs and sample densities
#                                                               against each other
#
# The steps run in order and hand on their result: calcODF, discreteSample, calcDensity; a step not
# named still runs, untimed. Each port writes <port>_<step>.txt (the best time) and
# profile_<port>_<step>.json to bench/out/dubna, and <port>_odf_values.bin, <port>_density_values.bin
# at the orientations of at.bin.
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out", "dubna")
sys.path.insert(0, os.path.join(ROOT, "tools", "graph"))

ALGORITHM = [
  {"name": "pole figure inversion",
   "entry": {"matlab": ["@PoleFigure.calcODF", "@MLSSolver.calcODF"],
             "python": ["mtex.analysis.pf2odf.calcODF", "mtex.analysis.pf2odf.MLSSolver.calcODF"]},
   "cases": {"dubna": "the seven Dubna quartz pole figures, 321, superposed third, default options"},
   "tolerance": 1e-2},
  {"name": "discrete sampling",
   "entry": {"matlab": ["@SO3Fun.discreteSample", "@SO3FunRBF.discreteSample"],
             "python": ["mtex.functions.characteristics.discreteSample"]},
   "cases": {"dubna": "1e7 orientations from the Dubna ODF"},
   "tolerance": 5e-2},
  {"name": "density estimation",
   "entry": {"matlab": ["@rotation.calcDensity"], "python": ["mtex.functions.density.calcDensity"]},
   "cases": {"dubna": "1e7 orientations of the Dubna ODF, 321, default halfwidth"},
   "tolerance": 5e-2},
]
STEPS = {"calcODF": "pole figure inversion", "discreteSample": "discrete sampling", "calcDensity": "density estimation"}
RESULT = {"calcODF": "odf", "discreteSample": "ori", "calcDensity": "odf2"}
REPS = {"calcODF": 3, "discreteSample": 2, "calcDensity": 3}
NAT = 1000


def export():
  os.makedirs(OUT, exist_ok=True)
  q = np.random.default_rng(1).standard_normal((NAT, 4))
  (q / np.linalg.norm(q, axis=1, keepdims=True)).astype("<f8").tofile(os.path.join(OUT, "at.bin"))


def timed(f):
  t = time.perf_counter()
  out = f()
  return out, time.perf_counter() - t


def run(steps):
  import profiling
  from mtex import calcDensity, mtexdata, orientation, rotation
  from mtex.analysis.pf2odf import calcODF
  A = np.fromfile(os.path.join(OUT, "at.bin"), "<f8").reshape(-1, 4)
  pf = mtexdata("dubna")
  rng = np.random.default_rng(0)
  calls = {"calcODF": lambda s: calcODF(pf), "discreteSample": lambda s: s["odf"].discreteSample(10**7, rng=rng),
           "calcDensity": lambda s: calcDensity(s["ori"])}
  state = {}
  for step in STEPS:
    f = lambda: calls[step](state)
    if step in steps:
      if step == "calcODF":
        f()
      out, best = min((timed(f) for _ in range(REPS[step])), key=lambda r: r[1])
      _, wall, recs = profiling.profileCall(f)
      profiling.write(os.path.join(OUT, f"profile_python_{step}.json"), STEPS[step], "python", "dubna", best, recs)
      open(os.path.join(OUT, f"python_{step}.txt"), "w").write(f"{best}\n")
      print(f"{step:16s} {best:6.2f} s, profiled {wall:6.2f} s", flush=True)
    else:
      out = f()
    state[RESULT[step]] = out
  at = orientation(rotation(A), state["odf"].CS, state["odf"].SS)
  for key, name in (("odf", "odf"), ("odf2", "density")):
    np.real(np.asarray(state[key].eval(at))).reshape(-1).astype("<f8").tofile(os.path.join(OUT, f"python_{name}_values.bin"))


# MATLAB through the bridge of ../master, one step per call, the state kept in globals
def matlab(steps):
  bridge = os.path.join(os.path.dirname(ROOT), "master", "docs", "agents", "matlab-bridge")
  for step in STEPS:
    cmd = (f"addpath('{os.path.join(HERE, 'matlab')}'); addpath('{os.path.join(ROOT, 'tools', 'graph')}'); "
           f"bench_dubna_density('{OUT}', '{step}', {int(step in steps)}, {REPS[step]});")
    r = subprocess.run([os.path.join(bridge, ".venv", "bin", "python"), os.path.join(bridge, "matlab_run.py"), cmd],
                       timeout=290)
    if r.returncode:
      sys.exit(f"MATLAB failed on {step}")


# the Julia port prints one line per step after a compiling call
def julia():
  r = subprocess.run(["julia", "--project=" + os.path.join(os.path.dirname(ROOT), "mtexJulia", "dev"), "-t", "10",
                      os.path.join(HERE, "dubna_density.jl")], capture_output=True, text=True, timeout=290)
  print(r.stdout, r.stderr[-2000:])
  names = {"odf = dubna": "calcODF", "discreteSample(1e7)": "discreteSample", "calcDensity(ori)": "calcDensity"}
  for line in r.stdout.splitlines():
    m = re.match(r"(.+?)\s+([\d.]+) s$", line)
    if m and m.group(1).strip() in names:
      open(os.path.join(OUT, f"julia_{names[m.group(1).strip()]}.txt"), "w").write(f"{m.group(2)}\n")


def load(port, name):
  p = os.path.join(OUT, f"{port}_{name}.bin")
  return np.fromfile(p, "<f8") if os.path.exists(p) else None


def compare():
  ports = ("python", "matlab", "julia")
  print(f"{'step':16s}" + "".join(f"{p:>10s}" for p in ports))
  for step in STEPS:
    t = [os.path.join(OUT, f"{p}_{step}.txt") for p in ports]
    print(f"{step:16s}" + "".join(f"{float(open(x).read()):9.2f}s" if os.path.exists(x) else f"{'-':>10s}" for x in t))
  tol = {a["name"]: a["tolerance"] for a in ALGORITHM}
  odf = {p: load(p, "odf_values") for p in ports[:2]}
  den = {p: load(p, "density_values") for p in ports[:2]}
  print(f"\nmaximum deviation relative to the maximum, at {NAT} orientations")
  # the samples differ between the ports, so the density checks sampling and estimation together
  rows = [("calcODF python ~ matlab", odf["python"], odf["matlab"], tol["pole figure inversion"]),
          ("sample density python ~ matlab", den["python"], den["matlab"], tol["density estimation"])]
  for label, a, b, t in rows:
    if a is not None and b is not None:
      d = np.max(np.abs(a - b)) / np.max(np.abs(b))
      print(f"  {label:32s} {d:.2e}  {'ok' if d <= t else 'DIFFERS'} (tolerance {t:g})")


if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "compare", sys.argv[2:] or list(STEPS)
  {"export": export, "run": lambda: run(names), "matlab": lambda: matlab(names), "julia": julia,
   "compare": compare}[cmd]()
