# the frame of a cross-port benchmark: each operation timed, profiled and its results written;
# MATLAB through the bridge; the table of times and deviations. crosscheckRun.m is the MATLAB half.
#
# An operation writes to <out>: <port>_<op>.txt (the best time), <port>_<op>.bin (its results as
# float64 rows, the first K) and profile_<port>_<op>.json, which tools/graph/build.py reads.
# crosscheck.jl is the Julia half, run by julia() on bench/<name>_ops.jl.
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

import profiling

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
BRIDGE = os.path.join(os.path.dirname(ROOT), "master", "docs", "agents", "matlab-bridge")
JULIA = [os.path.expanduser("~/.juliaup/bin/julia"), "--project=" + os.path.join(os.path.dirname(ROOT), "mtexJulia", "dev"), "-t", "auto"]
PORTS = ("python", "matlab", "julia")


def timed(f):
  t = time.perf_counter()
  out = f()
  return out, time.perf_counter() - t


# table: op -> (call, results as rows); ops run in table order, the ones not asked for only when
# a later one needs them (always); reps a number or one per op, after one warm up run as the
# Julia half does, so that no port's time holds compilation or the loading of a cache
def run(out, algorithms, table, ops, reps=5, always=()):
  algo = {case: a["name"] for a in algorithms for case in a["cases"]}
  for op, (f, result) in table.items():
    if op not in ops and op not in always:
      continue
    n = reps.get(op, 5) if isinstance(reps, dict) else reps
    f()
    res, best = min((timed(f) for _ in range(n)), key=lambda r: r[1])
    _, _, recs = profiling.profileCall(f)
    profiling.write(os.path.join(out, f"profile_python_{op}.json"), algo[op], "python", op, best, recs)
    open(os.path.join(out, f"python_{op}.txt"), "w").write(f"{best}\n")
    np.asarray(result(res), dtype=float).astype("<f8").tofile(os.path.join(out, f"python_{op}.bin"))
    print(f"{op:20s} {1e3 * best:9.1f} ms", flush=True)


def matlab(script, *args, timeout=290):
  cmd = (f"addpath('{os.path.join(ROOT, 'bench', 'matlab')}'); addpath('{HERE}'); "
         f"{script}({', '.join(args)});")
  r = subprocess.run([os.path.join(BRIDGE, ".venv", "bin", "python"), os.path.join(BRIDGE, "matlab_run.py"), cmd],
                     timeout=timeout)
  if r.returncode:
    sys.exit(f"MATLAB failed in {script}")


# the Julia half of a benchmark, each op passed with the algorithm its profile is filed under
def julia(script, algorithms, ops, timeout=590):
  algo = [f"{case}={a['name']}" for a in algorithms for case in a["cases"]]
  r = subprocess.run(JULIA + [os.path.join(ROOT, "bench", script)] + list(ops) + ["--"] + algo, timeout=timeout)
  if r.returncode:
    sys.exit(f"Julia failed in {script}")


def cellOf(ops):
  return "{" + ", ".join(repr(o) for o in ops) + "}"


# the table: times of every port, and the deviations of Python and Julia from MATLAB by deviation(op, a, b)
def compare(out, algorithms, cols, deviation):
  print(f"{'operation':20s}" + "".join(f"{p:>11s}" for p in PORTS) + "   deviation python ~ matlab, julia ~ matlab")
  for a in algorithms:
    print(a["name"])
    for op in a["cases"]:
      t = [os.path.join(out, f"{p}_{op}.txt") for p in PORTS]
      line = f"  {op:18s}" + "".join(f"{1e3 * float(open(x).read()):9.1f}ms" if os.path.exists(x) else f"{'-':>11s}" for x in t)
      mf = os.path.join(out, f"matlab_{op}.bin")
      tol = a["tolerance"][op] if isinstance(a["tolerance"], dict) else a["tolerance"]
      for p in ("python", "julia"):
        pf = os.path.join(out, f"{p}_{op}.bin")
        if not (os.path.exists(pf) and os.path.exists(mf)):
          line += f"   {'-':>10s}"
          continue
        pa, ma = (np.fromfile(x, "<f8").reshape(-1, cols(op)) for x in (pf, mf))
        d = deviation(op, pa, ma)
        line += f"   {d:.1e} {'ok' if tol is not None and d <= tol else 'DIFFERS' if tol is not None else ''}"
        line += "" if pa.shape == ma.shape else f" shapes {pa.shape} {ma.shape}"
      print(line)
