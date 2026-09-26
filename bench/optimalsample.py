# optimalSample against the Julia port, for docs/research/benchmark-sampling.md
#
#   .venv/bin/python bench/optimalsample.py
#   (cd ../mtexJulia && julia -t auto --project=dev ../pymtex/bench/optimalsample.jl)
#
# The same targets and the same deterministic starting nodes on both sides, a fixed number
# of L-BFGS steps with the tolerances at zero, best of three after one warm-up run; the
# discrepancy before and after says whether both solve the same problem.
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtex import Miller, S2FunHarmonic, SO3FunHarmonic, crystalFrame, degree, fibreODF, optimalSample, uniformODF, zvector  # noqa: E402

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)


def targets():
  fs = S2FunHarmonic(np.array([math.sqrt(4 * np.pi), 0, 0.6 * math.sqrt(4 * np.pi / 3), 0], dtype=complex))
  c = np.zeros(10, dtype=complex)
  c[0], c[5] = 1, 0.6 / math.sqrt(3)
  fr = SO3FunHarmonic(c)
  cs = crystalFrame("222")
  fibre = SO3FunHarmonic(0.5 * uniformODF(cs) + 0.5 * fibreODF(Miller(0, 0, 1, cs), zvector, halfwidth=20 * degree), bandwidth=32)
  return {"S2 1+0.6z": fs, "SO3 1+0.6(Rz).z": fr, "SO3 222 fibre": fibre}


ALGORITHM = {
  "name": "discrete sampling",
  "entry": {"matlab": ["@S2Fun.optimalSample"], "python": ["mtex.analysis.optimalsample.optimalSample"]},
  "cases": {"optimal_s2": "optimalSample of 1 + 0.6 z on the sphere, 100 and 1000 nodes",
            "optimal_so3": "optimalSample of SO(3) densities, a 222 fibre ODF up to 2000 nodes"},
  "tolerance": None}

CASES = [("S2 1+0.6z", 32, 100, 20, False), ("S2 1+0.6z", 32, 100, 20, True), ("S2 1+0.6z", 128, 1000, 20, False),
         ("SO3 1+0.6(Rz).z", 16, 100, 20, False), ("SO3 1+0.6(Rz).z", 16, 100, 20, True),
         ("SO3 222 fibre", 16, 500, 20, False), ("SO3 222 fibre", 32, 500, 20, False), ("SO3 222 fibre", 32, 2000, 50, False),
         ("SO3 222 fibre", 64, 500, 20, False)]


def main():
  fs = targets()
  print(f"python {sys.version.split()[0]}, numpy {np.__version__}")
  print("target,L,M,steps,weights,seconds,initial,final,iterations")
  rows = []
  for name, L, M, steps, opt in CASES:
    f = fs[name]
    solve = lambda: optimalSample(f, M, bandwidth=L, maxIter=steps, tol=0, tolWeights=0, tolJ=0, warmUp=5, optimizeWeights=opt, diagnostics=True)
    solve()
    best, r = np.inf, None
    for _ in range(3):
      t = time.perf_counter()
      r = solve()
      best = min(best, time.perf_counter() - t)
    print(f"{name},{L},{M},{steps},{int(opt)},{best:.4f},{r.history[0]:.6g},{r.discrepancy:.6g},{r.iterations}", flush=True)
    rows.append((name, L, M, steps, int(opt), best, r.history[0], r.discrepancy, r.iterations))
  with open(os.path.join(out, "optimalsample_python.csv"), "w") as fh:
    for row in rows:
      fh.write(",".join(str(x) for x in row) + "\n")


if __name__ == "__main__":
  main()
