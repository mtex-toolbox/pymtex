# classix and calcCluster against the Julia port, for docs/research/benchmark-sampling.md
#
#   .venv/bin/python bench/cluster.py            writes the point sets bench/cluster.jl reads
#   (cd ../mtexJulia && julia -t auto --project=dev ../mtexPython/bench/cluster.jl)
#
# Five Gaussian clouds in three dimensions at 10 000, 100 000 and 1 000 000 points for the
# raw algorithm, the same as directions, and four orientation clouds of a cubic crystal at
# 2 000 and 20 000 orientations through the embedding; best of three after a warm-up run.
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtex import calcCluster, classix, crystalFrame, degree, orientation, rotation, vector3d  # noqa: E402

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(7)
CENTERS = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3], [2, 2, 2]], dtype=float)


def points(n):
  return np.concatenate([c + 0.4 * rng.standard_normal((n // 5, 3)) for c in CENTERS])


def orientations(n):
  cs = crystalFrame("432")
  oc = orientation.byEuler(np.array([0, 40, 80, 20]) * degree, np.array([0, 30, 60, 80]) * degree, np.array([0, 10, 50, 70]) * degree, cs)
  rots = rotation.cat(*[rotation(oc[k].data) * rotation.byAxisAngle(vector3d.rand(n // 4, rng=rng), 2 * degree * np.abs(rng.standard_normal(n // 4)))
                        for k in range(4)])
  return orientation(rots, cs)


def timed(fun):
  fun()
  best = np.inf
  for _ in range(3):
    t = time.perf_counter()
    r = fun()
    best = min(best, time.perf_counter() - t)
  return best, r


def main():
  print(f"python {sys.version.split()[0]}, numpy {np.__version__}")
  print("case,n,seconds,clusters")
  rows = []
  for n in (10_000, 100_000, 1_000_000):
    X = points(n)
    np.savetxt(os.path.join(out, f"cluster_points_{n}.csv"), X, delimiter=",", fmt="%.10g")
    t, labels = timed(lambda: classix(X, 0.1, minPts=n // 100))
    print(f"classix,{n},{t:.4f},{labels.max() + 1}", flush=True)
    rows.append(("classix", n, t, labels.max() + 1))
    if n <= 100_000:
      v = vector3d(X).normalize()
      t, (cId, center) = timed(lambda: calcCluster(v))
      print(f"calcCluster directions,{n},{t:.4f},{center.size}", flush=True)
      rows.append(("calcCluster directions", n, t, center.size))
  for n in (2_000, 20_000):
    o = orientations(n)
    np.savetxt(os.path.join(out, f"cluster_quats_{n}.csv"), o.data.reshape(-1, 4), delimiter=",", fmt="%.12g")
    t, (cId, center) = timed(lambda: calcCluster(o, method="classix", radius=0.1, minPoints=n // 100))
    print(f"calcCluster orientations,{n},{t:.4f},{center.size}", flush=True)
    rows.append(("calcCluster orientations", n, t, center.size))
  with open(os.path.join(out, "cluster_python.csv"), "w") as fh:
    for row in rows:
      fh.write(",".join(str(x) for x in row) + "\n")


if __name__ == "__main__":
  main()
