# the geometry benchmark against Julia and MATLAB (docs/plans/benchmark.md): the quaternion
# products and every symmetry-reduced comparison, one million elements, best of five; the
# twins are bench/geometry.jl and bench/geometry.m, the table docs/research/benchmark-geometry.md
import sys
import time
import numpy as np
import mtex

n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
rng = np.random.default_rng(0)
cs, hx = mtex.crystalFrame("m-3m"), mtex.crystalFrame("6/mmm")
r, s = mtex.rotation.rand(n, rng=rng), mtex.rotation.rand(n, rng=rng)
v = mtex.vector3d.rand(n, rng=rng)
ori, ori2 = mtex.orientation.rand(n, cs, rng=rng), mtex.orientation.rand(n, cs, rng=rng)
ref = ori[0]
mori, mori2 = mtex.orientation.rand(n, cs, hx, rng=rng), mtex.orientation.rand(n, cs, hx, rng=rng)
mref = mori[0]
m = mtex.Miller(mtex.vector3d.rand(n, rng=rng), cs)


# the mean of the noSymmetry dot to the reference is the reduced dot, the check that the
# projection landed on the closest equivalent
def toRef(p, q):
  return np.mean(np.abs(np.sum(p.data * q.data, axis=-1)))


rows = [
  ("r * s", lambda: r * s, lambda q: np.mean(np.abs(q.data[:, 0]))),
  ("r * v", lambda: r * v, lambda w: np.mean(np.abs(w.data[:, 2]))),
  ("dot(ori, ori2)", lambda: mtex.dot(ori, ori2), np.mean),
  ("dot(ori, ref)", lambda: mtex.dot(ori, ref), np.mean),
  ("dot(mori, mori2)", lambda: mtex.dot(mori, mori2), np.mean),
  ("dot(mori, mref)", lambda: mtex.dot(mori, mref), np.mean),
  ("project2FR(m)", lambda: mtex.projectIntoFundamentalRegion(m), lambda w: np.mean(w.data[:, 2])),
  ("project2FR(ori, ref)", lambda: mtex.projectIntoFundamentalRegion(ori, ref), lambda p: toRef(p, ref)),
  ("project2FR(ori, ori2)", lambda: mtex.projectIntoFundamentalRegion(ori, ori2), lambda p: toRef(p, ori2)),
  ("project2FR(mori, mref)", lambda: mtex.projectIntoFundamentalRegion(mori, mref), lambda p: toRef(p, mref)),
  ("project2FR(mori, mori2)", lambda: mtex.projectIntoFundamentalRegion(mori, mori2), lambda p: toRef(p, mori2)),
]

print(f"Python, N = {n}, m-3m and m-3m -> 6/mmm, best of 5")
for name, f, check in rows:
  out = f()
  best = np.inf
  for _ in range(5):
    t = time.perf_counter()
    f()
    best = min(best, time.perf_counter() - t)
  print(f"{name:26s} {best * 1e3:9.1f} ms {best * 1e9 / n:8.1f} ns/elem   check {check(out):.4f}")
