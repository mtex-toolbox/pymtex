# E1 on the frame-free layer: the cost per element of the kernels on the (..., k) layout
import time
import numpy as np
import mtex

n = 1_000_000
rng = np.random.default_rng(0)
r = mtex.rotation.rand(n, rng=rng)
s = mtex.rotation.rand(n, rng=rng)
v = mtex.vector3d.rand(n, rng=rng)
r32, v32 = r.astype(np.float32), v.astype(np.float32)


def bench(name, f, reps=3):
  best = min(timeit(f) for _ in range(reps))
  print(f"{name:32s} {best * 1e9 / n:7.1f} ns per element")


def timeit(f):
  t = time.perf_counter()
  f()
  return time.perf_counter() - t


bench("product r * s", lambda: r * s)
bench("rotate r * v", lambda: r * v)
bench("angle(r, s)", lambda: mtex.angle(r, s))
bench("dot(r, s)", lambda: mtex.dot(r, s))
bench("Euler(r)", lambda: mtex.Euler(r))
bench("byEuler", lambda: mtex.rotation.byEuler(*mtex.Euler(r)))
bench("matrix(r)", lambda: mtex.matrix(r))
bench("byMatrix", lambda: mtex.rotation.byMatrix(mtex.matrix(r)))
bench("axis(r)", lambda: mtex.axis(r))
bench("homochoric(r)", lambda: mtex.homochoric(r))
bench("byHomochoric", lambda: mtex.rotation.byHomochoric(mtex.homochoric(r)))
bench("unique(r)", lambda: mtex.unique(r), reps=1)
bench("mean(r)", lambda: mtex.mean(r))
bench("angle(v, w)", lambda: mtex.angle(v, v))
bench("v + v", lambda: v + v)
bench("r[::2]", lambda: r[::2])
bench("cat(r, s)", lambda: mtex.cat(r, s))
bench("float32 product", lambda: r32 * r32)
bench("float32 rotate", lambda: r32 * v32)
