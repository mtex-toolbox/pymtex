# E2 of docs/PLAN.md: the Hamilton product and the search against the images of a reference
# as NumPy, cache-blocked NumPy, numba and C through ctypes; results in docs/research/benchmark-e2-kernels.md.
# gcc -O3 -march=native -shared -fPIC -o bench/e2_kernels.so bench/e2_kernels.c, then a venv with numba
import sys, time, ctypes, numpy as np
sys.path.insert(0, ".")
import mtex
from mtex.geometry.quaternion import qmul
from mtex.geometry.symreduce import bestDot, referenceImages, reductionGroups
n = 1_000_000
rng = np.random.default_rng(0)
r, s = mtex.rotation.rand(n, rng=rng), mtex.rotation.rand(n, rng=rng)
p, q = np.ascontiguousarray(r.data), np.ascontiguousarray(s.data)
cs, hx = mtex.crystalFrame("m-3m"), mtex.crystalFrame("6/mmm")
mo = mtex.orientation.rand(n, cs, hx, rng=rng); L, R = reductionGroups(mo)
es, _ = referenceImages(mo[0].data, L, R); es = np.ascontiguousarray(es); qs = np.ascontiguousarray(mo.data)
def best(f, reps=5):
  f(); return min((lambda t: (f(), time.perf_counter() - t)[1])(time.perf_counter()) for _ in range(reps))
def row(name, f, check=None):
  t = best(f); print(f"{name:40s} {t*1e9/n:8.1f} ns/elem" + (f"   check {check(f()):.4f}" if check else ""))
print("--- Hamilton product, 1e6")
row("NumPy, the port", lambda: qmul(p, q))
# cache blocking: the 28 passes on a chunk that stays in L2
pc, qc = np.ascontiguousarray(p.T), np.ascontiguousarray(q.T)
out = np.empty((4, n))
def blocked(block=2048):
  for lo in range(0, n, block):
    a1, b1, c1, d1 = pc[:, lo:lo+block]; a2, b2, c2, d2 = qc[:, lo:lo+block]; o = out[:, lo:lo+block]
    np.multiply(a1, a2, out=o[0]); o[0] -= b1 * b2; o[0] -= c1 * c2; o[0] -= d1 * d2
    np.multiply(b1, a2, out=o[1]); o[1] += a1 * b2; o[1] -= d1 * c2; o[1] += c1 * d2
    np.multiply(c1, a2, out=o[2]); o[2] += d1 * b2; o[2] += a1 * c2; o[2] -= b1 * d2
    np.multiply(d1, a2, out=o[3]); o[3] -= c1 * b2; o[3] += b1 * c2; o[3] += a1 * d2
  return out
row("NumPy, columns, cache blocked 2048", blocked)
row("NumPy, columns, cache blocked 16384", lambda: blocked(16384))
lib = ctypes.CDLL("./bench/e2_kernels.so")
co = np.empty((n, 4))
D = ctypes.c_void_p
lib.qmul.argtypes = [D, D, D, ctypes.c_long]; lib.bestdot.argtypes = [D, D, ctypes.c_long, ctypes.c_long, D, D]
def cq(): lib.qmul(p.ctypes.data, q.ctypes.data, co.ctypes.data, n); return co
row("C via ctypes, (n,4) rows", cq)
assert np.allclose(cq(), qmul(p, q))
try:
  import numba
  @numba.njit(cache=False, fastmath=False)
  def nq(p, q, out):
    for i in range(p.shape[0]):
      a0, a1, a2, a3 = p[i, 0], p[i, 1], p[i, 2], p[i, 3]; b0, b1, b2, b3 = q[i, 0], q[i, 1], q[i, 2], q[i, 3]
      out[i, 0] = a0*b0 - a1*b1 - a2*b2 - a3*b3; out[i, 1] = a1*b0 + a0*b1 - a3*b2 + a2*b3
      out[i, 2] = a2*b0 + a3*b1 + a0*b2 - a1*b3; out[i, 3] = a3*b0 - a2*b1 + a1*b2 + a0*b3
    return out
  @numba.njit(parallel=True)
  def nqp(p, q, out):
    for i in numba.prange(p.shape[0]):
      a0, a1, a2, a3 = p[i, 0], p[i, 1], p[i, 2], p[i, 3]; b0, b1, b2, b3 = q[i, 0], q[i, 1], q[i, 2], q[i, 3]
      out[i, 0] = a0*b0 - a1*b1 - a2*b2 - a3*b3; out[i, 1] = a1*b0 + a0*b1 - a3*b2 + a2*b3
      out[i, 2] = a2*b0 + a3*b1 + a0*b2 - a1*b3; out[i, 3] = a3*b0 - a2*b1 + a1*b2 + a0*b3
    return out
  @numba.njit
  def nbest(qs, es, val, idx):
    for i in range(qs.shape[0]):
      b = -1.0; k = 0
      for j in range(es.shape[0]):
        d = abs(qs[i, 0]*es[j, 0] + qs[i, 1]*es[j, 1] + qs[i, 2]*es[j, 2] + qs[i, 3]*es[j, 3])
        if d > b: b = d; k = j
      val[i] = b; idx[i] = k
    return val
  t = time.perf_counter(); nq(p[:10], q[:10], co[:10]); nqp(p[:10], q[:10], co[:10]); nbest(qs[:10], es, np.empty(10), np.empty(10, dtype=np.int64)); print(f"numba compile {time.perf_counter()-t:.1f} s")
  row("numba, (n,4) rows, one thread", lambda: nq(p, q, co))
  row(f"numba, parallel ({numba.get_num_threads()} threads)", lambda: nqp(p, q, co))
except ImportError:
  print("no numba"); nbest = None
print("--- search against 288 images, 1e6")
row("NumPy, the port (whole matrix)", lambda: bestDot(qs, es), lambda x: x[0].mean())
def blockedsearch(block=4096):
  k = np.empty(n, dtype=int); d = np.empty(n)
  for lo in range(0, n, block):
    dd = np.abs(qs[lo:lo+block] @ es.T); kk = dd.argmax(axis=1); k[lo:lo+block] = kk; d[lo:lo+block] = dd[np.arange(len(kk)), kk]
  return d, k
row("NumPy, blocked 4096", blockedsearch, lambda x: x[0].mean())
val, idx = np.empty(n), np.empty(n, dtype=np.int64)
def cb(): lib.bestdot(qs.ctypes.data, es.ctypes.data, n, len(es), val.ctypes.data, idx.ctypes.data); return val
row("C via ctypes", cb, lambda x: x.mean())
if nbest: row("numba, one thread", lambda: nbest(qs, es, val, idx), lambda x: x.mean())
