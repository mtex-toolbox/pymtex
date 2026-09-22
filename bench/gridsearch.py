# the SO(3) grid search against Julia and MATLAB: build an equispaced cubic grid, then the
# nearest search and the radius search (the one ODF reconstruction needs) on it. The twins
# are bench/gridsearch.jl and bench/gridsearch.m; the write-up is docs/research/benchmark-gridsearch.md
import sys
import time
import numpy as np
import mtex

res = 2.5 * mtex.degree
eps = 7.5 * mtex.degree
nn = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
nr = 2000
rng = np.random.default_rng(0)
cs = mtex.crystalFrame("m-3m")

t = time.perf_counter()
g = mtex.equispacedSO3Grid(cs, resolution=res)
tbuild = time.perf_counter() - t
o = mtex.orientation.rand(nn, cs, rng=rng)
mtex.find(g, o[:5]); mtex.find(g, o[:5], eps)   # warm the tree and numba


def best(f, reps=3):
  return min((lambda t: (f(), time.perf_counter() - t)[1])(time.perf_counter()) for _ in range(reps))


tn = best(lambda: mtex.find(g, o))
tr = best(lambda: mtex.find(g, o[:nr], eps))
Irad = mtex.find(g, o[:nr], eps)[0]
print(f"Python, cubic m-3m {res/mtex.degree:.1f} deg grid")
print(f"  build            {tbuild*1e3:8.0f} ms   ({g.size} points)")
print(f"  nearest {nn:6d}   {best(lambda: mtex.find(g, o))*1e3:8.1f} ms   {tn/nn*1e6:6.1f} us/query")
print(f"  radius  {nr:6d}   {tr*1e3:8.1f} ms   {tr/nr*1e6:6.1f} us/query   mean {np.diff(Irad.indptr).mean():.1f} nb @ {eps/mtex.degree:.1f} deg")
