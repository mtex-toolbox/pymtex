# E3: the harmonic transforms - the coefficient transforms, finufft at arbitrary nodes and
# the FFT on the Clenshaw-Curtis grid, at the bandwidths of docs/plans/functions.md 8
import sys
import time

import numpy as np

import mtex
from mtex import (S2FourierPlan, S2GridPlan, SO3FourierPlan, SO3GridPlan, deg2dim, quadratureS2Grid, quadratureSO3Grid,
                  setMTEXpref, sphericalTransform, sphericalTransformAdjoint, wignerTransform, wignerTransformAdjoint)

rng = np.random.default_rng(0)
J = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000

def timeit(f, reps=3):
  f()
  times = []
  for _ in range(reps):
    t = time.perf_counter()
    f()
    times.append(time.perf_counter() - t)
  return min(times)

def randomWigner(L):
  from mtex import degreeBlock
  fhat = rng.standard_normal(deg2dim(L + 1)) + 1j * rng.standard_normal(deg2dim(L + 1))
  for n in range(L + 1):
    F = degreeBlock(fhat, n)
    k = np.arange(-n, n + 1)
    sign = np.where((k[:, None] - k[None, :]) % 2, -1.0, 1.0)
    F[...] = (F + (sign * np.conj(F))[::-1, ::-1]) / 2
  return fhat

print(f"nodes J = {J}")
for backend in ("numba", "numpy"):
  setMTEXpref("kernels", backend)
  for L in (64, 128):
    fhat = randomWigner(L)
    G = wignerTransform(fhat, L, real=True)
    print(f"[{backend}] SO(3) L = {L}: wignerTransform {timeit(lambda: wignerTransform(fhat, L, real=True)):.3f} s, "
          f"adjoint {timeit(lambda: wignerTransformAdjoint(G, L, real=True)):.3f} s")
  if backend == "numpy":
    continue
  for L in (64,):
    fhat = randomWigner(L)
    rots = mtex.rotation.rand(J, rng=rng)
    t = time.perf_counter()
    p = SO3FourierPlan(rots, L)
    tp = time.perf_counter() - t
    f = p(fhat)
    w = np.full(J, 1 / J)
    print(f"SO(3) L = {L}, J = {J}: plan {tp:.3f} s, evaluate {timeit(lambda: p(fhat)):.3f} s, adjoint {timeit(lambda: p.adjoint(f, w)):.3f} s")
    g = quadratureSO3Grid(L)
    gp = SO3GridPlan(g, L)
    v = gp(fhat)
    print(f"SO(3) L = {L}, Clenshaw-Curtis grid of {len(g)} nodes: FFT evaluate {timeit(lambda: gp(fhat)):.3f} s, "
          f"adjoint {timeit(lambda: gp.adjoint(v, g.weights)):.3f} s")
  for L in (256,):
    fhat = rng.standard_normal((L + 1) ** 2) + 1j * rng.standard_normal((L + 1) ** 2)
    print(f"S2 L = {L}: sphericalTransform {timeit(lambda: sphericalTransform(fhat, L)):.3f} s, "
          f"adjoint {timeit(lambda: sphericalTransformAdjoint(sphericalTransform(fhat, L), L)):.3f} s")
    v = mtex.vector3d.rand(J, rng=rng)
    t = time.perf_counter()
    p = S2FourierPlan(v, L)
    tp = time.perf_counter() - t
    f = p(fhat)
    w = np.full(J, 1 / J)
    print(f"S2 L = {L}, J = {J}: plan {tp:.3f} s, evaluate {timeit(lambda: p(fhat)):.3f} s, adjoint {timeit(lambda: p.adjoint(f, w)):.3f} s")
    g = quadratureS2Grid(L)
    gp = S2GridPlan(g, L)
    vg = gp(fhat)
    print(f"S2 L = {L}, Clenshaw-Curtis grid of {len(g)} nodes: FFT evaluate {timeit(lambda: gp(fhat)):.3f} s, "
          f"adjoint {timeit(lambda: gp.adjoint(vg, g.weights)):.3f} s")
