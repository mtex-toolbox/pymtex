# the user facing harmonic operations against MATLAB's master of 2026-09-25, the cases of
# its speed commits; bench/matlab/bench_harmonics.m runs the same through the bridge
import sys
import time

import numpy as np

import mtex
from mtex import (S2FunHarmonic, SO3FunHarmonic, crystalFrame, deg2dim, orientation, quadratureS2Grid,
                  quadratureSO3Grid, rotation, specimenFrame, vector3d)
from mtex.functions.quadrature import bandwidthFor
from mtex.functions.transforms import isRealWigner, sphericalTransform, sphericalTransformAdjoint

rng = np.random.default_rng(0)
only = sys.argv[1:]


def timeit(f, reps=3):
  f()
  times = []
  for _ in range(reps):
    t = time.perf_counter()
    f()
    times.append(time.perf_counter() - t)
  return min(times)


def report(name, t):
  print(f"{name:48s} {1000 * t:9.1f} ms", flush=True)


def want(tag):
  return not only or tag in only


def randomODF(L, cs, ss):
  fhat = rng.standard_normal(deg2dim(L + 1)) + 1j * rng.standard_normal(deg2dim(L + 1))
  return SO3FunHarmonic(fhat, cs, ss, real=True)


tri = crystalFrame("1")
cub = crystalFrame("m-3m")
spec = specimenFrame.default

if want("eval"):
  for L, J in ((64, 10**6), (128, 10**4), (128, 10**6)):
    for name, cs in (("1", tri), ("m-3m", cub)):
      F = randomODF(L, cs, spec)
      ori = orientation(rotation.rand(J, rng=rng), cs, spec)
      report(f"SO3 eval L={L} J={J:.0e} {name}", timeit(lambda: F.eval(ori), 5))

if want("adjoint"):
  for L, J in ((64, 10**6), (128, 10**4), (128, 10**6)):
    for name, cs in (("1", tri), ("m-3m", cub)):
      ori = orientation(rotation.rand(J, rng=rng), cs, spec)
      v = rng.random(J)
      report(f"SO3 quadrature L={L} J={J:.0e} {name}",
             timeit(lambda: SO3FunHarmonic.quadrature(ori, v, bandwidth=L), 5))

if want("grid"):
  for name, cs, ss in (("1/1", tri, spec), ("432/1", crystalFrame("432"), spec),
                       ("622/222", crystalFrame("622"), specimenFrame("222"))):
    L = bandwidthFor(127, cs, ss)
    F = randomODF(L, cs, ss)
    g = quadratureSO3Grid(L, cs, ss)
    report(f"SO3 eval Clenshaw-Curtis L={L} {name}", timeit(lambda: F.eval(g), 2))
    v = F.eval(g)
    report(f"SO3 quadrature Clenshaw-Curtis L={L} {name}", timeit(lambda: SO3FunHarmonic.quadrature(g, v), 2))

if want("sym"):
  for name, cs, ss in (("622/222", crystalFrame("622"), specimenFrame("222")),
                       ("432/mmm", crystalFrame("432"), specimenFrame("mmm")),
                       ("432/1", crystalFrame("432"), spec)):
    L = 128
    F = randomODF(L, cs, ss)
    report(f"SO3 symmetrise L={L} {name}", timeit(lambda: F.symmetrise()))

if want("misc"):
  L = 128
  F = randomODF(L, tri, spec)
  report(f"SO3 isReal L={L}", timeit(lambda: isRealWigner(F._fhat)))
  r = rotation.rand(1, rng=rng)
  report(f"SO3 rotate L={L}", timeit(lambda: F.rotate(r)))
  report("SO3 rotate L=64", timeit(lambda: F.truncate(64).rotate(r)))

if want("s2"):
  L = 128
  fhat = rng.standard_normal((L + 1) ** 2) + 1j * rng.standard_normal((L + 1) ** 2)
  sF = S2FunHarmonic(fhat)
  sF.isReal = True
  v = vector3d.rand(10**6, rng=rng)
  report(f"S2 eval L={L} J=1e6", timeit(lambda: sF.eval(v), 2))
  vals = rng.random(10**6)
  report(f"S2 quadrature L={L} J=1e6", timeit(lambda: S2FunHarmonic.quadrature(v, vals, bandwidth=L), 2))
  # the bandwidth 1024 transforms take a minute before T3, so only on the tag 'big'
  for L in (256, 512, 1024) if "big" in only else (256, 512):
    fh = rng.standard_normal((L + 1) ** 2) + 1j * rng.standard_normal((L + 1) ** 2)
    report(f"S2 sphericalTransform L={L}", timeit(lambda: sphericalTransform(fh, L), 2))
    G = sphericalTransform(fh, L)
    report(f"S2 sphericalTransformAdjoint L={L}", timeit(lambda: sphericalTransformAdjoint(G, L), 2))
  for L in (256, 512, 1024):
    g = quadratureS2Grid(L)
    fh = rng.standard_normal((L + 1) ** 2) + 1j * rng.standard_normal((L + 1) ** 2)
    sF = S2FunHarmonic(fh)
    report(f"S2 eval Clenshaw-Curtis L={L}", timeit(lambda: sF.eval(g), 2))
  f = lambda v: np.exp(v.z)
  report("S2 quadrature of a callable L=256", timeit(lambda: S2FunHarmonic.quadrature(f, bandwidth=256), 2))
  sF = S2FunHarmonic(rng.standard_normal(129 ** 2) + 0j, crystalFrame("m-3m"))
  report("S2 symmetrise m-3m L=128", timeit(lambda: sF.symmetrise()))
  v = vector3d.rand(10**5, rng=rng)
  report("S2 calcDensity 1e5", timeit(lambda: v.calcDensity(), 2))
