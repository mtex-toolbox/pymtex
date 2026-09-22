# E2 for the maps layer: the spatial decomposition on square and hexagonal maps of 90 000,
# one million and nine million sites with three round holes and 0.5 % speckle, the setup of
# the Julia E5 (docs/research/benchmark-decomposition.md of the Julia port). Times the
# closing and the sites, the clipping through numba on one thread and on all, and through
# the C loop of decomposition_kernel.c on one thread, and the vertex welding.
#
#   .venv/bin/python bench/decomposition.py [sizes...]
import ctypes
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mtex import EBSD, crystalFrame, degree, phaseList, rotation, zvector   # noqa: E402
from mtex.maps.decomposition import buildSites, candidateLevels, localVoronoi, meshVertices   # noqa: E402
from mtex.maps.kernels import localVoronoiNumba   # noqa: E402

CS = crystalFrame("m-3m", mineral="bench")
SQ = np.eye(2)
HEX = np.array([[1.0, 0.5], [0.0, np.sqrt(3) / 2]])


def benchMap(A, n, rng):
  i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
  i, j = i.reshape(-1), j.reshape(-1)
  pos = np.stack([A[0, 0] * i + A[0, 1] * j, A[1, 0] * i + A[1, 1] * j, np.zeros(n * n)], axis=-1)
  rot = rotation.byAxisAngle(zvector, ((i // 20 + (j // 20) * 3) * 20 * degree))
  ph = np.ones(n * n, dtype=np.int8)
  for ci, cj, r in ((n // 4, n // 4, 1.5), (n // 2, n // 2, 4.0), (3 * n // 4, 3 * n // 4, 9.0)):
    ph[np.hypot(i - ci, j - cj) <= r] = 0
  ph[rng.random(n * n) < 0.005] = 0
  return EBSD(pos, rot, ph, phaseList(CS))


def timed(f, *args):
  t = time.perf_counter()
  out = f(*args)
  return out, time.perf_counter() - t


def cKernel():
  here = os.path.dirname(os.path.abspath(__file__))
  so = os.path.join(here, "decomposition_kernel.so")
  src = os.path.join(here, "decomposition_kernel.c")
  if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
    subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", so, src], check=True)
  lib = ctypes.CDLL(so)
  lib.localVoronoi.restype = ctypes.c_int64
  return lib


def runC(lib, sites, levels):
  ns = len(sites)
  cap = 4 * ns + 256
  cells = np.empty((cap, 2), dtype=np.int32)
  verts = np.empty((cap, 4), dtype=np.float64)
  unb = np.zeros(1, dtype=np.int64)
  P = ctypes.POINTER
  args = [np.ascontiguousarray(sites.pos, dtype=np.float64).ctypes.data_as(P(ctypes.c_double)),
          np.ascontiguousarray(sites.ij, dtype=np.int32).ctypes.data_as(P(ctypes.c_int32)),
          np.ascontiguousarray(sites.pixel, dtype=np.int32).ctypes.data_as(P(ctypes.c_int32)),
          np.ascontiguousarray(sites.isHole, dtype=np.uint8).ctypes.data_as(P(ctypes.c_uint8)),
          np.ascontiguousarray(sites.lookup, dtype=np.int32).ctypes.data_as(P(ctypes.c_int32)),
          ctypes.c_int64(sites.lookup.shape[0]), ctypes.c_int64(sites.lookup.shape[1]),
          ctypes.c_int64(sites.offset[0]), ctypes.c_int64(sites.offset[1]), ctypes.c_int64(ns)]
  keep = []
  for offs, d, W in levels:
    o = np.ascontiguousarray(offs, dtype=np.int64)
    dd = np.ascontiguousarray(d, dtype=np.float64)
    keep += [o, dd]
    args += [o.ctypes.data_as(P(ctypes.c_int64)), dd.ctypes.data_as(P(ctypes.c_double)), ctypes.c_int64(len(o)), ctypes.c_double(W)]
  args += [ctypes.c_double(0.05), cells.ctypes.data_as(P(ctypes.c_int32)), verts.ctypes.data_as(P(ctypes.c_double)),
           ctypes.c_int64(cap), unb.ctypes.data_as(P(ctypes.c_int64))]
  n = lib.localVoronoi(*args)
  assert n >= 0, n
  return cells[:n], verts[:n], int(unb[0])


def main(sizes):
  import numba
  rng = np.random.default_rng(0)
  lib = cKernel()
  print("| grid | sites | build | closing+sites | numba 1 thread | numba all threads | C 1 thread | weld | faces | vertices |")
  print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
  for A, name in ((SQ, "square"), (HEX, "hex")):
    for n in sizes:
      e, tBuild = timed(benchMap, A, n, rng)
      l = e.lattice
      P = l.plane.to2d(e.pos.data.reshape(-1, 3))
      present = e.isIndexed.reshape(-1)
      sites, tSites = timed(buildSites, l, P, present, 3.1)
      levels = candidateLevels(l.A, 3.1, P.dtype)
      localVoronoiNumba(sites, levels)                      # compile
      numba.set_num_threads(1)
      (cells, verts, unb), tOne = timed(localVoronoiNumba, sites, levels)
      numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)
      _, tAll = timed(localVoronoiNumba, sites, levels)
      (cellsC, vertsC, unbC), tC = timed(runC, lib, sites, levels)
      assert len(cellsC) == len(cells) and unb == unbC
      (V, F, cells2), tWeld = timed(meshVertices, cells, verts, l.dxy / 100)
      print(f"| {name} | {len(sites)} | {tBuild:.2f} | {tSites:.2f} | {tOne:.2f} | {tAll:.2f} | {tC:.2f} | {tWeld:.2f} | {len(F)} | {len(V)} |")
      sys.stdout.flush()


if __name__ == "__main__":
  main([int(s) for s in sys.argv[1:]] or [300, 1000])
