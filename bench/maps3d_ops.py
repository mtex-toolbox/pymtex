# the maps of volumes against MATLAB: the import of a DREAM.3D and an xnovo volume, the grain
# segmentation of both and the kernel average misorientation of the voxels; the Julia port has no
# maps of volumes, so there is no Julia half
#
#   .venv/bin/python bench/maps3d_ops.py run [op ...]        this port
#   .venv/bin/python bench/maps3d_ops.py matlab [op ...]     bench/matlab/bench_maps3d_ops.m through the bridge
#   .venv/bin/python bench/maps3d_ops.py compare
#
# The volumes are SmallIN100 of DREAM.3D, 100^3 voxels of a nickel superalloy, and the simulated
# xnovo volume of mtex-data, 50^3 voxels, a cylinder of nine phases; each port reads the files
# itself, so there is no export. A voxel result keeps every voxel whose three lattice indices are
# multiples of STRIDE, rows (x, y, z, ...); a grain result is a row (centroid, volume, numPixel) per
# grain. The ports number voxels and grains their own way, so compare matches rows by the position
# in the first three columns.
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "graph"))
import crosscheck  # noqa: E402

OUT = os.path.join(HERE, "out", "maps3d")
IN100 = os.path.expanduser("~/mtex/data/SmallIN100_MeshStats.dream3d")
XNOVO = os.path.join(os.path.dirname(os.path.dirname(HERE)), "mtex-data", "EBSD3", "SimulatedMultiPhase.h5")
STRIDE = {"in100": 7, "xnovo": 3}
ALGORITHM = [
  {"name": "3D EBSD import",
   "entry": {"matlab": ["@EBSD3.load", "loadEBSD_dream3d", "loadEBSD_xnovo"],
             "python": ["mtex.io.load3d.loadEBSD3", "mtex.io.load3d._loadEBSD_dream3d", "mtex.io.load3d._loadEBSD_xnovo"],
             "julia": []},
   "cases": {"load_dream3d": "EBSD3.load of SmallIN100, 100^3 voxels of DREAM.3D: position, phase and rotation of every 7th voxel per axis",
             "load_xnovo": "EBSD3.load of the simulated xnovo volume, 50^3 voxels of nine phases: every 3rd voxel per axis"},
   # the rotations of the same Euler angles or Rodrigues vectors, apart by rounding only
   "tolerance": 1e-12},
  {"name": "3D grain segmentation",
   "entry": {"matlab": ["@EBSD3square.calcGrains", "@grain3d.volume", "@grain3d.centroid"],
             "python": ["mtex.maps3d.calcgrains3d.calcGrains", "mtex.maps3d.grain3d.grain3d.volume",
                        "mtex.maps3d.grain3d.grain3d.centroid"],
             "julia": []},
   "cases": {"calcgrains3d_in100": "calcGrains of SmallIN100 at 5 degree: centroid, volume and voxel count of every grain",
             "calcgrains3d_sim": "calcGrains of the xnovo volume at 5 degree: centroid, volume and voxel count of every grain"},
   # the volumes are sums of unit cube faces by the divergence theorem, exact up to rounding
   "tolerance": 1e-10},
  {"name": "3D intragranular misorientation",
   "entry": {"matlab": ["@EBSD3.KAM"], "python": ["mtex.maps3d.intragranular3d.KAM"], "julia": []},
   "cases": {"kam3d_in100": "KAM, first order, of SmallIN100 within the grains of calcgrains3d_in100: every 7th voxel per axis"},
   # MATLAB takes small angles through acos
   "tolerance": 1e-6},
]
COLS = {"load_dream3d": 8, "load_xnovo": 8, "calcgrains3d_in100": 5, "calcgrains3d_sim": 5, "kam3d_in100": 4}


# the voxels whose lattice indices along every axis are multiples of the stride
def sampled(e, stride):
  xyz = np.asarray(e.pos.data, float).reshape(-1, 3)
  idx = np.column_stack([np.unique(np.round(c, 9), return_inverse=True)[1] for c in xyz.T])
  return np.all(idx % stride == 0, axis=1), xyz


def voxelRows(e, stride):
  keep, xyz = sampled(e, stride)
  q = np.asarray(e.rotations.data, float).reshape(-1, 4)
  return np.column_stack([xyz, np.asarray(e.phaseId).reshape(-1), q])[keep]


def grainRows(g):
  return np.column_stack([np.asarray(g.centroid.data, float).reshape(-1, 3), np.asarray(g.volume, float), g.numPixel])


def run(ops):
  from mtex import calcGrains, degree, loadEBSD3
  s = {}
  if any(op.endswith("in100") for op in ops):
    s["in100"] = loadEBSD3(IN100)
  if "calcgrains3d_sim" in ops:
    s["xnovo"] = loadEBSD3(XNOVO)

  # the KAM of the sampled voxels after their positions
  def kam(k):
    keep, xyz = sampled(s["in100"], STRIDE["in100"])
    return np.column_stack([xyz, np.asarray(k, float).reshape(-1)])[keep]

  table = {
    "load_dream3d": (lambda: loadEBSD3(IN100), lambda e: voxelRows(e, STRIDE["in100"])),
    "load_xnovo": (lambda: loadEBSD3(XNOVO), lambda e: voxelRows(e, STRIDE["xnovo"])),
    "calcgrains3d_in100": (lambda: calcGrains(s["in100"], angle=5 * degree), grainRows),
    "calcgrains3d_sim": (lambda: calcGrains(s["xnovo"], angle=5 * degree), grainRows),
    "kam3d_in100": (lambda: s["in100"].KAM(), kam)}
  crosscheck.run(OUT, ALGORITHM, table, ops, always=("calcgrains3d_in100",) if "kam3d_in100" in ops else ())


# MATLAB one op per call, each limited to 20 times Python's time per run plus loading, since a
# call that overruns keeps MATLAB busy after its client is gone: the first timeout ends the batch
def matlab(ops):
  for op in ops:
    f = os.path.join(OUT, f"python_{op}.txt")
    limit = min(290, 20 * float(open(f).read()) * 7 + 60) if os.path.exists(f) else 290
    print(f"{op}: limit {limit:.0f} s", flush=True)
    crosscheck.matlab("bench_maps3d_ops", repr(OUT), crosscheck.cellOf([op]), "1e7", repr(IN100), repr(XNOVO), timeout=limit)


# rows matched by their positions to the nearest within 1e-6; the voxels by their phase, which must
# agree, and the angle between their rotations where both are indexed; the grains by their voxel
# counts, which must agree, and their volumes relative to the largest; the KAM, less MATLAB's offset,
# relative to the largest
def deviation(op, a, b):
  from scipy.spatial import cKDTree
  d, j = cKDTree(b[:, :3]).query(a[:, :3])
  if len(a) != len(b) or d.max() > 1e-6 or len(set(j)) != len(j):
    print(f"  {op}: {len(a)} and {len(b)} rows, {int((d > 1e-6).sum())} of this port's without a match")
    return float("nan")
  va, vb = a[:, 3:], b[j, 3:]
  if op.startswith("load"):
    if (va[:, 0] != vb[:, 0]).any():
      print(f"  {op}: {int((va[:, 0] != vb[:, 0]).sum())} voxels of another phase")
      return float("nan")
    ok = (va[:, 0] > 0) & np.isfinite(va[:, 1:]).all(axis=1) & np.isfinite(vb[:, 1:]).all(axis=1)
    # the angle by the chord of the nearer of q and -q, which keeps its digits near zero
    chord = np.minimum(np.linalg.norm(va[ok, 1:] - vb[ok, 1:], axis=1), np.linalg.norm(va[ok, 1:] + vb[ok, 1:], axis=1))
    return float(np.max(4 * np.arcsin(chord / 2), initial=0))
  if op.startswith("calcgrains"):
    if (va[:, 1] != vb[:, 1]).any():
      print(f"  {op}: {int((va[:, 1] != vb[:, 1]).sum())} grains of another voxel count")
      return float("nan")
    return float(np.max(np.abs(va[:, 0] - vb[:, 0])) / np.max(np.abs(vb[:, 0])))
  # MATLAB adds 1e-5 to every angle it averages, to keep a zero in its sparse matrix
  vb = vb - 1e-5
  if (np.isnan(va) != np.isnan(vb)).any():
    print(f"  {op}: NaN in {int(np.isnan(va).sum())} Python and {int(np.isnan(vb).sum())} MATLAB rows")
  both = ~(np.isnan(va) | np.isnan(vb))
  return float(np.max(np.abs(va - vb)[both]) / np.max(np.abs(vb[both])))


if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "compare", sys.argv[2:] or list(COLS)
  os.makedirs(OUT, exist_ok=True)
  {"run": lambda: run(names),
   "matlab": lambda: matlab(names),
   "compare": lambda: crosscheck.compare(OUT, ALGORITHM, lambda op: COLS[op], deviation)}[cmd]()
