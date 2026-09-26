# the import against MATLAB: the EBSD maps of the vendor formats and the pole figures, read by
# each port from the same files
#
#   .venv/bin/python bench/io_ops.py run [op ...]        this port
#   .venv/bin/python bench/io_ops.py matlab [op ...]     bench/matlab/bench_io_ops.m through the bridge
#   .venv/bin/python bench/io_ops.py compare
#
# A map is written as rows (x, y, phase, a, b, c, d), the phase 0 for notIndexed and the rows
# matched by position; the pole figures as rows (x, y, z, intensity) in the order of the file.
# Both ports read the list the file holds, without putting it on its grid.
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "graph"))
import crosscheck  # noqa: E402

OUT = os.path.join(HERE, "out", "io")
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "master", "data")
FILES = {"load_ctf": os.path.join(DATA, "EBSD", "Forsterite.ctf"), "load_ang": os.path.join(DATA, "EBSD", "ferrite.ang"),
         "load_osc": os.path.join(DATA, "EBSD", "olivineopticalmap.ang"), "load_1C_1": "/home/hielscher/mtex/data/1C_1.ctf"}
ALGORITHM = [
  {"name": "EBSD import",
   "entry": {"matlab": ["@EBSD.load", "loadEBSD_ang", "loadEBSD_ctf"],
             "python": ["mtex.io.load.loadEBSD", "mtex.io.ang.loadEBSD_ang", "mtex.io.ctf.loadEBSD_ctf"]},
   "cases": {"load_ctf": "EBSD.load of Forsterite.ctf, every measurement",
             "load_ang": "EBSD.load of ferrite.ang, hexagonal grid",
             "load_osc": "EBSD.load of olivineopticalmap.ang, an optical map",
             "load_1C_1": "EBSD.load of the 2.5 million rows of 1C_1.ctf, one run"},
   # the rotation angle between the ports in radians; the files store single precision angles
   "tolerance": 1e-6},
  {"name": "pole figure import",
   "entry": {"matlab": ["@PoleFigure.load", "loadPoleFigure_dubna"],
             "python": ["mtex.io.loadpolefigure.loadPoleFigure", "mtex.io.loadpolefigure.loadPoleFigure_dubna"]},
   "cases": {"load_dubna": "PoleFigure.load of the seven Dubna quartz pole figures with their superposition"},
   "tolerance": 1e-12},
]
COLS = {"load_ctf": 7, "load_ang": 7, "load_osc": 7, "load_1C_1": 7, "load_dubna": 4}


# the seven files in the order of their names, as MATLAB's dir lists them
def dubnaFiles():
  import glob
  return sorted(glob.glob(os.path.join(DATA, "PoleFigure", "dubna", "*.cnv")))


def run(ops):
  from mtex import crystalFrame, loadEBSD, loadPoleFigure
  rows = lambda e: np.column_stack([np.asarray(e.pos.data, float).reshape(-1, 3)[:, :2], np.asarray(e.phaseId, float).reshape(-1),
                                    np.asarray(e.rotations.data, float).reshape(-1, 4)])
  cs = crystalFrame("321", [4.916, 4.916, 5.4054], mineral="Quartz")

  def pf(p):
    return np.concatenate([np.column_stack([np.asarray(r.data, float).reshape(-1, 3), np.asarray(i, float).reshape(-1)])
                           for r, i in zip(p.allR, p.allI)])

  table = {op: (lambda f=f: loadEBSD(f, noGrid=True), rows) for op, f in FILES.items()}
  table["load_dubna"] = (lambda: loadPoleFigure(dubnaFiles(), cs, superposition=[1, 1, [0.52, 1.23], 1, 1, 1, 1]), pf)
  crosscheck.run(OUT, ALGORITHM, table, ops, reps={"load_1C_1": 1})


# maps by position, then the phases exactly and the rotations by their angle on the indexed rows;
# MATLAB may put a hexagonal grid in and pad its short rows, which are notIndexed rows no file
# holds. Pole figures as sets of rows: the ports list the directions in their own order
def deviation(op, a, b):
  from scipy.spatial import cKDTree
  if op == "load_dubna":
    if a.shape != b.shape:
      return float("nan")
    key = lambda x: x[np.lexsort(np.round(x, 9).T[::-1])]
    a, b = key(a), key(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
  d, j = cKDTree(b[:, :2]).query(a[:, :2])
  if d.max() > 1e-6 or len(set(j)) != len(j):
    return float("nan")
  extra = np.setdiff1d(np.arange(len(b)), j)
  if len(extra):
    if np.any(b[extra, 2] != 0):
      return float("nan")
    print(f"  {op}: MATLAB pads the grid with {len(extra)} notIndexed rows")
  b = b[j]
  if (a[:, 2] != b[:, 2]).any():
    print(f"  {op}: the phase differs in {int((a[:, 2] != b[:, 2]).sum())} rows")
    return float("nan")
  ok = (a[:, 2] > 0) & np.all(np.isfinite(a[:, 3:]), axis=1) & np.all(np.isfinite(b[:, 3:]), axis=1)
  c = np.abs(np.sum(a[ok, 3:] * b[ok, 3:], axis=1))
  return float(np.max(2 * np.arccos(np.clip(c, -1, 1))))


if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "compare", sys.argv[2:] or list(COLS)
  {"run": lambda: run(names),
   "matlab": lambda: [crosscheck.matlab("bench_io_ops", repr(OUT), crosscheck.cellOf([op]), "1e8", timeout=290) for op in names],
   "compare": lambda: crosscheck.compare(OUT, ALGORITHM, lambda op: COLS[op], deviation)}[cmd]()
