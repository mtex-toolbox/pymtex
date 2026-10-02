# the import against MATLAB: the EBSD maps of the vendor formats and the pole figures, read by
# each port from the same files
#
#   .venv/bin/python bench/io_ops.py run [op ...]        this port
#   .venv/bin/python bench/io_ops.py matlab [op ...]     bench/matlab/bench_io_ops.m through the bridge
#   .venv/bin/python bench/io_ops.py julia [op ...]      bench/io_ops.jl, the Julia port on the same files
#   .venv/bin/python bench/io_ops.py compare
#
# A map is written as rows (x, y, phase, a, b, c, d), the phase 0 for notIndexed and the rows
# matched by position; the pole figures as rows (x, y, z, intensity) in the order of the file.
# Both ports read the list the file holds, without putting it on its grid. The export writes the
# copper29 map, its orientations turned, into a copy of its file; its rows are those of the map
# read back from what each port wrote, so the comparison checks what was written.
#
# The EDAX file of two maps is an APEX project, read by the extension .edaxh5 its config claims;
# ~/mtex/data/h5_edax_2datasets.edaxh5 is a link to the .h5 the file came as. copper29.h5oina is
# a large file of the data repository, fetched by mtexdatafile('copper29') into ~/.mtex/data.
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
         "load_osc": os.path.join(DATA, "EBSD", "olivineopticalmap.ang"), "load_1C_1": "/home/hielscher/mtex/data/1C_1.ctf",
         "load_ang_SAM": "/home/hielscher/mtex/data/SAM-1-X-clean.ang", "load_ang_PAGR": "/home/hielscher/mtex/data/PAGR.ang"}
COPPER29 = os.path.join(os.path.expanduser("~"), ".mtex", "data", "EBSD", "copper29.h5oina")
EDAX = "/home/hielscher/mtex/data/h5_edax_2datasets.edaxh5"
ALGORITHM = [
  {"name": "EBSD import",
   "entry": {"matlab": ["@EBSD.load", "loadEBSD_ang", "loadEBSD_ctf", "loadEBSD_h5"],
             "python": ["mtex.io.load.loadEBSD", "mtex.io.ang._loadEBSD_ang", "mtex.io.ctf._loadEBSD_ctf", "mtex.io.h5._loadEBSD_h5"],
             "julia": ["loadEBSD", "loadAng", "loadCtf", "loadH5"]},
   "cases": {"load_ctf": "EBSD.load of Forsterite.ctf, every measurement",
             "load_ang": "EBSD.load of ferrite.ang, hexagonal grid",
             "load_osc": "EBSD.load of olivineopticalmap.ang, an optical map",
             "load_1C_1": "EBSD.load of the 2.5 million rows of 1C_1.ctf, one run",
             "load_ang_SAM": "EBSD.load of the 1.5 million rows of SAM-1-X-clean.ang, four phases",
             "load_ang_PAGR": "EBSD.load of the 7.2 million rows of PAGR.ang, one run",
             "load_h5oina_copper29": "EBSD.load of the Oxford h5oina map copper29, its first data set",
             "load_h5_edax": "EBSD.load of the first of the two maps of an EDAX APEX file"},
   # the rotation angle between the ports in radians; the files store single precision angles
   "tolerance": 1e-6},
  {"name": "EBSD export",
   "entry": {"matlab": ["@EBSD.export", "exportEBSD_h5"],
             "python": ["mtex.io.exportebsd.export", "mtex.io.exportebsd._exportEBSD_h5"],
             "julia": ["saveH5", "h5WriteRows!"]},
   "cases": {"export_h5": "export of the copper29 map, turned by 10 degrees about (1,1,1), into a copy of its h5oina file"},
   # the rotation angle between the maps read back, in radians; the file stores the Euler angles
   # in single precision, which each port rounds from its own double precision turn
   "tolerance": 1e-6},
  {"name": "pole figure import",
   "entry": {"matlab": ["@PoleFigure.load", "loadPoleFigure_dubna"],
             "python": ["mtex.io.loadpolefigure.loadPoleFigure", "mtex.io.loadpolefigure.loadPoleFigure_dubna"],
             "julia": ["loadPoleFigure", "readDubna"]},
   "cases": {"load_dubna": "PoleFigure.load of the seven Dubna quartz pole figures with their superposition"},
   "tolerance": 1e-12},
]
COLS = {"load_ctf": 7, "load_ang": 7, "load_osc": 7, "load_1C_1": 7, "load_ang_SAM": 7, "load_ang_PAGR": 7,
        "load_h5oina_copper29": 7, "load_h5_edax": 7, "export_h5": 7, "load_dubna": 4}


# the seven files in the order of their names, as MATLAB's dir lists them
def dubnaFiles():
  import glob
  return sorted(glob.glob(os.path.join(DATA, "PoleFigure", "dubna", "*.cnv")))


def run(ops):
  import functools

  from mtex import crystalFrame, degree, export, loadEBSD, loadPoleFigure, rotate, rotation, vector3d
  rows = lambda e: np.column_stack([np.asarray(e.pos.data, float).reshape(-1, 3)[:, :2], np.asarray(e.phaseId, float).reshape(-1),
                                    np.asarray(e.rotations.data, float).reshape(-1, 4)])
  cs = crystalFrame("321", [4.916, 4.916, 5.4054], mineral="Quartz")

  def pf(p):
    return np.concatenate([np.column_stack([np.asarray(r.data, float).reshape(-1, 3), np.asarray(i, float).reshape(-1)])
                           for r, i in zip(p.allR, p.allI)])

  # the map the export writes, read and turned once
  @functools.cache
  def turned():
    return rotate(loadEBSD(COPPER29, grid=False), rotation.byAxisAngle(vector3d(1, 1, 1), 10 * degree), keepXY=True)

  # the export into this port's file, which the rows read back
  def exported():
    fname = os.path.join(OUT, "python_export_h5.h5oina")
    export(turned(), fname)
    return fname

  table = {op: (lambda f=f: loadEBSD(f, grid=False), rows) for op, f in FILES.items()}
  table["load_h5oina_copper29"] = (lambda: loadEBSD(COPPER29, grid=False), rows)
  table["load_h5_edax"] = (lambda: loadEBSD(EDAX, grid=False, dataSet=0, verbose=False), rows)
  table["export_h5"] = (exported, lambda f: rows(loadEBSD(f, grid=False)))
  table["load_dubna"] = (lambda: loadPoleFigure(dubnaFiles(), cs, superposition=[1, 1, [0.52, 1.23], 1, 1, 1, 1]), pf)
  crosscheck.run(OUT, ALGORITHM, table, ops, reps={"load_1C_1": 1, "load_ang_PAGR": 1})


# maps by position, then the phases exactly and the rotations by their angle on the indexed rows;
# MATLAB may put a hexagonal grid in and pad its short rows, which are notIndexed rows no file
# holds. A file that holds a position twice (SAM-1-X-clean.ang) is matched row by row, where both
# lists keep the order of the file. Pole figures as sets of rows: the ports list the directions in
# their own order
def deviation(op, a, b):
  from scipy.spatial import cKDTree
  if op == "load_dubna":
    if a.shape != b.shape:
      return float("nan")
    key = lambda x: x[np.lexsort(np.round(x, 9).T[::-1])]
    a, b = key(a), key(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
  if a.shape == b.shape and np.max(np.abs(a[:, :2] - b[:, :2])) <= 1e-6:
    j = np.arange(len(a))
  else:
    d, j = cKDTree(b[:, :2]).query(a[:, :2])
    if d.max() > 1e-6 or len(set(j)) != len(j):
      return float("nan")
  extra = np.setdiff1d(np.arange(len(b)), j)
  if len(extra):
    if np.any(b[extra, 2] != 0):
      return float("nan")
    print(f"  {op}: MATLAB pads the grid with {len(extra)} notIndexed rows")
  b = b[j]
  ok = (a[:, 2] > 0) & np.all(np.isfinite(a[:, 3:]), axis=1) & np.all(np.isfinite(b[:, 3:]), axis=1)
  c = np.abs(np.sum(a[ok, 3:] * b[ok, 3:], axis=1))
  angle = float(np.max(2 * np.arccos(np.clip(c, -1, 1)))) if ok.any() else 0.0
  # a phase that differs fails the case; the rows where it does are told apart from the rest
  differ = a[:, 2] != b[:, 2]
  if differ.any():
    lost = int((differ & (a[:, 2] == 0)).sum())
    print(f"  {op}: the phase differs in {int(differ.sum())} rows, {lost} of them notIndexed here and indexed in "
          f"MATLAB; the rotations of the other rows deviate by {angle:.1e}")
    return float("nan")
  return angle

if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "compare", sys.argv[2:] or list(COLS)
  {"run": lambda: run(names),
   "matlab": lambda: [crosscheck.matlab("bench_io_ops", repr(OUT), crosscheck.cellOf([op]), "1e8", timeout=290) for op in names],
   "julia": lambda: crosscheck.julia("io_ops.jl", ALGORITHM, names),
   "compare": lambda: crosscheck.compare(OUT, ALGORITHM, lambda op: COLS[op], deviation)}[cmd]()
