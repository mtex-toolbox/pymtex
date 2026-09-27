# the inverse pole figure colours against MATLAB: ipfHSVKey of 1e5 orientations, through the
# key's grid as a map plot takes them and exactly, in m-3m and 6/mmm
#
#   .venv/bin/python bench/plotting_ops.py export              the input
#   .venv/bin/python bench/plotting_ops.py run [op ...]        this port
#   .venv/bin/python bench/plotting_ops.py matlab [op ...]     bench/matlab/bench_plotting_ops.m through the bridge
#   .venv/bin/python bench/plotting_ops.py julia [op ...]      bench/plotting_ops.jl, the Julia port on the same input
#   .venv/bin/python bench/plotting_ops.py compare
#
# The colours are rows (r, g, b); MATLAB's orientation2color decides itself whether it goes
# through its grid, so both of this port's variants are compared with the one MATLAB result.
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools", "graph"))
import crosscheck  # noqa: E402

OUT = os.path.join(HERE, "out", "plotting")
SYM = {"m3m": "m-3m", "6mmm": "6/mmm"}
ALGORITHM = [
  {"name": "map plotting",
   "entry": {"matlab": ["@ipfColorKey.orientation2color"],
             "python": ["mtex.plotting.orientationkeys.ipfColorKey.orientation2color"],
             "julia": ["IPFColorKey", "color", "keyColors", "colorGrid"]},
   "cases": {"ipf_m3m": "ipfHSVKey(cs).orientation2color(ori), 1e5 orientations, m-3m, through the grid",
             "ipf_exact_m3m": "the same colours exactly, direction by direction",
             "ipf_6mmm": "ipfHSVKey(cs).orientation2color(ori), 1e5 orientations, 6/mmm, through the grid",
             "ipf_exact_6mmm": "the same colours exactly"},
   # the largest difference of a colour channel: the key's azimuth is MATLAB's within two degrees,
   # 5 of 255 (docs/plans/plotting.md S3), only the hue differs; the grid adds its interpolation
   "tolerance": {"ipf_m3m": 5e-2, "ipf_exact_m3m": 0.0196, "ipf_6mmm": 5e-2, "ipf_exact_6mmm": 0.0196}},
]
N = 10**5
COLS = {op: 3 for op in ALGORITHM[0]["cases"]}


def export():
  os.makedirs(OUT, exist_ok=True)
  x = np.random.default_rng(0).standard_normal((N, 4))
  (x / np.linalg.norm(x, axis=1, keepdims=True)).astype("<f8").tofile(os.path.join(OUT, "O.bin"))


def run(ops):
  from mtex import crystalFrame, ipfHSVKey, orientation, rotation
  O = rotation(np.fromfile(os.path.join(OUT, "O.bin"), "<f8").reshape(-1, 4))
  table = {}
  for s, sym in SYM.items():
    ori = orientation(O, crystalFrame(sym))

    def exact(ori=ori):
      key = ipfHSVKey(ori.CS)
      return key.dirMap.direction2color(key.crystalDirection(ori))

    table |= {f"ipf_{s}": (lambda ori=ori: ipfHSVKey(ori.CS).orientation2color(ori), lambda c: np.asarray(c, float).reshape(-1, 3)),
              f"ipf_exact_{s}": (exact, lambda c: np.asarray(c, float).reshape(-1, 3))}
  crosscheck.run(OUT, ALGORITHM, table, ops, reps=3)


def deviation(op, a, b):
  return float(np.max(np.abs(a - b))) if a.shape == b.shape else float("nan")


if __name__ == "__main__":
  cmd, names = sys.argv[1] if len(sys.argv) > 1 else "compare", sys.argv[2:] or list(COLS)
  {"export": export, "run": lambda: run(names),
   "matlab": lambda: crosscheck.matlab("bench_plotting_ops", repr(OUT), crosscheck.cellOf(names), str(N)),
   "julia": lambda: crosscheck.julia("plotting_ops.jl", ALGORITHM, names),
   "compare": lambda: crosscheck.compare(OUT, ALGORITHM, lambda op: COLS[op], deviation)}[cmd]()
