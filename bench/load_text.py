# the text loaders on the MTEX sample files and the large map, best of three, warm
#
#   .venv/bin/python bench/load_text.py
import os
import time

import mtex

ALGORITHM = {
  "name": "EBSD import",
  "entry": {"matlab": ["@EBSD.load", "loadEBSD_ang", "loadEBSD_ctf"],
            "python": ["mtex.io.load.loadEBSD", "mtex.io.ang.loadEBSD_ang", "mtex.io.ctf.loadEBSD_ctf"]},
  "cases": {"load_text": "loadEBSD of Forsterite.ctf, ferrite.ang, olivineopticalmap.ang and the 2.5 million rows of 1C_1.ctf, as a list and on the grid"},
  "tolerance": None}

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "frameSymmetry", "data", "EBSD")
FILES = [os.path.join(DATA, f) for f in ("Forsterite.ctf", "ferrite.ang", "olivineopticalmap.ang")]
FILES += [f for f in ("/home/hielscher/mtex/data/1C_1.ctf",) if os.path.exists(f)]

mtex.loadEBSD(FILES[0])
for f in FILES:
  best = [9e9, 9e9]
  for _ in range(3):
    t = time.perf_counter()
    e = mtex.loadEBSD(f, grid=False)
    best[0] = min(best[0], time.perf_counter() - t)
    t = time.perf_counter()
    mtex.loadEBSD(f)
    best[1] = min(best[1], time.perf_counter() - t)
  print("%-24s %9d rows  list %.3f s  grid %.3f s" % (os.path.basename(f), len(e), best[0], best[1]))
