# a profile of one call in the record format build.py maps onto the graph; profileDump.m writes
# the same records from MATLAB's profiler
#
#   {"algorithm", "port", "case", "wall", "functions": [{"file", "line", "name", "self", "total",
#    "calls", "children": [[index, total], ...]}]}
from __future__ import annotations

import cProfile
import json
import os
import pstats
import time


def profileCall(f):
  pr = cProfile.Profile()
  pr.enable()
  t = time.perf_counter()
  result = f()
  wall = time.perf_counter() - t
  pr.disable()
  st = pstats.Stats(pr).stats
  keys = list(st)
  index = {k: i for i, k in enumerate(keys)}
  recs = [dict(file=os.path.abspath(k[0]) if not k[0].startswith("~") else k[0], line=k[1], name=k[2],
               self=v[2], total=v[3], calls=v[1], children=[]) for k, v in st.items()]
  for k, v in st.items():
    for caller, cv in v[4].items():
      if caller in index:
        recs[index[caller]]["children"].append([index[k], cv[3]])
  return result, wall, recs


def write(path, algorithm, port, case, wall, recs):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  json.dump(dict(algorithm=algorithm, port=port, case=case, wall=wall, functions=recs), open(path, "w"))
