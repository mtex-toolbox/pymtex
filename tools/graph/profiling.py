# a profile of one call in the record format build.py maps onto the graph; profileDump.m writes
# the same records from MATLAB's profiler
#
#   {"algorithm", "port", "case", "wall", "cores", "threads", "functions": [{"file", "line", "name", "self",
#    "total", "calls", "children": [[index, total], ...]}]}
#
# wall is the best timed run; cores the CPU time of the process over all timed runs, every thread's
# counted, over their wall time, so that a clock coarser than one run still reads; threads the
# threads the port was given
from __future__ import annotations

import cProfile
import json
import os
import pstats
import time


# one call: its result, its wall time and the CPU time the process spent over it
def timed(f):
  c, t = time.process_time(), time.perf_counter()
  out = f()
  return out, time.perf_counter() - t, time.process_time() - c


# the threads this port runs its parallel kernels, finufft and scipy.fft with
def threads():
  from mtex.geometry.kernels import threads
  return threads()


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


# the best of the timed runs (result, wall, cpu) and the cores they kept busy together
def best(runs):
  runs = list(runs)
  wall, cpu = sum(r[1] for r in runs), sum(r[2] for r in runs)
  return min(runs, key=lambda r: r[1]), cpu / wall if wall > 0 else None


def write(path, algorithm, port, case, wall, recs, cores=None, threads=None):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  json.dump(dict(algorithm=algorithm, port=port, case=case, wall=wall, cores=cores, threads=threads, functions=recs),
            open(path, "w"))
