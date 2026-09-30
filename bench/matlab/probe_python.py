import json, sys, time, warnings
import numpy as np
import mtex
from mtex import degree
warnings.simplefilter("ignore")
fname, out = sys.argv[1], sys.argv[2]
t = time.perf_counter()
e = mtex.loadEBSD(fname, grid=False)
s = {"time": time.perf_counter() - t, "n": len(e), "phaseMap": e.phaseMap.tolist(), "counts": [int((e.phaseId == k).sum()) for k in range(len(e.CSList))],
     "minerals": e.CSList.mineralList, "pointGroups": [getattr(c, "pointGroup", "") for c in e.CSList],
     "abc": [np.concatenate([c.abc, c.abg / degree]).tolist() if hasattr(c, "abc") else [] for c in e.CSList]}
ind = e.isIndexed
s["nIndexed"] = int(ind.sum())
ks = np.flatnonzero(ind)
if ks.size:
  k = int(ks[0]); s["firstIndexed"] = k + 1
  s["firstEuler"] = (np.array(e.rotations[k].Euler()) / degree).tolist(); s["firstPos"] = e.pos.data[k, :2].tolist()
  M = np.array(e.rotations[ind].matrix()).reshape(-1, 3, 3); s["meanMatrix"] = M.mean(axis=0).tolist()
s["extent"] = [float(e.pos.x.min()), float(e.pos.x.max()), float(e.pos.y.min()), float(e.pos.y.max())]
s["props"] = {k: float(np.nanmean(np.asarray(v, dtype=float))) for k, v in e.prop.items()}
json.dump(s, open(out, "w"))
