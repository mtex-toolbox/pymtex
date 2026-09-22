import json, sys, numpy as np
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))   # matlab, python
name = sys.argv[3]
diffs = []
if a["n"] != b["n"]: diffs.append(f"n {a['n']} vs {b['n']}")
if a["counts"] != b["counts"]: diffs.append(f"counts {a['counts']} vs {b['counts']}")
if [m.lower() for m in a["minerals"]] != [m.lower() for m in b["minerals"]]: diffs.append(f"minerals {a['minerals']} vs {b['minerals']}")
if a["pointGroups"][1:] != b["pointGroups"][1:]: diffs.append(f"pointGroups {a['pointGroups']} vs {b['pointGroups']}")
for i, (x, y) in enumerate(zip(a["abc"], b["abc"])):
  if x and y and not np.allclose(x, y, rtol=1e-4): diffs.append(f"cell of phase {i}: {x} vs {y}")
if a.get("firstIndexed") != b.get("firstIndexed"): diffs.append(f"firstIndexed {a.get('firstIndexed')} vs {b.get('firstIndexed')}")
if "firstEuler" in a and "firstEuler" in b:
  da = np.abs((np.array(a["firstEuler"]) - np.array(b["firstEuler"]) + 180) % 360 - 180).max()
  if da > 1e-4: diffs.append(f"firstEuler {a['firstEuler']} vs {b['firstEuler']}")
  dm = np.abs(np.array(a["meanMatrix"]) - np.array(b["meanMatrix"])).max()
  if dm > 1e-4: diffs.append(f"meanMatrix differs by {dm:.2e}")
if not np.allclose(a["extent"], b["extent"], atol=1e-3): diffs.append(f"extent {a['extent']} vs {b['extent']}")
for k, v in a["props"].items():
  if k in b["props"]:
    if not np.isclose(v, b["props"][k], rtol=1e-6, atol=1e-9): diffs.append(f"prop {k} mean {v} vs {b['props'][k]}")
  else: diffs.append(f"prop {k} only in matlab")
for k in b["props"]:
  if k not in a["props"]: diffs.append(f"prop {k} only in python")
print("%-40s matlab %6.2f s  python %6.2f s  %s" % (name, a["time"], b["time"], "OK" if not diffs else "DIFF"))
for d in diffs: print("    ", d)
