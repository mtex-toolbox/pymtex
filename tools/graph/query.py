# questions to the graph of build.py
#
#   query.py find <text>                      nodes whose id contains the text
#   query.py node <node>                      where it is, counterparts, callers, callees, tests, notes, algorithms
#   query.py callers <node> [-d 3]            who reaches it, by depth; --dispatch also through ambiguous receivers
#   query.py callees <node> [-d 2]
#   query.py impact <node|file>               transitive callers in both ports, the tests and algorithms that reach it
#   query.py algo [<name>]                    an algorithm's entry points and where its time goes, port by port
#   query.py gaps <prefix>                    MATLAB functions under a folder without a Python counterpart
#   query.py layer <layer>                    a layer in both ports: counterparts, tests, notes, algorithms, per class
#
# A node is its id (m:@rotation.calcDensity, py:mtex.functions.density.calcDensity), a unique
# name, "@Class.method", "Class.method" or a source file path. A graph older than any of its
# sources is rebuilt first, about five seconds.
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = ("test", "bench")


class G:
  def __init__(self, path):
    g = json.load(open(path))
    self.g = g
    self.nodes = {n["id"]: n for n in g["nodes"]}
    self.out, self.inn = defaultdict(list), defaultdict(list)
    for e in g["edges"]:
      self.out[e["src"]].append(e)
      self.inn[e["dst"]].append(e)

  def resolve(self, q):
    if q in self.nodes:
      return [q]
    for p in ("m:", "py:", "m:@", "algo:"):
      if p + q in self.nodes:
        return [p + q]
    m = re.fullmatch(r"@?(\w+)[./](\w+)", q)
    if m:
      hits = [i for i, n in self.nodes.items() if n.get("cls") == m.group(1) and n.get("name") == m.group(2)]
      if hits:
        return hits
    hits = [i for i, n in self.nodes.items() if n.get("file") and (n["file"] == q or n["file"].endswith("/" + q))
            and n["kind"] not in ("local", "dispatch")]
    if hits:
      return hits
    return [i for i, n in self.nodes.items() if n.get("name") == q and n["kind"] in ("function", "method", "class")]

  def where(self, i):
    n = self.nodes[i]
    return f"{n.get('file', '')}:{n.get('line', '')}" if n.get("file") else ""

  # nodes reaching (or reached from) a node, by depth; dispatch nodes stand for any receiver
  def walk(self, start, depth, up=True, dispatch=False, kinds=("call", "dispatch")):
    seen, frontier, layers = set(start), list(start), []
    for _ in range(depth):
      nxt = []
      for i in frontier:
        for e in (self.inn[i] if up else self.out[i]):
          if e["kind"] not in kinds:
            continue
          j = e["src"] if up else e["dst"]
          if not dispatch and (e["resolution"] == "dispatch" and self.nodes.get(j, {}).get("kind") != "dispatch"):
            continue
          if not dispatch and self.nodes.get(j, {}).get("kind") == "dispatch" and up:
            continue
          if j not in seen:
            seen.add(j)
            nxt.append(j)
      if not nxt:
        break
      layers.append(nxt)
      frontier = nxt
    return layers

  def counterparts(self, i):
    return [(e["dst"], e["resolution"]) for e in self.out[i] if e["kind"] == "counterpart"] + \
           [(e["src"], e["resolution"]) for e in self.inn[i] if e["kind"] == "counterpart"]

  def tests(self, ids, depth=12, dispatch=False):
    found = set()
    for layer in self.walk(ids, depth, up=True, dispatch=dispatch):
      found |= {j for j in layer if self.nodes.get(j, {}).get("kind") in TESTS or
                "/tests/" in "/" + self.nodes.get(j, {}).get("file", "") and self.nodes[j]["kind"] == "script"}
    for i in ids:
      for t, cov in self.g.get("coverage", {}).get(i, {}).items():
        found.add(t)
    return sorted(found)


def show(G_, ids, limit=12):
  for i in ids[:limit]:
    n = G_.nodes.get(i, {})
    print(f"    {i}  {G_.where(i) if n else ''}")
  if len(ids) > limit:
    print(f"    ... {len(ids) - limit} more")


def cmdNode(G_, q):
  for i in G_.resolve(q):
    n = G_.nodes[i]
    print(f"{i}  [{n['kind']}]  {G_.where(i)}  {n.get('lines', '')} lines")
    if n.get("matlab"):
      print(f"  cites MATLAB: {', '.join(n['matlab'])}")
    cp = G_.counterparts(i)
    if cp:
      print("  counterparts:")
      for j, how in cp:
        print(f"    {j}  ({how})  {G_.where(j)}")
    callers = [e["src"] for e in G_.inn[i] if e["kind"] == "call"]
    callees = [e["dst"] for e in G_.out[i] if e["kind"] == "call"]
    print(f"  callers ({len(callers)}):")
    show(G_, sorted(callers), 8)
    print(f"  callees ({len(callees)}):")
    show(G_, sorted(callees), 16)
    t = G_.tests([i])
    print(f"  tests reaching it statically, exact receivers only ({len(t)}):")
    show(G_, t, 10)
    ns = [e["src"] for e in G_.inn[i] if e["kind"] == "note"]
    if ns:
      print("  notes:")
      for j in ns:
        print(f"    {G_.nodes[j]['file']}  {G_.nodes[j].get('title', '')}")
    al = [e["src"] for e in G_.inn[i] if e["kind"] == "entry"]
    if al:
      print(f"  entry point of: {', '.join(al)}")
    print()


def cmdWalk(G_, q, depth, up, dispatch):
  for i in G_.resolve(q):
    print(i)
    for d, layer in enumerate(G_.walk([i], depth, up, dispatch), 1):
      print(f"  depth {d} ({len(layer)}):")
      show(G_, sorted(layer), 30)


def cmdImpact(G_, q, dispatch):
  ids = G_.resolve(q)
  if not ids:
    sys.exit(f"no node {q}")
  cps = sorted({j for i in ids for j, _ in G_.counterparts(i)})
  for label, group in (("MATLAB/Python", ids), ("counterparts", cps)):
    if not group:
      continue
    reach = sorted({j for layer in G_.walk(group, 12, True, dispatch) for j in layer})
    print(f"{label}: {', '.join(group)}")
    print(f"  transitive callers: {len(reach)}")
    t = G_.tests(group, dispatch=dispatch)
    print(f"  tests and benchmarks ({len(t)}):")
    show(G_, t, 40)
    al = sorted({e["src"] for j in group + reach for e in G_.inn[j] if e["kind"] == "entry"})
    if al:
      print(f"  algorithms: {', '.join(al)}")


def fmt(t):
  return f"{t * 1e3:8.1f} ms" if t < 10 else f"{t:8.2f} s "


def cmdAlgo(G_, name):
  algos = [n for n in G_.nodes.values() if n["kind"] == "algorithm" and (not name or name in n["name"])]
  for a in algos:
    print(f"{a['name']}   benchmarks {', '.join(a['files'])}")
    for port, es in a["entry"].items():
      print(f"  {port}: {', '.join(es)}")
    if a.get("missing"):
      print(f"  MISSING entry points: {a['missing']}")
    profs = [p for p in G_.g["profiles"] if p["algorithm"] == a["name"]]
    for case in sorted({p["case"] for p in profs}):
      print(f"\n  case {case}: {a.get('cases', {}).get(case, '')}")
      byPort = {p["port"]: p for p in profs if p["case"] == case}
      for port, p in sorted(byPort.items()):
        print(f"    {port}: wall {fmt(p['wall']).strip()}")
        rows = sorted(p["functions"].items(), key=lambda kv: -kv[1]["self"])
        tot = sum(v["self"] for v in p["functions"].values()) or 1
        for fid, v in rows[:a.get("stages", 10)]:
          cp = [j for j, _ in G_.counterparts(fid)] if fid in G_.nodes else []
          print(f"      {fmt(v['self'])} self {100 * v['self'] / tot:5.1f} %  {fmt(v['total'])} total  "
                f"{v['calls']:7d} calls  {fid}" + (f"  ~ {cp[0]}" if cp else ""))
    print()


def cmdGaps(G_, prefix):
  mNodes = [n for n in G_.nodes.values() if n["port"] == "matlab" and n["kind"] in ("function", "method")
            and n.get("file", "").startswith(prefix) and not n.get("plumbing")]
  missing = [n for n in mNodes if not G_.counterparts(n["id"])]
  print(f"MATLAB under {prefix}: {len(mNodes)} functions and methods besides plumbing, "
        f"{len(mNodes) - len(missing)} with a Python counterpart")
  byDir = defaultdict(list)
  for n in missing:
    byDir[os.path.dirname(n["file"])].append(n["name"])
  for d, ns in sorted(byDir.items()):
    print(f"  {d}: {', '.join(sorted(set(ns)))}")


# the nodes some test reaches along exact edges, per port
def testReach(G_, port):
  start = [i for i, n in G_.nodes.items() if n["port"] == port and n["kind"] in TESTS]
  return {j for layer in G_.walk(start, 12, up=False) for j in layer}


def pct(a, b):
  return f"{a:5d} / {b:<5d} {100 * a / b:5.1f} %" if b else f"{a:5d} / {b:<5d}"


def cmdLayer(G_, layer):
  units = lambda port: [n for n in G_.nodes.values() if n["port"] == port and n.get("layer") == layer
                        and n["kind"] in ("function", "method") and not n.get("plumbing")
                        and not (port == "python" and n["name"].startswith("_") and not n["name"].startswith("__"))]
  hasNote = lambda i: any(e["kind"] == "note" for e in G_.inn[i])
  print(f"layer {layer}")
  print(f"{'':34s}{'MATLAB':>22s}{'Python':>24s}")
  m, p = units("matlab"), units("python")
  rm, rp = testReach(G_, "matlab"), testReach(G_, "python")
  rows = [("functions and methods", len(m), len(m), len(p), len(p)),
          ("with a counterpart", sum(bool(G_.counterparts(n["id"])) for n in m), len(m),
           sum(bool(G_.counterparts(n["id"])) for n in p), len(p)),
          ("reached by a test (static)", sum(n["id"] in rm for n in m), len(m), sum(n["id"] in rp for n in p), len(p)),
          ("run by a test (measured)", sum(n["id"] in G_.g["coverage"] for n in m), len(m),
           sum(n["id"] in G_.g["coverage"] for n in p), len(p)),
          ("mentioned in a note", sum(hasNote(n["id"]) for n in m), len(m), sum(hasNote(n["id"]) for n in p), len(p))]
  for label, a, b, c, d in rows:
    print(f"  {label:32s}{pct(a, b):>22s}  {pct(c, d):>22s}")
  plumb = sum(1 for n in G_.nodes.values() if n["port"] == "matlab" and n.get("layer") == layer and n.get("plumbing"))
  print(f"  MATLAB plumbing left out: {plumb} (subsref, display, get./set., ...)")
  algos = sorted({e["src"] for n in m + p for e in G_.inn[n["id"]] if e["kind"] == "entry"})
  print(f"  algorithms with an entry point here: {', '.join(a[5:] for a in algos) or 'none'}")
  print(f"\n  {'MATLAB class':24s} {'methods':>7s} {'matched':>7s} {'tested':>7s}  Python class")
  byCls = defaultdict(list)
  for n in m:
    byCls[n.get("cls") or "(functions)"].append(n)
  pyCls = {n["name"] for n in G_.nodes.values() if n["port"] == "python" and n["kind"] == "class"}
  for c, ns in sorted(byCls.items(), key=lambda kv: -len(kv[1])):
    matched = sum(bool(G_.counterparts(n["id"])) for n in ns)
    tested = sum(n["id"] in rm for n in ns)
    print(f"  {c:24s} {len(ns):7d} {matched:7d} {tested:7d}  {'yes' if c in pyCls else '-'}")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd")
  ap.add_argument("arg", nargs="?")
  ap.add_argument("-d", "--depth", type=int, default=3)
  ap.add_argument("--dispatch", action="store_true", help="also walk through method names of untyped receivers")
  ap.add_argument("--graph", default=os.path.join(HERE, "out", "graph.json"))
  a = ap.parse_args()
  sys.path.insert(0, HERE)
  import build
  if a.graph == os.path.join(HERE, "out", "graph.json") and build.stale():
    print("the graph is older than its sources, rebuilding", file=sys.stderr)
    build.main([])
  G_ = G(a.graph)
  if a.cmd == "find":
    for i in sorted(i for i in G_.nodes if a.arg.lower() in i.lower())[:200]:
      print(i, G_.where(i))
  elif a.cmd == "node":
    cmdNode(G_, a.arg)
  elif a.cmd in ("callers", "callees"):
    cmdWalk(G_, a.arg, a.depth, a.cmd == "callers", a.dispatch)
  elif a.cmd == "impact":
    cmdImpact(G_, a.arg, a.dispatch)
  elif a.cmd == "algo":
    cmdAlgo(G_, a.arg)
  elif a.cmd == "layer":
    cmdLayer(G_, a.arg or "geometry")
  elif a.cmd == "gaps":
    cmdGaps(G_, a.arg or "")
  else:
    sys.exit(__doc__)


if __name__ == "__main__":
  main()
