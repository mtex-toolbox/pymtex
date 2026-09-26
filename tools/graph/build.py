# the dependency graph of MTEX and its Python port, rebuilt from the sources, never edited
#
#   .venv/bin/python tools/graph/build.py [--matlab ../master] [--memory DIR ...]    writes tools/graph/out/graph.json
#
# Layers, each regenerated: the static call graphs of both ports (scan_matlab.py, scan_python.py),
# the counterpart edges between them (same class and method names, the MATLAB names a Python
# header or docstring cites, the names the imported port answers to), the notes that mention a node (agent memory, TODO.md, ADRs,
# docs/research, the research notes of MATLAB master's tests and of the Julia port), the research
# questions those notes ask (every subsection, or else every bullet, under a heading naming questions
# or a research agenda), the algorithms the benchmarks declare (bench/*.py, ALGORITHM) with the
# profiles their runs leave in bench/out/<algorithm>/, and the coverage the test runs leave in
# tools/graph/out/coverage_<port>.json.
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import scan_matlab  # noqa: E402
import scan_python  # noqa: E402

OUT = os.path.join(HERE, "out")
JULIA = os.path.join(os.path.dirname(ROOT), "mtexJulia")
MEMORY = [os.path.expanduser(f"~/.claude/projects/-home-hielscher-mtex-{p}/memory") for p in ("master", "pymtex")]
# names too common to link a note by: they would attach every note to every class
GENERIC = set("""eval plot mean size length find angle dot norm sum max min abs sqrt display char
inv times mtimes plus minus rotate project symmetrise conj real imag disp subsref subsasgn
cat horzcat vertcat isempty numel reshape unique round get set any all interp""".split())


class Index:
  # lookups over the nodes of both ports
  def __init__(self, nodes):
    self.byId = {n["id"]: n for n in nodes}
    self.byName = {}
    self.byFile = {}
    self.mClass, self.pClass = {}, {}
    for n in nodes:
      self.byName.setdefault((n["port"], n.get("name")), []).append(n["id"])
      if n.get("file") and n["kind"] not in ("local", "dispatch"):
        self.byFile.setdefault(n["file"], []).append(n["id"])
      if n["kind"] == "class":
        (self.mClass if n["port"] == "matlab" else self.pClass)[n["name"]] = n
    self.mMethods = {}
    for n in nodes:
      if n["port"] == "matlab" and n["kind"] == "method":
        self.mMethods.setdefault(n["cls"], {})[n["name"]] = n["id"]

  # the MATLAB method a class reaches under a name, walking its superclasses
  def mLookup(self, cls, name, seen=None):
    seen = seen or set()
    if cls in seen or cls not in self.mMethods and cls not in self.mClass:
      return None
    seen.add(cls)
    if name in self.mMethods.get(cls, {}):
      return self.mMethods[cls][name]
    for s in self.mClass.get(cls, {}).get("supers", []):
      r = self.mLookup(s, name, seen)
      if r:
        return r
    return None

  # MATLAB functions and methods of a name; a module function of Python stands for the methods of
  # every class the port also has, since it dispatches on its argument where MATLAB dispatches on the class
  def matlabNamed(self, name, limit=8, pyClasses=None):
    ids = [i for i in self.byName.get(("matlab", name), []) if self.byId[i]["kind"] in ("function", "method", "class")]
    if pyClasses is not None:
      ids = [i for i in ids if self.byId[i]["kind"] != "method" or self.byId[i]["cls"] in pyClasses]
      return ids if len(ids) <= 4 * limit else []
    return ids if len(ids) <= limit else []

  # the MATLAB nodes a cited name stands for: @Class, @Class/method, Class.method, a path, a function
  def resolveMatlab(self, ref):
    m = re.fullmatch(r"@?(\w+)[/.](\w+)", ref)
    if m and (m.group(1) in self.mClass or m.group(1) in self.mMethods):
      hit = self.mLookup(m.group(1), m.group(2))
      return [hit] if hit else []
    if ref.startswith("@") and ref[1:] in self.mClass:
      return [self.mClass[ref[1:]]["id"]]
    if "/" in ref:
      hits = [ids[0] for f, ids in self.byFile.items() if f.startswith("") and self.byId[ids[0]]["port"] == "matlab"
              and (f[:-2].endswith("/" + ref) or os.path.dirname(f).endswith("/" + ref))]
      return hits[:12]
    if ref in self.mClass:
      return [self.mClass[ref]["id"]]
    return self.matlabNamed(ref)


# Python spellings of MATLAB's operator and indexing methods
DUNDER = {"__getitem__": ["subsref", "subSet"], "__setitem__": ["subsasgn", "setSubSet"], "__repr__": ["display", "char"],
          "__str__": ["char", "display"], "__eq__": ["eq"], "__ne__": ["ne"], "__add__": ["plus"], "__sub__": ["minus"],
          "__neg__": ["uminus"], "__mul__": ["times", "mtimes"], "__rmul__": ["times", "mtimes"], "__matmul__": ["mtimes"],
          "__truediv__": ["rdivide", "mrdivide"], "__rtruediv__": ["ldivide", "mldivide"], "__pow__": ["power", "mpower"],
          "__len__": ["length", "numel"], "__abs__": ["abs"], "__call__": ["eval"], "__lt__": ["lt"], "__le__": ["le"],
          "__gt__": ["gt"], "__ge__": ["ge"], "__invert__": ["inv"], "cat": ["cat", "horzcat", "vertcat"], "T": ["transpose", "ctranspose"]}
REVERSE_DUNDER = {}
for _d, _ms in DUNDER.items():
  for _m in _ms:
    REVERSE_DUNDER.setdefault(_m, []).append(_d)
# MATLAB methods that exist for the language, not the mathematics: no counterpart is expected
PLUMBING = set("""subsref subsasgn subSet setSubSet numArgumentsFromSubscript end numel length size ndims isempty
isscalar loadobj saveobj copy delete display disp char dispLine dispData horzcat vertcat cat repmat reshape
transpose ctranspose permute squeeze ne isequal fullDouble""".split())
# method names every function-like class has, no evidence two classes are one
ARITHMETIC = set("""eval mtimes times plus minus uminus rdivide mrdivide power mpower inv mean plot rotate rotate_outer
symmetrise norm dot conj real abs sum max min eq ne cat""".split())
# the layer of a source file, one vocabulary for both ports
MLAYER = {"geometry": "geometry", "SO3Fun": "functions", "S2Fun": "functions", "S1Fun": "functions",
          "EBSDAnalysis": "maps", "PoleFigureAnalysis": "analysis", "ODFAnalysis": "analysis", "TensorAnalysis": "tensors",
          "interfaces": "io", "plotting": "plotting", "tools": "tools", "tests": "tests", "doc": "doc"}


def layerOf(n):
  f = n.get("file") or ""
  if n["port"] == "matlab":
    return MLAYER.get(f.split("/")[0], "other")
  if n["port"] == "python":
    parts = f.split("/")
    return parts[1] if parts[0] == "mtex" and len(parts) > 2 else parts[0] if parts[0] in ("tests", "bench") else "other"
  return None


def isPlumbing(n):
  return n["port"] == "matlab" and n["kind"] == "method" and (n["name"] in PLUMBING or n["name"].startswith(("get.", "set.")))


# the Python names a method answers to in MATLAB's vocabulary
def matlabNames(n):
  name, roles = n["name"], n.get("roles", [])
  out = DUNDER.get(name, []) + ([name] if not name.startswith("_") else [])
  out += ["get." + name] if "getter" in roles or name == "T" else []
  return out + (["set." + name] if "setter" in roles else [])


# a MATLAB class without a Python class of its name is spelled by the Python class sharing
# most of its method names (Miller by vector3d, crystalSymmetry by crystalFrame), else by its
# nearest ancestor that Python has
def classAliases(ix, pyNodes):
  pyNames = {}
  for n in pyNodes:
    if n["kind"] == "method":
      pyNames.setdefault(n["cls"], set()).update(x.split(".")[-1] for x in matlabNames(n))
  aliases = {}
  for c, methods in ix.mMethods.items():
    if c in ix.pClass:
      continue
    names = {m.split(".")[-1] for m in methods if m not in PLUMBING}
    best = max(pyNames.items(), key=lambda kv: len(names & kv[1] - ARITHMETIC), default=(None, set()))
    shared = len(names & best[1] - ARITHMETIC)
    if shared >= 3 and shared >= 0.25 * len(names):
      aliases.setdefault(best[0], []).append(c)
      continue
    # else the nearest MATLAB ancestor Python has: EBSDsquare is an EBSD whose grid pymtex keeps inside
    frontier = list(ix.mClass.get(c, {}).get("supers", []))
    while frontier:
      a = frontier.pop(0)
      if a in ix.pClass:
        aliases.setdefault(a, []).append(c)
        break
      frontier += ix.mClass.get(a, {}).get("supers", [])
  return aliases


# what the imported port answers to under MATLAB's names: methods a table or __getattr__ makes,
# inherited ones, module attributes; each found name links to the Python node defining it
def runtimeCounterparts(ix, pyNodes, aliases):
  import importlib
  import inspect
  sys.path.insert(0, ROOT)
  try:
    import mtex
  except Exception as err:
    print(f"runtime counterparts skipped: {err}")
    return []
  pyClass = {}
  for n in pyNodes:
    if n["kind"] == "class":
      pyClass.setdefault(n["name"], n["id"])
  byAlias = {c: p for p, cs in aliases.items() for c in cs}

  def nodeOf(obj, fallback):
    f = obj.fget if isinstance(obj, property) else getattr(obj, "__func__", obj)
    qual, mod = getattr(f, "__qualname__", ""), getattr(f, "__module__", "")
    nid = f"py:{mod}.{qual}"
    return nid if nid in ix.byId and "<" not in qual else fallback

  # a MATLAB class answers to its own methods and those it inherits
  def reachable(c, seen=None):
    seen = seen if seen is not None else set()
    if c in seen:
      return {}
    seen.add(c)
    out = {}
    for s in ix.mClass.get(c, {}).get("supers", []):
      out.update(reachable(s, seen))
    return out | ix.mMethods.get(c, {})

  edges = []
  for c in set(ix.mMethods) | set(ix.mClass):
    methods = reachable(c)
    pname = c if c in pyClass else byAlias.get(c)
    if not pname or pname not in pyClass:
      continue
    cid = pyClass[pname]
    modName = cid[len("py:class:"):].rsplit(".", 1)[0]
    cls = getattr(importlib.import_module(modName), pname, None)
    if not inspect.isclass(cls):
      continue
    for m, mid in methods.items():
      name = m.split(".")[-1]
      if m in PLUMBING or name in PLUMBING:
        continue
      # an _outer variant is the broadcasting call of its base, an operator its dunder
      for cand in [name, name[:-6] if name.endswith("_outer") else None] + REVERSE_DUNDER.get(name, []):
        obj = inspect.getattr_static(cls, cand, None) if cand else None
        if obj is not None:
          edges.append(dict(src=nodeOf(obj, cid), dst=mid, kind="counterpart", resolution="runtime"))
          break
  for n in ix.byId.values():
    if n["port"] == "matlab" and n["kind"] == "function" and n.get("layer") not in ("tests", "doc"):
      obj = getattr(mtex, n["name"], None)
      if obj is not None and (callable(obj) or not inspect.ismodule(obj)):
        src = nodeOf(obj, None) or (pyClass.get(n["name"]) if inspect.isclass(obj) else None)
        if src:
          edges.append(dict(src=src, dst=n["id"], kind="counterpart", resolution="runtime"))
  return edges


def counterparts(ix, pyNodes):
  edges = []
  aliases = classAliases(ix, pyNodes)
  for n in pyNodes:
    found = []
    if n["kind"] == "class" and n["name"] in ix.mClass:
      found.append((ix.mClass[n["name"]]["id"], "name"))
    elif n["kind"] == "class":
      found += [(ix.mClass[c]["id"], "alias") for c in aliases.get(n["name"], []) if c in ix.mClass]
    if n["kind"] == "method":
      for cls, how in [(n["cls"], "name")] + [(c, "alias") for c in aliases.get(n["cls"], [])]:
        for c in [cls] if n["name"] == "__init__" else matlabNames(n):
          hit = ix.mLookup(cls, c)
          if hit:
            found.append((hit, how))
    elif n["kind"] == "function" and not n["name"].startswith("_"):
      found += [(i, "name") for i in ix.matlabNamed(n["name"], pyClasses=ix.pClass)]
    for ref in n.get("matlab") or []:
      found += [(i, "cited") for i in ix.resolveMatlab(ref)]
    seen = set()
    for dst, how in found:
      if dst not in seen:
        seen.add(dst)
        edges.append(dict(src=n["id"], dst=dst, kind="counterpart", resolution=how))
  known = {(e["src"], e["dst"]) for e in edges}
  return edges + [e for e in runtimeCounterparts(ix, pyNodes, aliases) if (e["src"], e["dst"]) not in known
                  and not known.add((e["src"], e["dst"]))]


# the note files and the nodes each one mentions
def notes(ix, sources):
  nodes, edges = [], []
  for label, pattern in sources:
    for path in sorted(glob.glob(pattern)):
      text = open(path, encoding="utf-8", errors="replace").read()
      desc = re.search(r"^description:\s*(.*)$", text, re.M) or re.search(r"^#\s+(.*)$", text, re.M)
      nid = f"note:{label}/{os.path.basename(path)}"
      nodes.append(dict(id=nid, port="note", kind="note", name=os.path.basename(path), file=path,
                        title=(desc.group(1) if desc else "").strip()[:160]))
      edges += [dict(src=nid, dst=h, kind="note", resolution="mention") for h in sorted(mentions(ix, text))]
      for k, (line, title, body) in enumerate(questionBlocks(text)):
        qid = f"question:{label}/{os.path.basename(path)}#{k}"
        nodes.append(dict(id=qid, port="question", kind="question", name=title[:80], title=title[:160], file=path, line=line))
        edges.append(dict(src=qid, dst=nid, kind="asked", resolution="section"))
        edges += [dict(src=qid, dst=h, kind="question", resolution="mention") for h in sorted(mentions(ix, body))]
  return nodes, edges


# the nodes a text names: in backticks, as @Class/method, or as a source file
def mentions(ix, text):
  tick = re.compile(r"`([^`\n]{1,120})`")
  token = re.compile(r"[@\w./+-]*\w")
  cands = set()
  for span in tick.findall(text):
    cands |= set(token.findall(span))
  cands |= set(re.findall(r"@\w+/\w+", text))
  cands |= set(re.findall(r"[\w/@+-]+\.(?:m|py)\b", text))
  hits = set()
  for c in cands:
    hits |= set(mentioned(ix, c))
  return hits


QUESTION_HEADING = re.compile(r"questions|research agenda|open problems", re.I)


# the research questions of a note: under a heading that names questions, every subsection is one,
# or, where it has none, every top level list item; (line, title, text) each
def questionBlocks(text):
  lines = text.split("\n")
  head = [(i, len(m.group(1)), m.group(2).strip()) for i, l in enumerate(lines) if (m := re.match(r"(#{1,6})\s+(.*)", l))]
  out = []
  for k, (i, level, title) in enumerate(head):
    if level < 2 or not QUESTION_HEADING.search(title):
      continue
    end = next((j for j, lv, _ in head[k + 1:] if lv <= level), len(lines))
    subs = [(j, t) for j, lv, t in head[k + 1:] if j < end and lv == level + 1]
    if subs:
      for n, (j, t) in enumerate(subs):
        stop = subs[n + 1][0] if n + 1 < len(subs) else end
        out.append((j + 1, re.sub(r"^[\d.]+\s*", "", t), "\n".join(lines[j:stop])))
      continue
    items = [j for j in range(i + 1, end) if re.match(r"(?:[-*]|\d+[.)])\s+\S", lines[j])]
    for n, j in enumerate(items):
      stop = items[n + 1] if n + 1 < len(items) else end
      body = "\n".join(lines[j:stop])
      first = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", lines[j]).replace("**", "").strip()
      out.append((j + 1, first, body))
  return out


def mentioned(ix, c):
  c = c.strip("./")
  if c.endswith(".m"):
    return [ids[0] for f, ids in ix.byFile.items() if f == c or f.endswith("/" + c)][:3]
  if c.endswith(".py"):
    mod = "py:module:" + c[:-3].replace("/", ".")
    return [mod] if mod in ix.byId else []
  m = re.fullmatch(r"@?(\w+)[/.](\w+)", c)
  if m and (m.group(1) in ix.mClass or m.group(1) in ix.pClass):
    out = [ix.mLookup(m.group(1), m.group(2))]
    out += [i for i in ix.byName.get(("python", m.group(2)), []) if ix.byId[i].get("cls") == m.group(1)]
    return [o for o in out if o]
  if not re.fullmatch(r"[A-Za-z]\w{3,}", c) or c in GENERIC:
    return []
  if c in ix.mClass or c in ix.pClass:
    return [x["id"] for x in (ix.mClass.get(c), ix.pClass.get(c)) if x]
  ids = [i for p in ("matlab", "python") for i in ix.byName.get((p, c), [])
         if ix.byId[i]["kind"] in ("function", "method")]
  return ids if len(ids) <= 8 else []


# the ALGORITHM declarations of every benchmark: a literal dict or list of dicts, read without
# importing the script; benchmarks naming the same algorithm add their entry points and cases to one node
def algorithms(ix, benchDir):
  algos = {}
  for path in sorted(glob.glob(os.path.join(benchDir, "*.py"))):
    tree = ast.parse(open(path).read())
    for st in tree.body:
      if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ALGORITHM" for t in st.targets):
        try:
          decl = ast.literal_eval(st.value)
        except ValueError:
          print(f"{os.path.relpath(path, ROOT)}: ALGORITHM is not a literal, skipped")
          continue
        for a in decl if isinstance(decl, list) else [decl]:
          n = algos.setdefault(a["name"], dict(id=f"algo:{a['name']}", port="algorithm", kind="algorithm",
                                               name=a["name"], entry={}, cases={}, files=[], tolerance={}))
          for port, es in a["entry"].items():
            n["entry"].setdefault(port, [])
            n["entry"][port] += [e for e in es if e not in n["entry"][port]]
          n["cases"].update(a.get("cases", {}))
          tol = a.get("tolerance")
          n["tolerance"].update({c: tol.get(c) if isinstance(tol, dict) else tol for c in a.get("cases", {})})
          n["stages"] = max(n.get("stages", 0), a.get("stages", 10))
          n["files"].append(os.path.relpath(path, ROOT))
  nodes, edges = [], []
  for n in algos.values():
    n["missing"] = [e for port, es in n["entry"].items() for e in es
                    if (("m:" if port == "matlab" else "py:") + e) not in ix.byId]
    nodes.append(n)
    edges += [dict(src=n["id"], dst=("m:" if port == "matlab" else "py:") + e, kind="entry", resolution="declared")
              for port, es in n["entry"].items() for e in es]
  return nodes, edges


# the node a raw function record of a port's profiler stands for: MATLAB by file and function
# name, Python by file and the last definition starting at or before the record's line
def recordMapper(ix, matlabRoot):
  mFile, pFile = {}, {}
  for n in ix.byId.values():
    if n["port"] == "matlab" and n.get("file"):
      mFile.setdefault(os.path.join(matlabRoot, n["file"]), []).append(n)
    elif n["port"] == "python" and n.get("file") and n["kind"] not in ("module", "class"):
      pFile.setdefault(os.path.join(ROOT, n["file"]), []).append(n)

  def mapM(rec):
    cands = mFile.get(rec["file"], [])
    sub = rec["name"].split(">")[-1].split(".")[-1]
    hit = [n for n in cands if n["name"] == sub] or [n for n in cands if n["kind"] != "local"]
    return hit[0]["id"] if hit else None

  def mapP(rec):
    cands = [n for n in pFile.get(rec["file"], []) if n["line"] <= rec["line"]]
    return max(cands, key=lambda n: n["line"])["id"] if cands else None

  return lambda port, rec: mapM(rec) if port == "matlab" else mapP(rec)


# the profile a benchmark run left: raw function records mapped onto node ids
def profiles(ix, benchOut, matlabRoot):
  node = recordMapper(ix, matlabRoot)
  out = []
  for path in sorted(glob.glob(os.path.join(benchOut, "*", "profile_*.json"))):
    p = json.load(open(path))
    recs = p["functions"]
    ids = [node(p["port"], r) for r in recs]
    agg = {}
    for r, i in zip(recs, ids):
      key = i or f"external:{r['name']}"
      a = agg.setdefault(key, dict(self=0.0, total=0.0, calls=0, children={}))
      a["self"] += r["self"]
      a["total"] += r["total"]
      a["calls"] += r["calls"]
      for c, t in r.get("children", []):
        ck = ids[c] or f"external:{recs[c]['name']}"
        if ck != key:
          a["children"][ck] = a["children"].get(ck, 0.0) + t
    out.append(dict(algorithm=p["algorithm"], port=p["port"], case=p["case"], wall=p["wall"],
                    file=os.path.relpath(path, ROOT), functions=agg))
  return out


# the coverage the test runs left (coverage_python.py, coverageMatlab.m): node -> the tests running it
def coverage(ix, matlabRoot):
  node = recordMapper(ix, matlabRoot)
  cov = {}
  for path in sorted(glob.glob(os.path.join(OUT, "coverage_*.json"))):
    port = "python" if "python" in os.path.basename(path) else "matlab"
    for test, recs in json.load(open(path)).items():
      tid = ("py:" + re.sub(r"\[.*\]$", "", test).replace(".py::", ".").replace("/", ".").replace("::", ".")
             if port == "python" else "m:" + test)
      for r in recs:
        i = node(port, dict(file=r[0], line=r[1], name=r[2]))
        if i:
          cov.setdefault(i, {})[tid] = 1
  return cov


# whether a source the graph is built from changed after it: the two trees, the benchmarks and
# their profiles, the coverage runs and the notes
def stale(graph=os.path.join(OUT, "graph.json"), matlabRoot=None, memory=MEMORY):
  if not os.path.exists(graph):
    return True
  built = os.path.getmtime(graph)
  mRoot = matlabRoot or json.load(open(graph)).get("matlabRoot", os.path.join(os.path.dirname(ROOT), "master"))
  skip = scan_matlab.SKIP | {".venv", "__pycache__", "build", "matlab", "out"}
  roots = [(mRoot, (".m", ".md")), (os.path.join(ROOT, "mtex"), (".py",)), (os.path.join(ROOT, "tests"), (".py",)),
           (os.path.join(ROOT, "bench"), (".py",)), (HERE, (".py",)), (os.path.join(ROOT, "docs"), (".md",))]
  roots += [(d, (".md",)) for d in memory] + [(os.path.join(JULIA, "docs", "research"), (".md",))]
  for root, ext in roots:
    for dirpath, dirnames, filenames in os.walk(root):
      dirnames[:] = [d for d in dirnames if d not in skip]
      if any(f.endswith(ext) and os.path.getmtime(os.path.join(dirpath, f)) > built for f in filenames):
        return True
  results = glob.glob(os.path.join(ROOT, "bench", "out", "*", "profile_*.json")) + glob.glob(os.path.join(OUT, "coverage_*.json"))
  return any(os.path.getmtime(f) > built for f in results)


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--matlab", default=os.path.join(os.path.dirname(ROOT), "master"))
  ap.add_argument("--memory", nargs="*", default=MEMORY)
  args = ap.parse_args(argv)
  t = time.perf_counter()
  mRoot = os.path.abspath(args.matlab)
  mn, me = scan_matlab.build(mRoot)
  pn, pe = scan_python.build(ROOT)
  for n in mn + pn:
    n["layer"] = layerOf(n)
    if isPlumbing(n):
      n["plumbing"] = True
  ix = Index(mn + pn)
  ce = counterparts(ix, pn)
  sources = [(f"memory-{os.path.basename(os.path.dirname(d)).split('-')[-1]}", os.path.join(d, "*.md")) for d in args.memory]
  sources += [("matlab", os.path.join(mRoot, "TODO.md")),
             ("matlab-adr", os.path.join(mRoot, "docs", "adr", "*.md")),
             ("python-research", os.path.join(ROOT, "docs", "research", "*.md")),
             ("python-plans", os.path.join(ROOT, "docs", "plans", "*.md")),
             ("matlab-tests", os.path.join(mRoot, "tests", "*.md")),
             ("julia-research", os.path.join(JULIA, "docs", "research", "*.md"))]
  nn, ne = notes(ix, sources)
  an, ae = algorithms(ix, os.path.join(ROOT, "bench"))
  prof = profiles(ix, os.path.join(ROOT, "bench", "out"), mRoot)
  cov = coverage(ix, mRoot)
  g = dict(built=time.strftime("%Y-%m-%d %H:%M"), matlabRoot=mRoot, pythonRoot=ROOT,
           nodes=mn + pn + nn + an, edges=me + pe + ce + ne + ae, profiles=prof, coverage=cov)
  os.makedirs(OUT, exist_ok=True)
  json.dump(g, open(os.path.join(OUT, "graph.json"), "w"))
  kinds = {}
  for e in g["edges"]:
    kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
  print(f"{len(g['nodes'])} nodes, {len(g['edges'])} edges {kinds}, {len(prof)} profiles, "
        f"{len(an)} algorithms in {time.perf_counter() - t:.1f} s", file=sys.stderr)
  # the Obsidian vault, when there is one
  import obsidian
  if os.path.isdir(obsidian.VAULT):
    obsidian.write_vault(obsidian.VAULT)
  for a in an:
    if a["missing"]:
      print(f"algorithm {a['name']}: entry points not in the graph: {a['missing']}", file=sys.stderr)


if __name__ == "__main__":
  main()
