# the static call graph of the Julia port ../mtexJulia, read by scan_julia.jl with Julia's own parser
#
# Node ids: "jl:dot" a generic function (the packages extend one another's generics, so a name is
# one function across them), "jl:dot@lib/MTEXGeometry/src/vec3.jl:128" one of its methods, with
# the type of its first annotated argument as its class, "jl:type:Orientation" a type,
# "jl:file:<file>" a source file, which carries the MATLAB names its header cites, and
# "jl:test:<file>" / "jl:bench:<file>" a test or benchmark script as one unit. Files are relative
# to the Julia port; the Julia halves of this repository's benchmarks are ../pymtex/bench/*.jl.
# A method calls the generic; the generic reaches its methods by "dispatch" edges resolved
# "generic". A method's "matlab" lists the MATLAB names the comment above it cites; a generic or
# type the packages export is marked "exported".
from __future__ import annotations

import glob
import json
import os
import subprocess

from scan_python import matlabRefs

HERE = os.path.dirname(os.path.abspath(__file__))
JULIA_BIN = os.path.expanduser("~/.juliaup/bin/julia")
# the package of a source file and the layer it is, one vocabulary with the other ports
LAYER = {"MTEXGeometry": "geometry", "MTEXFunctions": "functions", "MTEXMaps": "maps", "MTEXInterfaces": "io",
         "MTEXPlotting": "plotting"}
# types too general to stand for a class
GENERAL = set("""IO Real Integer AbstractString String Symbol Number Bool Function Any AbstractFloat Float64 Int Nothing
AbstractDict Dict NamedTuple Tuple Val Type Union AbstractRange UnitRange Colon Pair MIME""".split())
# Julia's operators in MATLAB's vocabulary
OPERATORS = {"*": ["mtimes", "times"], "+": ["plus"], "-": ["minus", "uminus"], "/": ["mrdivide", "rdivide"],
             "\\": ["mldivide", "ldivide"], "^": ["mpower", "power"], "==": ["eq"], "<": ["lt"], "<=": ["le"],
             "adjoint": ["ctranspose"], "getindex": ["subsref"], "setindex!": ["subsasgn"], "show": ["display", "char"],
             "vcat": ["vertcat", "cat"], "hcat": ["horzcat", "cat"], "length": ["length", "numel"]}
# Julia type names whose MATLAB class is not spelled alike
CLASSES = {"Vec3": ["vector3d"], "Direction": ["vector3d"], "Axial": ["vector3d"], "UnitVec3": ["vector3d"],
           "AbstractVec3": ["vector3d"], "Vector3d": ["vector3d", "Miller"], "Framed": ["vector3d", "Miller"],
           "FramedList": ["vector3d", "Miller"], "Directions": ["vector3d"], "PointGroup": ["crystalSymmetry", "specimenSymmetry"],
           "Grains": ["grain2d"], "AbstractOrientations": ["orientation"], "SingleOrientation": ["orientation"],
           "Orientations": ["orientation"], "Rotations": ["rotation"], "RotationColumns": ["rotation"],
           "AbstractSO3Fun": ["SO3Fun"], "AbstractS2Fun": ["S2Fun"], "AbstractS1Fun": ["S1Fun"],
           "AbstractS2VectorField": ["S2VectorField"], "AbstractS2AxisField": ["S2AxisField"],
           "AbstractSO3VectorField": ["SO3VectorField"], "Plane": ["plane3d"], "Curvature": ["curvatureTensor"],
           "DislocationDensity": ["dislocationDensityTensor"], "Christoffel": ["ChristoffelTensor"],
           "TriplePoints": ["triplePointList"], "Tensor": ["tensor"], "IPFColorKey": ["ipfColorKey", "ipfHSVKey"],
           "Embedding": ["embedding"], "Embeddings": ["embedding"], "SO3Grid": ["SO3Grid", "orientation"],
           "AbstractOrientationGrid": ["SO3Grid"], "PolygonSet": ["grain2d"], "ReferenceFrame": ["referenceFrame"]}


def sources(juliaRoot, pyRoot):
  files = [os.path.relpath(f, juliaRoot) for f in sorted(glob.glob(os.path.join(juliaRoot, "lib", "*", "src", "**", "*.jl"), recursive=True))]
  files += [os.path.relpath(f, juliaRoot) for pat in ("test/*.jl", "bench/*.jl") for f in sorted(glob.glob(os.path.join(juliaRoot, pat)))]
  files += [os.path.relpath(f, juliaRoot) for f in sorted(glob.glob(os.path.join(pyRoot, "bench", "*.jl")))]
  return files


# the records of scan_julia.jl, one JSON object per definition; a file's records are kept in
# out/julia_records.json and read again only when the file changed, since the parse takes seconds
def records(juliaRoot, files):
  cachePath = os.path.join(HERE, "out", "julia_records.json")
  cache = json.load(open(cachePath)) if os.path.exists(cachePath) else {}
  # a changed scanner reads everything again
  scanner = os.path.getmtime(os.path.join(HERE, "scan_julia.jl"))
  if cache.pop("scan_julia.jl", None) != scanner:
    cache = {}
  mtime = {f: os.path.getmtime(os.path.join(juliaRoot, f)) for f in files}
  todo = [f for f in files if cache.get(f, [None])[0] != mtime[f]]
  if todo:
    r = subprocess.run([JULIA_BIN, "--startup-file=no", os.path.join(HERE, "scan_julia.jl"), juliaRoot] + todo,
                       capture_output=True, text=True, timeout=300)
    if r.returncode:
      raise RuntimeError(r.stderr[-2000:])
    fresh = {f: [] for f in todo}
    for line in r.stdout.splitlines():
      if line.strip():
        rec = json.loads(line)
        fresh[rec["file"]].append(rec)
    cache.update({f: [mtime[f], recs] for f, recs in fresh.items()})
    cache = {f: cache[f] for f in files}
    os.makedirs(os.path.dirname(cachePath), exist_ok=True)
    json.dump(cache | {"scan_julia.jl": scanner}, open(cachePath, "w"))
  return [rec for f in files for rec in cache[f][1]]


def layerOf(file):
  parts = file.split("/")
  if parts[0] == "lib" and len(parts) > 2:
    return "tensors" if "tensors" in parts else LAYER.get(parts[1], "other")
  return "tests" if parts[0] == "test" else "bench" if "bench" in parts else "other"


# the MATLAB names the comment block right above a line cites (a docstring's are added from its
# record), and those of a file's header
def comments(juliaRoot, files):
  above, header = {}, {}
  for f in files:
    try:
      lines = open(os.path.join(juliaRoot, f), encoding="utf-8").read().split("\n")
    except OSError:
      continue
    above[f] = lines
    head = []
    for line in lines:
      if not line.startswith("#"):
        break
      head.append(line[1:])
    header[f] = matlabRefs(" ".join(head))
  return above, header


def citedAbove(lines, line):
  block = []
  k = line - 2
  while 0 <= k < len(lines) and lines[k].lstrip().startswith("#"):
    block.append(lines[k].lstrip()[1:])
    k -= 1
  return matlabRefs(" ".join(reversed(block)))


def build(juliaRoot, pyRoot):
  files = sources(juliaRoot, pyRoot)
  recs = records(juliaRoot, files)
  above, header = comments(juliaRoot, files)
  alias = {r["name"]: r["of"] for r in recs if r["kind"] == "alias"}
  exported = {n for r in recs if r["kind"] == "export" for n in r["names"]}
  nodes, edges = {}, []

  def cls(t):
    seen = set()
    while t in alias and t not in CLASSES and t not in seen:
      seen.add(t)
      t = alias[t]
    return None if t is None or t in GENERAL else t

  for f in files:
    if f.startswith("lib/"):
      nodes[f"jl:file:{f}"] = dict(id=f"jl:file:{f}", port="julia", kind="module", name=os.path.basename(f)[:-3], file=f,
                                   line=1, matlab=header.get(f, []), layer=layerOf(f))
  for r in recs:
    if r["kind"] == "type" and r["name"]:
      tid = f"jl:type:{r['name']}"
      nodes.setdefault(tid, dict(id=tid, port="julia", kind="class", name=r["name"], cls=r["name"], file=r["file"],
                                 line=r["line"], supers=[r["super"]] if r["super"] else [], layer=layerOf(r["file"])))
  for r in recs:
    if r["kind"] != "method":
      continue
    name, f = r["name"], r["file"]
    gid = f"jl:{name}"
    g = nodes.setdefault(gid, dict(id=gid, port="julia", kind="function", name=name, file=f, line=r["line"], lines=0,
                                   qual=r["qual"], layer=layerOf(f)))
    # the generic sits where the port first defines it rather than extends it
    if g["qual"] and not r["qual"]:
      g.update(file=f, line=r["line"], qual="", layer=layerOf(f))
    lines = r["end"] - r["line"] + 1
    g["lines"] += lines
    mid = f"jl:{name}@{f}:{r['line']}"
    if mid in nodes:
      continue
    c = cls(r["type"])
    nodes[mid] = dict(id=mid, port="julia", kind="method", name=name, cls=c, file=f, line=r["line"], lines=lines,
                      qual=r["qual"], layer=layerOf(f),
                      matlab=citedAbove(above.get(f, []), r["line"]) + matlabRefs(r.get("doc", "")),
                      _calls=r["calls"], _refs=r["refs"])
    edges.append(dict(src=gid, dst=mid, kind="dispatch", resolution="generic"))
    if c and f"jl:type:{c}" in nodes:
      edges.append(dict(src=f"jl:type:{c}", dst=mid, kind="member", resolution="exact"))
  for r in recs:
    if r["kind"] == "script":
      kind = "test" if r["file"].startswith("test/") else "bench"
      sid = f"jl:{kind}:{r['file']}"
      nodes[sid] = dict(id=sid, port="julia", kind=kind, name=os.path.basename(r["file"])[:-3], file=r["file"], line=1,
                        layer=layerOf(r["file"]), _calls=r["calls"], _refs=r["refs"])
  # calls to the generics of the port; a name passed as a value counts when it is no common word
  for nid, n in nodes.items():
    called = {f"jl:{name}" for name in n.pop("_calls", [])} - {nid}
    edges += [dict(src=nid, dst=c, kind="call", resolution="name") for c in sorted(called) if c in nodes]
    refs = {f"jl:{name}" for name in n.pop("_refs", []) if len(name) > 3} - called - {nid}
    edges += [dict(src=nid, dst=c, kind="call", resolution="ref") for c in sorted(refs) if c in nodes]
  for n in nodes.values():
    if n["kind"] in ("function", "class") and n["name"] in exported:
      n["exported"] = True
  for t in [n for n in nodes.values() if n["kind"] == "class"]:
    for s in t["supers"]:
      if f"jl:type:{s}" in nodes:
        edges.append(dict(src=t["id"], dst=f"jl:type:{s}", kind="inherits", resolution="exact"))
  seen = set()
  edges = [e for e in edges if (e["src"], e["dst"], e["kind"]) not in seen and not seen.add((e["src"], e["dst"], e["kind"]))]
  return list(nodes.values()), edges


# the MATLAB classes a Julia type stands for: the table, else the same name in any case, with a
# Tensor suffix, or on SO(3) and the sphere (a kernel type serves both)
def matlabClasses(t, mClasses):
  if not t:
    return []
  if t in CLASSES:
    return [c for c in CLASSES[t] if c in mClasses]
  lower = {c.lower(): c for c in mClasses}
  return [lower[c.lower()] for c in (t, t + "Tensor", "SO3" + t, "S2" + t, "S1" + t) if c.lower() in lower]


# the counterparts of the Julia nodes in MATLAB and Python: a method by its class and name, a
# generic by name where no method of it matched, a type by its class, anything by what it cites
def counterparts(ix, jNodes):
  edges = []
  pyByClsName = {}
  for n in ix.byId.values():
    if n["port"] == "python" and n["kind"] == "method":
      pyByClsName.setdefault((n["cls"], n["name"]), n["id"])
  pyFunctions = {}
  for n in ix.byId.values():
    if n["port"] == "python" and n["kind"] == "function" and not n["name"].startswith("_"):
      pyFunctions.setdefault(n["name"], []).append(n["id"])
  matched = set()

  def names(name):
    base = name[:-1] if name.endswith("!") else name
    return OPERATORS.get(base, [base])

  def add(src, dst, how):
    edges.append(dict(src=src, dst=dst, kind="counterpart", resolution=how))

  for n in jNodes:
    if n["kind"] == "method":
      for mc in matlabClasses(n.get("cls"), ix.mClass.keys() | ix.mMethods.keys()):
        for nm in names(n["name"]) + ([mc] if n["name"] == n.get("cls") else []):
          # the class's own methods of the name, else the one it inherits
          hits = [i for i in ix.byName.get(("matlab", nm), []) if ix.byId[i].get("cls") == mc] or [ix.mLookup(mc, nm)]
          for hit in filter(None, hits):
            add(n["id"], hit, "name")
            matched.add(f"jl:{n['name']}")
          py = pyByClsName.get((mc, nm)) or pyByClsName.get((mc, "__init__" if nm == mc else None))
          if py:
            add(n["id"], py, "name")
    if n["kind"] == "class":
      for mc in matlabClasses(n["name"], ix.mClass.keys()):
        add(n["id"], ix.mClass[mc]["id"], "name")
        if mc in ix.pClass:
          add(n["id"], ix.pClass[mc]["id"], "name")
    for ref in n.get("matlab") or []:
      for hit in ix.resolveMatlab(ref):
        add(n["id"], hit, "cited")
  for n in jNodes:
    if n["kind"] != "function" or n["name"].startswith("_"):
      continue
    for nm in names(n["name"]):
      if n["id"] not in matched:
        for hit in ix.byName.get(("matlab", nm), []):
          if ix.byId[hit]["kind"] == "function" or len(ix.byName.get(("matlab", nm), [])) <= 8:
            add(n["id"], hit, "name")
      for py in pyFunctions.get(nm, []):
        add(n["id"], py, "name")
  seen = set()
  return [e for e in edges if (e["src"], e["dst"]) not in seen and not seen.add((e["src"], e["dst"]))]
