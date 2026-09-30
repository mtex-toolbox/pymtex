# what is left of the spellings docs/plans/syntax.md replaces, and the MATLAB counterparts the
# renames lost
#
#   .venv/bin/python tools/syntaxcheck.py              the old keywords and names still public
#   .venv/bin/python tools/syntaxcheck.py --baseline   writes the counterparts of the graph as the baseline
#   .venv/bin/python tools/syntaxcheck.py --graph      the MATLAB nodes that had a Python counterpart and lost it
#
# The keywords are read from the signatures of the public API and from the quoted keys the
# plotting reads out of **kw; the graph is tools/graph/out/graph.json, rebuilt by tools/graph/build.py.
import ast
import inspect
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH = os.path.join(ROOT, "tools", "graph", "out", "graph.json")
BASELINE = os.path.join(ROOT, "bench", "out", "syntax-counterparts-before.txt")

# the keyword spellings the plan replaces, and their replacements
KEYWORDS = {
  "tolerance": "tol", "maxit": "maxIter", "iterMax": "maxIter", "itermax": "maxIter", "iterMin": "minIter",
  "silent": "verbose", "MarkerSize": "markerSize", "DisplayName": "displayName", "EdgeAlpha": "edgeAlpha",
  "Length": "length", "linewidth": "lineWidth", "cs": "CS", "ss": "SS", "nothinning": "thinning",
  "noGhostCorrection": "ghostCorrection", "noScaling": "scaling", "noBoundary": "boundary",
  "noOrientFaces": "orientFaces", "noCenter": "center", "noLabel": "label", "noGrid": "grid",
  "noSymmetry": "symmetry", "noAntipodal": "antipodal", "noIntern": "intern", "noRobust": "robust",
  "noTitle": "title", "noSurface": "surface", "noSimplify": "simplify", "noRefine": "refine",
  "noPseudoDiagonal": "pseudoDiagonal", "noP2C": "p2c", "noDiagonal": "diagonal",
  "regularization": "regularisation", "fak": "factor", "nbins": "bins",
}

# strings that are no MTEX keywords: MATLAB's positional flags, which keep MATLAB's spelling, and
# matplotlib's spelling of what the plots hand to it
FLAG_STRINGS = {"cs", "ss", "Length", "silent", "noSymmetry", "linewidth"}

# the public names the plan removes or renames, and what takes their place
NAMES = {
  "plotPDF": "plotPF", "plotIPDF": "plotIPF", "plotPoleFigure": "plotPF", "calcPDF": "radon", "pdf": "radon",
  "ipdf": "radon", "eye": "id", "eq": "isclose", "eqTol": "isclose", "hist": "histogram",
  "project2FundamentalRegion": "projectIntoFundamentalRegion", "project2EulerFR": "projectIntoEulerRegion",
  "restrict2Upper": "restrictToUpper", "restrict2Lower": "restrictToLower", "SRight": "frameA",
  "SLeft": "frameB", "CS1": "frameA", "CS2": "frameB", "plotSpektra": "plotSpectrum", "calcBario": "barycentric",
  "numChilds": "numChildren", "nnodes": "numNodes", "nrings": "numRings", "nNeighbors": "numNeighbors",
  "angle_outer": "angle(outer=True)", "dot_outer": "dot(outer=True)", "addnoise": "addNoise",
  "addFeature_singleStep": "addFeature", "addFeature_circularSubgrain": "addFeature",
  "addFeature_simpleGradient": "addFeature", "fluorite_complex": "fluoriteComplex", "textureindex": "textureIndex",
  "plotx2east": "plottingConvention.default", "plotx2north": "plottingConvention.default",
  "plotx2south": "plottingConvention.default", "plotx2west": "plottingConvention.default",
  "plotzIntoPlane": "plottingConvention.default", "plotzOutOfPlane": "plottingConvention.default",
  "plotEBSD3": "plot", "plotGrains3d": "plot", "plotBoundary3d": "plot", "plot3dMap": "plot",
  "quiver3dMap": "quiver", "quiverVectors": "quiver3d", "plotPoints3d": "plot",
  "scatter3": "plot", "quiver3": "quiver3d", "export_VPSC": "export", "loadGrains_Dream3d": "loadGrains3d",
}
# the per-format loaders and exporters, private after R5
PREFIXES = ("loadEBSD_", "exportEBSD_", "loadODF_", "exportODF_", "loadPoleFigure_")
# where a name of NAMES stays legitimately: tensor.eye is a matrix, grains keep the *Boundary forms
KEEP = {("tensor", "eye")}
# the standard orientations leave the flat namespace, orientation keeps them
STANDARD = "mtex.geometry.standardorientations"


def publicApi():
  sys.path.insert(0, ROOT)
  import mtex
  for n in sorted(x for x in dir(mtex) if not x.startswith("_")):
    o = getattr(mtex, n)
    if inspect.ismodule(o):
      continue
    if inspect.isclass(o):
      yield n, None, o
      for m, v in o.__dict__.items():
        if not m.startswith("_"):
          yield n, m, getattr(v, "fget", None) or getattr(v, "__func__", v)
    elif callable(o):
      yield None, n, o


def keywordsLeft():
  out = []
  for cls, name, f in publicApi():
    try:
      params = inspect.signature(f).parameters if callable(f) and name else {}
    except (TypeError, ValueError):
      continue
    out += [(f"{cls + '.' if cls else ''}{name}", p) for p, v in params.items()
            if p in KEYWORDS and v.kind not in (v.VAR_POSITIONAL, v.VAR_KEYWORD)]
  # the keys the plotting and the loaders read out of **kw
  for dirpath, _, files in os.walk(os.path.join(ROOT, "mtex")):
    for fn in files:
      if not fn.endswith(".py"):
        continue
      path = os.path.join(dirpath, fn)
      for node in ast.walk(ast.parse(open(path, encoding="utf-8").read())):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in KEYWORDS \
           and node.value not in FLAG_STRINGS:
          out.append((os.path.relpath(path, ROOT) + f":{node.lineno}", node.value))
  return out


def namesLeft():
  import mtex
  out = []
  for cls, name, f in publicApi():
    if name is None:
      continue
    if name in NAMES and (cls, name) not in KEEP:
      out.append((f"{cls + '.' if cls else ''}{name}", NAMES[name]))
    if cls is None and name.startswith(PREFIXES):
      out.append((name, "private"))
    if cls is None and getattr(f, "__module__", "") == STANDARD and name not in ("byMiller", "variants"):
      out.append((name, f"orientation.{name}"))
  return out


def counterparts():
  g = json.load(open(GRAPH))
  return {e["dst"] for e in g["edges"] if e["kind"] == "counterpart" and e["src"].startswith("py:")}


if __name__ == "__main__":
  if "--baseline" in sys.argv:
    open(BASELINE, "w").write("\n".join(sorted(counterparts())) + "\n")
  elif "--graph" in sys.argv:
    before = set(open(BASELINE).read().split())
    lost = sorted(before - counterparts())
    print(f"{len(lost)} MATLAB nodes lost their Python counterpart")
    print("\n".join(lost))
  else:
    kws, names = keywordsLeft(), namesLeft()
    for where, k in kws:
      print(f"keyword  {where}: {k} -> {KEYWORDS[k]}")
    for where, new in names:
      print(f"name     {where} -> {new}")
    print(f"{len(kws)} old keywords, {len(names)} old names")
