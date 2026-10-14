# the function reference from the docstrings: a page per public function, class and member of
# mtex, docs/build/reference/<name>.md, named as MATLAB's reference page where the name reaches
# the same thing in Python (fibre.angle, EBSD.calcGrains, EBSD_index for the class), so that the
# site's MATLAB | Python switch pairs them, and by its Python name otherwise. The names of
# MATLAB's pages are docs/reference-matlab.txt, read once off the site.
# Usage:
#   .venv/bin/python docs/reference.py                     every page
#   .venv/bin/python docs/reference.py fibre.angle ...     these pages
#   .venv/bin/python docs/reference.py --matlab ../web     refresh docs/reference-matlab.txt
import inspect
import json
import os
import re
import sys

import numpy as np

import mtex

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pagesDir = os.path.join(root, "docs", "pages")
outDir = os.path.join(root, "docs", "build", "reference")
matlabList = os.path.join(root, "docs", "reference-matlab.txt")
SOURCE = "https://github.com/mtex-toolbox/pymtex/blob/main/{}#L{}"
sys.path.insert(0, os.path.join(root, "tools"))
from syntaxcheck import NAMES, PREFIXES  # noqa: E402

# the Python class a MATLAB class is folded into: its constructor is a function returning it,
# or its subclasses are one class here
ALIASES = {"Miller": "vector3d", "crystalSymmetry": "crystalFrame", "specimenSymmetry": "specimenFrame",
           "EBSDsquare": "EBSD", "EBSDhex": "EBSD", "EBSDgrid": "EBSD", "EBSD3square": "EBSD3",
           "S2FunHarmonicSym": "S2FunHarmonic", "S1Fun": "S1FunHarmonic", "S2VectorField": "S2VectorFieldHarmonic",
           "SO3VectorField": "SO3VectorFieldHarmonic", "S2AxisField": "S2AxisFieldHarmonic"}
# MATLAB's operator methods by their Python spelling
OPERATORS = {"subsref": "__getitem__", "eq": "__eq__", "ne": "__ne__", "plus": "__add__", "minus": "__sub__",
             "uminus": "__neg__", "times": "__mul__", "mtimes": "__mul__", "rdivide": "__truediv__",
             "mrdivide": "__truediv__", "power": "__pow__", "mpower": "__pow__", "abs": "__abs__", "inv": "__invert__",
             "eval": "__call__", "lt": "__lt__", "le": "__le__", "gt": "__gt__", "ge": "__ge__",
             "horzcat": "cat", "vertcat": "cat", "transpose": "T", "ctranspose": "T"}
# MATLAB methods of the language, not the mathematics, which Python has by NumPy's means
PLUMBING = set("""subsasgn subSet setSubSet numArgumentsFromSubscript end numel length size ndims isempty
isscalar loadobj saveobj copy delete display disp char dispLine dispData repmat reshape permute squeeze
isequal fullDouble subsind""".split())


# the names of MATLAB's reference pages
def matlabPages():
  with open(matlabList, encoding="utf-8") as fh:
    return [line.strip() for line in fh if line.strip()]


def writeMatlabPages(site):
  d = os.path.join(site, "pages", "function_reference_matlab")
  names = sorted(f[:-5] for f in os.listdir(d) if f.endswith(".html"))
  with open(matlabList, "w", encoding="utf-8") as fh:
    fh.write("\n".join(names) + "\n")
  print(f"{len(names)} MATLAB pages")


# the function behind an attribute: a property by its getter, a static or class method by its function
def unwrap(obj):
  if isinstance(obj, property):
    return obj.fget
  return getattr(obj, "__func__", obj)


def isOurs(obj):
  return getattr(unwrap(obj), "__module__", "").startswith("mtex")


def publicClass(name):
  obj = getattr(mtex, ALIASES.get(name, name), None)
  return obj if inspect.isclass(obj) and isOurs(obj) else None


# the Python spelling of a MATLAB name, or None: angle_outer is angle(outer=True), a file
# format's loader loadEBSD_ang the loader of every format
def renamed(name):
  new = NAMES.get(name, "").split("(")[0]
  if not new:
    new = next((p[:-1] for p in PREFIXES if name.startswith(p)), "")
  return new or None


# an attribute path of the flat namespace, plottingConvention.default
def lookup(path):
  obj = mtex
  for part in path.split("."):
    obj = getattr(obj, part, None)
  return obj


# what a MATLAB page name reaches in Python: (kind, object, class), or None
def resolve(page):
  if page.endswith("_index"):
    cls = publicClass(page[:-6])
    return ("class", cls, cls) if cls else None
  if "." not in page:
    obj = getattr(mtex, page, None) or lookup(renamed(page) or "_")
    if inspect.isclass(obj) and isOurs(obj):
      return ("class", obj, obj)
    return ("function", obj, None) if callable(obj) and isOurs(obj) else None
  c, m = page.split(".", 1)
  cls = publicClass(c)
  if cls is None or m in PLUMBING:
    return None
  if m == c:
    return ("constructor", cls, cls)
  for cand in (m, renamed(m), OPERATORS.get(m)):
    obj = inspect.getattr_static(cls, cand, None) if cand and cand.isidentifier() else None
    if obj is not None and isOurs(obj):
      return ("property" if isinstance(obj, property) else "method", obj, cls)
  # a method MATLAB has on the class is a function of the flat namespace here: calcGrains(ebsd)
  obj = getattr(mtex, m, None) or lookup(renamed(m) or "_")
  if callable(obj) and not inspect.isclass(obj) and isOurs(obj):
    return ("function", obj, cls)
  return None


# every page: name -> (kind, object, class, MATLAB's page or None)
def catalogue():
  pages, reached = {}, set()
  for p in matlabPages():
    hit = resolve(p)
    if hit:
      pages[p] = hit + (p,)
      reached.add(id(unwrap(hit[1])))
  for n in sorted(dir(mtex)):
    obj = getattr(mtex, n)
    if n.startswith("_") or not isOurs(obj) or not callable(obj):
      continue
    if inspect.isclass(obj):
      if f"{n}_index" not in pages:
        pages[f"{n}_index"] = ("class", obj, obj, None)
      for m, v in vars(obj).items():
        f = unwrap(v)
        if m.startswith("_") or not isOurs(v) or id(f) in reached or f"{n}.{m}" in pages:
          continue
        if callable(f):
          pages[f"{n}.{m}"] = ("property" if isinstance(v, property) else "method", v, obj, None)
    elif id(obj) not in reached and n not in pages:
      pages[n] = ("function", obj, None, None)
  return pages


# ------------------------------------------------------------------ rendering

# a default shown as it is written: an angle of whole or half degrees as 15*degree
class Shown:
  def __init__(self, text):
    self.text = text

  def __repr__(self):
    return self.text


def shownDefault(v):
  if isinstance(v, float) and v not in (0.0,) and np.isfinite(v):
    d = v / mtex.degree
    if abs(d * 2 - round(d * 2)) < 1e-9 and abs(d) < 3600:
      return Shown(f"{d:g}*degree")
  return v


def signature(name, obj):
  try:
    sig = inspect.signature(obj)
  except (TypeError, ValueError):
    return None
  params = [p.replace(annotation=inspect.Parameter.empty,
                      default=shownDefault(p.default) if p.default is not inspect.Parameter.empty else p.default)
            for p in sig.parameters.values()]
  return f"{name}{sig.replace(parameters=params, return_annotation=inspect.Signature.empty)}"


# a docstring as Markdown: prose as paragraphs, lines indented under the prose as Python code
def docBody(doc, link):
  out, code = [], []
  for line in doc.split("\n"):
    if line.startswith("  ") or (code and not line.strip()):
      code.append(line[2:])
      continue
    if code:
      out += ["", "```python", *[c for c in "\n".join(code).strip("\n").split("\n")], "```", ""]
      code = []
    out.append(re.sub(r"`([A-Za-z_][\w.]*)`", link, line))
  if code:
    out += ["", "```python", *"\n".join(code).strip("\n").split("\n"), "```"]
  return "\n".join(out).strip("\n")


# the doc pages calling a name: a function where it is called, a method only where no other
# class has a member of its name, since the source does not say what a receiver is
def usedOn(name, cls, calls, owners):
  if cls is not None and owners.get(name, 0) > 1:
    return []
  key = name if cls is None else "." + name
  return [page for page, names in calls.items() if key in names]


# the class count of every member name
def memberOwners():
  count = {}
  for n in dir(mtex):
    obj = getattr(mtex, n)
    if inspect.isclass(obj) and isOurs(obj):
      for m in vars(obj):
        count[m] = count.get(m, 0) + 1
  return count


# the names each doc page calls, `name(` and `.name(`, from its code lines
def pageCalls(sources):
  return {page: set(re.findall(r"(?<![\w.])([A-Za-z_]\w*)\(", src)) | {"." + m for m in re.findall(r"\.([A-Za-z_]\w*)\(", src)}
          for page, src in sources.items()}


def pageSources():
  found = {}
  for chapter in sorted(os.listdir(pagesDir)):
    d = os.path.join(pagesDir, chapter)
    if os.path.isdir(d):
      for f in sorted(os.listdir(d)):
        if f.endswith(".py"):
          src = open(os.path.join(d, f), encoding="utf-8").read()
          # code only: a light-format page keeps its prose in comment lines
          found[f[:-3]] = "\n".join(line for line in src.split("\n") if not line.lstrip().startswith("#"))
  return found


def source(obj):
  try:
    f = inspect.getsourcefile(unwrap(obj))
    line = inspect.getsourcelines(unwrap(obj))[1]
  except (TypeError, OSError):
    return None
  return SOURCE.format(os.path.relpath(f, root), line)


def render(name, entry, pages, calls, owners):
  kind, obj, cls, matlab = entry
  member = name.split(".", 1)[1] if "." in name and kind in ("method", "property") else None

  def link(m):
    t = m.group(1)
    for cand in (t, f"{cls.__name__}.{t}" if cls else None, f"{t}_index"):
      if cand and cand in pages and cand != name:
        return f"[`{t}`]({cand}_py.html)"
    return m.group(0)

  # the name a function is called by: plotx2north is plottingConvention.default
  called = getattr(obj, "__name__", None) or renamed(name) or name
  title = f"The class {obj.__name__}" if kind == "class" else name
  doc = inspect.getdoc(unwrap(obj)) or ""
  lines = [f"# {title}", ""]
  if kind == "property":
    lines += [f"A property of [{cls.__name__}]({cls.__name__}_index_py.html).", ""]
  elif kind == "method":
    lines += [f"A method of [{cls.__name__}]({cls.__name__}_index_py.html).", ""]
  summary, _, rest = doc.partition("\n\n")
  if summary:
    lines += [docBody(summary, link), ""]
  if kind in ("function", "method", "constructor"):
    shown = called if kind == "function" else f"{cls.__name__}.{member}" if member else cls.__name__
    sig = signature(shown, unwrap(obj) if kind != "constructor" else obj)
    if sig:
      lines += ["## Syntax", "", "```python", sig, "```", ""]
  if rest.strip():
    lines += [docBody(rest, link), ""]
  if kind == "class":
    lines += members(obj, pages)
  used = usedOn(member or called.split(".")[-1], cls if member else None, calls, owners)
  if used:
    lines += ["## Used on", "", ", ".join(f"[{p}]({p}_py.html)" for p in used[:12]), ""]
  if matlab:
    lines += [f"MATLAB's [{matlab}]({matlab}.html)", ""]
  src = source(obj)
  if src:
    lines += [f'<font size="2"><a href="{src}">source</a></font>', ""]
  return "\n".join(lines)


# the first sentence of a docstring, on one line, for a table cell
def firstSentence(doc):
  text = doc.split("\n\n")[0].replace("\n", " ").replace("|", "\\|")
  m = re.search(r"[.;](\s|$)", text)
  return text[:m.start()] if m else text


# the members of a class as a table of their summaries
def members(cls, pages):
  rows = []
  for n, entry in pages.items():
    if entry[2] is cls and entry[0] in ("method", "property", "function", "constructor") and n != f"{cls.__name__}_index":
      rows.append(f"| [{n.split('.', 1)[-1]}]({n}_py.html) | {firstSentence(inspect.getdoc(unwrap(entry[1])) or '')} |")
  if not rows:
    return []
  return ["## Members", "", "| | |", "|---|---|", *sorted(rows, key=str.lower), ""]


def build(names=()):
  pages = catalogue()
  calls, owners = pageCalls(pageSources()), memberOwners()
  os.makedirs(outDir, exist_ok=True)
  chosen = [n for n in pages if not names or n in names]
  for n in chosen:
    with open(os.path.join(outDir, n + ".md"), "w", encoding="utf-8") as fh:
      fh.write(render(n, pages[n], pages, calls, owners))
  index = {n: {"kind": e[0], "matlab": e[3]} for n, e in pages.items()}
  with open(os.path.join(outDir, "index.json"), "w", encoding="utf-8") as fh:
    json.dump(index, fh, indent=0, sort_keys=True)
  paired = sum(e[3] is not None for e in pages.values())
  print(f"{len(chosen)} pages written, {paired} of them MATLAB's names, {len(pages) - paired} Python's")
  return pages


if __name__ == "__main__":
  if sys.argv[1:2] == ["--matlab"]:
    writeMatlabPages(sys.argv[2])
  else:
    build(sys.argv[1:])
