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
import tokenize

import numpy as np

import mtex
from mtex.geometry._array import MtexArray

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pagesDir = os.path.join(root, "docs", "pages")
outDir = os.path.join(root, "docs", "build", "reference")
matlabList = os.path.join(root, "docs", "reference-matlab.txt")
# the page of every Python name, which openDoc and the links of the site read
namesFile = os.path.join(root, "mtex", "io", "referencenames.json")
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
# how an operator is written, and what it is when its method says nothing
OPERATOR_FORMS = {"__add__": ("a + b", "the sum"), "__radd__": ("b + a", "the sum"), "__sub__": ("a - b", "the difference"),
                  "__rsub__": ("b - a", "the difference"), "__mul__": ("a * b", "the product"),
                  "__rmul__": ("b * a", "the product"), "__matmul__": ("a @ b", "the matrix product"),
                  "__truediv__": ("a / b", "the quotient"), "__rtruediv__": ("b / a", "the quotient"),
                  "__neg__": ("-a", "the negative"), "__pow__": ("a ** n", "the power"), "__abs__": ("abs(a)", "the absolute value"),
                  "__invert__": ("~a", "the inverse"), "__eq__": ("a == b", "elementwise equality"),
                  "__ne__": ("a != b", "elementwise inequality"), "__lt__": ("a < b", "elementwise comparison"),
                  "__le__": ("a <= b", "elementwise comparison"), "__gt__": ("a > b", "elementwise comparison"),
                  "__ge__": ("a >= b", "elementwise comparison"), "__getitem__": ("a[ind]", "the elements selected by ind"),
                  "__setitem__": ("a[ind] = b", "the elements selected by ind replaced"), "__call__": ("f(x)", "the evaluation at x"),
                  "__len__": ("len(a)", "the length of the first axis")}

# the function plot hands each class to, shown on that class's plot page: the generic plot
# chooses by isinstance, its own docstring lists the forms
PLOT = {"EBSD": "mapplot.plotEBSD", "grain2d": "mapplot.plotGrains", "grainBoundary": "mapplot.plotBoundary",
        "triplePointList": "mapplot.plotTriplePoints", "mapImage": "mapplot.plotMapImage",
        "ODFSections": "sectionplot.plotSections", "PoleFigure": "polefigureplot._plotPoleFigure",
        "S1FunHarmonic": "polarplot.plotS1", "S1FunHandle": "polarplot.plotS1", "shape2d": "polarplot.plotShape",
        "crystalShape": "shapeplot.plotShape", "plane3d": "shapeplot.plotPlane", "orientationRegion": "spaceplot.plot3d",
        "sphericalRegion": "sphericalplot.plotRegion", "planarColorKey": "planarkey.plotPlanarKey",
        "SO3Fun": "sectionplot.plotSection", "referenceFrame": "sphericalplot.plotSymmetry"}


# the implementation plot uses for a class, or None
def plotImplementation(cls):
  import importlib
  for c in cls.__mro__:
    if c.__name__ in PLOT:
      mod, name = PLOT[c.__name__].split(".")
      return getattr(importlib.import_module(f"mtex.plotting.{mod}"), name)
  return None


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


# the docstring of an attribute, a property's own before its getter's; its lines keep their
# indentation relative to the body of the def, which inspect.getdoc loses when every line
# after the summary is a calling form, and which Python 3.13 strips from __doc__ already: the
# literal is read from the source, a docstring a factory writes into __doc__ taken as written
def docOf(obj):
  f = obj if isinstance(obj, property) and obj.__doc__ else unwrap(obj)
  raw = getattr(f, "__doc__", None)
  if not isinstance(raw, str) or not raw.strip():
    return inspect.getdoc(f) or ""
  # a callable instance, plottingConvention.default, by its class's docstring
  if not (inspect.isroutine(f) or inspect.isclass(f) or isinstance(f, property)) and raw == type(f).__doc__:
    f = type(f)
  literal = sourceDocstring(f.fget if isinstance(f, property) else f)
  if literal is None:
    first, *rest = raw.expandtabs().split("\n")
    return "\n".join([first.strip()] + [line.rstrip() for line in rest]).strip("\n")
  text, body = literal
  first, *rest = text.expandtabs().split("\n")
  return "\n".join([first.strip()] + [line[body:] if line[:body].strip() == "" else line.lstrip() for line in rest]).strip("\n")


# the docstring of a def or class as its source writes it, and the indentation of its body
def sourceDocstring(f):
  import ast
  import textwrap
  try:
    lines = inspect.getsourcelines(f)[0]
  except (TypeError, OSError, SyntaxError, tokenize.TokenError):
    return None
  text = textwrap.dedent("".join(lines))
  try:
    tree = ast.parse(text)
  except SyntaxError:
    return None
  node = tree.body[0] if tree.body else None
  if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
    return None
  doc = ast.get_docstring(node, clean=False)
  if doc is None or doc.strip() != (f.__doc__ or "").strip() and inspect.cleandoc(doc) != inspect.cleandoc(f.__doc__ or ""):
    return None
  head = next(line for line in text.split("\n") if line.lstrip().startswith(("def ", "class ", "async def ")))
  return doc, len(head) - len(head.lstrip()) + 2


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
    # T transposes on the containers, but is the Miller-Bravais index T on vector3d
    if cand == "T" and obj is not inspect.getattr_static(MtexArray, "T"):
      continue
    if obj is not None and isOurs(obj):
      if unwrap(obj) is mtex.plot and plotImplementation(cls):
        return ("function", plotImplementation(cls), cls)
      return ("property" if isinstance(obj, property) else "method", obj, cls)
  # a method MATLAB has on the class is a function of the flat namespace here: calcGrains(ebsd)
  # but not a forwarder, norm(x) for x.norm(), which the class does not answer
  obj = getattr(mtex, m, None) or lookup(renamed(m) or "_")
  if callable(obj) and not inspect.isclass(obj) and isOurs(obj) and not hasattr(obj, "forwardsTo"):
    return ("function", plotImplementation(cls) if obj is mtex.plot and plotImplementation(cls) else obj, cls)
  return None


# every page: name -> (kind, object, class, MATLAB's page or None)
def catalogue():
  pages, reached, docPages = {}, set(), set(pageSources())
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
      # a documentation page has the name already: SchmidFactor
      pages[f"mtex.{n}" if n in docPages else n] = ("function", obj, None, None)
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
# the sections of a docstring, MATLAB's help blocks: a header line of its own, its entries
# indented two spaces, 'name - text' for Input, Output and Keywords, a continuation indented
# further
SECTIONS = ("Input", "Output", "Keywords", "References", "See also")


def parseDoc(doc):
  lines = doc.split("\n")
  heads = [k for k, line in enumerate(lines) if line.strip() in SECTIONS and not line.startswith(" ")]
  main = "\n".join(lines[:heads[0]] if heads else lines).rstrip()
  sections = {}
  for k, h in enumerate(heads):
    body = lines[h + 1:heads[k + 1] if k + 1 < len(heads) else len(lines)]
    entries = []
    for line in body:
      if not line.strip():
        continue
      if line.startswith("  ") and not line.startswith("   ") or not entries:
        entries.append(line.strip())
      else:
        entries[-1] += " " + line.strip()
    sections[lines[h].strip()] = entries
  return main, sections


# a 'name - text' entry
def entry(e):
  name, _, text = e.partition(" - ")
  return name.strip(), text.strip()


# calling forms as MATLAB's page shows them, the explanation of a form as text above its own
# code box: a form's trailing '# comment' becomes the text, forms without one share a box
def renderForms(code, link):
  out, plain = [], []

  def flush():
    if plain:
      out.extend(["```python", *plain, "```", ""])
      plain.clear()

  for line in code:
    m = re.match(r"^(.*?\S)\s{2,}#\s*(.*)$", line)
    if m is None:
      plain.append(line)
      continue
    flush()
    text = m.group(2).strip()
    out.extend([re.sub(r"`([A-Za-z_][\w.]*)`", link, text[0].upper() + text[1:] + ("" if text.endswith((".", ":")) else ":")), "",
                "```python", m.group(1), "```", ""])
  flush()
  return out


def docBody(doc, link):
  out, code = [], []
  for line in doc.split("\n"):
    if line.startswith("  ") or (code and not line.strip()):
      code.append(line[2:])
      continue
    if code:
      out += [""] + renderForms("\n".join(code).strip("\n").split("\n"), link)
      code = []
    out.append(re.sub(r"`([A-Za-z_][\w.]*)`", link, line))
  if code:
    out += [""] + renderForms("\n".join(code).strip("\n").split("\n"), link)
  return "\n".join(out).strip("\n")


# the sections as tables and lists: Input and Output name and meaning, Keywords with the
# default the signature gives, References a list, See also links
def renderSections(sections, target, link, seeAlso):
  out = []
  try:
    params = inspect.signature(target).parameters if target is not None else {}
  except (TypeError, ValueError):
    params = {}
  for head in ("Input", "Output"):
    if head in sections:
      out += [f"## {head}", "", "| | |", "|---|---|"]
      out += [f"| `{n}` | {cell(t, link)} |" for n, t in map(entry, sections[head])] + [""]
  if "Keywords" in sections:
    out += ["## Keywords", "", "| | default | |", "|---|---|---|"]
    for n, t in map(entry, sections["Keywords"]):
      p = params.get(n)
      d = "" if p is None or p.default is inspect.Parameter.empty else f"`{shownDefault(p.default)!r}`"
      out.append(f"| `{n}` | {d} | {cell(t, link)} |")
    out.append("")
  if "References" in sections:
    out += ["## References", ""] + [f"- {e}" for e in sections["References"]] + [""]
  if "See also" in sections:
    names = " ".join(sections["See also"]).replace(",", " ").split()
    out += ["## See also", "", ", ".join(seeAlso(x) for x in names), ""]
  return out


# the calling forms, the first block of indented lines after the summary, and the prose
# around them
def splitForms(rest):
  lines = rest.strip("\n").split("\n") if rest.strip() else []
  if not lines or not lines[0].startswith("  "):
    return [], rest
  k = 0
  while k < len(lines) and (lines[k].startswith("  ") or not lines[k].strip() and k + 1 < len(lines) and lines[k + 1].startswith("  ")):
    k += 1
  return [l[2:] for l in lines[:k] if l.strip()], "\n".join(lines[k:])


# text for a table cell: one line, the bars escaped, the names in backticks linked
def cell(text, link):
  return re.sub(r"`([A-Za-z_][\w.]*)`", link, text.replace("|", "\\|"))


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


def render(name, entry_, pages, calls, owners, names=None, docPages=()):
  kind, obj, cls, matlab = entry_
  member = name.split(".", 1)[1] if "." in name and kind in ("method", "property") else None

  def link(m):
    t = m.group(1)
    for cand in (t, f"{cls.__name__}.{t}" if cls else None, f"{t}_index"):
      if cand and cand in pages and cand != name:
        return f"[`{t}`]({cand}_py.html)"
    return m.group(0)

  # the name a function is called by: plotx2north is plottingConvention.default
  called = getattr(obj, "__name__", None) or renamed(name) or name
  # what plot hands a class to is called as plot
  if cls is not None and obj is plotImplementation(cls):
    called = "plot"
  title = f"The class {obj.__name__}" if kind == "class" else name
  doc = docOf(obj)
  lines = [f"# {title}", ""]
  if kind == "property":
    lines += [f"A property of [{cls.__name__}]({cls.__name__}_index_py.html).", ""]
  elif kind == "method":
    lines += [f"A method of [{cls.__name__}]({cls.__name__}_index_py.html).", ""]
  operator = OPERATOR_FORMS.get(getattr(unwrap(obj), "__name__", ""))
  if operator and not doc:
    doc = f"{operator[1]}, the operator {operator[0]}"
  main, sections = parseDoc(doc)
  summary, _, rest = main.partition("\n\n")
  if summary:
    lines += [docBody(summary, link), ""]
  target = None
  # the calling forms are the syntax; without forms the signature is, and it closes the page
  # when no Keywords table gives the defaults
  forms, prose = splitForms(rest)
  sig = None
  if kind in ("function", "method", "constructor"):
    shown = called if kind == "function" else f"{cls.__name__}.{member}" if member else cls.__name__
    # a class or static method as it is called, without cls
    target = obj if kind == "constructor" else obj.__get__(None, cls) if isinstance(obj, (classmethod, staticmethod)) else unwrap(obj)
    sig = operator[0] if operator else signature(shown, target)
  if forms:
    lines += ["## Syntax", ""] + renderForms(forms, link)
  elif sig:
    lines += ["## Syntax", "", "```python", sig, "```", ""]
    sig = None
  if prose.strip():
    lines += [docBody(prose, link), ""]

  def seeAlso(x):
    page = (names or {}).get(x) or (x if x in pages else None) or (f"{x}_index" if f"{x}_index" in pages else None)
    if page is None and x in docPages:
      page = x
    # a function of the flat namespace written under MATLAB's name: exportPoleFigure on PoleFigure.export
    if page is None and "." not in x and callable(getattr(mtex, x, None)):
      f = unwrap(getattr(mtex, x))
      page = next((p for p, e in pages.items() if unwrap(e[1]) is f), None)
    # a method named without its class, or by a parent class: the member of that name,
    # this page's class first
    if page is None:
      head, _, last = x.rpartition(".")
      found = sorted(p for p in pages if p.endswith("." + last) and p.startswith(head))
      own = [p for p in found if p.split(".")[0] == name.split(".")[0]]
      page = (own or found or [None])[0]
    return f"[{x}]({page}_py.html)" if page else x

  lines += renderSections(sections, target, link, seeAlso)
  if sig and "Keywords" not in sections:
    lines += ["## Signature", "", "```python", sig, "```", ""]
  if kind == "class":
    lines += members(obj, pages)
  # a forwarder of the flat namespace, norm(x) for x.norm(): the methods it reaches
  if hasattr(unwrap(obj), "forwardsTo"):
    lines += forwarded(unwrap(obj).forwardsTo, pages, names or {})
  # a call of plot does not say what it draws
  used = [] if called == "plot" else usedOn(member or called.split(".")[-1], cls if member else None, calls, owners)
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


# the classes whose method a forwarder calls, each with its page and summary
def forwarded(member, pages, names):
  rows = []
  for n in sorted(dir(mtex), key=str.lower):
    c = getattr(mtex, n)
    if inspect.isclass(c) and isOurs(c) and inspect.getattr_static(c, member, None) is not None:
      page = names.get(f"{n}.{member}")
      if page:
        rows.append(f"| [{n}]({page}_py.html) | {firstSentence(docOf(inspect.getattr_static(c, member)))} |")
  return ["## Methods it calls", "", "| | |", "|---|---|", *rows, ""] if rows else []


# the members of a class as a table of their summaries
def members(cls, pages):
  rows = []
  for n, entry in pages.items():
    if entry[2] is cls and entry[0] in ("method", "property", "function", "constructor") and n != f"{cls.__name__}_index":
      rows.append(f"| [{n.split('.', 1)[-1]}]({n}_py.html) | {firstSentence(docOf(entry[1]))} |")
  if not rows:
    return []
  return ["## Members", "", "| | |", "|---|---|", *sorted(rows, key=str.lower), ""]


# the layers in the order of the imports, as the index shows them
LAYERS = {"geometry": "Geometry", "functions": "Function spaces", "tensors": "Tensors", "maps": "Maps",
          "maps3d": "Maps of volumes", "analysis": "Analysis", "io": "Import and export", "plotting": "Plotting",
          "wizard": "Import wizard"}
INDEX = "FunctionReference"


# the index: per layer its classes and the functions of the flat namespace, each linked to its
# first page; MATLAB's FunctionReference is its twin
def indexPage(pages):
  first = {}
  for n, e in pages.items():
    if e[0] != "class":
      first.setdefault(id(unwrap(e[1])), n)
  groups = {}
  for n in sorted(dir(mtex), key=str.lower):
    obj = getattr(mtex, n)
    if n.startswith("_") or not callable(obj) or not isOurs(obj):
      continue
    target = f"{n}_index" if inspect.isclass(obj) else first.get(id(unwrap(obj)))
    if not target:
      continue
    layer = getattr(unwrap(obj), "__module__", "mtex.other").split(".")[1]
    doc = firstSentence(inspect.getdoc(unwrap(obj)) or "")
    groups.setdefault(layer, ([], []))[0 if inspect.isclass(obj) else 1].append(f"| [{n}]({target}_py.html) | {doc} |")
  lines = ["# Function reference", "",
           "Every class and function of mtex, by the layer it belongs to; a class page lists its methods and properties.", ""]
  for layer in sorted(groups, key=lambda k: list(LAYERS).index(k) if k in LAYERS else len(LAYERS)):
    classes, functions = groups[layer]
    lines += [f"## {LAYERS.get(layer, layer)}", ""]
    for title, rows in (("Classes", classes), ("Functions", functions)):
      if rows:
        lines += [f"### {title}", "", "| | |", "|---|---|", *rows, ""]
  return "\n".join(lines)


# the page of every name as Python spells it: a class, a function of the flat namespace, a
# member as Class.member; a page named after MATLAB's name serves the Python name it reaches
# whether a page name is also a Python spelling, a function or Class.member of the flat namespace
def lookupName(name):
  head, _, member = name.partition(".")
  obj = getattr(mtex, head, None)
  if not member:
    return callable(obj) and isOurs(obj) and not inspect.isclass(obj)
  return inspect.isclass(obj) and isOurs(obj) and inspect.getattr_static(obj, member, None) is not None


def pythonNames(pages):
  members = {}
  for n in dir(mtex):
    obj = getattr(mtex, n)
    if inspect.isclass(obj) and isOurs(obj):
      for m in dir(obj):
        if not m.startswith("_"):
          members.setdefault(id(unwrap(inspect.getattr_static(obj, m, None))), []).append(f"{n}.{m}")
  # a page under the Python name itself first, then the pages reaching it under MATLAB's
  found = {n: n for n in pages if pages[n][0] != "class" and (lookupName(n) or n == getattr(unwrap(pages[n][1]), "__name__", None))}
  for n, (kind, obj, cls, _) in pages.items():
    if kind == "class":
      found.setdefault(obj.__name__, f"{obj.__name__}_index" if f"{obj.__name__}_index" in pages else n)
      continue
    f = unwrap(obj)
    name = getattr(f, "__name__", "")
    if kind == "function" and name and not name.startswith("_") and getattr(mtex, name, None) is f:
      found.setdefault(name, n)
    # a function MATLAB names apart that Python reaches by a path: plotx2north is plottingConvention.default
    path = renamed(n)
    if kind == "function" and path and "." in path and lookup(path) is not None:
      found.setdefault(path, n)
    if cls is not None and kind != "constructor":
      for full in members.get(id(f), []):
        if full.split(".")[0] == cls.__name__:
          found.setdefault(full, n)
  return dict(sorted(found.items()))


def writeNames(pages):
  names = pythonNames(pages)
  text = json.dumps(names, indent=0, sort_keys=True) + "\n"
  if not os.path.exists(namesFile) or open(namesFile, encoding="utf-8").read() != text:
    with open(namesFile, "w", encoding="utf-8") as fh:
      fh.write(text)
  return names


def build(names=()):
  pages = catalogue()
  calls, owners = pageCalls(pageSources()), memberOwners()
  os.makedirs(outDir, exist_ok=True)
  chosen = [n for n in pages if not names or n in names]
  pyNames, docPages = pythonNames(pages), set(pageSources())
  for n in chosen:
    with open(os.path.join(outDir, n + ".md"), "w", encoding="utf-8") as fh:
      fh.write(render(n, pages[n], pages, calls, owners, pyNames, docPages))
  if not names or INDEX in names:
    chosen.append(INDEX)
    with open(os.path.join(outDir, INDEX + ".md"), "w", encoding="utf-8") as fh:
      fh.write(indexPage(pages))
  index = {n: {"kind": e[0], "matlab": e[3]} for n, e in pages.items()}
  index[INDEX] = {"kind": "index", "matlab": INDEX}
  if not names:
    writeNames(pages)
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
