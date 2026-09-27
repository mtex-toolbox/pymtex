# the package as an editor reads it without running it: Pylance, pyright and Jedi see only
# what the source binds, so the names the layers assemble in loops, the methods other modules
# attach with install() and the help of a function must be written out. This tool writes them:
#
#   inits         every layer's __init__.py imports its modules' __all__ by name
#   declarations  a class body declares, under TYPE_CHECKING, the methods attached to it
#   docstrings    the comment above a public def or class becomes its docstring
#
#   .venv/bin/python tools/staticapi.py          rewrite the files
#   .venv/bin/python tools/staticapi.py --check  list what is out of date, as the test does
from __future__ import annotations

import ast
import functools
import importlib
import inspect
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(ROOT, "mtex")
LAYERS = ("geometry", "maps", "maps3d", "functions", "tensors", "analysis", "io", "plotting")
WIDTH = 120
MARKER = "# declared for the editor: the methods other modules attach (tools/staticapi.py)"


# ------------------------------------------------------------------ helpers

def read(path):
  with open(path, encoding="utf-8") as f:
    return f.read()


def modulePath(name):
  path = os.path.join(ROOT, *name.split("."))
  return path + ".py" if os.path.isfile(path + ".py") else os.path.join(path, "__init__.py")


# names joined by commas into lines of at most WIDTH characters, the first after head, the
# others indented under the first name
def wrapped(head, names, tail, indent=""):
  lines, line = [], indent + head
  for k, n in enumerate(names):
    piece = n + (", " if k < len(names) - 1 else tail)
    if len(line) + len(piece.rstrip()) > WIDTH and line.strip() != head.strip():
      lines.append(line.rstrip())
      line = indent + " " * len(head) + piece
    else:
      line += piece
  lines.append(line)
  return lines


# the syntax tree of a module as it is on disk
@functools.cache
def moduleTree(module):
  return ast.parse(read(modulePath(module)))


# the names a module binds at its top level, as a reader of the source sees them
@functools.cache
def moduleNames(module):
  return staticNames(moduleTree(module))


def staticNames(tree):
  names = set()
  for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
      names.add(node.name)
    elif isinstance(node, ast.Assign):
      for t in node.targets:
        names |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
      names.add(node.target.id)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
      names |= {(a.asname or a.name).split(".")[0] for a in node.names}
  return names


# the relative module path from the module src to the module dst
def relative(src, dst):
  pkg, target = src.split(".")[:-1], dst.split(".")
  common = 0
  while common < min(len(pkg), len(target)) and pkg[common] == target[common]:
    common += 1
  return "." * (len(pkg) - common + 1) + ".".join(target[common:])


# ------------------------------------------------------------------ inits

# the modules a layer's __init__.py imports, in order: the names of its loop on first use,
# its relative imports after
def layerModules(tree):
  for node in tree.body:
    if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
      return [e.value for e in node.iter.elts]
  return [node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module]


# a layer's __init__.py: its heading comments, one import per module of the names its
# __all__ lists, __all__ written out, then whatever the file holds after
def initSource(layer):
  path = os.path.join(PACKAGE, layer, "__init__.py")
  src = read(path)
  tree = ast.parse(src)
  lines = src.split("\n")
  modules = layerModules(tree)
  head = []
  for line in lines:
    if not line.startswith("#"):
      break
    head.append(line)
  # the statements the imports replace: the loop, __all__, importlib and the imports
  generated = [n for n in tree.body if isinstance(n, (ast.For, ast.ImportFrom, ast.Import))
               or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets))]
  generated = [n for n in generated if not (isinstance(n, ast.ImportFrom) and n.level == 0 and n.module != "importlib")]
  end = max(n.end_lineno for n in generated)
  tail = "\n".join(lines[end:]).strip("\n")
  exported = {}
  for m in modules:
    for name in importlib.import_module(f"mtex.{layer}.{m}").__all__:
      exported[name] = m
  order = list(dict.fromkeys(n for m in modules for n in importlib.import_module(f"mtex.{layer}.{m}").__all__))
  out = head[:]
  for m in modules:
    names = [n for n in importlib.import_module(f"mtex.{layer}.{m}").__all__ if exported[n] == m]
    names = list(dict.fromkeys(names))
    if names:
      out += wrapped(f"from .{m} import (", names, ")") if len(f"from .{m} import " + ", ".join(names)) > WIDTH else [f"from .{m} import " + ", ".join(names)]
    else:
      out.append(f"from .{m} import *  # noqa: F401,F403")
  out.append("")
  out += wrapped("__all__ = [", [f'"{n}"' for n in order], "]")
  if tail:
    if "_importlib" in tail:
      out.insert(len(head), "import importlib as _importlib")
    out += ["", "", tail] if tail.startswith(("def ", "#")) else ["", tail]
  return path, "\n".join(out) + "\n"


# ------------------------------------------------------------------ declarations

# a default the stub can write as it is: None, a bool, an integer, a string, a finite float
def _plain(value):
  if value is None or isinstance(value, (bool, int, str)):
    return True
  return isinstance(value, float) and value == value and abs(value) != float("inf")


# a docstring at an indent, a raw one when it holds a backslash
def docLines(doc, indent):
  quote = 'r"""' if "\\" in doc else '"""'
  text = doc.split("\n")
  out = [indent + quote + text[0]] + [(indent + t) if t else "" for t in text[1:]]
  out[-1] += '"""'
  return out


# a def standing for a function the editor cannot import: its parameters with their literal
# defaults, Any for the others, and its docstring
def stub(name, f, decorator, indent):
  try:
    sig = inspect.signature(f)
  except (TypeError, ValueError):
    sig = None
  params = []
  if sig is not None:
    star = False
    ps = list(sig.parameters.values())
    for k, p in enumerate(ps):
      if k and ps[k - 1].kind == p.POSITIONAL_ONLY and p.kind != p.POSITIONAL_ONLY:
        params.append("/")
      if p.kind == p.VAR_POSITIONAL:
        params.append("*" + p.name)
        star = True
        continue
      if p.kind == p.VAR_KEYWORD:
        params.append("**" + p.name)
        continue
      if p.kind == p.KEYWORD_ONLY and not star:
        params.append("*")
        star = True
      if p.default is p.empty:
        params.append(p.name)
      elif _plain(p.default):
        params.append(f"{p.name}={p.default!r}")
      else:
        params.append(f"{p.name}: Any = ...")
    if ps and ps[-1].kind == ps[-1].POSITIONAL_ONLY:
      params.append("/")
  else:
    params = ["*args", "**kwargs"]
  # the object comes first under the name an editor expects of a method
  first = {None: "self", "property": "self", "classmethod": "cls"}.get(decorator)
  if first and params and not params[0].startswith(("*", "/")):
    params[0] = first
  elif first:
    params.insert(0, first)
  out = [f"{indent}@{decorator}"] if decorator else []
  out.append(f"{indent}def {name}({', '.join(params)}) -> Any:")
  doc = inspect.getdoc(f)
  out += docLines(doc, indent + "  ") if doc and '"""' not in doc and not doc.endswith('"') else [f"{indent}  ..."]
  return out


# what a class body declares for one attachment: an import of the function under the name it
# is attached as, wrapped when it is a static or class method; a stub when no import reaches it
def declaration(cls, name, attr, indent):
  kind = "property" if isinstance(attr, property) else "staticmethod" if isinstance(attr, staticmethod) else "classmethod" if isinstance(attr, classmethod) else None
  f = attr.fget if kind == "property" else getattr(attr, "__func__", attr)
  f = getattr(f, "__wrapped__", f)
  module, fname = getattr(f, "__module__", None), getattr(f, "__name__", "")
  reachable = (module and module.startswith("mtex") and module != cls.__module__ and "<" not in f.__qualname__
               and f in (g := getattr(sys.modules[module], fname, None), getattr(g, "__wrapped__", None))
               and fname in moduleNames(module))
  if kind == "property" or not reachable:
    lines = stub(name, f, kind, indent)
    if kind == "property" and attr.fset is not None:
      lines += [f"{indent}@{name}.setter", f"{indent}def {name}(self, value) -> None:", f"{indent}  ..."]
    return [], lines
  src = relative(cls.__module__, module)
  if kind is None:
    return [f"from {src} import {fname}" + (f" as {name}" if fname != name else "")], []
  return [f"from {src} import {fname}"], [f"{indent}{name} = {kind}({fname})"]


# the definition of a class in its module
def classNode(cls):
  return next(n for n in ast.walk(moduleTree(cls.__module__)) if isinstance(n, ast.ClassDef) and n.name == cls.__name__)


# the classes with attachments, each with the block its body ends with
def declarationBlocks():
  import mtex  # noqa: F401
  from mtex.geometry.install import INSTALLED
  blocks = {}
  for cls, name, attr, declared in INSTALLED:
    if cls.__module__.startswith("mtex."):
      blocks.setdefault(cls, {})[name] = attr if declared is None else declared
  out = {}
  for cls, attrs in blocks.items():
    imports, body = [], []
    node = classNode(cls)
    own = {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    own |= {t.id for n in node.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    attrs = {n: a for n, a in attrs.items() if n not in own}
    if not attrs:
      continue
    for name, attr in attrs.items():
      i, b = declaration(cls, name, attr, "    ")
      imports += i
      body += b
    lines = ["  " + MARKER, "  if TYPE_CHECKING:"]
    if any("Any" in line for line in body):
      lines.append("    from typing import Any")
    grouped = {}
    for i in dict.fromkeys(imports):
      src, what = i.removeprefix("from ").split(" import ")
      grouped.setdefault(src, []).append(what)
    for src, names in grouped.items():
      head = f"from {src} import "
      lines += wrapped(head + "(", names, ")", "    ") if len(head + ", ".join(names)) + 4 > WIDTH else ["    " + head + ", ".join(names)]
    lines += body
    out[cls] = lines
  return out


# a module's source with its classes' blocks replaced, TYPE_CHECKING imported
def declaredSource(module, classes):
  path = modulePath(module)
  lines = read(path).split("\n")
  # the old blocks go first, from the bottom: the marker and the if statement after it
  tree = ast.parse("\n".join(lines))
  old = [n for n in ast.walk(tree) if isinstance(n, ast.If) and n.lineno >= 2 and lines[n.lineno - 2].strip() == MARKER]
  for n in sorted(old, key=lambda n: -n.lineno):
    start = n.lineno - 2
    start -= 1 if start > 0 and not lines[start - 1].strip() else 0
    del lines[start:n.end_lineno]
  tree = ast.parse("\n".join(lines))
  byName = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
  for cls, block in sorted(classes.items(), key=lambda cb: -byName[cb[0].__name__].end_lineno):
    node = byName[cls.__name__]
    lines[node.end_lineno:node.end_lineno] = [""] + block
  src = "\n".join(lines)
  if classes and not re.search(r"^from typing import .*\bTYPE_CHECKING\b", src, re.M):
    tree = ast.parse(src)
    imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and not (isinstance(n, ast.ImportFrom) and n.module == "__future__")]
    at = imports[0].lineno - 1 if imports else next(n.lineno - 1 for n in tree.body if not isinstance(n, ast.Expr))
    lines = src.split("\n")
    lines.insert(at, "from typing import TYPE_CHECKING")
    src = "\n".join(lines)
  return path, src


# ------------------------------------------------------------------ docstrings

# the functions attached under a public name, which want a docstring whatever their own name
def installedFunctions():
  import mtex  # noqa: F401
  from mtex.geometry.install import INSTALLED
  out = set()
  for _, name, attr, declared in INSTALLED:
    attr = attr if declared is None else declared
    f = attr.fget if isinstance(attr, property) else getattr(attr, "__func__", attr)
    if not name.startswith("_") and getattr(f, "__module__", "").startswith("mtex"):
      out.add((f.__module__, f.__qualname__))
  return out


def _isBanner(text):
  return "----" in text or "====" in text


# a source with the comment above every public def and class moved into its docstring
def docstringSource(src, module, installed):
  skipped = set()
  while True:
    lines = src.split("\n")
    targets = []

    def visit(body, prefix):
      for node in body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
          continue
        qual = prefix + node.name
        public = not node.name.startswith("_") or (module, qual) in installed
        if public and ast.get_docstring(node) is None and (qual, node.lineno) not in skipped:
          targets.append((qual, node))
        if isinstance(node, ast.ClassDef):
          visit(node.body, qual + ".")

    visit(ast.parse(src).body, "")
    # from the bottom, the source parsed anew after each move, since a move shifts the lines
    # above the ones it moves into
    for qual, node in sorted(targets, key=lambda t: -t[1].lineno):
      new = _moveComment(lines, node)
      if new is not None:
        src = "\n".join(new)
        break
      skipped.add((qual, node.lineno))
    else:
      return src


# the lines with the comment above node moved into its docstring, None when there is none
def _moveComment(lines, node):
  top = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
  indent = lines[node.lineno - 1][:len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())]
  k = top
  while k > 0 and lines[k - 1].startswith(indent + "#") and not lines[k - 1].startswith(indent + "#!"):
    k -= 1
  comment = lines[k:top]
  if not comment or any(_isBanner(c) for c in comment):
    return None
  doc = "\n".join(c[len(indent) + 1:].removeprefix(" ") for c in comment).strip()
  first = node.body[0]
  firstLine = min([d.lineno for d in getattr(first, "decorator_list", [])] + [first.lineno])
  if firstLine == node.lineno or '"""' in doc or doc.endswith('"') or not doc:
    return None
  # the header ends on the last line above the body that is neither blank nor a comment
  h = firstLine - 1
  while h > node.lineno and (not lines[h - 1].strip() or lines[h - 1].lstrip().startswith("#")):
    h -= 1
  bodyIndent = lines[firstLine - 1][:len(lines[firstLine - 1]) - len(lines[firstLine - 1].lstrip())]
  lines = lines[:]
  lines[h:h] = docLines(doc, bodyIndent)
  del lines[k:top]
  return lines


def packageFiles():
  for d, _, files in os.walk(PACKAGE):
    for f in sorted(files):
      if f.endswith(".py"):
        path = os.path.join(d, f)
        rel = os.path.relpath(path, ROOT)[:-3].replace(os.sep, ".")
        yield path, rel.removesuffix(".__init__")


# ------------------------------------------------------------------ the run

# every file the three steps would change, with its new content
def changes():
  sys.path.insert(0, ROOT)
  out = {}
  for layer in LAYERS:
    path, src = initSource(layer)
    out[path] = src
  byModule = {}
  for cls, block in declarationBlocks().items():
    byModule.setdefault(cls.__module__, {})[cls] = block
  for module, classes in byModule.items():
    path, src = declaredSource(module, classes)
    out[path] = src
  installed = installedFunctions()
  for path, module in packageFiles():
    out[path] = docstringSource(out.get(path) or read(path), module, installed)
  for p, s in out.items():
    compile(s, p, "exec")
  return {p: s for p, s in out.items() if s != read(p)}


# the public class attributes set at run time that neither the class body nor install() declares
def undeclared():
  import mtex
  from mtex.geometry.install import INSTALLED
  recorded = {(c, n) for c, n, *_ in INSTALLED}
  out = []
  for name in mtex.__all__:
    cls = getattr(mtex, name, None)
    if not isinstance(cls, type) or not cls.__module__.startswith("mtex"):
      continue
    try:
      node = classNode(cls)
    except StopIteration:
      continue
    bound = {n.name for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    bound |= {x.id for n in ast.walk(node) if isinstance(n, ast.Assign) for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)}
    bound |= {n.target.id for n in ast.walk(node) if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    for attr in vars(cls):
      if not attr.startswith("_") and attr not in bound and (cls, attr) not in recorded:
        out.append(f"{cls.__module__}.{cls.__name__}.{attr}")
  return out


if __name__ == "__main__":
  todo = changes()
  if "--check" in sys.argv:
    for p in sorted(todo):
      print("out of date:", os.path.relpath(p, ROOT))
    for a in undeclared():
      print("attached without install():", a)
    sys.exit(1 if todo else 0)
  for p, s in todo.items():
    with open(p, "w", encoding="utf-8") as f:
      f.write(s)
  print(f"{len(todo)} files written")
