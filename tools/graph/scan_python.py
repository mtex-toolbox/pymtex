# the static call graph of pymtex: module functions, classes, methods, tests, benchmarks
#
# Node ids: "py:mtex.functions.density.calcDensity" a function, "py:mtex.functions.so3fun.SO3FunHarmonic.eval"
# a method, "py:class:<module>.<Class>" a class, "py:dispatch:name" a method name several classes
# implement, reached on a receiver the scan cannot type, "py:module:<module>" a module. A node's
# "matlab" lists the MATLAB names its docstring cites, a module's those of its header comment.
from __future__ import annotations

import ast
import os
import re

ROOTS = ("mtex", "tests", "bench")
MATLAB_REF = re.compile(r"MATLAB[ '’s]*((?:@?[A-Za-z]\w*(?:[/.]\w+)?(?:,\s*|\s+and\s+|\s*))+)")


def moduleName(rel):
  parts = rel[:-3].split("/")
  if parts[-1] == "__init__":
    parts = parts[:-1]
  return ".".join(parts)


# the MATLAB names a comment or docstring cites: "MATLAB calcDensity, calcPDF" or "MATLAB @SO3FunHarmonic"
def matlabRefs(text):
  out = []
  for m in MATLAB_REF.finditer(text or ""):
    for w in re.split(r",\s*|\s+and\s+|\s+", m.group(1)):
      w = w.strip().rstrip(";.)")
      if re.fullmatch(r"@?[A-Za-z]\w*(?:[/.]\w+)?", w) and w not in ("does", "is", "the", "as", "has", "and", "gives", "keeps"):
        out.append(w)
      else:
        break
  return out


def headerComment(src):
  lines = []
  for line in src.split("\n"):
    if line.startswith("#"):
      lines.append(line[1:])
    elif line.strip():
      break
  return " ".join(lines)


class Module:
  def __init__(self, name, rel, tree, src, isPackage):
    self.name, self.rel, self.tree, self.src, self.isPackage = name, rel, tree, src, isPackage
    self.defs, self.imports, self.stars, self.verbs = {}, {}, [], {}

  def resolveRelative(self, level, mod):
    base = self.name.split(".")
    if not self.isPackage:
      base = base[:-1]
    base = base[:len(base) - (level - 1)] if level > 1 else base
    return ".".join(base + ([mod] if mod else []))


def scan(root):
  mods = {}
  for top in ROOTS:
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, top)):
      dirnames[:] = sorted(d for d in dirnames if d not in ("__pycache__", "out", "fixtures", "data"))
      for f in sorted(filenames):
        if not f.endswith(".py"):
          continue
        rel = os.path.relpath(os.path.join(dirpath, f), root)
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        try:
          tree = ast.parse(src)
        except SyntaxError:
          continue
        m = Module(moduleName(rel), rel, tree, src, f == "__init__.py")
        mods[m.name] = m
  for m in mods.values():
    for st in m.tree.body:
      if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        m.defs[st.name] = st
      # a verb made by funcs._method: angle(o1, o2) is the method o1.angle(o2)
      elif isinstance(st, ast.Assign) and isinstance(st.value, ast.Call) and ast.unparse(st.value.func) == "_method" \
          and st.value.args and isinstance(st.value.args[0], ast.Constant):
        for t in st.targets:
          if isinstance(t, ast.Name):
            m.verbs[t.id] = st.value.args[0].value
      elif isinstance(st, ast.ImportFrom):
        target = m.resolveRelative(st.level, st.module) if st.level else (st.module or "")
        for a in st.names:
          if a.name == "*":
            m.stars.append(target)
          else:
            m.imports[a.asname or a.name] = (target, a.name)
      elif isinstance(st, ast.Import):
        for a in st.names:
          m.imports[a.asname or a.name.split(".")[0]] = (a.name if a.asname else a.name.split(".")[0], None)
  return mods


class Graph:
  def __init__(self, root):
    self.root = root
    self.mods = scan(root)
    self.nodes, self.edges = {}, {}
    self.classes = {}
    self.methods = {}

  # the fully qualified definition a name in a module stands for: ("def", module, name) or ("module", name)
  def resolve(self, modName, name, seen=None):
    seen = seen or set()
    if (modName, name) in seen or modName not in self.mods:
      return ("module", f"{modName}.{name}") if f"{modName}.{name}" in self.mods else None
    seen.add((modName, name))
    m = self.mods[modName]
    if name in m.defs:
      return ("def", modName, name)
    if name in m.verbs:
      return ("verb", m.verbs[name])
    if name in m.imports:
      target, attr = m.imports[name]
      if attr is None:
        return ("module", target) if target in self.mods else None
      if f"{target}.{attr}" in self.mods:
        return ("module", f"{target}.{attr}")
      return self.resolve(target, attr, seen)
    for s in m.stars:
      r = self.resolve(s, name, seen)
      if r:
        return r
    if f"{modName}.{name}" in self.mods:
      return ("module", f"{modName}.{name}")
    return None

  def nodeOf(self, r):
    if not r or r[0] != "def":
      return None
    _, mod, name = r
    d = self.mods[mod].defs[name]
    if isinstance(d, ast.ClassDef):
      init = f"py:{mod}.{name}.__init__"
      return init if init in self.nodes else f"py:class:{mod}.{name}"
    return f"py:{mod}.{name}"

  def kindOf(self, mod, fname):
    if mod.startswith("tests.") and fname.startswith("test"):
      return "test"
    if mod.startswith("bench."):
      return "bench"
    return "function"

  def collect(self):
    header = {}
    for mn, m in self.mods.items():
      header[mn] = matlabRefs(headerComment(m.src))
      if mn.startswith("mtex"):
        self.nodes[f"py:module:{mn}"] = dict(id=f"py:module:{mn}", port="python", kind="module", name=mn.split(".")[-1],
                                             file=m.rel, line=1, matlab=header[mn])
      for name, d in m.defs.items():
        if isinstance(d, ast.ClassDef):
          cid = f"py:class:{mn}.{name}"
          bases = [ast.unparse(b) for b in d.bases]
          self.classes[cid] = dict(mod=mn, name=name, bases=bases, methods={})
          self.nodes[cid] = dict(id=cid, port="python", kind="class", name=name, cls=name, file=m.rel,
                                 line=d.lineno, supers=bases, matlab=matlabRefs(ast.get_docstring(d)))
          for st in d.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
              mid = f"py:{mn}.{name}.{st.name}"
              self.classes[cid]["methods"][st.name] = mid
              self.methods.setdefault(st.name, set()).add(mid)
              decs = [ast.unparse(x) for x in st.decorator_list]
              role = "getter" if "property" in decs or "classproperty" in decs else \
                     "setter" if any(x.endswith(".setter") for x in decs) else \
                     "static" if {"staticmethod", "classmethod"} & set(decs) else None
              # a property's getter and setter are one node holding both bodies
              if mid in self.nodes:
                old = self.nodes[mid]
                old["roles"] = sorted(set(old.get("roles", [])) | {role} - {None})
                old["lines"] += st.end_lineno - st.lineno + 1
                old["_ast"] = ast.Module(body=[old["_ast"], st], type_ignores=[])
                continue
              self.nodes[mid] = dict(id=mid, port="python", kind="method", name=st.name, cls=name, file=m.rel,
                                     line=st.lineno, lines=st.end_lineno - st.lineno + 1, roles=[role] if role else [],
                                     matlab=matlabRefs(ast.get_docstring(st)), _ast=st, _mod=mn, _cls=cid)
        else:
          fid = f"py:{mn}.{name}"
          self.nodes[fid] = dict(id=fid, port="python", kind=self.kindOf(mn, name), name=name, file=m.rel,
                                 line=d.lineno, lines=d.end_lineno - d.lineno + 1,
                                 matlab=matlabRefs(ast.get_docstring(d)), _ast=d, _mod=mn)
      # a benchmark or test script's top-level code is one node
      if mn.startswith(("bench.", "tests.")):
        sid = f"py:{mn}"
        body = [st for st in m.tree.body if not isinstance(st, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
        self.nodes[sid] = dict(id=sid, port="python", kind="bench" if mn.startswith("bench.") else "script",
                               name=mn.split(".")[-1], file=m.rel, line=1, matlab=header[mn],
                               _ast=ast.Module(body=body, type_ignores=[]), _mod=mn)

  # the method a class reaches under a name, walking its bases by name
  def lookup(self, cid, name, seen=None):
    seen = seen or set()
    if cid in seen or cid not in self.classes:
      return None
    seen.add(cid)
    c = self.classes[cid]
    if name in c["methods"]:
      return c["methods"][name]
    for b in c["bases"]:
      r = self.resolve(c["mod"], b.split(".")[-1])
      if r and r[0] == "def":
        hit = self.lookup(f"py:class:{r[1]}.{r[2]}", name, seen)
        if hit:
          return hit
    return None

  def addEdge(self, src, dst, res):
    if src != dst and dst:
      if (src, dst) not in self.edges or self.edges[(src, dst)]["resolution"] == "dispatch":
        self.edges[(src, dst)] = dict(src=src, dst=dst, kind="call", resolution=res)

  def dispatch(self, name):
    imps = self.methods.get(name, set())
    if len(imps) == 1:
      return next(iter(imps)), "method"
    did = f"py:dispatch:{name}"
    if did not in self.nodes:
      self.nodes[did] = dict(id=did, port="python", kind="dispatch", name=name)
      for mid in imps:
        self.edges[(did, mid)] = dict(src=did, dst=mid, kind="dispatch", resolution="dispatch")
    return did, "dispatch"

  # the definitions the names and attributes under some nodes of a body reach: (node, resolution)
  def refs(self, roots, mod, cid, local):
    for root in roots:
      for x in ast.walk(root):
        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id not in local:
          r = self.resolve(mod, x.id)
          if r and r[0] == "verb":
            if r[1] in self.methods:
              yield self.dispatch(r[1])
          else:
            yield self.nodeOf(r), "import"
        elif isinstance(x, ast.Attribute) and isinstance(x.ctx, ast.Load):
          v = x.value
          if isinstance(v, ast.Name) and v.id in ("self", "cls") and cid:
            yield self.lookup(cid, x.attr), "class"
          elif isinstance(v, ast.Name) and v.id not in local:
            r = self.resolve(mod, v.id)
            if r and r[0] == "module":
              yield self.nodeOf(self.resolve(r[1], x.attr)), "module"
            elif r and r[0] == "def" and isinstance(self.mods[r[1]].defs[r[2]], ast.ClassDef):
              yield self.lookup(f"py:class:{r[1]}.{r[2]}", x.attr), "static"
            elif x.attr in self.methods and not (r is None and v.id in self.mods[mod].imports):
              yield self.dispatch(x.attr)
          elif x.attr in self.methods and not x.attr.startswith("__"):
            yield self.dispatch(x.attr)

  # what a test asserts on: its assertions and the last assignment of each name they read
  def subject(self, tree):
    checks = []
    for x in ast.walk(tree):
      if isinstance(x, ast.Assert):
        checks.append(x.test)
      elif isinstance(x, ast.Call):
        f = ast.unparse(x.func)
        if "testing." in f or f.split(".")[-1] in ("close", "approx") or f.split(".")[-1].startswith("assert"):
          checks += x.args
      elif isinstance(x, ast.With) and any("raises" in ast.unparse(i.context_expr) or "warns" in ast.unparse(i.context_expr)
                                           for i in x.items):
        checks += x.body
    assigned = {}
    for x in ast.walk(tree):
      if isinstance(x, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and x.value is not None:
        for t in (x.targets if isinstance(x, ast.Assign) else [x.target]):
          for nm in ast.walk(t):
            if isinstance(nm, ast.Name):
              assigned[nm.id] = x.value
    read = {nm.id for c in checks for nm in ast.walk(c) if isinstance(nm, ast.Name) and isinstance(nm.ctx, ast.Load)}
    return checks + [assigned[k] for k in read if k in assigned]

  def link(self):
    for nid, n in list(self.nodes.items()):
      tree = n.get("_ast")
      if tree is None:
        continue
      mod, cid = n["_mod"], n.get("_cls")
      local = {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
      local |= {t.id for t in ast.walk(tree) if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store)}
      for dst, res in self.refs([tree], mod, cid, local):
        self.addEdge(nid, dst, res)
      # a test's subject: the definitions its assertions reach
      if n["kind"] == "test":
        for dst, res in self.refs(self.subject(tree), mod, cid, local):
          if dst and dst.startswith(("py:mtex", "py:class:mtex", "py:dispatch:")):
            self.edges[(nid, dst, "asserts")] = dict(src=nid, dst=dst, kind="asserts", resolution=res)
    for cid, c in self.classes.items():
      for mid in c["methods"].values():
        self.edges[(cid, mid)] = dict(src=cid, dst=mid, kind="member", resolution="exact")
      for b in c["bases"]:
        r = self.resolve(c["mod"], b.split(".")[-1])
        if r and r[0] == "def":
          self.edges[(cid, f"py:class:{r[1]}.{r[2]}")] = dict(src=cid, dst=f"py:class:{r[1]}.{r[2]}", kind="inherits", resolution="exact")

  def build(self):
    self.collect()
    self.link()
    nodes = [{k: v for k, v in n.items() if not k.startswith("_")} for n in self.nodes.values()]
    return nodes, [e for e in self.edges.values() if e["dst"] in self.nodes]


def build(root):
  return Graph(root).build()


if __name__ == "__main__":
  import json
  import sys
  import time
  t = time.perf_counter()
  n, e = build(sys.argv[1])
  print(f"{len(n)} nodes, {len(e)} edges in {time.perf_counter() - t:.1f} s", file=sys.stderr)
  json.dump(dict(nodes=n, edges=e), sys.stdout)
