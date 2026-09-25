# the static call graph of an MTEX MATLAB tree: functions, class methods, tests, doc pages
#
# Node ids: "m:name" a function on the path, "m:@Class.method" a method, "m:<dir>/private/name"
# a private function, "m:<parent>>sub" a local function, "m:dispatch:name" a method name that
# several classes implement and whose receiver the scan cannot tell apart.
from __future__ import annotations

import os
import re

SKIP = {".git", ".claude", "extern", "obsolete", "old", "compatibility", "userScripts",
        "templates", "docker", "data", "mex", ".github", "OldClasses", "tmp"}
KEYWORDS = set("""break case catch classdef continue else elseif end for function global if
otherwise parfor persistent return spmd switch try while methods properties events
enumeration arguments true false""".split())
IDENT = re.compile(r"(?<![\w.])(\.?)([A-Za-z]\w*)(?:\s*\.\s*([A-Za-z]\w*))?")
FUNC = re.compile(r"^\s*function\b\s*(?:(\[[^\]]*\]|[\w]+)\s*=\s*)?([\w.]+)\s*(?:\(([^)]*)\))?")
LHS = re.compile(r"^\s*(\[[^\]=]*\]|[A-Za-z]\w*(?:\s*[({][^=]*?[)}])?(?:\.\w+)*)\s*=(?!=)")


# the code of one line with comments and string literals removed, and the string literals
def stripLine(line):
  code, strs, i, n = [], [], 0, len(line)
  while i < n:
    c = line[i]
    if c == "%" or line.startswith("...", i):
      break
    # a quote right after a value is the transpose, anywhere else it opens a string
    if c == '"' or (c == "'" and not (code and re.match(r"[\w)\]}.']", code[-1]))):
      j = i + 1
      while j < n and not (line[j] == c and not line.startswith(c * 2, j)):
        j += 2 if line.startswith(c * 2, j) else 1
      strs.append(line[i + 1:j])
      code.append(" ")
      i = j + 1
      continue
    code.append(c)
    i += 1
  return "".join(code), strs


# the logical lines of a file, comments and strings stripped, with their line numbers
def codeLines(text):
  out, block = [], 0
  for k, line in enumerate(text.split("\n"), 1):
    s = line.strip()
    if s == "%{":
      block += 1
      continue
    if s == "%}" and block:
      block -= 1
      continue
    if block:
      continue
    code, strs = stripLine(line)
    out.append((k, code, strs))
  return out


def names(s):
  return [w for w in re.findall(r"[A-Za-z]\w*", s) if w not in KEYWORDS]


class Unit:
  # one function body: its id, where it is, what it may call
  def __init__(self, id, name, file, line, kind, cls=None, owner=None):
    self.id, self.name, self.file, self.line, self.kind, self.cls, self.owner = id, name, file, line, kind, cls, owner
    self.vars, self.refs, self.members, self.strings, self.lines = set(), [], [], [], 0


# the functions of one file, the first one the file's own, the rest local
def splitFile(rel, text, kind, cls):
  lines = codeLines(text)
  base = os.path.splitext(os.path.basename(rel))[0]
  units, cur, inClassdef, depthMethods = [], None, False, False
  for k, code, strs in lines:
    if re.match(r"^\s*classdef\b", code):
      inClassdef = True
      continue
    m = FUNC.match(code)
    if m:
      outs, fname, args = m.group(1) or "", m.group(2), m.group(3) or ""
      if inClassdef:
        uid = f"m:@{cls}.{fname}"
        u = Unit(uid, fname, rel, k, "method", cls=cls)
      elif not units:
        u = Unit(topId(rel, base, cls, kind), fname, rel, k, kind if kind != "method" else "method", cls=cls)
      else:
        u = Unit(f"{units[0].id}>{fname}", fname, rel, k, "local", cls=cls, owner=units[0].id)
      u.vars |= set(names(outs)) | set(names(args))
      units.append(u)
      cur = u
      continue
    if cur is None:
      if not units and not inClassdef and kind in ("test", "doc", "script"):
        cur = Unit(topId(rel, base, cls, kind), base, rel, k, kind)
        units.append(cur)
      else:
        continue
    cur.lines += 1
    lm = LHS.match(code)
    if lm:
      cur.vars |= {w for w in re.findall(r"(?<![\w.])([A-Za-z]\w*)", lm.group(1).split("(")[0].split("{")[0] if not lm.group(1).startswith("[") else lm.group(1))}
    fm = re.match(r"^\s*(?:par)?for\s+([A-Za-z]\w*)", code)
    if fm:
      cur.vars.add(fm.group(1))
    gm = re.match(r"^\s*(?:global|persistent)\s+(.*)", code)
    if gm:
      cur.vars |= set(names(gm.group(1)))
    body = code if not lm else code[lm.end():]
    for dot, a, b in IDENT.findall(body):
      if dot or a in KEYWORDS:
        continue
      cur.refs.append((a, b, k))
    for b in re.findall(r"\.\s*([A-Za-z]\w*)\s*\(", code):
      cur.members.append((b, k))
    cur.strings.extend(strs)
  return units


def topId(rel, base, cls, kind):
  d = os.path.dirname(rel)
  if kind == "method":
    return f"m:@{cls}.{base}"
  if "/private" in "/" + d or d.endswith("private"):
    return f"m:{d}/{base}"
  if kind in ("test", "doc", "script"):
    return f"m:{rel[:-2]}"
  return f"m:{base}"


def classOf(rel):
  for part in rel.split("/"):
    if part.startswith("@"):
      return part[1:]
  return None


def kindOf(rel, text):
  top = rel.split("/")[0]
  if top == "tests":
    return "test" if os.path.basename(rel).startswith(("check_", "benchmark_")) else "function"
  if top == "doc":
    return "doc"
  if classOf(rel) and "/private/" not in rel:
    return "method"
  if not re.search(r"^\s*(function|classdef)\b", text, re.M):
    return "script"
  return "function"


def scan(root):
  units, classes = [], {}
  for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames if d not in SKIP)
    for f in sorted(filenames):
      if not f.endswith(".m"):
        continue
      path = os.path.join(dirpath, f)
      rel = os.path.relpath(path, root)
      text = open(path, encoding="utf-8", errors="replace").read()
      cm = re.search(r"^\s*classdef\b(?:\s*\([^)]*\))?\s*(\w+)\s*(?:<\s*([^\n%]*))?", text, re.M)
      cls = classOf(rel)
      if cm:
        cls = cm.group(1)
        supers = [s.strip() for s in (cm.group(2) or "").split("&") if s.strip()]
        classes.setdefault(cls, {"file": rel, "supers": [], "methods": {}})
        classes[cls].update(file=rel, supers=[s for s in supers if re.match(r"^\w+$", s)])
      kind = "method" if cm else kindOf(rel, text)
      us = splitFile(rel, text, kind, cls)
      if cm and us:
        # the constructor keeps the class name, the class file itself is no unit
        pass
      units.extend(us)
  for u in units:
    if u.kind == "method" and u.cls:
      classes.setdefault(u.cls, {"file": None, "supers": [], "methods": {}})
      classes[u.cls]["methods"].setdefault(u.name, u.id)
  return units, classes


# the method a class reaches under a name, walking its superclasses
def lookup(classes, cls, name, seen=None):
  seen = seen or set()
  if cls in seen or cls not in classes:
    return None
  seen.add(cls)
  c = classes[cls]
  if name in c["methods"]:
    return c["methods"][name]
  for s in c["supers"]:
    r = lookup(classes, s, name, seen)
    if r:
      return r
  return None


def build(root):
  units, classes = scan(root)
  byId = {u.id: u for u in units}
  functions = {}
  privates = {}
  locals_ = {}
  for u in units:
    if u.kind == "local":
      locals_[(u.owner, u.name)] = u.id
    elif "/private/" in u.id or u.id.startswith("m:private/"):
      privates[(os.path.dirname(u.file), u.name)] = u.id
    elif u.kind in ("function",) and not u.cls:
      functions.setdefault(u.name, u.id)
  implementors = {}
  for c, d in classes.items():
    for name, mid in d["methods"].items():
      implementors.setdefault(name, set()).add(mid)

  nodes, edges = {}, {}
  for u in units:
    nodes[u.id] = dict(id=u.id, port="matlab", kind=u.kind, name=u.name, cls=u.cls,
                       file=u.file, line=u.line, lines=u.lines,
                       flags=sorted({s for s in u.strings if re.fullmatch(r"[A-Za-z]\w{1,30}", s)})[:60])
  for c, d in classes.items():
    if c not in {n["cls"] for n in nodes.values() if n["cls"]} and d["file"] is None:
      continue
    nodes.setdefault(f"m:class:{c}", dict(id=f"m:class:{c}", port="matlab", kind="class", name=c, cls=c,
                                          file=d["file"], supers=d["supers"]))

  def addEdge(src, dst, res):
    if src == dst:
      return
    key = (src, dst)
    if key not in edges or edges[key]["resolution"] == "dispatch":
      edges[key] = dict(src=src, dst=dst, kind="call", resolution=res)

  def dispatch(name):
    did = f"m:dispatch:{name}"
    if did not in nodes:
      nodes[did] = dict(id=did, port="matlab", kind="dispatch", name=name)
      for mid in implementors[name]:
        edges[(did, mid)] = dict(src=did, dst=mid, kind="dispatch", resolution="dispatch")
    return did

  for u in units:
    owner = u.owner or u.id
    d = os.path.dirname(u.file)
    privDirs = [os.path.join(d, "private"), d if d.endswith("private") else None]
    for a, b, _ in u.refs:
      if a in u.vars:
        continue
      if (owner, a) in locals_:
        addEdge(u.id, locals_[(owner, a)], "local")
        continue
      hit = next((privates[(p, a)] for p in privDirs if p and (p, a) in privates), None)
      if hit:
        addEdge(u.id, hit, "private")
        continue
      if a in classes:
        # a static method or constant reached through the class name
        if b and lookup(classes, a, b):
          addEdge(u.id, lookup(classes, a, b), "static")
        elif lookup(classes, a, a):
          addEdge(u.id, lookup(classes, a, a), "constructor")
        else:
          addEdge(u.id, f"m:class:{a}", "constructor")
        continue
      own = lookup(classes, u.cls, a) if u.cls else None
      if own:
        addEdge(u.id, own, "class")
        continue
      if a in functions:
        addEdge(u.id, functions[a], "function")
        continue
      if a in implementors:
        imps = implementors[a]
        addEdge(u.id, next(iter(imps)) if len(imps) == 1 else dispatch(a), "method" if len(imps) == 1 else "dispatch")
    for b, _ in u.members:
      if b in u.vars or b not in implementors:
        continue
      own = lookup(classes, u.cls, b) if u.cls else None
      imps = implementors[b]
      if own:
        addEdge(u.id, own, "class")
      else:
        addEdge(u.id, next(iter(imps)) if len(imps) == 1 else dispatch(b), "method" if len(imps) == 1 else "dispatch")
  for c, d in classes.items():
    for s in d["supers"]:
      if f"m:class:{c}" in nodes and f"m:class:{s}" in nodes:
        edges[(f"m:class:{c}", f"m:class:{s}")] = dict(src=f"m:class:{c}", dst=f"m:class:{s}", kind="inherits", resolution="exact")
    for mid in d["methods"].values():
      if f"m:class:{c}" in nodes and mid in nodes:
        edges[(f"m:class:{c}", mid)] = dict(src=f"m:class:{c}", dst=mid, kind="member", resolution="exact")
  return list(nodes.values()), list(edges.values())


if __name__ == "__main__":
  import json
  import sys
  import time
  t = time.perf_counter()
  n, e = build(sys.argv[1])
  print(f"{len(n)} nodes, {len(e)} edges in {time.perf_counter() - t:.1f} s", file=sys.stderr)
  json.dump(dict(nodes=n, edges=e), sys.stdout)
