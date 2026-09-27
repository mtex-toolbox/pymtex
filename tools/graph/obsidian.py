# the graph as an Obsidian vault, at the granularity a person reads: one note per class, per
# Python module, per MATLAB folder of functions, per algorithm, per layer and per note file
#
#   .venv/bin/python tools/graph/obsidian.py [vault]      default ~/mtex/graph-vault
#
# Writes <vault>/generated/, wiped and rewritten every time; <vault>/notes/ is the reader's own
# and never touched. Note names are unique across the vault, so a [[link]] resolves: a class
# under its name (the MATLAB and the Python class of one name are one note), "py.<module>",
# "m.<folder>", "algo <name>", "layer <name>", a memory entry under its name slug (the links
# between entries already use it), "adr <file>", "research <file>". build.py writes the vault
# whenever the vault folder exists.
from __future__ import annotations

import os
import re
import shutil
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from query import G, testReach  # noqa: E402
from scan_julia import matlabClasses  # noqa: E402

VAULT = os.path.expanduser("~/mtex/graph-vault")


# the note a node belongs to, or None for a test, a doc page, a benchmark or a dispatch node; a
# Julia method joins the MATLAB class its type stands for, else the note of its source file
def container(n, nodes, mClasses=()):
  if n["port"] == "matlab":
    if n["kind"] in ("method", "class") and n.get("cls"):
      return n["cls"]
    if n["kind"] == "local":
      return container(nodes[n["id"].split(">")[0]], nodes) if n["id"].split(">")[0] in nodes else None
    if n["kind"] == "function" and n.get("layer") not in ("tests", "doc"):
      d = os.path.dirname(n["file"])
      m = re.search(r"@(\w+)", d)
      return m.group(1) if m else "m." + d.replace("/private", "").replace("/", ".")
    return None
  if n["port"] == "python":
    if n["kind"] in ("method", "class"):
      return n["cls"]
    if n["kind"] in ("function", "module") and n.get("file", "").startswith("mtex/"):
      return "py." + n["file"][5:-3].replace("/", ".").replace(".__init__", "")
  if n["port"] == "julia" and n["kind"] == "method" and n["file"].startswith("lib/"):
    cs = matlabClasses(n.get("cls"), mClasses)
    return cs[0] if cs else "jl." + n["file"].split("/")[1] + "." + os.path.basename(n["file"])[:-3]
  return None


def slug(s):
  return re.sub(r"[\\/:*?\"<>|#^\[\]]", "_", s)


# a table cell escapes the pipe of an aliased link, or the table reads it as a column
def link(name, label=None, cell=False):
  return f"[[{slug(name)}{'\\|' if cell else '|'}{label}]]" if label else f"[[{slug(name)}]]"


def write(path, text):
  os.makedirs(os.path.dirname(path), exist_ok=True)
  open(path, "w").write(text)


def fmt(t):
  return f"{1e3 * t:.1f} ms" if t < 10 else f"{t:.2f} s"


README = """# The MTEX dependency graph

`generated/` is rewritten by `pymtex/tools/graph/build.py`; write your own notes in `notes/`
and link into it - they show up as backlinks there.

- `generated/layers/` one note per layer, the way in
- `generated/classes/` a class of MATLAB, Python and Julia under one name: its members (a Julia
  method by the type of its first argument), their counterparts in the other code bases,
  whether a test reaches them, what it calls and is called by
- `generated/modules/` a Python module (`py.`), a MATLAB folder of functions (`m.`) or a Julia
  source file (`jl.`) with the methods whose type is no MATLAB class
- `generated/algorithms/` the cross-port benchmarks: entry points, cases, time per port and
  where it goes
- `generated/memory/`, `generated/docs/` the agent memory, ADRs, research notes and plans
- `generated/questions/` the research questions the notes ask, linked to the classes they name

The local graph (a note plus depth 1 or 2) is the view to work with; the global one is a map.
With the Dataview plugin, for instance

```dataview
TABLE layer, members, withCounterpart FROM "generated/classes" WHERE members > 20 SORT withCounterpart / members
```
"""


# the vault from the current graph; a README at the root when there is none
def write_vault(vault):
  G_ = G(os.path.join(HERE, "out", "graph.json"))
  nodes = G_.nodes
  out = os.path.join(vault, "generated")
  shutil.rmtree(out, ignore_errors=True)
  os.makedirs(os.path.join(vault, "notes"), exist_ok=True)

  mClasses = {n["name"] for n in nodes.values() if n["port"] == "matlab" and n["kind"] == "class"}
  owner = {i: container(n, nodes, mClasses) for i, n in nodes.items()}
  members = defaultdict(list)
  for i, c in owner.items():
    if c and nodes[i]["kind"] not in ("class", "module", "local"):
      members[c].append(i)
  reached = testReach(G_, "matlab") | testReach(G_, "python") | testReach(G_, "julia")
  cov = G_.g.get("coverage", {})

  # the notes linked from the graph, under the names their files and their links use
  noteName = {}
  for i, n in nodes.items():
    if n["kind"] != "note":
      continue
    text = open(n["file"], encoding="utf-8", errors="replace").read() if os.path.exists(n["file"]) else ""
    base = os.path.splitext(n["name"])[0]
    if "/memory" in n["file"]:
      noteName[i] = base
    else:
      kind = "adr" if "/adr/" in n["file"] else "research" if "/research/" in n["file"] or "/tests/" in n["file"] else "plan" if "/plans/" in n["file"] else "todo"
      noteName[i] = f"{kind} {base}"
      write(os.path.join(out, "docs", slug(noteName[i]) + ".md"),
            f"---\ntype: {kind}\nfile: {n['file']}\n---\n# {n.get('title') or base}\n\n[{n['file']}](file://{n['file']})\n")

  # the research questions, each linked to the note that asks it and to the containers it names
  for i, n in nodes.items():
    if n["kind"] != "question":
      continue
    asked = [noteName[e["dst"]] for e in G_.out[i] if e["kind"] == "asked" and e["dst"] in noteName]
    named = sorted({owner[e["dst"]] for e in G_.out[i] if e["kind"] == "question" and owner.get(e["dst"])})
    write(os.path.join(out, "questions", slug(f"question {n['title'][:60]}") + ".md"),
          f"---\ntype: question\nfile: {n['file']}\nline: {n['line']}\n---\n# {n['title']}\n\n"
          f"[{n['file']}:{n['line']}](file://{n['file']})" + (f"\n\nasked in {', '.join(link(a) for a in asked)}" if asked else "")
          + (f"\n\nnames {' '.join(link(c) for c in named)}" if named else "") + "\n")

  # every memory entry, since the entries link each other whether or not they mention the code
  import build
  for d in build.MEMORY:
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
      if f.endswith(".md") and f != "MEMORY.md":
        # the entries link each other by file name or by name slug: the one is the note, the other its alias
        text = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
        m = re.search(r"^name:\s*(\S+)", text, re.M)
        if m and m.group(1) != f[:-3] and text.startswith("---\n"):
          text = f"---\naliases: [{m.group(1)}]\n" + text[4:]
        write(os.path.join(out, "memory", f), text)

  # the algorithms, and which containers they enter through
  algoOf = defaultdict(set)
  for i, n in nodes.items():
    if n["kind"] != "algorithm":
      continue
    name = f"algo {n['name']}"
    for e in G_.out[i]:
      if e["kind"] == "entry" and owner.get(e["dst"]):
        algoOf[owner[e["dst"]]].add(name)
    lines = [f"---\ntype: algorithm\n---\n# {n['name']}\n", f"benchmarks: {', '.join(n.get('files', []))}\n"]
    for port, es in n["entry"].items():
      lines.append(f"- {port}: " + ", ".join(link(owner.get(build.PREFIX[port] + e) or e, e) for e in es))
    profs = [p for p in G_.g["profiles"] if p["algorithm"] == n["name"]]
    for case in sorted({p["case"] for p in profs}):
      lines.append(f"\n## {case}\n\n{n.get('cases', {}).get(case, '')}\n")
      for p in sorted((p for p in profs if p["case"] == case), key=lambda p: p["port"]):
        top = sorted(p["functions"].items(), key=lambda kv: -kv[1]["self"])[:5]
        tops = ", ".join(link(owner[f], nodes[f]["name"]) if owner.get(f) else f.split(":")[-1][:40] for f, _ in top)
        lines.append(f"- **{p['port']}** {fmt(p['wall'])}: {tops}")
    write(os.path.join(out, "algorithms", slug(name) + ".md"), "\n".join(lines) + "\n")

  # the containers: members with their counterparts, the containers they call and are called by
  layers = defaultdict(list)
  for c, ms in sorted(members.items()):
    ports = sorted({nodes[i]["port"] for i in ms})
    layer = next((nodes[i].get("layer") for i in ms if nodes[i].get("layer")), "other")
    layers[layer].append(c)
    calls, calledBy, notes = defaultdict(int), defaultdict(int), set()
    rows = []
    for i in sorted(ms, key=lambda i: (nodes[i]["name"], nodes[i]["port"])):
      n = nodes[i]
      for e in G_.out[i]:
        if e["kind"] == "call" and e["resolution"] != "dispatch" and owner.get(e["dst"]) not in (None, c):
          calls[owner[e["dst"]]] += 1
      for e in G_.inn[i]:
        if e["kind"] == "call" and e["resolution"] != "dispatch" and owner.get(e["src"]) not in (None, c):
          calledBy[owner[e["src"]]] += 1
        elif e["kind"] == "note" and e["src"] in noteName:
          notes.add(noteName[e["src"]])
      cps = [j for j, _ in G_.counterparts(i)]
      cp = ", ".join(link(owner[j], nodes[j]["name"], cell=True) if owner.get(j) in members and owner[j] != c else nodes[j]["name"]
                     for j in cps if j in nodes)
      tested = "measured" if i in cov else "static" if i in reached else ""
      plumb = " (plumbing)" if n.get("plumbing") else ""
      rows.append(f"| {n['name']}{plumb} | {n['port']} | {cp} | {tested} | `{n.get('file', '')}:{n.get('line', '')}` |")
    top = lambda d: " ".join(link(k) for k, _ in sorted(d.items(), key=lambda kv: -kv[1])[:25])
    text = [f"---\ntype: {'class' if not c.startswith(('py.', 'm.', 'jl.')) else 'module'}\nlayer: {layer}\nports: [{', '.join(ports)}]\n"
            f"members: {len(ms)}\nwithCounterpart: {sum(1 for i in ms if G_.counterparts(i))}\n---\n# {c}\n",
            f"layer {link('layer ' + layer)}" + ("; algorithms " + ", ".join(link(a) for a in sorted(algoOf[c])) if algoOf[c] else ""),
            "\n## Members\n\n| name | port | counterpart | tested | where |\n|---|---|---|---|---|", *rows,
            f"\n## Calls\n\n{top(calls)}", f"\n## Called by\n\n{top(calledBy)}"]
    if notes:
      text.append("\n## Notes\n\n" + " ".join(link(x) for x in sorted(notes)))
    write(os.path.join(out, "classes" if not c.startswith(("py.", "m.", "jl.")) else "modules", slug(c) + ".md"), "\n".join(text) + "\n")

  for layer, cs in layers.items():
    write(os.path.join(out, "layers", slug(f"layer {layer}") + ".md"),
          f"---\ntype: layer\n---\n# layer {layer}\n\n" + "\n".join(f"- {link(c)}" for c in sorted(cs, key=str.lower)) + "\n")
  if not os.path.exists(os.path.join(vault, "README.md")):
    write(os.path.join(vault, "README.md"), README)
  print(f"{sum(len(f) for _, _, f in os.walk(out))} notes in {out}", file=sys.stderr)


if __name__ == "__main__":
  import build
  vault = sys.argv[1] if len(sys.argv) > 1 else VAULT
  if build.stale():
    build.main([])
  os.makedirs(vault, exist_ok=True)
  write_vault(vault)
