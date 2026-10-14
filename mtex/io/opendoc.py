# the documentation pages as scripts to run section by section: openDoc('EBSDTutorial') writes
# the page as a cell script, a "# %%" line before every section, into the current folder and
# opens it in VS Code, or with notebook=True as a Jupyter notebook. A function, class or method,
# by name or itself, opens its page of the function reference in the browser: openDoc('fibre.angle'),
# openDoc(calcGrains); referencenames.json, written by docs/reference.py, names the page of every
# Python name. The pages are the sources of the documentation, shipped in the wheel as
# mtex/docpages and read from docs/pages in a checkout; a page is in the jupytext light format,
# a paragraph of comment lines prose, a paragraph of code one cell, "# +" and "# -" around a
# cell of several paragraphs.
from __future__ import annotations

import difflib
import json
import os
import re
import shutil
import subprocess
from typing import TYPE_CHECKING, Any

from ..geometry.display import aligned

if TYPE_CHECKING:
  from .vocabulary import DocPageName

SITE = "https://mtex-toolbox.github.io/"

# the chapters in the order of the documentation
CHAPTERS = ("Tutorials", "GeneralConcepts", "Vectors", "Rotations", "CrystalGeometry", "CrystalOrientations",
            "Misorientations", "ODFAnalysis", "PoleFigureAnalysis", "EBSDAnalysis", "Grains", "GrainBoundaries",
            "EBSD3Analysis", "Tensors", "Elasticity", "Plasticity", "PhaseTransitions", "SphericalFunctions",
            "SO3Functions", "Plotting")


def pagesDir():
  """the folder of the pages: the installed copy, else the checkout's"""
  here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  for d in (os.path.join(here, "docpages"), os.path.join(os.path.dirname(here), "docs", "pages")):
    if os.path.isdir(d):
      return d
  raise FileNotFoundError("the documentation pages are not installed with this copy of MTEX")


def pageCells(src):
  """the paragraphs of a page: ("prose", markdown) or ("code", source)"""
  blocks, para, wrapped = [], [], False
  for line in src.split("\n") + [""]:
    if line.strip() == "# +":
      wrapped = True
      continue
    if line.strip() == "# -":
      wrapped = False
      blocks.append(("code", "\n".join(para).strip("\n")))
      para = []
      continue
    if line.strip() == "" and not wrapped:
      if para:
        prose = all(l.startswith("#") for l in para)
        text = "\n".join(l[2:] if l.startswith("# ") else l[1:] for l in para) if prose else "\n".join(para)
        blocks.append(("prose" if prose else "code", text))
      para = []
    else:
      para.append(line)
  return [(k, t) for k, t in blocks if t.strip()]


def webLinks(text, names):
  """the links and pictures of prose, relative on the website, to the website: a page to its Python
  version when it has one, a picture to the Python images"""
  def page(m):
    target, anchor = m.group(1), m.group(2) or ""
    return f"]({SITE}{target}{'_py' if target in names else ''}.html{anchor})"
  text = re.sub(r"(?<=!\[)([^\]]*)\]\(([^):/]+\.(?:png|jpg|svg))\)", lambda m: f"{m.group(1)}]({SITE}figures/python/{m.group(2)})", text)
  return re.sub(r"\]\(([A-Za-z0-9_.]+)\.html(#[^)]*)?\)", page, text)


def cellScript(src, names=()):
  """a page as a cell script: every paragraph of prose a markdown cell, every code paragraph a cell"""
  out = []
  for kind, text in pageCells(src):
    if kind == "prose":
      text = webLinks(text, set(names))
      out.append("# %% [markdown]\n" + "\n".join("# " + l if l else "#" for l in text.split("\n")))
    else:
      out.append("# %%\n" + text)
  return "\n\n".join(out) + "\n"


def asNotebook(src, names=()):
  """a page as a Jupyter notebook: every paragraph of prose a markdown cell, every code paragraph a cell"""
  def cell(kind, text):
    lines = text.split("\n")
    source = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    if kind == "prose":
      return {"cell_type": "markdown", "metadata": {}, "source": source}
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}
  cells = [cell(k, webLinks(t, set(names)) if k == "prose" else t) for k, t in pageCells(src)]
  meta = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
          "language_info": {"name": "python"}}
  return json.dumps({"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}, indent=1, ensure_ascii=False) + "\n"


class docIndex(list):
  """the documentation pages, (chapter, page, title) in the order of the documentation"""

  def __repr__(self):
    lines, chapter = [f"{len(self)} documentation pages; open one with openDoc('<page>')"], None
    for c, p, t in self:
      if c != chapter:
        lines.append(f"\n{c}")
        chapter = c
      lines.append(f"  {p:<34} {t}")
    return aligned("\n".join(lines))


def docPages():
  """every page with its chapter and its title"""
  root = pagesDir()
  found = [c for c in os.listdir(root) if os.path.isdir(os.path.join(root, c))]
  chapters = [c for c in CHAPTERS if c in found] + sorted(set(found) - set(CHAPTERS))
  pages = docIndex()
  for c in chapters:
    for f in sorted(os.listdir(os.path.join(root, c)), key=str.lower):
      if f.endswith(".py"):
        with open(os.path.join(root, c, f), encoding="utf-8") as fh:
          first = fh.readline()
        pages.append((c, f[:-3], first[4:].strip() if first.startswith("# # ") else f[:-3]))
  return pages


def referencePage(x):
  """the page of the function reference of a name or an object, or None"""
  path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "referencenames.json")
  with open(path, encoding="utf-8") as fh:
    names = json.load(fh)
  if isinstance(x, str):
    candidates = [x]
  else:
    f = getattr(x, "__func__", x)
    owner = type(x.__self__) if hasattr(x, "__self__") and not isinstance(x.__self__, type) else getattr(x, "__self__", None)
    name = getattr(f, "__name__", "")
    qual = getattr(f, "__qualname__", "")
    candidates = [name] if isinstance(x, type) else []
    if owner is not None and isinstance(owner, type):
      candidates += [f"{c.__name__}.{name}" for c in owner.__mro__]
    candidates += [qual, name]
  # a member is documented on the class defining it when its subclass has no page of its own
  for c in candidates:
    head, _, member = c.partition(".")
    if c in names:
      return names[c]
    if member:
      import mtex
      cls = getattr(mtex, head, None)
      for base in getattr(cls, "__mro__", ())[1:]:
        if f"{base.__name__}.{member}" in names:
          return names[f"{base.__name__}.{member}"]
  return None


def openDoc(name: DocPageName | Any = None, folder=None, overwrite=False, edit=True, notebook=False):
  """a documentation page as a script to run section by section, or the reference page of a
  function in the browser

    pages = openDoc()                            # the list of the documentation pages
    openDoc('EBSDAnalysis')                      # the pages of one chapter
    path = openDoc('EBSDTutorial')               # the page as a script, opened in VS Code
    path = openDoc('EBSDTutorial', notebook=True)  # the page as a Jupyter notebook
    url = openDoc('fibre.angle')                 # the reference page of a function by its name
    url = openDoc(calcGrains)                    # the reference page of a function itself
    url = openDoc(ori.mean)                      # the reference page of a method

  A documentation page is written as <Page>.py, a "# %%" line before every section, into the
  current folder, and opened in VS Code, where every section runs on its own; with
  notebook=True it is written as the Jupyter notebook <Page>.ipynb. A file already there is
  kept unless overwrite=True. A function, class or method, by its name or itself, opens its
  page of the function reference on the MTEX website.

  Input
    name - str, a documentation page, a chapter or the name of a function; or the function,
        class or method itself
  Output
    out - the list of pages for no name, the pages of a chapter, the path of the file written,
        or the address of the reference page
  Keywords
    folder - the folder the script is written into, by default the current one
    overwrite - write the page again over a file of its name
    edit - open the script in VS Code, or the reference page in the browser; False only
        writes the file or returns the address
    notebook - write a Jupyter notebook instead of a script
  See also
    mtexdata, Installation, MATLABtoPython"""
  if name is not None and not isinstance(name, str):
    page = referencePage(name)
    if page is None:
      raise ValueError(f"no reference page for {name!r}")
    return _browse(page, edit)
  pages = docPages()
  if name is None:
    return pages
  chapters = {c.lower(): c for c, _, _ in pages}
  if name.lower() in chapters:
    return docIndex(e for e in pages if e[0] == chapters[name.lower()])
  byName = {p.lower(): (c, p) for c, p, _ in pages}
  if name.lower().removesuffix(".py") not in byName and referencePage(name):
    return _browse(referencePage(name), edit)
  if name.lower().removesuffix(".py") not in byName:
    close = difflib.get_close_matches(name, [p for _, p, _ in pages], n=3, cutoff=0.6)
    hint = f"; did you mean {', '.join(close)}?" if close else "; openDoc() lists them"
    raise ValueError(f"no documentation page named {name!r}{hint}")
  chapter, page = byName[name.lower().removesuffix(".py")]
  path = os.path.abspath(os.path.join(folder or os.getcwd(), page + (".ipynb" if notebook else ".py")))
  if overwrite or not os.path.exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(os.path.join(pagesDir(), chapter, page + ".py"), encoding="utf-8") as fh:
      src = fh.read()
    with open(path, "w", encoding="utf-8") as fh:
      fh.write((asNotebook if notebook else cellScript)(src, [p for _, p, _ in pages]))
  if edit:
    code = shutil.which("code")
    if code:
      subprocess.Popen([code, "--reuse-window", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
      print(f"wrote {path}; open it from the file list of your editor")
  return path


# the address of a reference page, opened in the browser unless edit=False
def _browse(page, edit):
  url = f"{SITE}{page}_py.html"
  if edit:
    import webbrowser
    webbrowser.open(url)
  return url


__all__ = ["openDoc"]
