# the documentation pages as scripts to run section by section: openDoc('EBSDTutorial') writes
# the page as a cell script, a "# %%" line before every section, into the current folder and
# opens it in VS Code. The pages are the sources of the documentation, shipped in the wheel as
# mtex/docpages and read from docs/pages in a checkout; a page is in the jupytext light format,
# a paragraph of comment lines prose, a paragraph of code one cell, "# +" and "# -" around a
# cell of several paragraphs.
import difflib
import os
import re
import shutil
import subprocess

SITE = "https://mtex-toolbox.github.io/"

# the chapters in the order of the documentation
CHAPTERS = ("Tutorials", "GeneralConcepts", "Vectors", "Rotations", "CrystalGeometry", "CrystalOrientations",
            "Misorientations", "ODFAnalysis", "PoleFigureAnalysis", "EBSDAnalysis", "Grains", "GrainBoundaries",
            "EBSD3Analysis", "Tensors", "Elasticity", "Plasticity", "PhaseTransitions", "SphericalFunctions",
            "SO3Functions", "Plotting")


# the folder of the pages: the installed copy, else the checkout's
def pagesDir():
  here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  for d in (os.path.join(here, "docpages"), os.path.join(os.path.dirname(here), "docs", "pages")):
    if os.path.isdir(d):
      return d
  raise FileNotFoundError("the documentation pages are not installed with this copy of MTEX")


# the paragraphs of a page: ("prose", markdown) or ("code", source)
def pageCells(src):
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


# the links and pictures of prose, relative on the website, to the website: a page to its Python
# version when it has one, a picture to the Python images
def webLinks(text, names):
  def page(m):
    target, anchor = m.group(1), m.group(2) or ""
    return f"]({SITE}{target}{'_py' if target in names else ''}.html{anchor})"
  text = re.sub(r"(?<=!\[)([^\]]*)\]\(([^):/]+\.(?:png|jpg|svg))\)", lambda m: f"{m.group(1)}]({SITE}images/python/{m.group(2)})", text)
  return re.sub(r"\]\(([A-Za-z0-9_.]+)\.html(#[^)]*)?\)", page, text)


# a page as a cell script: every paragraph of prose a markdown cell, every code paragraph a cell
def cellScript(src, names=()):
  out = []
  for kind, text in pageCells(src):
    if kind == "prose":
      text = webLinks(text, set(names))
      out.append("# %% [markdown]\n" + "\n".join("# " + l if l else "#" for l in text.split("\n")))
    else:
      out.append("# %%\n" + text)
  return "\n\n".join(out) + "\n"


class docIndex(list):
  """the documentation pages, (chapter, page, title) in the order of the documentation"""

  def __repr__(self):
    lines, chapter = [f"{len(self)} documentation pages; open one with openDoc('<page>')"], None
    for c, p, t in self:
      if c != chapter:
        lines.append(f"\n{c}")
        chapter = c
      lines.append(f"  {p:<34} {t}")
    return "\n".join(lines)


# every page with its chapter and its title
def docPages():
  root = pagesDir()
  found = [c for c in os.listdir(root) if os.path.isdir(os.path.join(root, c))]
  chapters = [c for c in CHAPTERS if c in found] + sorted(set(found) - set(CHAPTERS))
  pages = docIndex()
  for c in chapters:
    for f in sorted(os.listdir(os.path.join(root, c)), key=str.lower):
      if f.endswith(".py"):
        first = open(os.path.join(root, c, f), encoding="utf-8").readline()
        pages.append((c, f[:-3], first[4:].strip() if first.startswith("# # ") else f[:-3]))
  return pages


def openDoc(name=None, folder=None, overwrite=False, edit=True):
  """a documentation page as a script to run section by section

  openDoc() lists the pages, openDoc('EBSDAnalysis') the pages of a chapter. openDoc('EBSDTutorial')
  writes the page as EBSDTutorial.py, a "# %%" line before every section, into the current folder
  (or folder=) and opens it in VS Code, unless edit=False; an existing file is kept unless
  overwrite=True. Returns the path of the script.
  """
  pages = docPages()
  if name is None:
    return pages
  chapters = {c.lower(): c for c, _, _ in pages}
  if name.lower() in chapters:
    return docIndex(e for e in pages if e[0] == chapters[name.lower()])
  byName = {p.lower(): (c, p) for c, p, _ in pages}
  if name.lower().removesuffix(".py") not in byName:
    close = difflib.get_close_matches(name, [p for _, p, _ in pages], n=3, cutoff=0.6)
    hint = f"; did you mean {', '.join(close)}?" if close else "; openDoc() lists them"
    raise ValueError(f"no documentation page named {name!r}{hint}")
  chapter, page = byName[name.lower().removesuffix(".py")]
  path = os.path.abspath(os.path.join(folder or os.getcwd(), page + ".py"))
  if overwrite or not os.path.exists(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    src = open(os.path.join(pagesDir(), chapter, page + ".py"), encoding="utf-8").read()
    with open(path, "w", encoding="utf-8") as fh:
      fh.write(cellScript(src, [p for _, p, _ in pages]))
  if edit:
    code = shutil.which("code")
    if code:
      subprocess.Popen([code, "--reuse-window", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
      print(f"wrote {path}; open it from the file list of your editor")
  return path


__all__ = ["openDoc"]
