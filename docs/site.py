# the built pages into the MTEX website, the Jekyll site of mtex-toolbox.github.io: every
# docs/build/<Page>.md becomes pages/documentation_python/<Page>.md with its front matter,
# the permalink <Page>_py.html beside MATLAB's <Page>.html, its figures in figures/python/
# (the site's checkout of mtex-toolbox/figures, served at /figures/), the page as a cell
# script in python_scripts/ and one entry in _data/python_pages.yml, the table the site's
# MATLAB | Python switch reads.
# Run docs/make.py first. Usage:
#   .venv/bin/python docs/site.py ../web              every built page
#   .venv/bin/python docs/site.py ../web CrystalShapes ...
# The whole export also writes the function reference of docs/reference.py into
# pages/function_reference_python/ and lists its pages with a MATLAB twin in python_pages.yml.
import io
import json
import os
import re
import shutil
import sys

import numpy as np
from PIL import Image

from mtex.io.opendoc import cellScript

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pagesDir = os.path.join(root, "docs", "pages")
buildDir = os.path.join(root, "docs", "build")
referenceDir = os.path.join(buildDir, "reference")
statusFile = os.path.join(root, "docs", "plans", "remaining-pages.md")
EDIT = "https://github.com/mtex-toolbox/pymtex/blob/main/docs/pages/{}/{}.py"
# where the figures are served, relative to a page at the site root
FIG = "figures/python"

# what a page that is not checked against MATLAB tells its reader
NOTES = {
  "Adapted": "This page is adapted for Python and differs from the MATLAB page.",
  "Partial": "Parts of this page are not yet available in Python; the MATLAB page has them all.",
  "Written": "This page is not yet compared with the MATLAB page.",
}


# the status word of every page in the table of remaining-pages.md
def statuses():
  found = {}
  for line in open(statusFile, encoding="utf-8"):
    cells = [c.strip() for c in line.split("|")]
    if len(cells) > 4 and "/" in cells[1] and cells[3]:
      found[cells[2]] = cells[3].split()[0].strip("*")
  return found


# the chapter of every page source
def chapters():
  found = {}
  for chapter in sorted(os.listdir(pagesDir)):
    d = os.path.join(pagesDir, chapter)
    if os.path.isdir(d):
      found.update({f[:-3]: chapter for f in os.listdir(d) if f.endswith(".py")})
  return found


# the permalinks the site has, read off the front matter of its pages
def permalinks(site):
  found = set()
  for dirpath, _, files in os.walk(os.path.join(site, "pages"), followlinks=True):
    if dirpath.endswith("documentation_python"):
      continue
    for f in files:
      if f.endswith((".html", ".md")):
        with open(os.path.join(dirpath, f), encoding="utf-8", errors="replace") as fh:
          head = fh.read(600)
        m = re.search(r"^permalink:\s*/?(\S+?)\s*$", head, re.M)
        if m:
          found.add(m.group(1))
  return found


# the inline math $x$ as kramdown's $$x$$, outside code
def math(text):
  return re.sub(r"(?<![\$\\])\$(?!\$)([^\$\n]+?)(?<![\$\\])\$(?!\$)", r"$$\1$$", text)


# a line of prose apart from its code spans
def prose(line, fn):
  parts = re.split(r"(`[^`]*`)", line)
  return "".join(p if p.startswith("`") else fn(p) for p in parts)


# a link to a page: its Python page when there is one, else the site's page, else the text
def relink(line, python, known):
  def one(m):
    text, target, anchor = m.group(1), m.group(2), m.group(3) or ""
    if target in python:
      return f"[{text}]({target}_py.html)"
    if f"{target}.html" in known:
      return f"[{text}]({target}.html{anchor})"
    return text
  return re.sub(r"(?<!!)\[([^\]]*)\]\(([A-Za-z0-9_.]+)\.html(#[^)]*)?\)", one, line)


# a figure of the page as the site shows MATLAB's, halved by the class mtex-figure; a figure
# holding 3D scenes links to its interactive page
def figure(line, name):
  gen = re.compile(rf"{re.escape(name)}-\d+(?:-\d+)?\.png")
  m = re.fullmatch(r"\[!\[\]\(([^)]+)\)\]\(([^)]+)\)", line)
  if m and gen.fullmatch(m.group(1)):
    return (f'<center class="mtex-figure"><a href="{FIG}/{m.group(2)}">'
            f'<img class="inline" src="{FIG}/{m.group(1)}"></a></center>'), [m.group(1), m.group(2)]
  m = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
  if m and gen.fullmatch(m.group(2)):
    return f'<center class="mtex-figure"><img class="inline" src="{FIG}/{m.group(2)}"></center>', [m.group(2)]
  used = re.findall(r"\]\(([^)/:]+\.(?:png|jpg|svg|html))\)", line)
  used = [u for u in used if u.endswith((".png", ".jpg", ".svg")) or u.startswith(name + "-")]
  for u in used:
    line = line.replace(f"]({u})", f"]({FIG}/{u})")
  return line, used


# the page body for the site and the files it uses
def body(md, name, python, known):
  out, used, fence = [], [], False
  lines = md.split("\n")
  # the title is the layout's
  if lines and lines[0].startswith("# "):
    lines = lines[1:]
  for line in lines:
    if line.startswith("```"):
      fence = not fence
    elif not fence:
      line, files = figure(line, name)
      used += files
      if not line.startswith("<center"):
        line = prose(line, lambda p: relink(math(p), python, known))
    out.append(line)
  return "\n".join(out).strip("\n"), used


# a figure as 256 colours where that is invisible, a mean error under one level in a hundred on
# white, else lossless: a map of many orientations keeps its colours; the same pixels give the
# same bytes
def quantise(src):
  im = Image.open(src)
  im = im.convert("RGBA")
  rgba = np.asarray(im)
  if (rgba[..., 3] == 255).all():
    im = im.convert("RGB")
  q = im.quantize(256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
  if np.abs(white(q) - white(im)).mean() > 2.55:
    q = im
  buf = io.BytesIO()
  q.save(buf, "PNG", optimize=True)
  return buf.getvalue()


# an image laid on white, as the page shows it
def white(im):
  x = np.asarray(im.convert("RGBA"), dtype=float)
  a = x[..., 3:] / 255
  return x[..., :3] * a + 255 * (1 - a)


# a file into figures/python, written only when its bytes differ, so that a rebuild leaves
# an unchanged figure alone and the site's figure diff shows only real changes
def place(src, dst):
  data = quantise(src) if src.endswith(".png") else open(src, "rb").read()
  if os.path.exists(dst) and open(dst, "rb").read() == data:
    return False
  open(dst, "wb").write(data)
  return True


# the status of a page: its word in remaining-pages.md, New for a page without a MATLAB page
def pageStatus(name, status, known):
  return status.get(name) or ("Written" if f"{name}.html" in known else "New")


def frontMatter(title, name, status, counterpart):
  lines = ["---", "title: '" + title.replace("'", "''") + "'", "sidebar: documentation_sidebar",
           f"permalink: {name}_py.html", "folder: documentation", "toc: false", "search: exclude", "lang: python"]
  if counterpart:
    lines.append(f"counterpart: {name}.html")
  lines.append(f"status: {status}")
  if status in NOTES:
    lines.append("note: '" + NOTES[status] + "'")
  return "\n".join(lines + ["---"])


# the function reference of docs/reference.py: name -> its MATLAB page or None
def referencePages():
  path = os.path.join(referenceDir, "index.json")
  if not os.path.exists(path):
    return {}
  return {n: e["matlab"] for n, e in json.load(open(path, encoding="utf-8")).items()}


def referenceFrontMatter(title, name, matlab):
  lines = ["---", "title: '" + title.replace("'", "''") + "'", "sidebar: function_reference_sidebar",
           f"permalink: {name}_py.html", "folder: function_reference", "toc: false", "search: exclude", "lang: python"]
  if matlab:
    lines.append(f"counterpart: {matlab}.html")
  return "\n".join(lines + ["status: Reference", "---"])


# the reference pages into pages/function_reference_python; their links are final, written by
# docs/reference.py, so they are not relinked
def exportReference(site, reference):
  out = os.path.join(site, "pages", "function_reference_python")
  os.makedirs(out, exist_ok=True)
  used = set()
  for n, matlab in reference.items():
    md = open(os.path.join(referenceDir, n + ".md"), encoding="utf-8").read()
    used.update(re.findall(r"\]\(figures/python_reference/([^/()]+\.png)\)", md))
    title, _, text = md.partition("\n")
    page = referenceFrontMatter(title[2:].strip(), n, matlab) + "\n{% raw %}\n" + text.strip("\n") + "\n{% endraw %}\n"
    path = os.path.join(out, n + ".md")
    if not os.path.exists(path) or open(path, encoding="utf-8").read() != page:
      open(path, "w", encoding="utf-8").write(page)
  for f in os.listdir(out):
    if f.endswith(".md") and f[:-3] not in reference:
      os.remove(os.path.join(out, f))
  # Reference examples are executed by docs/reference.py; publish their captured figures.
  images = os.path.join(site, "figures", "python_reference")
  if used:
    os.makedirs(images, exist_ok=True)
    for filename in used:
      shutil.copyfile(os.path.join(referenceDir, "images", filename), os.path.join(images, filename))
  if os.path.isdir(images):
    for filename in os.listdir(images):
      if filename.endswith(".png") and filename not in used:
        os.remove(os.path.join(images, filename))
  # the page of every Python name, for the links in the code of the pages (js/mx-reflinks.js)
  with open(os.path.join(root, "mtex", "io", "referencenames.json"), encoding="utf-8") as fh:
    names = {k: v for k, v in json.load(fh).items() if v in reference}
  with open(os.path.join(site, "python_refs.json"), "w", encoding="utf-8") as fh:
    json.dump(names, fh, separators=(",", ":"), sort_keys=True)
  navigation = os.path.join(referenceDir, "navigation.json")
  if os.path.exists(navigation):
    shutil.copyfile(navigation, os.path.join(site, "python_reference.json"))
  print(f"{len(reference)} reference pages, {len(names)} names linked")


def export(site, names):
  status, chapter, known = statuses(), chapters(), permalinks(site)
  built = sorted(f[:-3] for f in os.listdir(buildDir) if f.endswith(".md") and f[:-3] in chapter)
  # a page that stopped is not published
  ok = [n for n in built if "**The page stopped here.**" not in open(os.path.join(buildDir, n + ".md"), encoding="utf-8").read()]
  for n in sorted(set(built) - set(ok)):
    print(f"skipped {n}: the page stopped")
  reference = referencePages()
  # a link to a reference page MATLAB has goes to its Python twin
  python = set(ok) | set(reference)
  pageOut = os.path.join(site, "pages", "documentation_python")
  # the figures go into the site's checkout of mtex-toolbox/figures, which the site's
  # tools/deploy-figures.sh publishes
  if not os.path.isdir(os.path.join(site, "figures", ".git")):
    sys.exit(f"{site}/figures is not a checkout of mtex-toolbox/figures; clone it there first:\n"
             f"  git clone git@github.com:mtex-toolbox/figures.git {site}/figures")
  imageOut = os.path.join(site, *FIG.split("/"))
  os.makedirs(pageOut, exist_ok=True)
  os.makedirs(imageOut, exist_ok=True)
  scriptOut = os.path.join(site, "python_scripts")
  os.makedirs(scriptOut, exist_ok=True)
  os.makedirs(os.path.join(site, "_data"), exist_ok=True)
  written, used = 0, set()
  for n in ok if not names else [n for n in ok if n in names]:
    md = open(os.path.join(buildDir, n + ".md"), encoding="utf-8").read()
    title = md.split("\n", 1)[0][2:].strip()
    text, files = body(md, n, python, known)
    page = (frontMatter(title, n, pageStatus(n, status, known), f"{n}.html" in known) + "\n{% raw %}\n"
            + f'<font size="2"><a href="python_scripts/{n}.py" download>download as script</a> ·'
            + f' <a href="{EDIT.format(chapter[n], n)}">edit page</a></font>\n\n'
            + "<!--introduction-->\n\n" + text + "\n{% endraw %}\n")
    open(os.path.join(pageOut, n + ".md"), "w", encoding="utf-8").write(page)
    # the page as a cell script to run section by section, as openDoc writes it
    src = open(os.path.join(pagesDir, chapter[n], n + ".py"), encoding="utf-8").read()
    open(os.path.join(scriptOut, n + ".py"), "w", encoding="utf-8").write(cellScript(src, python))
    for f in files:
      src = os.path.join(buildDir, f)
      if not os.path.exists(src):
        src = os.path.join(pagesDir, chapter[n], f)
      if os.path.exists(src):
        written += place(src, os.path.join(imageOut, f))
        used.add(f)
      else:
        print(f"missing {f} of {n}")
  # the whole site: the table of pages, and no page or image left from an earlier export
  if not names:
    with open(os.path.join(site, "_data", "python_pages.yml"), "w", encoding="utf-8") as fh:
      fh.write("# written by pymtex docs/site.py: the documentation pages that have a Python version\n")
      for n in ok:
        fh.write(f"{n}: {{status: {pageStatus(n, status, known)}}}\n")
      # the switch needs only the reference pages with a MATLAB twin
      for n in sorted(n for n, matlab in reference.items() if matlab and n not in ok):
        fh.write(f"{n}: {{status: Reference}}\n")
    for f in os.listdir(pageOut):
      if f.endswith(".md") and f[:-3] not in ok:
        os.remove(os.path.join(pageOut, f))
    for f in os.listdir(imageOut):
      if f not in used:
        os.remove(os.path.join(imageOut, f))
    for f in os.listdir(scriptOut):
      if f[:-3] not in ok:
        os.remove(os.path.join(scriptOut, f))
    exportReference(site, reference)
  print(f"{len(names) or len(ok)} pages, {written} images written")


if __name__ == "__main__":
  if len(sys.argv) < 2:
    sys.exit("usage: docs/site.py <site> [pages ...]")
  export(sys.argv[1], sys.argv[2:])
