# build the documentation pages: every script in docs/pages/<Chapter>/ is executed cell by
# cell and becomes docs/build/<Page>.md with its figures beside it. A page is in the
# jupytext light format: a paragraph of comment lines is prose, a paragraph of code is one
# cell, "# +" and "# -" wrap a cell of several paragraphs. Usage:
#   .venv/bin/python docs/make.py              every page
#   .venv/bin/python docs/make.py VectorDefinition ...
import ast
import contextlib
import io
import os
import shutil
import sys
import traceback
import warnings

# the pages show the defaults, never the settings file of whoever builds them
os.environ["MTEX_SETTINGS"] = ""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mtex.io.opendoc import pageCells as parse

np.set_printoptions(legacy="1.25", precision=4, suppress=True)

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pagesDir = os.path.join(root, "docs", "pages")
buildDir = os.path.join(root, "docs", "build")
HEAD, TAIL, LIMIT = 15, 10, 30


# an output block longer than LIMIT lines keeps HEAD and TAIL lines around an ellipsis
def cut(text):
  lines = text.rstrip("\n").split("\n")
  if len(lines) > LIMIT:
    lines = lines[:HEAD] + ["⋮"] + lines[-TAIL:]
  return "\n".join(lines)


# the display of a trailing expression: a number as MATLAB's format short, an integer
# without decimals, NaN and Inf as MATLAB writes them, nothing for the handles a plot
# returns, the repr of everything else
def show(value):
  if isinstance(value, (bool, np.bool_)) or value is None:
    return None if value is None else str(bool(value))
  if isinstance(value, np.ndarray) and value.ndim == 0:
    value = value[()]
  if isinstance(value, (int, float, np.integer, np.floating)):
    x = float(value)
    if not np.isfinite(x):
      return "NaN" if np.isnan(x) else ("Inf" if x > 0 else "-Inf")
    if x == int(x) and abs(x) < 1e15:
      return str(int(x))
    return f"{x:.4f}" if 1e-3 <= abs(x) < 1e5 else f"{x:.4e}"
  items = value if isinstance(value, (list, tuple)) else [value]
  while any(isinstance(x, (list, tuple)) for x in items):
    items = [y for x in items for y in (x if isinstance(x, (list, tuple)) else [x])]
  items = [x for x in items if x is not None]
  if not items and isinstance(value, (list, tuple)):
    return None
  if items and all(type(x).__module__.startswith(("matplotlib", "mpl_toolkits", "mtex.plotting")) for x in items):
    return None
  return repr(value)


# run one cell in the namespace: what it prints, then the display of a trailing expression
def runCell(code, ns):
  tree = ast.parse(code)
  last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
  buf = io.StringIO()
  # a warning is part of the output, as MATLAB's diary shows it
  with contextlib.redirect_stdout(buf), warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    warnings.simplefilter("ignore", ResourceWarning)
    if last is not None:
      tree.body = tree.body[:-1]
    exec(compile(tree, "<cell>", "exec"), ns)
    if last is not None:
      text = show(eval(compile(ast.Expression(last.value), "<cell>", "eval"), ns))
      if text is not None:
        print(text)
  return "".join(f"Warning: {w.message}\n" for w in caught) + buf.getvalue()


def build(page):
  chapter, name = page
  src = open(os.path.join(pagesDir, chapter, name + ".py"), encoding="utf-8").read()
  # the pictures a chapter keeps beside its pages go along
  for f in os.listdir(os.path.join(pagesDir, chapter)):
    if f.endswith((".png", ".jpg", ".svg")):
      shutil.copy(os.path.join(pagesDir, chapter, f), os.path.join(buildDir, f))
  # the figures of the previous build go, so that none outlives the cell that made it
  for f in os.listdir(buildDir):
    if f.startswith(name + "-") and f.endswith((".png", ".html")):
      os.remove(os.path.join(buildDir, f))
  ns = {"__name__": "__mtexdoc__"}
  out, k, ok = [], 0, True
  # the last look of every figure, by the figure, since a closed figure frees its number
  snapshots = {}
  plt.close("all")
  for kind, text in parse(src):
    if kind == "prose":
      out.append(text + "\n")
      continue
    k += 1
    out.append("```python\n" + text + "\n```\n")
    try:
      printed = runCell(text, ns)
    except Exception:
      printed = traceback.format_exc()
      ok = False
    if printed.strip():
      out.append("```text\n" + cut(printed) + "\n```\n")
    # figures live on across the cells, as in MATLAB's publish, and a figure is shown after
    # the cell that made or changed it
    changed = []
    for num in plt.get_fignums():
      fig = plt.figure(num)
      # a figure has changed when it is new, holds more artists or looks from elsewhere; a
      # settling layout is no change
      count = (sum(len(ax.get_children()) for ax in fig.axes),
               tuple((getattr(ax, "elev", None), getattr(ax, "azim", None), getattr(ax, "roll", None), ax.axison) for ax in fig.axes),
               getattr(fig, "_mtexSceneVersion", None),
               # a panel nextAxis(k) repainted, which may hold as many artists as before
               getattr(fig, "_mtexRepaints", 0),
               # a colour bar moved elsewhere, as mtexColorbar('southoutside') after one on the right,
               # or over another range, as after setColorRange
               tuple((ax._colorbar.orientation, tuple(round(float(x), 9) for x in ax.get_xlim() + ax.get_ylim()))
                     for ax in fig.axes if getattr(ax, "_colorbar", None) is not None))
      if snapshots.get(fig) == count:
        continue
      snapshots[fig] = count
      buf = io.BytesIO()
      fig.savefig(buf, format="png", dpi=120)
      changed.append((buf.getvalue(), getattr(fig, "_mtexScenes", None) or []))
    # a figure holding 3D scenes is its picture linked to the interactive page of the first,
    # the pages of the others linked below it
    for f, (data, scenes) in enumerate(changed, 1):
      png = f"{name}-{k}.png" if len(changed) == 1 else f"{name}-{k}-{f}.png"
      open(os.path.join(buildDir, png), "wb").write(data)
      if not scenes:
        out.append(f"![]({png})\n")
        continue
      from mtex.plotting.render3d import writeHTML
      pages = [png[:-4] + (".html" if j == 0 else f"-{j + 1}.html") for j in range(len(scenes))]
      for scene, page in zip(scenes, pages):
        writeHTML(scene, os.path.join(buildDir, page), title=f"{name}, figure {k}")
      out.append(f"[![]({png})]({pages[0]})\n")
      if len(pages) > 1:
        out.append("3D views: " + ", ".join(f"[{j + 1}]({p})" for j, p in enumerate(pages)) + "\n")
    if not ok:
      out.append("**The page stopped here.**\n")
      print(f"FAILED {chapter}/{name} at cell {k}\n{printed}")
      break
  plt.close("all")
  open(os.path.join(buildDir, name + ".md"), "w", encoding="utf-8").write("\n".join(out))
  # a picture kept beside the page, a screenshot the page shows, goes into the build as it is
  for f in os.listdir(os.path.join(pagesDir, chapter)):
    if f.endswith(".png"):
      shutil.copyfile(os.path.join(pagesDir, chapter, f), os.path.join(buildDir, f))
  return ok


def pages(names):
  found = []
  for chapter in sorted(os.listdir(pagesDir)):
    d = os.path.join(pagesDir, chapter)
    if os.path.isdir(d):
      found += [(chapter, f[:-3]) for f in sorted(os.listdir(d)) if f.endswith(".py")]
  return [p for p in found if not names or p[1] in names]


# every page runs in a process of its own, as a MATLAB page runs in a fresh workspace: the
# session default frame and convention, a colour set on an interned frame, the figures
def buildApart(page):
  import subprocess
  r = subprocess.run([sys.executable, os.path.abspath(__file__), "--one", page[0], page[1]], cwd=root)
  return r.returncode == 0


if __name__ == "__main__":
  os.makedirs(buildDir, exist_ok=True)
  os.chdir(root)
  if len(sys.argv) > 1 and sys.argv[1] == "--one":
    sys.exit(0 if build((sys.argv[2], sys.argv[3])) else 1)
  failed = [p for p in pages(sys.argv[1:]) if not buildApart(p)]
  sys.exit(1 if failed else 0)
