# open a documentation page as a notebook in VS Code: the page is converted with jupytext
# into docs/build/<Page>.ipynb, one cell per paragraph of the light format, and handed to
# the code command. The notebook is a copy for running the page section by section; the
# page itself stays the only source. Usage:
#   .venv/bin/python docs/notebook.py VectorDefinition ...
import os
import subprocess
import sys

import jupytext

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pagesDir = os.path.join(root, "docs", "pages")
buildDir = os.path.join(root, "docs", "build")


# the path of a page given its name, searched through the chapters
def find(name):
  for chapter in sorted(os.listdir(pagesDir)):
    path = os.path.join(pagesDir, chapter, name + ".py")
    if os.path.isfile(path):
      return path
  sys.exit(f"no page named {name} in {pagesDir}")


# the notebook of a page, written into the build directory
def convert(name):
  nb = jupytext.read(find(name), fmt="py:light")
  nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (.venv)", "language": "python"}
  out = os.path.join(buildDir, name + ".ipynb")
  jupytext.write(nb, out, fmt="ipynb")
  return out


if __name__ == "__main__":
  os.makedirs(buildDir, exist_ok=True)
  files = [convert(name) for name in sys.argv[1:]] or sys.exit(__doc__ or "give the name of a page")
  subprocess.run(["code", root, *files], check=False)
