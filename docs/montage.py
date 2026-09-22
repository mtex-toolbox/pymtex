# the figures of a page beside MATLAB's: the MATLAB PNGs of docs/matlab/<Page>/ in the left
# column, the built ones of docs/build/ on the right, into docs/build/_montage_<Page>.png,
# for one look at a whole page. Usage: .venv/bin/python docs/montage.py EBSDImport
import os
import re
import sys

from PIL import Image

root = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def key(f):
  m = re.search(r"-(\d+)(?:-(\d+))?\.png$", f)
  return (int(m.group(1)), int(m.group(2) or 0))


def montage(page, width=420, height=260):
  matlabDir = os.path.join(root, "matlab", page)
  build = os.path.join(root, "build")
  ml = sorted([f for f in os.listdir(matlabDir) if f.endswith(".png")], key=key) if os.path.isdir(matlabDir) else []
  py = sorted([f for f in os.listdir(build) if f.startswith(page + "-") and f.endswith(".png")], key=key)
  n = max(len(ml), len(py), 1)
  out = Image.new("RGB", (2 * width, n * height), "white")
  for col, (d, files) in enumerate(((matlabDir, ml), (build, py))):
    for i, f in enumerate(files):
      im = Image.open(os.path.join(d, f)).convert("RGB")
      im.thumbnail((width, height))
      out.paste(im, (col * width, i * height))
  path = os.path.join(build, f"_montage_{page}.png")
  out.save(path)
  print(path, len(ml), "MATLAB figures,", len(py), "built")


if __name__ == "__main__":
  for page in sys.argv[1:]:
    montage(page)
