import ast
import json
import os

import pytest

from mtex import openDoc
from mtex.io.opendoc import asNotebook, cellScript, docPages, pageCells, pagesDir

PAGE = """# # A page
#
# Some prose with a [link](Other.html#part) and a ![](picture.png).

x = 1
x

# +
y = 2

z = x + y
# -

# More prose."""


def test_a_page_splits_into_prose_and_code():
  cells = pageCells(PAGE)
  assert [k for k, _ in cells] == ["prose", "code", "code", "prose"]
  assert cells[0][1].startswith("# A page\n\nSome prose")
  assert cells[2][1] == "y = 2\n\nz = x + y"


def test_the_cell_script_marks_every_section_and_links_the_website():
  s = cellScript(PAGE, {"Other"})
  assert s.count("# %% [markdown]") == 2 and s.count("# %%\n") == 2
  assert "(https://mtex-toolbox.github.io/Other_py.html#part)" in s
  assert "(https://mtex-toolbox.github.io/images/python/picture.png)" in s
  assert "(https://mtex-toolbox.github.io/Other.html#part)" in cellScript(PAGE)
  ast.parse(s)


def test_every_page_becomes_a_valid_script():
  pages = docPages()
  assert pages[0][0] == "Tutorials" and len(pages) > 200
  names = {p for _, p, _ in pages}
  for c, p, _ in pages:
    ast.parse(cellScript(open(os.path.join(pagesDir(), c, p + ".py"), encoding="utf-8").read(), names))


def test_the_index_and_a_chapter_are_listed():
  pages = openDoc()
  assert ("Tutorials", "EBSDTutorial", "EBSD Tutorial") in pages
  assert "EBSDTutorial" in repr(pages)
  tutorials = openDoc("tutorials")
  assert {c for c, _, _ in tutorials} == {"Tutorials"}


def test_an_unknown_page_suggests_the_close_ones():
  with pytest.raises(ValueError, match="EBSDTutorial"):
    openDoc("EBSDTutorail")


def test_a_page_is_written_once_and_kept(tmp_path):
  path = openDoc("ebsdtutorial", folder=tmp_path, edit=False)
  assert path == str(tmp_path / "EBSDTutorial.py")
  assert open(path, encoding="utf-8").read().startswith("# %% [markdown]\n# # EBSD Tutorial")
  open(path, "w").write("edited")
  openDoc("EBSDTutorial", folder=tmp_path, edit=False)
  assert open(path).read() == "edited"
  openDoc("EBSDTutorial", folder=tmp_path, edit=False, overwrite=True)
  assert open(path).read() != "edited"


def test_a_page_as_a_notebook(tmp_path):
  nb = json.loads(asNotebook(PAGE, {"Other"}))
  assert nb["nbformat"] == 4 and nb["cells"][0]["cell_type"] == "markdown"
  assert "Other_py.html#part" in "".join(nb["cells"][0]["source"])
  assert [c["cell_type"] for c in nb["cells"]] == [k if k == "code" else "markdown" for k, _ in pageCells(PAGE)]
  path = openDoc("EBSDTutorial", folder=tmp_path, edit=False, notebook=True)
  assert path == str(tmp_path / "EBSDTutorial.ipynb")
  cells = json.load(open(path, encoding="utf-8"))["cells"]
  for c in cells:
    if c["cell_type"] == "code":
      ast.parse("".join(c["source"]))
