import importlib.util
import inspect
import os
import re

import pytest

import mtex
from mtex.functions.halfwidth import selectHalfwidth

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("reference", os.path.join(ROOT, "docs", "reference.py"))
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)


@pytest.fixture(scope="module")
def pages():
  return ref.catalogue()


@pytest.fixture(scope="module")
def rendered(pages):
  calls, owners = ref.pageCalls(ref.pageSources()), ref.memberOwners()
  return {n: ref.render(n, e, pages, calls, owners) for n, e in pages.items()}


def test_every_public_name_has_a_page(pages):
  shown = {id(ref.unwrap(e[1])) for e in pages.values()}
  missing = []
  for n in dir(mtex):
    obj = getattr(mtex, n)
    if n.startswith("_") or not callable(obj) or not ref.isOurs(obj):
      continue
    if id(obj) not in shown:
      missing.append(n)
    if inspect.isclass(obj):
      missing += [f"{n}.{m}" for m, v in vars(obj).items()
                  if not m.startswith("_") and ref.isOurs(v) and callable(ref.unwrap(v)) and id(ref.unwrap(v)) not in shown]
  assert not missing, missing[:20]


def test_navigation_includes_every_public_class_and_function(pages):
  chapters = ref.navigation(pages)
  items = {item["title"]: item for chapter in chapters for item in chapter["items"]}
  expected = {n for n in dir(mtex) if not n.startswith("_")
              and callable(getattr(mtex, n)) and ref.isOurs(getattr(mtex, n))} | set(ref.MODULE_FUNCTIONS)
  assert set(items) == expected
  assert "items" in items["mapImageList"]
  assert "items" in items["neper"]
  assert "items" in items["voxelLattice"]
  assert {m["title"] for m in items["trueEbsd"]["items"]} >= {"T", "calcDistortion", "undistort"}
  assert "display" not in {m["title"] for m in items["trueEbsd"]["items"]}
  assert "crystalSymmetry_index_py.html" in items["crystalFrame"]["aliases"]
  assert "orientation2color" in {m["title"] for m in items["ipfHSVKey"]["items"]}
  assert "[selectHalfwidth](selectHalfwidth_py.html)" in ref.indexPage(pages)
  ebsd = [chapter for chapter in chapters if chapter["title"] == "EBSD Analysis"]
  assert len(ebsd) == 1
  assert {item["title"] for item in ebsd[0]["items"]} >= {"EBSD", "EBSD3", "grain2d", "grain3d"}
  assert ref.indexPage(pages).count("## EBSD Analysis\n") == 1
  assert '<a id="maps"></a>' in ref.indexPage(pages) and '<a id="maps-of-volumes"></a>' in ref.indexPage(pages)


def test_navigation_links_only_to_published_python_pages(pages):
  for chapter in ref.navigation(pages):
    for item in chapter["items"]:
      urls = [item["url"], *item.get("aliases", []), *(m["url"] for m in item.get("items", []))]
      assert all(url.endswith("_py.html") and url[:-8] in pages for url in urls), item["title"]


def test_every_python_link_reaches_a_page(rendered):
  # a link to <name>_py.html names a reference page or a documentation page
  targets = set(rendered) | set(ref.pageSources())
  broken = {(n, t) for n, text in rendered.items() for t in re.findall(r"\(([\w.]+)_py\.html\)", text) if t not in targets}
  assert not broken, sorted(broken)[:20]


def test_a_matlab_page_is_what_its_name_reaches_in_python():
  assert ref.resolve("fibre.angle")[:2] == ("method", inspect.getattr_static(mtex.fibre, "angle"))
  assert ref.resolve("EBSD.calcGrains")[:2] == ("function", mtex.calcGrains)
  assert ref.resolve("Miller.angle")[2] is mtex.vector3d
  assert ref.resolve("EBSD_index")[:2] == ("class", mtex.EBSD)
  assert ref.resolve("vector3d.subsasgn") is None
  # a forwarder of the flat namespace is no method of a class without it, T no transpose of vector3d
  assert ref.resolve("EBSD.isnan") is None and ref.resolve("vector3d.ctranspose") is None
  # a name syntax.md replaced reaches its replacement
  assert ref.resolve("vector3d.angle_outer")[1] is inspect.getattr_static(mtex.vector3d, "angle")
  assert ref.resolve("loadEBSD_ang")[1] is mtex.loadEBSD
  # a documented module function can be published without adding it to the flat namespace
  assert ref.resolve("selectHalfwidth")[1] is selectHalfwidth


def test_the_syntax_shows_the_keywords_and_angles_in_degrees(rendered):
  assert "| `return_nearest` | `False` |" in rendered["fibre.angle"]
  assert "| `angle` | `15*degree` |" in rendered["EBSD.calcGrains"]
  # a class method without cls, the calling forms of a docstring kept as code
  assert "rotation.byAxisAngle(v, omega)\n```" in rendered["rotation.byAxisAngle"]
  assert "```python\nrot = rotation.byAxisAngle(v, omega)" in rendered["rotation.byAxisAngle"]
  assert "MATLAB's [fibre.angle](fibre.angle.html)" in rendered["fibre.angle"]


def test_no_page_takes_the_name_of_a_documentation_page(pages):
  assert not set(pages) & set(ref.pageSources())


def test_every_page_says_what_it_is(pages):
  # a docstring, or the operator form for an operator
  empty = sorted({n for n, e in pages.items() if not ref.docOf(e[1])
                  and getattr(ref.unwrap(e[1]), "__name__", "") not in ref.OPERATOR_FORMS})
  assert not empty, empty


def test_a_class_plot_page_shows_what_plot_draws_it_with():
  assert all(ref.plotImplementation(getattr(mtex, c)) for c in ref.PLOT)
  assert ref.resolve("EBSD.plot")[1].__name__ == "plotEBSD"
  # a class with a plot of its own keeps it
  assert ref.resolve("EBSD3.plot")[1] is mtex.EBSD3.plot


def test_the_shipped_names_match_the_reference(pages):
  # mtex/io/referencenames.json, which openDoc reads: run docs/reference.py when this fails
  import json
  with open(ref.namesFile, encoding="utf-8") as fh:
    assert json.load(fh) == ref.pythonNames(pages)


def test_a_keywords_section_lists_exactly_the_keywords_of_the_signature(pages):
  # every parameter with a default is named, and nothing the function does not take
  import inspect
  wrong = {}
  for n, e in pages.items():
    main, sections = ref.parseDoc(ref.docOf(e[1]))
    if "Keywords" not in sections:
      continue
    f = ref.unwrap(e[1])
    params = inspect.signature(f).parameters
    given = {ref.entry(x)[0] for x in sections["Keywords"]}
    inputs = {ref.entry(x)[0] for x in sections.get("Input", [])}
    # a parameter of the port's own code, _internal, is no keyword of the user's
    keywords = {k for k, p in params.items() if p.default is not inspect.Parameter.empty and k not in inputs and not k.startswith("_")}
    if given != keywords:
      wrong[n] = (sorted(given - set(params)), sorted(keywords - given))
  assert not wrong, wrong


def test_the_calling_forms_keep_their_explanations_out_of_the_code(rendered):
  import re
  page = rendered["EBSD.calcGrains"]
  syntax = page.split("## Syntax", 1)[1].split("\n## ", 1)[0]
  boxes = re.findall(r"```python\n(.*?)```", syntax, re.S)
  assert len(boxes) > 3 and not any("#" in b for b in boxes) and "```\n\nHigh and low angle boundaries.\n" in syntax


def test_class_properties_are_rendered_as_a_table(rendered):
  page = rendered["crystalShape.crystalShape"]
  properties = page.split("## Properties\n", 1)[1].split("\n## ", 1)[0]
  assert "| `V` |" in properties and "| `F` |" in properties
  assert "zero-based vertex indices" in properties
  assert "## References" in page


def test_slip_system_example_runs_and_its_figure_is_exported(tmp_path, monkeypatch, pages):
  import numpy as np
  from PIL import Image
  name = "crystalShape.plotSlipSystem"
  _, sections = ref.parseDoc(ref.docOf(pages[name][1]))
  build = tmp_path / "reference"
  output = ref.buildExample(name, sections["Example"], build / "images")
  filename = name + "-01.png"
  pixels = np.array(Image.open(build / "images" / filename).convert("RGB")).astype(int)
  # The executed example must draw both systems, not merely create an empty figure.
  assert np.count_nonzero(pixels[:, :, 0] - pixels[:, :, 2] > 80) > 100
  assert np.count_nonzero(pixels[:, :, 2] - pixels[:, :, 0] > 80) > 100
  page = ref.render(name, pages[name], pages, {}, {}, exampleOutput=output)
  assert "## Example" in page and "figures/python_reference/" + filename in page
  (build / (name + ".md")).write_text(page)
  spec = importlib.util.spec_from_file_location("docs_site", os.path.join(ROOT, "docs", "site.py"))
  site = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(site)
  monkeypatch.setattr(site, "referenceDir", str(build))
  destination = tmp_path / "site"
  site.exportReference(str(destination), {name: name})
  assert (destination / "figures" / "python_reference" / filename).read_bytes() == (build / "images" / filename).read_bytes()
  assert "## Example" in (destination / "pages" / "function_reference_python" / (name + ".md")).read_text()


def test_example_keeps_indented_code_and_comments_between_prose(tmp_path):
  example = """Add three terms.

  total = 0
  for k in range(3):  # retain the loop body in this block
    total += k

Print the result in a later block.

  print(total)
"""
  assert "```text\n3\n```" in ref.buildExample("sum", example, tmp_path)


def test_example_plotting_conventions_do_not_leak_into_the_next_example(tmp_path):
  mtex.referenceFrame.reset()
  original = str(mtex.plottingConvention.default)
  ref.buildExample("changed", "  from mtex import plottingConvention\n  plottingConvention.default('x↑→y')", tmp_path)
  result = ref.buildExample("fresh", "  from mtex import plottingConvention\n  print(str(plottingConvention.default))", tmp_path)
  assert original in result


def test_class_constants_show_their_own_help(rendered):
  assert "identity" in rendered["rotation.id"]
  assert "unit vector along x" in rendered["vector3d.X"]
  assert "Levi-Civita symbol" in rendered["tensor.leviCivita"]
  assert "A property of" in rendered["rotation.id"]
  assert "a property of the class itself" not in rendered["rotation.id"]


def test_calling_forms_are_valid_python(rendered):
  import ast
  invalid = {}
  for name, page in rendered.items():
    syntax = page.split("## Syntax\n", 1)
    if len(syntax) < 2:
      continue
    syntax = syntax[1].split("\n## ", 1)[0]
    for code in re.findall(r"```python\n(.*?)```", syntax, re.S):
      try:
        ast.parse(code)
      except SyntaxError as err:
        invalid[name] = str(err)
  assert not invalid, invalid
