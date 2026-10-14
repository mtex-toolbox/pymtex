import importlib.util
import inspect
import os
import re

import pytest

import mtex

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


def test_the_syntax_shows_the_keywords_and_angles_in_degrees(rendered):
  assert "return_nearest=False" in rendered["fibre.angle"]
  assert "angle=15*degree" in rendered["EBSD.calcGrains"]
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
