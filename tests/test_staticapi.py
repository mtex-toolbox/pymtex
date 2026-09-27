import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("staticapi", os.path.join(ROOT, "tools", "staticapi.py"))
sa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sa)


def test_the_package_is_written_out():
  # what an editor reads must match what the package does: run tools/staticapi.py when this fails
  stale = [os.path.relpath(p, ROOT) for p in sa.changes()]
  assert not stale, f"out of date, run .venv/bin/python tools/staticapi.py: {stale}"
  assert not sa.undeclared(), "attach methods with mtex.geometry.install.install"


def test_a_comment_moves_into_the_docstring():
  src = '''# the measures of a set
# of outlines
class polygonSet:

  # one outline
  @property
  def keyhole(self):
    # the loops
    return 1

  # an inner helper keeps its comment
  def _inner(self):
    return 2

  def bare(self):
    return 3


# ---------------------------------------------- a banner stays
def after(x):
  return x
'''
  out = sa.docstringSource(src, "m", set())
  assert '''class polygonSet:
  """the measures of a set
  of outlines"""

  @property
  def keyhole(self):
    """one outline"""
    # the loops
    return 1

  # an inner helper keeps its comment
  def _inner(self):''' in out
  assert "# ---------------------------------------------- a banner stays\ndef after(x):\n  return x" in out
  # a second pass changes nothing
  assert sa.docstringSource(out, "m", set()) == out


def test_an_installed_private_function_gets_a_docstring():
  src = "# the rotation of a spin\ndef _spinExp(W):\n  return W\n"
  assert '"""the rotation of a spin"""' in sa.docstringSource(src, "m", {("m", "_spinExp")})


def test_a_backslash_makes_a_raw_docstring():
  out = sa.docstringSource("# the sum \\sum_k\ndef f():\n  return 0\n", "m", set())
  assert 'r"""the sum \\sum_k"""' in out


def test_relative_imports():
  assert sa.relative("mtex.geometry.vector3d", "mtex.plotting.sphericalplot") == "..plotting.sphericalplot"
  assert sa.relative("mtex.geometry.grids.s2grid", "mtex.plotting.spherical") == "...plotting.spherical"
  assert sa.relative("mtex.maps.ebsd", "mtex.maps.traces") == ".traces"


def test_a_stub_keeps_literal_defaults_and_the_docstring():
  def make(cs=None, simple=False, *, eps=1e-3, grid=object()):
    """the crystal shape"""
  lines = sa.stub("quartz", make, "staticmethod", "    ")
  assert lines == ["    @staticmethod", "    def quartz(cs=None, simple=False, *, eps=0.001, grid: Any = ...) -> Any:",
                   '      """the crystal shape"""']


def test_a_stub_method_takes_the_object_first():
  assert sa.stub("sqrt", lambda *a: a, None, "")[0] == "def sqrt(self, *a) -> Any:"


@pytest.mark.parametrize("width", [30, 120])
def test_wrapped_lines_stay_within_the_width(width, monkeypatch):
  monkeypatch.setattr(sa, "WIDTH", width)
  lines = sa.wrapped("from .x import (", [f"name{k}" for k in range(20)], ")")
  assert all(len(line) <= width for line in lines) and lines[-1].endswith(")")
