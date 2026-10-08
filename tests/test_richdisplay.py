import pytest

from mtex import crystalFrame, degree, orientation, vector3d
from mtex.plotting.richdisplay import displayedClasses, registerDisplays, reprHTML, reprPretty


class _printer:
  def __init__(self):
    self.out = ""

  def text(self, s):
    self.out += s


def test_the_html_bolds_the_name_and_explains_each_frame():
  o = orientation.byEuler(10 * degree, 20 * degree, 30 * degree, crystalFrame("m-3m", mineral="Iron"))
  h = reprHTML(o)
  assert h.startswith('<pre style="line-height: 1.25; margin: 0"><b>orientation</b> (<span title="crystalFrame')
  # the frame's own display is the tooltip of its label, escaped
  assert "mineral : Iron" in h and ">Iron</span> → <span" in h and "phi1" in h and h.endswith("</pre>")


def test_a_header_without_frames_stays_plain():
  # a frame-free vector names only the convention it is drawn in, a frame names no other frame
  h = reprHTML(vector3d(1, 2, 3))
  assert h.split("\n")[0].endswith("<b>vector3d</b> (" + repr(vector3d(1, 2, 3)).split("(", 1)[1].split("\n")[0]) and "<span" not in h
  assert "<span" not in reprHTML(crystalFrame("432"))


def test_the_terminal_display_is_the_text_with_the_name_bold():
  p = _printer()
  reprPretty(vector3d(1, 2, 3), p, False)
  assert p.out.startswith("\033[1mvector3d\033[0m (") and p.out.endswith("  1  2  3")


def test_every_class_with_a_display_is_registered():
  names = {c.__name__ for c in displayedClasses()}
  assert {"vector3d", "rotation", "orientation", "crystalFrame", "EBSD", "SO3FunRBF", "tensor"} <= names


def test_registration_needs_ipython():
  # outside IPython nothing happens, inside the two formatters know the classes
  registerDisplays()
  globalipapp = pytest.importorskip("IPython.testing.globalipapp")
  globalipapp.start_ipython()
  ip = globalipapp.get_ipython()
  registerDisplays()
  html = ip.display_formatter.formatters["text/html"]
  assert html.lookup_by_type(orientation) is reprHTML
  assert ip.display_formatter.formatters["text/plain"].lookup_by_type(orientation) is reprPretty
