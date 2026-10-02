# the one spelling of the style keywords of the plots (mtex/plotting/keywords.py)
import pytest

from mtex.plotting.keywords import lineKeywords, patchKeywords, refuseVariants


def test_a_style_keyword_in_MTEX_spelling_passes():
  assert refuseVariants({"lineWidth": 2, "markerSize": 4, "displayName": "a"}) == {"lineWidth": 2, "markerSize": 4, "displayName": "a"}


def test_MATLAB_spelling_of_a_style_keyword_is_refused_with_the_one_to_use():
  with pytest.raises(TypeError, match="'lineWidth'"):
    refuseVariants({"LineWidth": 2})


def test_matplotlib_spelling_of_a_style_keyword_is_refused():
  with pytest.raises(TypeError, match="'markerSize'"):
    refuseVariants({"markersize": 2})


def test_MTEX_marker_color_is_the_marker_face_color():
  with pytest.raises(TypeError, match="'markerFaceColor'"):
    refuseVariants({"MarkerColor": "r"})


def test_lineKeywords_hands_the_properties_of_a_line_to_matplotlib():
  assert lineKeywords({"lineWidth": 2, "displayName": "a", "color": "k"}) == {"linewidth": 2, "label": "a", "color": "k"}


def test_patchKeywords_hands_the_properties_of_a_patch_to_matplotlib():
  assert patchKeywords({"lineWidth": 2, "faceColor": "r"}) == {"linewidth": 2, "facecolor": "r"}


# the keys PlotKeywords declares, read from the source, the class existing only for the editor
def _completedKeys():
  import ast
  import mtex.plotting.keywords as k
  tree = ast.parse(open(k.__file__, encoding="utf-8").read())
  cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "PlotKeywords")
  return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}


def test_the_editor_completes_every_keyword_the_plots_split_off():
  from mtex.plotting.keywords import STYLE_KEYS
  from mtex.plotting.sphericalplot import AXES_KEYS, PROJECTION_KEYS, WHERE_KEYS
  assert set(STYLE_KEYS + PROJECTION_KEYS + AXES_KEYS + WHERE_KEYS) <= _completedKeys()


def test_every_plot_declares_its_keywords_for_the_editor():
  # a public or attached function of the drawing modules taking **kw annotates it Unpack[PlotKeywords]
  import ast
  import glob
  import os
  import mtex.plotting
  from mtex.geometry.install import INSTALLED
  shown = [getattr(declare or f, "__func__", declare or f) for _, _, f, declare in INSTALLED]
  attached = {(getattr(f, "__module__", None), getattr(f, "__name__", None)) for f in shown}
  skip = {"mtexfigure.py", "sections.py", "keywords.py", "render3d.py", "scene3d.py", "richdisplay.py", "vocabulary.py", "__init__.py"}
  bare = []
  for path in glob.glob(os.path.join(os.path.dirname(mtex.plotting.__file__), "*.py")):
    name = os.path.basename(path)
    module = "mtex.plotting." + name[:-3]
    for n in ast.parse(open(path, encoding="utf-8").read()).body if name not in skip else []:
      if isinstance(n, ast.FunctionDef) and n.args.kwarg and (not n.name.startswith("_") or (module, n.name) in attached):
        if n.args.kwarg.annotation is None:
          bare.append(f"{module}.{n.name}")
  assert not bare
