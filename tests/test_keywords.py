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


@pytest.mark.parametrize("flag", sorted(__import__("mtex.plotting.keywords", fromlist=["MATLAB_FLAGS"]).MATLAB_FLAGS))
def test_a_MATLAB_option_the_port_does_not_take_is_refused_with_what_to_write(flag):
  from mtex.plotting.keywords import MATLAB_FLAGS
  with pytest.raises(TypeError, match=f"'{flag}'.*" + MATLAB_FLAGS[flag][:12].replace("(", "\\(").replace(")", "\\)")):
    refuseVariants({flag: True}, flags=(flag,))


# MATLAB's documented options of the plots of directions, maps, grains, boundaries and triple
# points (their help's Options and Flags), each with a value it takes
MATLAB_OPTIONS = {
  "scatter": ["numbered", "dynamicMarkerSize", "grid", "markerAlpha", "markerEdgeAlpha", "markerFaceAlpha", "marker",
              "markerFaceColor", "markerEdgeColor", "markerSize"],
  "plotVectors": ["smooth", "contourf", "contour", "marker", "markerSize", "markerFaceColor", "markerEdgeColor"],
  "text": ["textColor", "textAboveMarker"],
  "quiver": ["arrowSize", "autoArrowSize", "maxHeadSize"],
  "plotGrains": ["noBoundary", "displayName", "region", "scale", "micronbar", "refFrame", "refFrameDirs", "refFrameLabels"],
  "plotEBSD": ["micronbar", "refFrame", "refFrameDirs", "refFrameLabels", "backend", "displayName", "region", "points", "exact"],
  "plotBoundary": ["smooth", "lineWidth", "lineColor", "edgeAlpha", "region", "displayName"],
  "plotTriplePoints": ["marker", "markerSize", "markerEdgeColor", "markerFaceColor", "lineWidth"],
}
VALUES = {"region": (33000, 34000, 4000, 5000), "backend": "patch", "displayName": "x", "scale": 0.3, "refFrameDirs": None,
          "refFrameLabels": ["x", "y", "z"], "lineColor": "r", "markerEdgeColor": "r", "markerFaceColor": "r", "marker": "s",
          "markerSize": 3, "lineWidth": 1, "edgeAlpha": 0.5, "textColor": "r", "arrowSize": 0.2, "maxHeadSize": 0.3,
          "markerAlpha": 0.5, "markerEdgeAlpha": 0.5, "markerFaceAlpha": 0.5, "micronbar": False, "refFrame": False}


@pytest.fixture(scope="module")
def drawn():
  import os
  matplotlib = pytest.importorskip("matplotlib")
  matplotlib.use("Agg")
  from mtex import calcGrains, degree, loadEBSD, plot, quiver, scatter, text, vector3d
  ebsd = loadEBSD(os.path.join(os.path.dirname(__file__), "fixtures", "ebsd", "Forsterite_cut.ctf"))
  grains = calcGrains(ebsd["indexed"], angle=10 * degree)
  v = vector3d.rand(5, rng=__import__("numpy").random.default_rng(1))
  return {"scatter": lambda **k: scatter(v, **k), "plotVectors": lambda **k: plot(v, **k),
          "text": lambda **k: text(v, ["a"] * 5, **k), "quiver": lambda **k: quiver(v, v, **k),
          "plotGrains": lambda **k: plot(grains, **k), "plotEBSD": lambda **k: plot(ebsd, **k),
          "plotBoundary": lambda **k: plot(grains.boundary, **k), "plotTriplePoints": lambda **k: plot(grains.triplePoints, **k)}


@pytest.mark.parametrize("call, option", [(c, o) for c, options in MATLAB_OPTIONS.items() for o in options])
def test_a_MATLAB_option_of_a_plot_is_drawn_or_refused_by_name_never_handed_to_matplotlib(drawn, call, option):
  import matplotlib.pyplot as plt
  try:
    drawn[call](**{option: VALUES.get(option, True)})
  except TypeError as e:
    assert repr(option) in str(e) and "unexpected keyword" not in str(e)
  finally:
    plt.close("all")
