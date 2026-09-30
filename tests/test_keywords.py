# the one spelling of the style keywords of the plots (mtex/plotting/keywords.py)
import pytest

from mtex.plotting.keywords import lineKeywords, refuseVariants


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
