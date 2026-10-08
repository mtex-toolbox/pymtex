import pytest

import mtex
from mtex.io.completer import openStringArgument, registerCompleter, stringCompletions


def complete(text):
  return stringCompletions(text, vars(mtex))


def test_a_call_offers_the_values_of_its_literal_annotation():
  assert complete('plottingConvention.default("') == ("", ["y↑→x", "y↓→x", "x↑→y", "x↓→y", "x←↑y", "x←↓y", "y←↑x", "y←↓x"])


def test_the_typed_part_filters_the_values():
  assert complete("crystalFrame('4/m") == ("4/m", ["4/m", "4/mmm", "4/m2/m2/m", "4/m-32/m"])


def test_the_typed_part_matches_whatever_its_case():
  assert complete('openDoc("ebsdtu') == ("ebsdtu", ["EBSDTutorial"])


def test_open_doc_offers_the_pages_and_the_chapters():
  assert complete('openDoc("Tens') == ("Tens", ["TensorArithmetics", "TensorAverage", "TensorDefinition", "TensorImport",
                                                "TensorVisualisation", "Tensors"])


def test_a_keyword_unpacked_from_the_plot_keywords_offers_its_values():
  assert complete('plot(v, color="re') == ("re", ["rebeccapurple", "red"])


def test_the_overloads_offer_the_values_of_each():
  assert complete('mtexdata("forst') == ("forst", ["forsterite"])


def test_a_position_beyond_the_named_ones_takes_the_star_annotation():
  assert complete('orientation.byEuler(1, 2, 3, "Bu') == ("Bu", ["Bunge"])


def test_an_argument_without_literal_annotation_offers_nothing():
  assert complete('crystalFrame("m-3m", mineral="') is None


def test_a_string_outside_a_call_is_no_argument():
  assert openStringArgument('x = "m') is None


def test_a_string_after_a_starred_argument_has_no_position():
  assert openStringArgument('f(*a, "m') == ("f", "(", None, "m")


def test_a_call_on_a_call_result_is_not_resolved():
  assert openStringArgument('ebsd("indexed").plot(color="r') is None


def test_a_call_over_several_lines_is_read():
  assert openStringArgument('crystalFrame(\n  "m-3') == ("crystalFrame", "(", 0, "m-3")


def test_a_closed_string_before_the_cursor_counts_as_an_argument():
  assert openStringArgument("f('a', b='x', '") == ("f", "(", 2, "")


def test_a_subscript_gives_the_elements_before():
  assert openStringArgument("gB['Fo', '") == ("gB", "[", ("Fo",), "")


def test_a_map_offers_its_phase_names():
  ebsd = mtex.mtexdata("forsterite", verbose=False)
  assert stringCompletions('ebsd["', {"ebsd": ebsd}) == ("", ["Forsterite", "Enstatite", "Diopside", "Silicon", "indexed", "notIndexed"])


def test_a_boundary_offers_the_second_phase_after_an_abbreviated_first():
  grains = mtex.calcGrains(mtex.mtexdata("forsterite", verbose=False)["indexed"])
  assert stringCompletions("grains.boundary['fo', 'E", {"grains": grains}) == ("E", ["Enstatite"])


def test_the_ipython_shell_completes_inside_the_quotes():
  globalipapp = pytest.importorskip("IPython.testing.globalipapp")
  from IPython.core.completer import provisionalcompleter
  globalipapp.start_ipython()
  shell = globalipapp.get_ipython()
  shell.user_ns.update(vars(mtex))
  registerCompleter(shell)
  line = 'plottingConvention.default("y↓'
  with provisionalcompleter():
    found = [(c.text, c.start) for c in shell.Completer.completions(line, len(line))]
  assert found == [("y↓→x", len(line) - 2)]
