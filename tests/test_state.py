import os

from mtex.wizard.state import WizardState


def test_a_state_change_calls_the_hooks_bound_to_it():
  st = WizardState(os.path.dirname(__file__))
  assert st.folder == os.path.abspath(os.path.dirname(__file__)) and st.status[0] == "idle" and not st.hasMap
  seen = []
  st.on("status", lambda: seen.append(st.status[0]))
  st.on("tab", lambda: seen.append("tab"))
  st.status = ("loading", "reading")
  st.changed("status", "tab", "nothing bound")
  assert seen == ["loading", "tab"]
  assert st.currentTab is None and len(st.conventions) == 8
  assert "loading" in repr(st)



def test_currentKind_is_the_kind_of_the_tab_shown():
  st = WizardState()
  st.tabs, st.tabKinds, st.tab = ["phase", "pole figures"], ["phase", "polefigures"], 1
  assert st.currentKind == "polefigures"


def test_currentKind_is_none_without_tabs():
  assert WizardState().currentKind is None
