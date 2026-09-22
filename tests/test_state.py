import os

from mtex.wizard.state import FileRow, WizardState


def test_state_and_hooks():
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


def test_file_row():
  r = FileRow("a.ang", "/x/a.ang", 1, size=12)
  assert not r.isDir and not r.open and r.depth == 1
