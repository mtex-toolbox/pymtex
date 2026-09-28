import os
import sys

import pytest

from mtex.wizard.desktop import APP_ID, desktopEntry, installDesktopEntry

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="a Linux desktop only")


def test_the_entry_and_the_icon_are_written_once(tmp_path):
  entry = installDesktopEntry(str(tmp_path))
  icon = tmp_path / "icons" / "hicolor" / "scalable" / "apps" / (APP_ID + ".svg")
  assert entry == str(tmp_path / "applications" / (APP_ID + ".desktop")) and icon.is_file()
  text = open(entry, encoding="utf-8").read()
  assert text == desktopEntry(str(icon))
  assert f"Exec={sys.executable} -m mtex.wizard %f" in text and f"StartupWMClass={APP_ID}" in text
  stamp = os.stat(entry).st_mtime_ns
  installDesktopEntry(str(tmp_path))
  assert os.stat(entry).st_mtime_ns == stamp


def test_an_entry_of_another_python_is_replaced(tmp_path):
  entry = installDesktopEntry(str(tmp_path))
  open(entry, "w").write("[Desktop Entry]\nExec=/old/python -m mtex.wizard\n")
  installDesktopEntry(str(tmp_path))
  assert sys.executable in open(entry).read()
