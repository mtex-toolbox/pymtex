# the spellings docs/plans/syntax.md replaced stay replaced (tools/syntaxcheck.py)
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("syntaxcheck", os.path.join(ROOT, "tools", "syntaxcheck.py"))
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


def test_no_public_keyword_is_spelled_the_old_way():
  assert sc.keywordsLeft() == []


def test_no_old_name_is_public():
  assert sc.namesLeft() == []
