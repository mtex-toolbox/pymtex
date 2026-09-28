# a pytest plugin recording which pymtex functions each test runs, for build.py
#
#   PYTHONPATH=tools/graph .venv/bin/python -m pytest -q -p coverage_python [tests ...]
#
# writes tools/graph/out/coverage_python.json: {test id: [[file, first line, qualified name], ...]},
# and tools/graph/out/durations_python.json: {test id: seconds}.
# sys.monitoring reports each function's first start per test and is then silenced for it, so
# the recording costs about one callback per function and test.
from __future__ import annotations

import json
import os
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
MTEX = os.path.join(os.path.dirname(os.path.dirname(HERE)), "mtex")
TOOL = sys.monitoring.COVERAGE_ID
seen, results, durations = set(), {}, {}


def started(code, offset):
  if code.co_filename.startswith(MTEX):
    seen.add((code.co_filename, code.co_firstlineno, code.co_qualname))
  return sys.monitoring.DISABLE


def pytest_configure(config):
  sys.monitoring.use_tool_id(TOOL, "mtex-graph")
  sys.monitoring.register_callback(TOOL, sys.monitoring.events.PY_START, started)
  sys.monitoring.set_events(TOOL, sys.monitoring.events.PY_START)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
  seen.clear()
  sys.monitoring.restart_events()
  t = time.perf_counter()
  yield
  durations[item.nodeid] = time.perf_counter() - t
  results[item.nodeid] = sorted(seen)


def pytest_unconfigure(config):
  sys.monitoring.set_events(TOOL, 0)
  sys.monitoring.free_tool_id(TOOL)
  os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
  path = os.path.join(HERE, "out", "coverage_python.json")
  old = json.load(open(path)) if os.path.exists(path) else {}
  json.dump(old | results, open(path, "w"))
  path = os.path.join(HERE, "out", "durations_python.json")
  old = json.load(open(path)) if os.path.exists(path) else {}
  json.dump(old | durations, open(path, "w"))
