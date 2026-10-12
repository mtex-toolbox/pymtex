import os

# the tests see the defaults, never the settings file of whoever runs them
os.environ["MTEX_SETTINGS"] = ""

import numpy as np
import pytest

import mtex


# skip a test whose classes are not built yet, so the semantics file is complete before the code
def need(*names):
  missing = [n for n in names if not hasattr(mtex, n)]
  if missing:
    pytest.skip(f"not built yet: {', '.join(missing)}")


@pytest.fixture(autouse=True)
def fresh_session():
  reset = getattr(mtex, "reset", None)
  if reset is not None:
    reset()
  yield
  releaseWidgets()


# the widgets a test left, closed and destroyed on the main thread, their children with them, so
# that the Python wrappers the garbage collector frees later hold nothing: it runs on whichever
# thread allocates, and a widget it frees on a worker (one of SciPy's query threads) waits in
# ~QWidget for the GUI thread, which waits in join for that worker
def releaseWidgets():
  import sys
  if "PySide6.QtWidgets" not in sys.modules:
    return
  from PySide6.QtCore import QCoreApplication, QEvent
  from PySide6.QtWidgets import QApplication
  app = QApplication.instance()
  left = app.topLevelWidgets() if app is not None else []
  if not left:
    return
  for w in left:
    w.close()
    w.deleteLater()
  QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def close(a, b, tol=1e-12):
  return np.allclose(np.asarray(a), np.asarray(b), atol=tol, rtol=0)


IN100 = os.path.join(os.path.dirname(__file__), "..", "..", "master", "data", "EBSD", "SmallIN100_MeshStats.dream3d")


# the IN100 volume of ../master/data and its grains at 10 degrees, as MATLAB's pages segment it,
# built once per run; the tests only read them
@pytest.fixture(scope="session")
def in100():
  if not os.path.exists(IN100):
    pytest.skip("the IN100 volume of ../master/data")
  from mtex import EBSD3, calcGrains, degree
  ebsd = EBSD3.load(IN100)
  return ebsd, calcGrains(ebsd, angle=10 * degree)
