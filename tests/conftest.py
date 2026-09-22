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


def close(a, b, tol=1e-12):
  return np.allclose(np.asarray(a), np.asarray(b), atol=tol, rtol=0)
