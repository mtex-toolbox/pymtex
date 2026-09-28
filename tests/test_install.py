import functools
import inspect

import pytest

import mtex
from mtex.geometry import install as installModule
from mtex.geometry.install import INSTALLED, install


# the attachments a test makes are taken back, so that tools/staticapi.py never sees them
@pytest.fixture(autouse=True)
def restoreRecord():
  n = len(INSTALLED)
  yield
  del INSTALLED[n:]


class _target:
  pass


def _twice(x, factor=2):
  """x times factor"""
  return factor * x.value


def test_a_function_becomes_a_method_under_its_own_name():
  install(_target, _twice)
  t = _target()
  t.value = 3
  assert t._twice() == 6 and t._twice(factor=3) == 9
  assert INSTALLED[-1] == (_target, "_twice", _twice, None)


def test_several_classes_and_another_name():
  class other:
    value = 5
  install((_target, other), _twice, "double")
  assert other().double() == 10
  assert [(c, n) for c, n, *_ in INSTALLED[-2:]] == [(_target, "double"), (other, "double")]


def test_static_class_methods_and_properties():
  install(_target, staticmethod(lambda a: a + 1), "succ")
  install(_target, classmethod(lambda cls: cls.__name__), "name")
  install(_target, property(lambda self: 42), "answer")
  assert _target.succ(1) == 2 and _target.name() == "_target" and _target().answer == 42


def test_a_callable_that_is_no_function_binds_the_object():
  install(_target, functools.partial(_twice, factor=4), "quadruple")
  t = _target()
  t.value = 2
  assert t.quadruple() == 8


def test_the_declared_function_is_recorded():
  install(_target, _twice, "shown", declare=abs)
  assert INSTALLED[-1][3] is abs


def test_attached_methods_keep_signature_and_help():
  # plot(v) and v.plot() are one function, KAM keeps its keywords
  assert mtex.EBSD.KAM.__doc__ and "threshold" in inspect.signature(mtex.EBSD.KAM).parameters
  assert "bandwidth" in inspect.signature(mtex.stiffnessTensor.velocity).parameters


def test_every_attachment_is_recorded():
  assert installModule.__all__ == []
  names = {(c.__name__, n) for c, n, *_ in INSTALLED}
  assert {("vector3d", "plot"), ("EBSD", "KAM"), ("SO3FunHarmonic", "calcMDF"), ("crystalShape", "quartz"),
          ("fibre", "alpha"), ("orientation", "Bain"), ("PoleFigure", "load")} <= names
