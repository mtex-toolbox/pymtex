# the family of functions on SO(3), MATLAB's SO3Fun, as the base class of every representation
from mtex import SO3Fun, SO3FunHarmonic, SO3FunHandle, crystalFrame, orientation, uniformODF, unimodalODF


def test_SO3Fun_isTheBaseOfEveryRepresentation():
  cs = crystalFrame("432")
  fs = [unimodalODF(orientation.id(cs)), uniformODF(cs), SO3FunHarmonic(unimodalODF(orientation.id(cs))), SO3FunHandle(lambda o: o.angle(), cs)]
  assert all(isinstance(f, SO3Fun) for f in fs)


def test_SO3Fun_isNoBaseOfAnOrientation():
  assert not isinstance(orientation.id(crystalFrame("432")), SO3Fun)
