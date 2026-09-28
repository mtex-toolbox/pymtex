# the poles of a list of orientations (docs/plans/plotting.md S6), as ../mtexJulia/test/poles.jl pins them
import numpy as np
import pytest

from mtex import Miller, crystalFrame, degree, orientation, rotation, specimenFrame, vector3d, xvector, zvector
from mtex.plotting.poles import dynamicMarkerSize, inversePoles, orbit, poles

rng = np.random.default_rng(6)


# five cubic orientations under orthorhombic specimen symmetry
def cubicOrientations():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  return cs, ss, orientation(rotation.rand(5, rng=rng), cs, ss)


def test_poles_oneColumnPerDistinctEquivalentAndSpecimenElement():
  cs, ss, os_ = cubicOrientations()
  ps = poles(os_, Miller(1, 0, 0, cs))
  # one column per distinct equivalent of h and per element of the specimen group, an axis
  # and its antipode counted once, in the specimen frame, as axes since the pole figure
  # cannot tell h from -h (MATLAB ori.SS * (ori * symmetrise(h)))
  assert ps.shape == (5, 3 * ss.numSym()) and ps.frame is ss and ps.antipodal
  assert poles(os_, Miller(1, 1, 1, cs)).shape == (5, 4 * ss.numSym())
  assert poles(os_, Miller(1, 2, 3, cs)).shape == (5, 24 * ss.numSym())


def test_poles_areSpecimenElementsTimesOrientationOfEquivalents():
  cs, ss, os_ = cubicOrientations()
  h = Miller(1, 0, 0, cs)
  ps = poles(os_, h)
  # every pole is a specimen element times the orientation applied to an equivalent of h
  o = os_[0]
  eq = h.symmetrise(unique=True, antipodal=True)
  images = [(rotation(s) * (rotation(o) * vector3d(eq[i].data))).data.reshape(3) for i in range(eq.size) for s in ss.rotations()]
  for k in range(ps.shape[1]):
    got = ps[0, k].data.reshape(3)
    assert min(np.linalg.norm(np.abs(got) - np.abs(x)) for x in images) < 1e-9


def test_poles_areDirectionsWhereNoElementReversesTheDirection():
  ss = specimenFrame("222")
  # a crystal without inversion and a direction no element reverses gives directions, not
  # axes; the two-folds of 321 reverse c, the three-fold of 3 does not
  cs3 = crystalFrame("3")
  o3 = orientation(rotation.rand(2, rng=rng), cs3, ss)
  assert not poles(o3, Miller(0, 0, 0, 1, cs3)).antipodal and poles(o3, Miller(0, 0, 0, 1, cs3)).shape == (2, ss.numSym())
  assert poles(o3, Miller(1, 0, -1, 0, cs3)).shape == (2, 3 * ss.numSym()) and not poles(o3, Miller(1, 0, -1, 0, cs3)).antipodal
  assert poles(o3, Miller(0, 0, 0, 1, cs3), antipodal=True).antipodal
  o32 = orientation(rotation.rand(2, rng=rng), crystalFrame("321"), ss)
  assert poles(o32, Miller(0, 0, 0, 1, crystalFrame("321"))).antipodal


def test_poles_refusesOtherFrameAndSeveralDirections():
  cs, ss, os_ = cubicOrientations()
  # a direction of another frame is refused, and one at a time
  with pytest.raises(ValueError):
    poles(os_, vector3d(xvector, frame=ss))
  with pytest.raises(ValueError):
    poles(os_, Miller([1, 1], [0, 1], [0, 1], cs))


def test_inversePoles_areTheOrbitOfTheSpecimenDirectionInTheCrystalFrame():
  cs, ss, os_ = cubicOrientations()
  hs = inversePoles(os_, zvector)
  # z is fixed by the two-fold about it, so one column, in the crystal frame, an axis
  assert hs.shape == (5, 1) and hs.frame is cs and hs.antipodal
  assert inversePoles(os_, vector3d(1, 2, 3)).shape == (5, 4)      # a general orbit of 222
  o = os_[1]
  assert np.linalg.norm(np.abs(hs[1, 0].data) - np.abs((rotation(o).inv() * zvector).data)) < 1e-9


def test_inversePoles_withoutSpecimenSymmetryIsOneDirection():
  cs = crystalFrame("m-3m")
  # without specimen symmetry the specimen direction is one direction and not an axis
  o1 = orientation(rotation.rand(3, rng=rng), cs, specimenFrame("1"))
  assert inversePoles(o1, zvector).shape == (3, 1) and not inversePoles(o1, zvector).antipodal


def test_orbit_sizes():
  cs = crystalFrame("m-3m")
  assert orbit([1, 0, 0], cs).shape == (6, 3) and orbit([1, 0, 0], cs, antipodal=True).shape == (3, 3)
  assert orbit([1, 2, 3], cs).shape == (48, 3) and orbit([1, 0, 0], specimenFrame("1")).shape == (1, 3)
  assert orbit([1, 0, 0], specimenFrame("222"), antipodal=True).shape == (1, 3)


def test_dynamicMarkerSize_shrinksWithTheCountBetweenBounds():
  assert dynamicMarkerSize(0) == 10 and dynamicMarkerSize(40) == 10
  assert 1.5 < dynamicMarkerSize(5000) < 5 and dynamicMarkerSize(10 ** 7) == 1.5
