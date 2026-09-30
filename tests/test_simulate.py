# a map whose answer is known, MATLAB simulateEBSD and the Julia simulate.jl
import numpy as np
import pytest

from mtex import angle, calcGrains, crystalFrame, degree, orientation, simulateEBSD, vector3d, xvector, yvector


def job():
  eS = simulateEBSD()
  eS.xdim, eS.ydim = 60, 30
  eS.CS = crystalFrame("mmm", mineral="Kryptonite")
  eS.ori0 = orientation.byEuler(np.array([0, 45, 0]) * degree, eS.CS)
  return eS.makeMap()


def test_the_map_of_ori0():
  eS = job()
  e = eS.EBSDsim
  assert e.isGridded and e.shape == (30, 60) and e.CS is eS.CS and eS.CS.mineral == "Kryptonite"
  assert np.max(np.asarray(angle(eS.ori0, e.orientations))) < 1e-7
  assert np.allclose(e.extent(), [1, 60, 1, 30]) and np.all(e.emptyProp == 1)
  # the default job: a triclinic kryptonite of 100 x 100 pixels
  d = simulateEBSD()
  assert d.CS.pointGroup == "1" and d.CS.mineral == "kryptonite" and d.makeMap().EBSDsim.shape == (100, 100)


@pytest.mark.parametrize("fun", ["uniform", "logn"])
def test_the_noise_stays_below_its_maximum(fun):
  eS = job()
  eS.noiseFun, eS.noiseMax, eS.rng = fun, 2 * degree, np.random.default_rng(3)
  eS.addNoise()
  a = np.asarray(angle(eS.ori0, eS.EBSDsim.orientations))
  assert a.max() <= 2 * degree + 1e-9 and a.max() > 1.5 * degree
  # the lognormal noise is mostly small, the uniform spread over the whole range
  assert (np.median(a) < 0.1 * degree) if fun == "logn" else (0.6 * degree < np.median(a) < 1.4 * degree)
  eS.noiseFun = "gauss"
  with pytest.raises(ValueError):
    eS.addNoise()


def test_a_step_is_a_boundary_of_the_step_angle():
  eS = job()
  eS.axS, eS.moriAngle = yvector, 3 * degree
  eS.addFeature('singleStep')
  e = eS.EBSDsim
  dom = eS.domainID
  assert dom.shape == e.shape and 0.4 < dom.mean() < 0.6
  a = np.asarray(angle(eS.ori0, e.orientations))
  assert np.allclose(a[dom], 3 * degree) and np.allclose(a[~dom], 0, atol=1e-7)
  grains = calcGrains(e, angle=[10 * degree, 1 * degree])
  ib = grains.innerBoundary
  byId = np.zeros(e.size, dtype=bool)
  byId[e.id.reshape(-1)] = dom.reshape(-1)
  step = ib[np.flatnonzero(byId[ib.ebsdId[:, 0]] ^ byId[ib.ebsdId[:, 1]])]
  assert len(step) == len(ib) and np.allclose(step.misorientation.angle(), 3 * degree)


def test_a_gradient_and_a_subgrain():
  eS = job()
  eS.axS, eS.gradDir, eS.moriAngle = yvector, xvector, 0.1 * degree
  eS.addFeature('simpleGradient')
  e = eS.EBSDsim
  a = np.asarray(angle(eS.ori0, e.orientations))
  # the turn grows by moriAngle per step along x and not at all along y
  assert np.allclose(a, e.pos.x * 0.1 * degree)
  eS = job()
  eS.axS, eS.moriAngle = vector3d(0, 0, 1), 5 * degree
  eS.addFeature('circularSubgrain')
  inside = np.hypot(eS.EBSDsim.pos.x - 30, eS.EBSDsim.pos.y - 15) <= 7.5 - 1e-6
  assert np.all(eS.domainID[inside]) and eS.domainID.sum() < 2 * inside.sum()
  a = np.asarray(angle(eS.ori0, eS.EBSDsim.orientations))
  assert np.allclose(a[eS.domainID], 5 * degree)


def test_an_unknown_feature_is_refused():
  eS = simulateEBSD()
  with pytest.raises(ValueError):
    eS.addFeature('spiral')
