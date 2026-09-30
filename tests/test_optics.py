# optics: the refractive index, the birefringence, the optical axes and the colour of a
# birefringent plate (docs/plans/functions.md P4d: T8 and P4i after the Julia tests)
import numpy as np

from mtex import (Miller, birefringence, crystalFrame, degree, opticalAxis, refractiveIndex, refractiveIndexTensor, rotation,
                  spectralTransmission, vector3d, xvector, yvector, zvector)

rng = np.random.default_rng(24)
OLIVINE = crystalFrame("mmm", [4.762, 10.225, 5.994], mineral="olivine")


def olivineIndex():
  return refractiveIndexTensor(np.diag([1.670, 1.635, 1.651]), OLIVINE)


def test_refractiveIndex_alongThePrincipalAxes():
  rI = olivineIndex()
  assert np.isclose(refractiveIndex(rI, xvector), 1.670) and np.isclose(rI.refractiveIndex(Miller(0, 1, 0, OLIVINE)), 1.635)


def test_birefringence_alongAPrincipalAxis():
  # along a principal axis the two indices are the other two, the polarisations the axes
  dn, fast, slow = birefringence(olivineIndex(), zvector)
  assert np.isclose(dn, 1.670 - 1.635) and fast.antipodal and np.isclose(abs(fast.data[1]), 1) and np.isclose(abs(slow.data[0]), 1)


def test_opticalAxis_ofABiaxialCrystalAreTwoWithoutBirefringence():
  # along the optical axes there is none, and there are two of them for a biaxial crystal
  rI = olivineIndex()
  axes = opticalAxis(rI)
  assert axes.size == 2 and axes.antipodal and np.allclose(birefringence(rI, axes)[0], 0, atol=1e-12)
  assert np.isclose(axes[0].angle(axes[1], symmetry=False), 2 * np.arctan(np.sqrt((1.651 - 1.635) / (1.670 - 1.651))), atol=1e-9)


def test_opticalAxis_ofAUniaxialCrystalIsTheDistinguishedAxis():
  # a uniaxial crystal has one, along the distinguished axis
  uni = refractiveIndexTensor(np.diag([1.5, 1.5, 1.7]))
  assert opticalAxis(uni).size == 1 and np.isclose(abs(opticalAxis(uni).data[2]), 1) and opticalAxis(refractiveIndexTensor(np.eye(3))).size == 0


def test_birefringence_atAListOfDirectionsAndOnTheSphere():
  # a list of directions, and the fields on the sphere
  rI = olivineIndex()
  vs = vector3d.rand(5, rng=rng)
  dns, fasts, slows = birefringence(rI, vs)
  assert dns.shape == (5,) and fasts.shape == (5,) and np.allclose(fasts.dot(vs, symmetry=False), 0, atol=1e-12) and np.allclose(fasts.dot(slows, symmetry=False), 0, atol=1e-12)
  dF, fF, sF = birefringence(rI)
  v = Miller(1, 1, 1, OLIVINE)
  assert np.isclose(np.real(dF(v)), birefringence(rI, v)[0]) and np.isclose(abs(fF(v).dot(birefringence(rI, v)[1])), 1)
  assert np.allclose(rI.birefringence(v)[0], birefringence(rI, v)[0])


def test_spectralTransmission_takesThePolariserAsDirectionOrAngle():
  rI = olivineIndex()
  v = Miller(1, 1, 1, OLIVINE)
  rgb = spectralTransmission(rI, v, 30000, tau=30 * degree)
  assert rgb.shape == (3,) and np.all(np.isfinite(rgb))
  # the polariser direction and the angle to the fast wave say the same thing
  _, fast, _ = birefringence(rI, v)
  polarizer = rotation.byAxisAngle(vector3d(v.data), 30 * degree) * vector3d(fast.data)
  assert np.allclose(spectralTransmission(rI, v, 30000, polarizationDirection=polarizer), rgb)


def test_spectralTransmission_crossedPassesNothingParallelEverything():
  # crossed polarisers with the fast wave along the polariser pass nothing, parallel ones everything
  rI = olivineIndex()
  v = Miller(1, 1, 1, OLIVINE)
  dark = spectralTransmission(rI, v, 30000, tau=0)
  bright = spectralTransmission(rI, v, 30000, tau=0, phi=0)
  assert np.allclose(dark, 0, atol=1e-12) and np.all(bright > 0)


def test_spectralTransmission_givesOneColourPerDirection():
  # a list of directions gives one colour each, with one polariser or one per direction
  rI = olivineIndex()
  vs = vector3d.rand(4, rng=rng)
  assert spectralTransmission(rI, vs, 22800).shape == (4, 3)
  assert spectralTransmission(rI, vs, 22800, polarizationDirection=xvector).shape == (4, 3)


def test_spectralTransmission_onTheSphereAgreesWithTheDirections():
  # the three channels on the sphere agree with the direction route
  rI = olivineIndex()
  v = Miller(1, 1, 1, OLIVINE)
  channels = spectralTransmission(rI, 30000, tau=15 * degree)
  direct = spectralTransmission(rI, v, 30000, tau=15 * degree)
  assert len(channels) == 3 and channels[0].antipodal and channels[0].frame is OLIVINE
  assert np.allclose([np.real(c(v)) for c in channels], direct, atol=2e-2)
  assert np.allclose([np.real(c(v)) for c in rI.spectralTransmission(30000, tau=15 * degree)], direct, atol=2e-2)


def test_a_list_of_tensors_pairs_with_the_directions():
  from mtex import orientation
  rI = olivineIndex()
  ori = orientation(rotation.rand(6, rng=rng), OLIVINE)
  # rotating the tensor into the specimen or the directions into the crystal is the same
  byTensor = (ori * rI).spectralTransmission(zvector, 22800, polarizationDirection=xvector)
  byDirection = rI.spectralTransmission(ori.inv() * zvector, 22800, polarizationDirection=ori.inv() * xvector)
  assert byTensor.shape == (6, 3) and np.allclose(byTensor, byDirection, atol=1e-10)
  dn, fast, _ = birefringence(ori * rI, zvector)
  assert dn.shape == (6,) and fast.shape == (6,) and np.allclose(dn, birefringence(rI, ori.inv() * zvector)[0])


def test_opticalAxis_showsItsCoordinates():
  # the optical axes show their coordinates
  assert opticalAxis(olivineIndex()).dispStyle == "xyz"
