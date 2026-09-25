# the texture characteristics (docs/plans/functions.md F6: C1 to C7, check_calcComponents,
# check_axisDistribution, and the definitions checked on functions with known answers)
import numpy as np
import pytest
from numpy import pi

from mtex import (S2FunHandle, Miller, S2DeLaValleePoussinKernel, S2FunBingham, S2FunHarmonic, SO3DeLaValleePoussinKernel, SO3FunHarmonic,
                  angle, bandwidthFor, calcAngleDistribution, calcAxisDistribution, calcComponents, calcDensity, calcError,
                  calcKernel, calcMDF, calcMIndex, centerSpecimen, conv, crystalFrame, degree, discreteSample, entropy,
                  equispacedS2Grid, fibreODF, fibreVolume, max, mean, min, norm, orientation, quadratureSO3Grid, rotate,
                  rotation, specimenFrame, textureindex, unimodalODF, uniformODF, vector3d, volume, yvector, zvector)

rng = np.random.default_rng(71)


def test_C1_scalars():
  cs, ss = crystalFrame("m-3m"), specimenFrame("1")
  u = uniformODF(cs, ss)
  assert np.isclose(textureindex(u), 1) and abs(entropy(u)) < 1e-10
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=15 * degree)
  H = SO3FunHarmonic(odf)
  assert np.isclose(textureindex(H), norm(H) ** 2) and np.isclose(textureindex(odf), textureindex(H), rtol=1e-6)
  # the entropy of a density is at most zero, and -int f log f by an independent rule
  e = entropy(H)
  assert e < 0
  g = quadratureSO3Grid(bandwidthFor(80, cs, ss), cs, ss)
  v = H(g.nodes())
  assert np.isclose(e, -np.sum(g.weights * np.where(v > 0, v * np.log(np.where(v > 0, v, 1)), 0)), rtol=1e-3)
  assert np.isclose(entropy(odf), e, rtol=1e-2) and np.isclose(H.entropy(), e)
  # the volume of a ball around the mode of a sharp unimodal ODF is nearly one, of the
  # uniform ODF the share of the ball
  c = orientation(rotation.rand(rng=rng), cs, ss)
  sharp = unimodalODF(c, halfwidth=5 * degree)
  assert 0.95 < volume(sharp, c, 20 * degree) <= 1
  assert np.isclose(volume(u, c, 10 * degree), 24 * (10 * degree - np.sin(10 * degree)) / pi, rtol=1e-2)
  assert volume(u, c, 60 * degree) == 1
  # the fibre volume of a fibre ODF around its own fibre
  h, r = Miller(1, 1, 1, cs), vector3d.rand(rng=rng)
  fodf = fibreODF(h, r, ss, halfwidth=5 * degree)
  assert fibreVolume(fodf, h, r, 15 * degree) > 0.9
  assert np.isclose(fibreVolume(u, h, r, 10 * degree), min(1, 8 * (1 - np.cos(10 * degree)) / 2), rtol=5e-2)
  s = S2FunHarmonic(S2DeLaValleePoussinKernel(halfwidth=20 * degree))
  assert np.isclose(textureindex(s), norm(s) ** 2) and entropy(s) < 0
  assert abs(entropy(S2FunHarmonic([np.sqrt(4 * pi)]))) < 1e-10


def test_C2_extrema():
  cs, ss = crystalFrame("432"), specimenFrame("1")
  c1, c2 = orientation(rotation.rand(rng=rng), cs, ss), orientation(rotation.rand(rng=rng), cs, ss)
  odf = 0.7 * unimodalODF(c1, halfwidth=8 * degree) + 0.3 * unimodalODF(c2, halfwidth=8 * degree)
  # one maximum is a value and an orientation
  v, o = max(odf)
  assert isinstance(v, float) and isinstance(o, orientation)
  assert angle(o, c1) < 0.5 * degree and np.isclose(v, odf(c1), rtol=1e-3)
  assert odf.max()[0] == v
  # several are arrays, the highest first
  v2, o2 = max(odf, numLocal=2)
  assert len(v2) == 2 and v2[0] >= v2[1] and angle(o2[1], c2) < 0.5 * degree
  H = SO3FunHarmonic(odf)
  vh, oh = max(H)
  assert angle(oh, c1) < 0.5 * degree and np.isclose(vh, H(oh), rtol=1e-6) and np.isclose(vh, v, rtol=5e-2)
  vm, om = min(odf)
  assert vm < 1e-3 and np.isclose(vm, odf(om))
  # on the sphere
  s = S2FunHarmonic(rotate(S2FunBingham([-10.0, -4.0, 0.0]), rotation.byAxisAngle(yvector, 0.4)), bandwidth=30)
  vs, ps = max(s)
  assert angle(ps, rotation.byAxisAngle(yvector, 0.4) * zvector) < 0.3 * degree and np.isclose(vs, s(ps), rtol=1e-5)
  cs6 = crystalFrame("6/mmm")
  sf = S2FunHarmonic(S2DeLaValleePoussinKernel(halfwidth=20 * degree), cs6)
  vf, pf = max(sf)
  assert pf.frame is cs6 and abs(vector3d(pf.data).dot(zvector)) > np.cos(0.3 * degree)
  # several maxima of a symmetric function are distinct orbits, not copies of one mode
  cs = crystalFrame("m-3m")
  adf = S2FunHarmonic(S2FunHandle(calcAxisDistribution(cs), cs, bandwidth=64, antipodal=True))
  va, pa = max(adf, numLocal=2)
  assert pa.size == 2 and np.all(np.diff(va) <= 0)
  for i in range(2):
    for j in range(i + 1, 2):
      assert angle(pa[i], pa[j]) > 2 * degree
  assert np.isclose(va[0], np.max(np.real(adf(equispacedS2Grid(resolution=0.5 * degree)))), rtol=1e-3)
  assert max([3, 1, 2]) == 3 and min(4, 2) == 2


def test_C3_components():
  cs = crystalFrame("m-3m")
  c1, c2 = orientation(rotation.byEuler(0, 0, 0, "Bunge"), cs), orientation(rotation.byEuler(30 * degree, 40 * degree, 50 * degree, "Bunge"), cs)
  odf = 0.6 * unimodalODF(c1, halfwidth=6 * degree) + 0.4 * unimodalODF(c2, halfwidth=6 * degree)
  # seeded, with a radius
  seed = orientation(rotation.rand(20, rng=rng), cs, c1.SS)
  modes, weights, centerId = calcComponents(odf, seed=seed, radius=5 * degree, exact=True)
  assert len(weights) == modes.size and np.all((0 <= centerId) & (centerId < modes.size)) and len(centerId) == seed.size
  tm, tw, _ = calcComponents(odf, seed=seed, radius=1e-4 * degree, exact=True)
  assert len(tw) == tm.size and np.sum(tw) < 1e-4
  # the weights sum to one over the full radius
  assert abs(np.sum(calcComponents(odf, radius=pi, exact=True)[1]) - 1) < 1e-10
  # the known components: two modes at the centres with weights 0.6 and 0.4
  odf7 = 0.6 * unimodalODF(c1, halfwidth=7 * degree) + 0.4 * unimodalODF(c2, halfwidth=7 * degree)
  modes, weights, centerId = calcComponents(odf7)
  assert modes.size == 2
  assert np.min(angle(modes, c1)) < 1 * degree and np.min(angle(modes, c2)) < 1 * degree
  assert np.max(np.abs(np.sort(weights)[::-1] - [0.6, 0.4])) < 0.02 and np.all(centerId >= 0)
  # the centres of a radial basis function are its own seeds
  rb = unimodalODF(orientation(rotation.cat(rotation(c1), rotation(c2)), cs), halfwidth=7 * degree, weights=[0.6, 0.4])
  om, ow, _ = rb.calcComponents()
  assert om.size == 2 and np.max(np.abs(np.sort(ow)[::-1] - [0.6, 0.4])) < 0.02


def test_C4_misorientations():
  cs = crystalFrame("mmm")
  # the uniform angle distribution has mean one; Mackenzie's mean angle of the cubic one; the
  # uniform ODF gives the same, a list of random misorientations a histogram of it in percent
  p, omega = calcAngleDistribution(cs, cs)
  assert np.isclose(np.mean(p), 1)
  pc, oc = calcAngleDistribution(crystalFrame("m-3m"), crystalFrame("m-3m"))
  assert np.isclose(oc[-1], 62.8 * degree, atol=0.01 * degree)
  assert np.isclose(np.sum(oc * pc) / np.sum(pc), 40.73 * degree, rtol=1e-2)
  pf, of = calcAngleDistribution(uniformODF(cs, cs), resolution=2 * degree)
  assert np.allclose(of, omega) and np.mean(np.abs(pf - p)) < 0.01
  h, mid = calcAngleDistribution(orientation.rand(20000, cs, cs, rng=rng))
  assert np.isclose(np.sum(h), 100) and np.mean(np.abs(h / 100 * len(h) - np.interp(mid, omega, p))) < 0.1
  # the MDF of the uniform ODF is uniform, its M-index zero
  assert calcMIndex(uniformODF(cs), resolution=2 * degree) < 2e-2
  m = calcMDF(uniformODF(cs))
  assert m.antipodal and np.isclose(mean(m), 1) and m.CS is cs and m.SS is cs
  assert np.max(np.abs(m(rotation.rand(20, rng=rng)) - 1)) < 1e-6
  # the MDF of a sharp unimodal ODF concentrates at the identity: a large M-index
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs), halfwidth=5 * degree)
  assert calcMIndex(odf, resolution=2 * degree) > 0.5
  m2 = calcMDF(odf, odf)
  assert m2(rotation.id) > 10 * m2(rotation.rand(rng=rng))
  # two different ODFs give a misorientation distribution between the crystal frames
  cs2 = crystalFrame("4/mmm")
  odf2 = unimodalODF(orientation(rotation.rand(rng=rng), cs2), halfwidth=15 * degree)
  m12 = calcMDF(odf, odf2)
  assert m12.CS is cs2 and m12.SS is cs and not m12.antipodal and np.isclose(mean(m12), 1)


def test_C5_axis_distribution():
  cs = crystalFrame("mmm")
  h = vector3d(equispacedS2Grid(resolution=20 * degree, antipodal=True).data)
  odf = 0.4 * uniformODF(cs, cs, antipodal=True) + 0.6 * unimodalODF(orientation(rotation.byEuler(20 * degree, 30 * degree, 10 * degree, "Bunge"), cs, cs, antipodal=True), halfwidth=15 * degree)
  whole = calcAxisDistribution(odf, h, resolution=2 * degree)
  assert np.array_equal(whole, calcAxisDistribution(odf, h, resolution=2 * degree, minAngle=0, maxAngle=np.inf))
  lower = calcAxisDistribution(odf, h, resolution=2 * degree, maxAngle=60 * degree)
  upper = calcAxisDistribution(odf, h, resolution=2 * degree, minAngle=60 * degree)
  assert np.max(np.abs(lower + upper - whole)) / np.max(whole) < 5e-3
  tol = 1e-3 * np.max(whole)
  assert np.all(lower <= whole + tol) and np.all(upper <= whole + tol) and np.any(lower > tol) and np.any(upper > tol)
  # the uniform ODF against the integral of the uniform density on a fine rule
  u = uniformODF(cs, cs, antipodal=True)
  fine = calcAxisDistribution(u, h, resolution=0.5 * degree)
  coarse = calcAxisDistribution(u, h, resolution=2 * degree)
  assert np.max(np.abs(coarse - fine)) / np.max(fine) < 3e-2
  win = calcAxisDistribution(u, h, resolution=2 * degree, minAngle=20 * degree, maxAngle=40 * degree)
  winf = calcAxisDistribution(u, h, resolution=0.5 * degree, minAngle=20 * degree, maxAngle=40 * degree)
  assert np.max(np.abs(win - winf)) / np.max(winf) < 3e-2
  # the inverted window is empty
  assert np.max(np.abs(calcAxisDistribution(odf, h, resolution=2 * degree, minAngle=40 * degree, maxAngle=20 * degree))) < 1e-10
  # the axis distribution is a spherical function of mean one in the crystal frame
  adf = calcAxisDistribution(u, resolution=2.5 * degree, bandwidth=32)
  assert adf.frame is cs and adf.antipodal and np.isclose(mean(adf), 1, rtol=2e-2)


def test_C6_sampling():
  cs, ss = crystalFrame("432"), specimenFrame("1")
  c = orientation(rotation.rand(rng=rng), cs, ss)
  odf = unimodalODF(c, halfwidth=10 * degree)
  o = discreteSample(odf, 2000, rng=rng)
  assert isinstance(o, orientation) and o.size == 2000 and o.CS is cs
  # the share of samples near the centre is the volume there, and the density estimated
  # with the same kernel is the ODF convolved with it
  share = volume(odf, c, 20 * degree)
  assert abs(volume(o, c, 20 * degree) - share) < 0.05
  est = calcDensity(o, halfwidth=10 * degree)
  assert np.isclose(est(c), conv(odf, SO3DeLaValleePoussinKernel(halfwidth=10 * degree))(c), rtol=0.15)
  # a composition of kernel forms and a harmonic function are sampled too
  mix = 0.5 * odf + 0.5 * uniformODF(cs, ss)
  om = discreteSample(mix, 1000, rng=rng)
  assert om.size == 1000 and abs(volume(om, c, 20 * degree) - 0.5 * share) < 0.06
  oh = discreteSample(SO3FunHarmonic(odf), 1000, rng=rng, resolution=3 * degree)
  assert abs(volume(oh, c, 20 * degree) - share) < 0.06
  s = S2FunHarmonic(S2DeLaValleePoussinKernel(halfwidth=10 * degree))
  d = discreteSample(s, 1000, rng=rng)
  assert d.size == 1000 and np.count_nonzero(d.angle(zvector) < 20 * degree) > 900
  assert discreteSample(S2FunHarmonic(s, antipodal=True), 5, rng=rng).antipodal
  # a symmetric density is sampled over the whole sphere, the sector nodes spread by the group
  adf = calcAxisDistribution(crystalFrame("m-3m"))
  da = discreteSample(adf, 4000, rng=rng)
  assert da.antipodal and np.count_nonzero(da.z < 0) > 100 and np.mean(np.real(adf(da))) > 1.05


def test_C7_kernel_selection_and_specimen_centering():
  cs = crystalFrame("321")
  model = 0.5 * uniformODF(cs) + 0.5 * unimodalODF(orientation(rotation.byEuler(35 * degree, 45 * degree, 0.0, "Bunge"), cs), halfwidth=10 * degree)
  ori = discreteSample(model, 2000, rng=rng)
  hw = {m: calcKernel(ori, m, rng=rng).halfwidth() / degree for m in ("klcv", "ruleOfThumb", "magicRule")}
  # every rule gives a usable width, the cross validation the narrowest
  assert all(2 <= v <= 40 for v in hw.values()) and hw["klcv"] <= hw["magicRule"]
  assert calcError(calcDensity(ori, kernel=calcKernel(ori, rng=rng)), model) < 0.25
  with pytest.raises(ValueError):
    calcKernel(ori, "nothing")
  # an orthotropic texture mounted askew comes back
  cubic, ss = crystalFrame("m-3m"), specimenFrame("222")
  odf = unimodalODF(orientation(rotation.byEuler(np.array([135, 60]) * degree, np.array([45, 54.73]) * degree, np.array([120, 45]) * degree, "Bunge"), cubic, ss), halfwidth=12 * degree)
  mount = rotation.byEuler(15 * degree, 12 * degree, -5 * degree, "Bunge")
  os = discreteSample(odf, 1000, rng=rng)
  tilted = calcDensity(orientation(mount * rotation(os), cubic), halfwidth=10 * degree)
  _, correction, v1, v2 = centerSpecimen(tilted)
  assert abs(v1.dot(v2)) < 1e-6
  assert angle(mount, correction.inv()) < 5 * degree


# the compiled sampler of a radial basis function draws from the same distribution as the
# NumPy form: the moments up to degree 4 of both samples agree to their sampling error, with
# symmetry on both sides and a uniform part, and for an exchange symmetric function
def test_rbf_sampler_kernel_against_numpy():
  from mtex import setMTEXpref
  rng = np.random.default_rng(11)
  cs, ss = crystalFrame("321"), specimenFrame("222")
  mixed = 0.7 * unimodalODF(orientation(rotation.rand(3, rng=rng), cs, ss), halfwidth=12 * degree) + 0.3 * uniformODF(cs, ss)
  exchange = unimodalODF(orientation(rotation.rand(2, rng=rng), cs, cs, antipodal=True), halfwidth=15 * degree)
  assert exchange.antipodal
  for f in (mixed, exchange):
    moments = {}
    for kernels in ("numba", "numpy"):
      setMTEXpref("kernels", kernels)
      o = discreteSample(f, 200000, rng=np.random.default_rng(3))
      moments[kernels] = SO3FunHarmonic.quadrature(o, np.ones(o.size), bandwidth=4).fhat
    setMTEXpref("kernels", "numba")
    assert np.max(np.abs(moments["numba"] - moments["numpy"])) < 0.05
