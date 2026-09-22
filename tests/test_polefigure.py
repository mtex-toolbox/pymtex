# the pole figure container, its arithmetic, the simulation from an ODF, the quadrature
# weights, the errors and the corrections (docs/plans/functions.md P4a: P1 to P8 after the
# Julia tests; check_calcPoleFigureSuperposition on the container)
import numpy as np
import pytest

from mtex import (Miller, PoleFigure, S2FunHarmonic, SchulzDefocusingKernel, calcError, calcErrorPF, calcNormalization,
                  calcPoleFigure, calcQuadratureWeights, correct, crystalFrame, degree, equispacedS2Grid, isOutlier, max, mean, min,
                  noisepf, orientation, pdf, refine, regularS2Grid, rotation, specimenFrame, unimodalODF, uniformODF, union,
                  unique, vector3d, zvector)
from mtex.functions.polefigure import _poleDensity

rng = np.random.default_rng(3)


def test_P1_construction_and_accessors():
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  r = regularS2Grid(resolution=10 * degree, antipodal=True)
  h = Miller(1, 0, 0, cs)
  pf = PoleFigure(h, r, np.ones(r.shape))
  assert pf.numPF == 1 and len(pf) == 1 and pf.allI[0].shape == r.shape
  assert pf.CS is cs and pf.SS is ss and pf.frame is ss
  assert pf.antipodal and pf.allH[0].size == 1 and pf.c[0] == [1.0]
  # the specimen directions of a measurement are axes, unless the measurement is complete
  d = regularS2Grid(resolution=20 * degree)
  assert PoleFigure(h, d, np.ones(d.shape)).antipodal
  assert not PoleFigure(h, d, np.ones(d.shape), complete=True).antipodal
  # a superposition gets one coefficient per direction, the mean by default
  hs = Miller([1, 0], [1, 1], [0, 1], cs)
  pfs = PoleFigure(hs, r, np.ones(r.size))
  assert pfs.numPF == 1 and pfs.allH[0].size == 2 and np.allclose(pfs.c[0], [0.5, 0.5])
  assert np.allclose(PoleFigure(hs, r, np.ones(r.size), superposition=[0.52, 1.23]).c[0], [0.52, 1.23])
  # what does not fit is rejected
  with pytest.raises(ValueError):
    PoleFigure(h, r, np.ones(3))
  with pytest.raises(ValueError):
    PoleFigure(h, r, np.ones(r.size), superposition=[1.0, 1.0])
  with pytest.raises(ValueError):
    PoleFigure(vector3d(1, 0, 0), r, np.ones(r.size))
  # a data set holds several pole figures; MATLAB's numPF, select and cat
  both = PoleFigure([h, hs], r, [np.ones(r.size), 2 * np.ones(r.size)], superposition=[None, [0.52, 1.23]])
  assert both.numPF == 2 and both.CS is cs and both.SS is ss
  assert both.allR[0].shape == r.shape and both.allI[1].shape == r.shape
  assert np.allclose(both.c[1], [0.52, 1.23]) and both.c[0] == [1.0]
  assert both.select(1).numPF == 1 and both[1].allH[0].size == 2 and both[[0, 1]].numPF == 2
  assert [p.numPF for p in both] == [1, 1]
  assert PoleFigure.cat(both, pf).numPF == 3
  assert both.intensities.size == 2 * r.size and both.h.size == 3 and both.r.size == 2 * r.size
  # frames given positionally land on directions that name none
  plain = PoleFigure(vector3d(1, 0, 0), vector3d(r.data), np.ones(r.size), cs, specimenFrame("222"))
  assert plain.CS is cs and plain.SS.numSym() == 4
  # a mask over the measured directions keeps what it says, one pole figure at a time
  keep = both.r.theta < 60 * degree
  sub = both[keep]
  assert sub.intensities.size == np.count_nonzero(keep) and sub.numPF == 2
  # the intensities are assigned as one vector or one number
  both.intensities = np.arange(2 * r.size)
  assert both.allI[1].ravel()[0] == r.size
  both.intensities = 3
  assert np.all(both.intensities == 3)
  assert "PoleFigure" in repr(both) and "36 x 10 points" in repr(both)


def test_P2_quadrature_weights():
  # on a regular grid the weights are exact: they integrate every harmonic they see
  r = regularS2Grid(resolution=10 * degree, antipodal=True)
  w = calcQuadratureWeights(r)
  assert np.isclose(np.sum(w), 1) and np.all(w > 0) and w.shape == r.shape
  f = S2FunHarmonic(rng.standard_normal(36) + 1j * rng.standard_normal(36), real=True, antipodal=True)
  assert abs(np.sum(w * np.real(f(r))) - mean(f)) < 1e-7
  # the full sphere as well
  d = regularS2Grid(resolution=6 * degree)
  wd = calcQuadratureWeights(d)
  g = S2FunHarmonic(rng.standard_normal(49) + 1j * rng.standard_normal(49), real=True)
  assert np.isclose(np.sum(wd), 1)
  assert abs(np.sum(wd * np.real(g(d))) - mean(g)) < 1e-7
  # scattered points: the Voronoi weights are positive and close, the least norm weights exact
  v = vector3d.rand(400, antipodal=True, rng=rng)
  wv = calcQuadratureWeights(v)
  assert np.isclose(np.sum(wv), 1) and np.all(wv > 0)
  assert abs(np.sum(wv * np.real(f(v))) - mean(f)) < 1e-2
  wh = calcQuadratureWeights(v, method="harmonic")
  assert np.isclose(np.sum(wh), 1)
  assert abs(np.sum(wh * np.real(f(v))) - mean(f)) < 1e-8
  # equal weights would not do it
  assert abs(np.sum(np.real(f(v))) / v.size - mean(f)) > 1e-3
  assert r.calcQuadratureWeights().shape == r.shape


def test_P3_simulation_from_an_ODF():
  # a group without an element sending h to -h, so the pole figure is direction valued
  cs, ss = crystalFrame("3"), specimenFrame.default
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=15 * degree)
  r = regularS2Grid(resolution=10 * degree, antipodal=True)
  # check_calcPoleFigureSuperposition: the default is the mean of the components
  h23 = Miller([1, 0], [0, 1], [-1, -1], [1, 1], cs)
  h = [Miller(1, 0, 0, cs), h23]
  pfs = calcPoleFigure(odf, h, r)
  assert pfs.numPF == 2 and np.allclose(pfs.c[1], [0.5, 0.5]) and pfs.c[0] == [1.0]
  ref = 0.5 * _poleDensity(odf, h23[0], pfs.allR[1]) + 0.5 * _poleDensity(odf, h23[1], pfs.allR[1])
  assert np.max(np.abs(pfs.allI[1] - ref)) < 1e-10
  # coefficients handed over are used as they are
  pfe = calcPoleFigure(odf, h, r, superposition=[None, [0.52, 1.23]])
  assert np.allclose(calcPoleFigure(odf, h[0], r, superposition=2.0).allI[0], 2 * pfs.allI[0])
  assert np.allclose(calcPoleFigure(odf, [h[0]], r, superposition=[2.0]).allI[0], 2 * pfs.allI[0])
  assert PoleFigure(h[0], r, np.ones(r.size), superposition=2.0).c[0] == [2.0]
  ref2 = 0.52 * _poleDensity(odf, h23[0], pfs.allR[1]) + 1.23 * _poleDensity(odf, h23[1], pfs.allR[1])
  assert np.max(np.abs(pfe.allI[1] - ref2)) < 1e-10
  # a plain pole figure is its pole density function, on the grid it was asked for
  assert pfs.allI[0].shape == r.shape
  assert np.allclose(pfs.allI[0], _poleDensity(odf, h[0], pfs.allR[0]))
  # a measurement sees the even part: the pole figure of r and of -r agree
  P = pdf(odf, h[0])
  vs = vector3d.rand(20, rng=rng)
  assert not P.antipodal
  assert np.allclose(_poleDensity(odf, h[0], vector3d(vs, antipodal=True)), np.real(P(vs) + P(-vs)) / 2)
  # every pole density has mean one, up to what the 10 degree rule integrates of a 15 degree kernel
  assert abs(mean(pfs)[0] - 1) < 1e-3
  assert np.allclose(calcPoleFigure(uniformODF(cs, ss), h[0], r).allI[0], 1)
  # a Miller list gives one pole figure per element, a grid keyword the regular grid
  three = calcPoleFigure(odf, Miller([1, 0, 0], [0, 1, 0], [0, 0, 1], cs), r)
  assert three.numPF == 3 and three.allH[2].size == 1
  assert calcPoleFigure(odf, h[0], resolution=15 * degree).allR[0].shape == (24, 7)
  # one set of specimen directions per pole figure
  r2 = [regularS2Grid(resolution=10 * degree, antipodal=True), regularS2Grid(resolution=15 * degree, antipodal=True)]
  assert [p.intensities.size for p in calcPoleFigure(odf, h, r2)] == [x.size for x in r2]
  # the pole figures an ODF predicts on the directions of a measurement, either order
  sim = calcPoleFigure(odf, pfs)
  assert sim.numPF == 2 and np.allclose(sim.c[1], pfs.c[1]) and np.allclose(sim.allI[1], pfs.allI[1])
  assert np.allclose(calcPoleFigure(pfs, odf).allI[0], pfs.allI[0]) and np.allclose(odf.calcPoleFigure(h, r).allI[0], pfs.allI[0])


def test_P4_arithmetic():
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=20 * degree)
  r = regularS2Grid(resolution=10 * degree, antipodal=True)
  pf = calcPoleFigure(odf, Miller(1, 0, 0, cs), r)
  I = pf.allI[0]
  assert np.allclose((pf * 2).allI[0], 2 * I) and np.allclose((2 * pf).allI[0], 2 * I)
  assert np.allclose((pf + 1).allI[0], I + 1) and np.allclose((1 - pf).allI[0], 1 - I)
  assert np.all((pf - pf).allI[0] == 0) and np.allclose((-pf).allI[0], -I)
  assert np.isclose(max(max(pf, 1))[0], np.maximum(np.max(I), 1)) and min(max(pf, 1))[0] >= 1
  assert np.allclose((pf / pf).allI[0], 1) and np.allclose((pf * pf).allI[0], I * I)
  assert abs(mean(pf)[0] - 1) < 1e-4
  assert abs(mean((pf * 7).normalize())[0] - 1) < 1e-12
  assert np.allclose(pf.scale(3).allI[0], 3 * I)
  # a data set is corrected in one line, with one factor for all or one per pole figure
  both = PoleFigure.cat(pf, pf * 3)
  assert np.allclose(mean(both - 1), [mean(pf)[0] - 1, mean(pf * 3)[0] - 1])
  assert np.allclose(max(both), [np.max(I), 3 * np.max(I)])
  assert np.allclose((both / [1, 3]).allI[1], I)
  with pytest.raises(ValueError):
    both / [1, 2, 3]
  # turning the specimen turns its pole figures
  rot = rotation.byAxisAngle(zvector, 30 * degree)
  turned = pf.rotate(rot)
  assert np.allclose(turned.allI[0], I) and turned.antipodal and turned.SS is ss
  assert np.max(np.abs(calcPoleFigure(odf.rotate(rot), Miller(1, 0, 0, cs), turned.allR[0]).allI[0] - I)) < 1e-5
  # a rotation that is no element of the specimen group strips the group
  sym = PoleFigure(Miller(1, 0, 0, cs), r, I, specimenFrame("222"))
  assert sym.rotate(rotation.byAxisAngle(zvector, np.pi)).SS.numSym() == 4
  assert sym.rotate(rotation.byAxisAngle(zvector, 30 * degree)).SS.numSym() == 1


def test_P5_normalisation_and_errors():
  cs, ss = crystalFrame("m-3m"), specimenFrame.default
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=20 * degree)
  r = regularS2Grid(resolution=10 * degree, antipodal=True)
  pfs = calcPoleFigure(odf, Miller([1, 1], [0, 1], [0, 1], cs), r)
  # the scaling that carries the mass of the first onto the second
  assert np.isclose(calcNormalization(pfs[0], pfs[0] * 3)[0], 1 / 3)
  assert np.allclose(calcNormalization(pfs, pfs * 2), [0.5, 0.5])
  # the recalculation of the ODF the data came from has no error, in every mode
  for mode in ("relative", "RP", "l1", "l2"):
    assert np.all(calcError(pfs, odf, mode) < 1e-9)
  # a different ODF has one, and the error is invariant under a scaling of the data
  other = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=20 * degree)
  assert np.all(calcError(pfs, other) > 0.1)
  assert np.allclose(calcError(pfs[0] * 5, odf), calcError(pfs[0], odf), atol=1e-12)
  # the error pole figure lives on the same directions, the RP mode only where it is defined
  e = calcErrorPF(pfs[0], other)
  assert e.intensities.size == pfs[0].intensities.size and np.all(e.intensities >= 0)
  assert calcErrorPF(pfs[0], other, "RP").intensities.size <= pfs[0].intensities.size
  with pytest.raises(ValueError):
    calcError(pfs, other, "l3")
  # normalising against an ODF
  assert abs(mean((pfs[0] * 4).normalize(odf))[0] - mean(pfs[0])[0]) < 1e-6
  # the error between two ODFs still goes through the same name
  assert calcError(odf, odf) < 1e-12 and calcError(odf, other, "l2") > 0


def test_P6_corrections_outliers_noise():
  cs, ss = crystalFrame("m-3m"), specimenFrame("222")
  odf = unimodalODF(orientation(rotation.rand(rng=rng), cs, ss), halfwidth=20 * degree)
  r = regularS2Grid(resolution=10 * degree, antipodal=True, frame=ss)
  pf = calcPoleFigure(odf, Miller(1, 0, 0, cs), r)
  I = pf.allI[0]
  # a background and a defocusing curve are taken off again
  bg = pf._rebuild([np.full(I.shape, 0.3)])
  d = pf._rebuild([0.5 + 0.5 * np.cos(r.theta) ** 2])
  raw = pf * d + bg
  assert np.max(np.abs(correct(raw, background=bg, defocusing=d).allI[0] - I)) < 1e-12
  assert np.max(np.abs(correct(raw, background=0.3, defocusing=d).allI[0] - I)) < 1e-12
  # the defocusing curve measured on a coarser ring set is interpolated in the polar angle
  th = np.arange(0, 91, 10) * degree
  curve = PoleFigure(Miller(1, 0, 0, cs), vector3d.byPolar(th, np.zeros(th.size), antipodal=True), 0.5 + 0.5 * np.cos(th) ** 2, ss)
  assert np.max(np.abs(correct(pf * d, defocusing=curve).allI[0] - I)) < 1e-3
  # a defocusing background is taken off the defocusing first
  assert np.max(np.abs(correct(pf * d, defocusing=d + 0.2, defocusingBackground=0.2).allI[0] - I)) < 1e-12
  # a spike is an outlier, the rest is not
  spiked = pf._rebuild([I.ravel().copy()])
  spiked.allI[0].flat[100] += 20 * np.max(I)
  out = isOutlier(spiked)
  assert out[100] and np.count_nonzero(out) < 5
  # Poisson noise keeps the mean and the shape
  noisy = noisepf(pf, 100, rng=rng)
  assert noisy.allI[0].shape == I.shape and abs(mean(noisy)[0] / 100 - mean(pf)[0]) < 0.05
  assert min(noisepf(pf, 10, background=5, nonnegative=True, rng=rng))[0] >= 0
  # equivalent specimen directions are merged by unique, keeping the pole density
  u = unique(pf)
  assert u.intensities.size == 91                      # the 222 sector of the 360 point half grid
  assert np.max(np.abs(u.allI[0] - _poleDensity(odf, pf.allH[0], u.allR[0]))) < 1e-7
  # pole figures of the same crystal directions are joined
  keep = np.zeros(r.size, dtype=bool)
  keep[:180] = True
  half1, half2 = pf[keep], pf[~keep]
  m = union(half1, half2, calcPoleFigure(odf, Miller(1, 1, 1, cs), r))
  assert m.numPF == 2 and m.allI[0].size == 360 and union(half1, None).numPF == 1


def test_P7_the_defocusing_curve():
  cs = crystalFrame("m-3m")
  r = regularS2Grid(resolution=10 * degree, maxTheta=80 * degree, antipodal=True)
  pf = PoleFigure(Miller(1, 0, 0, cs), r, np.ones(r.shape))
  psi = SchulzDefocusingKernel(r=100, w=0.5)
  # the correction is one at the pole, grows with the tilt and diverges at 90 degree
  assert psi.eval(1.0) == 1 and psi.eval(0.0) == np.inf
  curve = psi(np.cos(np.linspace(0, 80 * degree, 9)))
  assert np.all(np.diff(curve) >= -1e-9) and curve[-1] > 1.5
  # pinned against MATLAB's SchulzDefocusingKernel
  assert np.allclose(psi(np.cos(np.array([30, 50, 70]) * degree)), [1.0, 1.685394, 3.885520], atol=1e-5)
  d = psi.defocusing(pf)
  assert d.allI[0].shape == pf.allI[0].shape and np.all(np.isfinite(d.allI[0]))
  assert np.isclose(d.allI[0][0, 0], 1)                # the pole is not defocused
  # a measurement damped by the curve is recovered by the correction
  assert np.max(np.abs(correct(pf * d, defocusing=d).allI[0] - 1)) < 1e-12


def test_P8_a_pole_figure_as_a_function_and_refine():
  cs, ss = crystalFrame("m-3m"), specimenFrame("1")
  odf = unimodalODF(orientation.byEuler(30 * degree, 50 * degree, 10 * degree, cs, ss), halfwidth=20 * degree)
  h = Miller(1, 0, 0, cs)
  r = regularS2Grid(resolution=5 * degree, antipodal=True, frame=ss)
  pf = calcPoleFigure(odf, h, r)
  f = S2FunHarmonic(pf)
  assert f.frame is pf.SS and f.antipodal
  # half of what the point density of a five degree grid resolves
  assert f.bandwidth == 23
  assert S2FunHarmonic(pf, bandwidth=12).bandwidth == 12
  # the function gives the intensities back, up to what the bandwidth drops
  vals = np.real(f(r))
  assert np.sqrt(np.mean((vals - pf.allI[0]) ** 2)) < 0.05 * np.max(pf.allI[0])
  # a list without a resolution is read at the bandwidth its count supports
  rl = vector3d.rand(500, antipodal=True, rng=rng)
  assert S2FunHarmonic(PoleFigure(h, rl, rng.random(500), ss)).bandwidth == 14
  with pytest.raises(ValueError):
    S2FunHarmonic(PoleFigure.cat(pf, pf))
  # refining a cap adds points inside it and none below its rim
  v = equispacedS2Grid(resolution=20 * degree, maxTheta=80 * degree)
  w, new = refine(v)
  assert w.size == v.size + new.size and new.size > v.size / 2
  assert np.all(new.theta <= 80 * degree + 1e-6)
  assert v.refine()[1].size == new.size
